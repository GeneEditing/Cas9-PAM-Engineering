#!/usr/bin/env python3
"""Final v2 local reproduction orchestrator. Uses existing project environments/data."""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE/'common'))
from support import (require, fasta, table, write_table, digest, select_msa, index_msa, POSITIONS, WT)

STAGES = ['01_pammla_64m','02_shortlist','03_progen2','04_esmc',
          '05_joint_screen','06_colabfold','07_final_ranking']
MODELS = ['220924_NN_rand_seed0_ROS','220924_NN_rand_seed2_ROS','220924_NN_rand_seed3_ROS']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_config(path):
    path = Path(path).resolve(); c = json.loads(path.read_text())
    root = Path(c['project_root']).expanduser().resolve()
    def absolute(value):
        p = Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (root/p).resolve())
    c['project_root'] = str(root)
    for key in ['reference_run','run_dir','reference_fasta','train_variants','colabfold_cache']:
        c[key] = absolute(c[key])
    c['msa_roots'] = [absolute(p) for p in c['msa_roots']]
    c['environments'] = {k:absolute(v) for k,v in c['environments'].items()}
    require(c['run_dir'] != c['reference_run'], 'Use a new run directory; reference run must remain unchanged')
    return c


def environment(c):
    env = os.environ.copy()
    for name in ['PROGEN2_INPUT','PROGEN2_RUN','PROGEN2_EXPECTED_N']:
        env.pop(name, None)
    env.update(CAS9_ROOT=c['project_root'], CAS9_RUN=c['run_dir'],
               CUDA_VISIBLE_DEVICES=str(c['gpu']), TF_FORCE_GPU_ALLOW_GROWTH='true',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    # PAMmla pickles can reference project-local encoder definitions.
    env['PYTHONPATH']=str(Path(c['project_root'])/'PAMmla') + os.pathsep + env.get('PYTHONPATH','')
    # Use environment's C++ runtime when present (copied pyarrow environments).
    lib = str(Path(c['environments']['pammla'])/'lib')
    env['LD_LIBRARY_PATH']=lib+os.pathsep+env.get('LD_LIBRARY_PATH','')
    return env


def execute(command,c,log_name,cwd=None,env_name=None):
    run=Path(c['run_dir']); log=run/'logs'/log_name; log.parent.mkdir(parents=True,exist_ok=True)
    env=environment(c)
    if env_name:
        prefix=Path(c['environments'][env_name])
        env['PATH']=str(prefix/'bin')+os.pathsep+env.get('PATH','')
        env['LD_LIBRARY_PATH']=str(prefix/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    command=list(map(str,command))
    record={'time':now(),'cwd':str(cwd or PACKAGE),'command':command,'log':str(log)}
    with (run/'metadata/commands.jsonl').open('a') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print('\n$ '+shlex.join(command),flush=True)
    with log.open('a') as f:
        f.write('\nSTART '+now()+'\n'); f.flush()
        with subprocess.Popen(command,cwd=cwd or PACKAGE,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    print(line,end='',flush=True); f.write(line); f.flush()
                code=proc.wait()
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                raise
        f.write('EXIT '+str(code)+' '+now()+'\n')
    require(code==0, f'Command failed (exit {code}); see {log}')


def python(c,name):
    return Path(c['environments'][name])/'bin/python'


def preflight(c):
    root=Path(c['project_root']); issues=[]
    required=[Path(c['reference_fasta']),Path(c['train_variants']),
              root/'progen2/tokenizer.json',root/'progen2/checkpoints/progen2-base/config.json',
              root/'progen2/models/progen/modeling_progen.py',root/'esmc_models/ESMC-600M/config.json',
              Path(c['colabfold_cache']),
              Path(__file__).resolve().parent.parent/'inputs/reference_front38.tsv']
    for name in MODELS:
        required += [root/'PAMmla'/name/p for p in ['saved_model','encoder.pkl','mean_std.pkl','data_parameters.pkl']]
    required += [python(c,k) for k in ['pammla','progen2','esmc','colabfold']]
    required += [Path(c['environments']['colabfold'])/'bin/colabfold_batch']
    # Check the full local ptm weight set before starting a potentially long 64M run.
    cache=Path(c['colabfold_cache'])
    present={p.name for p in cache.rglob('params_model_*_ptm.npz')} if cache.is_dir() else set()
    for i in range(1,6):
        name=f'params_model_{i}_ptm.npz'
        if name not in present: issues.append('MISSING ColabFold parameter '+name+' under '+str(cache))
    for folder in [root/'progen2/checkpoints/progen2-base',root/'esmc_models/ESMC-600M']:
        if not any(folder.glob('*.bin')) and not any(folder.glob('*.safetensors')):
            issues.append('MISSING model weight files under '+str(folder))
    for p in required:
        if not p.exists(): issues.append('MISSING '+str(p))
    require(not issues, '\n'.join(issues))
    ref=fasta(c['reference_fasta'])
    require(len(ref)==3421 and 'SpCas9_WT' in ref, 'Reference FASTA must contain 3420 candidates + SpCas9_WT')
    wt=ref['SpCas9_WT']
    require(len(wt)==1368 and ''.join(wt[p-1] for p in POSITIONS)==WT,'Reference WT mismatch')
    genotypes=set()
    for cid,seq in ref.items():
        require(len(seq)==1368 and set(seq)<=set('ACDEFGHIKLMNPQRSTVWY'),'Invalid sequence '+cid)
        if cid=='SpCas9_WT': continue
        g=cid.split('|')[-1]
        require(len(g)==6 and g==''.join(seq[p-1] for p in POSITIONS),'ID/genotype mismatch '+cid)
        require(g not in genotypes,'Duplicate reference genotype '+g); genotypes.add(g)
        require(all(a==b or i in POSITIONS for i,(a,b) in enumerate(zip(wt,seq),1)), 'Changes outside six positions '+cid)
    train=[s.strip() for s in Path(c['train_variants']).read_text().splitlines() if s.strip()]
    require(bool(train) and all(len(s)==6 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') for s in train),'Invalid training genotype list')
    reference_selected=table(Path(__file__).resolve().parent.parent/'inputs/reference_front38.tsv')
    require(len(reference_selected)==38,'Reference front table must contain 38 candidates')
    print('Checking saved MSA query sequences...',flush=True)
    index=index_msa(c['msa_roots'])
    missing=[r['candidate_id'] for r in reference_selected if ref.get(r['candidate_id']) not in index]
    require(not missing,'Missing exact-query MSA before starting long run: '+', '.join(missing))
    for name,code in {
        'pammla':'import tensorflow, pandas, pyarrow, sklearn; print("PAMmla imports OK", tensorflow.__version__, pandas.__version__)',
        'progen2':'import torch, tokenizers, transformers; assert torch.cuda.is_available(); print("ProGen2 CUDA OK", torch.__version__)',
        'esmc':'import torch, transformers; assert torch.cuda.is_available(); print("ESM-C CUDA OK", torch.__version__)',
        'colabfold':'import jax, colabfold; assert any(d.platform=="gpu" for d in jax.devices()); print("ColabFold GPU OK", jax.__version__)'
    }.items():
        execute([python(c,name),'-c',code],c,'preflight_'+name+'.log',env_name=name)
    execute([Path(c['environments']['colabfold'])/'bin/colabfold_batch','--help'],c,'colabfold_help.log',env_name='colabfold')
    help_text=(Path(c['run_dir'])/'logs/colabfold_help.log').read_text()
    for flag in ['--data','--model-type','--num-models','--num-recycle','--num-relax','--rank','--random-seed','--num-seeds','--max-seq','--max-extra-seq']:
        require(flag in help_text,'Installed ColabFold missing option '+flag)
    print('PASS: project inputs, four environments, GPU and all 38 saved MSAs',flush=True)


def initialize(c):
    run=Path(c['run_dir']); run.mkdir(parents=True,exist_ok=True)
    for folder in ['inputs','metadata','logs','pids','work','scripts_snapshot','results']:
        (run/folder).mkdir(exist_ok=True)
    for stage in STAGES:
        (run/'results'/stage).mkdir(exist_ok=True)
    # Lock configuration and code to this run so resume cannot mix configurations.
    snapshot=run/'metadata/config.resolved.json'
    package_files=[PACKAGE/'pipeline.py']+sorted((PACKAGE/'common').glob('*.py'))+sorted((PACKAGE/'steps').glob('*/*.py'))
    hashes={str(p.relative_to(PACKAGE)):digest(p) for p in package_files}
    hash_file=run/'metadata/code_sha256.json'
    if snapshot.exists() and not any((run/'results').rglob('*.*')) and not any((run/'metadata').glob('*.done.json')):
        snapshot.unlink()
    if snapshot.exists():
        require(json.loads(snapshot.read_text())==c,'Configuration changed for existing run. Use a new run_dir.')
        require(hash_file.exists() and json.loads(hash_file.read_text())==hashes,'Package code changed for existing run. Use a new run_dir.')
    else:
        require(not any((run/'results').rglob('*.*')),'Run directory contains prior outputs without this package metadata; choose a new run_dir')
        snapshot.write_text(json.dumps(c,indent=2)+'\n')
        hash_file.write_text(json.dumps(hashes,indent=2)+'\n')
        for p in package_files:
            dst=run/'scripts_snapshot'/p.relative_to(PACKAGE); dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dst)
    # Snapshot two small authoritative inputs, checking rather than overwriting on resume.
    for source,dest in [(Path(c['reference_fasta']),run/'inputs/reference_candidates.fasta'),
                        (Path(c['train_variants']),run/'inputs/train_variants.txt')]:
        require(source.is_file(),'Missing input: '+str(source))
        if dest.exists(): require(digest(source)==digest(dest),'Input changed: '+str(source)+'; choose a new run_dir')
        else: shutil.copy2(source,dest)
    return run


def link(source,dest):
    if dest.is_symlink() or dest.exists():
        require(dest.resolve()==source.resolve(),'Conflicting work path: '+str(dest))
    else: dest.symlink_to(source, target_is_directory=source.is_dir())


def script_file(name):
    if name == 'support.py': return PACKAGE/'common'/name
    matches = list((PACKAGE/'steps').glob('*/'+name))
    require(len(matches) == 1, 'Expected one step script: '+name)
    return matches[0]

def scripts(c,name,env='pammla',args=(),cwd=None):
    execute([python(c,env),'-u',script_file(name),*args],c,name+'.log',cwd,env)


def run_stage(c,n):
    run=Path(c['run_dir']); root=Path(c['project_root']); tag=STAGES[n-1]
    if n==1:
        work=run/'work/01_pammla_64m'; work.mkdir(exist_ok=True)
        for name in MODELS: link(root/'PAMmla'/name,work/name)
        link(run/'results/01_pammla_64m',work/'predictions_64m')
        for shard in range(400):
            target=run/'results/01_pammla_64m'/f'shard_{shard:03d}.parquet'
            if target.exists():
                print('Existing shard; full validation at stage end:',target.name,flush=True)
                continue
            execute([python(c,'pammla'),'-u',script_file('batch_predict.py'),str(shard)],
                    c,f'01_pammla_64m/shard_{shard:03d}.log',work,'pammla')
        scripts(c,'support.py',args=['shards'])
    elif n==2:
        work=run/'results/02_shortlist'
        link(run/'results/01_pammla_64m',work/'predictions_64m')
        link(run/'inputs/train_variants.txt',work/'train_variants.txt')
        scripts(c,'shortlist.py',cwd=work)
        scripts(c,'support.py',args=['sequences'])
    elif n==3:
        scripts(c,'support.py',args=['recover03'])
        scripts(c,'score_progen2.py','progen2')
        scripts(c,'support.py',args=['03'])
    elif n==4:
        scripts(c,'support.py',args=['recover04'])
        folder=run/'results/04_esmc'
        scripts(c,'score_esmc.py','esmc',[
            '--input',run/'inputs/03_progen2/candidates_with_wt.fasta',
            '--summary',folder/'esmc_summary.csv','--detail',folder/'esmc_per_mutation.csv',
            '--wt-id','SpCas9_WT','--mask-batch-size',str(c['esmc_mask_batch_size'])])
        scripts(c,'support.py',args=['04'])
    elif n==5:
        scripts(c,'joint_screen.py')
        screen_dir = run/'results/05_joint_screen'
        selected_path = screen_dir/'03_fronts_1_to_4.tsv'
        selected = table(selected_path)
        require(len(selected)==39 and len({r['candidate_id'] for r in selected})==39,
                'Expected the recorded 39 unique Front1-4 candidates; see screening_report.txt')

        reference_path = Path(__file__).resolve().parent.parent/'inputs/reference_front38.tsv'
        reference = table(reference_path)
        reference_ids = {r['candidate_id'] for r in reference}
        require(len(reference)==len(reference_ids)==38,
                'Historical reference must contain 38 unique candidates')

        fixed = [r for r in selected if r['candidate_id'] in reference_ids]
        excluded = [r['candidate_id'] for r in selected
                    if r['candidate_id'] not in reference_ids]
        require(len(fixed)==38 and
                {r['candidate_id'] for r in fixed}==reference_ids and
                excluded==['PAMmla_00422|MRSCYQ'],
                '39-to-38 branch differs from the recorded run; stop without changing thresholds')

        actual39_path = screen_dir/'03_fronts_1_to_4_actual39.tsv'
        require(not actual39_path.exists(), 'Actual-39 audit copy already exists')
        shutil.copy2(selected_path, actual39_path)

        ids_path = screen_dir/'03_fronts_1_to_4_ids.txt'
        if ids_path.exists():
            shutil.copy2(ids_path, screen_dir/'03_fronts_1_to_4_actual39_ids.txt')
        write_table(selected_path, fixed)
        ids_path.write_text(
            '\n'.join(r['candidate_id'] for r in fixed)+'\n'
        )

        branch = {
            'actual_new_selected': 39,
            'fixed_reference_cohort': 38,
            'excluded_for_cohort_comparison': excluded,
            'method': 'Filter the new 39-row selected table by historical candidate IDs; retain current scores',
            'actual39_sha256': digest(actual39_path),
            'historical_ids_source_sha256': digest(reference_path),
        }
        (screen_dir/'reference38_branch.json').write_text(
            json.dumps(branch, indent=2)+'\n'
        )
        print('Recorded 39 candidates; continuing with fixed historical 38', flush=True)
    elif n==6:
        select_msa(run,c['msa_roots'],run/'results/05_joint_screen/03_fronts_1_to_4.tsv')
        execute([Path(c['environments']['colabfold'])/'bin/colabfold_batch',
                 run/'inputs/06_colabfold_msa',run/'results/06_colabfold/predictions',
                 '--data',c['colabfold_cache'],'--model-type','alphafold2_ptm',
                 '--num-models','1','--num-recycle','3','--num-relax','0','--rank','ptm',
                 '--random-seed','0','--num-seeds','1','--max-seq','512','--max-extra-seq','5120'],
                c,'06_colabfold.log',env_name='colabfold')
        scripts(c,'support.py',args=['structure'])
    elif n==7:
        scripts(c,'final_rank.py'); scripts(c,'score_breakdown.py'); scripts(c,'export_mutations.py')
        current=table(run/'results/07_final_ranking/final_top10.tsv')
        reference=Path(c['reference_run'])/'results/07_final_ranking/final_top10.tsv'
        comparison={'current_top10':[r['genotype'] for r in current],
                    'note':'New GPU/ColabFold results may differ; no historical structure scores substituted.'}
        if reference.exists():
            previous=[r['genotype'] for r in table(reference)]
            comparison.update(reference_top10=previous,same_order=previous==comparison['current_top10'],
                              same_set=set(previous)==set(comparison['current_top10']))
        (run/'results/07_final_ranking/top10_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print('COMPLETED',tag,now(),flush=True)


def outputs(run,n):
    patterns={1:['results/01_pammla_64m/shard_*.parquet'],
              2:['results/02_shortlist/*.csv','inputs/03_progen2/*.fasta'],
              3:['results/03_progen2/*.tsv','results/03_progen2/*.txt'],
              4:['results/04_esmc/*.csv'],
              5:['results/05_joint_screen/*.tsv','results/05_joint_screen/*.json','results/05_joint_screen/*.txt'],
              6:['results/06_colabfold/colabfold_structural_summary_all38.tsv',
                 'results/06_colabfold/predictions/*rank_001_*.json','results/06_colabfold/predictions/*rank_001_*.pdb',
                 'inputs/07_final_ranking/*','metadata/msa_manifest.tsv','inputs/06_colabfold_msa/*.a3m'],
              7:['results/07_final_ranking/*']}
    files=sorted(set(p for pattern in patterns[n] for p in run.glob(pattern) if p.is_file()))
    require(bool(files),'No stage outputs found')
    # Structural score tables and small outputs use content hashes. Large parquet uses metadata
    # after exhaustive stage-01 semantic validation, avoiding repeated multi-GB reads on resume.
    return {str(p.relative_to(run)): {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
            **({'sha256':digest(p)} if p.suffix!='.parquet' else {})} for p in files}


def done(run,n):
    marker=run/'metadata'/f'{n:02d}.done.json'
    if not marker.exists(): return False
    old=json.loads(marker.read_text())
    require(old['outputs']==outputs(run,n),'Completed outputs changed for stage '+str(n)+'. Use a new run_dir to recompute.')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default=str(PACKAGE/'config.json'))
    ap.add_argument('action',choices=['check','run','step','status'])
    ap.add_argument('stage',type=int,nargs='?',choices=range(1,8))
    args=ap.parse_args(); c=load_config(args.config); run=Path(c['run_dir'])
    if args.action=='status':
        print('Run:',run)
        for i,tag in enumerate(STAGES,1):
            print(('DONE' if (run/'metadata'/f'{i:02d}.done.json').exists() else 'PENDING'),tag)
        report=run/'results/07_final_ranking/final_ranking_report.txt'
        if report.exists(): print(report.read_text())
        return
    if args.action=='step': require(args.stage is not None,'Specify stage 1–7')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'pipeline.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This run already has an active pipeline')
        run=initialize(c)
        (run/'pids/pipeline.pid').write_text(str(os.getpid())+'\n')
        if args.action in {'check','run'}:
            preflight(c)
        if args.action=='check': return
        stages=range(1,8) if args.action=='run' else [args.stage]
        for n in stages:
            for previous in range(1,n):
                require(done(run,previous),f'Stage {previous} must complete first')
            if done(run,n):
                print('SKIP validated completed stage',n,flush=True); continue
            run_stage(c,n)
            marker=run/'metadata'/f'{n:02d}.done.json'
            temp=marker.with_suffix('.tmp')
            temp.write_text(json.dumps({'completed':now(),'outputs':outputs(run,n)},indent=2)+'\n'); temp.replace(marker)
        print('\nResults:',run/'results',flush=True)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('\nInterrupted. Completed outputs retained.',file=sys.stderr); sys.exit(130)
    except Exception as exc:
        print('\nSTOP:',exc,file=sys.stderr); sys.exit(1)
