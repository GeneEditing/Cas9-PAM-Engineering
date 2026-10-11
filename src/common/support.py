"""Pipeline glue: no changes to model scoring or selection formulas."""
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

POSITIONS = [1135, 1136, 1218, 1219, 1335, 1337]
AA = 'GPAVLIMCFYWHKRQNEDST'
WT = 'DSGERT'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fasta(path):
    result = {}; name = None; chunks = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('>'):
            if name is not None:
                require(name not in result, 'Duplicate FASTA ID: ' + name)
                result[name] = ''.join(chunks)
            name = line[1:].split()[0]; chunks = []
        else:
            require(name is not None, 'Sequence before FASTA header')
            chunks.append(line)
    if name is not None:
        require(name not in result, 'Duplicate FASTA ID: ' + name)
        result[name] = ''.join(chunks)
    return result


def a3m_query(path):
    # Only read first sequence; alignment headers may contain repeated IDs.
    chunks = []; started = False
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if started:
                    break
                started = True
            elif started:
                chunks.append(line)
    return re.sub('[a-z.\\-]', '', ''.join(chunks))


def table(path, delimiter='\t'):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_table(path, rows, delimiter='\t'):
    require(bool(rows), 'Cannot write empty table: ' + str(path))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter,
                           lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def write_fasta(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join('>' + k + '\n' + v + '\n' for k, v in records.items()))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_shards(run):
    import numpy as np
    import pandas as pd
    folder = run / 'results/01_pammla_64m'
    files = sorted(folder.glob('shard_*.parquet'))
    require([p.name for p in files] == [f'shard_{i:03d}.parquet' for i in range(400)],
            'Expected shards 000–399; missing or unexpected shard files')
    pams = ['N'+a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
    for i, path in enumerate(files):
        df = pd.read_parquet(path)
        require(len(df) == 160000 and df.index.is_unique, 'Invalid rows/index: ' + str(path))
        require(set(df.columns) == set(pams), 'Expected 64 PAM columns: ' + str(path))
        prefix = AA[i//20] + AA[i%20]
        require(all(len(v)==6 and v.startswith(prefix) and set(v)<=set(AA) for v in df.index),
                'Wrong shard genotypes: ' + str(path))
        require(np.isfinite(df.to_numpy()).all(), 'Non-finite PAM score: ' + str(path))
    print('PASS: 400 shards, 64,000,000 unique variants, 64 finite PAM scores each')


def prepare_sequences(run):
    rows = table(run/'results/02_shortlist/final_shortlist_v2.csv', ',')
    reference = fasta(run/'inputs/reference_candidates.fasta')
    wt = reference['SpCas9_WT']
    require(len(wt)==1368 and ''.join(wt[p-1] for p in POSITIONS)==WT, 'WT mismatch')
    require(len(rows)==3420 and len({r['variant'] for r in rows})==3420, 'Expected 3420 unique variants')
    by_genotype = {k.split('|')[-1]: (k,v) for k,v in reference.items() if k != 'SpCas9_WT'}
    require(set(by_genotype)=={r['variant'] for r in rows},
            'Shortlist differs from reference genotype set; stop before GPU scoring')
    records = {'SpCas9_WT': wt}; meta = []
    for row in rows:
        g = row['variant']; seq = list(wt)
        for p,a in zip(POSITIONS,g):
            seq[p-1] = a
        seq = ''.join(seq); cid, old_seq = by_genotype[g]
        require(seq==old_seq, 'Full-length reference mismatch: '+cid)
        expected_mut = ';'.join(f'{a}{p}{b}' for a,p,b in zip(WT,POSITIONS,g) if a!=b)
        require(row['mutations']==expected_mut, 'Mutation annotation mismatch: '+cid)
        records[cid]=seq
        meta.append(dict(row,candidate_id=cid,sequence_length=len(seq)))
    write_fasta(run/'inputs/03_progen2/candidates_with_wt.fasta',records)
    write_table(run/'results/02_shortlist/candidate_metadata.csv',meta,',')
    print('PASS: 3420 candidates + WT; historical IDs retained by exact genotype/sequence mapping')


def clean_scores(path, id_column, delimiter, expected, statuses):
    """Keep latest successful record per ID; preserve the append log as an audit copy."""
    rows=table(path,delimiter); selected={}
    for row in rows:
        cid=row.get(id_column)
        if cid in expected and row.get('status') in statuses and None not in row.values() and None not in row:
            selected[cid]=row
    missing=set(expected)-set(selected)
    require(not missing, f'Incomplete scores: {len(missing)} missing/error IDs; rerun stage. Examples: {sorted(missing)[:5]}')
    require(all(r.get(id_column) in expected for r in rows if r.get(id_column)), 'Unexpected score IDs')
    for cid,row in selected.items():
        cols = ['ll_lr_mean','ll_rl_mean','ll_bi_mean'] if id_column=='sequence_id' else ['mutctx_mean_delta','mutctx_min_delta']
        if id_column=='sequence_id':
            require(row['sequence_sha256']==hashlib.sha256(expected[cid].encode()).hexdigest(), 'Scored sequence hash mismatch')
        if cid!='SpCas9_WT':
            require(all(math.isfinite(float(row[c])) for c in cols), 'Non-finite score: '+cid)
    normalized=list(selected.values())
    if rows != normalized:
        backup=path.with_name(path.name+'.append-history.'+digest(path)[:12])
        if not backup.exists(): shutil.copy2(path,backup)
        write_table(path,normalized,delimiter)
    print('PASS:',path.name,len(normalized),'successful rows')


def recover_partial_scores(run, stage):
    """Normalize append files before resume; archive original bytes, keep valid success rows."""
    records=fasta(run/'inputs/03_progen2/candidates_with_wt.fasta')
    folder=run/('results/03_progen2' if stage=='03' else 'results/04_esmc')
    paths=[(folder/'progen2_full_raw.tsv','sequence_id','\t')] if stage=='03' else [
        (folder/'esmc_summary.csv','candidate_id',','),
        (folder/'esmc_per_mutation.csv','candidate_id',',')]
    for path,key,sep in paths:
        if not path.exists() or path.stat().st_size==0: continue
        with path.open(newline='') as f:
            reader=csv.DictReader(f,delimiter=sep); fields=reader.fieldnames; rows=list(reader)
        require(fields and key in fields,'Invalid score header: '+str(path))
        selected={}
        for row in rows:
            cid=row.get(key)
            if None in row or None in row.values(): continue
            require(cid in records,'Unknown score ID: '+str(cid))
            if 'status' in row:
                if row['status'] not in {'ok','wt','identical_to_wt'}: continue
                if stage=='03':
                    require(row['sequence_sha256']==hashlib.sha256(records[cid].encode()).hexdigest(),
                            'Existing score refers to changed sequence: '+cid)
                numeric=['ll_lr_mean','ll_rl_mean','ll_bi_mean'] if stage=='03' else ['mutctx_mean_delta','mutctx_min_delta']
                if cid!='SpCas9_WT':
                    try: valid=all(math.isfinite(float(row[k])) for k in numeric)
                    except (ValueError,KeyError): valid=False
                    if not valid: continue
                selected[cid]=row
            else:
                try: valid=all(math.isfinite(float(row[k])) for k in ['wtctx_delta','mutctx_delta'])
                except (ValueError,KeyError): valid=False
                if valid: selected[(cid,row['mutation'])]=row
        valid_rows=list(selected.values())
        if rows!=valid_rows:
            backup=path.with_name(path.name+'.append-history.'+digest(path)[:12])
            if not backup.exists(): shutil.copy2(path,backup)
            tmp=path.with_name(path.name+'.tmp')
            with tmp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,delimiter=sep,lineterminator='\n')
                writer.writeheader(); writer.writerows(valid_rows)
            tmp.replace(path)
            print('Recovered append file:',path.name,'valid rows:',len(valid_rows))


def validate_scores(run,stage):
    records=fasta(run/'inputs/03_progen2/candidates_with_wt.fasta')
    if stage=='03':
        clean_scores(run/'results/03_progen2/progen2_full_raw.tsv','sequence_id','\t',records,{'ok'})
    else:
        clean_scores(run/'results/04_esmc/esmc_summary.csv','candidate_id',',',records,{'ok','wt','identical_to_wt'})


def index_msa(roots):
    index={}; seen=set()
    for root in roots:
        root=Path(root)
        if not root.is_dir(): continue
        for path in sorted(root.rglob('*.a3m')):
            resolved=path.resolve()
            if resolved in seen: continue
            seen.add(resolved)
            query=a3m_query(path)
            if query: index.setdefault(query,[]).append(path)
    return index


def select_msa(run, roots, selected_path):
    records=fasta(run/'inputs/03_progen2/candidates_with_wt.fasta')
    selected=table(selected_path)
    require(len(selected)==38 and len({r['candidate_id'] for r in selected})==38, 'Expected 38 unique Front1–4 candidates')
    index=index_msa(roots); missing=[]; rows=[]
    target=run/'inputs/06_colabfold_msa'; target.mkdir(parents=True,exist_ok=True)
    for row in selected:
        cid=row['candidate_id']; sequence=records[cid]; paths=index.get(sequence,[])
        if not paths: missing.append(cid); continue
        chosen=paths[0]; output=target/(cid.replace('|','_')+'.a3m')
        shutil.copy2(chosen,output)
        rows.append({'candidate_id':cid,'source':str(chosen),'input':str(output),
                     'sha256':digest(output),'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                     'matching_files':len(paths)})
    require(not missing,'Missing exact-query saved MSA; no remote search or substituted sequence: '+', '.join(missing))
    expected={Path(r['input']).name for r in rows}
    require({p.name for p in target.glob('*.a3m')}==expected, 'Unexpected stale A3M files in input directory')
    write_table(run/'metadata/msa_manifest.tsv',rows)
    write_fasta(run/'inputs/07_final_ranking/all38_candidates.fasta',
                {r['candidate_id']:records[r['candidate_id']] for r in selected})
    print('PASS: 38 exact-query A3M inputs, including any separately prepared candidate')


def summarize_structure(run):
    selected=table(run/'results/05_joint_screen/03_fronts_1_to_4.tsv')
    seqs=fasta(run/'inputs/07_final_ranking/all38_candidates.fasta')
    folder=run/'results/06_colabfold/predictions'; merged=[]
    require(len(selected)==38,'Expected 38 selected candidates')
    for row in selected:
        cid=row['candidate_id']; name=cid.replace('|','_'); seq=seqs[cid]
        require(''.join(seq[p-1] for p in POSITIONS)==row['genotype'],'Genotype mismatch')
        scores=list(folder.glob(name+'_scores_rank_001_*.json'))
        pdbs=list(folder.glob(name+'_unrelaxed_rank_001_*.pdb'))
        require(len(scores)==len(pdbs)==1,'Missing/ambiguous rank1 structure: '+cid)
        data=json.loads(scores[0].read_text()); values=list(map(float,data['plddt'])); ptm=float(data['ptm'])
        require(len(values)==len(seq) and all(math.isfinite(x) and 0<=x<=100 for x in values),'Invalid pLDDT: '+cid)
        require(math.isfinite(ptm) and 0<=ptm<=1,'Invalid pTM: '+cid)
        require(any(l.startswith('ATOM  ') for l in pdbs[0].read_text().splitlines()),'Empty PDB: '+cid)
        mean=sum(values)/len(values); local=[values[p-1] for p in POSITIONS]
        merged.append(dict(row,mean_plddt=mean,ptm=ptm,six_site_mean_plddt=sum(local)/6,
                           six_site_min_plddt=min(local),pass_structural_gate=mean>=80 and ptm>=0.6 and min(local)>=70,
                           rank1_pdb=str(pdbs[0]),rank1_scores_json=str(scores[0])))
    write_table(run/'results/06_colabfold/colabfold_structural_summary_all38.tsv',merged)
    failed=[r['candidate_id'] for r in merged if not r['pass_structural_gate']]
    require(not failed,'Structural gate failed; original ranking requires 38 pass: '+', '.join(failed))
    write_table(run/'inputs/07_final_ranking/final_ranking_input.tsv',merged)
    print('PASS: 38/38 structures meet original gate')


if __name__=='__main__':
    run=Path(os.environ['CAS9_RUN']); action=sys.argv[1]
    if action=='shards': check_shards(run)
    elif action=='sequences': prepare_sequences(run)
    elif action in {'03','04'}: validate_scores(run,action)
    elif action in {'recover03','recover04'}: recover_partial_scores(run,action[-2:])
    elif action=='structure': summarize_structure(run)
    else: raise SystemExit('Unknown action: '+action)
