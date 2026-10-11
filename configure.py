#!/usr/bin/env python3
"""Write paths for this unpacked copy; optionally create missing Conda environments."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
install_missing = sys.argv[1:] == ["--install-missing"]
if sys.argv[1:] not in ([], ["--install-missing"]):
    raise SystemExit("用法：python3 configure.py [--install-missing]")

template = json.loads((root / "src/config.template.json").read_text(encoding="utf-8"))
config = dict(template)
current_file = root / "src/config.json"
if current_file.is_file():
    current = json.loads(current_file.read_text(encoding="utf-8"))
    for key in ("gpu", "esmc_mask_batch_size"):
        if key in current:
            config[key] = current[key]
config["gpu"] = os.environ.get("CAS9_GPU", config["gpu"])

conda = shutil.which("conda")
named_envs = {}
if conda:
    result = subprocess.run(
        [conda, "env", "list", "--json"], capture_output=True, text=True
    )
    if result.returncode == 0:
        for location in json.loads(result.stdout).get("envs", []):
            path = Path(location)
            named_envs[path.name] = path

env_paths = {}
for key in ("pammla", "progen2", "esmc", "colabfold"):
    env_name = f"cas9-{key}"
    override = os.environ.get(f"CAS9_{key.upper()}_ENV")
    local = root / ".envs" / env_name
    historical = Path(template["environments"][key])

    if override:
        chosen = Path(override).expanduser().resolve()
        if not (chosen / "bin/python").is_file():
            raise SystemExit(f"指定的环境不存在：{chosen}")
    else:
        candidates = (local, named_envs.get(env_name), historical)
        chosen = next(
            (path for path in candidates
             if path is not None and (path / "bin/python").is_file()),
            None,
        )
        if chosen is None:
            chosen = local
            if install_missing:
                if not conda:
                    raise SystemExit("未找到 Conda，无法自动创建四套运行环境")
                if chosen.exists():
                    raise SystemExit(f"环境目录存在但不完整，请先检查：{chosen}")
                spec = root / "requirements" / f"{key}.environment.yml"
                print(f"创建环境 {env_name}：{chosen}", flush=True)
                subprocess.run(
                    [conda, "env", "create", "--yes",
                     "--prefix", str(chosen), "--file", str(spec)],
                    check=True,
                )
                if not (chosen / "bin/python").is_file():
                    raise SystemExit(f"Conda 创建后仍未找到 Python：{chosen}")

    env_paths[key] = str(chosen)
    print(f"{key}：{chosen}", flush=True)

config.update(
    project_root=str(root / "resources"),
    reference_run=str(root),
    run_dir=str(root / "new_run"),
    reference_fasta=str(root / "inputs/reference_candidates.fasta"),
    train_variants=str(root / "inputs/train_variants.txt"),
    colabfold_cache=str(root / "resources/colabfold/cache"),
    msa_roots=[str(root / "inputs/06_colabfold_msa")],
    environments=env_paths,
)
current_file.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print("配置已更新：", current_file)
