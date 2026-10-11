#!/usr/bin/env python3
"""Check or download the exact model files listed in WEIGHT_MANIFEST.tsv."""
import csv
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models/WEIGHT_MANIFEST.tsv"
BASE_URL = os.environ.get(
    "CAS9_WEIGHT_BASE_URL", ""
).rstrip("/")
CHUNK = 8 * 1024 * 1024


def matches(path, size, expected_hash):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == expected_hash


def load_manifest():
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["relative_path", "size_bytes", "sha256"]:
            raise RuntimeError("权重清单表头不正确")
        rows = list(reader)

    if len(rows) != 30:
        raise RuntimeError(f"权重清单应有 30 个文件，实际 {len(rows)} 个")

    entries = []
    seen = set()
    for row in rows:
        name = row["relative_path"]
        parts = name.split("/")
        if (not name.startswith("resources/") or
                any(part in ("", ".", "..") for part in parts) or
                "\\" in name or name in seen):
            raise RuntimeError(f"不安全或重复的清单路径：{name}")
        seen.add(name)
        size = int(row["size_bytes"])
        expected_hash = row["sha256"].lower()
        if size < 1 or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(f"清单大小或 SHA256 无效：{name}")
        entries.append((name, size, expected_hash))
    return entries


def fetch(name, size, expected_hash):
    destination = ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.commonpath((str(ROOT), str(destination.parent.resolve()))) != str(ROOT):
        raise RuntimeError(f"目标目录超出提交包：{name}")
    if destination.is_symlink():
        raise RuntimeError(f"目标文件是符号链接：{name}")

    partial = destination.with_name(destination.name + ".part")
    url = BASE_URL + "/" + quote(name, safe="/")
    print(f"下载：{name}（{size / 1024**2:.1f} MiB）", flush=True)

    for attempt in range(1, 4):
        if matches(partial, size, expected_hash):
            os.replace(partial, destination)
            print(f"校验通过：{name}", flush=True)
            return

        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= size:
            partial.unlink()
            offset = 0

        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-"):
                    raise RuntimeError("服务器返回的续传范围不正确")

                written = offset if append else 0
                reported = written // (128 * 1024**2)
                with partial.open("ab" if append else "wb") as output:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
                        if written > size:
                            raise RuntimeError("下载大小超过清单记录")
                        if written // (128 * 1024**2) > reported:
                            reported = written // (128 * 1024**2)
                            print(f"  {written / size:.0%}", flush=True)
        except (OSError, RuntimeError) as error:
            print(f"第 {attempt} 次下载中断：{error}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"下载失败：{name}") from error
            time.sleep(2)
            continue

        if matches(partial, size, expected_hash):
            os.replace(partial, destination)
            print(f"校验通过：{name}", flush=True)
            return

        if partial.stat().st_size >= size:
            partial.unlink()
        print(f"第 {attempt} 次下载未通过 SHA256/大小校验", flush=True)
        if attempt == 3:
            raise RuntimeError(f"校验失败：{name}")
        time.sleep(2)


def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else "fetch"
    if mode not in ("check", "fetch"):
        raise RuntimeError("用法：download_weights.py [check|fetch]")

    missing = [
        item for item in load_manifest()
        if not matches(ROOT / item[0], item[1], item[2])
    ]
    if mode == "check":
        print(f"权重检查：30 个文件，{len(missing)} 个缺失或校验失败")
        for name, _, _ in missing:
            print("缺失或不匹配：", name)
        return 1 if missing else 0

    if missing and not BASE_URL:
        raise RuntimeError(
            "缺少模型文件。请按 models/MODEL_CARD.md 从官方来源下载并放入指定目录，"
            "或设置 CAS9_WEIGHT_BASE_URL 指向你有权使用的可信镜像。"
        )

    for item in missing:
        fetch(*item)
    print("PASS：30 个模型文件均与 SHA256 清单一致")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("停止：", error, file=sys.stderr)
        sys.exit(1)
