#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  run|step)
    python3 "$ROOT/configure.py" --install-missing
    python3 "$ROOT/models/download_weights.py" fetch
    ;;
  check)
    python3 "$ROOT/configure.py"
    python3 "$ROOT/models/download_weights.py" check
    ;;
  status)
    python3 "$ROOT/configure.py"
    ;;
  weights)
    exec python3 "$ROOT/models/download_weights.py" fetch
    ;;
esac

exec python3 "$ROOT/src/pipeline.py" "$@"
