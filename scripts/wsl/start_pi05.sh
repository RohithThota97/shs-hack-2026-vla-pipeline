#!/bin/bash
# Serve an openpi pi0.5 checkpoint to the Isaac Sim scripts (foreground; keep this terminal open). From our team
# repo's tools/wsl_start_pi05.sh.
#   bash scripts/wsl/start_pi05.sh <checkpoint_dir> [port] [openpi_dir]
# checkpoint_dir has params/ and assets/fr3_duo/. Log: outputs/diagnose/pi05_server.log
set -e
CKPT=${1:?usage: start_pi05.sh <checkpoint_dir> [port] [openpi_dir]}
PORT=${2:-8090}
DIR=${3:-$HOME/openpi}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CKPT=$(cd "$CKPT" && pwd) || { echo "no checkpoint folder $1" >&2; exit 1; }   # absolute: we cd to the repo below
[ -d "$CKPT/params" ] || { echo "$CKPT has no params/ (an openpi checkpoint has params/ + assets/fr3_duo/)" >&2; exit 1; }
mkdir -p "$REPO/outputs/diagnose"
echo "pi0.5 from $CKPT on :$PORT — log: $REPO/outputs/diagnose/pi05_server.log"
pkill -f "hackpipe.*serve --pi05" 2>/dev/null || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
exec "$DIR/.venv/bin/python" -u -m hackpipe serve --pi05 "$CKPT" --port "$PORT" \
  > "$REPO/outputs/diagnose/pi05_server.log" 2>&1 < /dev/null
