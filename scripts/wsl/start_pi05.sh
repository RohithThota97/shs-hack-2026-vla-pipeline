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
mkdir -p "$REPO/outputs/diagnose"
pkill -f "hackpipe.*serve --pi05" 2>/dev/null || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
exec "$DIR/.venv/bin/python" -u -m hackpipe serve --pi05 "$CKPT" --port "$PORT" \
  > "$REPO/outputs/diagnose/pi05_server.log" 2>&1 < /dev/null
