#!/usr/bin/env bash
# Run on the GPU machine (Linux + NVIDIA), from inside the cloned repo. The laptop side is scripts/remote/laptop.sh.
#
#   ./scripts/remote/gpu.sh setup <data folder>      once: venv, LeRobot 0.6.1, datasets linked as data/raw/<name>
#   ./scripts/remote/gpu.sh console [state.zip]      import the laptop's state (if given), start the console
#   ./scripts/remote/gpu.sh console [state.zip] --public    same, plus a public cloudflared link (no SSH needed)
#   ./scripts/remote/gpu.sh status                   console running? writer? datasets / models
#   ./scripts/remote/gpu.sh export                   registry + small outputs → ~/gpu_state.zip (for the laptop)
#   ./scripts/remote/gpu.sh stop                     stop the console (training jobs it started keep running)
#
# <data folder>: the dataset zips / folders downloaded from the team's Google Drive (never from HF / GitHub).
# The console listens on 127.0.0.1 only; reach it with `laptop.sh connect` (SSH tunnel) or the --public link.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3.12}
PORT=${PORT:-8501}
STATE_DIR=${HACKPIPE_REMOTE_DIR:-$HOME/.hackpipe}
LEROBOT_SPEC="lerobot[smolvla,dataset]==0.6.1"
HP=.venv/bin/hackpipe
mkdir -p "$STATE_DIR"

die() { echo "ERROR: $*" >&2; exit 1; }
alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

setup() {
  local data=${1:-}
  [[ -n "$data" ]] || die "usage: gpu.sh setup <folder with the dataset zips / folders>"
  [[ -d "$data" ]] || die "no folder $data"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  command -v "$PY" >/dev/null || die "$PY not found (Python 3.12 needed; or run with PYTHON=/path/to/python3.12)"
  command -v ffmpeg >/dev/null || echo "WARNING: no ffmpeg on PATH (sudo apt install ffmpeg) — video decoding may fail"
  [[ -d .venv ]] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e ".[dev,video,hub,console]" "$LEROBOT_SPEC" matplotlib
  .venv/bin/pip uninstall -y -q torchao 2>/dev/null || true     # an old torchao breaks LeRobot's start-up
  .venv/bin/lerobot-train --help >/dev/null || die "lerobot-train does not start (see the error above)"
  .venv/bin/python - <<'EOF'
import av, torch
assert av.__version__.startswith("15."), f"PyAV {av.__version__}: LeRobot 0.6.1 needs av>=15,<16"
assert torch.cuda.is_available(), "torch does not see the GPU (driver / CUDA build mismatch)"
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))
EOF
  echo "== linking datasets from $data"
  .venv/bin/python - "$data" <<'EOF'
import json, sys
from pathlib import Path
from hackpipe.pipeline.watch import link, linked, pinned_names, unlinked
found = unlinked(dirs=[Path(sys.argv[1]).expanduser().resolve()])
for d in found:
    print(f"linked data/raw/{link(d['path'], d['name']).name}: {d['episodes']} episodes, {d['frames']} frames "
          f"← {d['path']}", flush=True)
names = {Path(p).name for p in linked().values()}
missing = [n for n in pinned_names().values() if n not in names]
if missing:
    sys.exit(f"missing {missing}: the registry's demo IDs need these exact datasets (episodes, frames must match "
             "configs/project.yaml → dataset.pinned_names)")
print("data/raw:", sorted(names))
EOF
  .venv/bin/python -m pytest -q 2>&1 | tail -1
  echo
  echo "Setup done. Next: get hackpipe_state.zip from the laptop (laptop.sh send), then"
  echo "  ./scripts/remote/gpu.sh console ~/hackpipe_state.zip"
}

console() {
  local zip="" public=0
  for a in "$@"; do [[ "$a" == "--public" ]] && public=1 || zip=$a; done
  [[ -x $HP ]] || die "run ./scripts/remote/gpu.sh setup <data folder> first"
  if alive "$STATE_DIR/console.pid"; then
    echo "console already running (pid $(cat "$STATE_DIR/console.pid")); ./scripts/remote/gpu.sh stop to restart"
  else
    if [[ -n "$zip" ]]; then
      [[ -f "$zip" ]] || die "no file $zip"
      $HP state import "$zip" --replace          # the registry here is backed up first; this machine becomes writer
    fi
    $HP dataset list | grep -E '"version"|"status"' | paste - - || true
    .venv/bin/python -c "import secrets; print(secrets.token_urlsafe(16))" > "$STATE_DIR/console_token"
    chmod 600 "$STATE_DIR/console_token"
    HACKPIPE_CONSOLE_TOKEN=$(cat "$STATE_DIR/console_token") STREAMLIT_SERVER_ADDRESS=127.0.0.1 \
      STREAMLIT_SERVER_HEADLESS=true nohup $HP console --port "$PORT" > "$STATE_DIR/console.log" 2>&1 &
    echo $! > "$STATE_DIR/console.pid"
    sleep 5
    alive "$STATE_DIR/console.pid" || { tail -20 "$STATE_DIR/console.log"; die "console did not start"; }
    echo "console running on 127.0.0.1:$PORT (log $STATE_DIR/console.log)"
  fi
  local token; token=$(cat "$STATE_DIR/console_token")
  if (( public )); then
    local cf="$STATE_DIR/cloudflared"
    [[ -x "$cf" ]] || { curl -fsSL -o "$cf" \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x "$cf"; }
    alive "$STATE_DIR/tunnel.pid" || {
      nohup "$cf" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$STATE_DIR/tunnel.log" 2>&1 &
      echo $! > "$STATE_DIR/tunnel.pid"; }
    local url=""
    for _ in $(seq 60); do
      url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$STATE_DIR/tunnel.log" | head -1 || true)
      [[ -n "$url" ]] && break; sleep 1
    done
    [[ -n "$url" ]] || die "no tunnel URL yet — see $STATE_DIR/tunnel.log"
    echo
    echo "Public console link (send it privately, only to the team):"
    echo "  $url/?token=$token"
  else
    echo
    echo "On the laptop:  ./scripts/remote/laptop.sh connect $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)"
  fi
}

status() {
  alive "$STATE_DIR/console.pid" && echo "console: running (pid $(cat "$STATE_DIR/console.pid"))" || echo "console: stopped"
  alive "$STATE_DIR/tunnel.pid" && echo "public tunnel: running" || true
  $HP writer
  $HP dataset list | grep -E '"version"|"status"' | paste - - || true
  $HP jobs 2>/dev/null | tail -5 || true
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
}

export_state() {
  $HP state export "$HOME/gpu_state.zip"
  echo
  echo "Stop writing here now (don't train / approve) until the laptop sends a new state back."
  echo "On the laptop:  ./scripts/remote/laptop.sh fetch $(whoami)@$(hostname)"
  echo "(no SSH: copy ~/gpu_state.zip to Drive; on the laptop: .venv/bin/hackpipe state import <zip> --replace)"
}

stop() {
  for f in console tunnel; do
    if alive "$STATE_DIR/$f.pid"; then       # hackpipe console runs streamlit as a child: stop both
      pkill -P "$(cat "$STATE_DIR/$f.pid")" 2>/dev/null || true
      kill "$(cat "$STATE_DIR/$f.pid")" 2>/dev/null || true
      echo "stopped $f"
    fi
    rm -f "$STATE_DIR/$f.pid"
  done
}

case "${1:-}" in
  setup) shift; setup "$@" ;;
  console) shift; console "$@" ;;
  status) status ;;
  export) export_state ;;
  stop) stop ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
