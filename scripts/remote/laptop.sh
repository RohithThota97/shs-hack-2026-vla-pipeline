#!/usr/bin/env bash
# Run on the laptop (macOS / Linux), from inside the repo. The GPU side is scripts/remote/gpu.sh.
#
#   ./scripts/remote/laptop.sh send USER@HOST      export this registry and copy it to the GPU machine
#   ./scripts/remote/laptop.sh connect USER@HOST   SSH tunnel → opens http://localhost:8501 with the console token
#   ./scripts/remote/laptop.sh fetch USER@HOST     GPU machine hands over (console stopped, writer released), its
#                                                  state (models, evals) is imported here
#
# Needs SSH access to the GPU machine (ssh USER@HOST works). The repo there: REMOTE_REPO, a path relative to the
# remote home or absolute (default shs-hack-2026-vla-pipeline). No SSH? Move the zips through Drive and use
# `gpu.sh console … --public` instead. The laptop console must be stopped for send / fetch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT=${PORT:-8501}
REMOTE_PORT=${REMOTE_PORT:-8501}        # the GPU console's port (gpu.sh PORT)
REMOTE_REPO=${REMOTE_REPO:-shs-hack-2026-vla-pipeline}
HP=.venv/bin/hackpipe

die() { echo "ERROR: $*" >&2; exit 1; }
host=${2:-}
[[ -n "$host" || -z "${1:-}" ]] || die "usage: laptop.sh ${1} USER@HOST"
remote() { ssh "$host" "cd $(printf %q "$REMOTE_REPO") && $*"; }
no_local_console() {
  if pgrep -f "streamlit run .*hackpipe" >/dev/null; then
    die "the console is running on this laptop: stop it first (one machine writes the registry at a time)"
  fi
}
versions() {
  .venv/bin/python -c "
from hackpipe.common.registry import Registry
r = Registry()
print('datasets:', ', '.join(f\"{d['version']} {d['status']}\" for d in r.datasets()) or 'none')
print('models:  ', ', '.join(f\"{m['version']} {m['status']}\" for m in r.models()) or 'none')"
}

send() {
  no_local_console
  local zip; zip="$(mktemp -d)/hackpipe_state.zip"
  $HP state export --out "$zip"
  scp "$zip" "$host:hackpipe_state.zip"
  echo
  echo "Sent. Don't train / approve on this laptop until you fetch the state back."
  echo "On the GPU machine:  cd $REMOTE_REPO && ./scripts/remote/gpu.sh console ~/hackpipe_state.zip"
}

connect() {
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $PORT is already in use here (a laptop console?): stop it, or PORT=8502 ./scripts/remote/laptop.sh connect $host"
  fi
  local token
  token=$(remote ./scripts/remote/gpu.sh token) || die "no console token on $host: start the console there first"
  local url="http://localhost:$PORT/?token=$token"
  echo "Tunnel open — console at $url"
  echo "(keep this window open; Ctrl+C closes the tunnel, training on the GPU machine keeps running)"
  ( sleep 3; command -v open >/dev/null && open "$url" || xdg-open "$url" 2>/dev/null || true ) &
  ssh -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$REMOTE_PORT" "$host"
}

fetch() {
  no_local_console
  remote ./scripts/remote/gpu.sh export
  mkdir -p "$HOME/Desktop"
  scp "$host:gpu_state.zip" "$HOME/Desktop/gpu_state.zip"
  $HP state import "$HOME/Desktop/gpu_state.zip" --replace     # the laptop registry is backed up first
  versions
  echo
  echo "Imported: this laptop is the writer again. Checkpoints stay on the GPU machine."
}

case "${1:-}" in
  send) send ;;
  connect) connect ;;
  fetch) fetch ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
