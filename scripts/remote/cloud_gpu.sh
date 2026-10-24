#!/usr/bin/env bash
# One script for a rented cloud GPU (RunPod / Lambda / any Linux + NVIDIA machine with SSH). Run on the laptop,
# from inside the repo. Uses scripts/remote/gpu.sh (on the pod) and scripts/remote/laptop.sh (here).
#
#   ./scripts/remote/cloud_gpu.sh up "ssh root@1.2.3.4 -p 22045 -i ~/.ssh/hackpipe_gpu"
#        everything: SSH alias, tools + Python 3.12 on the pod, repo, data copy (from this laptop, ~4.4 GB),
#        pipeline install + GPU check, registry sent, console started, then the tunnel → http://localhost:8501
#        (safe to re-run: finished steps are skipped; the registry is sent only once)
#   ./scripts/remote/cloud_gpu.sh connect      open the console again (tunnel; Ctrl+C closes it, training goes on)
#   ./scripts/remote/cloud_gpu.sh status       pod: console, writer, datasets, running jobs, GPU
#   ./scripts/remote/cloud_gpu.sh down         BEFORE deleting the pod: the last checkpoint of every model →
#                                              ~/hackpipe_checkpoints/, registry + runs back to this laptop
#
# Paste the "SSH over exposed TCP" line from the pod's Connect tab (RunPod) or `ssh ubuntu@IP` (Lambda).
# Data goes straight from this laptop to the pod (rsync over SSH): no public links, nothing on HF / GitHub.
set -euo pipefail
cd "$(dirname "$0")/../.."
ALIAS=hackpipe-gpu
REPO_URL=https://github.com/RohithThota97/shs-hack-2026-vla-pipeline.git
BASE=${REMOTE_BASE:-/workspace}          # RunPod's volume disk survives a pod stop; elsewhere an absolute path,
                                         # e.g. export REMOTE_BASE=/home/ubuntu (Lambda)
export REMOTE_REPO="$BASE/shs-hack-2026-vla-pipeline"
DATA="$BASE/hackpipe_data"
DATASETS=(fr3_duo_handover_84 fr3_duo_handover_lighting10 fr3_duo_rand500)
PORT=${PORT:-8501}

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "=== $*"; }
r() { ssh "$ALIAS" "$@"; }              # a key with a passphrase: run `ssh-add ~/.ssh/hackpipe_gpu` once first

write_alias() {     # "ssh root@IP -p PORT -i KEY" → Host hackpipe-gpu in ~/.ssh/config
  local line=$1 user_host="" port=22 key=""
  read -r -a w <<< "$line"
  local i=0
  while (( i < ${#w[@]} )); do
    case "${w[$i]}" in
      ssh) ;;
      -p) i=$((i + 1)); port=${w[$i]} ;;
      -i) i=$((i + 1)); key=${w[$i]} ;;
      -*) ;;
      *@*) user_host=${w[$i]} ;;
    esac
    i=$((i + 1))
  done
  [[ -n "$user_host" ]] || die "no user@host in: $line"
  key=${key:-$HOME/.ssh/hackpipe_gpu}
  key=${key/#\~/$HOME}
  [[ -f "$key" ]] || die "SSH key $key not found (ssh-keygen -t ed25519 -f ~/.ssh/hackpipe_gpu, then add the .pub to the cloud account)"
  mkdir -p ~/.ssh && touch ~/.ssh/config && chmod 600 ~/.ssh/config
  # replace an earlier block of ours
  awk '/^# >>> hackpipe-gpu/{skip=1} !skip{print} /^# <<< hackpipe-gpu/{skip=0}' ~/.ssh/config > ~/.ssh/config.tmp
  cat >> ~/.ssh/config.tmp <<EOF
# >>> hackpipe-gpu (scripts/remote/cloud_gpu.sh)
Host $ALIAS
  HostName ${user_host#*@}
  User ${user_host%@*}
  Port $port
  IdentityFile $key
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
# <<< hackpipe-gpu
EOF
  mv ~/.ssh/config.tmp ~/.ssh/config
  echo "~/.ssh/config: Host $ALIAS → ${user_host} port $port"
}

local_console_stopped() {
  if pgrep -f "streamlit run .*hackpipe" >/dev/null; then
    read -r -p "The console is running on this laptop; stop it so the pod becomes the only registry writer? [y/N] " a
    [[ "$a" =~ ^[Yy] ]] || die "stop the laptop console first"
    pkill -f "streamlit run .*hackpipe" || true
    sleep 2
  fi
}

up() {
  [[ -n "${1:-}" ]] || die 'usage: cloud_gpu.sh up "ssh root@IP -p PORT -i KEY"'
  step "1/7 SSH alias"
  write_alias "$1"
  r true || die "cannot SSH into the pod (is it running? is the public key in the cloud account?)"
  r nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no NVIDIA GPU visible on the pod"

  step "2/7 tools + Python 3.12 on the pod"
  r "set -e
     command -v git >/dev/null && command -v ffmpeg >/dev/null && command -v rsync >/dev/null || {
       export DEBIAN_FRONTEND=noninteractive
       SUDO=\$( [ \$(id -u) -eq 0 ] || echo sudo )
       \$SUDO apt-get update -qq && \$SUDO apt-get install -y -qq git ffmpeg rsync >/dev/null; }
     [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
     ~/.local/bin/uv python install 3.12 >/dev/null
     echo python: \$(~/.local/bin/uv python find 3.12)"

  step "3/7 repo on the pod ($REMOTE_REPO)"
  r "set -e; mkdir -p $BASE
     if [ -d $REMOTE_REPO/.git ]; then git -C $REMOTE_REPO pull -q --ff-only; else git clone -q $REPO_URL $REMOTE_REPO; fi
     git -C $REMOTE_REPO log --oneline -1"

  step "4/7 data from this laptop → $DATA (resumes if interrupted)"
  r "mkdir -p $DATA"
  for d in "${DATASETS[@]}"; do
    [[ -e "data/raw/$d/meta/info.json" ]] || die "data/raw/$d is missing on this laptop"
    echo "-- $d ($(du -shL "data/raw/$d" | cut -f1))"
    rsync -a -L --partial --progress -e ssh "data/raw/$d/" "$ALIAS:$DATA/$d/" | tail -1
  done

  step "5/7 pipeline install + checks on the pod (first time: ~5 min)"
  r "cd $REMOTE_REPO && PYTHON=\$(~/.local/bin/uv python find 3.12) ./scripts/remote/gpu.sh setup $DATA"

  step "6/7 registry → pod, console on the pod"
  if r "test -f $REMOTE_REPO/registry/registry.db"; then
    echo "the pod already has a registry: not sent again (the pod is the writer; bring it back with: down)"
  else
    local_console_stopped
    ./scripts/remote/laptop.sh send "$ALIAS"
  fi
  if r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh status" | grep -q "console: running"; then
    echo "console already running on the pod"
  else
    local zip=""
    r "test -f ~/hackpipe_state.zip && ! test -f ~/.hackpipe/imported" && zip="~/hackpipe_state.zip"
    r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh console $zip && { [ -z '$zip' ] || touch ~/.hackpipe/imported; }"
  fi

  step "7/7 tunnel → http://localhost:$PORT (keep this window open; Ctrl+C closes only the tunnel)"
  echo "In the console: Train → timing run → step budget → Start training. The Train page shows the pod's GPU."
  connect
}

connect() {
  local_console_stopped
  ./scripts/remote/laptop.sh connect "$ALIAS"
}

status() {
  r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh status"
}

down() {
  step "1/2 checkpoints → ~/hackpipe_checkpoints/ (the last one of every model: weights + config)"
  mkdir -p ~/hackpipe_checkpoints
  local found=0
  for m in $(r "cd $REMOTE_REPO/outputs/train 2>/dev/null && for d in */; do [ -e \$d/checkpoints/last/pretrained_model/model.safetensors ] && echo \${d%/}; done" || true); do
    found=1
    echo "-- $m"
    rsync -a -L --partial --progress -e ssh "$ALIAS:$REMOTE_REPO/outputs/train/$m/checkpoints/last/pretrained_model/" \
      "$HOME/hackpipe_checkpoints/$m/pretrained_model/" | tail -1
  done
  (( found )) || echo "no finished checkpoints on the pod"
  step "2/2 registry + runs back to this laptop (stops the pod's console; refused while a job runs)"
  local_console_stopped
  ./scripts/remote/laptop.sh fetch "$ALIAS"
  echo
  echo "Done. Next:"
  echo "  - upload ~/hackpipe_checkpoints/<model> to Google Drive (keep the last 2), then"
  echo "  - TERMINATE the pod in the cloud console (it is billed until you do)"
  echo "  - start the laptop console again: .venv/bin/hackpipe console"
}

case "${1:-}" in
  up) shift; up "$@" ;;
  connect) connect ;;
  status) status ;;
  down) down ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
