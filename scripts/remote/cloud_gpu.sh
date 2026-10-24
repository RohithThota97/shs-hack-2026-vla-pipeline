#!/usr/bin/env bash
# One script for a rented cloud GPU (RunPod / Lambda / any Linux + NVIDIA machine with SSH). Run on the laptop,
# from inside the repo. Uses scripts/remote/gpu.sh (on the pod) and scripts/remote/laptop.sh (here).
#
#   ./scripts/remote/cloud_gpu.sh up "ssh root@1.2.3.4 -p 22045 -i ~/.ssh/hackpipe_gpu"
#        everything: SSH alias, tools + Python 3.12 on the pod, repo at this laptop's commit, data copy (from this
#        laptop, ~4.4 GB), pipeline install + GPU check, registry handed over, console started, then the tunnel →
#        http://localhost:8501. Safe to re-run: finished steps are skipped.
#   ./scripts/remote/cloud_gpu.sh connect      open the console again (tunnel; Ctrl+C closes it, training goes on)
#   ./scripts/remote/cloud_gpu.sh status       pod: console, writer, datasets, running jobs, GPU
#   ./scripts/remote/cloud_gpu.sh down         BEFORE deleting the pod: the last checkpoint of every model →
#                                              ~/hackpipe_checkpoints/, registry + runs back to this laptop
#
# Registry: one writer at a time. `up` hands the laptop's registry to the pod once (marker ~/.hackpipe/cloud_writer
# on this laptop); `down` takes it back and removes the marker; the next `up` hands it over again.
# Remote folder: /workspace when the pod has it (RunPod's volume, survives a stop), else the remote home; set
# REMOTE_BASE=/abs/path to choose. Data goes from this laptop to the pod over SSH: no public links, nothing on HF.
set -euo pipefail
cd "$(dirname "$0")/../.."
ALIAS=hackpipe-gpu
REPO_URL=https://github.com/RohithThota97/shs-hack-2026-vla-pipeline.git
# datasets copied to the pod; e.g. HACKPIPE_DATASETS="fr3_duo_handover_84 fr3_duo_handover_lighting10" for a
# handover-only run (the two pinned sources are required, the cube data only for runs that train on it)
read -r -a DATASETS <<< "${HACKPIPE_DATASETS:-fr3_duo_handover_84 fr3_duo_handover_lighting10 fr3_duo_rand500}"
PORT=${PORT:-8501}
MARKER="$HOME/.hackpipe/cloud_writer"
CONSOLE_PATTERN="streamlit run .*hackpipe"      # same pattern as laptop.sh's no_local_console

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "=== $*"; }
r() { ssh "$ALIAS" "$@"; }                     # a key with a passphrase: `ssh-add ~/.ssh/hackpipe_gpu` once first

remote_paths() {                               # BASE, REMOTE_REPO (laptop.sh reads it), DATA, UV
  if [[ -n "${REMOTE_BASE:-}" ]]; then
    BASE=$REMOTE_BASE
  else
    BASE=$(r 'if [ -d /workspace ] && [ -w /workspace ]; then echo /workspace; else echo "$HOME"; fi') \
      || die "cannot SSH into $ALIAS"
  fi
  [[ "$BASE" == /* ]] || die "REMOTE_BASE must be an absolute path, got $BASE"
  export REMOTE_REPO="$BASE/shs-hack-2026-vla-pipeline"
  DATA="$BASE/hackpipe_data"
  UV="$BASE/.uv/bin/uv"                        # on the same disk as the venv: a restarted pod keeps both
}

write_alias() {     # "ssh root@IP -p PORT -i KEY" → Host hackpipe-gpu in ~/.ssh/config (edited in place)
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
  mkdir -p ~/.ssh
  [[ -e ~/.ssh/config ]] || { touch ~/.ssh/config; chmod 600 ~/.ssh/config; }
  local begins ends
  begins=$(grep -c '^# >>> hackpipe-gpu' ~/.ssh/config || true)
  ends=$(grep -c '^# <<< hackpipe-gpu' ~/.ssh/config || true)
  [[ "$begins" == "$ends" ]] || die "~/.ssh/config has a hackpipe-gpu block without its end marker: fix it by hand"
  local tmp; tmp=$(mktemp)
  awk '/^# >>> hackpipe-gpu/{skip=1} !skip{print} /^# <<< hackpipe-gpu/{skip=0}' ~/.ssh/config > "$tmp"
  cat >> "$tmp" <<EOF
# >>> hackpipe-gpu (scripts/remote/cloud_gpu.sh)
Host $ALIAS
  HostName ${user_host#*@}
  User ${user_host%@*}
  Port $port
  IdentityFile $key
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
# <<< hackpipe-gpu
EOF
  cat "$tmp" > ~/.ssh/config                   # in place: keeps a dotfiles symlink and the file's permissions
  rm -f "$tmp"
  echo "~/.ssh/config: Host $ALIAS → ${user_host} port $port"
}

local_console_stopped() {
  pgrep -f "$CONSOLE_PATTERN" >/dev/null || return 0
  read -r -p "The console is running on this laptop; stop it so the pod becomes the only registry writer? [y/N] " a
  [[ "$a" =~ ^[Yy] ]] || die "stop the laptop console first"
  pkill -f "$CONSOLE_PATTERN" || true
  for _ in $(seq 20); do pgrep -f "$CONSOLE_PATTERN" >/dev/null || return 0; sleep 1; done
  die "the laptop console is still running"
}

pod_jobs() {        # running / queued job ids on the pod ("" = none); fails if the pod can't be asked
  r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh status" | sed -n 's/^running jobs: //p'
}

up() {
  [[ -n "${1:-}" ]] || die 'usage: cloud_gpu.sh up "ssh root@IP -p PORT -i KEY"'
  step "1/7 SSH alias"
  write_alias "$1"
  r true || die "cannot SSH into the pod (is it running? is the public key in the cloud account?)"
  r nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no NVIDIA GPU visible on the pod"
  remote_paths
  echo "remote folder: $BASE"

  step "2/7 tools + Python 3.12 on the pod"
  r "set -e
     command -v git >/dev/null && command -v ffmpeg >/dev/null && command -v rsync >/dev/null || {
       export DEBIAN_FRONTEND=noninteractive
       SUDO=\$( [ \$(id -u) -eq 0 ] || echo sudo )
       \$SUDO apt-get update -qq && \$SUDO apt-get install -y -qq git ffmpeg rsync >/dev/null; }
     mkdir -p $BASE/.uv
     [ -x $UV ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$BASE/.uv/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
     UV_PYTHON_INSTALL_DIR=$BASE/.uv/python $UV python install 3.12 >/dev/null
     echo python: \$(UV_PYTHON_INSTALL_DIR=$BASE/.uv/python $UV python find 3.12)"

  step "3/7 repo on the pod ($REMOTE_REPO) at this laptop's commit"
  local sha; sha=$(git rev-parse HEAD)
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo "WARNING: uncommitted changes here are not on the pod"
  r "set -e
     [ -d $REMOTE_REPO/.git ] || git clone -q $REPO_URL $REMOTE_REPO
     git -C $REMOTE_REPO fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
     if git -C $REMOTE_REPO cat-file -e $sha^{commit} 2>/dev/null; then git -C $REMOTE_REPO checkout -q $sha
     else echo 'WARNING: commit $sha is not on GitHub (push it): using origin/main'; git -C $REMOTE_REPO checkout -q origin/main; fi
     git -C $REMOTE_REPO log --oneline -1"

  step "4/7 data from this laptop → $DATA (resumes if interrupted)"
  r "mkdir -p $DATA"
  for d in "${DATASETS[@]}"; do
    [[ -e "data/raw/$d/meta/info.json" ]] || die "data/raw/$d is missing on this laptop"
    echo "-- $d ($(du -shL "data/raw/$d" | cut -f1))"
    rsync -a -L --partial --progress -e ssh "data/raw/$d/" "$ALIAS:$DATA/$d/" | tail -1
  done

  step "5/7 pipeline install + checks on the pod (first time: ~5 min)"
  r "cd $REMOTE_REPO && PYTHON=\$(UV_PYTHON_INSTALL_DIR=$BASE/.uv/python $UV python find 3.12) ./scripts/remote/gpu.sh setup $DATA"

  step "6/7 registry → pod, console on the pod"
  if [[ -f "$MARKER" ]]; then
    echo "the pod already holds the registry since $(cat "$MARKER") (hand-over marker $MARKER): not sent again"
    r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh console"
  else
    local_console_stopped
    ./scripts/remote/laptop.sh send "$ALIAS"
    r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh stop >/dev/null; ./scripts/remote/gpu.sh console ~/hackpipe_state.zip"
    mkdir -p "$(dirname "$MARKER")"
    echo "$(date '+%Y-%m-%d %H:%M') ($ALIAS)" > "$MARKER"
  fi

  step "7/7 tunnel → http://localhost:$PORT (keep this window open; Ctrl+C closes only the tunnel)"
  echo "In the console: Train → timing run → step budget → Start training. The Train page shows the pod's GPU."
  ./scripts/remote/laptop.sh connect "$ALIAS"
}

connect() {         # laptop.sh refuses when the local port is busy (e.g. the laptop console): PORT=8502 then
  remote_paths
  ./scripts/remote/laptop.sh connect "$ALIAS"
}

status() {
  remote_paths
  r "cd $REMOTE_REPO && ./scripts/remote/gpu.sh status"
}

down() {
  remote_paths
  local jobs
  jobs=$(pod_jobs) || die "cannot reach the pod — don't delete it yet"
  [[ -z "$jobs" ]] || die "jobs still running on the pod: $jobs — wait for them (cloud_gpu.sh status) first"
  step "1/2 checkpoints → ~/hackpipe_checkpoints/ (the last one of every model: weights + config)"
  mkdir -p ~/hackpipe_checkpoints
  local models
  models=$(r "cd $REMOTE_REPO/outputs/train 2>/dev/null || exit 0
              for d in */; do [ -e \$d/checkpoints/last/pretrained_model/model.safetensors ] && echo \${d%/}; done; true") \
    || die "cannot list the pod's checkpoints — don't delete it yet"
  for m in $models; do
    echo "-- $m"
    rsync -a -L --partial --progress -e ssh "$ALIAS:$REMOTE_REPO/outputs/train/$m/checkpoints/last/pretrained_model/" \
      "$HOME/hackpipe_checkpoints/$m/pretrained_model/" | tail -1
  done
  [[ -n "$models" ]] || echo "no finished checkpoints on the pod"
  step "2/2 registry + runs back to this laptop (stops the pod's console, releases its writer role)"
  local_console_stopped
  ./scripts/remote/laptop.sh fetch "$ALIAS"
  rm -f "$MARKER"
  echo
  echo "Done: this laptop is the writer again. Next:"
  echo "  - upload ~/hackpipe_checkpoints/<model> to Google Drive (keep the last 2), then"
  echo "  - TERMINATE the pod in the cloud console (it is billed until you do)"
  echo "  - start the laptop console again: .venv/bin/hackpipe console"
}

case "${1:-}" in
  up) shift; up "$@" ;;
  connect) connect ;;
  status) status ;;
  down) down ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
