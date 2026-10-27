#!/bin/bash
# Install uv, clone openpi and create its environment (WSL / Linux, for `hackpipe serve --pi05` and
# `hackpipe diagnose pi05`). From our team repo's tools/wsl_install_openpi.sh.
#   bash scripts/wsl/install_openpi.sh [openpi_dir]          (default ~/openpi)
set -e
DIR=${1:-$HOME/openpi}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
if [ ! -d "$DIR" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$DIR"
fi
cd "$DIR"
git log --oneline -1
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# hackpipe itself (light core: pyyaml, numpy, pandas, pyarrow) for `python -m hackpipe serve --pi05`
uv pip install -e "$REPO"
.venv/bin/python -c "import jax, openpi; print('jax', jax.__version__, jax.devices())"
echo INSTALL_OK
