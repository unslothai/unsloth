#!/bin/bash
# One-off e2e: exercise _install_torch_resumable_wheels against the CPU index.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"
TMPROOT="$(mktemp -d)"
trap 'rm -rf "$TMPROOT"' EXIT

uv venv "$TMPROOT/venv" --python 3.12 >/dev/null
VENV_PY="$TMPROOT/venv/bin/python"
STUDIO_HOME="$TMPROOT/studio"
mkdir -p "$STUDIO_HOME/share"

extract() { sed -n "/^$1()/,/^}/p" "$INSTALL_SH"; }

eval "$(extract download)"
eval "$(extract download_resumable)"
substep() { echo "SUBSTEP: $*"; }
run_install_cmd_retry() { shift; "$@"; }
eval "$(extract _pytorch_wheel_arch_suffix)"
eval "$(extract _constraint_ver_prefix)"
eval "$(extract _pytorch_simple_listing)"
eval "$(extract _pick_simple_index_wheel)"
eval "$(extract _resolve_simple_index_wheel_url)"
eval "$(extract _resolve_torch_wheel_versions)"
eval "$(extract _install_torch_resumable_wheels)"

export TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
export TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
export TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
export TORCHAUDIO_CONSTRAINT="torchaudio>=2.6,<2.11.0"
export _VENV_PY="$VENV_PY"
export STUDIO_HOME="$STUDIO_HOME"
export _ARCH="$(uname -m)"

echo "=== resumable torch install (cpu index) ==="
_install_torch_resumable_wheels \
    "$TORCH_CONSTRAINT" "$TORCHVISION_CONSTRAINT" "$TORCHAUDIO_CONSTRAINT"

echo "=== verify imports ==="
"$VENV_PY" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
"

echo "=== cached wheels ==="
ls -lh "$STUDIO_HOME/share/torch-wheel-cache"

echo "=== resume check: re-run should reuse partial cache ==="
_install_torch_resumable_wheels \
    "$TORCH_CONSTRAINT" "$TORCHVISION_CONSTRAINT" "$TORCHAUDIO_CONSTRAINT" \
    --force-reinstall

echo "OK"
