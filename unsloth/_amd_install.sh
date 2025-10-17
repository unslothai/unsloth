#!/usr/bin/env bash
# _amd_install.sh
# Non-interactive installer: build tools, PyTorch (ROCm 6.4), bitsandbytes (HIP), and Unsloth from source.
# Usage:
#   bash _amd_install.sh
#

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends build-essential cmake git

pip install \
    torch==2.8.0 torchvision torchaudio torchao==0.13.0 xformers \
    --index-url https://download.pytorch.org/whl/rocm6.4

WORKDIR="$(pwd)"
TMPDIR="$(mktemp -d)"
cd "$TMPDIR"
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
arch
cmake -DCOMPUTE_BACKEND=hip -S .
make -j"$(nproc)"
pip install .
cd "$WORKDIR"
rm -rf "$TMPDIR"

pip install --no-deps unsloth unsloth-zoo
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
