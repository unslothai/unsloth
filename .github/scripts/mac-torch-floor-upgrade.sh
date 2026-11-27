#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Helpers for studio-mac-torch-floor-upgrade.yml, run against $UNSLOTH_STUDIO_HOME's venv.
#   probe          print "PROBE <python> <torch> <torchvision> <torchaudio> mps <bool>", check MPS SDPA
#   seed T TV TA   put the existing home on that torch trio (an older home, as a user would have)
set -euo pipefail

VPY="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python"

_probe() {
    "$VPY" - <<'PY'
import sys
import torch, torchvision, torchaudio
mps = torch.backends.mps.is_available()
print("PROBE", sys.version.split()[0], torch.__version__, torchvision.__version__, torchaudio.__version__, "mps", mps)
if mps:
    q = torch.randn(1, 8, 1024, 64, dtype = torch.bfloat16)
    ref = torch.nn.functional.scaled_dot_product_attention(q.double(), q.double(), q.double())
    out = torch.nn.functional.scaled_dot_product_attention(q.to("mps"), q.to("mps"), q.to("mps")).cpu().double()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim = 0).item()
    print("PROBE sdpa_cos", round(cos, 6))
    assert cos > 0.999, cos
try:
    import torchcodec
    print("PROBE torchcodec", torchcodec.__version__)
except Exception as e:  # informational: Python 3.10 homes may have none
    print("PROBE torchcodec_unavailable", type(e).__name__)
PY
}

_seed() {
    # install.sh keeps its uv under the Studio home when UNSLOTH_STUDIO_HOME is set, so look there first.
    _uv=""
    for _c in "$UNSLOTH_STUDIO_HOME/uv/uv" "$(command -v uv || true)" "$HOME/.local/bin/uv"; do
        if [ -n "$_c" ] && [ -x "$_c" ]; then _uv="$_c"; break; fi
    done
    [ -n "$_uv" ] || { echo "no uv found under $UNSLOTH_STUDIO_HOME/uv, PATH or ~/.local/bin" >&2; exit 1; }
    "$_uv" pip install --python "$VPY" "torch==$1" "torchvision==$2" "torchaudio==$3" \
        --default-index https://download.pytorch.org/whl/cpu
    _probe
}

case "${1:-}" in
    probe) _probe ;;
    seed) shift; _seed "$@" ;;
    *) echo "usage: $0 probe | seed TORCH TORCHVISION TORCHAUDIO" >&2; exit 2 ;;
esac
