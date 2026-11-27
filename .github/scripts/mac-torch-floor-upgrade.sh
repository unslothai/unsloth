#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Helpers for studio-mac-torch-floor-upgrade.yml, run against $UNSLOTH_STUDIO_HOME's venv.
#   probe          print "PROBE <python> <torch> <torchvision> <torchaudio> mps <bool>", check MPS SDPA
#   seed T TV TA   put the existing home on that torch trio (an older home, as a user would have)
#   check-route T INSTALL_LOG PROBE_LOG
#                  the re-run of a home on torch T either kept it (probe still T) or moved it to a
#                  ">=LO,<HI" route (probe inside it); anything else, or no route line at all, fails
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

_version_lt() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$1" ]
}

_check_route() {
    _from="$1"
    _from_re="$(printf '%s' "$_from" | sed 's/\./\\./g')"
    _got="$(sed -nE 's/^PROBE [0-9.]+ ([0-9]+\.[0-9]+\.[0-9]+)[^ ]* .*/\1/p' "$3" | head -n 1)"
    [ -n "$_got" ] || { echo "no PROBE version line in $3" >&2; exit 1; }
    if grep -Eq "existing install has torch $_from_re -- keeping it" "$2"; then
        [ "$_got" = "$_from" ] || { echo "install log keeps torch $_from but the home has $_got" >&2; exit 1; }
        echo "route: kept $_from"
        return
    fi
    _route="$(grep -oE "existing install has torch $_from_re -- moving to torch>=[0-9.]+,<[0-9.]+" "$2" | head -n 1 || true)"
    [ -n "$_route" ] || { echo "install log neither keeps nor moves torch $_from" >&2; exit 1; }
    _lo="$(printf '%s' "$_route" | sed -E 's/.*>=([0-9.]+),<.*/\1/')"
    _hi="$(printf '%s' "$_route" | sed -E 's/.*,<([0-9.]+)$/\1/')"
    if _version_lt "$_got" "$_lo" || ! _version_lt "$_got" "$_hi"; then
        echo "install log moves torch $_from to >=$_lo,<$_hi but the home has $_got" >&2
        exit 1
    fi
    echo "route: moved $_from to $_got (>=$_lo,<$_hi)"
}

case "${1:-}" in
    probe) _probe ;;
    seed) shift; _seed "$@" ;;
    check-route) shift; _check_route "$@" ;;
    *) echo "usage: $0 probe | seed TORCH TORCHVISION TORCHAUDIO | check-route TORCH INSTALL_LOG PROBE_LOG" >&2; exit 2 ;;
esac
