#!/usr/bin/env bash
# Update the baked llama.cpp prebuilt in place, inside a running container,
# without pulling a new image. Downloads the newest portable llama.cpp bundle
# (the same target-pinned, sha256-verified bundle the image is built with) and
# atomically swaps it into $UNSLOTH_LLAMA_CPP_PATH, so the next GGUF export /
# model load uses it.
#
#   docker exec <container> unsloth-llama-update            # latest release
#   docker exec <container> unsloth-llama-update --tag b9773-mix-1f1aaa4
#   docker exec <container> unsloth-llama-update --check    # report only, no download
#
# This reuses the build-time fetcher, which resolves the latest release via the
# GitHub /releases/latest redirect (no API token, not rate-limited) and installs
# the portable CUDA bundle that runs on CPU and every supported GPU. That makes
# it work the same in a CPU-only or a --gpus container, unlike the host-probing
# installer behind the in-app banner.
#
# Persistence: the swap lands in the container's writable layer (survives
# docker restart). To keep it across a full recreate, mount the prebuilt dir on
# a named volume: -v unsloth_llama:/opt/unsloth/llama.cpp
set -euo pipefail

INSTALL_DIR="${UNSLOTH_LLAMA_CPP_PATH:-/opt/unsloth/llama.cpp}"
STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
FETCHER="${UNSLOTH_LLAMA_FETCHER:-/usr/local/lib/unsloth/fetch_llama_prebuilt.py}"
REPO="unslothai/llama.cpp"
TAG="latest"
CHECK_ONLY=0

usage() { sed -n '2,24p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)          TAG="$2"; shift 2;;
        --install-dir)  INSTALL_DIR="$2"; shift 2;;
        --check)        CHECK_ONLY=1; shift;;
        -h|--help)      usage; exit 0;;
        *) echo "unsloth-llama-update: unknown argument: $1" >&2; usage; exit 2;;
    esac
done

[ -f "$FETCHER" ] || { echo "unsloth-llama-update: fetcher not found at $FETCHER" >&2; exit 1; }

# Any python works (the fetcher is stdlib-only); prefer the Studio venv, then base.
PY=""
for cand in \
    "$STUDIO_HOME/unsloth_studio/bin/python" \
    /opt/unsloth-venv/bin/python \
    python3 python; do
    command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
    [ -x "$cand" ] && { PY="$cand"; break; }
done
[ -n "$PY" ] || { echo "unsloth-llama-update: no python found" >&2; exit 1; }

# amd64 -> linux-x64-cuda12 portable; arm64 -> linux-arm64-cuda13 portable.
case "$(uname -m)" in
    x86_64|amd64) ARCH="amd64";;
    aarch64|arm64) ARCH="arm64";;
    *) echo "unsloth-llama-update: unsupported arch $(uname -m)" >&2; exit 1;;
esac

installed_tag() {
    "$PY" - "$INSTALL_DIR" <<'PY' 2>/dev/null || echo "unknown"
import json, os, sys
p = os.path.join(sys.argv[1], "UNSLOTH_PREBUILT_INFO.json")
try:
    d = json.load(open(p)); print(d.get("tag") or d.get("release_tag") or d.get("upstream_tag") or "unknown")
except Exception:
    print("unknown")
PY
}

resolve_latest() {
    "$PY" - "$FETCHER" "$REPO" <<'PY' 2>/dev/null || echo ""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("flp", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.resolve_latest_tag(sys.argv[2]))
PY
}

CUR="$(installed_tag)"
echo "[llama-update] install dir: $INSTALL_DIR"
echo "[llama-update] installed:   $CUR"

if [ "$CHECK_ONLY" = "1" ]; then
    LATEST="$(resolve_latest)"
    echo "[llama-update] latest:      ${LATEST:-unknown}"
    if [ -n "$LATEST" ] && [ "$LATEST" != "$CUR" ]; then
        echo "[llama-update] an update is available (run without --check to apply)"
    else
        echo "[llama-update] up to date"
    fi
    exit 0
fi

# Fetch into a sibling temp dir (same filesystem as INSTALL_DIR, so the swap is
# an atomic rename), then swap. On any failure the existing install is untouched.
parent="$(dirname "$INSTALL_DIR")"
work="$(mktemp -d "$parent/.llamaupd.XXXXXX")"
trap 'rm -rf "$work" "${INSTALL_DIR}.old.$$" 2>/dev/null || true' EXIT
new="$work/llama.cpp"

echo "[llama-update] fetching llama.cpp '$TAG' ($ARCH portable) ..."
"$PY" "$FETCHER" "$TAG" "$ARCH" "$new"

# Preserve the Studio ownership marker so setup.sh keeps recognising the dir.
[ -e "$INSTALL_DIR/.unsloth-studio-owned" ] && touch "$new/.unsloth-studio-owned"

echo "[llama-update] swapping into place ..."
mv "$INSTALL_DIR" "${INSTALL_DIR}.old.$$"
if mv "$new" "$INSTALL_DIR"; then
    rm -rf "${INSTALL_DIR}.old.$$"
else
    echo "[llama-update] swap failed; restoring previous install" >&2
    mv "${INSTALL_DIR}.old.$$" "$INSTALL_DIR"
    exit 1
fi

echo "[llama-update] installed now: $(installed_tag)"
echo "[llama-update] done (reload your model / re-run export to use it)"
