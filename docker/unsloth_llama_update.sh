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
# Persistence: unmounted, the swap lands in the container's writable layer
# (survives docker restart). To keep it across a full recreate, mount the dir
# on a named volume (-v unsloth_llama:/opt/unsloth/llama.cpp); the updater
# detects the mount and swaps the bundle contents inside the volume.
set -euo pipefail

INSTALL_DIR="${UNSLOTH_LLAMA_CPP_PATH:-/opt/unsloth/llama.cpp}"
STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
FETCHER="${UNSLOTH_LLAMA_FETCHER:-/usr/local/lib/unsloth/fetch_llama_prebuilt.py}"
REPO="unslothai/llama.cpp"
TAG="latest"
CHECK_ONLY=0

usage() { sed -n '2,21p' "$0"; }

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

# the fetcher is stdlib-only, so any python works
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

# The FULL "release_tag" is what --check compares against resolve_latest_tag; the
# marker's "tag" is the normalized base build, so preferring it reports "an update is
# available" forever on any install the in-app updater wrote.
installed_tag() {
    "$PY" - "$INSTALL_DIR" <<'PY' 2>/dev/null || echo "unknown"
import json, os, sys
p = os.path.join(sys.argv[1], "UNSLOTH_PREBUILT_INFO.json")
try:
    d = json.load(open(p)); print(d.get("release_tag") or d.get("tag") or d.get("upstream_tag") or "unknown")
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
    # resolve_latest swallows every failure into "", so an empty value means the
    # lookup never happened; "up to date" would report a state it could not observe
    if [ -z "$LATEST" ]; then
        echo "[llama-update] could not reach the release feed; update status UNKNOWN" >&2
        echo "[llama-update] (retry once the container has network access)" >&2
        exit 1
    fi
    if [ "$LATEST" != "$CUR" ]; then
        echo "[llama-update] an update is available (run without --check to apply)"
    else
        echo "[llama-update] up to date"
    fi
    exit 0
fi

# a sibling temp dir is on the same filesystem, so the swap is an atomic rename
parent="$(dirname "$INSTALL_DIR")"

# A named volume mounted AT the install dir cannot be renamed (EBUSY), so swap its
# CONTENTS instead. UNSLOTH_LLAMA_UPDATE_IN_PLACE=1/0 overrides autodetection.
IN_PLACE="${UNSLOTH_LLAMA_UPDATE_IN_PLACE:-}"
if [ -z "$IN_PLACE" ]; then
    IN_PLACE=0
    if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$INSTALL_DIR" 2>/dev/null; then
        IN_PLACE=1
    elif [ "$(stat -c %d "$INSTALL_DIR" 2>/dev/null)" != "$(stat -c %d "$parent" 2>/dev/null)" ]; then
        IN_PLACE=1   # filesystem boundary = a volume, on hosts without mountpoint(1)
    fi
fi
if [ "$IN_PLACE" = "1" ]; then
    # work + backup live UNDER the install dir, so every swap step is a same-fs rename
    work="$(mktemp -d "$INSTALL_DIR/.llamaupd.XXXXXX")"
    backup="$INSTALL_DIR/.old.$$"
else
    work="$(mktemp -d "$parent/.llamaupd.XXXXXX")"
    backup="${INSTALL_DIR}.old.$$"
fi
swap_done=0
drained=0
# The exit handler must never delete $backup while it is the ONLY copy: restore
# first, remove only once the new tree is active.
cleanup() {
    if [ "$swap_done" -ne 1 ]; then
        if [ "$IN_PLACE" = "1" ]; then
            # every old entry lives in exactly one of $backup / $INSTALL_DIR, so a
            # same-named entry in the install dir is a half-moved NEW one
            if [ -d "$backup" ]; then
                _restore_fail=0
                # The loop below only sees names the OLD tree had, so a file the
                # new release introduced would survive into a mixed-version dir and
                # ggml dlopens every libggml-*.so next to the binaries. Gated on
                # "drained": before that, an entry here can be the only old copy.
                if [ "$drained" = "1" ]; then
                    find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 \
                         ! -path "$work" ! -path "$backup" \
                         -exec rm -rf {} + 2>/dev/null || true
                fi
                for _e in "$backup"/* "$backup"/.[!.]* "$backup"/..?*; do
                    { [ -e "$_e" ] || [ -L "$_e" ]; } || continue
                    _b="$(basename "$_e")"
                    if [ -e "$INSTALL_DIR/$_b" ] || [ -L "$INSTALL_DIR/$_b" ]; then
                        rm -rf "${INSTALL_DIR:?}/$_b" 2>/dev/null || true
                    fi
                    mv "$_e" "$INSTALL_DIR/" 2>/dev/null || _restore_fail=1
                done
                if [ "$_restore_fail" -eq 0 ]; then
                    rmdir "$backup" 2>/dev/null || true
                else
                    echo "[llama-update] CRITICAL: restore failed; previous install preserved at $backup" >&2
                fi
            fi
        elif [ ! -e "$INSTALL_DIR" ] && [ -e "$backup" ]; then
            if ! mv "$backup" "$INSTALL_DIR" 2>/dev/null; then
                echo "[llama-update] CRITICAL: restore failed; previous install preserved at $backup" >&2
            fi
        fi
    fi
    rm -rf "$work" 2>/dev/null || true
    if [ "$swap_done" = "1" ]; then
        rm -rf "$backup" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
new="$work/llama.cpp"

echo "[llama-update] fetching llama.cpp '$TAG' ($ARCH portable) ..."
"$PY" "$FETCHER" "$TAG" "$ARCH" "$new"

# preserve the Studio ownership marker so setup.sh keeps recognising the dir
[ -e "$INSTALL_DIR/.unsloth-studio-owned" ] && touch "$new/.unsloth-studio-owned"

echo "[llama-update] swapping into place ..."
if [ "$IN_PLACE" = "1" ]; then
    mkdir "$backup"
    find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 \
         ! -path "$work" ! -path "$backup" -exec mv -t "$backup" {} +
    # every old entry is now in $backup, so from here the trap may clear the install
    # dir before restoring; set -e means a failed drain never reaches this
    drained=1
    if find "$new" -mindepth 1 -maxdepth 1 -exec mv -t "$INSTALL_DIR" {} +; then
        swap_done=1
    else
        echo "[llama-update] swap failed; restoring previous install" >&2
        exit 1
    fi
else
    mv "$INSTALL_DIR" "$backup"
    if mv "$new" "$INSTALL_DIR"; then
        swap_done=1
    else
        echo "[llama-update] swap failed; restoring previous install" >&2
        mv "$backup" "$INSTALL_DIR"
        exit 1
    fi
fi

echo "[llama-update] installed now: $(installed_tag)"
echo "[llama-update] done (reload your model / re-run export to use it)"
