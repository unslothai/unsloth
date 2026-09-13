#!/usr/bin/env bash
# Update Unsloth Studio in place, inside a running container, without pulling a
# new image. Updates ONLY the Studio Python packages (the backend code and the
# pre-built frontend, which ships inside the unsloth wheel) and restarts the
# Studio service. The torch/CUDA stack is left untouched.
#
#   docker exec <container> unsloth-studio-update              # latest PyPI release
#   docker exec <container> unsloth-studio-update --ref main   # latest git main (builds its frontend)
#   docker exec <container> unsloth-studio-update --with-deps  # also update deps
#   docker exec <container> unsloth-studio-update --no-restart # update, restart later
#
# Not `unsloth studio update`: that re-runs the full installer, which re-probes the
# host GPU for torch wheels and in a CPU-only container downgrades torch to CPU/cu126.
#
# Persistence: the update is written to the container's writable layer, so it
# survives `docker restart`. To keep it across a full `docker rm` + `docker run`
# (and to keep your chats/users/models), run Studio with its home on a named
# volume: -v unsloth_studio_home:/opt/unsloth-studio
set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
REF=""
ZOO_REF=""
NO_DEPS="--no-deps"
RESTART=1
PACKAGES="unsloth unsloth_zoo"

usage() { sed -n '2,21p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)         REF="$2"; shift 2;;
        --zoo-ref)     ZOO_REF="$2"; shift 2;;
        --with-deps)   NO_DEPS=""; shift;;
        --no-restart)  RESTART=0; shift;;
        --packages)    PACKAGES="$2"; shift 2;;
        -h|--help)     usage; exit 0;;
        *) echo "unsloth-studio-update: unknown argument: $1" >&2; usage; exit 2;;
    esac
done

PY=""
for cand in \
    "$STUDIO_HOME/unsloth_studio/bin/python" \
    "$STUDIO_HOME/unsloth_studio/bin/python3"; do
    [ -x "$cand" ] && { PY="$cand"; break; }
done
if [ -z "$PY" ] && [ -L "$STUDIO_HOME/bin/unsloth" ]; then
    venv_bin="$(dirname "$(readlink -f "$STUDIO_HOME/bin/unsloth")")"
    [ -x "$venv_bin/python" ] && PY="$venv_bin/python"
fi
[ -n "$PY" ] || { echo "unsloth-studio-update: could not find the Studio venv under $STUDIO_HOME" >&2; exit 1; }

# Where the editable source tree really is: the Studio home may present it as a symlink
# into the image's copy, and replacing that link with a directory would take the tree
# out of the image's hands.
SRC="$(readlink -f "$STUDIO_HOME/src" 2>/dev/null || true)"
[ -n "$SRC" ] || SRC="$STUDIO_HOME/src"
SRC_DIR="$(dirname "$SRC")"

version_of() { "$PY" -c "from importlib.metadata import version; print(version('unsloth'))" 2>/dev/null || echo "unknown"; }

log() { echo "[studio-update] $*"; }

log "Studio venv: $PY"
log "before: unsloth $(version_of)"

# The tree Studio restarts into must import AND serve its UI: `unsloth studio` exits 1
# when studio/frontend/dist is missing, and three quick exits leave it FATAL. From / so
# a `studio` directory in the caller's cwd cannot stand in for the installed package.
studio_tree_ok() {
    (cd / && "$PY" -) <<'PY'
import os, sys
try:
    import studio.backend.main  # noqa: F401
    import studio
except Exception as exc:
    print(f"[studio-update] ERROR: 'import studio.backend.main' failed: {exc}")
    sys.exit(1)
index = os.path.join(os.path.dirname(studio.__file__), "frontend", "dist", "index.html")
if not os.path.isfile(index):
    print(f"[studio-update] ERROR: no built frontend at {index}")
    sys.exit(1)
PY
}

# How each package is installed right now (editable tree, pinned commit or release), so
# a failed update can put it back exactly.
ROLLBACK="$(mktemp)"
"$PY" - "$ROLLBACK" unsloth unsloth_zoo <<'PY'
import json, sys
from importlib.metadata import distribution, PackageNotFoundError
out = []
for name in sys.argv[2:]:
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        continue
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    url = info.get("url", "")
    if info.get("dir_info", {}).get("editable") and url.startswith("file://"):
        out.append("-e " + url[len("file://"):])
    elif "vcs_info" in info:
        out.append(f"{name} @ git+{url}@{info['vcs_info']['commit_id']}")
    elif url:
        out.append(f"{name} @ {url}")
    else:
        out.append(f"{name}=={dist.version}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY

STAGE=""
PREV_SRC=""
SWAPPED=0
restore() {
    log "restoring the previous install"
    if [ "$SWAPPED" = "1" ] && [ -d "$PREV_SRC" ]; then
        rm -rf "$SRC"
        mv "$PREV_SRC" "$SRC"
        SWAPPED=0
    fi
    "$PY" -m pip install --no-deps -r "$ROLLBACK" >/dev/null \
        || log "CRITICAL: pip could not reinstall: $(tr '\n' ' ' < "$ROLLBACK")"
}
cleanup() {
    [ -n "$STAGE" ] && rm -rf "$STAGE"
    rm -f "$ROLLBACK"
}
trap cleanup EXIT

# --ref installs the same shape the image was built with: an editable source tree whose
# frontend is built here with the bundled Node. A plain `pip install git+...` has no
# studio/frontend/dist (only release wheels ship one) and no oxc-validator runtime.
build_source_tree() {
    local tree="$1" node_bin="$STUDIO_HOME/node/bin" oxc
    [ -x "$node_bin/npm" ] || { log "ERROR: no bundled Node at $node_bin; cannot build the frontend for --ref"; return 1; }
    (
        export PATH="$node_bin:$PATH"
        cd "$tree/studio/frontend"
        log "installing frontend dependencies"
        npm ci --no-fund --no-audit --loglevel=error || npm install --no-fund --no-audit --loglevel=error
        log "building the frontend"
        npm run build
        oxc="$tree/studio/backend/core/data_recipe/oxc-validator"
        if [ -f "$oxc/package.json" ]; then
            cd "$oxc"
            log "installing the oxc validator runtime"
            npm install --no-fund --no-audit --loglevel=error
        fi
    ) || return 1
    rm -rf "$tree/studio/frontend/node_modules"
    [ -f "$tree/studio/frontend/dist/index.html" ] || { log "ERROR: the frontend build produced no dist/index.html"; return 1; }
}

if [ -n "$REF" ]; then
    # unsloth-zoo does NOT track unsloth's tags
    _zoo_ref="$ZOO_REF"
    if [ -z "$_zoo_ref" ]; then
        # git documents status 2 for "reached the remote, no matching ref"; any other
        # non-zero means the lookup never happened, and falling through to main pairs
        # the requested unsloth revision with an unrelated zoo one across a private API
        _ls_rc=0
        git ls-remote --exit-code https://github.com/unslothai/unsloth-zoo.git \
            "$REF" >/dev/null 2>&1 || _ls_rc=$?
        if [ "$_ls_rc" = "0" ]; then
            _zoo_ref="$REF"
        elif [ "$_ls_rc" = "2" ]; then
            _zoo_ref="main"
            log "unsloth-zoo has no ref '${REF}'; using zoo main"
        else
            echo "unsloth-studio-update: could not reach unslothai/unsloth-zoo (git ls-remote exit ${_ls_rc}); refusing to guess the zoo ref." >&2
            echo "unsloth-studio-update: retry, or pin it yourself with --zoo-ref <ref>." >&2
            exit 1
        fi
    fi
    log "installing from git: unsloth @${REF}, unsloth-zoo @${_zoo_ref}"
    # a sibling of the real src, so the swap below is a same-filesystem rename
    STAGE="$(mktemp -d "$SRC_DIR/.src-update.XXXXXX")"
    if ! { git -C "$STAGE" init -q \
            && git -C "$STAGE" remote add origin https://github.com/unslothai/unsloth \
            && git -C "$STAGE" fetch -q --depth 1 origin "$REF" \
            && git -C "$STAGE" checkout -q FETCH_HEAD; }; then
        echo "unsloth-studio-update: could not fetch unsloth ref '${REF}'; nothing was changed." >&2
        exit 1
    fi
    rm -rf "$STAGE/.git"
    if ! build_source_tree "$STAGE"; then
        echo "unsloth-studio-update: the frontend for '${REF}' did not build; nothing was changed." >&2
        exit 1
    fi
    PREV_SRC="$SRC_DIR/.src-prev.$$"
    mv "$SRC" "$PREV_SRC"
    if ! mv "$STAGE" "$SRC"; then
        mv "$PREV_SRC" "$SRC"
        echo "unsloth-studio-update: could not move the new source tree into place; nothing was changed." >&2
        exit 1
    fi
    STAGE=""
    SWAPPED=1
    # shellcheck disable=SC2086
    if ! "$PY" -m pip install $NO_DEPS -e "$SRC" \
            "git+https://github.com/unslothai/unsloth-zoo.git@${_zoo_ref}#egg=unsloth_zoo"; then
        restore
        echo "unsloth-studio-update: pip could not install '${REF}'; the previous install is back in place." >&2
        exit 1
    fi
else
    log "installing latest release of: $PACKAGES"
    # shellcheck disable=SC2086
    if ! "$PY" -m pip install -U $NO_DEPS $PACKAGES; then
        restore
        echo "unsloth-studio-update: pip failed; the previous install is back in place." >&2
        exit 1
    fi
fi

log "after:  unsloth $(version_of)"

# Restarting into a tree that cannot start kills a Studio that is serving fine and
# parks supervisord's program in FATAL, so check first and put the old one back.
if ! studio_tree_ok; then
    restore
    if studio_tree_ok; then
        log "the update could not start Studio, so the previous install was restored and Studio was not restarted."
    else
        log "CRITICAL: the previous install could not be restored cleanly; Studio will fail to start until this is fixed."
    fi
    log "If a new dependency is missing, re-run with --with-deps."
    exit 1
fi
if [ "$SWAPPED" = "1" ]; then
    rm -rf "$PREV_SRC"
fi

if [ "$RESTART" = "1" ]; then
    SUPCTL="$(command -v supervisorctl || true)"
    [ -n "$SUPCTL" ] || SUPCTL="/opt/unsloth-venv/bin/supervisorctl"
    # `status` exits 3 for a program that exists but is not running (STOPPED, EXITED,
    # FATAL), which is exactly the Studio a failed earlier update left behind.
    _st=0
    [ -x "$SUPCTL" ] && { "$SUPCTL" status studio >/dev/null 2>&1 || _st=$?; } || _st=4
    if [ "$_st" = "0" ] || [ "$_st" = "3" ]; then
        log "restarting the studio service"
        "$SUPCTL" restart studio || true
        _wait="${UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT:-180}"
        _deadline=$(( $(date +%s) + _wait ))
        _up=0
        while [ "$(date +%s)" -lt "$_deadline" ]; do
            curl -sf -o /dev/null --max-time 3 http://127.0.0.1:8000/api/health && { _up=1; break; }
            sleep 2
        done
        if [ "$_up" = "1" ]; then
            log "Studio is answering on port 8000"
        elif [ "$_wait" -gt 0 ]; then
            log "WARNING: Studio did not answer on port 8000 within ${_wait}s; see docker logs"
        fi
    else
        log "supervisor not managing 'studio' here; restart Studio yourself"
        log "  (e.g. 'docker restart <container>')"
    fi
else
    log "--no-restart: restart Studio to load the update"
    log "  docker exec <container> supervisorctl restart studio"
fi

log "done"
