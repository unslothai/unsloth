#!/usr/bin/env bash
# Update Unsloth Studio in place, inside a running container, without pulling a
# new image. Updates ONLY the Studio Python packages (the backend code and the
# pre-built frontend, which ships inside the unsloth wheel) and restarts the
# Studio service. The torch/CUDA stack is left untouched.
#
#   docker exec <container> unsloth-studio-update              # latest PyPI release
#   docker exec <container> unsloth-studio-update --ref main   # latest git main
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

version_of() { "$PY" -c "from importlib.metadata import version; print(version('unsloth'))" 2>/dev/null || echo "unknown"; }

echo "[studio-update] Studio venv: $PY"
echo "[studio-update] before: unsloth $(version_of)"

if [ -n "$REF" ]; then
    SPECS="git+https://github.com/unslothai/unsloth.git@${REF}#egg=unsloth"
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
            echo "[studio-update] unsloth-zoo has no ref '${REF}'; using zoo main"
        else
            echo "unsloth-studio-update: could not reach unslothai/unsloth-zoo (git ls-remote exit ${_ls_rc}); refusing to guess the zoo ref." >&2
            echo "unsloth-studio-update: retry, or pin it yourself with --zoo-ref <ref>." >&2
            exit 1
        fi
    fi
    SPECS="$SPECS git+https://github.com/unslothai/unsloth-zoo.git@${_zoo_ref}#egg=unsloth_zoo"
    echo "[studio-update] installing from git: unsloth @${REF}, unsloth-zoo @${_zoo_ref}"
else
    SPECS="$PACKAGES"
    echo "[studio-update] installing latest release of: $PACKAGES"
fi

# shellcheck disable=SC2086
"$PY" -m pip install -U $NO_DEPS $SPECS

echo "[studio-update] after:  unsloth $(version_of)"

# Restarting into code that cannot import kills a process that is serving fine and
# leaves supervisord's studio program in FATAL, which it never leaves on its own.
if ! "$PY" -c "import studio.backend.main" >/dev/null 2>&1; then
    echo "[studio-update] ERROR: 'import studio.backend.main' failed after update." >&2
    echo "[studio-update] A new dependency may be missing. Re-run with --with-deps:" >&2
    echo "[studio-update]   unsloth-studio-update --with-deps" >&2
    echo "[studio-update] NOT restarting Studio, so the process already running keeps" >&2
    echo "[studio-update] serving whatever it has already imported. That is the only" >&2
    echo "[studio-update] thing still working: the venv on disk is ALREADY replaced." >&2
    echo "[studio-update] Anything it imports lazily from here on fails the same way," >&2
    echo "[studio-update] any restart parks Studio in FATAL, and because the venv lives" >&2
    echo "[studio-update] under \$UNSLOTH_STUDIO_HOME, a persisted home keeps it broken" >&2
    echo "[studio-update] across docker rm + docker run. Fix it before restarting." >&2
    echo "[studio-update] Once fixed:  supervisorctl restart studio" >&2
    exit 1
fi

if [ "$RESTART" = "1" ]; then
    SUPCTL="$(command -v supervisorctl || true)"
    [ -n "$SUPCTL" ] || SUPCTL="/opt/unsloth-venv/bin/supervisorctl"
    if [ -x "$SUPCTL" ] && "$SUPCTL" status studio >/dev/null 2>&1; then
        echo "[studio-update] restarting the studio service"
        "$SUPCTL" restart studio
    else
        echo "[studio-update] supervisor not managing 'studio' here; restart Studio yourself"
        echo "[studio-update]   (e.g. 'docker restart <container>')"
    fi
else
    echo "[studio-update] --no-restart: restart Studio to load the update"
    echo "[studio-update]   docker exec <container> supervisorctl restart studio"
fi

echo "[studio-update] done"
