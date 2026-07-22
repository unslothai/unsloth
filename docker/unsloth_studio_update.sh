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
# Why not `unsloth studio update`: that command re-runs the full installer,
# which re-probes the host GPU to pick torch wheels. In a CPU-only container
# (run without --gpus) it finds no GPU and can downgrade torch to CPU/cu126,
# breaking CUDA. This helper only touches the Studio packages, so it is safe in
# both GPU and CPU containers.
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

# Resolve the Studio venv python. Prefer the venv directly; fall back to
# following the launcher symlink ($STUDIO_HOME/bin/unsloth -> venv/bin/unsloth).
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

# Build the package specs. With --ref, install from git so you can track main
# (or any branch/tag/sha); otherwise take the latest PyPI release.
if [ -n "$REF" ]; then
    SPECS="git+https://github.com/unslothai/unsloth.git@${REF}#egg=unsloth"
    # unsloth-zoo does NOT track unsloth's tags (different cadence). Use --zoo-ref
    # if given; else the unsloth ref only when the zoo repo has it, falling back to
    # main.
    _zoo_ref="$ZOO_REF"
    if [ -z "$_zoo_ref" ]; then
        if git ls-remote --exit-code https://github.com/unslothai/unsloth-zoo.git \
                "$REF" >/dev/null 2>&1; then
            _zoo_ref="$REF"
        else
            _zoo_ref="main"
            echo "[studio-update] unsloth-zoo has no ref '${REF}'; using zoo main"
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

# Sanity: the backend must still import after the swap (a missing --no-deps
# transitive dep shows up here). Non-fatal: just warn with the remedy.
if ! "$PY" -c "import studio.backend.main" >/dev/null 2>&1; then
    echo "[studio-update] WARNING: 'import studio.backend.main' failed after update." >&2
    echo "[studio-update] A new dependency may be missing. Re-run with --with-deps:" >&2
    echo "[studio-update]   unsloth-studio-update --with-deps" >&2
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
