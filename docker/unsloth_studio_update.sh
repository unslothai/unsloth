#!/usr/bin/env bash
# Update Unsloth Studio in place, inside a running container, without pulling a
# new image. Updates ONLY the Studio Python packages (the backend code and the
# pre-built frontend, which ships inside the unsloth wheel) and restarts the
# Studio service. The torch/CUDA stack is left untouched.
#
#   docker exec <container> unsloth-studio-update              # latest PyPI release
#   docker exec <container> unsloth-studio-update --ref main   # latest git main (builds its frontend)
#   docker exec <container> unsloth-studio-update --with-deps  # also update deps (torch/CUDA stay pinned)
#   docker exec <container> unsloth-studio-update --no-restart # update, restart later
#   --zoo-ref <ref>      unsloth-zoo ref to pair with --ref (default: the same ref, else main)
#   --packages "<specs>" what a release update installs (default: unsloth unsloth_zoo)
#   UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT=<seconds>  how long to wait for /api/health after the
#                        restart (default 180). The previous install is kept until Studio
#                        answers; if it does not, or the restart fails, it goes back and the
#                        helper exits 1. 0 skips the wait: the update is committed as soon as
#                        the restart command succeeds, with no proof that it can serve.
#   UNSLOTH_NPM_REGISTRY=<url>  npm registry for the --ref frontend build, as in the installer
#
# One update at a time: a second invocation while one runs exits 1 without touching
# anything.
#
# Not `unsloth studio update`: that re-runs the full installer, which re-probes the
# host GPU for torch wheels and in a CPU-only container downgrades torch to CPU/cu126.
#
# Persistence: the updated packages live in the image's copy of Studio, so they
# survive `docker restart` but not `docker rm`; pull a new image for a lasting update.
# Studio's data (accounts, chats, models) is separate: keep it with
# -v unsloth-studio:/opt/unsloth-studio, which never pins Studio's code.
set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
REF=""
ZOO_REF=""
NO_DEPS="--no-deps"
RESTART=1
PACKAGES="unsloth unsloth_zoo"

# the header comment, up to the first line that is not one
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { print }' "$0"; }

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
# out of the image's hands. pip is still pointed at the home path (SRC_INSTALL), so the
# recorded install keeps resolving through the link, whichever image serves it.
SRC="$(readlink -f "$STUDIO_HOME/src" 2>/dev/null || true)"
[ -n "$SRC" ] || SRC="$STUDIO_HOME/src"
SRC_DIR="$(dirname "$SRC")"
SRC_INSTALL="$STUDIO_HOME/src"
[ -e "$SRC_INSTALL" ] || SRC_INSTALL="$SRC"

# From /: `python -` and `python -c` put the caller's cwd first on sys.path (python -m pip
# drops it itself), so a checkout there with its own unsloth.egg-info would answer for
# the venv's installed distribution.
version_of() { (cd / && "$PY" -c "from importlib.metadata import version; print(version('unsloth'))") 2>/dev/null || echo "unknown"; }

log() { echo "[studio-update] $*"; }

# The package record and dependency snapshot of the install being replaced, kept beside
# the source tree from the moment pip starts until the update is committed, so a run
# that is killed outright leaves the next one enough to finish the restore.
KEEP_ROLLBACK="$SRC_DIR/.src-update.rollback"
KEEP_FREEZE="$SRC_DIR/.src-update.freeze"

# Puts recorded packages back: the dependency snapshot as pinned, the packages by force
# (pip takes a same-version editable tree as already satisfying `unsloth==<version>`
# and would leave the new tree's metadata in place), and what the update added taken
# out. Returns 1 when pip could not do all of it.
reinstall_recorded() {
    local rollback="$1" freeze="$2" ok=0 _absent
    if [ -n "$freeze" ] && [ -s "$freeze" ]; then
        "$PY" -m pip install --no-deps -r "$freeze" >/dev/null \
            || { log "CRITICAL: pip could not put the previous dependency set back; re-run with --with-deps once the cause is fixed"; ok=1; }
    fi
    if [ -n "$rollback" ] && [ -s "$rollback" ]; then
        "$PY" -m pip install --no-deps --force-reinstall -r "$rollback" >/dev/null \
            || { log "CRITICAL: pip could not reinstall: $(tr '\n' ' ' < "$rollback")"; ok=1; }
        _absent="$(sed -n 's/^# absent: //p' "$rollback" | tr '\n' ' ')"
        if [ -n "${_absent// /}" ]; then
            # shellcheck disable=SC2086
            "$PY" -m pip uninstall -y $_absent >/dev/null \
                || { log "CRITICAL: pip could not remove what the update added: $_absent"; ok=1; }
        fi
    fi
    return "$ok"
}

log "Studio venv: $PY"
log "before: unsloth $(version_of)"

# Two updaters at once would each take the other's staging and previous trees for
# leftovers (below) and swap over each other's src. The lock is held on an open fd for
# the whole run, so the kernel drops it however the run ends, SIGKILL included; the name
# matches the linker's scratch pattern so it is never linked into the home.
LOCK="$SRC_DIR/.src-update.lock"
command -v flock >/dev/null 2>&1 || { echo "unsloth-studio-update: flock (util-linux) is missing; refusing to run unlocked." >&2; exit 1; }
exec 9>>"$LOCK" || { echo "unsloth-studio-update: cannot open $LOCK" >&2; exit 1; }
if ! flock -n 9; then
    echo "unsloth-studio-update: another unsloth-studio-update is running (holding $LOCK); wait for it to finish, then retry. Nothing was changed." >&2
    exit 1
fi

# A run that was killed outright (docker stop ends in SIGKILL, so no trap ran) can leave
# its previous tree beside src as .src-prev.*, or its staging tree as .src-update.*.
# Nothing else writes those names here, and the lock above makes this the only updater,
# so at start they are always leftovers. A previous tree is a swap that was never
# committed (commit_update renames it away before deleting it): the tree in src was
# never proven to serve, so the previous one goes back over it; with no src at all the
# kill landed between the two moves. Everything else is cleared.
shopt -s nullglob
_prev=("$SRC_DIR"/.src-prev.*)
if [ "${#_prev[@]}" = "1" ] && [ -d "${_prev[0]}" ]; then
    if [ -e "$SRC" ]; then
        log "an interrupted update left its previous tree at ${_prev[0]}; putting it back over the unverified one"
        _drop="$(mktemp -d "$SRC_DIR/.src-update.XXXXXX")" && rmdir "$_drop"
        mv -T "$SRC" "$_drop"
    else
        log "recovering the source tree an interrupted update left at ${_prev[0]}"
    fi
    mv -T "${_prev[0]}" "$SRC"
fi
# the same kill after pip had started: the packages it replaced go back before anything
# else is recorded as the previous install
if [ -s "$KEEP_ROLLBACK" ]; then
    log "an interrupted update left its package record at $KEEP_ROLLBACK; putting the previous packages back first"
    if reinstall_recorded "$KEEP_ROLLBACK" "$KEEP_FREEZE"; then
        rm -f "$KEEP_ROLLBACK" "$KEEP_FREEZE"
    else
        echo "unsloth-studio-update: could not put the previous packages back (see the CRITICAL lines above); fix the cause and run this again. Nothing else was changed." >&2
        exit 1
    fi
fi
if [ -d "$SRC" ]; then
    for _stale in "$SRC_DIR"/.src-update.* "$SRC_DIR"/.src-prev.*; do
        [ "$_stale" = "$LOCK" ] && continue
        log "removing $_stale, left behind by an earlier update"
        rm -rf "$_stale"
    done
fi
shopt -u nullglob

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

STAGE=""
PREV_SRC=""
SWAPPED=0
INSTALLING=0
DONE=0
ROLLBACK=""
FREEZE=""
CONSTRAINTS=""
RESTORE_FAILED=0
# Returns 1 when the previous install is not fully back; the kept record then stays
# for the next run to finish from.
restore() {
    log "restoring the previous install"
    # a restore runs to completion: a signal now would leave a half-restored install,
    # and the exit trap must not run it a second time
    trap '' INT TERM
    INSTALLING=0
    if [ "$SWAPPED" = "1" ] && [ -d "$PREV_SRC" ]; then
        rm -rf "$SRC"
        if mv -T "$PREV_SRC" "$SRC"; then
            SWAPPED=0
        else
            log "CRITICAL: could not put $PREV_SRC back at $SRC; move it there by hand"
            RESTORE_FAILED=1
        fi
    fi
    reinstall_recorded "$ROLLBACK" "$FREEZE" || RESTORE_FAILED=1
    [ "$RESTORE_FAILED" = "1" ] && return 1
    rm -f "$KEEP_ROLLBACK" "$KEEP_FREEZE"
    return 0
}
# A failed install ends here: exit 1, saying how far the restore got.
fail_after_restore() {
    if restore; then
        echo "unsloth-studio-update: $1; the previous install is back in place." >&2
    else
        echo "unsloth-studio-update: $1; the previous source tree is back but pip could not put every previous package back (see the CRITICAL lines above). Fix the cause and run this again: the next run finishes the restore first." >&2
    fi
    exit 1
}
# Kept from the moment pip can change the venv until commit_update or a finished restore.
keep_record() {
    cp -- "$ROLLBACK" "$KEEP_ROLLBACK"
    if [ -n "$FREEZE" ] && [ -s "$FREEZE" ]; then cp -- "$FREEZE" "$KEEP_FREEZE"; else rm -f "$KEEP_FREEZE"; fi
}
# Runs on every exit. An interrupt (Ctrl-C, or a TERM) after the swap started would
# otherwise leave the half-installed tree in place and the previous one beside it.
cleanup() {
    # a second signal while the cleanup itself restores must not cut it short
    trap '' INT TERM
    # a release-path pip that was interrupted can have replaced the packages half-way:
    # the recorded previous pins go back the same as after a swap
    if [ "$DONE" != "1" ] && { [ "$SWAPPED" = "1" ] || [ "$INSTALLING" = "1" ]; }; then
        log "interrupted after the install started; putting the previous install back"
        restore || log "the previous install is not fully back; the next run finishes the restore first"
    fi
    [ -n "$STAGE" ] && rm -rf "$STAGE"
    [ -n "$ROLLBACK" ] && rm -f "$ROLLBACK"
    [ -n "$FREEZE" ] && rm -f "$FREEZE"
    [ -n "$CONSTRAINTS" ] && rm -f "$CONSTRAINTS"
    # the lock is still held on fd 9 until exit; a waiter that opened this inode blocks
    # on it until then, and one that starts later creates a fresh file
    rm -f "$LOCK"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# How each package is installed right now (editable tree, pinned commit or release), so
# a failed update can put it back exactly. Every --packages target is recorded, and one
# that is not installed yet is noted so a restore can take it out again.
ROLLBACK="$(mktemp)"
_names="unsloth unsloth_zoo"
for _p in $PACKAGES; do _names="$_names ${_p%%[<>=!~\[@ ]*}"; done
# shellcheck disable=SC2086
(cd / && "$PY" - "$ROLLBACK" $_names) <<'PY'
import json, sys
from importlib.metadata import distribution, PackageNotFoundError
out = []
for name in dict.fromkeys(sys.argv[2:]):
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        out.append("# absent: " + name)
        continue
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    url = info.get("url", "")
    if info.get("dir_info", {}).get("editable") and url.startswith("file://"):
        # the URI as recorded: pip accepts `-e file:///path`, and a path with spaces
        # comes back percent-encoded, which as a bare path is not a valid editable
        out.append("-e " + url)
    elif "vcs_info" in info:
        out.append(f"{name} @ git+{url}@{info['vcs_info']['commit_id']}")
    elif url:
        out.append(f"{name} @ {url}")
    else:
        out.append(f"{name}=={dist.version}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY

# --with-deps lets pip move every dependency, and putting unsloth back alone would leave
# that new dependency set under the old code. Snapshot it first (editable trees are
# handled by ROLLBACK) so restore can pin it back. The torch/CUDA stack is pinned as
# constraints for the install itself: the image links the venv's nvidia libraries into
# the base venv, so a re-resolved torch would write over the base image's copy.
if [ -z "$NO_DEPS" ]; then
    FREEZE="$(mktemp)"
    if ! "$PY" -m pip freeze --exclude-editable > "$FREEZE" 2>/dev/null; then
        echo "unsloth-studio-update: pip freeze failed, so a failed --with-deps update could not be rolled back; nothing was changed." >&2
        exit 1
    fi
    CONSTRAINTS="$(mktemp)"
    grep -E '^(torch|torchvision|torchaudio|triton|nvidia-|xformers|bitsandbytes)' "$FREEZE" > "$CONSTRAINTS" || true
fi
DEP_ARGS=()
if [ -z "$NO_DEPS" ] && [ -s "$CONSTRAINTS" ]; then
    DEP_ARGS=(-c "$CONSTRAINTS")
fi

# --ref installs the same shape the image was built with: an editable source tree whose
# frontend is built here with the bundled Node. A plain `pip install git+...` has no
# studio/frontend/dist (only release wheels ship one) and no oxc-validator runtime.
NPM_REGISTRY_ARGS=()
[ -n "${UNSLOTH_NPM_REGISTRY:-}" ] && NPM_REGISTRY_ARGS=(--registry "$UNSLOTH_NPM_REGISTRY")
build_source_tree() {
    local tree="$1" npm_bin="" oxc
    if [ -x "$STUDIO_HOME/node/bin/npm" ]; then
        npm_bin="$STUDIO_HOME/node/bin/npm"
    else
        # the installer uses the system Node when it is new enough and installs none of its own
        npm_bin="$(command -v npm || true)"
    fi
    [ -n "$npm_bin" ] || { log "ERROR: no npm (neither $STUDIO_HOME/node/bin/npm nor one on PATH); cannot build the frontend for --ref"; return 1; }
    # errexit is ignored inside a `( ... ) || return` list, so each step fails explicitly
    (
        export PATH="$(dirname "$npm_bin"):$PATH"
        cd "$tree/studio/frontend" || exit 1
        log "installing frontend dependencies"
        if [ -f package-lock.json ]; then
            # the lockfile is the build's contract (.npmrc: `npm ci`, never `npm install`)
            npm ci --no-fund --no-audit --loglevel=error ${NPM_REGISTRY_ARGS[@]+"${NPM_REGISTRY_ARGS[@]}"} \
                || { log "ERROR: npm ci failed (lockfile drift or the registry; UNSLOTH_NPM_REGISTRY=<url> for a mirror)"; exit 1; }
        else
            log "no package-lock.json in this ref; resolving with npm install"
            npm install --no-fund --no-audit --loglevel=error ${NPM_REGISTRY_ARGS[@]+"${NPM_REGISTRY_ARGS[@]}"} || exit 1
        fi
        log "building the frontend"
        npm run build || { log "ERROR: npm run build failed"; exit 1; }
        oxc="$tree/studio/backend/core/data_recipe/oxc-validator"
        if [ -f "$oxc/package.json" ]; then
            cd "$oxc" || exit 1
            log "installing the oxc validator runtime"
            # the same lockfile rule as the frontend
            if [ -f package-lock.json ]; then
                npm ci --no-fund --no-audit --loglevel=error ${NPM_REGISTRY_ARGS[@]+"${NPM_REGISTRY_ARGS[@]}"} \
                    || { log "ERROR: npm ci failed for the oxc validator (lockfile drift or the registry)"; exit 1; }
            else
                npm install --no-fund --no-audit --loglevel=error ${NPM_REGISTRY_ARGS[@]+"${NPM_REGISTRY_ARGS[@]}"} || exit 1
            fi
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
    if [ ! -d "$SRC" ]; then
        echo "unsloth-studio-update: no source tree at $SRC (a wheel install has none, and a broken src link points at nothing); --ref needs the image's Studio checkout. Use a plain update instead; nothing was changed." >&2
        exit 1
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
    # a fresh name (mktemp) and -T: an existing directory of the same name would make mv
    # nest the tree inside it, and restore would then move that wrapper over src
    PREV_SRC="$(mktemp -d "$SRC_DIR/.src-prev.XXXXXX")" && rmdir "$PREV_SRC"
    # SWAPPED before the first move: a signal between the two moves must still restore
    keep_record
    SWAPPED=1
    mv -T "$SRC" "$PREV_SRC"
    if ! mv -T "$STAGE" "$SRC"; then
        mv -T "$PREV_SRC" "$SRC" && SWAPPED=0
        echo "unsloth-studio-update: could not move the new source tree into place; nothing was changed." >&2
        exit 1
    fi
    STAGE=""
    _spec="$SRC_INSTALL"
    # with dependencies: the backend's own requirement set is the `studio` extra
    [ -z "$NO_DEPS" ] && _spec="${SRC_INSTALL}[studio]"
    INSTALLING=1
    # shellcheck disable=SC2086
    if ! "$PY" -m pip install $NO_DEPS ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} -e "$_spec" \
            "git+https://github.com/unslothai/unsloth-zoo.git@${_zoo_ref}#egg=unsloth_zoo"; then
        fail_after_restore "pip could not install '${REF}'"
    fi
else
    _pkgs="$PACKAGES"
    if [ -z "$NO_DEPS" ]; then
        # a release update with dependencies must bring the backend's requirement set too
        _pkgs="$(printf '%s\n' $PACKAGES | sed 's/^unsloth$/unsloth[studio]/' | tr '\n' ' ')"
    fi
    log "installing latest release of: $_pkgs"
    keep_record
    INSTALLING=1
    # shellcheck disable=SC2086
    if ! "$PY" -m pip install -U $NO_DEPS ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} $_pkgs; then
        fail_after_restore "pip failed"
    fi
fi

log "after:  unsloth $(version_of)"

# Restarting into a tree that cannot start kills a Studio that is serving fine and
# parks supervisord's program in FATAL, so check first and put the old one back.
if ! studio_tree_ok; then
    if restore && studio_tree_ok; then
        log "the update could not start Studio, so the previous install was restored and Studio was not restarted."
    else
        log "CRITICAL: the previous install could not be restored cleanly; Studio will fail to start until this is fixed."
    fi
    log "If a new dependency is missing, re-run with --with-deps."
    exit 1
fi
# The previous tree stays beside src until the restarted service proves it can serve:
# an import that passes says nothing about a backend that dies at startup. Committing
# sets DONE before the previous tree goes, so a signal after that cannot "restore" the
# old package pins on top of the new tree.
commit_update() {
    DONE=1
    rm -f "$KEEP_ROLLBACK" "$KEEP_FREEZE"
    if [ "$SWAPPED" = "1" ]; then
        # renamed before it is deleted: a kill during the delete must not leave a half
        # tree that the next run takes for the previous one and puts back
        _gone="$(mktemp -d "$SRC_DIR/.src-update.XXXXXX")" && rmdir "$_gone"
        mv -T "$PREV_SRC" "$_gone" || _gone="$PREV_SRC"
        rm -rf "$_gone"
    fi
}
# The restarted service did not come up: put the previous install back and start that,
# so the container is not left without a Studio.
back_out() {
    log "ERROR: $1; putting the previous install back"
    local _clean=1
    restore || _clean=0
    "$SUPCTL" restart studio >/dev/null 2>&1 || "$SUPCTL" start studio >/dev/null 2>&1 || true
    if "$SUPCTL" status studio; then
        log "the previous install is running again"
    else
        log "CRITICAL: the previous install is back but supervisorctl could not start it either (see docker logs)"
    fi
    if [ "$_clean" = "1" ]; then
        echo "unsloth-studio-update: $1; the previous install is back in place." >&2
    else
        echo "unsloth-studio-update: $1; the previous source tree is back but pip could not put every previous package back (see the CRITICAL lines above). Fix the cause and run this again: the next run finishes the restore first." >&2
    fi
    exit 1
}

if [ "$RESTART" = "1" ]; then
    SUPCTL="$(command -v supervisorctl || true)"
    [ -n "$SUPCTL" ] || SUPCTL="/opt/unsloth-venv/bin/supervisorctl"
    # `status` exits 3 for a program that exists but is not running (STOPPED, EXITED,
    # FATAL), which is exactly the Studio a failed earlier update left behind.
    _st=0
    [ -x "$SUPCTL" ] && { "$SUPCTL" status studio >/dev/null 2>&1 || _st=$?; } || _st=4
    if [ "$_st" = "0" ] || [ "$_st" = "3" ]; then
        # a program that is not running gets `start`: `restart` first stops it, which
        # supervisorctl reports as an error
        _cmd=restart
        [ "$_st" = "3" ] && _cmd=start
        log "${_cmd}ing the studio service"
        if ! "$SUPCTL" "$_cmd" studio; then
            back_out "supervisorctl $_cmd studio failed"
        fi
        _wait="${UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT:-180}"
        case "$_wait" in ''|*[!0-9]*) log "ignoring UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT='${_wait}' (not a number of seconds); using 180"; _wait=180;; esac
        if [ "$_wait" -eq 0 ]; then
            # no validation asked for: the restart command succeeded, and that is all
            # this run knows
            commit_update
            log "health check skipped (UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT=0); the update is committed"
        else
            _deadline=$(( $(date +%s) + _wait ))
            _up=0
            while [ "$(date +%s)" -lt "$_deadline" ]; do
                curl -sf -o /dev/null --max-time 3 http://127.0.0.1:8000/api/health && { _up=1; break; }
                sleep 2
            done
            if [ "$_up" = "1" ]; then
                commit_update
                log "Studio is answering on port 8000"
            else
                back_out "Studio did not answer on port 8000 within ${_wait}s (see docker logs)"
            fi
        fi
    else
        # nothing here can start Studio, so the import and frontend checks above are
        # the whole validation
        commit_update
        log "supervisor not managing 'studio' here; restart Studio yourself"
        log "  (e.g. 'docker restart <container>')"
    fi
else
    commit_update
    log "--no-restart: restart Studio to load the update"
    log "  docker exec <container> supervisorctl restart studio"
fi

log "done"
