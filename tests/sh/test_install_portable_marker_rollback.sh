#!/usr/bin/env bash
# Regression test: the .unsloth-portable-root marker must roll back with the venv.
#
# install.sh publishes the marker in _export_portable_roots (before the uv bootstrap) and
# deletes one in _clear_stale_portable_marker (earlier still), yet the exit and signal
# handlers only used to put the previous ENVIRONMENT back. So a `--portable` run over a
# normal install that died in uv / setup.sh / the shim restored the normal venv under a
# marker that keeps storage_roots redirecting the HF caches and the projects root, and a
# normal reinstall over a portable install restored the portable venv with no marker left,
# so `source .../activate` writes outside the volume that install was contained in.
#
# The snippet below is install.sh verbatim -- flag parsing, destination resolution, the
# marker helpers, the publish, the reset, and the whole venv-rollback + trap block -- with a
# forced failure standing in for the first step that can fail. The last section runs the REAL
# storage_roots resolver over the restored trees, so this fails if either half stops agreeing.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
BACKEND="$ROOT/studio/backend"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
# Just the slot variables -- the four marker slots and the launcher pair beside them: the
# helpers that read them live inside blockR, beside the venv rollback, and the two writers
# record inline so each stays runnable on its own.
blockM="$(blk '/^_PORTABLE_MARKER_PATH_1=""$/ {grab=1} grab {print} /^_PORTABLE_SHIM_BACKUP=""$/ {exit}')"
blockE="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockD="$(blk '/^_clear_stale_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockR="$(blk '/^_VENV_ROLLBACK_DIR=""$/ {grab=1} grab {print} /^trap ._on_install_signal 143. TERM$/ {exit}')"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_MARKER_PRIOR_2'*) : ;; *) echo "FAIL: blockM extraction broke"; exit 1 ;; esac
# The portable launcher rolls back on the same handlers; its slot is declared with them.
# tests/sh/test_install_portable_shim_conversion.sh is what exercises it.
case "$blockM" in *'_PORTABLE_SHIM_BACKUP'*) : ;; *) echo "FAIL: blockM lost the launcher slot"; exit 1 ;; esac
case "$blockE" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockE extraction broke"; exit 1 ;; esac
case "$blockE" in *'_PORTABLE_MARKER_PRIOR_1'*) : ;; *) echo "FAIL: blockE stopped recording the publish"; exit 1 ;; esac
case "$blockD" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
case "$blockD" in *'_PORTABLE_MARKER_PRIOR_1'*) : ;; *) echo "FAIL: blockD stopped recording the reset"; exit 1 ;; esac
case "$blockR" in *'_on_install_signal'*) : ;; *) echo "FAIL: blockR extraction broke"; exit 1 ;; esac
case "$blockR" in *'_restore_studio_venv_replacement'*) : ;; *) echo "FAIL: blockR lost the venv rollback"; exit 1 ;; esac
# The restore and commit helpers must stay inside the venv-rollback block: the handlers that
# call them are extracted with it, here and by tests/sh/test_install_rollback_lifecycle.sh,
# and a helper defined outside that range makes every one of those snippets exit 127.
case "$blockR" in *'_restore_portable_marker_slot() {'*) : ;; *) echo "FAIL: the marker restore is defined outside the rollback block"; exit 1 ;; esac
case "$blockR" in *'_commit_portable_marker() {'*) : ;; *) echo "FAIL: the marker commit is defined outside the rollback block"; exit 1 ;; esac
# Neither writer may depend on a helper defined elsewhere, or lifting it out on its own
# (which tests/sh/test_install_portable_marker_reset.sh does) stops working.
case "$blockD" in *'_portable_marker_prior'*) echo "FAIL: blockD calls a helper it does not define"; exit 1 ;; *) : ;; esac
case "$blockE" in *'_portable_marker_prior'*) echo "FAIL: blockE calls a helper it does not define"; exit 1 ;; *) : ;; esac

# Both handlers really call the restore, and the installer really commits it. Grepping the
# file, not the snippet: the snippet stops before the call sites of the commit.
_exit_block="$(blk '/^_on_install_exit\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
_signal_block="$(blk '/^_on_install_signal\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
case "$_exit_block" in *'_restore_portable_marker'*) : ;; *) echo "FAIL: the EXIT handler does not restore the marker"; exit 1 ;; esac
case "$_signal_block" in *'_restore_portable_marker'*) : ;; *) echo "FAIL: the signal handler does not restore the marker"; exit 1 ;; esac
grep -q '^_commit_portable_marker$' "$INSTALL" || { echo "FAIL: install.sh never commits the marker"; exit 1; }
# The commit has to sit past every step that can fail the install: after the setup.sh gate,
# or a failing setup would keep a marker whose environment was just rolled back.
_commit_line=$(grep -n '^_commit_portable_marker$' "$INSTALL" | head -n1 | cut -d: -f1)
_setup_gate=$(grep -n '^    exit "\$_SETUP_EXIT"$' "$INSTALL" | head -n1 | cut -d: -f1)
_launch_exit=$(grep -n '^            exit "\$_LAUNCH_EXIT"$' "$INSTALL" | head -n1 | cut -d: -f1)
[ -n "$_setup_gate" ] || { echo "FAIL: could not locate the setup.sh failure gate"; exit 1; }
[ -n "$_launch_exit" ] || { echo "FAIL: could not locate the autostart exit"; exit 1; }
[ "$_commit_line" -gt "$_setup_gate" ] || { echo "FAIL: the marker commits before setup.sh can fail the install"; exit 1; }
[ "$_commit_line" -lt "$_launch_exit" ] || { echo "FAIL: the marker commits after the autostart, which can exit nonzero"; exit 1; }

# set -e is on in install.sh, so a helper ending in a bare `[ cond ] && action` would kill
# the run the moment the condition is false.
SNIP='set -e
C_WARN=""
substep() { printf "  . %s\n" "$1"; }
rollback_substep() { printf "  R %s\n" "$1"; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockM"'
'"$blockE"'
_export_portable_roots
'"$blockD"'
_clear_stale_portable_marker
_UNSLOTH_LOGIN_PATH="$PATH"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
'"$blockR"'
# ── everything above is install.sh verbatim ──
printf "reached|%s\n" "$STUDIO_HOME"
# The real install moves the existing environment aside here (the legacy-layout migration
# and the venv replacement both go through this function).
if [ -d "$VENV_DIR" ]; then _start_studio_venv_replacement "$VENV_DIR"; fi
case "${FAIL_MODE:-fail}" in
    ok)
        _commit_studio_venv_replacement
        _commit_portable_marker
        ;;
    launch)
        # Install committed, then the post-install autostart returns nonzero.
        _commit_studio_venv_replacement
        _commit_portable_marker
        exit 7
        ;;
    signal)
        kill -INT $$
        sleep 5
        ;;
    *)
        exit 7
        ;;
esac'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
# mktemp, not a counter: the increment in `H="$(new_home)"` would happen in a subshell,
# every case would share one HOME, and the trees would contaminate each other.
new_home() { mktemp -d "$T/home.XXXXXX"; }

marker_state() { # dir
    if [ -f "$1/.unsloth-portable-root" ]; then printf present; else printf gone; fi
}
marker_body() { # dir
    cat "$1/.unsloth-portable-root" 2>/dev/null || printf '(none)'
}
venv_tag() { # venv dir
    cat "$1/tag" 2>/dev/null || printf '(none)'
}
# The rollback copies must not survive as litter either.
strays() { # studio home
    set -- "$1"/unsloth_studio.rollback.*
    if [ -e "$1" ]; then printf '%s' "$#"; else printf 0; fi
}

run_install() { # fakehome [env assignments and args, "--" separates]
    _home="$1"; shift
    _env=""
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do _env="$_env $1"; shift; done
    [ "$#" -eq 0 ] || shift
    # env -i: the caller's own UNSLOTH_* would mask the branches under test.
    # shellcheck disable=SC2086
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" $_env \
        bash -c "$SNIP" _ "$@" > "$T/out" 2>"$T/err"
    _rc=$?
    if ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet never reached the install (rc=%s)\n%s\n%s\n' \
            "$_rc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
    printf '%s' "$_rc"
}

# ── C. --portable publishes the marker before anything is installed ─────────────────

# C1. Over an existing NORMAL install. The failure restores that install, so the marker the
# run published must go with it, or the restored tree reads as portable forever.
H="$(new_home)"
mkdir -p "$H/.unsloth/studio/unsloth_studio"
printf 'normal\n' > "$H/.unsloth/studio/unsloth_studio/tag"
rc="$(run_install "$H" -- --portable)"
check "C1 a failed --portable exits nonzero" 7 "$rc"
check "C1 the previous normal environment is back" normal "$(venv_tag "$H/.unsloth/studio/unsloth_studio")"
check "C1 the marker it published is gone again" gone "$(marker_state "$H/.unsloth")"
check "C1 no rollback copy is left behind" 0 "$(strays "$H/.unsloth/studio")"

# C2. Over an existing PORTABLE install, the marker was already the user's. Rolling back
# must restore it, not delete it: nothing about that install changed.
H2="$(new_home)"
mkdir -p "$H2/.unsloth/studio/unsloth_studio"
printf 'portable\n' > "$H2/.unsloth/studio/unsloth_studio/tag"
printf '%s\n' "$H2/.unsloth" > "$H2/.unsloth/.unsloth-portable-root"
rc="$(run_install "$H2" -- --portable)"
check "C2 a failed portable reinstall exits nonzero" 7 "$rc"
check "C2 the marker that was already there survives" present "$(marker_state "$H2/.unsloth")"
check "C2 and still names the same root" "$H2/.unsloth" "$(marker_body "$H2/.unsloth")"

# C3. A fresh portable install into an empty root: nothing to restore, the marker still goes.
H3="$(new_home)"
mkdir -p "$H3/vol"
rc="$(run_install "$H3" "UNSLOTH_HOME=$H3/vol" --)"
check "C3 a failed fresh portable install exits nonzero" 7 "$rc"
check "C3 leaves no marker in the half-built root" gone "$(marker_state "$H3/vol")"

# C4. The install COMMITTED and only the post-install autostart failed. The exit status is
# nonzero, so the handler runs, and it must keep everything the install just wrote.
H4="$(new_home)"
mkdir -p "$H4/.unsloth/studio/unsloth_studio"
printf 'normal\n' > "$H4/.unsloth/studio/unsloth_studio/tag"
rc="$(run_install "$H4" FAIL_MODE=launch -- --portable)"
check "C4 a failed autostart still exits nonzero" 7 "$rc"
check "C4 but the portable marker is kept" present "$(marker_state "$H4/.unsloth")"
check "C4 and names the root it installed" "$H4/.unsloth" "$(marker_body "$H4/.unsloth")"

# C5. A successful portable install keeps its marker (the whole point of publishing it).
H5="$(new_home)"
mkdir -p "$H5/.unsloth/studio/unsloth_studio"
printf 'normal\n' > "$H5/.unsloth/studio/unsloth_studio/tag"
rc="$(run_install "$H5" FAIL_MODE=ok -- --portable)"
check "C5 a successful portable install exits 0" 0 "$rc"
check "C5 and keeps its marker" present "$(marker_state "$H5/.unsloth")"

# ── D. A normal reinstall clears the marker before anything is installed ────────────

# D1. Nested layout: the marker lives one level above STUDIO_HOME. A failure restores the
# portable venv, so the marker it is reached through has to come back with it.
H6="$(new_home)"
mkdir -p "$H6/.unsloth/studio/unsloth_studio"
printf 'portable\n' > "$H6/.unsloth/studio/unsloth_studio/tag"
printf '%s\n' "$H6/.unsloth" > "$H6/.unsloth/.unsloth-portable-root"
rc="$(run_install "$H6" --)"
check "D1 a failed normal reinstall exits nonzero" 7 "$rc"
check "D1 the previous portable environment is back" portable "$(venv_tag "$H6/.unsloth/studio/unsloth_studio")"
check "D1 and so is the marker that resolves it" present "$(marker_state "$H6/.unsloth")"
check "D1 with its original contents" "$H6/.unsloth" "$(marker_body "$H6/.unsloth")"

# D2. Flat layout: the master root IS the Studio root, so the marker sits in the directory
# being installed into (slot 1 rather than the parent slot).
H7="$(new_home)"
mkdir -p "$H7/flat/unsloth_studio"
printf 'portable\n' > "$H7/flat/unsloth_studio/tag"
printf '%s\n' "$H7/flat" > "$H7/flat/.unsloth-portable-root"
rc="$(run_install "$H7" "UNSLOTH_STUDIO_HOME=$H7/flat" --)"
check "D2 a failed flat-root reinstall exits nonzero" 7 "$rc"
check "D2 restores the flat portable marker" present "$(marker_state "$H7/flat")"
check "D2 with its original contents" "$H7/flat" "$(marker_body "$H7/flat")"

# D3. Both spellings at once: <root>/studio/<marker> and <root>/<marker> are cleared in the
# same run, which is why the rollback keeps two slots rather than one.
H8="$(new_home)"
mkdir -p "$H8/vol/studio/unsloth_studio"
printf 'portable\n' > "$H8/vol/studio/unsloth_studio/tag"
printf '%s\n' "$H8/vol" > "$H8/vol/.unsloth-portable-root"
printf '%s\n' "$H8/vol/studio" > "$H8/vol/studio/.unsloth-portable-root"
rc="$(run_install "$H8" "UNSLOTH_STUDIO_HOME=$H8/vol/studio" --)"
check "D3 a failed two-marker reinstall exits nonzero" 7 "$rc"
check "D3 restores the parent marker" present "$(marker_state "$H8/vol")"
check "D3 restores the studio-level marker" present "$(marker_state "$H8/vol/studio")"
check "D3 the parent marker keeps its contents" "$H8/vol" "$(marker_body "$H8/vol")"
check "D3 the studio marker keeps its contents" "$H8/vol/studio" "$(marker_body "$H8/vol/studio")"

# D4. A SUCCESSFUL normal reinstall is the case the reset exists for: the marker stays gone.
H9="$(new_home)"
mkdir -p "$H9/.unsloth/studio/unsloth_studio"
printf 'portable\n' > "$H9/.unsloth/studio/unsloth_studio/tag"
printf '%s\n' "$H9/.unsloth" > "$H9/.unsloth/.unsloth-portable-root"
rc="$(run_install "$H9" FAIL_MODE=ok --)"
check "D4 a successful normal reinstall exits 0" 0 "$rc"
check "D4 and the stale marker stays removed" gone "$(marker_state "$H9/.unsloth")"

# D5. Committed, then the autostart fails. The conversion must stand.
H10="$(new_home)"
mkdir -p "$H10/.unsloth/studio/unsloth_studio"
printf 'portable\n' > "$H10/.unsloth/studio/unsloth_studio/tag"
printf '%s\n' "$H10/.unsloth" > "$H10/.unsloth/.unsloth-portable-root"
rc="$(run_install "$H10" FAIL_MODE=launch --)"
check "D5 a failed autostart still exits nonzero" 7 "$rc"
check "D5 but the conversion to a normal install stands" gone "$(marker_state "$H10/.unsloth")"

# ── Signals. Ctrl-C is the interruption users actually produce. ─────────────────────
H11="$(new_home)"
mkdir -p "$H11/.unsloth/studio/unsloth_studio"
printf 'portable\n' > "$H11/.unsloth/studio/unsloth_studio/tag"
printf '%s\n' "$H11/.unsloth" > "$H11/.unsloth/.unsloth-portable-root"
rc="$(run_install "$H11" FAIL_MODE=signal --)"
check "S1 an interrupted reinstall exits 130" 130 "$rc"
check "S1 the previous portable environment is back" portable "$(venv_tag "$H11/.unsloth/studio/unsloth_studio")"
check "S1 and so is its marker" present "$(marker_state "$H11/.unsloth")"

H12="$(new_home)"
mkdir -p "$H12/.unsloth/studio/unsloth_studio"
printf 'normal\n' > "$H12/.unsloth/studio/unsloth_studio/tag"
rc="$(run_install "$H12" FAIL_MODE=signal -- --portable)"
check "S2 an interrupted --portable exits 130" 130 "$rc"
check "S2 and drops the marker it published" gone "$(marker_state "$H12/.unsloth")"

# ── The runtime half. Without this the checks above only prove a file moved. ────────
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
home = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "portable": sr.portable_mode(),
    "unsloth_home": str(home) if home else None,
}))
PYEOF
    probe() { # fakehome field [env...]
        _phome="$1"; _pfield="$2"; shift 2
        # shellcheck disable=SC2086
        _pout=$(env -i HOME="$_phome" PATH="$PATH" _BACKEND="$BACKEND" "$@" \
            python3 "$PROBE" 2>"$T/perr")
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            printf '  FAIL  storage_roots probe produced no output\n%s\n' "$(cat "$T/perr")"
            fails=$((fails + 1))
            printf 'probe-failed'
            return 0
        fi
        printf '%s' "$_pjson" | _FIELD="$_pfield" python3 -c \
            'import json,os,sys; v=json.load(sys.stdin)[os.environ["_FIELD"]]; print("null" if v is None else str(v).lower() if isinstance(v,bool) else v)'
    }
    # C1 restored a NORMAL install. The runtime must agree, with no environment to help it.
    check "the tree C1 restored reads as non-portable" false "$(probe "$H" portable)"
    check "and resolves no master root" null "$(probe "$H" unsloth_home)"
    # D1 restored a PORTABLE install, and the marker is the only signal that survives on
    # disk, so an `unsloth` reached past the shim has to find the root through it.
    check "the tree D1 restored still reads as portable" true "$(probe "$H6" portable)"
    check "and resolves the portable root it was contained in" "$H6/.unsloth" "$(probe "$H6" unsloth_home)"
    # D4 committed the conversion; the runtime must see a normal install.
    check "the tree D4 converted reads as non-portable" false "$(probe "$H9" portable)"
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
