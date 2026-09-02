#!/usr/bin/env bash
# Regression test: a NORMAL install over a tree an earlier `--portable` run
# created drops the .unsloth-portable-root marker it would be read through.
#
# The marker is the only portable signal that survives on disk, and
# storage_roots.portable_mode() is true whenever unsloth_home() is, so a stale
# one keeps the reinstalled tree redirecting the HF caches and the projects root
# with UNSLOTH_PORTABLE=0 powerless to turn it off. The last section runs the
# REAL resolver over the tree the installer blocks just produced, so this test
# fails if either half stops agreeing.
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

blockA="$(awk '
    /^# ── Parse flags ──$/ {grab = 1}
    grab {print}
    /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen = 1}
    seen && /^fi$/ {exit}
' "$INSTALL")"

blockB="$(awk '
    /^_resolve_studio_destinations\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

blockD="$(awk '
    /^_clear_stale_portable_marker\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockD" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
case "$blockD" in *'_PORTABLE_MODE'*) : ;; *) echo "FAIL: blockD lost the portable guard"; exit 1 ;; esac
# The installer really calls it, and after the destinations are resolved.
grep -q '^_clear_stale_portable_marker$' "$INSTALL" || { echo "FAIL: install.sh never calls _clear_stale_portable_marker"; exit 1; }
_call_line=$(grep -n '^_clear_stale_portable_marker$' "$INSTALL" | head -n1 | cut -d: -f1)
_resolve_line=$(grep -n '^_resolve_studio_destinations$' "$INSTALL" | head -n1 | cut -d: -f1)
[ "$_call_line" -gt "$_resolve_line" ] || { echo "FAIL: the marker reset runs before STUDIO_HOME is resolved"; exit 1; }
# setup.sh reads the marker too (_setup_portable_mode), so the reset has to come first.
_setup_line=$(grep -n 'SETUP_SH=' "$INSTALL" | head -n1 | cut -d: -f1)
[ -n "$_setup_line" ] || { echo "FAIL: could not locate the setup.sh handoff"; exit 1; }
[ "$_call_line" -lt "$_setup_line" ] || { echo "FAIL: the marker reset runs after setup.sh"; exit 1; }

# set -e is on: a bare `[ cond ] && action` inside the block would kill the run.
SNIP='set -e
C_WARN=""
substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockD"'
_clear_stale_portable_marker
printf "reached|%s\n" "$STUDIO_HOME"'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
# mktemp, not a counter: the increment in `H="$(new_home)"` would happen in a
# subshell, every case would share one HOME, and the trees would contaminate.
new_home() { mktemp -d "$T/home.XXXXXX"; }

# env -i: the caller's own UNSLOTH_* would mask the branches under test.
run_install() { # fakehome [env assignments and args, "--" separates]
    _home="$1"; shift
    _env=""
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do _env="$_env $1"; shift; done
    [ "$#" -eq 0 ] || shift
    # shellcheck disable=SC2086
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" $_env \
        bash -c "$SNIP" _ "$@" > "$T/out" 2>"$T/err"
    _rc=$?
    if [ "$_rc" -ne 0 ] || ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet aborted (rc=%s)\n%s\n%s\n' "$_rc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
}

marker_state() { # path
    if [ -f "$1/.unsloth-portable-root" ]; then printf present; else printf gone; fi
}

# ── 1. The default collision: `--portable` with no --root uses $HOME/.unsloth,
# whose studio/ child is exactly the default NORMAL install root.
H="$(new_home)"
mkdir -p "$H/.unsloth/studio/unsloth_studio/bin" "$H/.unsloth/share" "$H/.unsloth/bin"
printf '%s\n' "$H/.unsloth" > "$H/.unsloth/.unsloth-portable-root"
run_install "$H" --
check "normal reinstall drops the marker above the default studio root" gone "$(marker_state "$H/.unsloth")"

# ── 2. A portable run must keep its own marker.
H2="$(new_home)"
mkdir -p "$H2/.unsloth/studio"
printf '%s\n' "$H2/.unsloth" > "$H2/.unsloth/.unsloth-portable-root"
run_install "$H2" -- --portable
check "--portable keeps the marker" present "$(marker_state "$H2/.unsloth")"

# ── 3. Same, seeded from the environment rather than the flag.
H3="$(new_home)"
mkdir -p "$H3/.unsloth/studio"
printf '%s\n' "$H3/.unsloth" > "$H3/.unsloth/.unsloth-portable-root"
run_install "$H3" UNSLOTH_PORTABLE=1 --
check "UNSLOTH_PORTABLE=1 keeps the marker" present "$(marker_state "$H3/.unsloth")"

# ── 3b. --shortcuts-only installs nothing, so it must not convert a tree back.
H3b="$(new_home)"
mkdir -p "$H3b/.unsloth/studio"
printf '%s\n' "$H3b/.unsloth" > "$H3b/.unsloth/.unsloth-portable-root"
run_install "$H3b" -- --shortcuts-only
check "--shortcuts-only keeps the marker" present "$(marker_state "$H3b/.unsloth")"

# ── 4. Flat layout: the master root IS the Studio root, so the marker sits in
# the directory the normal install is pointed at.
H4="$(new_home)"
mkdir -p "$H4/flat/unsloth_studio"
printf '%s\n' "$H4/flat" > "$H4/flat/.unsloth-portable-root"
run_install "$H4" "UNSLOTH_STUDIO_HOME=$H4/flat" --
check "normal reinstall drops the marker in a flat portable root" gone "$(marker_state "$H4/flat")"

# ── 5. ...unless a NESTED portable install still resolves through that marker.
H5="$(new_home)"
mkdir -p "$H5/vol/studio/unsloth_studio"
printf '%s\n' "$H5/vol" > "$H5/vol/.unsloth-portable-root"
run_install "$H5" "UNSLOTH_STUDIO_HOME=$H5/vol" --
check "a marker another install still uses is kept" present "$(marker_state "$H5/vol")"

# ── 6. Scoping: <root>/studio is the only child a parent marker names. Some
# other directory under a portable root is a different tree.
H6="$(new_home)"
mkdir -p "$H6/vol/studio/unsloth_studio" "$H6/vol/other"
printf '%s\n' "$H6/vol" > "$H6/vol/.unsloth-portable-root"
run_install "$H6" "UNSLOTH_STUDIO_HOME=$H6/vol/other" --
check "a sibling install keeps the portable root's marker" present "$(marker_state "$H6/vol")"

# ── 7. Nothing to remove is not an error (fresh machine, set -e still on).
H7="$(new_home)"
run_install "$H7" --
check "a fresh install with no marker still completes" gone "$(marker_state "$H7/.unsloth")"

# ── 8. The runtime half. Case 1's tree, through the real resolver: portable
# before the reset, normal after it. Without this the shell checks above only
# prove a file moved.
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
    probe() { # fakehome [env...]
        _phome="$1"; shift
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
        printf '%s' "$_pjson" | python3 -c 'import json,sys; print(str(json.load(sys.stdin)["portable"]).lower())'
    }
    H8="$(new_home)"
    mkdir -p "$H8/.unsloth/studio/unsloth_studio/bin"
    printf '%s\n' "$H8/.unsloth" > "$H8/.unsloth/.unsloth-portable-root"
    check "the stale marker really does read as portable" true "$(probe "$H8")"
    # portable_mode() reads UNSLOTH_PORTABLE as an opt-IN and falls through to
    # unsloth_home(), so 0 is not an off switch: that is why the cleanup has to
    # happen at install time. Flip this expectation if the runtime ever grows an
    # explicit opt-out.
    check "and UNSLOTH_PORTABLE=0 cannot turn it off" true "$(probe "$H8" UNSLOTH_PORTABLE=0)"
    run_install "$H8" --
    check "after a normal reinstall the runtime reads as non-portable" false "$(probe "$H8")"
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
