#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Behaviour tests for the #7803 fix: the Python request handed to uv, and the
# guard that recreates a venv left on a skipped interpreter by an earlier run.
# The real helpers and the real guard block are extracted from install.sh and
# executed against a stubbed uv, so this cannot drift into testing a copy.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

_HELPERS=$(mktemp)
{
    # Quiet stand-ins for the reporting helpers the extracted functions call.
    printf 'substep() { :; }\nrollback_substep() { :; }\n'
    sed -n '/^PYTHON_SKIP=/p' "$INSTALL_SH"
    sed -n '/^_python_skip_applies()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_python_is_skipped()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_python_request()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_start_studio_venv_replacement()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_discard_venv_for_recreate()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_restore_studio_venv_replacement()/,/^}/p' "$INSTALL_SH"
} > "$_HELPERS"
for _needed in _python_skip_applies _python_is_skipped _python_request _start_studio_venv_replacement \
               _discard_venv_for_recreate _restore_studio_venv_replacement; do
    grep -q "^$_needed()" "$_HELPERS" || {
        echo "  FAIL: could not extract $_needed from install.sh"
        exit 1
    }
done
# shellcheck disable=SC1090
. "$_HELPERS"

echo "=== the request handed to uv ==="

assert_eq "a bare 3.13 asks for its own series minus the bad patch" \
    ">=3.13,<3.14,!=3.13.8" "$(_python_request 3.13)"
# Not a floor: an offline host, or a uv whose manifest predates 3.13.9, may still
# have a good cached 3.13.7, and ">=3.13.9" would refuse it and fail the install.
assert_eq "the request never becomes a floor above the bad patch" \
    "" "$(_python_request 3.13 | grep -o '>=3\.13\.9' || true)"
assert_eq "a minor with nothing skipped still gets its own series" \
    ">=3.12,<3.13" "$(_python_request 3.12)"
assert_eq "an explicit patch from --python is the user's choice" \
    "3.13.8" "$(_python_request 3.13.8)"
assert_eq "a --python path is not a version and is passed through" \
    "/usr/bin/python3.13" "$(_python_request /usr/bin/python3.13)"
# The exclusions are generated from PYTHON_SKIP, so adding a patch there is the
# only edit a future bad release needs.
_saved_skip="$PYTHON_SKIP"
PYTHON_SKIP="3.13.8 3.13.20 3.12.4"
assert_eq "every skipped patch in the series is excluded" \
    ">=3.13,<3.14,!=3.13.8,!=3.13.20" "$(_python_request 3.13)"
assert_eq "a skipped patch from another series is not" \
    ">=3.12,<3.13,!=3.12.4" "$(_python_request 3.12)"
PYTHON_SKIP="$_saved_skip"

echo "=== values that are not a plain X.Y ==="

# dash aborts the whole install on "Illegal number", so anything that could
# reach the arithmetic has to be turned away before it.
assert_eq "a relative path whose first segment looks like a version" \
    "3.13/bin/python" "$(_python_request 3.13/bin/python)"
assert_eq "a Windows-style path" \
    "C:\\Python313\\python.exe" "$(_python_request 'C:\Python313\python.exe')"
assert_eq "a prerelease tag is not arithmetic" \
    "3.13rc1" "$(_python_request 3.13rc1)"
assert_eq "a uv download name is passed through" \
    "cpython-3.13-macos-aarch64-none" "$(_python_request cpython-3.13-macos-aarch64-none)"

echo "=== --no-torch does not need a torch-capable interpreter ==="

_saved_skip_torch="${SKIP_TORCH:-false}"
SKIP_TORCH=true
assert_eq "the request is left alone when torch is never installed" \
    "3.13" "$(_python_request 3.13)"
if _python_is_skipped "3.13.8"; then
    assert_eq "a skipped patch is usable without torch" "no" "yes"
else
    assert_eq "a skipped patch is usable without torch" "no" "no"
fi
SKIP_TORCH="$_saved_skip_torch"
if _python_is_skipped "3.13.8"; then
    assert_eq "and is skipped again once torch is back" "yes" "yes"
else
    assert_eq "and is skipped again once torch is back" "yes" "no"
fi

echo "=== the uv version probe on an image with no awk ==="

# The comment on that block says an unreadable version counts as "uv present".
# Without the guard the pipeline exits 127 and set -e kills the install first,
# which is exactly the host the block exists to keep working.
_PROBE=$(mktemp)
sed -n '/^        _uv_prev_ver=\$(uv --version/,/_uv_prev_ver=""$/p' "$INSTALL_SH" > "$_PROBE"
[ -s "$_PROBE" ] || { echo "  FAIL: could not extract the uv version probe"; exit 1; }
_probe_work=$(mktemp -d)
mkdir -p "$_probe_work/bin"
printf '#!/bin/sh\necho "uv 0.9.2"\n' > "$_probe_work/bin/uv"
chmod +x "$_probe_work/bin/uv"
# PATH is narrowed inside the child, not around it: narrowing it around the
# child would hide `sh` itself and the test would pass for the wrong reason.
_probe_out=$(sh -c "PATH='$_probe_work/bin'; export PATH; set -e; . '$_PROBE'; echo \"SURVIVED:\${_uv_prev_ver:-empty}\"" 2>&1 || true)
assert_eq "no awk means an unreadable version, not a dead install" \
    "SURVIVED:empty" "$(printf '%s' "$_probe_out" | tail -1)"
rm -rf "$_probe_work" "$_PROBE"

echo "=== the skip list ==="

if _python_is_skipped "3.13.8"; then
    assert_eq "3.13.8 is skipped" "yes" "yes"
else
    assert_eq "3.13.8 is skipped" "yes" "no"
fi
if _python_is_skipped "3.13.12"; then
    assert_eq "a good patch is not skipped" "no" "yes"
else
    assert_eq "a good patch is not skipped" "no" "no"
fi
if _python_is_skipped ""; then
    assert_eq "an unreadable version is not skipped" "no" "yes"
else
    assert_eq "an unreadable version is not skipped" "no" "no"
fi

echo "=== the venv guard ==="

_GUARD=$(mktemp)
sed -n '/^# The request above only decides/,/^fi$/p' "$INSTALL_SH" > "$_GUARD"
[ -s "$_GUARD" ] || { echo "  FAIL: could not extract the venv guard"; exit 1; }

# Runs the guard against a fake venv whose python reports $1, with uv stubbed.
# Echoes the request uv was asked for, or nothing when the guard did not fire.
run_guard() {
    _reported="$1"
    _user_python="${2:-}"
    _work=$(mktemp -d)
    mkdir -p "$_work/venv/bin"
    cat > "$_work/venv/bin/python" <<EOF
#!/bin/sh
echo "$_reported"
EOF
    chmod +x "$_work/venv/bin/python"

    (
        set -e
        STUDIO_HOME="$_work"
        VENV_DIR="$_work/venv"
        _VENV_ROLLBACK_DIR=""
        _VENV_ROLLBACK_TARGET="$VENV_DIR"
        _VENV_ROLLBACK_ACTIVE=false
        _USER_PYTHON="$_user_python"
        PYTHON_VERSION="3.13"
        # shellcheck disable=SC1090
        . "$_HELPERS"
        _run_uv_venv() {
            shift  # label
            shift  # target dir
            shift  # --python
            echo "REQUEST=$1" >&2
            mkdir -p "$VENV_DIR/bin"
            printf '#!/bin/sh\necho 3.13.12\n' > "$VENV_DIR/bin/python"
            chmod +x "$VENV_DIR/bin/python"
        }
        # shellcheck disable=SC1090
        . "$_GUARD"
    ) 2>&1 >/dev/null | sed -n 's/^REQUEST=//p'
    rm -rf "$_work"
}

assert_eq "a venv left on 3.13.8 is recreated on the screened request" \
    ">=3.13,<3.14,!=3.13.8" "$(run_guard 3.13.8)"
assert_eq "a healthy venv is left alone" \
    "" "$(run_guard 3.13.12)"
assert_eq "an unreadable interpreter is left alone" \
    "" "$(run_guard '')"
assert_eq "--python is honoured even on a skipped version" \
    "" "$(run_guard 3.13.8 /usr/bin/python3.13)"

echo "=== a failed recreate must not cost the user their environment ==="

# The legacy-layout migration moves $STUDIO_HOME/.venv into $VENV_DIR without
# arming _start_studio_venv_replacement, so the guard runs with no rollback in
# place. If it removed the venv outright, a `uv venv` that cannot resolve an
# interpreter (offline, or a uv older than the requested patch) would leave the
# machine with nothing. $_rollback_active mirrors whether a replacement is
# already in flight; $_recreate_rc is what the stubbed uv returns.
# A separate `sh`, not a subshell: `( ... ) || true` puts the subshell in an ||
# list, which switches set -e off for everything inside it, so the guard would
# never abort the way it does in the real installer.
_DRIVER=$(mktemp)
cat > "$_DRIVER" <<'DRIVER'
STUDIO_HOME="$1"
VENV_DIR="$1/venv"
_VENV_ROLLBACK_DIR=""
_VENV_ROLLBACK_TARGET="$VENV_DIR"
_VENV_ROLLBACK_ACTIVE=false
_USER_PYTHON=""
PYTHON_VERSION="3.13"
# shellcheck disable=SC1090
. "$2"
if [ "$3" = true ]; then
    # Stand in for the main path, which moved the user's real venv aside itself
    # and then created the fresh one the guard is about to replace.
    mkdir -p "$1/already-preserved"
    _VENV_ROLLBACK_DIR="$1/already-preserved"
    _VENV_ROLLBACK_ACTIVE=true
fi
_stub_rc="$4"
_run_uv_venv() { return "$_stub_rc"; }
set -e
# What _on_install_exit does for a non-zero status.
trap '[ "$?" -eq 0 ] || _restore_studio_venv_replacement' EXIT
# shellcheck disable=SC1090
. "$5"
DRIVER

run_guard_failure() {
    _rollback_active="$1"
    _recreate_rc="$2"
    _work=$(mktemp -d)
    mkdir -p "$_work/venv/bin"
    printf '#!/bin/sh\necho 3.13.8\n' > "$_work/venv/bin/python"
    chmod +x "$_work/venv/bin/python"
    # Only present in the environment the user already had.
    : > "$_work/venv/USER_DATA"

    sh "$_DRIVER" "$_work" "$_HELPERS" "$_rollback_active" "$_recreate_rc" "$_GUARD" \
        >/dev/null 2>&1 || true

    if [ -f "$_work/venv/USER_DATA" ]; then echo "preserved"; else echo "lost"; fi
    rm -rf "$_work"
}

assert_eq "a migrated venv survives a recreate that fails" \
    "preserved" "$(run_guard_failure false 1)"
assert_eq "a rollback copy is not clobbered when one is already in flight" \
    "lost" "$(run_guard_failure true 1)"
assert_eq "a recreate that works still replaces the environment" \
    "lost" "$(run_guard_failure false 0)"

rm -f "$_HELPERS" "$_GUARD" "$_DRIVER"

# ── The --python / UNSLOTH_PYTHON range gate (#8495) ──
# The version was never range-checked, so an unsupported minor reached `uv venv`
# and failed minutes later inside dependency resolution with a bare "pyarrow"
# (constraints.txt pins pyarrow==23.0.1, which has no cp39 wheel on any platform;
# matplotlib, pymupdf, pymupdf4llm and fastmcp are 3.10+ too).
#
# The check is inline argument parsing, not a function, so run the real prefix of
# install.sh: everything up to the Tauri helpers, which is before any network or
# filesystem work. A copy of the logic here could not catch it moving.
# The real block, extracted like every other test here: it is inline argument
# validation rather than a function, so it is pulled out by its own first and last
# lines and driven with _USER_PYTHON pre-set. A reimplementation would keep passing
# after install.sh changed, which is the failure mode this file exists to avoid.
_GATE=$(mktemp)
{
    # $2 is the --shortcuts-only flag: the gate only judges a run that will
    # actually select an interpreter.
    printf '_USER_PYTHON="$1"\n_SHORTCUTS_ONLY="${2:-false}"\n'
    # The verdicts moved into _check_python_request so a path-style request reaches
    # them too, so the function comes along with the block that calls it.
    sed -n '/^_check_python_request()/,/^}/p' "$INSTALL_SH"
    awk '/^if \[ -n "\$_USER_PYTHON" \] && \[ "\$_SHORTCUTS_ONLY" != true \]; then$/{f=1} f{print} f && /^fi$/{exit}' "$INSTALL_SH"
} > "$_GATE"
grep -q '_req_minor' "$_GATE" || { echo "  FAIL: could not extract the python range gate"; FAIL=$((FAIL + 1)); }

run_python_gate() {  # version -> "rejected" | "accepted"
    if sh "$_GATE" "$1" >/dev/null 2>&1; then echo "accepted"; else echo "rejected"; fi
}

assert_eq "3.9 is rejected before uv is asked for a venv" \
    "rejected" "$(run_python_gate 3.9)"
# `unsloth studio update` re-runs the installer with --shortcuts-only to refresh the
# launcher, handing it the caller's whole environment (unsloth_cli/commands/studio.py).
# That run selects no interpreter, so a stale UNSLOTH_PYTHON=3.9 exported years ago
# must not fail it: the helper only prints the status, so the update would look
# successful with the launcher left unwritten.
assert_eq "--shortcuts-only does not judge the python request" \
    "accepted" "$(if sh "$_GATE" 3.9 true >/dev/null 2>&1; then echo accepted; else echo rejected; fi)"
assert_eq "2.7 is rejected" \
    "rejected" "$(run_python_gate 2.7)"
# 3.10 is rejected with everything below it: both bundled Data Designer plugins
# declare requires-python >= 3.11 and install_python_stack.py installs them
# unconditionally, so a 3.10 run dies on a local project uv refuses, near the end
# of setup -- the late failure this gate exists to replace.
assert_eq "3.10 is rejected: the Data Designer plugins need 3.11" \
    "rejected" "$(run_python_gate 3.10)"
assert_eq "3.11 is the floor and is allowed" \
    "accepted" "$(run_python_gate 3.11)"
assert_eq "3.13 is allowed" \
    "accepted" "$(run_python_gate 3.13)"
# Newer than tested is a warning, not a failure: unproven is not known-broken.
assert_eq "3.14 warns but continues" \
    "accepted" "$(run_python_gate 3.14)"
# Not versions at all, and not this gate's business.
assert_eq "an explicit interpreter path passes through" \
    "accepted" "$(run_python_gate /usr/bin/python3.12)"
# A path names an interpreter as surely as "3.9" does. Real executables, built here,
# because the gate asks the interpreter rather than parsing its name.
_FAKE_DIR=$(mktemp -d)
printf '#!/bin/sh\necho "3.9"\n' > "$_FAKE_DIR/py39"
printf '#!/bin/sh\necho "3.12"\n' > "$_FAKE_DIR/py312"
chmod +x "$_FAKE_DIR/py39" "$_FAKE_DIR/py312"
assert_eq "a path to an unsupported interpreter is rejected" \
    "rejected" "$(run_python_gate "$_FAKE_DIR/py39")"
assert_eq "a path to a supported interpreter is accepted" \
    "accepted" "$(run_python_gate "$_FAKE_DIR/py312")"
assert_eq "a path that cannot be run is left to the steps that resolve it" \
    "accepted" "$(run_python_gate "$_FAKE_DIR/does-not-exist")"
rm -rf "$_FAKE_DIR"
assert_eq "a uv download name passes through" \
    "accepted" "$(run_python_gate cpython-3.12-linux-aarch64-none)"
assert_eq "a prerelease string passes through instead of aborting dash" \
    "accepted" "$(run_python_gate 3.13rc1)"
assert_eq "an empty request is left to the default" \
    "accepted" "$(run_python_gate "")"

case "$(sh "$_GATE" 3.9 2>&1 || true)" in
    *pyarrow*) assert_eq "the rejection message names pyarrow" "yes" "yes" ;;
    *) assert_eq "the rejection message names pyarrow" "yes" "no" ;;
esac
case "$(sh "$_GATE" 3.14 2>&1 || true)" in
    *WARNING*) assert_eq "3.14 says why it is unproven" "yes" "yes" ;;
    *) assert_eq "3.14 says why it is unproven" "yes" "no" ;;
esac

# 3.15+ is not "unproven", it is impossible: pyproject.toml declares
# requires-python = ">=3.9,<3.15", so uv refuses the unsloth package itself there
# whatever wheels exist. Warning and continuing recreated the late resolver failure
# this gate was added to replace.
assert_eq "3.15 is rejected, not merely warned about" \
    "rejected" "$(run_python_gate 3.15)"
assert_eq "3.16 is rejected" \
    "rejected" "$(run_python_gate 3.16)"
assert_eq "4.0 is rejected" \
    "rejected" "$(run_python_gate 4.0)"
case "$(sh "$_GATE" 3.15 2>&1 || true)" in
    *requires-python*) assert_eq "the 3.15 rejection names requires-python" "yes" "yes" ;;
    *) assert_eq "the 3.15 rejection names requires-python" "yes" "no" ;;
esac
# The ceiling the message quotes has to be the one pyproject.toml actually declares.
case "$(grep -m1 '^requires-python' "$SCRIPT_DIR/../../pyproject.toml")" in
    *'<3.15'*) assert_eq "pyproject still pins the ceiling this gate enforces" "yes" "yes" ;;
    *) assert_eq "pyproject still pins the ceiling this gate enforces" "yes" "no" ;;
esac

rm -f "$_GATE"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
