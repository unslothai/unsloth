#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Exercises install.sh's real rollback helpers without downloading the Studio stack.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
PASS=0
FAIL=0

ok()  { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

ROLLBACK_BLOCK=$(sed -n '/^_VENV_ROLLBACK_DIR=""/,/^trap '\''_on_install_signal 143'\'' TERM$/p' "$INSTALL_SH")
if ! printf '%s\n' "$ROLLBACK_BLOCK" | grep -q '^_on_install_signal() {'; then
    echo "  FAIL: could not extract rollback lifecycle block from install.sh"
    exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run_signal_case() {
    _signal="$1"
    _expected_status="$2"
    _case_dir="$WORK/signal-$_signal"
    mkdir -p "$_case_dir/unsloth_studio"
    printf 'old\n' > "$_case_dir/unsloth_studio/generation"
    _harness="$_case_dir/harness.sh"
    {
        printf '%s\n' 'set -e'
        printf '%s\n' 'substep() { :; }'
        printf '%s\n' 'rollback_substep() { substep "$@"; }'
        printf '%s\n' 'C_WARN=""'
        printf "STUDIO_HOME='%s'\n" "$_case_dir"
        printf "VENV_DIR='%s/unsloth_studio'\n" "$_case_dir"
        printf '%s\n' "$ROLLBACK_BLOCK"
        printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
        printf '%s\n' 'mkdir -p "$VENV_DIR"'
        printf '%s\n' 'printf "partial\n" > "$VENV_DIR/generation"'
        printf 'kill -%s $$\n' "$_signal"
        printf '%s\n' 'exit 99'
    } > "$_harness"

    set +e
    dash "$_harness" >/dev/null 2>&1
    _status=$?
    set -e
    if [ "$_status" = "$_expected_status" ]; then
        ok "dash $_signal exits with $_expected_status"
    else
        bad "dash $_signal exits with $_expected_status (got $_status)"
    fi
    if [ "$(cat "$_case_dir/unsloth_studio/generation" 2>/dev/null)" = "old" ]; then
        ok "dash $_signal restores the previous environment"
    else
        bad "dash $_signal did not restore the previous environment"
    fi
    if ! find "$_case_dir" -maxdepth 1 -name 'unsloth_studio.rollback.*' -print -quit | grep -q .; then
        ok "dash $_signal leaves no rollback copy"
    else
        bad "dash $_signal left a rollback copy"
    fi
}

echo "=== install.sh signal rollback ==="
run_signal_case INT 130
run_signal_case TERM 143
run_signal_case HUP 129

echo "=== install.sh transition boundaries ==="
START_BOUNDARY_DIR="$WORK/start-boundary"
mkdir -p "$START_BOUNDARY_DIR/unsloth_studio"
printf 'old\n' > "$START_BOUNDARY_DIR/unsloth_studio/generation"
START_BOUNDARY_HARNESS="$START_BOUNDARY_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'rollback_substep() { substep "$@"; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$START_BOUNDARY_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$START_BOUNDARY_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' 'mv() { command mv "$@"; kill -TERM $$; }'
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
} > "$START_BOUNDARY_HARNESS"
set +e
dash "$START_BOUNDARY_HARNESS" >/dev/null 2>&1
_start_boundary_status=$?
set -e
if [ "$_start_boundary_status" -eq 143 ] \
   && [ "$(cat "$START_BOUNDARY_DIR/unsloth_studio/generation" 2>/dev/null)" = "old" ]; then
    ok "signal immediately after rollback rename restores the old environment"
else
    bad "rollback state was not published before rename"
fi

COMMIT_BOUNDARY_DIR="$WORK/commit-boundary"
mkdir -p "$COMMIT_BOUNDARY_DIR/unsloth_studio"
printf 'old\n' > "$COMMIT_BOUNDARY_DIR/unsloth_studio/generation"
COMMIT_BOUNDARY_HARNESS="$COMMIT_BOUNDARY_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'rollback_substep() { substep "$@"; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$COMMIT_BOUNDARY_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$COMMIT_BOUNDARY_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
    printf '%s\n' 'mkdir -p "$VENV_DIR"'
    printf '%s\n' 'printf "new\n" > "$VENV_DIR/generation"'
    printf '%s\n' 'rm() { kill -TERM $$; }'
    printf '%s\n' '_commit_studio_venv_replacement'
} > "$COMMIT_BOUNDARY_HARNESS"
set +e
dash "$COMMIT_BOUNDARY_HARNESS" >/dev/null 2>&1
_commit_boundary_status=$?
set -e
if [ "$_commit_boundary_status" -eq 143 ] \
   && [ "$(cat "$COMMIT_BOUNDARY_DIR/unsloth_studio/generation" 2>/dev/null)" = "new" ]; then
    ok "signal during committed-backup deletion keeps the new environment"
else
    bad "signal during committed-backup deletion restored a partial backup"
fi

echo "=== install.sh successful cleanup ==="
PRUNE_DIR="$WORK/prune"
mkdir -p "$PRUNE_DIR/unsloth_studio"
printf 'old\n' > "$PRUNE_DIR/unsloth_studio/generation"
PRUNE_HARNESS="$PRUNE_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'rollback_substep() { substep "$@"; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$PRUNE_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$PRUNE_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
    printf '%s\n' 'mkdir -p "$VENV_DIR"'
    printf '%s\n' 'printf "new\n" > "$VENV_DIR/generation"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.20000101000000.999999999"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.20000101000001.$$"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.user-data"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/outside"'
    printf '%s\n' 'ln -s "$STUDIO_HOME/outside" "$STUDIO_HOME/unsloth_studio.rollback.20000101000002.999999998"'
    printf '%s\n' '_commit_studio_venv_replacement'
} > "$PRUNE_HARNESS"

sh "$PRUNE_HARNESS" >/dev/null 2>&1
if [ "$(cat "$PRUNE_DIR/unsloth_studio/generation" 2>/dev/null)" = "new" ]; then
    ok "successful replacement keeps the new environment"
else
    bad "successful replacement lost the new environment"
fi
if [ ! -d "$PRUNE_DIR/unsloth_studio.rollback.20000101000000.999999999" ]; then
    ok "successful install removes an orphan from a dead PID"
else
    bad "successful install left an orphan from a dead PID"
fi
_active_count=$(find "$PRUNE_DIR" -maxdepth 1 -type d -name 'unsloth_studio.rollback.20000101000001.*' | wc -l)
if [ "$_active_count" -eq 1 ]; then
    ok "successful install preserves a concurrent installer's rollback"
else
    bad "successful install removed a concurrent installer's rollback"
fi
if [ -d "$PRUNE_DIR/unsloth_studio.rollback.user-data" ]; then
    ok "stale cleanup preserves names outside the generated format"
else
    bad "stale cleanup removed a non-generated rollback name"
fi
if [ -L "$PRUNE_DIR/unsloth_studio.rollback.20000101000002.999999998" ] \
   && [ -d "$PRUNE_DIR/outside" ]; then
    ok "stale cleanup does not follow rollback symlinks"
else
    bad "stale cleanup mutated a rollback symlink target"
fi

echo "=== install.sh commits before the post-setup tail ==="
# The environment is final once studio setup returns, so nothing in the wiring below it may
# reach the exit trap that restores the previous environment.
_commit_calls=$(grep -c '^[[:space:]]*_commit_studio_venv_replacement$' "$INSTALL_SH")
_commit_at=$(grep -n '^[[:space:]]*_commit_studio_venv_replacement$' "$INSTALL_SH" | head -1 | cut -d: -f1)
_setup_gate_at=$(grep -n '^if \[ "\$_SETUP_EXIT" -eq 0 \]; then$' "$INSTALL_SH" | head -1 | cut -d: -f1)
# The first thing install.sh mutates outside the venv once setup has returned.
_shim_at=$(grep -n '^mkdir -p "\$_LOCAL_BIN"$' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ "$_commit_calls" -eq 1 ] && [ -n "$_commit_at" ] && [ -n "$_setup_gate_at" ] && [ -n "$_shim_at" ] \
   && [ "$_setup_gate_at" -lt "$_commit_at" ] && [ "$_commit_at" -lt "$_shim_at" ]; then
    ok "the replacement is committed inside the setup-succeeded gate, before anything is wired"
else
    bad "the replacement is committed too late (gate=$_setup_gate_at commit=$_commit_at shim=$_shim_at calls=$_commit_calls)"
fi
# ...and first in that gate: anything added ahead of it is one more command inside the window.
_gate_first=$(sed -n "$((_setup_gate_at + 1)),\$p" "$INSTALL_SH" \
    | grep -vE '^[[:space:]]*(#|$)' | head -1 | sed 's/^[[:space:]]*//')
if [ "$_gate_first" = "_commit_studio_venv_replacement" ]; then
    ok "nothing runs between studio setup succeeding and the commit"
else
    bad "the setup-succeeded gate runs something before the commit ($_gate_first)"
fi

# install.sh's own tail, gate to gate, so the cases below run its real commit call site. Both
# anchors are checked: without the closing one sed would run to EOF and carry unrelated code.
_tail_ends=$(grep -c '^if \[ "\$_SETUP_EXIT" -ne 0 \]; then$' "$INSTALL_SH")
TAIL_BLOCK=$(sed -n '/^if \[ "\$_SETUP_EXIT" -eq 0 \]; then$/,/^if \[ "\$_SETUP_EXIT" -ne 0 \]; then$/p' "$INSTALL_SH" \
    | sed '$d')
if [ "$_tail_ends" -ne 1 ] \
   || ! printf '%s\n' "$TAIL_BLOCK" | grep -q '^_persist_login_path_dir() {'; then
    echo "  FAIL: could not extract the post-setup tail from install.sh"
    exit 1
fi

# The state the tail inherits: a replacement in flight and a new environment on disk.
write_tail_harness() {  # case dir, login shell, "no-exe" to leave the shim's target absent
    {
        printf '%s\n' 'set -e'
        printf '%s\n' 'substep() { printf "%s\n" "$1" >> "$STUDIO_HOME/steps.log"; }'
        printf '%s\n' 'rollback_substep() { substep "$@"; }'
        printf '%s\n' 'step() { printf "%s\n" "$2" >> "$STUDIO_HOME/steps.log"; }'
        printf '%s\n' 'tauri_clear_install_error() { :; }'
        # The tail ends with this call; writing a launcher is not what these cases are about.
        printf '%s\n' 'create_studio_shortcuts() { return 0; }'
        printf '%s\n' 'TAURI_MODE=false'
        printf '%s\n' 'OS=linux'
        printf '%s\n' 'C_WARN=""'
        printf "STUDIO_HOME='%s'\n" "$1"
        printf "VENV_DIR='%s/unsloth_studio'\n" "$1"
        printf '%s\n' "$ROLLBACK_BLOCK"
        printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
        printf '%s\n' 'mkdir -p "$VENV_DIR/bin"'
        if [ "${3:-}" != no-exe ]; then
            printf '%s\n' 'printf "#!/bin/sh\\n" > "$VENV_DIR/bin/unsloth"'
            printf '%s\n' 'chmod +x "$VENV_DIR/bin/unsloth"'
        fi
        printf '%s\n' 'VENV_ABS_BIN="$VENV_DIR/bin"'
        printf '%s\n' 'printf "new\n" > "$VENV_DIR/generation"'
        printf '%s\n' '_SETUP_EXIT=0'
        printf "HOME='%s/home'\n" "$1"
        printf '%s\n' 'export HOME'
        printf "SHELL='%s'\n" "$2"
        printf '%s\n' 'export SHELL'
        # The tail reads all four from the environment; an exported ZDOTDIR would send the
        # fixture's write to the developer's own ~/.zshrc.
        printf '%s\n' 'unset ZDOTDIR ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL'
        printf '%s\n' '_LOCAL_BIN="$HOME/.local/bin"'
        printf '%s\n' '_STUDIO_HOME_REDIRECT=default'
        printf '%s\n' '_UNSLOTH_LOGIN_PATH="/usr/bin:/bin"'
        printf '%s\n' '_UNSLOTH_UV_BIN_DIR="$HOME/.local/uvbin"'
        printf '%s\n' "$TAIL_BLOCK"
    } > "$1/harness.sh"
}

# An unwritable profile costs the PATH entry it would have added, not the install.
run_readonly_profile_case() {
    _case="$1"
    _case_dir="$WORK/readonly-$_case"
    _case_home="$_case_dir/home"
    mkdir -p "$_case_dir/unsloth_studio" "$_case_home/.local/bin" \
        "$_case_home/$(dirname "$3")"
    printf 'old\n' > "$_case_dir/unsloth_studio/generation"
    printf '# unwritable\n' > "$_case_home/$3"
    chmod 444 "$_case_home/$3"
    if true 2>/dev/null >> "$_case_home/$3"; then
        bad "unwritable $_case profile could not be set up (this user appends to it regardless)"
        return 0
    fi
    write_tail_harness "$_case_dir" "$2"

    set +e
    sh "$_case_dir/harness.sh" >/dev/null 2>&1
    _status=$?
    set -e
    if [ "$_status" -eq 0 ]; then
        ok "an unwritable $_case profile does not fail the install"
    else
        bad "an unwritable $_case profile failed the install (exit $_status)"
    fi
    if [ "$(cat "$_case_home/$3")" = "# unwritable" ]; then
        ok "an unwritable $_case profile is left as it was"
    else
        bad "an unwritable $_case profile was modified"
    fi
    if grep -q "could not write $_case_home/$3; add ~/.local/bin" \
        "$_case_dir/steps.log" 2>/dev/null; then
        ok "an unwritable $_case profile is reported to the user"
    else
        bad "an unwritable $_case profile is silently skipped"
    fi
    # One unwritable profile must not cost the other five the uv loop writes.
    if grep -q '/.local/uvbin' "$_case_home/.profile" 2>/dev/null; then
        ok "the profiles that can be written still get their PATH entry ($_case)"
    else
        bad "one unwritable $_case profile stopped the remaining profiles from being written"
    fi
}

run_readonly_profile_case zsh /bin/zsh .zshrc
run_readonly_profile_case fish /usr/bin/fish .config/fish/conf.d/unsloth.fish

# The tail can still refuse outright: a real directory at the shim path is user data it will not
# delete. That refusal must cost the shim, not the environment.
REFUSE_DIR="$WORK/tail-refusal"
mkdir -p "$REFUSE_DIR/unsloth_studio" "$REFUSE_DIR/home/.local/bin/unsloth"
printf 'old\n' > "$REFUSE_DIR/unsloth_studio/generation"
write_tail_harness "$REFUSE_DIR" /bin/bash
set +e
sh "$REFUSE_DIR/harness.sh" >/dev/null 2>&1
_refusal_status=$?
set -e
if [ "$_refusal_status" -eq 1 ]; then
    ok "a directory at the shim path still refuses the install"
else
    bad "a directory at the shim path no longer refuses the install (exit $_refusal_status)"
fi
if [ "$(cat "$REFUSE_DIR/unsloth_studio/generation" 2>/dev/null)" = "new" ]; then
    ok "a refused shim keeps the environment just installed"
else
    bad "a refused shim rolled back the environment just installed"
fi
if ! find "$REFUSE_DIR" -maxdepth 1 -name 'unsloth_studio.rollback.*' -print -quit | grep -q .; then
    ok "a refused shim leaves no rollback copy"
else
    bad "a refused shim left a rollback copy"
fi

# An unwritable bin directory is only a failed install when what it holds is not this run's shim.
run_readonly_bin_case() {  # name, what the existing entry points at, expected status, [no-exe]
    _bin_dir="$WORK/readonly-bin-$1"
    _bin_home="$_bin_dir/home"
    mkdir -p "$_bin_dir/unsloth_studio" "$_bin_home/.local/bin"
    printf 'old\n' > "$_bin_dir/unsloth_studio/generation"
    # The harness writes the executable; the entry already there either resolves to it or not.
    ln -sfn "$2" "$_bin_home/.local/bin/unsloth"
    chmod 555 "$_bin_home/.local/bin"
    if true 2>/dev/null > "$_bin_home/.local/bin/probe"; then
        rm -f "$_bin_home/.local/bin/probe"
        chmod 755 "$_bin_home/.local/bin"
        bad "unwritable bin directory holding $1 could not be set up (this user writes it anyway)"
        return 0
    fi
    write_tail_harness "$_bin_dir" /bin/bash "${4:-}"
    set +e
    sh "$_bin_dir/harness.sh" >/dev/null 2>"$_bin_dir/stderr"
    _bin_status=$?
    set -e
    chmod 755 "$_bin_home/.local/bin"
    if [ "$_bin_status" -eq "$3" ]; then
        ok "an unwritable bin directory holding $1 exits $3"
    else
        bad "an unwritable bin directory holding $1 exits $3 (got $_bin_status)"
    fi
    if [ "$3" -eq 0 ]; then
        if grep -q "kept the existing shim" "$_bin_dir/steps.log" 2>/dev/null; then
            ok "keeping the existing shim is reported rather than passed over in silence"
        else
            bad "keeping the existing shim is not reported"
        fi
    elif grep -qF "run '$_bin_dir/unsloth_studio/bin/unsloth' directly" "$_bin_dir/stderr"; then
        ok "refusing $1 says how to start Unsloth without the shim"
    else
        bad "refusing $1 does not say how to start Unsloth without the shim"
    fi
}

# Absolute as install.sh writes it, relative as something else might: both resolve to it.
run_readonly_bin_case absolute-shim "$WORK/readonly-bin-absolute-shim/unsloth_studio/bin/unsloth" 0
run_readonly_bin_case relative-shim ../../../unsloth_studio/bin/unsloth 0
# One resolving elsewhere, one naming the exact path install.sh writes but resolving nowhere.
run_readonly_bin_case another-command /bin/false 1
run_readonly_bin_case dangling-shim \
    "$WORK/readonly-bin-dangling-shim/unsloth_studio/bin/unsloth" 1 no-exe

echo "=== install.ps1 rollback wiring ==="
if grep -q '^    function Remove-StaleStudioVenvRollbacks {' "$INSTALL_PS1" \
   && grep -q '^    Remove-StaleStudioVenvRollbacks$' "$INSTALL_PS1"; then
    ok "Windows installer prunes stale rollbacks after success"
else
    bad "Windows installer does not wire stale rollback cleanup"
fi
if grep -q '^    } finally {$' "$INSTALL_PS1" \
   && grep -A3 '^    } finally {$' "$INSTALL_PS1" | grep -q 'Restore-StudioVenvRollback'; then
    ok "Windows replacement is protected by finally"
else
    bad "Windows replacement lacks finally rollback"
fi
if grep -A18 '^    function Remove-StudioVenvTreeWithRetry {' "$INSTALL_PS1" \
   | grep -q 'ErrorAction Stop'; then
    ok "Windows rollback deletion failures are observable and retried"
else
    bad "Windows rollback deletion still hides failures"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "ALL PASSED"
