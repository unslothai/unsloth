#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the shell-rc PATH append in install.sh (_persist_local_bin_on_path).
#
# History: the append was three unguarded `echo ... >> "$_SHELL_PROFILE"` lines. With
# `set -e` at the top of install.sh, an rc file that could not be written aborted the
# installer at its very last cosmetic step -- after the venv, llama.cpp and the shim were
# already in place -- so the user saw "Unsloth Studio Installed" followed by a hard
# failure, and the Tauri/AppImage path reported the whole install as failed. Immutable
# and managed homes hit this every time: NixOS / home-manager symlink ~/.bashrc into the
# read-only Nix store, and chezmoi / stow / yadm do the same.
#
# The contract now:
#   * a writable rc gets the export appended and a normal "path" step;
#   * an rc that already mentions .local/bin is left alone (idempotent re-runs);
#   * an empty rc path is a no-op (no shell rc was found);
#   * a NON-WRITABLE rc warns, prints the line to add by hand, and still returns 0 so
#     `set -e` cannot kill a successful install;
#   * the three lines go in as ONE redirect, so a partial write cannot leave a dangling
#     "# Added by Unsloth installer" comment with no export under it.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        echo "  ---- output ----"; echo "$_haystack" | sed 's/^/  | /'
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        echo "  ---- output ----"; echo "$_haystack" | sed 's/^/  | /'
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

# ── Extract the function under test ──
_FN_FILE=$(mktemp)
sed -n '/^_persist_local_bin_on_path()/,/^}/p' "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q '_persist_local_bin_on_path()' "$_FN_FILE"; then
    echo "FAIL: could not extract _persist_local_bin_on_path from install.sh"
    echo "      (it must stay a top-level function so this test can reach it)"
    exit 1
fi

_HARNESS=$(mktemp)
cat > "$_HARNESS" <<'HARNESS'
C_WARN=''; C_ERR=''; C_OK=''; C_DIM=''; C_RST=''
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
HARNESS

_SH="${BASH:-/bin/bash}"

# `set -e` inside the runner mirrors install.sh: if the function returns non-zero the
# runner dies before echoing RC, which is exactly the regression being guarded.
_run() {  # $1 = rc path
    ( "$_SH" -c "set -e; . '$_HARNESS'; . '$_FN_FILE'; _persist_local_bin_on_path '$1'; echo \"RC=\$?\"" 2>&1 )
}

_TMP=$(mktemp -d)
trap 'rm -rf "$_TMP" "$_FN_FILE" "$_HARNESS"' EXIT

echo "=== writable rc: appends the export ==="
_rc="$_TMP/.bashrc"; : > "$_rc"
_out=$(_run "$_rc")
assert_contains "reports the append" "$_out" "STEP path added ~/.local/bin to PATH"
assert_contains "returns 0"          "$_out" "RC=0"
assert_contains "export written"     "$(cat "$_rc")" 'export PATH="$HOME/.local/bin:$PATH"'
assert_contains "marker written"     "$(cat "$_rc")" '# Added by Unsloth installer'

echo "=== already mentions .local/bin: left untouched (idempotent) ==="
_rc2="$_TMP/.bashrc_existing"
printf 'export PATH="$HOME/.local/bin:$PATH"\n' > "$_rc2"
_before=$(cat "$_rc2")
_out=$(_run "$_rc2")
assert_contains     "returns 0"        "$_out" "RC=0"
assert_not_contains "no second append" "$_out" "added ~/.local/bin"
if [ "$(cat "$_rc2")" = "$_before" ]; then
    echo "  PASS: file unchanged"; PASS=$((PASS + 1))
else
    echo "  FAIL: file changed on a re-run"; FAIL=$((FAIL + 1))
fi

echo "=== empty rc path (no shell rc found): no-op ==="
_out=$(_run "")
assert_contains     "returns 0"  "$_out" "RC=0"
assert_not_contains "no step"    "$_out" "STEP path"

echo "=== READ-ONLY rc: warns, does not fail the install ==="
# The NixOS / home-manager / chezmoi shape. Root ignores the write bit, so skip there
# rather than reporting a false pass.
_rc3="$_TMP/.bashrc_ro"; : > "$_rc3"; chmod 0444 "$_rc3"
if [ "$(id -u)" = "0" ] || { echo x >> "$_rc3"; } 2>/dev/null; then
    echo "  SKIP: cannot make a file unwritable as this user (running as root?)"
    : > "$_rc3"
else
    _out=$(_run "$_rc3")
    assert_contains     "returns 0 (set -e survives)" "$_out" "RC=0"
    assert_contains     "warns about the rc"          "$_out" "could not write"
    assert_contains     "prints the manual line"      "$_out" 'export PATH="$HOME/.local/bin:$PATH"'
    assert_contains     "reassures install is fine"   "$_out" "only the PATH line is missing"
    assert_not_contains "does not claim success"      "$_out" "added ~/.local/bin to PATH in"
    if [ ! -s "$_rc3" ]; then
        echo "  PASS: read-only file left empty (no partial write)"; PASS=$((PASS + 1))
    else
        echo "  FAIL: read-only file was modified"; FAIL=$((FAIL + 1))
    fi
fi
chmod 0644 "$_rc3" 2>/dev/null || true

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
