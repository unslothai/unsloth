#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests for the torch-trio --overrides guard on the Step-2 unsloth installs in
# install.sh. A released unsloth wheel can pin an older torch (2026.7.2 declares
# torch<2.11.0); without the overrides file a with-deps PyPI resolve downgrades
# the trio Step 1 installed, and the flavor guard misses it (PyPI's torch 2.10
# default is itself cu128-flavored). Same assertion pattern as test_torch_constraint.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_true() {
    _label="$1"; _ok="$2"
    if [ "$_ok" = "0" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== test_unsloth_torch_override ==="

# 1. Every with-deps unsloth install carries the overrides expansion (local,
#    generic, migrated); the --no-deps no-torch paths need no guard.
_local_block=$(grep -A2 '"install unsloth (local)"' "$INSTALL_SH")
printf '%s' "$_local_block" | grep -q -- '--overrides "\$_UNSLOTH_TORCH_OVERRIDES"'
assert_true "local (with-deps) unsloth install passes --overrides" "$?"

_generic_block=$(grep -A2 '"install unsloth" uv pip install' "$INSTALL_SH")
printf '%s' "$_generic_block" | grep -q -- '--overrides "\$_UNSLOTH_TORCH_OVERRIDES"'
assert_true "generic (with-deps) unsloth install passes --overrides" "$?"

_migrated_block=$(grep -A3 '"install unsloth (migrated)"' "$INSTALL_SH")
printf '%s' "$_migrated_block" | grep -q -- '--overrides "\$_UNSLOTH_TORCH_OVERRIDES"'
assert_true "migrated (with-deps) unsloth install passes --overrides" "$?"

_no_torch_block=$(grep -A2 '"install unsloth (no-torch)"' "$INSTALL_SH")
if printf '%s' "$_no_torch_block" | grep -q -- '--overrides'; then _rc=1; else _rc=0; fi
assert_true "no-torch (--no-deps) unsloth install has no overrides" "$_rc"

_migrated_nt_block=$(grep -A2 '"install unsloth (migrated no-torch)"' "$INSTALL_SH")
if printf '%s' "$_migrated_nt_block" | grep -q -- '--overrides'; then _rc=1; else _rc=0; fi
assert_true "migrated no-torch (--no-deps) unsloth install has no overrides" "$_rc"

# 2. The overrides file is only built when SKIP_TORCH=false.
grep -B2 '_torch_trio_pins=\$(' "$INSTALL_SH" | grep -q 'SKIP_TORCH" = false'
assert_true "overrides file build is gated on SKIP_TORCH=false" "$?"

# 3. The pin-collection snippet emits exact ==pins for the installed trio (run
#    the embedded python against this test's interpreter).
_snippet=$(sed -n '/_torch_trio_pins=\$("\$_VENV_PY" -c "/,/^" 2>\/dev\/null)/p' "$INSTALL_SH" \
    | sed '1s/.*-c "//' | sed '$d')
_out=$(python3 -c "$_snippet" 2>&1) || true
# torch may or may not be importable on the test host; the snippet must not
# crash and every line it does emit must be an exact pkg==version pin.
if [ -n "$_out" ]; then
    printf '%s\n' "$_out" | grep -vqE '^(torch|torchvision|torchaudio)==.+$' && _rc=1 || _rc=0
else
    _rc=0
fi
assert_true "pin snippet emits only exact trio ==pins (or nothing)" "$_rc"

# 4. The temp overrides file is cleaned up after Step 2.
grep -q 'rm -f "\$_UNSLOTH_TORCH_OVERRIDES"' "$INSTALL_SH"
assert_true "overrides temp file is removed after the unsloth installs" "$?"

# 5. Any UV_OVERRIDE env file is folded in (the CLI --overrides flag would
#    otherwise replace it, dropping e.g. the macOS arm64 darwin overrides).
grep -q 'for _ov_file in \${UV_OVERRIDE:-}' "$INSTALL_SH"
assert_true "UV_OVERRIDE env files are merged into the overrides file" "$?"

# 6. Exit and signal traps share cleanup, so a failed or interrupted Step 2
#    cannot leak the overrides file.
sed -n '/_on_install_exit() {/,/^}/p' "$INSTALL_SH" | grep -q '_cleanup_install_temporaries'
_exit_cleanup_rc=$?
sed -n '/_on_install_signal() {/,/^}/p' "$INSTALL_SH" | grep -q '_cleanup_install_temporaries'
_signal_cleanup_rc=$?
sed -n '/_cleanup_install_temporaries() {/,/^}/p' "$INSTALL_SH" \
    | grep -q 'rm -f "\$_UNSLOTH_TORCH_OVERRIDES"'
_cleanup_body_rc=$?
if [ "$_exit_cleanup_rc" -eq 0 ] && [ "$_signal_cleanup_rc" -eq 0 ] \
   && [ "$_cleanup_body_rc" -eq 0 ]; then
    _rc=0
else
    _rc=1
fi
assert_true "exit and signal traps remove the overrides temp file" "$_rc"

# 7. The UV_OVERRIDE fold filters inherited files instead of cat-ing them (run
#    the extracted awk program on sample files): (a) inherited torch-trio lines
#    are dropped so the generated exact pins win (uv intersects duplicates);
#    (b) every line is newline-terminated so an unterminated file cannot join
#    two requirements into one.
_awk_prog=$(sed -n "s/.*awk '\(.*\)' \"\$_ov_file\".*/\1/p" "$INSTALL_SH")
[ -n "$_awk_prog" ]
assert_true "UV_OVERRIDE fold uses the trio-filtering awk program" "$?"

_ov_dir=$(mktemp -d)
printf '%s' 'transformers>=4.57.6' > "$_ov_dir/ov1.txt" # no trailing newline
cat > "$_ov_dir/ov2.txt" <<'EOF'
# comment survives
torch<2.11.0
torchvision==0.25.0
torchaudio!=2.11.0
torchmetrics==1.0
anyio<4.14.0
EOF
_merged="$_ov_dir/merged.txt"
printf '%s\n' 'torch==2.11.0+cu128' > "$_merged"
for _f in "$_ov_dir/ov1.txt" "$_ov_dir/ov2.txt"; do
    awk "$_awk_prog" "$_f" >> "$_merged"
done

grep -qx 'transformers>=4.57.6' "$_merged"
assert_true "no-trailing-newline override stays a separate requirement line" "$?"

if grep -qx 'torchmetrics==1.0' "$_merged" && grep -qx 'anyio<4.14.0' "$_merged"; then
    _rc=0
else
    _rc=1
fi
assert_true "unrelated inherited overrides are preserved" "$_rc"

if grep -qE '^(torch|torchvision|torchaudio)([[:space:]<>=!~;@[]|$)' "$_merged" \
    && [ "$(grep -cE '^(torch|torchvision|torchaudio)([[:space:]<>=!~;@[]|$)' "$_merged")" != "1" ]; then
    _rc=1
else
    _rc=0
fi
grep -qx 'torch==2.11.0+cu128' "$_merged" || _rc=1
assert_true "inherited torch-trio lines are dropped; generated pin wins" "$_rc"
rm -rf "$_ov_dir"

# 5. The beside-the-caller override file must not be world-readable. The merge
#    above copies every inherited non-torch requirement into it, and a direct URL
#    requirement can carry credentials, so a caller's umask 022 would expose them
#    for the length of a torch install. The mktemp fallback is already 0600.
_creation=$(sed -n '/_UNSLOTH_TORCH_OVERRIDES="$_ov_dir\/.unsloth-torch-overrides/,/^            fi$/p' "$INSTALL_SH")
# `if` rather than a bare pipeline: this file runs under `set -e`, so a failing
# grep would abort the suite instead of reporting one FAIL.
if printf '%s' "$_creation" | grep -q 'umask 077'; then _rc=0; else _rc=1; fi
assert_true "the adjacent override file is created under umask 077" "$_rc"

if printf '%s' "$_creation" | grep -q 'chmod 600'; then _rc=0; else _rc=1; fi
assert_true "and chmod'd too, since \`: >\` truncates without changing an existing mode" "$_rc"

# Drive the real construct. A stale file from a recycled PID is the case the
# umask alone cannot fix, which is why both halves are needed.
_mode_dir=$(mktemp -d)
(
    umask 022
    _f="$_mode_dir/.unsloth-torch-overrides.$$.txt"
    if (umask 077; : > "$_f") 2>/dev/null; then chmod 600 "$_f" 2>/dev/null || true; fi
    stat -c %a "$_f" > "$_mode_dir/fresh"

    _s="$_mode_dir/stale.txt"
    : > "$_s"; chmod 644 "$_s"
    if (umask 077; : > "$_s") 2>/dev/null; then
        stat -c %a "$_s" > "$_mode_dir/stale_umask_only"
        chmod 600 "$_s" 2>/dev/null || true
    fi
    stat -c %a "$_s" > "$_mode_dir/stale_both"
)
if [ "$(cat "$_mode_dir/fresh")" = "600" ]; then _rc=0; else _rc=1; fi
assert_true "a freshly created override file is 0600 under umask 022" "$_rc"
if [ "$(cat "$_mode_dir/stale_umask_only")" = "644" ]; then _rc=0; else _rc=1; fi
assert_true "umask alone leaves a stale file 0644, so the chmod is load-bearing" "$_rc"
if [ "$(cat "$_mode_dir/stale_both")" = "600" ]; then _rc=0; else _rc=1; fi
assert_true "umask plus chmod brings a stale file back to 0600" "$_rc"
rm -rf "$_mode_dir"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
