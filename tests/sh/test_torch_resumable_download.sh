#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Guards install.sh resumable torch wheel helpers (#8456).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _want="$2"; _got="$3"
    if [ "$_want" = "$_got" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (want '$_want', got '$_got')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

_FN_FILE=$(mktemp)
sed -n '/^download_resumable()/,/^}/p' "$INSTALL_SH" > "$_FN_FILE"
sed -n '/^_constraint_ver_prefix()/,/^}/p' "$INSTALL_SH" >> "$_FN_FILE"
sed -n '/^_pick_simple_index_wheel()/,/^}/p' "$INSTALL_SH" >> "$_FN_FILE"

_HARNESS=$(mktemp)
cat > "$_HARNESS" <<'HARNESS'
_ARCH=x86_64
HARNESS

_BIN=$(mktemp -d)
_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }

echo "=== download_resumable prefers wget -c ==="
rm -f "$_BIN"/*
_mk wget 'echo "wget $@" >> '"$_BIN"'/log"; exit 0'
PATH="$_BIN:$PATH" . "$_HARNESS"
PATH="$_BIN:$PATH" . "$_FN_FILE"
PATH="$_BIN:$PATH" download_resumable "https://example.com/torch.whl" "/tmp/torch.whl"
assert_contains "wget -c used" "$(cat "$_BIN/log")" "wget -c"

echo "=== _constraint_ver_prefix parses torch ranges ==="
. "$_HARNESS"
. "$_FN_FILE"
assert_eq "torch>=2.11.0,<2.12.0" "2.11." "$(_constraint_ver_prefix 'torch>=2.11.0,<2.12.0')"
assert_eq "torchvision==0.26.*" "0.26." "$(_constraint_ver_prefix 'torchvision==0.26.*')"

echo "=== _pick_simple_index_wheel selects newest matching href ==="
_LISTING='<a href="torch-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"></a>
<a href="torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"></a>'
. "$_HARNESS"
. "$_FN_FILE"
assert_eq "newest torch 2.11 wheel" \
    "torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "$(_pick_simple_index_wheel "$_LISTING" "torch" "2.11." "cp312" "x86_64")"
_HREF='torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=abc123'
_NAME=$(printf '%s' "${_HREF##*/}" | sed 's/%2[Bb]/+/g; s/[?#].*//')
assert_eq "wheel basename strips sha256 fragment" \
    "torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "$_NAME"

echo "=== _install_torch_default_index falls back to resumable path ==="
assert_contains "uv failure warns" "$(cat "$INSTALL_SH")" \
    "retrying with resumable wheel downloads (wget -c)"
assert_contains "helper present" "$(cat "$INSTALL_SH")" "_install_torch_resumable_wheels()"

rm -f "$_FN_FILE" "$_HARNESS"
rm -rf "$_BIN"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
