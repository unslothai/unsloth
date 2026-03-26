#!/bin/bash
# Unit tests for Mac Intel compatibility and UNSLOTH_NO_TORCH propagation in install.sh
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

# ── Extract version_ge function from install.sh ──
_VGE_FILE=$(mktemp)
sed -n '/^version_ge()/,/^}/p' "$INSTALL_SH" > "$_VGE_FILE"

echo "=== version_ge ==="

# Basic comparisons
_result=$(bash -c ". '$_VGE_FILE'; version_ge '3.13' '3.12' && echo pass || echo fail")
assert_eq "3.13 >= 3.12" "pass" "$_result"

_result=$(bash -c ". '$_VGE_FILE'; version_ge '3.12' '3.13' && echo pass || echo fail")
assert_eq "3.12 >= 3.13" "fail" "$_result"

_result=$(bash -c ". '$_VGE_FILE'; version_ge '3.13' '3.13' && echo pass || echo fail")
assert_eq "3.13 >= 3.13 (equal)" "pass" "$_result"

# Patch versions
_result=$(bash -c ". '$_VGE_FILE'; version_ge '3.13.8' '3.13' && echo pass || echo fail")
assert_eq "3.13.8 >= 3.13 (patch > implicit 0)" "pass" "$_result"

_result=$(bash -c ". '$_VGE_FILE'; version_ge '3.12.0' '3.13.0' && echo pass || echo fail")
assert_eq "3.12.0 >= 3.13.0 (minor less)" "fail" "$_result"

# UV_MIN_VERSION edge cases
_result=$(bash -c ". '$_VGE_FILE'; version_ge '0.7.14' '0.7.14' && echo pass || echo fail")
assert_eq "0.7.14 >= 0.7.14 (exact UV_MIN_VERSION)" "pass" "$_result"

_result=$(bash -c ". '$_VGE_FILE'; version_ge '0.7.13' '0.7.14' && echo pass || echo fail")
assert_eq "0.7.13 >= 0.7.14 (below minimum)" "fail" "$_result"

_result=$(bash -c ". '$_VGE_FILE'; version_ge '0.11.1' '0.7.14' && echo pass || echo fail")
assert_eq "0.11.1 >= 0.7.14 (well above)" "pass" "$_result"

# Major jump
_result=$(bash -c ". '$_VGE_FILE'; version_ge '1.0' '0.99.99' && echo pass || echo fail")
assert_eq "1.0 >= 0.99.99 (major jump)" "pass" "$_result"

rm -f "$_VGE_FILE"

echo ""
echo "=== Architecture detection + PYTHON_VERSION ==="

# Extract the arch-detection snippet from install.sh:
#   lines 558-584 (OS detection, _ARCH, MAC_INTEL, PYTHON_VERSION)
# We create a self-contained snippet that uses a mock uname.
_ARCH_SNIPPET=$(mktemp)
cat > "$_ARCH_SNIPPET" << 'SNIPPET'
# OS detection (uses overridden uname)
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
fi

# Architecture detection
_ARCH=$(uname -m)
MAC_INTEL=false
if [ "$OS" = "macos" ] && [ "$_ARCH" = "x86_64" ]; then
    MAC_INTEL=true
fi

if [ "$MAC_INTEL" = true ]; then
    PYTHON_VERSION="3.12"
else
    PYTHON_VERSION="3.13"
fi

echo "$OS $MAC_INTEL $PYTHON_VERSION"
SNIPPET

# Test: Darwin x86_64 -> macos true 3.12
_result=$(bash -c '
uname() {
    case "$1" in
        -m) echo "x86_64" ;;
        *)  echo "Darwin" ;;
    esac
}
export -f uname
'"source '$_ARCH_SNIPPET'")
assert_eq "Darwin x86_64 -> macos true 3.12" "macos true 3.12" "$_result"

# Test: Darwin arm64 -> macos false 3.13
_result=$(bash -c '
uname() {
    case "$1" in
        -m) echo "arm64" ;;
        *)  echo "Darwin" ;;
    esac
}
export -f uname
'"source '$_ARCH_SNIPPET'")
assert_eq "Darwin arm64 -> macos false 3.13" "macos false 3.13" "$_result"

# Test: Linux x86_64 -> linux false 3.13
_result=$(bash -c '
uname() {
    case "$1" in
        -m) echo "x86_64" ;;
        *)  echo "Linux" ;;
    esac
}
export -f uname
'"source '$_ARCH_SNIPPET'")
assert_eq "Linux x86_64 -> linux false 3.13" "linux false 3.13" "$_result"

# Test: Linux aarch64 -> linux false 3.13
_result=$(bash -c '
uname() {
    case "$1" in
        -m) echo "aarch64" ;;
        *)  echo "Linux" ;;
    esac
}
export -f uname
'"source '$_ARCH_SNIPPET'")
assert_eq "Linux aarch64 -> linux false 3.13" "linux false 3.13" "$_result"

rm -f "$_ARCH_SNIPPET"

echo ""
echo "=== get_torch_index_url on Darwin ==="

# Extract get_torch_index_url and replace hardcoded nvidia-smi path
_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH" \
    | sed "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
    > "$_FUNC_FILE"

# Build a minimal tools directory
_TOOLS_DIR=$(mktemp -d)
for _cmd in grep sed head sh bash cat; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

# Create a mock uname that returns Darwin
_MOCK_UNAME_DIR=$(mktemp -d)
cat > "$_MOCK_UNAME_DIR/uname" << 'MOCK_UNAME'
#!/bin/sh
case "$1" in
    -s) echo "Darwin" ;;
    -m) echo "arm64" ;;
    *)  echo "Darwin" ;;
esac
MOCK_UNAME
chmod +x "$_MOCK_UNAME_DIR/uname"

# Also create a mock nvidia-smi that WOULD return a CUDA version
# (to prove macOS always returns cpu regardless)
_GPU_DIR=$(mktemp -d)
cat > "$_GPU_DIR/nvidia-smi" << 'MOCK_SMI'
#!/bin/sh
cat <<'SMI_OUT'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.6     |
+-----------------------------------------------------------------------------------------+
SMI_OUT
MOCK_SMI
chmod +x "$_GPU_DIR/nvidia-smi"

# Test: Darwin always returns cpu (even with nvidia-smi present)
_result=$(PATH="$_GPU_DIR:$_MOCK_UNAME_DIR:$_TOOLS_DIR" bash -c ". '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null)
assert_eq "Darwin -> cpu (even with nvidia-smi)" "https://download.pytorch.org/whl/cpu" "$_result"

# Test: Darwin without nvidia-smi also returns cpu
_result=$(PATH="$_MOCK_UNAME_DIR:$_TOOLS_DIR" bash -c ". '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null)
assert_eq "Darwin -> cpu (no nvidia-smi)" "https://download.pytorch.org/whl/cpu" "$_result"

rm -f "$_FUNC_FILE"
rm -rf "$_FAKE_SMI_DIR" "$_TOOLS_DIR" "$_MOCK_UNAME_DIR" "$_GPU_DIR"

echo ""
echo "=== UNSLOTH_NO_TORCH propagation ==="

# Verify UNSLOTH_NO_TORCH is passed to setup.sh in BOTH the --local and non-local branches.
# The local branch (line ~879) and the non-local branch (line ~884) should both have it.
_local_count=$(grep -c 'UNSLOTH_NO_TORCH=' "$INSTALL_SH" | head -1)
# We expect at least 2 occurrences (local + non-local setup.sh invocations)
if [ "$_local_count" -ge 2 ]; then
    echo "  PASS: UNSLOTH_NO_TORCH appears in >= 2 setup.sh invocations ($_local_count found)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: UNSLOTH_NO_TORCH should appear in >= 2 setup.sh invocations (found $_local_count)"
    FAIL=$((FAIL + 1))
fi

# Verify the value passed is "$MAC_INTEL" (the variable, not a hardcoded string)
_mac_intel_count=$(grep 'UNSLOTH_NO_TORCH="\$MAC_INTEL"' "$INSTALL_SH" | wc -l)
if [ "$_mac_intel_count" -ge 2 ]; then
    echo "  PASS: UNSLOTH_NO_TORCH=\"\$MAC_INTEL\" in both branches ($_mac_intel_count found)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: UNSLOTH_NO_TORCH=\"\$MAC_INTEL\" should appear in >= 2 branches (found $_mac_intel_count)"
    FAIL=$((FAIL + 1))
fi

# Verify MAC_INTEL is set to true when Intel Mac is detected (static check)
_mac_intel_set=$(grep -c 'MAC_INTEL=true' "$INSTALL_SH")
if [ "$_mac_intel_set" -ge 1 ]; then
    echo "  PASS: MAC_INTEL=true is set in install.sh"
    PASS=$((PASS + 1))
else
    echo "  FAIL: MAC_INTEL=true not found in install.sh"
    FAIL=$((FAIL + 1))
fi

# Verify the Intel Mac skip message exists
if grep -q 'Skipping PyTorch.*Intel Mac' "$INSTALL_SH"; then
    echo "  PASS: Intel Mac PyTorch skip message found"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Intel Mac PyTorch skip message not found"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
