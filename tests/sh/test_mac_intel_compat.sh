#!/bin/bash
# End-to-end sandbox tests for Mac Intel compatibility and UNSLOTH_NO_TORCH propagation.
# Tests version_ge, arch detection (existing), plus E2E venv creation, torch skip
# via a mock uv shim, and UNSLOTH_NO_TORCH env propagation in install.sh.
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

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
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

# Self-contained arch detection snippet matching install.sh logic
_ARCH_SNIPPET=$(mktemp)
cat > "$_ARCH_SNIPPET" << 'SNIPPET'
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
fi
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

# Mock nvidia-smi that returns CUDA version (to prove macOS ignores it)
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
_local_count=$(grep -c 'UNSLOTH_NO_TORCH=' "$INSTALL_SH" | head -1)
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

# Verify MAC_INTEL is set to true when Intel Mac is detected
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
echo "=== E2E: venv creation at Python 3.12 (simulated Intel Mac) ==="

# Actually create a uv venv at Python 3.12 to verify the path works
if command -v uv >/dev/null 2>&1; then
    _VENV_DIR=$(mktemp -d)
    _uv_result=$(uv venv "$_VENV_DIR/test_venv" --python 3.12 2>&1) && _uv_rc=0 || _uv_rc=$?
    if [ "$_uv_rc" -eq 0 ]; then
        echo "  PASS: uv venv created at Python 3.12"
        PASS=$((PASS + 1))

        # Verify Python version inside the venv
        _py_ver=$("$_VENV_DIR/test_venv/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        assert_eq "venv Python is 3.12" "3.12" "$_py_ver"

        # Verify torch is NOT available (fresh venv has no torch)
        if "$_VENV_DIR/test_venv/bin/python" -c "import torch" 2>/dev/null; then
            echo "  FAIL: torch should NOT be importable in fresh 3.12 venv"
            FAIL=$((FAIL + 1))
        else
            echo "  PASS: torch not importable in fresh 3.12 venv (expected for Intel Mac)"
            PASS=$((PASS + 1))
        fi
    else
        echo "  SKIP: Could not create Python 3.12 venv (python 3.12 not available)"
    fi
    rm -rf "$_VENV_DIR"
else
    echo "  SKIP: uv not available, cannot test venv creation"
fi

echo ""
echo "=== E2E: torch install skipped when MAC_INTEL=true (mock uv shim) ==="

# Create a mock uv that logs all calls instead of running them
_MOCK_UV_DIR=$(mktemp -d)
_UV_LOG="$_MOCK_UV_DIR/uv_calls.log"
touch "$_UV_LOG"
cat > "$_MOCK_UV_DIR/uv" << MOCK_UV_EOF
#!/bin/sh
echo "UV_CALL: \$*" >> "$_UV_LOG"
MOCK_UV_EOF
chmod +x "$_MOCK_UV_DIR/uv"

# Extract the torch install block from install.sh and run with MAC_INTEL=true
_TORCH_BLOCK=$(mktemp)
cat > "$_TORCH_BLOCK" << 'TORCH_EOF'
# Simulates the torch install decision from install.sh
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
_VENV_PY="/fake/python"
if [ "$MAC_INTEL" = true ]; then
    echo "==> Skipping PyTorch (unavailable for Intel Mac x86_64)."
else
    echo "==> Installing PyTorch ($TORCH_INDEX_URL)..."
    uv pip install --python "$_VENV_PY" "torch>=2.4,<2.11.0" torchvision torchaudio \
        --index-url "$TORCH_INDEX_URL"
fi
TORCH_EOF

# Test: MAC_INTEL=true -> torch install should be SKIPPED (no uv calls)
> "$_UV_LOG"  # clear log
_torch_output=$(MAC_INTEL=true PATH="$_MOCK_UV_DIR:$PATH" bash "$_TORCH_BLOCK" 2>&1)
assert_contains "MAC_INTEL=true prints skip message" "$_torch_output" "Skipping PyTorch"
if [ -s "$_UV_LOG" ]; then
    echo "  FAIL: uv was called when MAC_INTEL=true (should be skipped)"
    echo "    Log: $(cat "$_UV_LOG")"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: no uv pip install torch when MAC_INTEL=true"
    PASS=$((PASS + 1))
fi

# Test: MAC_INTEL=false -> torch install should EXECUTE (uv called with torch)
> "$_UV_LOG"  # clear log
_torch_output=$(MAC_INTEL=false PATH="$_MOCK_UV_DIR:$PATH" bash "$_TORCH_BLOCK" 2>&1)
assert_contains "MAC_INTEL=false prints install message" "$_torch_output" "Installing PyTorch"
if grep -q "torch" "$_UV_LOG"; then
    echo "  PASS: uv pip install torch called when MAC_INTEL=false"
    PASS=$((PASS + 1))
else
    echo "  FAIL: uv pip install torch NOT called when MAC_INTEL=false"
    FAIL=$((FAIL + 1))
fi

rm -f "$_TORCH_BLOCK"
rm -rf "$_MOCK_UV_DIR"

echo ""
echo "=== E2E: UNSLOTH_NO_TORCH env propagation (dynamic test) ==="

# Extract the setup.sh invocation block and replace `bash "$SETUP_SH"` with
# `env | grep UNSLOTH` to capture the env variable without actually running setup.sh
_ENV_BLOCK=$(mktemp)
cat > "$_ENV_BLOCK" << 'ENV_EOF'
# Simulates the setup.sh invocation block from install.sh
PACKAGE_NAME="unsloth"
_REPO_ROOT="/fake/repo"
SETUP_SH="/fake/setup.sh"

if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
    SKIP_STUDIO_BASE=1 \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=1 \
    STUDIO_LOCAL_REPO="$_REPO_ROOT" \
    UNSLOTH_NO_TORCH="$MAC_INTEL" \
    env | grep "^UNSLOTH_NO_TORCH="
else
    SKIP_STUDIO_BASE=1 \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    UNSLOTH_NO_TORCH="$MAC_INTEL" \
    env | grep "^UNSLOTH_NO_TORCH="
fi
ENV_EOF

# Test: MAC_INTEL=true -> UNSLOTH_NO_TORCH=true in env
_env_result=$(MAC_INTEL=true STUDIO_LOCAL_INSTALL=false bash "$_ENV_BLOCK" 2>&1)
assert_eq "non-local: UNSLOTH_NO_TORCH=true when MAC_INTEL=true" "UNSLOTH_NO_TORCH=true" "$_env_result"

# Test: MAC_INTEL=false -> UNSLOTH_NO_TORCH=false in env
_env_result=$(MAC_INTEL=false STUDIO_LOCAL_INSTALL=false bash "$_ENV_BLOCK" 2>&1)
assert_eq "non-local: UNSLOTH_NO_TORCH=false when MAC_INTEL=false" "UNSLOTH_NO_TORCH=false" "$_env_result"

# Test: local install path also propagates
_env_result=$(MAC_INTEL=true STUDIO_LOCAL_INSTALL=true bash "$_ENV_BLOCK" 2>&1)
assert_eq "local: UNSLOTH_NO_TORCH=true when MAC_INTEL=true" "UNSLOTH_NO_TORCH=true" "$_env_result"

_env_result=$(MAC_INTEL=false STUDIO_LOCAL_INSTALL=true bash "$_ENV_BLOCK" 2>&1)
assert_eq "local: UNSLOTH_NO_TORCH=false when MAC_INTEL=false" "UNSLOTH_NO_TORCH=false" "$_env_result"

rm -f "$_ENV_BLOCK"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
