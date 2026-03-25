#!/bin/bash
# Unit tests for get_torch_index_url() from install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract only the get_torch_index_url function from install.sh
# Also replace the hardcoded /usr/bin/nvidia-smi fallback with a
# controllable path so we can test the "no GPU" scenario on GPU machines.
_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH" \
    | sed "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
    > "$_FUNC_FILE"

# Save system PATH so we always have basic tools (uname, grep, head, etc.)
_SYS_PATH="/usr/local/bin:/usr/bin:/bin"

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

# Helper: create a mock nvidia-smi that prints a given CUDA version string
make_mock_smi() {
    _dir=$(mktemp -d)
    cat > "$_dir/nvidia-smi" <<MOCK
#!/bin/sh
cat <<'SMI_OUT'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: $1     |
+-----------------------------------------------------------------------------------------+
SMI_OUT
MOCK
    chmod +x "$_dir/nvidia-smi"
    echo "$_dir"
}

# Build a minimal tools directory with symlinks to essential commands
# (uname, grep, head, etc.) but WITHOUT nvidia-smi.
_TOOLS_DIR=$(mktemp -d)
for _cmd in uname grep sed head sh bash cat; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

# Helper: run get_torch_index_url with a custom PATH
# $1 = directory with mock nvidia-smi (prepended to PATH), or "none" for no-GPU test
run_func() {
    _mock_dir="$1"
    if [ "$_mock_dir" = "none" ]; then
        # Minimal PATH with only basic tools, no nvidia-smi anywhere
        PATH="$_TOOLS_DIR" bash -c ". '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
    else
        # Put mock nvidia-smi dir first, then basic tools
        PATH="$_mock_dir:$_TOOLS_DIR" bash -c ". '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
    fi
}

echo "=== test_get_torch_index_url ==="

# 1) No nvidia-smi available -> cpu
_result=$(run_func "none")
assert_eq "no nvidia-smi -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"

# 2) CUDA 12.6 -> cu126
_dir=$(make_mock_smi "12.6")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.6 -> cu126" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_dir"

# 3) CUDA 12.8 -> cu128
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.8 -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 4) CUDA 13.0 -> cu130
_dir=$(make_mock_smi "13.0")
_result=$(run_func "$_dir")
assert_eq "CUDA 13.0 -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 5) CUDA 12.4 -> cu124
_dir=$(make_mock_smi "12.4")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.4 -> cu124" "https://download.pytorch.org/whl/cu124" "$_result"
rm -rf "$_dir"

# 6) CUDA 11.8 -> cu118
_dir=$(make_mock_smi "11.8")
_result=$(run_func "$_dir")
assert_eq "CUDA 11.8 -> cu118" "https://download.pytorch.org/whl/cu118" "$_result"
rm -rf "$_dir"

# 7) CUDA 10.2 (too old) -> cpu
_dir=$(make_mock_smi "10.2")
_result=$(run_func "$_dir")
assert_eq "CUDA 10.2 -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 8) Unparseable nvidia-smi output -> cu126 default
_dir=$(mktemp -d)
cat > "$_dir/nvidia-smi" <<'MOCK'
#!/bin/sh
echo "something completely unexpected"
MOCK
chmod +x "$_dir/nvidia-smi"
_result=$(run_func "$_dir")
assert_eq "unparseable -> cu126" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_dir"

rm -f "$_FUNC_FILE"
rm -rf "$_FAKE_SMI_DIR"
rm -rf "$_TOOLS_DIR"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
