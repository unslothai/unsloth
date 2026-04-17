#!/bin/bash
# Tests for TORCH_CONSTRAINT variable in install.sh and tokenizers in no-torch-runtime.txt.
# Follows the same assertion pattern as test_mac_intel_compat.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
NO_TORCH_RT="$SCRIPT_DIR/../../studio/backend/requirements/no-torch-runtime.txt"
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

# ── Helper: create a mock python that reports a given minor version ──
make_mock_python() {
    _minor="$1"
    _venv_dir="$2"
    mkdir -p "$_venv_dir/bin"
    cat > "$_venv_dir/bin/python" <<MOCK_EOF
#!/bin/bash
if echo "\$@" | grep -q "sys.version_info.minor"; then
    echo "$_minor"
else
    echo "0"
fi
MOCK_EOF
    chmod +x "$_venv_dir/bin/python"
}

# ── Helper: run the TORCH_CONSTRAINT snippet with given params ──
run_constraint_snippet() {
    _skip_torch="$1"
    _os="$2"
    _arch="$3"
    _py_minor="$4"
    _venv_dir="$5"

    make_mock_python "$_py_minor" "$_venv_dir"

    bash -c "
        SKIP_TORCH=\"$_skip_torch\"
        OS=\"$_os\"
        _ARCH=\"$_arch\"
        VENV_DIR=\"$_venv_dir\"
        TORCH_CONSTRAINT=\"torch>=2.4,<2.11.0\"
        if [ \"\$SKIP_TORCH\" = false ] && [ \"\$OS\" = \"macos\" ] && [ \"\$_ARCH\" = \"arm64\" ]; then
            _PY_MINOR=\$(\"\$VENV_DIR/bin/python\" -c \"import sys; print(sys.version_info.minor)\" 2>/dev/null || echo \"0\")
            if [ \"\$_PY_MINOR\" -ge 13 ] 2>/dev/null; then
                TORCH_CONSTRAINT=\"torch>=2.6,<2.11.0\"
            fi
        fi
        echo \"\$TORCH_CONSTRAINT\"
    " 2>/dev/null
}

# ======================================================================
# Structural checks
# ======================================================================
echo "=== Structural: TORCH_CONSTRAINT in install.sh ==="

_SH_CONTENT=$(cat "$INSTALL_SH")

_count=$(grep -c 'TORCH_CONSTRAINT="torch>=2.4,<2.11.0"' "$INSTALL_SH" || true)
assert_eq "default TORCH_CONSTRAINT assignment exists" "1" "$_count"

_count=$(grep -c 'TORCH_CONSTRAINT="torch>=2.6,<2.11.0"' "$INSTALL_SH" || true)
assert_eq "tightened TORCH_CONSTRAINT assignment exists" "1" "$_count"

_count=$(grep -c '"\$TORCH_CONSTRAINT"' "$INSTALL_SH" || true)
_has_var=$([ "$_count" -ge 1 ] && echo "yes" || echo "no")
assert_eq "\$TORCH_CONSTRAINT used in pip install" "yes" "$_has_var"

# Hardcoded torch>=2.4,<2.11.0 should only appear once (the default assignment)
_hardcoded=$(grep -c '"torch>=2.4,<2.11.0"' "$INSTALL_SH" || true)
assert_eq "hardcoded torch>=2.4 appears exactly once" "1" "$_hardcoded"

echo ""
echo "=== Structural: tokenizers in no-torch-runtime.txt ==="

_has_tokenizers=$(grep -c '^tokenizers$' "$NO_TORCH_RT" || true)
assert_eq "tokenizers present as standalone line" "1" "$_has_tokenizers"

# tokenizers before transformers
_tok_line=$(grep -n '^tokenizers$' "$NO_TORCH_RT" | head -1 | cut -d: -f1)
_tf_line=$(grep -n '^transformers' "$NO_TORCH_RT" | head -1 | cut -d: -f1)
_tok_first=$([ "$_tok_line" -lt "$_tf_line" ] && echo "yes" || echo "no")
assert_eq "tokenizers before transformers" "yes" "$_tok_first"

# torch itself NOT in no-torch file
_has_torch=$(grep -c '^torch$' "$NO_TORCH_RT" || true)
assert_eq "torch not in no-torch-runtime.txt" "0" "$_has_torch"

echo ""
echo "=== Structural: install.ps1 unchanged ==="

_PS1_CONTENT=$(cat "$INSTALL_PS1")
_ps1_has_var=$(echo "$_PS1_CONTENT" | grep -c 'TORCH_CONSTRAINT\|TorchConstraint' || true)
assert_eq "install.ps1 has no TORCH_CONSTRAINT variable" "0" "$_ps1_has_var"

_ps1_hardcoded=$(echo "$_PS1_CONTENT" | grep -c '"torch>=2.4,<2.11.0"' || true)
_ps1_has_hc=$([ "$_ps1_hardcoded" -ge 1 ] && echo "yes" || echo "no")
assert_eq "install.ps1 has hardcoded torch constraint" "yes" "$_ps1_has_hc"

# ======================================================================
# Runtime: mocked platform/version combos
# ======================================================================
echo ""
echo "=== Runtime: TORCH_CONSTRAINT with mocked inputs ==="

TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

# 1. arm64 macOS py3.13 -> tightened
_result=$(run_constraint_snippet false macos arm64 13 "$TMPDIR_BASE/v1")
assert_eq "arm64+macos+py313 -> tightened" "torch>=2.6,<2.11.0" "$_result"

# 2. arm64 macOS py3.14 -> tightened (future-proofed)
_result=$(run_constraint_snippet false macos arm64 14 "$TMPDIR_BASE/v2")
assert_eq "arm64+macos+py314 -> tightened" "torch>=2.6,<2.11.0" "$_result"

# 3. arm64 macOS py3.12 -> default
_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v3")
assert_eq "arm64+macos+py312 -> default" "torch>=2.4,<2.11.0" "$_result"

# 4. arm64 macOS py3.11 -> default
_result=$(run_constraint_snippet false macos arm64 11 "$TMPDIR_BASE/v4")
assert_eq "arm64+macos+py311 -> default" "torch>=2.4,<2.11.0" "$_result"

# 5. Linux x86_64 py3.13 -> default (Linux unaffected)
_result=$(run_constraint_snippet false linux x86_64 13 "$TMPDIR_BASE/v5")
assert_eq "linux+x86_64+py313 -> default" "torch>=2.4,<2.11.0" "$_result"

# 6. Linux aarch64 py3.13 -> default (guard checks OS=macos)
_result=$(run_constraint_snippet false linux aarch64 13 "$TMPDIR_BASE/v6")
assert_eq "linux+aarch64+py313 -> default" "torch>=2.4,<2.11.0" "$_result"

# 7. Intel Mac x86_64 py3.12 -> default (arch mismatch)
_result=$(run_constraint_snippet false macos x86_64 12 "$TMPDIR_BASE/v7")
assert_eq "macos+x86_64+py312 -> default" "torch>=2.4,<2.11.0" "$_result"

# 8. SKIP_TORCH=true arm64 macOS py3.13 -> block skipped, default
_result=$(run_constraint_snippet true macos arm64 13 "$TMPDIR_BASE/v8")
assert_eq "SKIP_TORCH=true -> default" "torch>=2.4,<2.11.0" "$_result"

# 9. WSL py3.13 -> default
_result=$(run_constraint_snippet false wsl x86_64 13 "$TMPDIR_BASE/v9")
assert_eq "wsl+py313 -> default" "torch>=2.4,<2.11.0" "$_result"

# 10. py_minor=0 (failed query fallback) -> default
_result=$(run_constraint_snippet false macos arm64 0 "$TMPDIR_BASE/v10")
assert_eq "py_minor=0 fallback -> default" "torch>=2.4,<2.11.0" "$_result"

# 11. Boundary: py_minor=12 -> NOT tightened
_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v11")
assert_eq "boundary py_minor=12 -> default" "torch>=2.4,<2.11.0" "$_result"

# 12. Boundary: py_minor=13 -> tightened
_result=$(run_constraint_snippet false macos arm64 13 "$TMPDIR_BASE/v12")
assert_eq "boundary py_minor=13 -> tightened" "torch>=2.6,<2.11.0" "$_result"

# 13. Intel Mac py3.13 -> default (arch=x86_64, not arm64)
_result=$(run_constraint_snippet false macos x86_64 13 "$TMPDIR_BASE/v13")
assert_eq "macos+x86_64+py313 -> default" "torch>=2.4,<2.11.0" "$_result"

# ======================================================================
# Mock uv integration
# ======================================================================
echo ""
echo "=== Mock uv: verify constraint passed to uv ==="

# arm64 + py313 -> uv receives torch>=2.6
_UV_LOG="$TMPDIR_BASE/uv_log_tight.txt"
make_mock_python 13 "$TMPDIR_BASE/uv_venv1"
cat > "$TMPDIR_BASE/mock_uv_tight" <<UVEOF
#!/bin/bash
echo "\$@" >> $_UV_LOG
UVEOF
chmod +x "$TMPDIR_BASE/mock_uv_tight"

bash -c "
    SKIP_TORCH=false
    OS=\"macos\"
    _ARCH=\"arm64\"
    VENV_DIR=\"$TMPDIR_BASE/uv_venv1\"
    TORCH_CONSTRAINT=\"torch>=2.4,<2.11.0\"
    if [ \"\$SKIP_TORCH\" = false ] && [ \"\$OS\" = \"macos\" ] && [ \"\$_ARCH\" = \"arm64\" ]; then
        _PY_MINOR=\$(\"\$VENV_DIR/bin/python\" -c \"import sys; print(sys.version_info.minor)\" 2>/dev/null || echo \"0\")
        if [ \"\$_PY_MINOR\" -ge 13 ] 2>/dev/null; then
            TORCH_CONSTRAINT=\"torch>=2.6,<2.11.0\"
        fi
    fi
    \"$TMPDIR_BASE/mock_uv_tight\" pip install --python \"\$VENV_DIR/bin/python\" \"\$TORCH_CONSTRAINT\" torchvision torchaudio
" 2>/dev/null
_uv_got=$(cat "$_UV_LOG" 2>/dev/null || echo "")
assert_contains "mock uv arm64+py313 receives torch>=2.6" "$_uv_got" "torch>=2.6,<2.11.0"

# arm64 + py312 -> uv receives torch>=2.4
_UV_LOG2="$TMPDIR_BASE/uv_log_default.txt"
make_mock_python 12 "$TMPDIR_BASE/uv_venv2"
cat > "$TMPDIR_BASE/mock_uv_default" <<UVEOF
#!/bin/bash
echo "\$@" >> $_UV_LOG2
UVEOF
chmod +x "$TMPDIR_BASE/mock_uv_default"

bash -c "
    SKIP_TORCH=false
    OS=\"macos\"
    _ARCH=\"arm64\"
    VENV_DIR=\"$TMPDIR_BASE/uv_venv2\"
    TORCH_CONSTRAINT=\"torch>=2.4,<2.11.0\"
    if [ \"\$SKIP_TORCH\" = false ] && [ \"\$OS\" = \"macos\" ] && [ \"\$_ARCH\" = \"arm64\" ]; then
        _PY_MINOR=\$(\"\$VENV_DIR/bin/python\" -c \"import sys; print(sys.version_info.minor)\" 2>/dev/null || echo \"0\")
        if [ \"\$_PY_MINOR\" -ge 13 ] 2>/dev/null; then
            TORCH_CONSTRAINT=\"torch>=2.6,<2.11.0\"
        fi
    fi
    \"$TMPDIR_BASE/mock_uv_default\" pip install --python \"\$VENV_DIR/bin/python\" \"\$TORCH_CONSTRAINT\" torchvision torchaudio
" 2>/dev/null
_uv_got2=$(cat "$_UV_LOG2" 2>/dev/null || echo "")
assert_contains "mock uv arm64+py312 receives torch>=2.4" "$_uv_got2" "torch>=2.4,<2.11.0"

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
