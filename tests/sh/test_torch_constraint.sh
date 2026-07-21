#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
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
        _TORCH_CEILING=\"2.12.0\"
        TORCH_CONSTRAINT=\"torch>=2.4,<\${_TORCH_CEILING}\"
        if [ \"\$SKIP_TORCH\" = false ] && [ \"\$OS\" = \"macos\" ] && [ \"\$_ARCH\" = \"arm64\" ]; then
            _PY_MINOR=\$(\"\$VENV_DIR/bin/python\" -c \"import sys; print(sys.version_info.minor)\" 2>/dev/null || echo \"0\")
            if [ \"\$_PY_MINOR\" -ge 13 ] 2>/dev/null; then
                TORCH_CONSTRAINT=\"torch>=2.6,<\${_TORCH_CEILING}\"
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

# The supported line is centralized in per-file ceiling variables so a future
# torch 2.12 bump is a three-line change; the default range admits torch 2.11.
_count=$(grep -c '_TORCH_CEILING="2.12.0"' "$INSTALL_SH" || true)
assert_eq "torch ceiling variable defined once" "1" "$_count"
_count=$(grep -c '_TORCHVISION_CEILING="0.27.0"' "$INSTALL_SH" || true)
assert_eq "torchvision ceiling variable defined once" "1" "$_count"
_count=$(grep -c '_TORCHAUDIO_CEILING="2.12.0"' "$INSTALL_SH" || true)
assert_eq "torchaudio ceiling variable defined once" "1" "$_count"

_count=$(grep -c 'TORCH_CONSTRAINT="torch>=2.4,<${_TORCH_CEILING}"' "$INSTALL_SH" || true)
assert_eq "default TORCH_CONSTRAINT composes the ceiling" "1" "$_count"
_count=$(grep -c 'TORCH_CONSTRAINT="torch>=2.6,<${_TORCH_CEILING}"' "$INSTALL_SH" || true)
assert_eq "tightened TORCH_CONSTRAINT composes the ceiling" "1" "$_count"

_count=$(grep -c '"\$TORCH_CONSTRAINT"' "$INSTALL_SH" || true)
_has_var=$([ "$_count" -ge 1 ] && echo "yes" || echo "no")
assert_eq "\$TORCH_CONSTRAINT used in pip install" "yes" "$_has_var"

# No stray hardcoded default ranges outside the ceiling-composed assignments
# (the curated ROCm >=2.11 floors are deliberately literal).
_hardcoded=$(grep -c '"torch>=2.4,<2.11.0"\|"torch>=2.4,<2.12.0"' "$INSTALL_SH" || true)
assert_eq "no hardcoded default torch range remains" "0" "$_hardcoded"

# Companions must be bounded to torch's window everywhere, never bare: torchaudio
# 2.11 dropped its exact torch pin, so a bare companion can drift from a capped torch.
_count=$(grep -c 'TORCHVISION_CONSTRAINT="torchvision>=0.19,<${_TORCHVISION_CEILING}"' "$INSTALL_SH" || true)
assert_eq "torchvision default composes the ceiling" "1" "$_count"
_count=$(grep -c 'TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<${_TORCHAUDIO_CEILING}"' "$INSTALL_SH" || true)
assert_eq "torchaudio default composes the ceiling" "1" "$_count"
_count=$(grep -c 'TORCHVISION_CONSTRAINT="torchvision"$' "$INSTALL_SH" || true)
assert_eq "no bare torchvision companion remains" "0" "$_count"
_count=$(grep -c 'TORCHAUDIO_CONSTRAINT="torchaudio"$' "$INSTALL_SH" || true)
assert_eq "no bare torchaudio companion remains" "0" "$_count"

# Widening keys off the final leaf (_torch_index_leaf), not the full URL, so a
# mirror base path with cu*/rocm7.2 but a cpu/older-rocm leaf is not mis-widened.
_cuda_case=$(grep -c 'cu\[0-9\]\*)' "$INSTALL_SH" || true)
_has_cuda_case=$([ "$_cuda_case" -ge 1 ] && echo "yes" || echo "no")
assert_eq "cu* index case adjusts TORCH_CONSTRAINT" "yes" "$_has_cuda_case"
_leaf_case=$(grep -c 'case "\$_torch_index_leaf" in' "$INSTALL_SH" || true)
_has_leaf_constraint=$([ "$_leaf_case" -ge 2 ] && echo "yes" || echo "no")
assert_eq "constraint case anchors on _torch_index_leaf" "yes" "$_has_leaf_constraint"

echo ""
echo "=== Structural: tokenizers in no-torch-runtime.txt ==="

# Package-name boundary is anything not valid in a PEP 508 name, or EOL.
# Covers `tokenizers`, `tokenizers<=0.23.0`, `tokenizers[extra]`,
# `tokenizers; python_version<"3.13"`, etc., but NOT `tokenizers-foo`.
_TOK_RE='^tokenizers([^a-zA-Z0-9._-]|$)'

_has_tokenizers=$(grep -cE "$_TOK_RE" "$NO_TORCH_RT" || true)
assert_eq "tokenizers package listed" "1" "$_has_tokenizers"

# Regression guard for #5359: the tokenizers line must carry an upper
# bound that excludes 0.23.1+. transformers in the allowed 4.56..5.3
# window rejects 0.23.1 at import time with
#   `tokenizers<=0.23.0,>=0.22.0 is required, but found 0.23.1`.
# Accept both `<=0.23.0` and the functionally equivalent `<0.23.1`.
# Two-stage grep: pick lines that start with the tokenizers package
# name (PEP 508 name boundary), then require a safe upper bound.
_has_safe_pin=$(grep -E "$_TOK_RE" "$NO_TORCH_RT" \
    | grep -cE '(<=[[:space:]]*0\.23\.0|<[[:space:]]*0\.23\.1)' \
    || true)
assert_eq "tokenizers pinned with upper bound excluding 0.23.1+" "1" "$_has_safe_pin"

# tokenizers before transformers
_tok_line=$(grep -nE "$_TOK_RE" "$NO_TORCH_RT" | head -1 | cut -d: -f1)
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

_ps1_hardcoded=$(echo "$_PS1_CONTENT" | grep -c '"torch>=2.4,<2.12.0"' || true)
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
assert_eq "arm64+macos+py313 -> tightened" "torch>=2.6,<2.12.0" "$_result"

# 2. arm64 macOS py3.14 -> tightened (future-proofed)
_result=$(run_constraint_snippet false macos arm64 14 "$TMPDIR_BASE/v2")
assert_eq "arm64+macos+py314 -> tightened" "torch>=2.6,<2.12.0" "$_result"

# 3. arm64 macOS py3.12 -> default
_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v3")
assert_eq "arm64+macos+py312 -> default" "torch>=2.4,<2.12.0" "$_result"

# 4. arm64 macOS py3.11 -> default
_result=$(run_constraint_snippet false macos arm64 11 "$TMPDIR_BASE/v4")
assert_eq "arm64+macos+py311 -> default" "torch>=2.4,<2.12.0" "$_result"

# 5. Linux x86_64 py3.13 -> default (Linux unaffected)
_result=$(run_constraint_snippet false linux x86_64 13 "$TMPDIR_BASE/v5")
assert_eq "linux+x86_64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

# 6. Linux aarch64 py3.13 -> default (guard checks OS=macos)
_result=$(run_constraint_snippet false linux aarch64 13 "$TMPDIR_BASE/v6")
assert_eq "linux+aarch64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

# 7. Intel Mac x86_64 py3.12 -> default (arch mismatch)
_result=$(run_constraint_snippet false macos x86_64 12 "$TMPDIR_BASE/v7")
assert_eq "macos+x86_64+py312 -> default" "torch>=2.4,<2.12.0" "$_result"

# 8. SKIP_TORCH=true arm64 macOS py3.13 -> block skipped, default
_result=$(run_constraint_snippet true macos arm64 13 "$TMPDIR_BASE/v8")
assert_eq "SKIP_TORCH=true -> default" "torch>=2.4,<2.12.0" "$_result"

# 9. WSL py3.13 -> default
_result=$(run_constraint_snippet false wsl x86_64 13 "$TMPDIR_BASE/v9")
assert_eq "wsl+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

# 10. py_minor=0 (failed query fallback) -> default
_result=$(run_constraint_snippet false macos arm64 0 "$TMPDIR_BASE/v10")
assert_eq "py_minor=0 fallback -> default" "torch>=2.4,<2.12.0" "$_result"

# 11. Boundary: py_minor=12 -> NOT tightened
_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v11")
assert_eq "boundary py_minor=12 -> default" "torch>=2.4,<2.12.0" "$_result"

# 12. Boundary: py_minor=13 -> tightened
_result=$(run_constraint_snippet false macos arm64 13 "$TMPDIR_BASE/v12")
assert_eq "boundary py_minor=13 -> tightened" "torch>=2.6,<2.12.0" "$_result"

# 13. Intel Mac py3.13 -> default (arch=x86_64, not arm64)
_result=$(run_constraint_snippet false macos x86_64 13 "$TMPDIR_BASE/v13")
assert_eq "macos+x86_64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

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
# ROCm 2.11 floor: leaf is lowercased before the gfx*/rocm* allowlist match
# ======================================================================
echo ""
echo "=== ROCm 2.11 floor case (leaf normalization) ==="

# Structural: install.sh lowercases _torch_index_leaf before the floor case, so the
# canonical gfx120X-all (capital X) matches gfx120x-all.
_has_lc=$(grep -c '_torch_index_leaf=$(printf .* | tr .\[:upper:\]. .\[:lower:\].)' "$INSTALL_SH" || true)
_has_lc_ok=$([ "$_has_lc" -ge 1 ] && echo "yes" || echo "no")
assert_eq "install.sh lowercases _torch_index_leaf" "yes" "$_has_lc_ok"

# Runtime: replicate install.sh's normalization + floor case and assert both gfx120X-all
# and gfx120x-all get the floor, while non-2.11 leaves keep the default.
run_floor_case() {
    _url="$1"
    bash -c '
        TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
        TORCHVISION_CONSTRAINT="torchvision"
        TORCHAUDIO_CONSTRAINT="torchaudio"
        _torch_index_leaf="${1%/}"
        _torch_index_leaf="${_torch_index_leaf##*/}"
        _torch_index_leaf=$(printf "%s" "$_torch_index_leaf" | tr "[:upper:]" "[:lower:]")
        case "$_torch_index_leaf" in
            rocm7.2|gfx120x-all|gfx1151|gfx1150)
                TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
                TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
                TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
                ;;
        esac
        echo "$TORCH_CONSTRAINT"
    ' _ "$_url"
}

assert_eq "gfx120X-all (capital) -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx120X-all')"
assert_eq "gfx120X-all trailing slash -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx120X-all/')"
assert_eq "gfx120x-all (lowercase) -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx120x-all')"
assert_eq "gfx1151 -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx1151')"
assert_eq "gfx1150 -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx1150')"
assert_eq "rocm7.2 -> 2.11 floor" "torch>=2.11.0,<2.12.0" \
    "$(run_floor_case 'https://download.pytorch.org/whl/rocm7.2')"
assert_eq "gfx110X-all -> default (no floor)" "torch>=2.4,<2.11.0" \
    "$(run_floor_case 'https://repo.amd.com/rocm/whl/gfx110X-all')"
assert_eq "rocm6.4 -> default (no floor)" "torch>=2.4,<2.11.0" \
    "$(run_floor_case 'https://download.pytorch.org/whl/rocm6.4')"
assert_eq "cu128 -> default (no floor)" "torch>=2.4,<2.11.0" \
    "$(run_floor_case 'https://download.pytorch.org/whl/cu128')"
assert_eq "cpu -> default (no floor)" "torch>=2.4,<2.11.0" \
    "$(run_floor_case 'https://download.pytorch.org/whl/cpu')"

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
