#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests for TORCH_CONSTRAINT in install.sh and tokenizers in no-torch-runtime.txt.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
NO_TORCH_RT="$SCRIPT_DIR/../../studio/backend/requirements/no-torch-runtime.txt"
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

echo "=== Structural: TORCH_CONSTRAINT in install.sh ==="

_SH_CONTENT=$(cat "$INSTALL_SH")

_count=$(grep -c '_TORCH_CEILING="2.12.0"' "$INSTALL_SH" || true)
assert_eq "torch ceiling variable defined once" "1" "$_count"
_count=$(grep -c '_TORCHVISION_CEILING="0.27.0"' "$INSTALL_SH" || true)
assert_eq "torchvision ceiling variable defined once" "1" "$_count"
_count=$(grep -c '_TORCHAUDIO_CEILING="2.12.0"' "$INSTALL_SH" || true)
assert_eq "torchaudio ceiling variable defined once" "1" "$_count"

# Branch triples are indented; the column-0 composed default is the one anchored.
_count=$(grep -c '^TORCH_CONSTRAINT="torch>=2.4,<${_TORCH_CEILING}"$' "$INSTALL_SH" || true)
assert_eq "default TORCH_CONSTRAINT assignment exists at top level" "1" "$_count"

_count=$(grep -c 'TORCH_CONSTRAINT="torch>=2.6,<${_TORCH_CEILING}"' "$INSTALL_SH" || true)
_has=$([ "$_count" -ge 1 ] && echo "yes" || echo "no")
assert_eq "tightened TORCH_CONSTRAINT assignment exists" "yes" "$_has"

_count=$(grep -c '"\$TORCH_CONSTRAINT"' "$INSTALL_SH" || true)
_has_var=$([ "$_count" -ge 1 ] && echo "yes" || echo "no")
assert_eq "\$TORCH_CONSTRAINT used in pip install" "yes" "$_has_var"

# An install line spelling the pin out would ignore the branch's choice.
_literal=$(grep -cE 'uv pip install .*"torch>=' "$INSTALL_SH" || true)
assert_eq "no pip install hardcodes a torch pin" "0" "$_literal"

# Curated per-index overrides may still cap literally (gfx906 / MI50 on rocm6.3).
_hardcoded=$(grep -E '"torch>=2\.4,<2\.11\.0"|"torch>=2\.4,<2\.12\.0"' "$INSTALL_SH" \
    | grep -c -v '^[[:space:]]*TORCH_CONSTRAINT=' || true)
assert_eq "no hardcoded default torch range off a TORCH_CONSTRAINT= assignment" "0" "$_hardcoded"

# The gfx906 / MI50 reroute keeps its own sub-2.11 cap. Existence, not a count: more may appear.
_count=$(grep -cE '^[[:space:]]+TORCH_CONSTRAINT="torch>=2\.4,<2\.11\.0"$' "$INSTALL_SH" || true)
_has_sub211_cap=$([ "$_count" -ge 1 ] && echo "yes" || echo "no")
assert_eq "gfx906 reroute caps torch below 2.11 for the rocm6.3 index" "yes" "$_has_sub211_cap"

# Companions must always be bounded: torchaudio 2.11 dropped its exact torch pin, so a bare
# companion beside a <2.11 torch resolves a mismatched build.
_total=$(grep -cE '^[[:space:]]*TORCHVISION_CONSTRAINT="' "$INSTALL_SH" || true)
_bounded=$(grep -cE '^[[:space:]]*TORCHVISION_CONSTRAINT="torchvision>=[0-9][0-9.]*,<([0-9][0-9.]*|[$][{]_TORCHVISION_CEILING[}])"$' "$INSTALL_SH" || true)
assert_eq "every torchvision constraint is upper-bounded" "$_total" "$_bounded"
_total=$(grep -cE '^[[:space:]]*TORCHAUDIO_CONSTRAINT="' "$INSTALL_SH" || true)
_bounded=$(grep -cE '^[[:space:]]*TORCHAUDIO_CONSTRAINT="torchaudio>=[0-9][0-9.]*,<([0-9][0-9.]*|[$][{]_TORCHAUDIO_CEILING[}])"$' "$INSTALL_SH" || true)
assert_eq "every torchaudio constraint is upper-bounded" "$_total" "$_bounded"

_count=$(grep -c '^TORCHVISION_CONSTRAINT="torchvision>=0.19,<${_TORCHVISION_CEILING}"$' "$INSTALL_SH" || true)
assert_eq "torchvision default composes the ceiling" "1" "$_count"
_count=$(grep -c '^TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<${_TORCHAUDIO_CEILING}"$' "$INSTALL_SH" || true)
assert_eq "torchaudio default composes the ceiling" "1" "$_count"
_count=$(grep -c 'TORCHVISION_CONSTRAINT="torchvision"$' "$INSTALL_SH" || true)
assert_eq "no bare torchvision companion remains" "0" "$_count"
_count=$(grep -c 'TORCHAUDIO_CONSTRAINT="torchaudio"$' "$INSTALL_SH" || true)
assert_eq "no bare torchaudio companion remains" "0" "$_count"

# Widening keys off the final leaf, so a mirror base path with cu*/rocm7.2 is not mis-widened.
_cuda_case=$(grep -c 'cu\[0-9\]\*)' "$INSTALL_SH" || true)
_has_cuda_case=$([ "$_cuda_case" -ge 1 ] && echo "yes" || echo "no")
assert_eq "cu* index case adjusts TORCH_CONSTRAINT" "yes" "$_has_cuda_case"
_leaf_case=$(grep -c 'case "\$_torch_index_leaf" in' "$INSTALL_SH" || true)
_has_leaf_constraint=$([ "$_leaf_case" -ge 2 ] && echo "yes" || echo "no")
assert_eq "constraint case anchors on _torch_index_leaf" "yes" "$_has_leaf_constraint"

echo ""
echo "=== Structural: tokenizers in no-torch-runtime.txt ==="

# PEP 508 name boundary: matches `tokenizers<=...`, `tokenizers[x]`, not `tokenizers-foo`.
_TOK_RE='^tokenizers([^a-zA-Z0-9._-]|$)'

_has_tokenizers=$(grep -cE "$_TOK_RE" "$NO_TORCH_RT" || true)
assert_eq "tokenizers package listed" "1" "$_has_tokenizers"

# The tokenizers line must exclude 0.23.1+: transformers 4.56..5.3 rejects it at import.
# Accept both `<=0.23.0` and `<0.23.1`.
_has_safe_pin=$(grep -E "$_TOK_RE" "$NO_TORCH_RT" \
    | grep -cE '(<=[[:space:]]*0\.23\.0|<[[:space:]]*0\.23\.1)' \
    || true)
assert_eq "tokenizers pinned with upper bound excluding 0.23.1+" "1" "$_has_safe_pin"

# tokenizers before transformers
_tok_line=$(grep -nE "$_TOK_RE" "$NO_TORCH_RT" | head -1 | cut -d: -f1)
_tf_line=$(grep -n '^transformers' "$NO_TORCH_RT" | head -1 | cut -d: -f1)
_tok_first=$([ "$_tok_line" -lt "$_tf_line" ] && echo "yes" || echo "no")
assert_eq "tokenizers before transformers" "yes" "$_tok_first"

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

echo ""
echo "=== Runtime: TORCH_CONSTRAINT with mocked inputs ==="

TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

_result=$(run_constraint_snippet false macos arm64 13 "$TMPDIR_BASE/v1")
assert_eq "arm64+macos+py313 -> tightened" "torch>=2.6,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos arm64 14 "$TMPDIR_BASE/v2")
assert_eq "arm64+macos+py314 -> tightened" "torch>=2.6,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v3")
assert_eq "arm64+macos+py312 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos arm64 11 "$TMPDIR_BASE/v4")
assert_eq "arm64+macos+py311 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false linux x86_64 13 "$TMPDIR_BASE/v5")
assert_eq "linux+x86_64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false linux aarch64 13 "$TMPDIR_BASE/v6")
assert_eq "linux+aarch64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos x86_64 12 "$TMPDIR_BASE/v7")
assert_eq "macos+x86_64+py312 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet true macos arm64 13 "$TMPDIR_BASE/v8")
assert_eq "SKIP_TORCH=true -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false wsl x86_64 13 "$TMPDIR_BASE/v9")
assert_eq "wsl+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

# py_minor=0 is the failed-query fallback.
_result=$(run_constraint_snippet false macos arm64 0 "$TMPDIR_BASE/v10")
assert_eq "py_minor=0 fallback -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos arm64 12 "$TMPDIR_BASE/v11")
assert_eq "boundary py_minor=12 -> default" "torch>=2.4,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos arm64 13 "$TMPDIR_BASE/v12")
assert_eq "boundary py_minor=13 -> tightened" "torch>=2.6,<2.12.0" "$_result"

_result=$(run_constraint_snippet false macos x86_64 13 "$TMPDIR_BASE/v13")
assert_eq "macos+x86_64+py313 -> default" "torch>=2.4,<2.12.0" "$_result"

echo ""
echo "=== Mock uv: verify constraint passed to uv ==="

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

echo ""
echo "=== ROCm 2.11 floor case (leaf normalization) ==="

# install.sh lowercases the leaf so canonical gfx120X-all matches gfx120x-all.
_has_lc=$(grep -c '_torch_index_leaf=$(printf .* | tr .\[:upper:\]. .\[:lower:\].)' "$INSTALL_SH" || true)
_has_lc_ok=$([ "$_has_lc" -ge 1 ] && echo "yes" || echo "no")
assert_eq "install.sh lowercases _torch_index_leaf" "yes" "$_has_lc_ok"

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

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
