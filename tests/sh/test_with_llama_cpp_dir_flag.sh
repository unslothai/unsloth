#!/bin/bash
# Static wiring check: --with-llama-cpp-dir must be handled consistently by install.sh,
# install.ps1, studio/setup.sh and studio/setup.ps1 via UNSLOTH_LOCAL_LLAMA_CPP_DIR.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
ENV_VAR="UNSLOTH_LOCAL_LLAMA_CPP_DIR"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _file="$2"; _needle="$3"
    if grep -qF -- "$_needle" "$_file"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle' in $(basename "$_file"))"
        FAIL=$((FAIL + 1))
    fi
}

assert_min_count() {
    _label="$1"; _file="$2"; _pattern="$3"; _min="$4"
    _n=$(grep -cE -- "$_pattern" "$_file" || true)
    if [ "$_n" -ge "$_min" ]; then
        echo "  PASS: $_label (found $_n, need >= $_min)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (found $_n in $(basename "$_file"), need >= $_min)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=== install.sh: parses --with-llama-cpp-dir and forwards the env var ==="

assert_contains \
    "install.sh: accepts --with-llama-cpp-dir flag" \
    "$INSTALL_SH" "--with-llama-cpp-dir"
assert_contains \
    "install.sh: validates the path exists before forwarding" \
    "$INSTALL_SH" 'if [ ! -d "$_WITH_LLAMA_CPP_DIR" ]; then'
# Forwarded on both the local and non-local setup invocations.
assert_min_count \
    "install.sh: forwards $ENV_VAR on both setup invocations" \
    "$INSTALL_SH" "$ENV_VAR=\"\\\$_WITH_LLAMA_CPP_DIR\"" 2

echo ""
echo "=== install.ps1: parses --with-llama-cpp-dir and forwards the env var ==="

assert_contains \
    "install.ps1: accepts --with-llama-cpp-dir flag" \
    "$INSTALL_PS1" '"--with-llama-cpp-dir"'
assert_contains \
    "install.ps1: errors when flag is given with no path argument" \
    "$INSTALL_PS1" "--with-llama-cpp-dir requires a path argument"
assert_contains \
    "install.ps1: validates the path exists before forwarding" \
    "$INSTALL_PS1" "--with-llama-cpp-dir path does not exist"
assert_contains \
    "install.ps1: exports $ENV_VAR for setup.ps1" \
    "$INSTALL_PS1" "\$env:$ENV_VAR ="
assert_contains \
    "install.ps1: clears $ENV_VAR after the setup run" \
    "$INSTALL_PS1" "Remove-Item Env:$ENV_VAR"

echo ""
echo "=== studio/setup.sh: reads the env var, links, and skips download+build ==="

assert_contains \
    "setup.sh: reads $ENV_VAR" \
    "$SETUP_SH" "$ENV_VAR"
assert_contains \
    "setup.sh: symlinks the local dir into the canonical install location" \
    "$SETUP_SH" 'ln -sfn "$_RESOLVED_LOCAL" "$LLAMA_CPP_DIR"'
assert_contains \
    "setup.sh: disables the source build when the local dir is linked" \
    "$SETUP_SH" "_NEED_LLAMA_SOURCE_BUILD=false"
assert_contains \
    "setup.sh: skips the prebuilt download when the local dir is linked" \
    "$SETUP_SH" "_SKIP_PREBUILT_INSTALL=true"
assert_contains \
    "setup.sh: link branch gates the prebuilt/compile chain" \
    "$SETUP_SH" 'if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then'

echo ""
echo "=== studio/setup.ps1: reads the env var, junctions, and skips download+build ==="

assert_contains \
    "setup.ps1: reads $ENV_VAR" \
    "$SETUP_PS1" "\$env:$ENV_VAR"
assert_contains \
    "setup.ps1: creates a directory junction into the canonical location" \
    "$SETUP_PS1" "mklink /J"
assert_contains \
    "setup.ps1: falls back to a copy when the junction can't be created" \
    "$SETUP_PS1" "Copy-Item -Recurse -LiteralPath \$ResolvedLocal -Destination \$LlamaCppDir"
assert_contains \
    "setup.ps1: disables the source build when the local dir is linked" \
    "$SETUP_PS1" '$NeedLlamaSourceBuild = $false'
assert_contains \
    "setup.ps1: link branch gates the prebuilt/compile chain" \
    "$SETUP_PS1" 'if ($LocalLlamaCppLinked) {'
assert_contains \
    "setup.ps1: linked-dir case short-circuits the build chain" \
    "$SETUP_PS1" 'step "llama.cpp" "linked (skipping build)"'

echo ""
echo "=== both setup scripts: validate against every layout the backend resolves ==="

# Must match LlamaCppBackend._layout_candidates() (root, build/bin, build/bin/Release).
assert_contains \
    "setup.sh: accepts root-level or build/bin llama-server layouts" \
    "$SETUP_SH" '[ -x "$1/llama-server" ] || [ -x "$1/build/bin/llama-server" ]'
assert_contains \
    "setup.ps1: accepts the build\\bin (non-Release) llama-server.exe layout" \
    "$SETUP_PS1" 'Join-Path $ResolvedLocal "build\bin\llama-server.exe"'
assert_contains \
    "setup.ps1: accepts the root-level llama-server.exe layout" \
    "$SETUP_PS1" 'Join-Path $ResolvedLocal "llama-server.exe"'

echo ""
echo "=== both setup scripts: a local dir pointing at the canonical path is a no-op ==="

assert_contains \
    "setup.sh: ignores a local dir equal to the canonical install location" \
    "$SETUP_SH" 'if [ "$_RESOLVED_LOCAL" = "$_CANON_LLAMA_CPP_DIR" ]; then'
assert_contains \
    "setup.ps1: ignores a local dir equal to the canonical install location" \
    "$SETUP_PS1" '$LocalIsCanonical = ($ResolvedLocal -eq $LlamaCppDir)'
assert_contains \
    "setup.ps1: the canonical check actually gates the no-op branch" \
    "$SETUP_PS1" 'if ($LocalIsCanonical) {'

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
