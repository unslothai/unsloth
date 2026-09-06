#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Exercise the real install-time uv cache selector under POSIX sh and bash.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

HELPERS=$(awk '
    /^_configure_uv_cache\(\) \{/ { grab = 1 }
    /^_prepare_studio_uv_cache_for_launch\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$INSTALL_SH")
for _helper in _configure_uv_cache _prepare_studio_uv_cache_for_launch; do
    if ! printf '%s\n' "$HELPERS" | grep -q "^${_helper}() {"; then
        echo "  FAIL: could not extract $_helper from install.sh"
        exit 1
    fi
done

WORK=$(mktemp -d)
PROBE="$WORK/probe.sh"
UV_STUB_DIR="$WORK/bin"
export UV_STUB_DIR
trap 'rm -rf "$WORK"' EXIT INT TERM
mkdir -p "$UV_STUB_DIR"
cat > "$UV_STUB_DIR/uv" <<'UV'
#!/bin/sh
[ "$1" = cache ] && [ "$2" = dir ] || exit 2
printf '%s\n' "$TEST_UV_EFFECTIVE_CACHE"
UV
chmod +x "$UV_STUB_DIR/uv"
printf '%s\n' "$HELPERS" > "$PROBE"
cat >> "$PROBE" <<'PROBE'
step() { printf 'message=%s\n' "$2"; }
C_WARN=""
case "$1" in
    unset) unset UV_CACHE_DIR ;;
    value) UV_CACHE_DIR=$2 ;;
    *) exit 2 ;;
esac
_ISOLATE_UV_CACHE=$3
HOME=$4
case "$5" in
    unset) unset XDG_CACHE_HOME ;;
    value) XDG_CACHE_HOME=$6 ;;
    *) exit 2 ;;
esac
STUDIO_HOME=$7

TEST_UV_EFFECTIVE_CACHE=$9
export TEST_UV_EFFECTIVE_CACHE
PATH="$UV_STUB_DIR:$PATH"
export PATH
_configure_uv_cache
_child=$($8 -c 'printf "%s" "${UV_CACHE_DIR+x}:$UV_CACHE_DIR"')
printf 'value=%s\nmode=%s\nchild=%s\n' "$UV_CACHE_DIR" "$_UV_CACHE_MODE" "$_child"
_prepare_studio_uv_cache_for_launch
printf 'launch=%s\n' "$UV_CACHE_DIR"
PROBE

run_case() { # shell, label, state, input, isolate, home, xdg-state, xdg, root, effective, value, mode, message, launch
    _shell=$1
    _label=$2
    _state=$3
    _input=$4
    _isolate=$5
    _home=$6
    _xdg_state=$7
    _xdg=$8
    _root=$9
    shift 9
    _effective=$1
    _expected=$2
    _mode=$3
    _message=$4
    _launch=$5
    _actual=$($_shell "$PROBE" "$_state" "$_input" "$_isolate" "$_home" "$_xdg_state" "$_xdg" "$_root" "$_shell" "$_effective")
    _wanted=$(printf 'message=%s\nvalue=%s\nmode=%s\nchild=x:%s\nlaunch=%s' \
        "$_message" "$_expected" "$_mode" "$_expected" "$_launch")
    if [ "$_actual" = "$_wanted" ]; then
        ok "$_shell: $_label"
    else
        bad "$_shell: $_label (expected [$_wanted], got [$_actual])"
    fi
}

echo "=== test_install_uv_cache_root ==="
for shell in sh bash; do
    command -v "$shell" >/dev/null 2>&1 || continue
    CASE="$WORK/$shell case"
    HOME_DIR="$CASE/home with spaces"
    XDG_DIR="$CASE/xdg with spaces"
    ROOT="$CASE/studio root"
    STUDIO_CACHE="$ROOT/cache/uv"
    HOME_CACHE="$HOME_DIR/.cache/uv"
    XDG_CACHE="$XDG_DIR/uv"
    OVERRIDE="$CASE/caller cache/uv artifacts"
    mkdir -p "$HOME_DIR" "$ROOT"

    run_case "$shell" "missing default selects Studio" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$HOME_CACHE" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"

    mkdir -p "$HOME_CACHE/CACHEDIR.TAG/inside"
    : > "$HOME_CACHE/CACHEDIR.TAG/inside/payload"
    : > "$HOME_CACHE/.gitignore"
    run_case "$shell" "marker-only default stays Studio and is not traversed" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$HOME_CACHE" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"
    if [ -f "$HOME_CACHE/CACHEDIR.TAG/inside/payload" ] && [ -f "$HOME_CACHE/.gitignore" ]; then
        ok "$shell: marker-only probe is non-destructive"
    else
        bad "$shell: marker-only probe modified the shared cache"
    fi

    mkdir -p "$HOME_CACHE/sdists-v9" "$HOME_CACHE/interpreter-v4/key"
    : > "$HOME_CACHE/sdists-v9/.git"
    : > "$HOME_CACHE/sdists-v9/.gitignore"
    : > "$HOME_CACHE/interpreter-v4/key/metadata.msgpack"
    : > "$HOME_CACHE/.lock"
    run_case "$shell" "uv venv scaffolding stays Studio-owned" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$HOME_CACHE" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"

    mkdir -p "$HOME_CACHE/archive-v0/package"
    : > "$HOME_CACHE/archive-v0/package/payload.py"
    run_case "$shell" "populated HOME cache is reused" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$HOME_CACHE" "$HOME_CACHE" shared \
        "reusing existing shared cache ($HOME_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    run_case "$shell" "blank override still reuses populated default" value "   " false \
        "$HOME_DIR" unset "" "$ROOT" "$HOME_CACHE" "$HOME_CACHE" shared \
        "reusing existing shared cache ($HOME_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    mkdir -p "$XDG_CACHE/wheels-v5/package"
    : > "$XDG_CACHE/wheels-v5/package/cached.whl"
    run_case "$shell" "XDG default wins over HOME" unset "" false \
        "$HOME_DIR" value "$XDG_DIR" "$ROOT" "$XDG_CACHE" "$XDG_CACHE" shared \
        "reusing existing shared cache ($XDG_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    EMPTY_XDG="$CASE/empty xdg"
    EMPTY_XDG_CACHE="$EMPTY_XDG/uv"
    mkdir -p "$EMPTY_XDG_CACHE"
    run_case "$shell" "empty XDG does not fall back to populated HOME" unset "" false \
        "$HOME_DIR" value "$EMPTY_XDG" "$ROOT" "$EMPTY_XDG_CACHE" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"

    CONFIG_CACHE="$CASE/uv.toml cache"
    mkdir -p "$CONFIG_CACHE/wheels-v5/package"
    : > "$CONFIG_CACHE/wheels-v5/package/torch.whl"
    run_case "$shell" "uv-configured cache is resolved and reused" unset "" false \
        "$HOME_DIR" value "$EMPTY_XDG" "$ROOT" "$CONFIG_CACHE" "$CONFIG_CACHE" shared \
        "reusing existing shared cache ($CONFIG_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    run_case "$shell" "custom override is exact" value "$OVERRIDE" false \
        "$HOME_DIR" value "$XDG_DIR" "$ROOT" "$XDG_CACHE" "$OVERRIDE" custom \
        "preserving custom UV_CACHE_DIR ($OVERRIDE)" "$OVERRIDE"

    run_case "$shell" "custom override wins over isolation" value "$OVERRIDE" true \
        "$HOME_DIR" value "$XDG_DIR" "$ROOT" "$XDG_CACHE" "$OVERRIDE" custom \
        "preserving custom UV_CACHE_DIR ($OVERRIDE)" "$OVERRIDE"

    run_case "$shell" "forced isolation uses Studio despite populated default" unset "" true \
        "$HOME_DIR" value "$XDG_DIR" "$ROOT" "$XDG_CACHE" "$STUDIO_CACHE" isolated \
        "forced Studio cache isolation ($STUDIO_CACHE); already-cached packages may download again" "$STUDIO_CACHE"

    run_case "$shell" "unresolvable default safely selects Studio" unset "" false \
        "" unset "" "$ROOT" "" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"

    # wheels-v* is .msgpack/.http only on uv 0.10: one `--dry-run` leaves a file.
    META_CACHE="$CASE/metadata only/uv"
    mkdir -p "$META_CACHE/wheels-v6/pypi/torch" "$META_CACHE/simple-v20/pypi" \
        "$META_CACHE/sdists-v9/pypi/pkg"
    : > "$META_CACHE/wheels-v6/pypi/torch/2.11.0-cp313-none-any.msgpack"
    : > "$META_CACHE/wheels-v6/pypi/torch/2.11.0.http"
    : > "$META_CACHE/simple-v20/pypi/torch.rkyv"
    : > "$META_CACHE/sdists-v9/pypi/pkg/revision.rev"
    : > "$META_CACHE/sdists-v9/pypi/pkg/download.lock"
    run_case "$shell" "metadata-only default is not a warm cache" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$META_CACHE" "$STUDIO_CACHE" studio \
        "using new Studio-owned cache ($STUDIO_CACHE)" "$STUDIO_CACHE"

    mkdir -p "$META_CACHE/archive-v0/hash/torch"
    : > "$META_CACHE/archive-v0/hash/torch/_C.so"
    run_case "$shell" "package bytes beside metadata do count" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$META_CACHE" "$META_CACHE" shared \
        "reusing existing shared cache ($META_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    # builds-v* is what modern uv calls the old built-wheels-* bucket.
    BUILDS_CACHE="$CASE/builds cache/uv"
    mkdir -p "$BUILDS_CACHE/builds-v0/pkg"
    : > "$BUILDS_CACHE/builds-v0/pkg/module.py"
    run_case "$shell" "builds-v0 counts as package data" unset "" false \
        "$HOME_DIR" unset "" "$ROOT" "$BUILDS_CACHE" "$BUILDS_CACHE" shared \
        "reusing existing shared cache ($BUILDS_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
        "$STUDIO_CACHE"

    # `find` needs -L to descend a symlink; Get-ChildItem -Recurse already does.
    LINK_CACHE="$CASE/symlinked bucket/uv"
    LINK_TARGET="$CASE/symlinked bucket/elsewhere"
    mkdir -p "$LINK_CACHE" "$LINK_TARGET/pkg"
    : > "$LINK_TARGET/pkg/payload.so"
    if ln -s "$LINK_TARGET" "$LINK_CACHE/archive-v0" 2>/dev/null; then
        run_case "$shell" "symlinked bucket is still inspected" unset "" false \
            "$HOME_DIR" unset "" "$ROOT" "$LINK_CACHE" "$LINK_CACHE" shared \
            "reusing existing shared cache ($LINK_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
            "$STUDIO_CACHE"
    fi

    # Unreadable is not empty: falling back is right, doing it silently is not.
    DENIED_CACHE="$CASE/denied bucket/uv"
    mkdir -p "$DENIED_CACHE/archive-v0/pkg"
    : > "$DENIED_CACHE/archive-v0/pkg/payload.so"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 000 "$DENIED_CACHE/archive-v0" 2>/dev/null; then
        run_case "$shell" "unreadable bucket says why it fell back" unset "" false \
            "$HOME_DIR" unset "" "$ROOT" "$DENIED_CACHE" "$STUDIO_CACHE" studio \
            "using new Studio-owned cache ($STUDIO_CACHE); part of $DENIED_CACHE could not be read, so cached packages may download again" \
            "$STUDIO_CACHE"
        chmod 755 "$DENIED_CACHE/archive-v0" 2>/dev/null || true
    fi

    DEEP_CACHE="$CASE/denied leaf/uv"
    mkdir -p "$DEEP_CACHE/archive-v0/visible" "$DEEP_CACHE/archive-v0/aaa hidden"
    : > "$DEEP_CACHE/archive-v0/visible/payload.so"
    : > "$DEEP_CACHE/archive-v0/aaa hidden/other.so"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 000 "$DEEP_CACHE/archive-v0/aaa hidden" 2>/dev/null; then
        run_case "$shell" "unreadable leaf keeps a warm cache warm" unset "" false \
            "$HOME_DIR" unset "" "$ROOT" "$DEEP_CACHE" "$DEEP_CACHE" shared \
            "reusing existing shared cache ($DEEP_CACHE) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate" \
            "$STUDIO_CACHE"
        chmod 755 "$DEEP_CACHE/archive-v0/aaa hidden" 2>/dev/null || true
    fi
done

_resolve_line=$(grep -n '^_resolve_studio_destinations$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_configure_line=$(grep -n '^_configure_uv_cache$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_uv_line=$(grep -n '^# ── Install uv ──$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_venv_line=$(grep -n '^# ── Create venv ' "$INSTALL_SH" | head -n1 | cut -d: -f1)
if [ -n "$_resolve_line" ] && [ -n "$_configure_line" ] && [ -n "$_uv_line" ] && [ -n "$_venv_line" ] \
   && [ "$_resolve_line" -lt "$_uv_line" ] && [ "$_uv_line" -lt "$_configure_line" ] \
   && [ "$_configure_line" -lt "$_venv_line" ]; then
    ok "helper runs after uv setup and before venv setup"
else
    bad "helper ordering (resolve=$_resolve_line uv=$_uv_line configure=$_configure_line venv=$_venv_line)"
fi

for _required in \
    '_ISOLATE_UV_CACHE=false' \
    '--isolated-uv-cache) _ISOLATE_UV_CACHE=true' \
    'UNSLOTH_ISOLATE_UV_CACHE' \
    'export UNSLOTH_ISOLATE_UV_CACHE=1' \
    'unset UV_CACHE_DIR' \
    '_prepare_studio_uv_cache_for_launch'; do
    if grep -Fq -- "$_required" "$INSTALL_SH"; then
        ok "source contract: $_required"
    else
        bad "missing source contract: $_required"
    fi
done

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
