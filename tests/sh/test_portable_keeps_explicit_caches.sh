#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: portable mode defaults cache locations, it does not seize them.
#
# storage_roots._setup_cache_env() has always yielded to a non-blank explicit value, and main's
# own _configure_uv_cache prints "preserving custom UV_CACHE_DIR" for one. The portable shell
# paths did not: _export_portable_roots, the generated bin/unsloth shim and share/studio.conf
# all exported UV_CACHE_DIR, PIP_CACHE_DIR, NPM_CONFIG_CACHE, BUN_INSTALL_CACHE_DIR and
# CUDA_CACHE_PATH unconditionally. The same install therefore honoured a user's own multi-GB
# wheel cache or silently refilled a new one depending on which entry point they used.
#
# Identity (UNSLOTH_HOME, UNSLOTH_PORTABLE, UNSLOTH_STUDIO_HOME) stays forced, and so does
# UV_NO_MODIFY_PATH, which is a promise not to write outside the root rather than a location.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

_CACHES="UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR UV_PYTHON_BIN_DIR \
UV_INSTALL_DIR NPM_CONFIG_CACHE BUN_INSTALL_CACHE_DIR CUDA_CACHE_PATH PIP_CACHE_DIR"

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# ── 1. _export_portable_roots, run for real ──────────────────────────
blockT="$(grep '^_trim_ws() ' "$INSTALL")"
blockB="$(awk '/^_resolve_studio_destinations\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"
blockE="$(awk '/^_export_portable_roots\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"
case "$blockE" in *'UV_CACHE_DIR'*) : ;; *) echo "FAIL: blockE extraction broke"; exit 1 ;; esac
case "$blockE" in *'_epr_default'*) : ;; *) echo "FAIL: blockE no longer defaults its caches"; exit 1 ;; esac

SNIP='set -e
substep() { :; }
'"$blockT"'
'"$blockB"'
'"$blockE"'
_PORTABLE_MODE=true
_PORTABLE_FLAT=false
_UNSLOTH_ROOT="$FIXTURE_ROOT"
_resolve_studio_destinations
_export_portable_roots
for v in '"$_CACHES"' UNSLOTH_HOME UNSLOTH_STUDIO_HOME UV_NO_MODIFY_PATH; do
    eval "printf \"%s=%s\n\" \"\$v\" \"\${$v:-}\""
done'

R="$T/root"
mkdir -p "$T/home" "$T/mine"
explicit=""
for v in $_CACHES; do explicit="$explicit $v=$T/mine/$v"; done

# shellcheck disable=SC2086
out="$(env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$R" \
    $explicit sh -c "$SNIP" _ 2>/dev/null)"

kept=0; taken=""
for v in $_CACHES; do
    got="$(printf '%s\n' "$out" | sed -n "s/^$v=//p")"
    if [ "$got" = "$T/mine/$v" ]; then kept=$((kept + 1)); else taken="$taken $v"; fi
done
check "the installer keeps every explicit cache" "10 " "$kept $taken"
check "identity is still forced" "$R" "$(printf '%s\n' "$out" | sed -n 's/^UNSLOTH_HOME=//p')"
check "UV_NO_MODIFY_PATH is still forced" "1" \
    "$(printf '%s\n' "$out" | sed -n 's/^UV_NO_MODIFY_PATH=//p')"

# With nothing set, the caches must still land under the root: the default is the whole point.
out2="$(env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$R" \
    sh -c "$SNIP" _ 2>/dev/null)"
outside=""
for v in $_CACHES; do
    got="$(printf '%s\n' "$out2" | sed -n "s/^$v=//p")"
    case "$got" in "$R"/*) ;; *) outside="$outside $v:$got" ;; esac
done
check "with nothing set they all default under the root" "" "$outside"

# A blank inherited value counts as unset, as it does in the resolver.
out3="$(env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$R" \
    UV_CACHE_DIR="" PIP_CACHE_DIR="   " sh -c "$SNIP" _ 2>/dev/null)"
check "a blank UV_CACHE_DIR is treated as unset" "$R/cache/uv" \
    "$(printf '%s\n' "$out3" | sed -n 's/^UV_CACHE_DIR=//p')"
check "a whitespace PIP_CACHE_DIR is treated as unset" "$R/cache/pip" \
    "$(printf '%s\n' "$out3" | sed -n 's/^PIP_CACHE_DIR=//p')"

# ── 2. the generated shim, executed ──────────────────────────────────
# Built from install.sh's own writer lines so a change to them shows up here.
shim_lines="$(awk '/^    if \{$/ {g=1} g {print} g && /_shim_tmp" 2>\/dev\/null/ {exit}' "$INSTALL")"
case "$shim_lines" in
    *'UV_CACHE_DIR'*) : ;;
    *) echo "FAIL: shim writer extraction broke"; exit 1 ;;
esac
# The leading quote matters: the guarded form starts the emitted line with `[ -n ...`, so only
# an unconditional writer produces a line beginning `"export UV_CACHE_DIR=`.
case "$shim_lines" in
    *'"export UV_CACHE_DIR='*)
        echo "FAIL: the shim exports UV_CACHE_DIR unconditionally again"; exit 1 ;;
    *) : ;;
esac
case "$shim_lines" in
    *'"export UV_NO_MODIFY_PATH=1"'*) : ;;
    *) echo "FAIL: the shim stopped forcing UV_NO_MODIFY_PATH"; exit 1 ;;
esac

S="$T/shimroot"
mkdir -p "$S/bin" "$S/studio/unsloth_studio/bin"
{
    printf '%s\n' "#!/bin/sh"
    printf '%s\n' "export UNSLOTH_HOME='$S'"
    printf '%s\n' "export UNSLOTH_PORTABLE=1"
    for v in $_CACHES; do
        printf '[ -n "${%s:-}" ] || export %s=%s\n' "$v" "$v" "'$S/cache/$v'"
    done
    printf '%s\n' "exec '$S/studio/unsloth_studio/bin/unsloth' \"\$@\""
} > "$S/bin/unsloth"
chmod +x "$S/bin/unsloth"
{
    printf '%s\n' '#!/bin/sh'
    printf '%s\n' 'for v in '"$_CACHES"'; do eval "printf \"%s=%s\n\" \"\$v\" \"\${$v:-}\""; done'
} > "$S/studio/unsloth_studio/bin/unsloth"
chmod +x "$S/studio/unsloth_studio/bin/unsloth"

# shellcheck disable=SC2086
sout="$(env -i HOME="$T/home" PATH="$PATH" $explicit "$S/bin/unsloth" studio 2>/dev/null)"
skept=0; staken=""
for v in $_CACHES; do
    got="$(printf '%s\n' "$sout" | sed -n "s/^$v=//p")"
    if [ "$got" = "$T/mine/$v" ]; then skept=$((skept + 1)); else staken="$staken $v"; fi
done
check "the shim keeps every explicit cache" "10 " "$skept $staken"

sout2="$(env -i HOME="$T/home" PATH="$PATH" "$S/bin/unsloth" studio 2>/dev/null)"
check "and still defaults them under the root" "$S/cache/UV_CACHE_DIR" \
    "$(printf '%s\n' "$sout2" | sed -n 's/^UV_CACHE_DIR=//p')"

# ── 3. share/studio.conf, sourced ────────────────────────────────────
conf_lines="$(sed -n "/UNSLOTH_EXE='\$_css_quoted_exe'/,/studio\.conf\"\$/p" "$INSTALL")"
case "$conf_lines" in *'UV_CACHE_DIR'*) : ;; *) echo "FAIL: conf writer extraction broke"; exit 1 ;; esac
case "$conf_lines" in
    *'"export UV_CACHE_DIR='*)
        echo "FAIL: studio.conf exports UV_CACHE_DIR unconditionally again"; exit 1 ;;
    *) : ;;
esac

# ── 4. the one relocation that costs a download gets said out loud ──
# Portable mode moves HF_HUB_CACHE under the root, so an existing shared model cache stays on
# disk but stops being read and every model is fetched again. Nothing moves it; the user is
# told. Silent when they named HF_HOME/HF_HUB_CACHE themselves (the resolver leaves those
# alone) and silent on a machine with no cache to strand.
# Its own snippet: SNIP stubs substep to a no-op, which would swallow the very line under test.
SNIP_LOUD="$(printf '%s\n' "$SNIP" | sed 's/^substep() { :; }$/substep() { printf "SUBSTEP %s\\n" "$1"; }/')"
case "$SNIP_LOUD" in
    *'SUBSTEP'*) : ;;
    *) echo "FAIL: could not make substep audible"; exit 1 ;;
esac
notice() { # extra env assignments
    # shellcheck disable=SC2086
    env -i HOME="$T/hfhome" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$R" $1 \
        sh -c "$SNIP_LOUD" _ 2>&1 | grep -c 'moves the model cache under the root' | tr -d ' '
}
mkdir -p "$T/hfhome/.cache/huggingface/hub/models--x"
: > "$T/hfhome/.cache/huggingface/hub/models--x/blob"
check "a warm shared model cache is reported" "1" "$(notice "")"
check "silent when HF_HUB_CACHE was named"    "0" "$(notice "HF_HUB_CACHE=$T/mine/hf")"
check "silent when HF_HOME was named"         "0" "$(notice "HF_HOME=$T/mine/hfhome")"
rm -rf "$T/hfhome/.cache"
check "silent on a machine with no model cache" "0" "$(notice "")"

# XDG_CACHE_HOME moves the shared cache, and huggingface_hub follows it: HF_HUB_CACHE
# defaults to HF_HOME/hub and HF_HOME to $XDG_CACHE_HOME/huggingface. The probe used to look
# under $HOME/.cache regardless, so the users whose cache is somewhere unusual were the ones
# who never heard it was about to be stranded.
mkdir -p "$T/xdg/huggingface/hub/models--x"
: > "$T/xdg/huggingface/hub/models--x/blob"
check "a cache under XDG_CACHE_HOME is reported too" "1" \
    "$(notice "XDG_CACHE_HOME=$T/xdg")"
# ...and it is that directory the notice names, not the $HOME one.
xdg_says() {
    env -i HOME="$T/hfhome" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$R" \
        XDG_CACHE_HOME="$T/xdg" sh -c "$SNIP_LOUD" _ 2>&1 \
        | grep -q "$T/xdg/huggingface/hub" && printf yes || printf no
}
# Named on more than one line (the stranding line and the HF_HUB_CACHE suggestion), so this
# asks whether it appears at all rather than counting.
check "and the notice names the XDG path" "yes" "$(xdg_says)"

# With XDG_CACHE_HOME set, a cache sitting at $HOME/.cache is NOT what huggingface_hub reads,
# so warning about it would send the user to delete the wrong directory.
mkdir -p "$T/hfhome/.cache/huggingface/hub/models--y"
: > "$T/hfhome/.cache/huggingface/hub/models--y/blob"
rm -rf "$T/xdg"
check "silent about \$HOME/.cache when XDG points elsewhere" "0" \
    "$(notice "XDG_CACHE_HOME=$T/xdg")"

# A relative XDG_CACHE_HOME is invalid per the spec, so the probe falls back to $HOME/.cache
# rather than resolving against whatever directory the installer was launched from.
check "a relative XDG_CACHE_HOME falls back to \$HOME" "1" \
    "$(notice "XDG_CACHE_HOME=relative/path")"
rm -rf "$T/hfhome/.cache"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
