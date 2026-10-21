#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The ROCm-on-WSL env drop-in must not add an empty LD_LIBRARY_PATH element (the cwd) when the
# variable starts unset, a re-run must repair an older drop-in, and a second run changes nothing.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
HELPER_SH="$SCRIPT_DIR/../../scripts/install_rocm_wsl_strixhalo.sh"

_tmp="$(mktemp -d)"
trap 'rm -rf "$_tmp"' EXIT
mkdir -p "$_tmp/rocm/lib" "$_tmp/profile.d" "$_tmp/home"
: > "$_tmp/rocm/lib/librocdxg.so"
_dropin="$_tmp/profile.d/unsloth-rocm-wsl.sh"

# Every function the writer needs, with the system paths pointed into the sandbox.
_fns="$(sed -n '/^_rocm_wsl_dropin_stale() {$/,/^}$/p; /^_persist_rocm_wsl_dropin() {$/,/^}$/p' "$INSTALL_SH" \
    | sed "s|/etc/profile.d/unsloth-rocm-wsl.sh|$_dropin|g; s|/opt/rocm|$_tmp/rocm|g")"
case "$_fns" in
    *"_persist_rocm_wsl_dropin()"*) ;;
    *) bad "could not extract _persist_rocm_wsl_dropin"; summary ;;
esac

_run_writer() {
    env -i PATH="$PATH" bash -c 'id() { echo 0; }; eval "$1"; _persist_rocm_wsl_dropin' _ "$_fns"
}

_ld_after_source() {
    # A login shell with LD_LIBRARY_PATH unset, as /etc/profile.d sees it.
    env -i PATH=/usr/bin:/bin sh -c '. "$1"; printf "%s" "$LD_LIBRARY_PATH"' _ "$1"
}

_no_empty_element() {
    case ":$1:" in
        *::*) return 1 ;;
    esac
    return 0
}

echo "=== install.sh drop-in ==="
_run_writer
[ -r "$_dropin" ] || bad "drop-in not written"
_ld="$(_ld_after_source "$_dropin")"
assert_eq "drop-in with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"
if _no_empty_element "$_ld"; then ok "no empty LD_LIBRARY_PATH element"; else bad "empty element in '$_ld'"; fi
_ld="$(env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/x sh -c '. "$1"; printf "%s" "$LD_LIBRARY_PATH"' _ "$_dropin")"
assert_eq "drop-in keeps an existing LD_LIBRARY_PATH after ROCm" "$_tmp/rocm/lib:/x" "$_ld"

cp "$_dropin" "$_tmp/first"
_run_writer
if cmp -s "$_tmp/first" "$_dropin"; then ok "second run is byte-identical"; else bad "second run changed the drop-in"; fi

# An older install left the trailing-colon form: a re-run must rewrite it.
printf '# >>> Unsloth ROCm-on-WSL (gfx1151) >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n# <<< Unsloth ROCm-on-WSL (gfx1151) <<<\n' "$_tmp/rocm" > "$_dropin"
_run_writer
if cmp -s "$_tmp/first" "$_dropin"; then ok "stale drop-in rewritten on re-run"; else bad "stale drop-in kept on re-run"; fi

# A drop-in an operator edited (no stale form) is left alone.
printf 'export LD_LIBRARY_PATH=/custom\n' > "$_dropin"
_run_writer
assert_eq "current drop-in not overwritten" "export LD_LIBRARY_PATH=/custom" "$(cat "$_dropin")"

echo "=== helper env block ==="
_step4="$(sed -n '/^# ── Step 4: persist environment/,/^# ── Step 5:/p' "$HELPER_SH" \
    | sed "s|/etc/profile.d/unsloth-rocm-wsl.sh|$_tmp/helper_env.sh|g")"
[ -n "$_step4" ] || bad "could not extract the helper's Step 4"
_run_helper_step4() {
    env -i PATH="$PATH" HOME="$_tmp/home" ROCM_DIR="$_tmp/rocm" SUDO="" \
        bash -c 'say() { :; }; eval "$1"; printf "%s" "$LD_LIBRARY_PATH"' _ "$_step4"
}
printf '# user line\n' > "$_tmp/home/.bashrc"
_proc_ld="$(_run_helper_step4)"
assert_eq "helper's own process LD_LIBRARY_PATH" "$_tmp/rocm/lib" "$_proc_ld"
_ld="$(_ld_after_source "$_tmp/helper_env.sh")"
assert_eq "helper drop-in with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"
_ld="$(_ld_after_source "$_tmp/home/.bashrc")"
assert_eq "helper ~/.bashrc block with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"
cp "$_tmp/home/.bashrc" "$_tmp/bashrc_first"
_run_helper_step4 >/dev/null
if cmp -s "$_tmp/bashrc_first" "$_tmp/home/.bashrc"; then ok "helper second run leaves ~/.bashrc byte-identical"; else bad "helper second run changed ~/.bashrc"; fi

# An older helper's ~/.bashrc block is replaced, the user's own lines kept.
printf '# user line\n# >>> Unsloth ROCm-on-WSL >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n# <<< Unsloth ROCm-on-WSL <<<\n# after\n' "$_tmp/rocm" > "$_tmp/home/.bashrc"
_run_helper_step4 >/dev/null
assert_not_contains "stale ~/.bashrc block removed" "$(cat "$_tmp/home/.bashrc")" 'lib:${LD_LIBRARY_PATH:-}'
assert_contains "user lines kept" "$(cat "$_tmp/home/.bashrc")" "# after"
_ld="$(_ld_after_source "$_tmp/home/.bashrc")"
assert_eq "repaired ~/.bashrc with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"

summary
