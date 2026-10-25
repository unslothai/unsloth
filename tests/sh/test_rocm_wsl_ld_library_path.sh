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
_fns="$(sed -n '/^_rocm_wsl_dropin_stale() {$/,/^}$/p; /^_repair_rocm_wsl_bashrc() {$/,/^}$/p; /^_persist_rocm_wsl_dropin() {$/,/^}$/p' "$INSTALL_SH" \
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

# An older install left the trailing-colon form: a re-run must fix that line and keep the rest.
printf '# >>> Unsloth ROCm-on-WSL (gfx1151) >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\nexport HSA_CUSTOM=1\n# <<< Unsloth ROCm-on-WSL (gfx1151) <<<\n' "$_tmp/rocm" > "$_dropin"
_run_writer
assert_not_contains "stale drop-in line fixed on re-run" "$(cat "$_dropin")" 'lib:${LD_LIBRARY_PATH:-}'
assert_contains "other drop-in lines kept" "$(cat "$_dropin")" "export HSA_CUSTOM=1"
_ld="$(_ld_after_source "$_dropin")"
assert_eq "fixed drop-in with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"
cp "$_dropin" "$_tmp/fixed_first"
_run_writer
if cmp -s "$_tmp/fixed_first" "$_dropin"; then ok "fixed drop-in unchanged by another run"; else bad "another run changed the fixed drop-in"; fi

# A drop-in an operator edited (no stale form) is left alone.
printf 'export LD_LIBRARY_PATH=/custom\n' > "$_dropin"
_run_writer
assert_eq "current drop-in not overwritten" "export LD_LIBRARY_PATH=/custom" "$(cat "$_dropin")"

echo "=== non-root re-run: passwordless sudo, an explained prompt only on a terminal ==="
# Fake sudo: logs its args; without -n it stands in for a password prompt (accepted when
# FAKE_SUDO_PW=1), with -n it refuses (no cached credentials) unless FAKE_SUDO_OK=1 (a NOPASSWD rule).
mkdir -p "$_tmp/fakebin"
cat > "$_tmp/fakebin/sudo" <<'FAKE'
#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_SUDO_LOG"
if [ "$1" != "-n" ]; then
    echo "PROMPTED" >> "$FAKE_SUDO_LOG"
    [ "${FAKE_SUDO_PW:-0}" = "1" ] && exec "$@"
    exit 1
fi
shift
[ "${FAKE_SUDO_OK:-0}" = "1" ] || exit 1
exec "$@"
FAKE
chmod +x "$_tmp/fakebin/sudo"
_stale_dropin() {
    printf 'export LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n' "$_tmp/rocm" > "$_dropin"
}
# $1 = NOPASSWD rule, $2 = a terminal to ask on, $3 = the password is typed.
_run_writer_nonroot() {
    env -i PATH="$_tmp/fakebin:$PATH" FAKE_SUDO_LOG="$_tmp/sudo.log" FAKE_SUDO_OK="$1" FAKE_TTY="${2:-0}" FAKE_SUDO_PW="${3:-0}" TAURI_MODE="${4:-false}" \
        bash -c 'id() { echo 1000; }; substep() { printf "%s\n" "$1"; }; _can_read_tty() { [ "$FAKE_TTY" = "1" ]; }; eval "$1"; _persist_rocm_wsl_dropin' _ "$_fns"
}
_stale_dropin; : > "$_tmp/sudo.log"
_out="$(_run_writer_nonroot 0)"
assert_not_contains "no terminal: no interactive sudo attempted" "$(cat "$_tmp/sudo.log")" "PROMPTED"
assert_contains "repair used sudo -n" "$(cat "$_tmp/sudo.log")" "-n sed"
assert_contains "hint printed when sudo -n is refused" "$_out" "re-run the installer"
_stale_dropin; : > "$_tmp/sudo.log"
_out="$(_run_writer_nonroot 1 1)"
assert_not_contains "passwordless sudo repairs the stale drop-in" "$(cat "$_dropin")" 'lib:${LD_LIBRARY_PATH:-}'
assert_not_contains "passwordless sudo: no prompt even on a terminal" "$(cat "$_tmp/sudo.log")" "PROMPTED"
assert_eq "no hint when the repair ran" "" "$_out"
_stale_dropin; : > "$_tmp/sudo.log"
_out="$(_run_writer_nonroot 0 1 1)"
assert_not_contains "terminal: the explained prompt repairs the drop-in" "$(cat "$_dropin")" 'lib:${LD_LIBRARY_PATH:-}'
assert_contains "terminal: says why before sudo asks" "$_out" "sudo may ask for your password"
assert_not_contains "terminal: no hint after the repair" "$_out" "re-run the installer"
_out="$(_run_writer_nonroot 0 1 1)"
assert_eq "repaired drop-in: a re-run asks nothing" "" "$_out"
_stale_dropin; : > "$_tmp/sudo.log"
_out="$(_run_writer_nonroot 0 1 0)"
assert_contains "terminal, password refused: hint printed" "$_out" "re-run the installer"
# Desktop (--tauri) runs the installer in a background process group: an inherited terminal cannot answer.
_stale_dropin; : > "$_tmp/sudo.log"
_out="$(_run_writer_nonroot 0 1 1 true)"
assert_not_contains "--tauri with a terminal: no interactive sudo" "$(cat "$_tmp/sudo.log")" "PROMPTED"
assert_contains "--tauri with a terminal: hint printed" "$_out" "re-run the installer"

echo "=== install.sh repairs an old helper's ~/.bashrc block ==="
mkdir -p "$_tmp/dotfiles" "$_tmp/home2"
printf '# user line\n# >>> Unsloth ROCm-on-WSL >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n# <<< Unsloth ROCm-on-WSL <<<\n' "$_tmp/rocm" > "$_tmp/dotfiles/bashrc"
ln -s "$_tmp/dotfiles/bashrc" "$_tmp/home2/.bashrc"
env -i PATH="$PATH" HOME="$_tmp/home2" bash -c 'eval "$1"; _repair_rocm_wsl_bashrc' _ "$_fns"
if [ -L "$_tmp/home2/.bashrc" ]; then ok "symlinked ~/.bashrc stays a link"; else bad "~/.bashrc link replaced by a file"; fi
assert_not_contains "old block line repaired" "$(cat "$_tmp/dotfiles/bashrc")" 'lib:${LD_LIBRARY_PATH:-}'
_ld="$(_ld_after_source "$_tmp/home2/.bashrc")"
assert_eq "repaired block with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"
cp "$_tmp/dotfiles/bashrc" "$_tmp/repaired_first"
env -i PATH="$PATH" HOME="$_tmp/home2" bash -c 'eval "$1"; _repair_rocm_wsl_bashrc' _ "$_fns"
if cmp -s "$_tmp/repaired_first" "$_tmp/dotfiles/bashrc"; then ok "repair is idempotent"; else bad "second repair changed ~/.bashrc"; fi

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

# An older helper's ~/.bashrc block is replaced through a symlink, the user's own lines kept.
printf '# user line\n# >>> Unsloth ROCm-on-WSL >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n# <<< Unsloth ROCm-on-WSL <<<\n# after\n' "$_tmp/rocm" > "$_tmp/dotfiles/helper_bashrc"
rm -f "$_tmp/home/.bashrc"; ln -s "$_tmp/dotfiles/helper_bashrc" "$_tmp/home/.bashrc"
_run_helper_step4 >/dev/null
if [ -L "$_tmp/home/.bashrc" ]; then ok "helper keeps a symlinked ~/.bashrc a link"; else bad "helper replaced the ~/.bashrc link"; fi
assert_not_contains "stale ~/.bashrc block removed" "$(cat "$_tmp/home/.bashrc")" 'lib:${LD_LIBRARY_PATH:-}'
assert_contains "user lines kept" "$(cat "$_tmp/home/.bashrc")" "# after"
_ld="$(_ld_after_source "$_tmp/home/.bashrc")"
assert_eq "repaired ~/.bashrc with LD_LIBRARY_PATH unset" "$_tmp/rocm/lib" "$_ld"

# A ~/.bashrc it cannot edit must not abort the helper after ROCm is installed.
if [ "$(id -u)" != "0" ]; then
    mkdir -p "$_tmp/ro"
    printf '# >>> Unsloth ROCm-on-WSL >>>\nexport LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n# <<< Unsloth ROCm-on-WSL <<<\n' "$_tmp/rocm" > "$_tmp/ro/.bashrc"
    chmod a-w "$_tmp/ro" "$_tmp/ro/.bashrc"
    if env -i PATH="$PATH" HOME="$_tmp/ro" ROCM_DIR="$_tmp/rocm" SUDO="" \
        bash -euo pipefail -c 'say() { :; }; eval "$1"' _ "$_step4" >/dev/null 2>&1; then
        ok "read-only ~/.bashrc does not abort the helper"
    else
        bad "read-only ~/.bashrc aborted the helper"
    fi
    chmod u+w "$_tmp/ro" "$_tmp/ro/.bashrc"
fi

summary
