#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: a bare `bash studio/setup.sh` against a portable install must stay inside
# the root, the way every other entry point already does.
#
# setup.sh recovers the master root from <studio>/.unsloth-master-root so that a standalone
# run finds node/, llama.cpp/ and whisper.cpp/ at the right level. It restored only the bun
# cache alongside it, and the two it did not restore both write outside the root:
#
#   * UV_INSTALL_DIR / UV_NO_MODIFY_PATH -- uv reinstalls itself whenever `command -v uv`
#     misses, which on a portable install is every run, because uv lives at <root>/bin and
#     nothing ever put that on PATH. astral's cascade then ends at $HOME/.local/bin and
#     _setup_persist_uv_path appends a PATH line to ~/.profile, ~/.bashrc, ~/.zshrc and
#     ~/.config/fish/conf.d/unsloth.fish. A binary and a permanent shell edit that both
#     survive the `rm -rf <root>` the portable install advertises.
#   * NPM_CONFIG_CACHE -- bun does not read npm's configuration, and npm runs on paths bun
#     never covers: `npm install -g bun` runs before bun exists on the managed-Node path, the
#     frontend `npm install` is the fallback whenever bun is absent, and the oxc-validator
#     `npm install` sits outside the frontend guard entirely. npm's POSIX default is $HOME/.npm.
#
# install.sh's _export_portable_roots, share/studio.conf, the generated bin/unsloth shim and
# the CLI's _portable_root_env all set these three; setup.sh is the one that did not.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# ── Lift the real code ───────────────────────────────────────────────
blockPIN="$(awk '
    /^# The rest of the portable environment a bare `bash studio\/setup\.sh` did not inherit\./ {grab = 1}
    grab {print}
    grab && /^fi$/ {exit}
' "$SETUP")"
blockTRIM="$(grep '^_setup_trim_ws() ' "$SETUP")"
blockHAS="$(awk '/^_setup_path_has_dir\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}' "$SETUP")"
blockPERSIST="$(awk '/^_setup_persist_uv_path\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}' "$SETUP")"
blockDEST="$(grep -n '' "$SETUP" | sed -n "$(grep -n '_siup_dest="\${UV_INSTALL_DIR' "$SETUP" | head -1 | cut -d: -f1),+2p" | cut -d: -f2-)"

# Self-validating: a stale range must read as a broken test, never as a pass.
case "$blockPIN"     in *'UV_INSTALL_DIR'*)     : ;; *) echo "FAIL: the pin block extraction broke"; exit 1 ;; esac
case "$blockPIN"     in *'NPM_CONFIG_CACHE'*)   : ;; *) echo "FAIL: the pin block lost NPM_CONFIG_CACHE"; exit 1 ;; esac
case "$blockPIN"     in *'UV_NO_MODIFY_PATH'*)  : ;; *) echo "FAIL: the pin block lost UV_NO_MODIFY_PATH"; exit 1 ;; esac
case "$blockTRIM"    in *'_setup_trim_ws'*)     : ;; *) echo "FAIL: _setup_trim_ws extraction broke"; exit 1 ;; esac
case "$blockPERSIST" in *'Added by Unsloth setup'*) : ;; *) echo "FAIL: _setup_persist_uv_path extraction broke"; exit 1 ;; esac
case "$blockDEST"    in *'HOME/.local/bin'*)    : ;; *) echo "FAIL: the uv destination cascade extraction broke"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
R="$T/vol/portable"; mkdir -p "$R/bin"
R="$(CDPATH= cd -P -- "$R" && pwd -P)"

echo
echo "[1] the three values the standalone path now derives"
pin() { # $1 = UNSLOTH_HOME, rest = extra env assignments
    _uh="$1"; shift
    # shellcheck disable=SC2086
    env -i PATH="$PATH" HOME="$T/home" UNSLOTH_HOME="$_uh" $* bash -c '
set -eu
'"$blockTRIM"'
'"$blockPIN"'
printf "%s|%s|%s\n" "${UV_INSTALL_DIR:-}" "${UV_NO_MODIFY_PATH:-}" "${NPM_CONFIG_CACHE:-}"'
}
check "uv installs into the root"        "$R/bin|1|$R/cache/npm" "$(pin "$R")"
check "a normal install sets none of them" "||"                  "$(pin "")"
check "an explicit UV_INSTALL_DIR wins"  "$T/mine|1|$R/cache/npm" "$(pin "$R" UV_INSTALL_DIR=$T/mine)"
check "an explicit NPM_CONFIG_CACHE wins" "$R/bin|1|$T/npm"      "$(pin "$R" NPM_CONFIG_CACHE=$T/npm)"
# Blank counts as unset, matching install.sh's _epr_default and the resolvers.
check "a blank value is treated as unset" "$R/bin|1|$R/cache/npm" "$(pin "$R" UV_INSTALL_DIR= NPM_CONFIG_CACHE=)"

echo
echo "[2] what that changes on disk: uv's destination and the user's shell files"
# The point of the pin is not the variable, it is that nothing lands outside the root. Run the
# real destination cascade and the real startup-file writer against a fixture HOME.
rc_probe() { # $1 = pin on/off
    _H="$T/rc_$1"; rm -rf "$_H"; mkdir -p "$_H"
    : > "$_H/.bashrc"; : > "$_H/.zshrc"; : > "$_H/.profile"
    _pin_src=""
    [ "$1" = on ] && _pin_src="$blockPIN"
    env -i PATH=/usr/bin:/bin HOME="$_H" UNSLOTH_HOME="$R" bash -c '
set -eu
'"$blockTRIM"'
'"$_pin_src"'
'"$blockHAS"'
'"$blockPERSIST"'
'"$blockDEST"'
_SETUP_LOGIN_PATH="/usr/bin:/bin"
_setup_persist_uv_path "$_siup_dest"
printf "%s\n" "$_siup_dest"'
}
dest_off="$(rc_probe off)"
dest_on="$(rc_probe on)"
check "without the pin uv would land outside the root" "$T/rc_off/.local/bin" "$dest_off"
check "with the pin uv lands in the root"              "$R/bin"              "$dest_on"

touched() { # $1 = fixture home
    _n=0
    for _f in "$1/.profile" "$1/.bashrc" "$1/.zshrc" "$1/.config/fish/conf.d/unsloth.fish"; do
        [ -s "$_f" ] && _n=$((_n+1))
    done
    printf '%s' "$_n"
}
# The control: this fixture really does get its startup files rewritten without the fix, so a
# pass below is the pin working rather than the probe never reaching the writer.
check "without the pin the shell startup files are rewritten" "4" "$(touched "$T/rc_off")"
check "with the pin not one of them is touched"               "0" "$(touched "$T/rc_on")"

echo
echo "[3] the pin runs before every consumer in setup.sh"
# Pure ordering was the whole defect, so the ordering is pinned too. npm in particular has
# three call sites and the last one sits outside the frontend guard, so "before the first"
# is not enough.
pin_line=$(grep -n '^# The rest of the portable environment a bare `bash studio/setup\.sh` did not inherit\.' "$SETUP" | head -1 | cut -d: -f1)
check "found the pin block" "yes" "$([ -n "$pin_line" ] && echo yes || echo no)"
late=""
while IFS=: read -r _ln _; do
    [ -n "$_ln" ] || continue
    [ "$_ln" -gt "$pin_line" ] || late="$late $_ln"
done <<EOF
$(grep -n 'npm install' "$SETUP" | grep -vE '^[0-9]+:[[:space:]]*#')
EOF
check "every npm install runs after the npm cache pin" "" "$late"
uv_line=$(grep -n '^_setup_install_uv_pinned() {$' "$SETUP" | head -1 | cut -d: -f1)
check "the uv installer is defined after the pin too" "yes" \
    "$([ -n "$uv_line" ] && [ "$pin_line" -lt "$uv_line" ] && echo yes || echo no)"

echo
echo "[4] live npm: the default really is outside the root"
if command -v npm >/dev/null 2>&1; then
    bare="$(env -i PATH="$PATH" HOME="$T/home" npm config get cache 2>/dev/null)"
    case "$bare" in
        "$R"/*) printf '  FAIL  %s\n' "npm's default unexpectedly sits in the root ($bare)"; fails=$((fails+1)) ;;
        *) printf '  PASS  %s\n' "npm defaults outside the root ($bare)" ;;
    esac
    pinned="$(env -i PATH="$PATH" HOME="$T/home" NPM_CONFIG_CACHE="$R/cache/npm" \
        npm config get cache 2>/dev/null)"
    check "NPM_CONFIG_CACHE moves it inside" "$R/cache/npm" "$pinned"
else
    printf '  SKIP  live npm probe (npm is not installed here)\n'
fi

echo
if [ "$fails" -eq 0 ]; then printf 'ALL PASS\n'; else printf '%s FAILURES\n' "$fails"; exit 1; fi
