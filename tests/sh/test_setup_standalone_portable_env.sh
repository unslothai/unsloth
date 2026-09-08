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
#   * PIP_CACHE_DIR -- uv is not the only installer setup.sh reaches for. fast_install falls
#     through to `python -m pip install` whenever `command -v uv` misses or the `uv pip
#     install` above it fails, the staged-root branch uses pip outright, the Colab path runs
#     `pip install -r`, and install_python_stack.py forces plain pip for the wheels uv's
#     filename check rejects. pip's default cache is $HOME/.cache/pip on Linux and
#     $HOME/Library/Caches/pip on macOS, so multi-GB Torch and CUDA wheels landed outside the
#     root, survived the advertised `rm -rf <root>`, and were re-downloaded by the next update.
#
# install.sh's _export_portable_roots, share/studio.conf, the generated bin/unsloth shim and
# the CLI's _portable_root_env all set these four; setup.sh is the one that did not.
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
case "$blockPIN"     in *'PIP_CACHE_DIR'*)     : ;; *) echo "FAIL: the pin block lost PIP_CACHE_DIR"; exit 1 ;; esac
# Everything install.sh's shim pins, this block pins too. Matched against the block with its
# COMMENTS STRIPPED: the comment above these lines names UV_PYTHON_INSTALL_DIR to explain why
# it matters, so a check against the raw text passes on the prose alone and keeps passing
# after the export itself is deleted. Verified by deleting it.
blockPIN_code="$(printf '%s\n' "$blockPIN" | sed -e 's/#.*$//')"
for _v in UV_PYTHON_INSTALL_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR UV_PYTHON_BIN_DIR CUDA_CACHE_PATH; do
    case "$blockPIN_code" in
        *"export $_v="*) : ;;
        *) echo "FAIL: the pin block lost $_v"; exit 1 ;;
    esac
done
case "$blockTRIM"    in *'_setup_trim_ws'*)     : ;; *) echo "FAIL: _setup_trim_ws extraction broke"; exit 1 ;; esac
case "$blockPERSIST" in *'Added by Unsloth setup'*) : ;; *) echo "FAIL: _setup_persist_uv_path extraction broke"; exit 1 ;; esac
case "$blockDEST"    in *'HOME/.local/bin'*)    : ;; *) echo "FAIL: the uv destination cascade extraction broke"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
R="$T/vol/portable"; mkdir -p "$R/bin"
R="$(CDPATH= cd -P -- "$R" && pwd -P)"

echo
echo "[1] the four values the standalone path now derives"
pin() { # $1 = UNSLOTH_HOME, rest = extra env assignments
    _uh="$1"; shift
    # shellcheck disable=SC2086
    env -i PATH="$PATH" HOME="$T/home" UNSLOTH_HOME="$_uh" $* bash -c '
set -eu
'"$blockTRIM"'
'"$blockPIN"'
printf "%s|%s|%s|%s\n" "${UV_INSTALL_DIR:-}" "${UV_NO_MODIFY_PATH:-}" "${NPM_CONFIG_CACHE:-}" "${PIP_CACHE_DIR:-}"'
}
_all="$R/bin|1|$R/cache/npm|$R/cache/pip"
check "uv installs into the root"        "$_all"                  "$(pin "$R")"
check "a normal install sets none of them" "|||"                  "$(pin "")"
check "an explicit UV_INSTALL_DIR wins"  "$T/mine|1|$R/cache/npm|$R/cache/pip" "$(pin "$R" UV_INSTALL_DIR=$T/mine)"
check "an explicit NPM_CONFIG_CACHE wins" "$R/bin|1|$T/npm|$R/cache/pip"      "$(pin "$R" NPM_CONFIG_CACHE=$T/npm)"
check "an explicit PIP_CACHE_DIR wins"    "$R/bin|1|$R/cache/npm|$T/pip"      "$(pin "$R" PIP_CACHE_DIR=$T/pip)"
# Blank counts as unset, matching install.sh's _epr_default and the resolvers.
check "a blank value is treated as unset" "$_all" "$(pin "$R" UV_INSTALL_DIR= NPM_CONFIG_CACHE= PIP_CACHE_DIR=)"

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
# Same for pip. fast_install's fallback arm is the main one, but it is not the only pip in
# this script: the Colab requirements install and the staged-root path both reach it too, and
# install_python_stack.py -- which setup.sh execs, so it inherits whatever is set here -- forces
# plain pip for the wheels uv's filename check rejects.
late_pip=""
while IFS=: read -r _ln _; do
    [ -n "$_ln" ] || continue
    [ "$_ln" -gt "$pin_line" ] || late_pip="$late_pip $_ln"
done <<EOF
$(grep -nE '(^|[^_[:alnum:]])pip install' "$SETUP" | grep -vE '^[0-9]+:[[:space:]]*#')
EOF
check "every pip install runs after the pip cache pin" "" "$late_pip"

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
echo "[5] live pip: the default really is outside the root"
# Asked of pip itself rather than asserted from the documented default, because that default
# is platform-dependent ($HOME/.cache/pip on Linux, $HOME/Library/Caches/pip on macOS) and
# both are outside the root, which is the only property that matters here.
_py=""
for _c in python3 python; do command -v "$_c" >/dev/null 2>&1 && { _py="$_c"; break; }; done
if [ -n "$_py" ] && env -i PATH="$PATH" HOME="$T/home" "$_py" -m pip --version >/dev/null 2>&1; then
    bare_pip="$(env -i PATH="$PATH" HOME="$T/home" "$_py" -m pip cache dir 2>/dev/null | tail -1)"
    case "$bare_pip" in
        "$R"/*) printf '  FAIL  %s\n' "pip's default unexpectedly sits in the root ($bare_pip)"; fails=$((fails+1)) ;;
        "") printf '  SKIP  live pip probe (pip cache dir printed nothing)\n' ;;
        *) printf '  PASS  %s\n' "pip defaults outside the root ($bare_pip)" ;;
    esac
    if [ -n "$bare_pip" ]; then
        pinned_pip="$(env -i PATH="$PATH" HOME="$T/home" PIP_CACHE_DIR="$R/cache/pip" \
            "$_py" -m pip cache dir 2>/dev/null | tail -1)"
        check "PIP_CACHE_DIR moves it inside" "$R/cache/pip" "$pinned_pip"
    fi
else
    printf '  SKIP  live pip probe (no usable pip here)\n'
fi

echo
if [ "$fails" -eq 0 ]; then printf 'ALL PASS\n'; else printf '%s FAILURES\n' "$fails"; exit 1; fi
