#!/usr/bin/env bash
# Regression test: a portable install must keep bun's package cache inside the root.
#
# setup.sh prefers bun over npm when it rebuilds a source frontend (`command -v bun`,
# and it installs bun itself on the managed-Node path), and bun reads NONE of npm's
# configuration. Checked against bun 1.4.0: with NPM_CONFIG_CACHE pinned and nothing
# else, `bun pm cache` still answers ~/.bun/install/cache and a real `bun install`
# writes its packages there -- outside the root the installer promises holds
# everything and the root its summary says `rm -rf` removes. bun's documented
# override is BUN_INSTALL_CACHE_DIR (https://bun.com/docs/pm/global-cache).
#
# Three generated environments have to carry it, because each one is where the next
# `bun install` gets its environment from: the installer's own exports, the portable
# shim, and share/studio.conf (what the launcher and `unsloth studio update` read).
# setup.sh derives it as well, for an update that arrives carrying only UNSLOTH_HOME.
#
# The live section only runs where bun is installed; the static sections are what
# keeps this meaningful on a machine without it.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
SETUP="$ROOT/studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockC="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockBUN="$(awk '
    /^# Keep bun.s package cache inside the portable root\./ {grab = 1}
    grab {print}
    grab && /^fi$/ {exit}
' "$SETUP")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockC" in *'NPM_CONFIG_CACHE='*) : ;; *) echo "FAIL: blockC extraction broke"; exit 1 ;; esac
case "$blockBUN" in *'BUN_INSTALL_CACHE_DIR'*) : ;; *) echo "FAIL: the setup.sh bun block extraction broke"; exit 1 ;; esac
# It has to sit before the bun install it is pinning, or the first `bun install`
# of the run has already written outside the root.
_bun_pin_line=$(grep -n '^# Keep bun.s package cache inside the portable root\.' "$SETUP" | head -n1 | cut -d: -f1)
_bun_use_line=$(grep -n '^if command -v bun &>/dev/null; then' "$SETUP" | head -n1 | cut -d: -f1)
[ -n "$_bun_pin_line" ] || { echo "FAIL: setup.sh no longer pins the bun cache"; exit 1; }
[ -n "$_bun_use_line" ] || { echo "FAIL: could not locate the bun install in setup.sh"; exit 1; }
[ "$_bun_pin_line" -lt "$_bun_use_line" ] || { echo "FAIL: setup.sh pins the bun cache after it uses bun"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home"

echo
echo "[1] the installer's own environment"
SNIP='substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockC"'
_export_portable_roots
printf "%s|%s\n" "$UNSLOTH_ROOT" "${BUN_INSTALL_CACHE_DIR:-}"'
R="$T/vol"; mkdir -p "$R"
R="$(CDPATH= cd -P -- "$R" && pwd -P)"
out="$(env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
    bash -c "$SNIP" _ --root "$R")"
check "install.sh pins the bun cache under the root" "$R/cache/bun" "${out#*|}"
out="$(env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" bash -c "$SNIP" _)"
check "a normal install sets nothing" "" "${out#*|}"

echo
echo "[2] the two files a later run reads its environment back out of"
shim_block="$(awk '
    /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ {if (!seen) {grab = 1}}
    grab {print}
    /_shim_tmp/ {seen = 1}
    grab && /^elif ! ln -sfn/ {exit}
' "$INSTALL")"
case "$shim_block" in *_shim_tmp*) : ;; *) echo "FAIL: shim block extraction broke"; exit 1 ;; esac
case "$shim_block" in
    *"export BUN_INSTALL_CACHE_DIR='\$_shim_root/cache/bun'"*)
        printf '  PASS  %s\n' "the portable shim exports BUN_INSTALL_CACHE_DIR" ;;
    *) printf '  FAIL  %s\n' "the portable shim exports BUN_INSTALL_CACHE_DIR"; fails=$((fails+1)) ;;
esac
conf_block="$(sed -n '/studio.conf: exe path/,/studio\.conf"$/p' "$INSTALL")"
case "$conf_block" in *'.unsloth-portable'*|*'UNSLOTH_EXE'*) : ;; *) echo "FAIL: conf block extraction broke"; exit 1 ;; esac
case "$conf_block" in
    *"export BUN_INSTALL_CACHE_DIR='\$_css_quoted_root/cache/bun'"*)
        printf '  PASS  %s\n' "studio.conf exports BUN_INSTALL_CACHE_DIR" ;;
    *) printf '  FAIL  %s\n' "studio.conf exports BUN_INSTALL_CACHE_DIR"; fails=$((fails+1)) ;;
esac

echo
echo "[3] setup.sh derives it when the caller only passed UNSLOTH_HOME"
setup_probe() { # unsloth_home preset
    env -i PATH="$PATH" HOME="$T/home" UNSLOTH_HOME="$1" BUN_INSTALL_CACHE_DIR="$2" \
        bash -c 'set -euo pipefail
verbose_substep() { :; }
'"$blockBUN"'
printf "%s" "${BUN_INSTALL_CACHE_DIR:-}"'
}
check "derived from UNSLOTH_HOME" "$R/cache/bun" "$(setup_probe "$R" "")"
check "an explicit setting wins" "$T/mine" "$(setup_probe "$R" "$T/mine")"
check "a normal install is left alone" "" "$(setup_probe "" "")"

echo
echo "[4] live bun: the npm pin alone does not contain it"
if command -v bun > /dev/null 2>&1; then
    printf '  ..    bun %s\n' "$(bun --version)"
    npm_only="$(env -i PATH="$PATH" HOME="$T/home" NPM_CONFIG_CACHE="$R/cache/npm" \
        bun pm cache 2>/dev/null)"
    case "$npm_only" in
        "$R"/*) printf '  FAIL  %s\n' "NPM_CONFIG_CACHE unexpectedly contains bun ($npm_only)"
                fails=$((fails+1)) ;;
        *) printf '  PASS  %s\n' "NPM_CONFIG_CACHE alone leaves bun outside the root ($npm_only)" ;;
    esac
    pinned="$(env -i PATH="$PATH" HOME="$T/home" NPM_CONFIG_CACHE="$R/cache/npm" \
        BUN_INSTALL_CACHE_DIR="$R/cache/bun" bun pm cache 2>/dev/null)"
    check "BUN_INSTALL_CACHE_DIR moves it inside" "$R/cache/bun" "$pinned"
else
    printf '  SKIP  live bun probe (bun is not installed here)\n'
fi

echo
if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
