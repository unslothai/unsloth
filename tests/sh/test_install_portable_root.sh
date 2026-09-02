#!/usr/bin/env bash
# Regression test: install.sh --portable / --root put every root under one
# directory. Runs the real blocks from install.sh against a fake HOME.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blockA="$(awk '
    /^# ── Parse flags ──$/ {grab = 1}
    grab {print}
    /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen = 1}
    seen && /^fi$/ {exit}
' "$INSTALL")"

blockB="$(awk '
    /^_resolve_studio_destinations\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

blockC="$(awk '
    /^_export_portable_roots\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockA" in *"--root) _next_is_root=true ;;"*) : ;; *) echo "FAIL: blockA lost --root"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockB" in *'STUDIO_HOME="$UNSLOTH_ROOT/studio"'*) : ;; *) echo "FAIL: blockB lost the portable branch"; exit 1 ;; esac
case "$blockC" in *"UV_CACHE_DIR="*) : ;; *) echo "FAIL: blockC extraction broke"; exit 1 ;; esac
case "$blockC" in *"UV_PYTHON_INSTALL_DIR="*) : ;; *) echo "FAIL: blockC lost the uv python dir"; exit 1 ;; esac
case "$blockC" in *"UV_PYTHON_BIN_DIR="*) : ;; *) echo "FAIL: blockC lost the uv python bin dir"; exit 1 ;; esac
case "$blockC" in *"NPM_CONFIG_CACHE="*) : ;; *) echo "FAIL: blockC lost the npm cache"; exit 1 ;; esac
case "$blockC" in *"CUDA_CACHE_PATH="*) : ;; *) echo "FAIL: blockC lost the cuda cache"; exit 1 ;; esac
case "$blockC" in *"PIP_CACHE_DIR="*) : ;; *) echo "FAIL: blockC lost the pip cache"; exit 1 ;; esac

SNIP='substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockC"'
_export_portable_roots
printf "%s|%s|%s|%s|%s|%s|%s|%s|%s\n" "$UNSLOTH_ROOT" "$STUDIO_HOME" "$DATA_DIR" "$_LOCAL_BIN" \
    "${UV_CACHE_DIR:-}" "${UV_PYTHON_INSTALL_DIR:-}" "$_STUDIO_HOME_REDIRECT" \
    "${UV_PYTHON_BIN_DIR:-}" "${PIP_CACHE_DIR:-}"'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# env -i: the caller's own UNSLOTH_* / UV_* / XDG_* would mask the leaks.
resolve() { # fakehome [args...]
    _home="$1"; shift
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" \
        bash -c "$SNIP" _ "$@"
}
field() { printf '%s' "$1" | cut -d'|' -f"$2"; }

new_home() { mktemp -d "$T/home.XXXXXX"; }

H="$(new_home)"
out="$(resolve "$H")"
check "default -> ~/.unsloth/studio"        "$H/.unsloth/studio"     "$(field "$out" 2)"
check "default -> ~/.local/share/unsloth"   "$H/.local/share/unsloth" "$(field "$out" 3)"
check "default -> ~/.local/bin"             "$H/.local/bin"          "$(field "$out" 4)"
check "default leaves UV_CACHE_DIR unset"   ""                       "$(field "$out" 5)"
check "default leaves PIP_CACHE_DIR unset"  ""                       "$(field "$out" 9)"
# A fake HOME takes the "home redirected" branch; either is fine as long as it is
# not "env", which skips the shell-rc and desktop-entry writes.
case "$(field "$out" 7)" in
    home|default) printf '  PASS  %s\n' "default is not env-override mode" ;;
    *) printf '  FAIL  %s : got [%s]\n' "default is not env-override mode" "$(field "$out" 7)"; fails=$((fails+1)) ;;
esac

H="$(new_home)"
out="$(resolve "$H" --portable)"
check "--portable root"            "$H/.unsloth"                "$(field "$out" 1)"
check "--portable studio home"     "$H/.unsloth/studio"         "$(field "$out" 2)"
check "--portable share"           "$H/.unsloth/share"          "$(field "$out" 3)"
check "--portable bin"             "$H/.unsloth/bin"            "$(field "$out" 4)"
check "--portable uv cache"        "$H/.unsloth/cache/uv"       "$(field "$out" 5)"
check "--portable uv python dir"   "$H/.unsloth/cache/uv-python" "$(field "$out" 6)"
check "--portable redirect marker" "env"                        "$(field "$out" 7)"
check "--portable uv python bin dir" "$H/.unsloth/bin"          "$(field "$out" 8)"
# pip is install_python_stack's fallback when uv fails, so ~/.cache/pip is reachable.
check "--portable pip cache"       "$H/.unsloth/cache/pip"      "$(field "$out" 9)"

H="$(new_home)"
mkdir -p "$T/elsewhere"
R="$(CDPATH= cd -P -- "$T/elsewhere" && pwd -P)"
out="$(resolve "$H" --root "$R")"
check "--root root"          "$R"                   "$(field "$out" 1)"
check "--root studio home"   "$R/studio"            "$(field "$out" 2)"
check "--root uv cache"      "$R/cache/uv"          "$(field "$out" 5)"

H="$(new_home)"
out="$(resolve "$H" "--root=$R")"
check "--root=DIR studio home" "$R/studio" "$(field "$out" 2)"

H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" UNSLOTH_HOME="$R" \
    bash -c "$SNIP" _)"
check "UNSLOTH_HOME implies portable" "$R/studio"   "$(field "$out" 2)"
check "UNSLOTH_HOME pins the uv cache" "$R/cache/uv" "$(field "$out" 5)"

H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$H/leftover" bash -c "$SNIP" _ --root "$R")"
check "--root beats UNSLOTH_STUDIO_HOME" "$R/studio" "$(field "$out" 2)"

# Before _PORTABLE_FLAT this installed to ~/.unsloth/studio instead.
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 UNSLOTH_STUDIO_HOME="$H/unsloth" bash -c "$SNIP" _)"
check "portable + studio home: root is the named path" "$H/unsloth"       "$(field "$out" 1)"
check "portable + studio home: nothing moves"          "$H/unsloth"       "$(field "$out" 2)"
check "portable + studio home: uv contained"           "$H/unsloth/cache/uv" "$(field "$out" 5)"

H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 STUDIO_HOME="$H/unsloth" bash -c "$SNIP" _)"
check "portable + STUDIO_HOME alias"                   "$H/unsloth"       "$(field "$out" 2)"

# storage_roots.portable_mode() strips and case-folds UNSLOTH_PORTABLE and install.ps1's
# guard is case-insensitive, so an installer that read these as off would write the normal
# roots while the runtime contained itself. The off values must stay off.
for _v in True " true " YES " ON "; do
    H="$(new_home)"
    out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
        UNSLOTH_PORTABLE="$_v" bash -c "$SNIP" _)"
    check "UNSLOTH_PORTABLE='$_v' is portable" "$H/.unsloth/cache/uv" "$(field "$out" 5)"
done
for _v in 0 false FALSE Off " no "; do
    H="$(new_home)"
    out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
        UNSLOTH_PORTABLE="$_v" bash -c "$SNIP" _)"
    check "UNSLOTH_PORTABLE='$_v' is not portable" "" "$(field "$out" 5)"
done

# `unsloth studio update` re-runs install.sh from the shim with UNSLOTH_HOME set,
# which made the refresh re-derive <root>/studio and exit "binary missing".
H="$(new_home)"
mkdir -p "$H/flat/unsloth_studio/bin"
F="$(CDPATH= cd -P -- "$H/flat" && pwd -P)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 UNSLOTH_HOME="$F" UNSLOTH_STUDIO_HOME="$F" bash -c "$SNIP" _)"
check "flat root survives a refresh"   "$F"     "$(field "$out" 2)"
check "flat root keeps its bin"        "$F/bin" "$(field "$out" 4)"

H="$(new_home)"
mkdir -p "$H/nested/studio/unsloth_studio/bin"
NR="$(CDPATH= cd -P -- "$H/nested" && pwd -P)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 UNSLOTH_HOME="$NR" UNSLOTH_STUDIO_HOME="$NR/studio" bash -c "$SNIP" _)"
check "nested root still nests on a refresh" "$NR/studio" "$(field "$out" 2)"

reject() { # label args...
    _label="$1"; shift
    if out="$(resolve "$(new_home)" "$@" 2>&1)"; then
        printf '  FAIL  %s : accepted, got [%s]\n' "$_label" "$out"; fails=$((fails+1))
    else
        case "$out" in
            *"--root requires a path argument"*) printf '  PASS  %s\n' "$_label" ;;
            *) printf '  FAIL  %s : wrong error [%s]\n' "$_label" "$out"; fails=$((fails+1)) ;;
        esac
    fi
}
reject "--root with no value"        --local --root
reject "--root with an empty value"  --local --root ""
reject "--root= with nothing after"  --local --root=
reject "--root followed by a flag"   --root --local
reject "--root with a blank value"   --local --root "   "
reject "--root= with a blank value"  --local "--root=   "

# A blank UNSLOTH_HOME used to pass the non-empty check, enable portable mode, then
# resolve to "" and export UV_INSTALL_DIR=/bin, UV_CACHE_DIR=/cache/uv.
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" UNSLOTH_HOME="   " \
    bash -c "$SNIP" _)"
check "blank UNSLOTH_HOME is not portable"   "$H/.unsloth/studio" "$(field "$out" 2)"
check "blank UNSLOTH_HOME sets no uv cache"  ""                   "$(field "$out" 5)"
check "blank UNSLOTH_HOME sets no uv bin"    ""                   "$(field "$out" 8)"

# Untrimmed, the blank won the precedence check and the real root was ignored.
H="$(new_home)"
mkdir -p "$H/aliased"
A="$(CDPATH= cd -P -- "$H/aliased" && pwd -P)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="  " STUDIO_HOME="$A" bash -c "$SNIP" _)"
check "blank UNSLOTH_STUDIO_HOME falls back to STUDIO_HOME" "$A" "$(field "$out" 2)"

H="$(new_home)"
mkdir -p "$H/custom"
C="$(CDPATH= cd -P -- "$H/custom" && pwd -P)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$C" bash -c "$SNIP" _)"
check "UNSLOTH_STUDIO_HOME studio home" "$C"       "$(field "$out" 2)"
check "UNSLOTH_STUDIO_HOME share"       "$C/share" "$(field "$out" 3)"
check "UNSLOTH_STUDIO_HOME sets no uv cache" ""    "$(field "$out" 5)"

H="$(new_home)"
if grep -q -- '--portable and --root are not supported with --tauri' "$INSTALL"; then
    printf '  PASS  %s\n' "--tauri rejects --portable"
else
    printf '  FAIL  %s\n' "--tauri rejects --portable"; fails=$((fails+1))
fi

H="$(new_home)"
out="$(resolve "$H" --root "$R")"
outside=""
for i in 2 3 4 5 6 8 9; do
    v="$(field "$out" "$i")"
    case "$v" in "$R"/*) ;; *) outside="$outside $i:$v" ;; esac
done
check "nothing escapes the portable root" "" "$outside"

shim_block="$(awk '
    /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ {if (!seen) {grab = 1}}
    grab {print}
    /_shim_tmp/ {seen = 1}
    grab && /^elif ! ln -sfn/ {exit}
' "$INSTALL")"
for v in UNSLOTH_HOME UNSLOTH_PORTABLE UNSLOTH_STUDIO_HOME UNSLOTH_LLAMA_CPP_PATH \
         UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR UV_NO_MODIFY_PATH \
         UV_INSTALL_DIR UV_TOOL_BIN_DIR NPM_CONFIG_CACHE CUDA_CACHE_PATH PIP_CACHE_DIR; do
    case "$shim_block" in
        *"export $v="*) printf '  PASS  %s\n' "portable shim exports $v" ;;
        *) printf '  FAIL  %s\n' "portable shim exports $v"; fails=$((fails+1)) ;;
    esac
done
case "$shim_block" in
    *'exec '*) printf '  PASS  %s\n' "portable shim execs the venv entry point" ;;
    *) printf '  FAIL  %s\n' "portable shim execs the venv entry point"; fails=$((fails+1)) ;;
esac

conf_block="$(sed -n '/studio.conf: exe path/,/studio\.conf"$/p' "$INSTALL")"
for v in UNSLOTH_HOME UNSLOTH_PORTABLE UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR \
         UV_INSTALL_DIR UV_TOOL_BIN_DIR NPM_CONFIG_CACHE CUDA_CACHE_PATH PIP_CACHE_DIR; do
    case "$conf_block" in
        *"export $v="*) printf '  PASS  %s\n' "studio.conf exports $v" ;;
        *) printf '  FAIL  %s\n' "studio.conf exports $v"; fails=$((fails+1)) ;;
    esac
done

path_warn_block="$(sed -n "/^# Warn if another 'unsloth' wins on PATH/,/^fi\$/p" "$INSTALL")"
case "$path_warn_block" in
    *'_shim_real'*) printf '  PASS  %s\n' "PATH warning knows about the portable shim" ;;
    *) printf '  FAIL  %s\n' "PATH warning knows about the portable shim"; fails=$((fails+1)) ;;
esac

UNINSTALL="$HERE/../../scripts/uninstall.sh"
if grep -q "export UNSLOTH_HOME=" "$UNINSTALL"; then
    printf '  PASS  %s\n' "uninstall.sh reads the master root from studio.conf"
else
    printf '  FAIL  %s\n' "uninstall.sh reads the master root from studio.conf"; fails=$((fails+1))
fi

done_block="$(sed -n '/portable install; everything lives in:/,/were left untouched/p' "$INSTALL")"
case "$done_block" in
    *"rm -rf '"*)
        printf '  PASS  %s\n' "closing message removes the root with rm -rf" ;;
    *) printf '  FAIL  %s\n' "closing message removes the root with rm -rf"; fails=$((fails+1)) ;;
esac
# See tests/sh/test_uninstall_portable_root_scope.sh: that command also removes a
# default install, so it must never be offered as the way to remove this root.
case "$done_block" in
    *"UNSLOTH_HOME='"*"scripts/uninstall.sh"*)
        printf '  FAIL  %s\n' "closing message does not offer UNSLOTH_HOME=... scripts/uninstall.sh"
        fails=$((fails+1)) ;;
    *) printf '  PASS  %s\n' "closing message does not offer UNSLOTH_HOME=... scripts/uninstall.sh" ;;
esac

# A layout dir the user pre-symlinked to another volume is not covered by that
# rm -rf, so the closing message has to say so.
case "$done_block" in
    *"_portable_escapes"*) printf '  PASS  %s\n' "closing message reports symlinks out of the root" ;;
    *) printf '  FAIL  %s\n' "closing message reports symlinks out of the root"; fails=$((fails+1)) ;;
esac

esc_block="$(awk '
    /^_portable_escapes\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"
case "$esc_block" in *'pwd -P'*) : ;; *) echo "FAIL: esc_block extraction broke"; exit 1 ;; esac

ESCSNIP='_PORTABLE_MODE=true
UNSLOTH_ROOT="$1"
'"$esc_block"'
_portable_escapes'
escapes() { bash -c "$ESCSNIP" _ "$1"; }

mkdir -p "$T/escroot/studio" "$T/escroot/share" "$T/bigdisk/cache"
ER="$(CDPATH= cd -P -- "$T/escroot" && pwd -P)"
BIG="$(CDPATH= cd -P -- "$T/bigdisk/cache" && pwd -P)"
check "a contained root reports no escapes" "" "$(escapes "$ER")"
ln -s "$BIG" "$ER/cache"
check "a cache symlinked off the root is named" "cache -> $BIG" "$(escapes "$ER")"
rm "$ER/cache"
mkdir -p "$ER/inside"
ln -s "$ER/inside" "$ER/cache"
check "a symlink inside the root is not an escape" "" "$(escapes "$ER")"
rm "$ER/cache"
rmdir "$ER/studio"
ln -s "$BIG" "$ER/studio"
check "a studio symlinked off the root is named" "studio -> $BIG" "$(escapes "$ER")"
rm "$ER/studio"
# setup.sh installs node and whisper.cpp beside studio/ at the master root, so a
# pre-symlinked one escapes the printed rm -rf exactly like llama.cpp does.
for _rt in llama.cpp node whisper.cpp; do
    ln -s "$BIG" "$ER/$_rt"
    check "a $_rt symlinked off the root is named" "$_rt -> $BIG" "$(escapes "$ER")"
    rm "$ER/$_rt"
done

# setup.sh clears the desktop app's WebView caches under $HOME. Portable mode
# prints "the desktop app and shell PATH were left untouched", so it must not.
SETUP="$HERE/../../studio/setup.sh"
blockW="$(awk '
    /^_clear_webview_caches\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$SETUP")"
blockP="$(awk '
    /^_setup_portable_mode\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$SETUP")"
guard="$(awk '
    /^if \[ -z "\$STAGE_ROOT" \] && \[ -x "\$VENV_DIR\/bin\/python" \]/ {grab = 1}
    grab {print}
    grab && /^fi$/ {exit}
' "$SETUP")"
# _setup_portable_mode normalizes through this helper, so the snippet needs it too.
blockT="$(grep '^_setup_trim_ws() ' "$SETUP")"
case "$blockW" in *"WebKitCache"*) : ;; *) echo "FAIL: blockW extraction broke"; exit 1 ;; esac
case "$blockP" in *"UNSLOTH_PORTABLE"*) : ;; *) echo "FAIL: blockP extraction broke"; exit 1 ;; esac
case "$blockT" in *"[[:space:]]"*) : ;; *) echo "FAIL: blockT extraction broke"; exit 1 ;; esac
case "$guard" in *"_clear_webview_caches"*) : ;; *) echo "FAIL: guard extraction broke"; exit 1 ;; esac

WVSNIP='substep() { :; }
'"$blockW"'
'"$blockT"'
'"$blockP"'
STAGE_ROOT=""
STUDIO_HOME="$HOME/.unsloth/studio"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
'"$guard"'
[ -e "$HOME/.local/share/ai.unsloth.studio/WebKitCache" ] && printf kept || printf cleared'

webview_case() { # label expected mode
    _label="$1"; _expected="$2"; _mode="${3:-none}"
    _h="$(new_home)"
    mkdir -p "$_h/.unsloth/studio/unsloth_studio/bin" \
        "$_h/.local/share/ai.unsloth.studio/WebKitCache" "$_h/portable"
    : > "$_h/.unsloth/studio/unsloth_studio/bin/python"
    chmod +x "$_h/.unsloth/studio/unsloth_studio/bin/python"
    : > "$_h/.local/share/ai.unsloth.studio/.webview-cache-cleared"
    printf '%s\n' "$_h/portable" > "$_h/portable/.unsloth-portable-root"
    case "$_mode" in
        env)    _got="$(env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" \
                    UNSLOTH_PORTABLE=1 bash -c "$WVSNIP" _)" ;;
        env_mixedcase)
                _got="$(env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" \
                    UNSLOTH_PORTABLE=" True " bash -c "$WVSNIP" _)" ;;
        marker) _got="$(env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" \
                    UNSLOTH_HOME="$_h/portable" bash -c "$WVSNIP" _)" ;;
        *)      _got="$(env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" \
                    bash -c "$WVSNIP" _)" ;;
    esac
    check "$_label" "$_expected" "$_got"
}
webview_case "normal install still clears the WebView cache"  cleared none
webview_case "portable install leaves the WebView cache"      kept    env
webview_case "portable marker alone leaves the WebView cache" kept    marker
# storage_roots.portable_mode() strips and case-folds, so this value IS portable to the
# runtime; reading it as off here would clear a cache the portable run promised to keep.
webview_case "UNSLOTH_PORTABLE=' True ' leaves the WebView cache" kept env_mixedcase

# Env-mode writes no rc PATH entry, so a deferred `unsloth studio` either misses or
# starts a DIFFERENT install; every hint outside an activated venv names the shim.
hint_block="$(awk '
    /^# Single-quote-escape so paths with spaces/ {grab = 1}
    grab {print}
    grab && /^_li_launch_q=/ {seen = 1}
    seen && /^fi$/ {exit}
' "$INSTALL")"
declined="$(awk '
    /step "launch" "to start later, run:"/ {grab = 1}
    grab {print; n++}
    n >= 2 {exit}
' "$INSTALL")"
case "$hint_block" in *'_li_shim_q'*) : ;; *) echo "FAIL: hint_block extraction broke"; exit 1 ;; esac
case "$declined" in *'substep'*) : ;; *) echo "FAIL: declined extraction broke"; exit 1 ;; esac

HINTSNIP='step() { :; }
substep() { printf "%s\n" "$1"; }
_LOCAL_BIN="$1"
VENV_DIR="$2"
_STUDIO_HOME_REDIRECT="$3"
'"$hint_block"'
'"$declined"
hint() { env -i PATH="$PATH" sh -c "$HINTSNIP" _ "$1" "$2" "$3"; }
check "deferred hint names the portable shim" \
    "'$R/bin/unsloth' studio -p 8888" "$(hint "$R/bin" "$R/studio/unsloth_studio" env)"
check "deferred hint stays bare on a default install" \
    "unsloth studio -p 8888" "$(hint "$H/.local/bin" "$H/.unsloth/studio/unsloth_studio" default)"
check "deferred hint quotes an apostrophe in the root" \
    "'/o'\\''brien/bin/unsloth' studio -p 8888" "$(hint "/o'brien/bin" /x env)"
# The one survivor is the line after `source .../activate`, where PATH really does resolve it.
check "only the post-activate hint is bare" \
    "1" "$(grep -c 'substep "unsloth studio -p 8888"' "$INSTALL")"

# Converting a default install in place leaves its launcher, desktop entry and .app
# under $HOME; they run this venv uncontained and write logs, PID and .command back
# there. Portable mode removes nothing outside the root, so it has to name them.
conv_block="${shim_block%elif ! ln -sfn*}fi"
CONVSNIP='substep() { printf "%s\n" "$1"; }
C_WARN=""
UNSLOTH_ROOT="$1"
STUDIO_HOME="$UNSLOTH_ROOT/studio"
DATA_DIR="$UNSLOTH_ROOT/share"
_LOCAL_BIN="$UNSLOTH_ROOT/bin"
_shim_path="$_LOCAL_BIN/unsloth"
VENV_DIR="$2"
_PORTABLE_MODE=true
'"$conv_block"

convert_case() { # fakehome owner_exe
    _cvh="$1"
    _cvr="$_cvh/root"
    _cvv="$_cvh/.unsloth/studio/unsloth_studio"
    mkdir -p "$_cvv/bin" "$_cvh/.local/share/unsloth" "$_cvh/.local/share/applications" \
        "$_cvh/Desktop" "$_cvh/Applications/Unsloth Studio.app/Contents/MacOS" \
        "$_cvr/studio" "$_cvr/share" "$_cvr/bin"
    printf '#!/bin/sh\n' > "$_cvv/bin/unsloth"
    chmod +x "$_cvv/bin/unsloth"
    printf "UNSLOTH_EXE='%s'\n" "$2" > "$_cvh/.local/share/unsloth/studio.conf"
    printf 'Exec="%s/.local/share/unsloth/launch-studio.sh"\n' "$_cvh" \
        > "$_cvh/.local/share/applications/unsloth-studio.desktop"
    cp "$_cvh/.local/share/applications/unsloth-studio.desktop" \
        "$_cvh/Desktop/unsloth-studio.desktop"
    printf "exec '%s/.local/share/unsloth/launch-studio.sh' \"\$@\"\n" "$_cvh" \
        > "$_cvh/Applications/Unsloth Studio.app/Contents/MacOS/launch-studio"
    env -i HOME="$_cvh" PATH="$PATH" sh -c "$CONVSNIP" _ "$_cvr" "$_cvv"
}
named() { case "$1" in *"$2"*) echo yes ;; *) echo no ;; esac; }

H="$(new_home)"
out="$(convert_case "$H" "$H/.unsloth/studio/unsloth_studio/bin/unsloth")"
check "conversion names the stale launcher data dir" \
    "yes" "$(named "$out" "  $H/.local/share/unsloth")"
check "conversion names the stale desktop entry" \
    "yes" "$(named "$out" "$H/.local/share/applications/unsloth-studio.desktop")"
check "conversion names the stale Desktop copy" \
    "yes" "$(named "$out" "$H/Desktop/unsloth-studio.desktop")"
check "conversion names the stale macOS app" \
    "yes" "$(named "$out" "$H/Applications/Unsloth Studio.app")"
# The symlink warning uses -ef for the same reason: never name somebody else's install.
H="$(new_home)"
out="$(convert_case "$H" "$T/somebody-elses-venv/bin/unsloth")"
check "conversion says nothing about an unrelated install" \
    "no" "$(named "$out" "older launcher")"

# The flag is the answer to unslothai/unsloth#8865; undiscoverable, it is not delivered.
README="$HERE/../../README.md"
for needle in '--portable' '--root' 'UNSLOTH_PORTABLE=1' 'UNSLOTH_HOME=/abs/path'; do
    if grep -qF -e "$needle" "$README"; then
        printf '  PASS  %s\n' "README documents $needle"
    else
        printf '  FAIL  %s\n' "README documents $needle"; fails=$((fails+1))
    fi
done

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
