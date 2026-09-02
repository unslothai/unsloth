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

SNIP='substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockC"'
_export_portable_roots
printf "%s|%s|%s|%s|%s|%s|%s|%s\n" "$UNSLOTH_ROOT" "$STUDIO_HOME" "$DATA_DIR" "$_LOCAL_BIN" \
    "${UV_CACHE_DIR:-}" "${UV_PYTHON_INSTALL_DIR:-}" "$_STUDIO_HOME_REDIRECT" \
    "${UV_PYTHON_BIN_DIR:-}"'

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
for i in 2 3 4 5 6 8; do
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
         UV_INSTALL_DIR UV_TOOL_BIN_DIR NPM_CONFIG_CACHE CUDA_CACHE_PATH; do
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
         UV_INSTALL_DIR UV_TOOL_BIN_DIR NPM_CONFIG_CACHE CUDA_CACHE_PATH; do
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
    *"UNSLOTH_HOME='"*"scripts/uninstall.sh"*)
        printf '  PASS  %s\n' "closing message names UNSLOTH_HOME for the uninstaller" ;;
    *) printf '  FAIL  %s\n' "closing message names UNSLOTH_HOME for the uninstaller"; fails=$((fails+1)) ;;
esac

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
