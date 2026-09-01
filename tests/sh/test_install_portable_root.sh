#!/usr/bin/env bash
# Regression test: install.sh --portable / --root put every root under one
# directory (issue #8865), and leave the default install byte-identical.
#
# Extracts the real flag parser, _resolve_studio_destinations and
# _export_portable_roots from install.sh by content anchors (not line numbers)
# and runs them against a hermetic fake HOME for each case.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Block A: the flag parser, from its banner down to the portable-root default.
blockA="$(awk '
    /^# ── Parse flags ──$/ {grab = 1}
    grab {print}
    /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen = 1}
    seen && /^fi$/ {exit}
' "$INSTALL")"

# Block B: _resolve_studio_destinations, anchored on the definition and closed
# on the first column-0 brace, so branches can be reordered inside it freely.
blockB="$(awk '
    /^_resolve_studio_destinations\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

# Block C: _export_portable_roots, same shape.
blockC="$(awk '
    /^_export_portable_roots\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$INSTALL")"

# Self-validate every extraction so a future install.sh refactor fails loudly
# here rather than silently testing an empty string.
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockA" in *"--root) _next_is_root=true ;;"*) : ;; *) echo "FAIL: blockA lost --root"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockB" in *'STUDIO_HOME="$UNSLOTH_ROOT/studio"'*) : ;; *) echo "FAIL: blockB lost the portable branch"; exit 1 ;; esac
case "$blockC" in *"UV_CACHE_DIR="*) : ;; *) echo "FAIL: blockC extraction broke"; exit 1 ;; esac
case "$blockC" in *"UV_PYTHON_INSTALL_DIR="*) : ;; *) echo "FAIL: blockC lost the uv python dir"; exit 1 ;; esac
case "$blockC" in *"UV_PYTHON_BIN_DIR="*) : ;; *) echo "FAIL: blockC lost the uv python bin dir"; exit 1 ;; esac
# npm and the NVIDIA JIT cache: measured escaping a real portable install
# (~/.npm/_logs, ~/.nv/ComputeCache) once uv was contained and they were what
# was left. Easy to drop in a refactor, invisible without a clean-env run.
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

# env -i, so the caller's own UNSLOTH_* / UV_* / XDG_* never reach the snippet:
# this box exports several of them and they would mask exactly the leaks the
# feature exists to close.
resolve() { # fakehome [args...]
    _home="$1"; shift
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" \
        bash -c "$SNIP" _ "$@"
}
field() { printf '%s' "$1" | cut -d'|' -f"$2"; }

new_home() { mktemp -d "$T/home.XXXXXX"; }

# ── 1. Default install: unchanged ──
H="$(new_home)"
out="$(resolve "$H")"
check "default -> ~/.unsloth/studio"        "$H/.unsloth/studio"     "$(field "$out" 2)"
check "default -> ~/.local/share/unsloth"   "$H/.local/share/unsloth" "$(field "$out" 3)"
check "default -> ~/.local/bin"             "$H/.local/bin"          "$(field "$out" 4)"
check "default leaves UV_CACHE_DIR unset"   ""                       "$(field "$out" 5)"
# A fake HOME differs from the passwd-DB home, so this is the "home redirected"
# branch rather than "default". The two produce identical paths by design and
# only the marker differs; what matters here is that it is not "env", which is
# what would skip the shell-rc and desktop-entry writes.
case "$(field "$out" 7)" in
    home|default) printf '  PASS  %s\n' "default is not env-override mode" ;;
    *) printf '  FAIL  %s : got [%s]\n' "default is not env-override mode" "$(field "$out" 7)"; fails=$((fails+1)) ;;
esac

# ── 2. --portable with no --root: contained, still at the familiar location ──
H="$(new_home)"
out="$(resolve "$H" --portable)"
check "--portable root"            "$H/.unsloth"                "$(field "$out" 1)"
check "--portable studio home"     "$H/.unsloth/studio"         "$(field "$out" 2)"
check "--portable share"           "$H/.unsloth/share"          "$(field "$out" 3)"
check "--portable bin"             "$H/.unsloth/bin"            "$(field "$out" 4)"
check "--portable uv cache"        "$H/.unsloth/cache/uv"       "$(field "$out" 5)"
check "--portable uv python dir"   "$H/.unsloth/cache/uv-python" "$(field "$out" 6)"
check "--portable redirect marker" "env"                        "$(field "$out" 7)"
# Separate from UV_PYTHON_INSTALL_DIR: `uv python install` puts the interpreter
# in the install dir but its python3.x symlinks here, and unset that is
# ~/.local/bin -- observed leaving 3 entries in the home on an otherwise clean
# portable install.
check "--portable uv python bin dir" "$H/.unsloth/bin"          "$(field "$out" 8)"

# ── 3. --root DIR ──
H="$(new_home)"
mkdir -p "$T/elsewhere"
R="$(CDPATH= cd -P -- "$T/elsewhere" && pwd -P)"
out="$(resolve "$H" --root "$R")"
check "--root root"          "$R"                   "$(field "$out" 1)"
check "--root studio home"   "$R/studio"            "$(field "$out" 2)"
check "--root uv cache"      "$R/cache/uv"          "$(field "$out" 5)"

# ── 4. --root=DIR, the attached form ──
H="$(new_home)"
out="$(resolve "$H" "--root=$R")"
check "--root=DIR studio home" "$R/studio" "$(field "$out" 2)"

# ── 5. UNSLOTH_HOME, for `curl ... | sh` where no flag can be passed ──
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" UNSLOTH_HOME="$R" \
    bash -c "$SNIP" _)"
check "UNSLOTH_HOME implies portable" "$R/studio"   "$(field "$out" 2)"
check "UNSLOTH_HOME pins the uv cache" "$R/cache/uv" "$(field "$out" 5)"

# ── 6. Portable outranks a leftover UNSLOTH_STUDIO_HOME ──
# Otherwise a stale export from an earlier install would split the tree in two,
# which is the one outcome the flag promises not to produce.
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$H/leftover" bash -c "$SNIP" _ --root "$R")"
check "--root beats UNSLOTH_STUDIO_HOME" "$R/studio" "$(field "$out" 2)"

# ── 6b. UNSLOTH_PORTABLE=1 contains an existing UNSLOTH_STUDIO_HOME in place ──
# The natural way to ask for this is `UNSLOTH_PORTABLE=1
# UNSLOTH_STUDIO_HOME=/home/me/unsloth`. Before the _PORTABLE_FLAT branch that
# silently ignored the named path and installed to ~/.unsloth/studio, leaving
# the user with an install nowhere near where they asked for it.
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 UNSLOTH_STUDIO_HOME="$H/unsloth" bash -c "$SNIP" _)"
check "portable + studio home: root is the named path" "$H/unsloth"       "$(field "$out" 1)"
check "portable + studio home: nothing moves"          "$H/unsloth"       "$(field "$out" 2)"
check "portable + studio home: uv contained"           "$H/unsloth/cache/uv" "$(field "$out" 5)"

# The STUDIO_HOME alias has to reach the same place, or the two spellings of the
# same request would produce two different installs.
H="$(new_home)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_PORTABLE=1 STUDIO_HOME="$H/unsloth" bash -c "$SNIP" _)"
check "portable + STUDIO_HOME alias"                   "$H/unsloth"       "$(field "$out" 2)"

# ── 7. UNSLOTH_STUDIO_HOME alone still behaves exactly as before ──
H="$(new_home)"
mkdir -p "$H/custom"
C="$(CDPATH= cd -P -- "$H/custom" && pwd -P)"
out="$(env -i HOME="$H" PATH="$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$C" bash -c "$SNIP" _)"
check "UNSLOTH_STUDIO_HOME studio home" "$C"       "$(field "$out" 2)"
check "UNSLOTH_STUDIO_HOME share"       "$C/share" "$(field "$out" 3)"
check "UNSLOTH_STUDIO_HOME sets no uv cache" ""    "$(field "$out" 5)"

# ── 8. --portable is rejected with --tauri ──
# The desktop app resolves ~/.unsloth/studio in Rust and never sees a
# per-session variable, so it would launch a Studio that is not there.
H="$(new_home)"
if grep -q -- '--portable and --root are not supported with --tauri' "$INSTALL"; then
    printf '  PASS  %s\n' "--tauri rejects --portable"
else
    printf '  FAIL  %s\n' "--tauri rejects --portable"; fails=$((fails+1))
fi

# ── 9. Every path the snippet produced stays inside the root ──
H="$(new_home)"
out="$(resolve "$H" --root "$R")"
outside=""
for i in 2 3 4 5 6 8; do
    v="$(field "$out" "$i")"
    case "$v" in "$R"/*) ;; *) outside="$outside $i:$v" ;; esac
done
check "nothing escapes the portable root" "" "$outside"

# ── 10. The shim is a wrapper, not a symlink, and carries the roots ──
# A symlink exports nothing, so `bin/unsloth` from a fresh shell would find the
# interpreter (the backend infers STUDIO_HOME from sys.prefix) and still send uv
# to ~/.cache/uv. That is a half-contained install that looks fine right up
# until it downloads several GB into the home directory.
shim_block="$(awk '
    /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ {if (!seen) {grab = 1}}
    grab {print}
    /_shim_tmp/ {seen = 1}
    grab && /^elif ! ln -sfn/ {exit}
' "$INSTALL")"
for v in UNSLOTH_HOME UNSLOTH_PORTABLE UNSLOTH_STUDIO_HOME UNSLOTH_LLAMA_CPP_PATH \
         UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR UV_NO_MODIFY_PATH \
         NPM_CONFIG_CACHE CUDA_CACHE_PATH; do
    case "$shim_block" in
        *"export $v="*) printf '  PASS  %s\n' "portable shim exports $v" ;;
        *) printf '  FAIL  %s\n' "portable shim exports $v"; fails=$((fails+1)) ;;
    esac
done
case "$shim_block" in
    *'exec '*) printf '  PASS  %s\n' "portable shim execs the venv entry point" ;;
    *) printf '  FAIL  %s\n' "portable shim execs the venv entry point"; fails=$((fails+1)) ;;
esac

# ── 11. studio.conf restates them for the launcher ──
# The desktop launcher and `unsloth studio update` start from a fresh shell that
# inherits nothing, so a conf without these repopulates ~/.cache/uv on update.
conf_block="$(sed -n '/studio.conf: exe path/,/studio\.conf"$/p' "$INSTALL")"
for v in UNSLOTH_HOME UNSLOTH_PORTABLE UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR \
         NPM_CONFIG_CACHE CUDA_CACHE_PATH; do
    case "$conf_block" in
        *"export $v="*) printf '  PASS  %s\n' "studio.conf exports $v" ;;
        *) printf '  FAIL  %s\n' "studio.conf exports $v"; fails=$((fails+1)) ;;
    esac
done

# ── 12. The PATH warning does not fire on our own portable shim ──
# The old symlink shared a realpath with the venv entry point; a wrapper script
# resolves to itself, so the "another 'unsloth' wins on PATH" check saw the shim
# as a foreign install and told the user to avoid the very path the installer
# had just recommended. Observed on a real --root install before the fix.
path_warn_block="$(sed -n "/^# Warn if another 'unsloth' wins on PATH/,/^fi\$/p" "$INSTALL")"
case "$path_warn_block" in
    *'_shim_real'*) printf '  PASS  %s\n' "PATH warning knows about the portable shim" ;;
    *) printf '  FAIL  %s\n' "PATH warning knows about the portable shim"; fails=$((fails+1)) ;;
esac

# ── 13. The uninstaller reads the master root back out of studio.conf ──
# The venv is at <root>/studio, so the uninstaller's three-dirname walk lands on
# the Studio root and would leave <root>/{bin,cache,llama.cpp,node} behind.
UNINSTALL="$HERE/../../scripts/uninstall.sh"
if grep -q "export UNSLOTH_HOME=" "$UNINSTALL"; then
    printf '  PASS  %s\n' "uninstall.sh reads the master root from studio.conf"
else
    printf '  FAIL  %s\n' "uninstall.sh reads the master root from studio.conf"; fails=$((fails+1))
fi

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
