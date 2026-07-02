#!/usr/bin/env bash
# Behavioral test for the --with-llama-cpp-dir linking block in studio/setup.sh.
# The companion test_with_llama_cpp_dir_flag.sh is a static wiring check; this one
# actually RUNS the real link logic (extracted from setup.sh by content anchors,
# not line numbers) against hermetic fake dirs and asserts the outcomes Lee asked
# for: an external built dir gets linked, neither the prebuilt download nor the
# source build is armed, an unbuilt dir is rejected, a relink doesn't destroy the
# target, and pointing at the canonical path is a no-op. POSIX symlinks here;
# the Windows junction path is covered by the backend test suite.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Extract the two helpers + the whole `UNSLOTH_LOCAL_LLAMA_CPP_DIR` if-block.
# Starts at the quantize-shim helper, ends at the first column-0 `fi` after the
# `if [ -n "${UNSLOTH_LOCAL_LLAMA_CPP_DIR..` guard (inner ifs are indented).
block="$(awk '
    /^_link_local_llama_quantize_shim\(\) \{/ {grab=1}
    grab {print}
    /^if \[ -n "\$\{UNSLOTH_LOCAL_LLAMA_CPP_DIR/ {inif=1}
    inif && /^fi$/ {exit}
' "$SETUP")"

# Self-validate the extraction so a future setup.sh refactor fails loudly here.
case "$block" in *'ln -sfn "$_RESOLVED_LOCAL" "$LLAMA_CPP_DIR"'*) : ;;
    *) echo "FAIL: link block extraction broke (no ln -sfn)"; exit 1 ;; esac
case "$block" in *'_has_local_llama_server'*) : ;;
    *) echo "FAIL: link block extraction broke (no _has_local_llama_server)"; exit 1 ;; esac

# Stub setup.sh's logging + ownership helpers, seed the vars the block reads,
# then run the extracted block and print the resulting state.
PREAMBLE='
set -u
step() { :; }; substep() { :; }; verbose_substep() { :; }
_assert_studio_owned_or_absent() { :; }
C_ERR=""
_STUDIO_HOME_IS_CUSTOM=false
_NEED_LLAMA_SOURCE_BUILD=UNSET
_SKIP_PREBUILT_INSTALL=UNSET
'
EPILOGUE='
echo "LINKED=$_LOCAL_LLAMA_CPP_LINKED"
echo "NEED_BUILD=$_NEED_LLAMA_SOURCE_BUILD"
echo "SKIP_PREBUILT=$_SKIP_PREBUILT_INSTALL"
if [ -L "$LLAMA_CPP_DIR" ]; then echo "ISLINK=1"; echo "TARGET=$(readlink "$LLAMA_CPP_DIR")"; else echo "ISLINK=0"; fi
'
SNIP="$PREAMBLE"$'\n'"$block"$'\n'"$EPILOGUE"

# run_link <local_dir> <canonical_llama_cpp_dir> -> prints state lines; RC in $RC
run_link() {
    OUT="$(env -i PATH="$PATH" HOME="$T" \
        UNSLOTH_LOCAL_LLAMA_CPP_DIR="$1" LLAMA_CPP_DIR="$2" \
        bash -c "$SNIP" 2>/dev/null)"
    RC=$?
}
val() { printf '%s\n' "$OUT" | grep "^$1=" | head -1 | cut -d= -f2-; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# Some environments (Windows git-bash without native symlinks) make `ln -s` copy
# instead of link. The symlink-identity assertions (ISLINK / readlink target)
# only run where real symlinks exist; the link/skip/no-data-loss assertions run
# everywhere, including CI (Linux), where the link path is the real one.
ln -s "$T" "$T/.symprobe" 2>/dev/null
if [ -L "$T/.symprobe" ]; then SYMLINKS=1; else SYMLINKS=0; fi
rm -rf "$T/.symprobe"

# A built external tree (CMake layout) + a flat/`make` tree (root-level binary).
# The fake binary is a shebang script so the `-x` test in _has_local_llama_server
# holds on both Linux (chmod +x) and Windows git-bash (MSYS treats #!-files as
# executable), without needing a real platform binary.
mk_exe()    { printf '#!/bin/sh\necho fake\n' > "$1"; chmod +x "$1"; }
mk_built()  { mkdir -p "$1/build/bin"; mk_exe "$1/build/bin/llama-server"; }
mk_flat()   { mkdir -p "$1"; mk_exe "$1/llama-server"; }

# 1. External CMake build -> linked, and BOTH install paths disarmed.
EXT1="$T/ext_cmake"; mk_built "$EXT1"; : > "$EXT1/keep.txt"
CANON1="$T/home1/llama.cpp"; mkdir -p "$(dirname "$CANON1")"
run_link "$EXT1" "$CANON1"
check "cmake build: linked"            "true"  "$(val LINKED)"
check "cmake build: source build off"  "false" "$(val NEED_BUILD)"
check "cmake build: prebuilt skipped"  "true"  "$(val SKIP_PREBUILT)"
if [ "$SYMLINKS" = 1 ]; then
    check "cmake build: canonical is a symlink" "1" "$(val ISLINK)"
    check "cmake build: link points at external" "$(CDPATH= cd -P -- "$EXT1" && pwd -P)" "$(val TARGET)"
else
    printf '  SKIP  cmake build: symlink-identity (no real symlinks here)\n'
fi

# 2. Flat / make tree (root-level llama-server, no build/bin) -> still linked
#    (the new layout-candidate acceptance; the old check rejected this).
EXT2="$T/ext_flat"; mk_flat "$EXT2"
CANON2="$T/home2/llama.cpp"; mkdir -p "$(dirname "$CANON2")"
run_link "$EXT2" "$CANON2"
check "flat build: linked (root-level llama-server accepted)" "true" "$(val LINKED)"

# 3. Unbuilt tree -> rejected (non-zero exit, no link created).
EXT3="$T/ext_empty"; mkdir -p "$EXT3"
CANON3="$T/home3/llama.cpp"; mkdir -p "$(dirname "$CANON3")"
run_link "$EXT3" "$CANON3"
check "unbuilt tree: rejected (exit != 0)" "yes" "$([ "$RC" -ne 0 ] && echo yes || echo no)"
check "unbuilt tree: no link left behind"  "no"  "$([ -L "$CANON3" ] && echo yes || echo no)"

# 4. Relink over a stale link must NOT destroy the (new) target's contents.
OLD="$T/ext_old"; mk_built "$OLD"
NEW="$T/ext_new"; mk_built "$NEW"; : > "$NEW/precious.txt"
CANON4="$T/home4/llama.cpp"; mkdir -p "$(dirname "$CANON4")"
ln -sfn "$OLD" "$CANON4"   # simulate a prior --with-llama-cpp-dir run
run_link "$NEW" "$CANON4"
if [ "$SYMLINKS" = 1 ]; then
    check "relink: now points at the new external" "$(CDPATH= cd -P -- "$NEW" && pwd -P)" "$(val TARGET)"
fi
check "relink: new target's contents preserved" "yes" "$([ -f "$NEW/precious.txt" ] && echo yes || echo no)"
check "relink: old target's contents preserved" "yes" "$([ -f "$OLD/build/bin/llama-server" ] && echo yes || echo no)"

# 5. Pointing at the canonical path itself is a no-op reuse: linked, not turned
#    into a self-referential symlink, contents untouched.
CANON5="$T/home5/llama.cpp"; mk_built "$CANON5"; : > "$CANON5/keep.txt"
run_link "$CANON5" "$CANON5"
check "canonical no-op: linked"             "true" "$(val LINKED)"
check "canonical no-op: not made a symlink" "0"    "$(val ISLINK)"
check "canonical no-op: contents preserved" "yes"  "$([ -f "$CANON5/keep.txt" ] && echo yes || echo no)"

echo ""
if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
