#!/usr/bin/env bash
# Regression test: studio/setup.sh normalizes UNSLOTH_HOME once, up front, before
# it derives any runtime path from it.
#
# UNSLOTH_HOME is the portable master root and node/, llama.cpp/ and whisper.cpp/
# hang off it. install.sh trims, tilde-expands and resolves the same variable
# (_trim_ws + `CDPATH= cd -P -- ... && pwd -P`) and storage_roots.py strips it,
# but setup.sh used to read it raw behind a bare `[ -n ... ]`. A whitespace-only
# value therefore beat a real custom root and installed Node into a directory
# literally named " " in the working directory, and a relative value installed it
# under the working directory -- a DIFFERENT one for Node and for llama.cpp, since
# `cd "$SCRIPT_DIR"` runs between the two derivations.
#
# `unsloth studio update` reaches this: _run_setup_script() hands setup.sh
# {**os.environ}, and _ensure_studio_env_exported() only fills UNSLOTH_HOME when
# it is falsy, so " " survives; pin_relative_overrides() in _system_dir_guard.py
# only runs from a Windows system directory, so a relative value survives on
# POSIX.
#
# The real blocks are extracted from setup.sh by content anchors (not line
# numbers) and run against a hermetic fake HOME from a hermetic working
# directory, one mktemp -d per case.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Block A: the override/normalization prologue, up to _STUDIO_HOME_IS_CUSTOM.
# Same extraction as test_studio_home_node_dir.sh, so the two cannot drift.
blockA="$(awk '
    /^_studio_override_var=""/ {grab=1}
    grab {print}
    /_STUDIO_HOME_IS_CUSTOM=true/ {seen=1}
    seen && /^fi$/ {exit}
' "$SETUP")"
# Block B: the whole _NODE_PARENT conditional -> NODE_DIR, anchored on the
# NODE_DIR assignment and walked back to the top-level `if` that feeds it, so
# branches can be added or reordered without touching this file.
blockB="$(awk '
    /^if / {start = NR}
    {buf[NR] = $0}
    /^NODE_DIR="\$_NODE_PARENT\/node"/ {
        if (start == 0) exit
        for (i = start; i <= NR; i++) print buf[i]
        exit
    }
' "$SETUP")"
# Block C: the llama.cpp derivation, _PORTABLE_ROOT through LLAMA_CPP_DIR.
blockC="$(awk '
    /^_PORTABLE_ROOT="\$\{UNSLOTH_HOME:-\}"/ {grab=1}
    grab {print}
    /^LLAMA_CPP_DIR="\$UNSLOTH_HOME\/llama\.cpp"/ {exit}
' "$SETUP")"

# Self-validate every extraction so a setup.sh refactor fails loudly here instead
# of quietly testing an empty string.
case "$blockA" in *"_STUDIO_HOME_IS_CUSTOM=true"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockA" in *'UNSLOTH_HOME'*) : ;; *) echo "FAIL: blockA no longer normalizes UNSLOTH_HOME"; exit 1 ;; esac
case "$blockB" in *'NODE_DIR="$_NODE_PARENT/node"'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockB" in *'UNSLOTH_HOME'*) : ;; *) echo "FAIL: blockB missed the UNSLOTH_HOME branch"; exit 1 ;; esac
case "$blockC" in *'LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"'*) : ;; *) echo "FAIL: blockC extraction broke"; exit 1 ;; esac
case "$blockC" in *'mkdir -p "$UNSLOTH_HOME"'*) : ;; *) echo "FAIL: blockC missed the root mkdir"; exit 1 ;; esac

# Node only. Runs entirely in $1, which is also where a raw relative root lands.
node_dir_for() { # cwd HOME UNSLOTH_HOME [UNSLOTH_STUDIO_HOME]
    ( cd "$1" || exit 1
      env -i HOME="$2" UNSLOTH_HOME="$3" UNSLOTH_STUDIO_HOME="${4:-}" \
          STUDIO_HOME="" UNSLOTH_STUDIO_STAGE_ROOT="" PATH="$PATH" \
          bash -c "$blockA
$blockB
echo \"\$NODE_DIR\"" 2>/dev/null | tail -1 )
}

# Node AND llama.cpp, with a `cd` in between: setup.sh really does cd into
# studio/ after the Node block and before the llama.cpp block, so a root that is
# still relative at that point names two different directories in one run.
node_and_llama_dir_for() { # cwd cd_target HOME UNSLOTH_HOME
    ( cd "$1" || exit 1
      env -i HOME="$3" UNSLOTH_HOME="$4" UNSLOTH_STUDIO_HOME="" \
          STUDIO_HOME="" UNSLOTH_STUDIO_STAGE_ROOT="" PATH="$PATH" \
          bash -c "$blockA
$blockB
cd \"\$1\" || exit 1
$blockC
echo \"\$NODE_DIR\"
echo \"\$LLAMA_CPP_DIR\"" _ "$2" 2>/dev/null | tail -2 )
}

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/fakehome" "$T/custom" "$T/elsewhere"
FAKEHOME="$(CDPATH= cd -P -- "$T/fakehome" && pwd -P)"
CUSTOM="$(CDPATH= cd -P -- "$T/custom" && pwd -P)"
ELSEWHERE="$(CDPATH= cd -P -- "$T/elsewhere" && pwd -P)"

# 1. Whitespace-only UNSLOTH_HOME is "unset", so a real custom studio home wins,
#    instead of Node landing in a directory named " " under the working dir.
C1="$(mktemp -d)"; C1="$(CDPATH= cd -P -- "$C1" && pwd -P)"
check "whitespace-only UNSLOTH_HOME yields to the custom studio home" \
    "$CUSTOM/node" "$(node_dir_for "$C1" "$FAKEHOME" " " "$CUSTOM")"

# 2. ... and with no other override it falls all the way back to ~/.unsloth.
C2="$(mktemp -d)"; C2="$(CDPATH= cd -P -- "$C2" && pwd -P)"
check "whitespace-only UNSLOTH_HOME falls back to ~/.unsloth" \
    "$FAKEHOME/.unsloth/node" "$(node_dir_for "$C2" "$FAKEHOME" "	 " "")"

# 3. A relative root is anchored to the launch directory, absolutely.
C3="$(mktemp -d)"; C3="$(CDPATH= cd -P -- "$C3" && pwd -P)"
check "relative UNSLOTH_HOME is anchored to the launch directory" \
    "$C3/rel/root/node" "$(node_dir_for "$C3" "$FAKEHOME" "rel/root" "")"

# 4. ... and stays that one directory across setup.sh's own mid-run `cd`.
C4="$(mktemp -d)"; C4="$(CDPATH= cd -P -- "$C4" && pwd -P)"
check "relative UNSLOTH_HOME names ONE root either side of the mid-run cd" \
    "$C4/rel/root/node
$C4/rel/root/llama.cpp" \
    "$(node_and_llama_dir_for "$C4" "$ELSEWHERE" "$FAKEHOME" "rel/root")"

# 5. Nothing is created under the directory setup.sh was cd'd into. blockC runs a
#    real `mkdir -p "$UNSLOTH_HOME"`, which is what used to litter the tree.
check "the mid-run cd target is left empty" "" "$(ls -A "$ELSEWHERE")"

# 6. Leading/trailing whitespace around a real absolute root is trimmed, not
#    carried into the path.
C6="$(mktemp -d)"; C6="$(CDPATH= cd -P -- "$C6" && pwd -P)"
check "surrounding whitespace is trimmed off an absolute root" \
    "$CUSTOM/node" "$(node_dir_for "$C6" "$FAKEHOME" "  $CUSTOM  " "")"

# 7. Tilde, which is not expanded on a quoted assignment, matches install.sh.
C7="$(mktemp -d)"; C7="$(CDPATH= cd -P -- "$C7" && pwd -P)"
check "~/ is expanded like install.sh does" \
    "$FAKEHOME/portable/node" "$(node_dir_for "$C7" "$FAKEHOME" "~/portable" "")"

# 8. setup.sh legitimately runs before the tree exists, so an absolute root that
#    is not there yet must survive verbatim rather than be dropped or truncated.
C8="$(mktemp -d)"; C8="$(CDPATH= cd -P -- "$C8" && pwd -P)"
check "a nonexistent absolute root is preserved" \
    "$C8/not/created/yet/node" "$(node_dir_for "$C8" "$FAKEHOME" "$C8/not/created/yet" "")"

# 9. An existing absolute root is resolved through symlinks, matching the
#    _RESOLVED_LOCAL compare further down setup.sh.
C9="$(mktemp -d)"; C9="$(CDPATH= cd -P -- "$C9" && pwd -P)"
ln -s "$CUSTOM" "$C9/link"
check "a symlinked root resolves to its real path" \
    "$CUSTOM/node" "$(node_dir_for "$C9" "$FAKEHOME" "$C9/link" "")"

rm -rf "$C1" "$C2" "$C3" "$C4" "$C6" "$C7" "$C8" "$C9"
if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
