#!/usr/bin/env bash
# setup.sh installs the isolated Node under <UNSLOTH_STUDIO_HOME>, matching
# node_runtime.managed_node_dir(). Logic is extracted from setup.sh by content anchors.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blockA="$(awk '
    /^_studio_override_var=""/ {grab=1}
    grab {print}
    /_STUDIO_HOME_IS_CUSTOM=true/ {seen=1}
    seen && /^fi$/ {exit}
' "$SETUP")"
# Block B: the whole _NODE_PARENT conditional, anchored on NODE_DIR and walked back to its
# top-level `if`, so branches can be added or reordered without breaking extraction.
blockB="$(awk '
    /^if / {start = NR}
    {buf[NR] = $0}
    /^NODE_DIR="\$_NODE_PARENT\/node"/ {
        if (start == 0) exit
        for (i = start; i <= NR; i++) print buf[i]
        exit
    }
' "$SETUP")"
SNIP="$blockA"$'\n'"$blockB"$'\n''echo "$NODE_DIR"'

case "$blockA" in *"_STUDIO_HOME_IS_CUSTOM=true"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'NODE_DIR="$_NODE_PARENT/node"'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
# Walking back could land on an unrelated `if`, so require both branches by name.
case "$blockB" in *'_STUDIO_HOME_IS_CUSTOM'*) : ;; *) echo "FAIL: blockB missed the custom-home branch"; exit 1 ;; esac
case "$blockB" in *'STAGE_ROOT'*) : ;; *) echo "FAIL: blockB missed the staging branch"; exit 1 ;; esac

node_dir_for() { # HOME UNSLOTH_STUDIO_HOME STUDIO_HOME [UNSLOTH_STUDIO_STAGE_ROOT]
    env -i HOME="$1" UNSLOTH_STUDIO_HOME="$2" STUDIO_HOME="$3" \
        UNSLOTH_STUDIO_STAGE_ROOT="${4:-}" PATH="$PATH" \
        bash -c "$SNIP" 2>/dev/null | tail -1
}

# Set UNSLOTH_STUDIO_STAGE_ROOT: blockA derives STAGE_ROOT/RUNTIME_ROOT and overwrites them.
staged_node_dir_for() { # HOME UNSLOTH_STUDIO_STAGE_ROOT UNSLOTH_STUDIO_HOME
    env -i HOME="$1" UNSLOTH_STUDIO_STAGE_ROOT="$2" UNSLOTH_STUDIO_HOME="$3" \
        STUDIO_HOME="" PATH="$PATH" \
        bash -c "$SNIP" 2>/dev/null | tail -1
}

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/custom" "$T/fakehome/.unsloth/studio"
CUSTOM="$(CDPATH= cd -P -- "$T/custom" && pwd -P)"
FAKEHOME="$(CDPATH= cd -P -- "$T/fakehome" && pwd -P)"
LEGACY="$FAKEHOME/.unsloth/studio"

check "UNSLOTH_STUDIO_HOME=<custom> -> <custom>/node" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "$CUSTOM" "")"
check "STUDIO_HOME alias -> <custom>/node" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "" "$CUSTOM")"
check "UNSLOTH_STUDIO_HOME wins over STUDIO_HOME" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "$CUSTOM" "$T/fakehome")"
check "legacy-valued override -> ~/.unsloth/node sibling" "$FAKEHOME/.unsloth/node" "$(node_dir_for "$FAKEHOME" "$LEGACY" "")"
check "no override -> ~/.unsloth/node" "$FAKEHOME/.unsloth/node" "$(node_dir_for "$FAKEHOME" "" "")"
# Staged update: STAGE_ROOT outranks a custom home, so Node never lands in the live home.
mkdir -p "$T/stage"
STAGE="$(CDPATH= cd -P -- "$T/stage" && pwd -P)"
check "stage root -> <stage>/node" "$STAGE/node" "$(staged_node_dir_for "$FAKEHOME" "$STAGE" "")"
check "stage root beats a custom studio home" "$STAGE/node" "$(staged_node_dir_for "$FAKEHOME" "$STAGE" "$CUSTOM")"

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
