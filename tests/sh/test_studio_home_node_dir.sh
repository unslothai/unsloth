#!/usr/bin/env bash
# Regression test: setup.sh installs the isolated Node under <UNSLOTH_STUDIO_HOME>
# (or the STUDIO_HOME alias), matching node_runtime.managed_node_dir(). Extracts
# the real STUDIO_HOME + NODE_DIR logic from setup.sh by content anchors (not line
# numbers) and runs it against a hermetic fake HOME for each override case.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Block A: studio override -> STUDIO_HOME -> _STUDIO_HOME_IS_CUSTOM.
blockA="$(awk '
    /^_studio_override_var=""/ {grab=1}
    grab {print}
    /_STUDIO_HOME_IS_CUSTOM=true/ {seen=1}
    seen && /^fi$/ {exit}
' "$SETUP")"
# Block B: the whole _NODE_PARENT conditional -> NODE_DIR.
#
# Anchored on the NODE_DIR assignment and walked BACK to the top-level `if` that
# feeds it, rather than on one branch's text. #9890 added a $STAGE_ROOT branch
# above the custom-home one, which demoted `if [ "$_STUDIO_HOME_IS_CUSTOM" ...`
# to an `elif` and left the old anchor matching nothing: blockB came out empty
# and this file failed on main for a setup.sh change that was entirely correct.
# Branches can now be added, reordered or removed without touching this.
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

# Self-validate the extraction so a future setup.sh refactor fails loudly here.
case "$blockA" in *"_STUDIO_HOME_IS_CUSTOM=true"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'NODE_DIR="$_NODE_PARENT/node"'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
# Walking back to the nearest `if` could land on an unrelated conditional, which
# would extract cleanly and then test nothing, so require both branches by name.
case "$blockB" in *'_STUDIO_HOME_IS_CUSTOM'*) : ;; *) echo "FAIL: blockB missed the custom-home branch"; exit 1 ;; esac
case "$blockB" in *'STAGE_ROOT'*) : ;; *) echo "FAIL: blockB missed the staging branch"; exit 1 ;; esac

# Block A derives STAGE_ROOT from UNSLOTH_STUDIO_STAGE_ROOT.
node_dir_for() { # HOME UNSLOTH_STUDIO_HOME STUDIO_HOME [UNSLOTH_STUDIO_STAGE_ROOT]
    env -i HOME="$1" UNSLOTH_STUDIO_HOME="$2" STUDIO_HOME="$3" \
        UNSLOTH_STUDIO_STAGE_ROOT="${4:-}" PATH="$PATH" \
        bash -c "$SNIP" 2>/dev/null | tail -1
}

# Drives UNSLOTH_STUDIO_STAGE_ROOT, the public knob: blockA derives STAGE_ROOT
# and RUNTIME_ROOT from it, so setting those two directly gets overwritten.
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

# 1. UNSLOTH_STUDIO_HOME = custom dir -> <custom>/node
check "UNSLOTH_STUDIO_HOME=<custom> -> <custom>/node" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "$CUSTOM" "")"
# 2. STUDIO_HOME alias = custom dir -> <custom>/node
check "STUDIO_HOME alias -> <custom>/node" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "" "$CUSTOM")"
# 3. UNSLOTH_STUDIO_HOME wins over STUDIO_HOME
check "UNSLOTH_STUDIO_HOME wins over STUDIO_HOME" "$CUSTOM/node" "$(node_dir_for "$FAKEHOME" "$CUSTOM" "$T/fakehome")"
# 4. Override = legacy default -> sibling ~/.unsloth/node
check "legacy-valued override -> ~/.unsloth/node sibling" "$FAKEHOME/.unsloth/node" "$(node_dir_for "$FAKEHOME" "$LEGACY" "")"
# 5. No override -> ~/.unsloth/node
check "no override -> ~/.unsloth/node" "$FAKEHOME/.unsloth/node" "$(node_dir_for "$FAKEHOME" "" "")"
# 6-7. Staged update (#9890): STAGE_ROOT sends Node under RUNTIME_ROOT, and it
# outranks a custom studio home, so a background update never writes its Node
# into the home the running Studio is serving from.
mkdir -p "$T/stage"
STAGE="$(CDPATH= cd -P -- "$T/stage" && pwd -P)"
check "stage root -> <stage>/node" "$STAGE/node" "$(staged_node_dir_for "$FAKEHOME" "$STAGE" "")"
check "stage root beats a custom studio home" "$STAGE/node" "$(staged_node_dir_for "$FAKEHOME" "$STAGE" "$CUSTOM")"

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
