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
# Block B: _STUDIO_HOME_IS_CUSTOM -> _NODE_PARENT -> NODE_DIR.
blockB="$(awk '
    /^if \[ "\$_STUDIO_HOME_IS_CUSTOM" = true \]; then/ {grab=1}
    grab {print}
    /^NODE_DIR="\$_NODE_PARENT\/node"/ {exit}
' "$SETUP")"
SNIP="$blockA"$'\n'"$blockB"$'\n''echo "$NODE_DIR"'

# Self-validate the extraction so a future setup.sh refactor fails loudly here.
case "$blockA" in *"_STUDIO_HOME_IS_CUSTOM=true"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'NODE_DIR="$_NODE_PARENT/node"'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac

node_dir_for() { # HOME UNSLOTH_STUDIO_HOME STUDIO_HOME
    env -i HOME="$1" UNSLOTH_STUDIO_HOME="$2" STUDIO_HOME="$3" PATH="$PATH" \
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

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
