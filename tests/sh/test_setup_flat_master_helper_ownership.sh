#!/usr/bin/env bash
# Regression test: the managed-helper ownership guards in setup.sh must fire for a
# portable MASTER root even when that root is spelled as the legacy ~/.unsloth/studio.
#
# `install.sh --portable` over an existing default install (UNSLOTH_STUDIO_HOME=
# ~/.unsloth/studio) builds the FLAT layout, where the master root IS the Studio root.
# node/, llama.cpp/ and whisper.cpp/ then hang off ~/.unsloth/studio, but the spelling
# stays the legacy one, so _STUDIO_HOME_IS_CUSTOM stays false and every guard keyed on it
# was skipped. install_node_prebuilt._swap_into_place renames the existing directory to
# .node.old-$$ and then shutil.rmtree()s it, so a user's own ~/.unsloth/studio/node was
# permanently deleted (prebuilt_core.swap_into_place and llama.cpp's activate_install_tree
# move and delete the same way).
#
# The key is the Studio root BEING the master root, not portable mode, matching
# sd_cpp_engine._root_is_portable_master: `install.sh --root ~/.unsloth` builds a NESTED
# master whose helpers are ~/.unsloth/node, ~/.unsloth/llama.cpp and ~/.unsloth/whisper.cpp
# -- exactly the unmarked directories every pre-marker default install already carries --
# so demanding a marker there would refuse to replace trees Unsloth genuinely owns.
#
# Extracted from setup.sh by content anchor, never by line number, and every extraction is
# self-validated so a refactor fails loudly here instead of going silently inert.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Block A: _setup_trim_ws / _setup_abs_path, the UNSLOTH_HOME + studio-override
# normalization, _STUDIO_HOME_IS_CUSTOM, and the ownership-strictness derivation. Same
# start anchor as test_studio_home_node_dir.sh so the two cannot drift; the end anchor is
# extended past _STUDIO_HOME_IS_CUSTOM to the strictness flag this file is about.
blockA="$(awk '
    /^_studio_override_var=""/ {grab=1}
    grab {print}
    /^    _STUDIO_STRICT_OWNERSHIP=true$/ {seen=1}
    seen && /^fi$/ {exit}
' "$SETUP")"

# Block B: the whole _NODE_PARENT conditional -> NODE_DIR. Anchored on the NODE_DIR
# assignment and walked BACK to the top-level `if`, so branches can be added or reordered.
blockB="$(awk '
    /^if / {start = NR}
    {buf[NR] = $0}
    /^NODE_DIR="\$_NODE_PARENT\/node"/ {
        if (start == 0) exit
        for (i = start; i <= NR; i++) print buf[i]
        exit
    }
' "$SETUP")"

# Block C: the real Node ownership guard, as setup.sh runs it before the installer.
# Anchored on the guarded call and walked BACK to the enclosing `if`, like blockB, so the
# condition can be rewritten without silently emptying this.
blockC="$(awk '
    {buf[NR] = $0}
    /^    if \[ / && !seen {start = NR}
    /_assert_studio_owned_or_absent "\$NODE_DIR"/ {seen = 1}
    seen && /^    fi$/ {
        if (start == 0) exit
        for (i = start; i <= NR; i++) print buf[i]
        exit
    }
' "$SETUP")"

case "$blockA" in *"_STUDIO_HOME_IS_CUSTOM=true"*) : ;;
    *) echo "FAIL: blockA lost the layout flag"; exit 1 ;; esac
case "$blockA" in *"_STUDIO_STRICT_OWNERSHIP=true"*) : ;;
    *) echo "FAIL: blockA lost the strictness flag"; exit 1 ;; esac
case "$blockA" in *"_STUDIO_ROOT_IS_MASTER_ROOT=true"*) : ;;
    *) echo "FAIL: blockA lost the master-root derivation"; exit 1 ;; esac
case "$blockA" in *'.unsloth-portable-root'*) : ;;
    *) echo "FAIL: blockA lost the portable-marker fallback"; exit 1 ;; esac
case "$blockA" in *'.unsloth-master-root'*) : ;;
    *) echo "FAIL: blockA lost the nested master-root record check"; exit 1 ;; esac
case "$blockB" in *'NODE_DIR="$_NODE_PARENT/node"'*) : ;;
    *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockC" in *'_assert_studio_owned_or_absent "$NODE_DIR"'*) : ;;
    *) echo "FAIL: blockC lost the Node ownership guard"; exit 1 ;; esac
case "$blockC" in *'_STUDIO_STRICT_OWNERSHIP'*) : ;;
    *) echo "FAIL: blockC is not gated on the strictness flag"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# setup.sh runs install steps at load, so lift the real ownership helpers instead.
python3 - "$SETUP" "$T/helpers.sh" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = []
for name in ("_studio_owned_adoptable", "_studio_dir_unsearchable", "_studio_dir_unreadable",
             "_studio_rstrip_slash", "_report_denied_ancestor", "_path_access_denied",
             "_assert_studio_owned_or_absent"):
    i = src.index(name + "() {")
    out.append(src[i:src.index("\n}\n", i) + 3])
pathlib.Path(sys.argv[2]).write_text("\n".join(out))
PY
grep -qF 'UNSLOTH_NODE_PREBUILT_INFO.json' "$T/helpers.sh" ||
    { echo "FAIL: helper extraction lost the Node adoption evidence"; exit 1; }

PREAMBLE='
set -u
step() { :; }; substep() { :; }; verbose_substep() { :; }
_clear_webview_caches() { :; }
setup_fail() { exit "$1"; }
C_ERR=""; C_WARN=""
'
EPILOGUE='
echo "STRICT=$_STUDIO_STRICT_OWNERSHIP"
echo "NODE_DIR=$NODE_DIR"
. "$HELPERS"
mkdir -p "$_NODE_PARENT"
'"$blockC"'
echo "VERDICT=ACCEPTED"
'
SNIP="$PREAMBLE"$'\n'"$blockA"$'\n'"$blockB"$'\n'"$EPILOGUE"

drive() { # HOME UNSLOTH_HOME UNSLOTH_STUDIO_HOME
    OUT="$(env -i PATH="$PATH" HELPERS="$T/helpers.sh" HOME="$1" UNSLOTH_HOME="$2" \
        UNSLOTH_STUDIO_HOME="$3" STUDIO_HOME="" UNSLOTH_STUDIO_STAGE_ROOT="" \
        bash -c "$SNIP" 2>/dev/null)"
}
val() { printf '%s\n' "$OUT" | grep "^$1=" | head -1 | cut -d= -f2-; }
verdict() { case "$OUT" in *VERDICT=ACCEPTED*) echo ACCEPTED ;; *) echo REFUSED ;; esac; }

# A default install: the venv, and a user's own unowned node/ at the place that layout
# puts the bundled runtime.
mk_home() { # dir node_rel
    mkdir -p "$1/.unsloth/studio/unsloth_studio/bin" "$1/$2/lib"
    printf 'home = /usr\n' > "$1/.unsloth/studio/unsloth_studio/pyvenv.cfg"
    printf 'three days of work\n' > "$1/$2/MY_NOTES.txt"
}

H_LEGACY="$T/h_legacy"; mk_home "$H_LEGACY" ".unsloth/node"
H_FLAT="$T/h_flat";     mk_home "$H_FLAT" ".unsloth/studio/node"
: > "$H_FLAT/.unsloth/studio/.unsloth-portable-root"
H_NEST="$T/h_nest";     mk_home "$H_NEST" ".unsloth/node"
: > "$H_NEST/.unsloth/.unsloth-portable-root"
printf '%s\n' "$H_NEST/.unsloth" > "$H_NEST/.unsloth/studio/.unsloth-master-root"
H_CUSTOM="$T/h_custom";  mk_home "$H_CUSTOM" ".unsloth/node"
mkdir -p "$T/custom/node"; printf 'mine\n' > "$T/custom/node/MY_NOTES.txt"

# ── the four layouts ──

# (a) Plain legacy non-portable. ~/.unsloth/node is the historical default cache: every
# pre-marker install has one and none of them are marked, so this must stay permissive.
drive "$H_LEGACY" "" ""
check "legacy default: not strict"        "false" "$(val STRICT)"
check "legacy default: node at ~/.unsloth" "$H_LEGACY/.unsloth/node" "$(val NODE_DIR)"
check "legacy default: unowned node still replaced" "ACCEPTED" "$(verdict)"

# (b) Plain custom root. Strict before this change and strict after it.
drive "$H_CUSTOM" "" "$T/custom"
check "custom root: strict"               "true" "$(val STRICT)"
check "custom root: unowned node refused" "REFUSED" "$(verdict)"

# (c) install.sh --root ~/.unsloth: a NESTED master that legitimately owns ~/.unsloth/node,
# beside its own llama.cpp and whisper.cpp. Keying on portable mode would break this.
drive "$H_NEST" "$H_NEST/.unsloth" "$H_NEST/.unsloth/studio"
check "nested legacy master: not strict"  "false" "$(val STRICT)"
check "nested legacy master: node at ~/.unsloth" "$H_NEST/.unsloth/node" "$(val NODE_DIR)"
check "nested legacy master: unowned node still replaced" "ACCEPTED" "$(verdict)"

# (d) THE BUG. UNSLOTH_STUDIO_HOME=~/.unsloth/studio UNSLOTH_PORTABLE=1 makes that
# directory the flat master, so node/ lands INSIDE it -- a level no legacy install used.
drive "$H_FLAT" "$H_FLAT/.unsloth/studio" "$H_FLAT/.unsloth/studio"
check "flat legacy master: strict"        "true" "$(val STRICT)"
check "flat legacy master: node inside the root" \
    "$H_FLAT/.unsloth/studio/node" "$(val NODE_DIR)"
check "flat legacy master: unowned node refused" "REFUSED" "$(verdict)"

# (e) The same install with nothing in the environment (a bare `bash setup.sh`, or the
# CLI's recovery path): the on-disk marker at the root has to carry it.
drive "$H_FLAT" "" "$H_FLAT/.unsloth/studio"
check "flat legacy master, marker only: strict" "true" "$(val STRICT)"

# (f) And the nested one must NOT be dragged in by its own marker, which sits one level up
# and is outranked by the .unsloth-master-root record inside the Studio root.
drive "$H_NEST" "" "$H_NEST/.unsloth/studio"
check "nested legacy master, marker only: not strict" "false" "$(val STRICT)"

# ── non-collapse: a tree Unsloth genuinely owns is still replaced ──
# Without these the fix could degrade into "never replace anything" and every update would
# break with "not marked as an Unsloth-owned Node install".

H_OWNED="$T/h_owned"; mk_home "$H_OWNED" ".unsloth/studio/node"
: > "$H_OWNED/.unsloth/studio/.unsloth-portable-root"
: > "$H_OWNED/.unsloth/studio/node/.unsloth-studio-owned"
drive "$H_OWNED" "$H_OWNED/.unsloth/studio" "$H_OWNED/.unsloth/studio"
check "flat legacy master: marked node replaced" "ACCEPTED" "$(verdict)"

# A pre-marker install of ours: no .unsloth-studio-owned, but installer metadata only our
# own install_node_prebuilt.py writes. _studio_owned_adoptable adopts it.
H_ADOPT="$T/h_adopt"; mk_home "$H_ADOPT" ".unsloth/studio/node"
: > "$H_ADOPT/.unsloth/studio/.unsloth-portable-root"
: > "$H_ADOPT/.unsloth/studio/node/UNSLOTH_NODE_PREBUILT_INFO.json"
drive "$H_ADOPT" "$H_ADOPT/.unsloth/studio" "$H_ADOPT/.unsloth/studio"
check "flat legacy master: our own unmarked node adopted" "ACCEPTED" "$(verdict)"
[ -f "$H_ADOPT/.unsloth/studio/node/.unsloth-studio-owned" ]
check "flat legacy master: adoption writes the marker" "0" "$?"

# An absent node/ is the fresh-install case and must never be refused.
H_FRESH="$T/h_fresh"
mkdir -p "$H_FRESH/.unsloth/studio/unsloth_studio/bin"
: > "$H_FRESH/.unsloth/studio/.unsloth-portable-root"
drive "$H_FRESH" "$H_FRESH/.unsloth/studio" "$H_FRESH/.unsloth/studio"
check "flat legacy master: fresh install accepted" "ACCEPTED" "$(verdict)"

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
