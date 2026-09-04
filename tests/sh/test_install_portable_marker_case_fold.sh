#!/usr/bin/env bash
# Regression test: the nested portable-marker cleanup folds case where the
# FILESYSTEM does, and only there.
#
# install.sh writes <master>/studio, but a case-insensitive filesystem opens that
# directory for a user who typed `Studio`, and getcwd() hands back the spelling
# that was typed rather than the one on disk (golang/go#20947; `pwd -P` resolves
# symlinks, not case). So UNSLOTH_STUDIO_HOME=<master>/Studio can name the very
# directory the installer created and arrive in $STUDIO_HOME spelled `Studio`.
# An exact `*/studio` match in _clear_stale_portable_marker left the parent marker
# in place while both readers went on honouring it: the normal reinstall reported
# success and the tree still resolved as portable, with UNSLOTH_PORTABLE=0
# powerless to turn it off.
#
# The fold used to be keyed on `uname` being Darwin. That is the wrong question.
# macOS is case-insensitive by DEFAULT, not by rule, and a case-sensitive APFS
# volume is exactly what an external disk carrying a portable install tends to be
# formatted as. There <master>/studio and <master>/Studio are two directories, and
# a platform-wide fold hands a separate normal install at <master>/Studio the
# nested portable sibling's master root and managed runtimes, and lets a normal
# reinstall of <master>/Studio delete the portable sibling's marker. So all four
# copies of the rule -- install.sh here, studio/setup.sh's _setup_portable_mode,
# storage_roots._inherits_parent_portable_marker and the CLI's twin of it -- now
# ask whether the two paths identify the same directory (st_dev/st_ino, spelled
# `-ef` in shell and Path.samefile in Python) instead of what platform this is.
#
# EVERY case below is therefore run under BOTH a faked Darwin `uname` and the real
# one, and asserted to give the SAME answer. A platform gate creeping back in
# fails here rather than on somebody's external disk.
#
# On the two directories/one directory distinction: a case-insensitive volume
# cannot be created on the CI host, so `studio` is made a symlink to the real
# `Studio` instead. That gives the two spellings the one property the predicate
# actually tests -- same st_dev/st_ino -- without pretending to simulate
# case-insensitive name lookup in general, which the predicate never relies on
# because it only ever probes the literal name `studio`.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
BACKEND="$ROOT/studio/backend"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Anchor uniqueness: a second line matching any of these would silently lift a
# different block and leave every assertion below testing the wrong text. Whole
# lines compared as literals, which is what the awk anchors below amount to.
count_exact() { awk -v s="$1" '$0 == s {n++} END {print n+0}' "$INSTALL"; }
while IFS= read -r anchor; do
    n=$(count_exact "$anchor")
    if [ "$n" -ne 1 ]; then
        printf 'FAIL: anchor [%s] matches %s lines, expected 1\n' "$anchor" "$n"; exit 1
    fi
done <<EOF
# ── Parse flags ──
_resolve_studio_destinations() {
_clear_stale_portable_marker() {
EOF

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockD="$(blk '/^_clear_stale_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockD" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
# The two halves of the rule. Without the fold the cased checks below would pass
# for the wrong reason; without the identity probe the case-sensitive checks would.
case "$blockD" in *"tr '[:upper:]' '[:lower:]'"*) : ;; *) echo "FAIL: blockD lost the case fold"; exit 1 ;; esac
case "$blockD" in *'-ef "$_spm_parent/studio"'*) : ;; *) echo "FAIL: blockD lost the identity probe"; exit 1 ;; esac
# The negative assertions run against the CODE only. The prose above the probe
# names `uname`, `readlink -f` and `realpath` to explain why none of them is
# consulted, and matching that would fail the block for saying so.
blockD_code="$(printf '%s\n' "$blockD" | grep -v '^[[:space:]]*#')"
case "$blockD_code" in *'-ef'*) : ;; *) echo "FAIL: the identity probe is in a comment, not the code"; exit 1 ;; esac
# ...and the platform gate must stay gone.
case "$blockD_code" in *'uname'*) echo "FAIL: blockD gates the fold on the platform again"; exit 1 ;; *) : ;; esac
# ${var,,} is a bashism and install.sh runs under macOS /bin/sh; readlink -f and
# realpath are not on the BSD side either.
case "$blockD_code" in *',,}'*) echo "FAIL: blockD uses a bash-only case fold"; exit 1 ;; *) : ;; esac
case "$blockD_code" in *'readlink -f'*|*'realpath '*) echo "FAIL: blockD uses a tool macOS lacks"; exit 1 ;; *) : ;; esac
# The fold changes the leaf, so stripping the literal `/studio` would derive the
# wrong parent for `Studio` and the removal would miss (or hit the wrong dir).
case "$blockD_code" in *'${STUDIO_HOME%/studio}'*) echo "FAIL: blockD still strips a literal /studio"; exit 1 ;; *) : ;; esac
# Lifted out on its own below, so it may not call a helper defined elsewhere.
case "$blockD_code" in *'_portable_marker_prior'*) echo "FAIL: blockD calls a helper it does not define"; exit 1 ;; *) : ;; esac

# set -e is on: a bare `[ cond ] && action` inside the block would kill the run.
SNIP='set -e
C_WARN=""
substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockD"'
_clear_stale_portable_marker
printf "reached|%s\n" "$STUDIO_HOME"'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home"

# A uname earlier on PATH than the real one. Only uname is shadowed; tr, grep and
# the rest still resolve normally. Every scenario runs under this AND without it.
mkdir -p "$T/fakebin"
printf '#!/bin/sh\nprintf "Darwin\\n"\n' > "$T/fakebin/uname"
chmod +x "$T/fakebin/uname"

new_root() { mktemp -d "$T/root.XXXXXX"; }

# A nested portable tree: <master>/<leaf> holding the venv, marker one level up.
build() { # master leaf
    mkdir -p "$1/$2/unsloth_studio/bin" "$1/bin" "$1/share"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
}

# The same tree, plus a lowercase alias so both spellings are ONE directory --
# what a case-insensitive volume gives you, and all the predicate looks at.
build_case_insensitive() { # master
    build "$1" Studio
    ln -s Studio "$1/studio"
}

convert() { # master leaf platform
    _cv_path="$PATH"
    [ "$3" = darwin ] && _cv_path="$T/fakebin:$PATH"
    env -i HOME="$T/home" PATH="$_cv_path" USER="${USER:-tester}" \
        UNSLOTH_STUDIO_HOME="$1/$2" bash -c "$SNIP" _ > "$T/out" 2>"$T/err"
    _cv_rc=$?
    if [ "$_cv_rc" -ne 0 ] || ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet aborted (rc=%s)\n%s\n%s\n' \
            "$_cv_rc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
}

marker_state() { # dir
    if [ -f "$1/.unsloth-portable-root" ]; then printf present; else printf gone; fi
}

echo
echo "[1] one directory under two spellings is still this install, on either platform"
for plat in darwin linux; do
    M1="$(new_root)"; build_case_insensitive "$M1"
    convert "$M1" Studio "$plat"
    check "$plat: Studio names the same dir, so the parent marker drops" \
        gone "$(marker_state "$M1")"
done

echo
echo "[2] two directories are two installs, on either platform"
# THE REGRESSION THIS FIX IS FOR. <master>/studio is a genuine nested portable
# install; <master>/Studio is a separate normal one that arrived through
# UNSLOTH_STUDIO_HOME. Reinstalling the second must not retire the first's marker.
for plat in darwin linux; do
    M2="$(new_root)"; build "$M2" studio
    mkdir -p "$M2/Studio/unsloth_studio/bin"
    convert "$M2" Studio "$plat"
    check "$plat: a distinct Studio keeps the portable sibling's marker" \
        present "$(marker_state "$M2")"
done

echo
echo "[3] a cased leaf with no lowercase sibling at all keeps it too"
# Nothing at <master>/studio to be the same directory AS, so the marker names a
# directory that is not this one. The probe declines, which leaves it in place.
for plat in darwin linux; do
    M3="$(new_root)"; build "$M3" Studio
    convert "$M3" Studio "$plat"
    check "$plat: no lowercase sibling, marker stays" present "$(marker_state "$M3")"
done

echo
echo "[4] the fold is not the whole rule"
# Lowercase must still work, or the identity probe has replaced the match.
for plat in darwin linux; do
    M4="$(new_root)"; build "$M4" studio
    convert "$M4" studio "$plat"
    check "$plat: lowercase studio still drops it" gone "$(marker_state "$M4")"
done
# ...and it must not turn every child of a marked root into a match. A name that
# does not fold to `studio` never reaches the identity probe at all, which is why
# the cheap name test stays in front of it.
for plat in darwin linux; do
    M5="$(new_root)"; build "$M5" studio
    mkdir -p "$M5/Other/unsloth_studio"
    convert "$M5" Other "$plat"
    check "$plat: a sibling keeps the portable root's marker" present "$(marker_state "$M5")"
done

echo
echo "[5] the parent it derives is the marked root, not a folded path"
# `${STUDIO_HOME%/studio}` cannot strip `Studio`, so a wrong derivation here
# either misses the marker (case 1 fails) or walks somewhere else entirely.
M6="$(new_root)"; build_case_insensitive "$M6"
mkdir -p "$M6/Studio/nested"
printf 'neighbour\n' > "$M6/Studio/.unsloth-portable-root"
convert "$M6" Studio darwin
check "darwin: the parent marker goes" gone "$(marker_state "$M6")"
check "darwin: the marker inside the studio dir goes too (flat slot)" gone "$(marker_state "$M6/Studio")"

echo
echo "[6] the flat layout is untouched by any of this"
M7="$(new_root)"
mkdir -p "$M7/unsloth_studio/bin"
printf '%s\n' "$M7" > "$M7/.unsloth-portable-root"
env -i HOME="$T/home" PATH="$T/fakebin:$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$M7" bash -c "$SNIP" _ > "$T/out" 2>"$T/err" || true
check "darwin: a flat root still loses its own marker" gone "$(marker_state "$M7")"

echo
echo "[7] the runtime half: both Python copies must agree about the tree"
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys, unittest.mock
sys.path.insert(0, os.environ["_BACKEND"])
sys.path.insert(0, os.environ["_REPO"])
# Imported BEFORE the platform patch: patching sys.platform to darwin makes
# urllib.request try to import the macOS-only _scproxy at import time. Both
# predicates read sys.platform at CALL time, so the patch still covers them --
# and the point of this test is that neither of them looks at it any more.
from utils.paths import storage_roots as sr
from unsloth_cli.commands import studio as cli
from pathlib import Path
root = Path(os.environ["UNSLOTH_STUDIO_HOME"])
with unittest.mock.patch.object(sys, "platform", os.environ["_PLATFORM"]):
    print("__JSON__" + json.dumps({
        "portable": sr.portable_mode(),
        "sr_inherits": sr._inherits_parent_portable_marker(root),
        "cli_inherits": cli._inherits_parent_portable_marker(root),
        "sr_parent": str(sr._parent_portable_root(root) or ""),
        "cli_parent": str(cli._parent_portable_root(root) or ""),
    }))
PYEOF
    field() { # studio_home platform field
        _pout=$(env -i HOME="$T/home" PATH="$PATH" _BACKEND="$BACKEND" _REPO="$ROOT" \
            _PLATFORM="$2" UNSLOTH_STUDIO_HOME="$1" python3 "$PROBE" 2>"$T/perr")
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            printf '  FAIL  storage_roots probe produced no output\n%s\n' "$(cat "$T/perr")" >&2
            fails=$((fails + 1)); printf 'probe-failed'; return 0
        fi
        printf '%s' "$_pjson" | _F="$3" python3 -c \
            'import json,os,sys; print(str(json.load(sys.stdin)[os.environ["_F"]]).lower())'
    }
    # The two copies never disagree, whatever the shape.
    agree() { # label studio_home platform expected
        for _f in sr_inherits cli_inherits; do
            check "$1 ($_f)" "$4" "$(field "$2" "$3" "$_f")"
        done
    }

    for plat in darwin linux; do
        # Same directory, two spellings: adopted, and the installer agrees.
        M8="$(new_root)"; build_case_insensitive "$M8"
        agree "$plat: one dir/two spellings inherits" "$M8/Studio" "$plat" true
        check "$plat: ...and reads as portable" true "$(field "$M8/Studio" "$plat" portable)"
        convert "$M8" Studio "$plat"
        check "$plat: ...and reads as plain once converted" false \
            "$(field "$M8/Studio" "$plat" portable)"

        # Two directories: the separate normal install adopts nothing, and the
        # nested portable one beside it keeps working.
        M9="$(new_root)"; build "$M9" studio
        mkdir -p "$M9/Studio/unsloth_studio/bin"
        agree "$plat: a distinct Studio does not inherit" "$M9/Studio" "$plat" false
        check "$plat: ...and reads as plain" false "$(field "$M9/Studio" "$plat" portable)"
        agree "$plat: the nested studio beside it still does" "$M9/studio" "$plat" true
        check "$plat: ...and still reads as portable" true "$(field "$M9/studio" "$plat" portable)"
    done
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

echo
if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
