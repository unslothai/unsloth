#!/usr/bin/env bash
# Regression test: the nested portable-marker cleanup folds case where the
# filesystem does.
#
# install.sh writes <master>/studio, but macOS ships a case-insensitive
# filesystem and getcwd() hands back the spelling that was typed rather than the
# one on disk (golang/go#20947; `pwd -P` resolves symlinks, not case). So
# UNSLOTH_STUDIO_HOME=<master>/Studio opens the very directory the installer
# created and arrives in $STUDIO_HOME spelled `Studio`.
#
# storage_roots._inherits_parent_portable_marker and setup.sh's
# _setup_portable_mode already fold that name on Darwin, so an exact `*/studio`
# match in _clear_stale_portable_marker left the parent marker in place while
# both readers went on honouring it: the normal reinstall reported success and
# the tree still resolved as portable, with UNSLOTH_PORTABLE=0 powerless to turn
# it off. The last section runs the REAL resolver over the tree the installer
# block just produced, so this fails if either half stops agreeing.
#
# The fold must stay Darwin-only: on Linux `studio` and `Studio` are two
# different directories and a marker above the second one is not ours.
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

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockD="$(blk '/^_clear_stale_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockD" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
# The fold itself, and the platform gate on it. Without both, the checks below
# would still pass on the two lowercase cases and prove nothing.
case "$blockD" in *'Darwin'*) : ;; *) echo "FAIL: blockD lost the Darwin gate"; exit 1 ;; esac
case "$blockD" in *"tr '[:upper:]' '[:lower:]'"*) : ;; *) echo "FAIL: blockD lost the case fold"; exit 1 ;; esac
# ${var,,} is a bashism and install.sh runs under macOS /bin/sh; readlink -f and
# realpath are not on the BSD side either.
case "$blockD" in *',,}'*) echo "FAIL: blockD uses a bash-only case fold"; exit 1 ;; *) : ;; esac
# The fold changes the leaf, so stripping the literal `/studio` would derive the
# wrong parent for `Studio` and the removal would miss (or hit the wrong dir).
case "$blockD" in *'${STUDIO_HOME%/studio}'*) echo "FAIL: blockD still strips a literal /studio"; exit 1 ;; *) : ;; esac
# Lifted out on its own below, so it may not call a helper defined elsewhere.
case "$blockD" in *'_portable_marker_prior'*) echo "FAIL: blockD calls a helper it does not define"; exit 1 ;; *) : ;; esac

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

# The fold only runs on macOS, so provoke it: a uname earlier on PATH than the
# real one. Only uname is shadowed; tr, grep and the rest still resolve normally.
mkdir -p "$T/fakebin"
printf '#!/bin/sh\nprintf "Darwin\\n"\n' > "$T/fakebin/uname"
chmod +x "$T/fakebin/uname"

new_root() { mktemp -d "$T/root.XXXXXX"; }

# A nested portable tree: <master>/<leaf> holding the venv, marker one level up.
build() { # master leaf
    mkdir -p "$1/$2/unsloth_studio/bin" "$1/bin" "$1/share"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
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
echo "[1] macOS: the same directory spelled Studio is still this install"
M1="$(new_root)"; build "$M1" Studio
convert "$M1" Studio darwin
check "darwin: Studio drops the parent marker" gone "$(marker_state "$M1")"

echo
echo "[2] Linux: Studio is a different directory, so the marker is not ours"
M2="$(new_root)"; build "$M2" Studio
convert "$M2" Studio linux
check "linux: Studio keeps the parent marker" present "$(marker_state "$M2")"

echo
echo "[3] the fold is not the whole rule"
# Lowercase must still work on Darwin, or the fold has replaced the match.
M3="$(new_root)"; build "$M3" studio
convert "$M3" studio darwin
check "darwin: lowercase studio still drops it" gone "$(marker_state "$M3")"
# ...and it must not turn every child of a marked root into a match.
M4="$(new_root)"; build "$M4" studio
mkdir -p "$M4/Other/unsloth_studio"
convert "$M4" Other darwin
check "darwin: a sibling keeps the portable root's marker" present "$(marker_state "$M4")"

echo
echo "[4] the parent it derives is the marked root, not a folded path"
# `${STUDIO_HOME%/studio}` cannot strip `Studio`, so a wrong derivation here
# either misses the marker (case 1 fails) or walks somewhere else entirely.
M5="$(new_root)"; build "$M5" Studio
mkdir -p "$M5/Studio/nested"
printf 'neighbour\n' > "$M5/Studio/.unsloth-portable-root"
convert "$M5" Studio darwin
check "darwin: the parent marker goes" gone "$(marker_state "$M5")"
check "darwin: the marker inside the studio dir goes too (flat slot)" gone "$(marker_state "$M5/Studio")"

echo
echo "[5] the flat layout is untouched by the fold"
M6="$(new_root)"
mkdir -p "$M6/unsloth_studio/bin"
printf '%s\n' "$M6" > "$M6/.unsloth-portable-root"
env -i HOME="$T/home" PATH="$T/fakebin:$PATH" USER="${USER:-tester}" \
    UNSLOTH_STUDIO_HOME="$M6" bash -c "$SNIP" _ > "$T/out" 2>"$T/err" || true
check "darwin: a flat root still loses its own marker" gone "$(marker_state "$M6")"

echo
echo "[6] the runtime half: storage_roots must agree about the tree"
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys, unittest.mock
sys.path.insert(0, os.environ["_BACKEND"])
# _inherits_parent_portable_marker folds on `sys.platform == "darwin"`, the same
# condition the installer block reads out of uname.
with unittest.mock.patch.object(sys, "platform", "darwin"):
    from utils.paths import storage_roots as sr
    home = sr.unsloth_home()
    print("__JSON__" + json.dumps({
        "portable": sr.portable_mode(),
        "unsloth_home": str(home) if home else None,
    }))
PYEOF
    probe() { # studio_home
        _pout=$(env -i HOME="$T/home" PATH="$PATH" _BACKEND="$BACKEND" \
            UNSLOTH_STUDIO_HOME="$1" python3 "$PROBE" 2>"$T/perr")
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            printf '  FAIL  storage_roots probe produced no output\n%s\n' "$(cat "$T/perr")"
            fails=$((fails + 1)); printf 'probe-failed'; return 0
        fi
        printf '%s' "$_pjson" | python3 -c \
            'import json,sys; print(str(json.load(sys.stdin)["portable"]).lower())'
    }
    M7="$(new_root)"; build "$M7" Studio
    check "before: the runtime reads the cased tree as portable" true "$(probe "$M7/Studio")"
    convert "$M7" Studio darwin
    check "after: the converted tree reads as non-portable" false "$(probe "$M7/Studio")"
    # The Linux half of the same agreement: there the resolver does not inherit
    # across a `Studio` child at all, so the marker the installer leaves behind
    # belongs to a `studio` install and costs this tree nothing.
    LPROBE="$T/probe_linux.py"
    sed 's/^with unittest.*$/if True:/' "$PROBE" > "$LPROBE"
    grep -q 'if True:' "$LPROBE" || { echo "FAIL: linux probe rewrite broke"; exit 1; }
    lprobe() { # studio_home
        env -i HOME="$T/home" PATH="$PATH" _BACKEND="$BACKEND" \
            UNSLOTH_STUDIO_HOME="$1" python3 "$LPROBE" 2>/dev/null \
            | sed -n 's/^__JSON__//p' \
            | python3 -c 'import json,sys; print(str(json.load(sys.stdin)["portable"]).lower())'
    }
    M8="$(new_root)"; build "$M8" Studio
    # On Linux both spellings can exist at once, and the marker belongs to this one.
    mkdir -p "$M8/studio/unsloth_studio/bin"
    convert "$M8" Studio linux
    check "linux: the marker stays, and the Studio child never reads it" false "$(lprobe "$M8/Studio")"
    check "linux: lowercase studio under the same root still does" true "$(lprobe "$M8/studio")"
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
