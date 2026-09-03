#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: a portable install that cannot publish <root>/.unsloth-portable-root must
# FAIL, not report success.
#
# The marker is the only portable signal that survives on disk, and every reader wants a regular
# FILE at exactly that path -- storage_roots.unsloth_home() tests `.is_file()`, and the CLI's
# venv inference does the same in _looks_like_installer_managed_studio_home. An earlier partial
# or manual setup that left a DIRECTORY there makes the `>` redirect fail; with `|| true` the
# installer carried on and printed its success summary, and the documented "activate the venv"
# path then resolved back to ~/.unsloth and wrote the caches, the projects root and studio.db
# outside the root the user selected. Silent, and only visible much later.
#
# Driven by extracting the real _export_portable_roots (plus the slot declarations it writes
# into) from install.sh and running it against fixtures, and the "does it read as portable"
# half is checked through the real storage_roots resolver rather than asserted.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL="$ROOT/install.sh"
BACKEND="$ROOT/studio/backend"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

blk() { awk "$1" "$INSTALL"; }
blockT="$(grep '^_trim_ws() ' "$INSTALL")"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockM="$(blk '/^_PORTABLE_MARKER_PATH_1=""$/ {grab=1} grab {print} /^_PORTABLE_SHIM_BACKUP=""$/ {exit}')"
blockE="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"

# Self-validate the extraction, or every assertion below is about "".
case "$blockT" in *_trim_ws*) : ;; *) echo "FAIL: blockT extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_MARKER_PRIOR_1'*) : ;; *) echo "FAIL: blockM extraction broke"; exit 1 ;; esac
case "$blockE" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockE extraction broke"; exit 1 ;; esac
case "$blockE" in *'_PORTABLE_MARKER_PRIOR_1'*) : ;; *) echo "FAIL: blockE stopped recording the publish"; exit 1 ;; esac
# The publish must not depend on a helper defined elsewhere: this block is lifted out and run
# on its own here and by tests/sh/test_install_portable_marker_rollback.sh, where an undefined
# name inside a condition goes silently inert instead of failing.
case "$blockE" in *'_portable_marker_publish'*) echo "FAIL: blockE calls a helper it does not define"; exit 1 ;; *) : ;; esac

SNIP='
set -e
'"$blockT"'
'"$blockB"'
'"$blockM"'
'"$blockE"'
substep() { :; }
_PORTABLE_MODE=true
_PORTABLE_FLAT=false
_UNSLOTH_ROOT="$_FIXTURE_ROOT"
_resolve_studio_destinations
_export_portable_roots
printf "PUBLISHED %s\n" "$_PORTABLE_MARKER_PATH_1"
printf "PRIOR %s\n" "$_PORTABLE_MARKER_PRIOR_1"
'

# Run the publish against $1; prints "<exit status>" and leaves stdout/stderr in $T/out,$T/err.
publish() { # root
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" _FIXTURE_ROOT="$1" \
        sh -c "$SNIP" _ > "$T/out" 2> "$T/err" && printf '0\n' || printf '%s\n' "$?"
}
said() { # needle
    if grep -qF -- "$1" "$T/out" "$T/err" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
}
mkdir -p "$T/home"
new_root() { mktemp -d "$T/root.XXXXXX"; }

echo "[1] the normal path still works"
R1="$(new_root)"
check "1 a fresh portable root publishes cleanly" 0 "$(publish "$R1")"
check "1 the marker is a regular file" yes "$([ -f "$R1/.unsloth-portable-root" ] && echo yes || echo no)"
check "1 naming the root" "$R1" "$(cat "$R1/.unsloth-portable-root")"
check "1 with no prior marker recorded" "PRIOR n" "$(grep '^PRIOR' "$T/out")"

echo "[2] a re-run over an existing marker still records the prior state for rollback"
R2="$(new_root)"
printf 'old\n' > "$R2/.unsloth-portable-root"
check "2 exits clean" 0 "$(publish "$R2")"
check "2 snapshots the old contents" "PRIOR yold" "$(grep '^PRIOR' "$T/out")"

echo "[3] a DIRECTORY at the marker path is fatal"
R3="$(new_root)"
mkdir -p "$R3/.unsloth-portable-root/leftover"
rc3="$(publish "$R3")"
check "3 the install fails instead of reporting success" 1 "$rc3"
check "3 the error names the marker path" yes "$(said "$R3/.unsloth-portable-root")"
check "3 and says a directory is in the way" yes "$(said "A directory is in its place")"
check "3 and says why it matters" yes "$(said "this install is not portable")"
# Nothing may be left pointing at a marker this run never wrote: the rollback slot has to be
# cleared, or a later restore would act on a path it does not own.
check "3 the rollback slot is released" "PUBLISHED " "$(grep '^PUBLISHED' "$T/out" || printf 'PUBLISHED \n')"
check "3 the directory is left exactly as it was" yes \
    "$([ -d "$R3/.unsloth-portable-root/leftover" ] && echo yes || echo no)"

echo "[4] any other unwritable marker path is fatal too, with the generic hint"
# A dangling symlink at the name: the root itself is writable, so the earlier -w check passes
# and this is the publish failing on its own. -f is false through a broken link, so the prior
# state still reads as absent and nothing is snapshotted for rollback.
R4="$(new_root)"
ln -s "$R4/nowhere/marker" "$R4/.unsloth-portable-root"
rc4="$(publish "$R4")"
check "4 the install fails" 1 "$rc4"
check "4 with the writable/space hint" yes "$(said "is writable and has free space")"
check "4 not the directory hint" no "$(said "A directory is in its place")"
check "4 the rollback slot is released" "PUBLISHED " "$(grep '^PUBLISHED' "$T/out" || printf 'PUBLISHED \n')"

echo "[5] why fatal: through the real resolver, a directory marker is NOT portable"
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
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
        printf '%s' "$_pjson" | python3 -c 'import json,sys; print(str(json.load(sys.stdin)["portable"]).lower())'
    }
    # A nested portable tree whose master root carries a real marker file: portable.
    P5="$(new_root)"
    mkdir -p "$P5/studio/unsloth_studio"
    printf '%s\n' "$P5" > "$P5/.unsloth-portable-root"
    check "5 a regular-file marker reads as portable" true "$(probe "$P5/studio")"
    # The same tree with a DIRECTORY at that name: every reader wants .is_file(), so the
    # install silently stops being portable. This is what the abort above prevents shipping.
    P6="$(new_root)"
    mkdir -p "$P6/studio/unsloth_studio" "$P6/.unsloth-portable-root"
    check "5 a directory at the same path does not" false "$(probe "$P6/studio")"
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
