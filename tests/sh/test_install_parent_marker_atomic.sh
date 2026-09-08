#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: a failed publication of the parent portable marker must leave no marker.
#
# `printf ... > "$marker"` truncates before printf runs, so a write that dies partway (ENOSPC,
# a disk quota, SIGXFSZ) left a ZERO-BYTE file at the exact path every reader tests for with
# -f, and the failure arm cleared the rollback slot so nothing removed it. Measured, on the
# code before the fix: both install.sh's own adoption and storage_roots.portable_mode() then
# reported portable=true for an install that was never made portable, and the install had
# already exited 1.
#
# Two ways in, and they need different answers:
#   a FIRST publication that fails must leave nothing behind, and
#   a RE-publication that fails must leave the previous root's bytes untouched -- a portable
#   install whose marker got truncated to zero would be read as normal by nothing and as
#   portable-with-no-root by everything, and converting it back moves gigabytes of cache.
# Staging into a sibling file and renaming gives both, because the rename is atomic and never
# happens at all when the write failed.
#
# ulimit -f 0 is the failure injection: the redirection still opens and truncates, and the
# write then fails, which is precisely the shape of the bug. The real ENOSPC needs a full
# filesystem, and a read-only directory is the wrong shape (the open fails, so no file is
# created and there was never anything to clean up).
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

# The publication block, lifted from the marker path assignment to its closing `fi`.
blockP="$(awk '
    /^    _PORTABLE_MARKER_PATH_1="\$UNSLOTH_ROOT\/\.unsloth-portable-root"$/ {g = 1}
    g {print}
    g && /^    _PMP_TMP=""$/ {exit}
' "$INSTALL")"
case "$blockP" in
    *'mv -f "$_pmp_tmp"'*) : ;;
    *) echo "FAIL: the parent marker is not published through a rename any more"; exit 1 ;;
esac
case "$blockP" in
    *'_PORTABLE_MARKER_PATH_1="$UNSLOTH_ROOT/.unsloth-portable-root"'*) : ;;
    *) echo "FAIL: publication extraction broke"; exit 1 ;;
esac
# The staging file has to be reachable by the install-wide cleanup, or a signal between the
# write and the rename leaves it in the user's root.
grep -q '\[ -n "${_PMP_TMP:-}" \] && rm -f "$_PMP_TMP"' "$INSTALL" \
    || { echo "FAIL: the staging file is not registered for cleanup"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' 'UNSLOTH_ROOT="$FIXTURE_ROOT"' "$blockP" \
    'echo PUBLISHED' > "$SNIP"

# SIGXFSZ is ignored so the oversized write returns EFBIG to printf instead of killing the
# shell outright. That is the point: it lets the script's OWN failure arm run, which is what
# is under test. A signal that kills the process mid-publication is a different case, covered
# by the install-wide cleanup registration checked above rather than here, because a lifted
# block has no traps armed.
publish() { # root [shell-prelude]
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$1" \
        sh -c "trap '' XFSZ 2>/dev/null; ${2:-:}; . '$SNIP'" > "$T/out" 2>"$T/err"
    printf '%s' "$?"
}
state() { if [ -e "$1" ]; then printf present; else printf gone; fi; }
mkdir -p "$T/home"

# 1. A first publication that fails partway leaves nothing at the marker path, so the next
# plain run sees a normal install rather than adopting an identity nobody established.
R1="$T/r1"; mkdir -p "$R1"
rc1="$(publish "$R1" 'ulimit -f 0')"
check "a failed first publication reports failure" "1" "$rc1"
check "and leaves no marker at all" gone "$(state "$R1/.unsloth-portable-root")"
check "and no staging file either"  "0" \
    "$(ls -a "$R1" | grep -c 'unsloth-portable-root\.')"

# 2. The same failure over an EXISTING portable install keeps the bytes that were there. The
# old in-place write truncated them, which is the worse half of this bug: a real portable
# install left describing no root.
R2="$T/r2"; mkdir -p "$R2"
printf '%s\n' "$R2" > "$R2/.unsloth-portable-root"
rc2="$(publish "$R2" 'ulimit -f 0')"
check "a failed re-publication reports failure"  "1" "$rc2"
check "and the previous root survives intact"    "$R2" \
    "$(cat "$R2/.unsloth-portable-root")"

# 3. The ordinary path still publishes, and publishes the root.
R3="$T/r3"; mkdir -p "$R3"
rc3="$(publish "$R3")"
check "a normal publication succeeds"        "0"   "$rc3"
check "and the marker names the root"        "$R3" "$(cat "$R3/.unsloth-portable-root")"
check "leaving no staging file behind"       "0"   \
    "$(ls -a "$R3" | grep -c 'unsloth-portable-root\.')"

# 4. A directory sitting at the marker path is still refused rather than moved into. `mv file
# dir` succeeds by putting the file INSIDE, which would report a published marker and leave
# the readers seeing a directory, which they all read as not-portable.
R4="$T/r4"; mkdir -p "$R4/.unsloth-portable-root"
rc4="$(publish "$R4")"
check "a directory in the marker's place fails" "1" "$rc4"
check "and nothing was moved into it"           "0" \
    "$(ls -a "$R4/.unsloth-portable-root" | grep -cv '^\.\{1,2\}$')"
# The message is read from this case rather than the truncated-write ones: ulimit -f 0 caps
# every write in that shell, stderr included, so there the diagnostic cannot reach the file.
check "and the failure names the marker"        "1" \
    "$(grep -c 'could not write the portable root marker' "$T/err")"
check "and says a directory is in the way"      "1" \
    "$(grep -c 'A directory is in its place' "$T/err")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
