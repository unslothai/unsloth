#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: publishing <studio>/.unsloth-master-root must not destroy a record that is
# already there when the write fails.
#
# `printf ... > "$_epr_record"` truncates on open, so a write that then fails -- a full volume,
# which is exactly what a portable root on a small or removable disk runs into -- left a
# PREVIOUS valid record at zero bytes. The failure arm clears _PORTABLE_MARKER_PATH_3, so the
# rollback did not put it back either, and storage_roots reads an empty record as no record and
# resolves $HOME/.unsloth instead of the root the user named.
#
# The failure is injected with `ulimit -f 0`, which lets the open succeed and fails the write,
# the same order ENOSPC does. The real block is lifted out of install.sh and run on its own.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

# Just the record publish: from its path assignment to the end of the failure arm's own
# reporting, stopping before the slot-1 unwind (which needs the marker helpers).
block="$(awk '
    /^    _epr_record="\$STUDIO_HOME\/\.unsloth-master-root"$/ {grab = 1}
    grab {print}
    grab && /back to \$HOME\/\.unsloth and write outside the root you selected/ {
        print "            return 1"; print "        fi"; print "    fi"; exit
    }
' "$INSTALL")"

# Self-validate, or every assertion below is about an empty string.
case "$block" in *'_epr_record='*) : ;; *) echo "FAIL: block extraction broke"; exit 1 ;; esac
case "$block" in *'.unsloth-master-root'*) : ;; *) echo "FAIL: block lost the record path"; exit 1 ;; esac
case "$block" in
    *'_PORTABLE_MARKER_PRIOR_3'*) : ;;
    *) echo "FAIL: block lost the rollback snapshot"; exit 1 ;;
esac
# The point of the fix. A bare `> "$_epr_record"` anywhere in the publish reintroduces it.
# Comments stripped first: the code comment explaining the fix names the very idiom it bans.
block_code="$(printf '%s\n' "$block" | sed -e 's/^[[:space:]]*#.*$//')"
case "$block_code" in
    *'> "$_epr_record"'*)
        echo "FAIL: the record is written with a truncating redirect again"; exit 1 ;;
    *) : ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

SNIP='
_publish() {
'"$block"'
    return 0
}
_PORTABLE_MARKER_PATH_1=""
_PORTABLE_MARKER_PRIOR_1=""
_PORTABLE_MARKER_PATH_3=""
_PORTABLE_MARKER_PRIOR_3=""
ulimit -f 0
_publish
printf "rc=%s\n" "$?"
'

run() { # studio_home root
    env -i HOME="$T/home" PATH="$PATH" STUDIO_HOME="$1" UNSLOTH_ROOT="$2" \
        sh -c "$SNIP" _ 2>/dev/null
}

# A working nested portable install: the record names the master root one level up.
S="$T/root/studio"
mkdir -p "$S"
printf '%s\n' "$T/root" > "$S/.unsloth-master-root"
before="$(cat "$S/.unsloth-master-root")"
run "$S" "$T/newroot" >/dev/null 2>&1

check "the previous record survives a failed write" "$before" \
    "$(cat "$S/.unsloth-master-root" 2>/dev/null || printf '<missing>')"
check "and it is not left empty"                    "no" \
    "$([ -s "$S/.unsloth-master-root" ] && echo no || echo yes)"
# No temp assertion here: ulimit -f 0 delivers SIGXFSZ, which kills the shell before any
# failure arm or trap can run. That is an artifact of the injection, not of a full disk --
# ENOSPC returns an error to printf instead. The staging file is registered with
# _cleanup_install_temporaries for the signal case; the arm itself is exercised below.
grep -q '_EPR_TMP' "$INSTALL" \
    || { echo "FAIL: the staging file is not registered for cleanup"; exit 1; }

# A write failure that reports rather than signals: the directory is read-only, so the staging
# file cannot be created at all. Here the failure arm really does run.
S3="$T/ro/studio"
mkdir -p "$S3"
printf '%s\n' "$T/ro" > "$S3/.unsloth-master-root"
chmod 555 "$S3"
out_all="$(env -i HOME="$T/home" PATH="$PATH" STUDIO_HOME="$S3" UNSLOTH_ROOT="$T/ro-new" \
    sh -c "$SNIP" _ 2>&1 || true)"
chmod 755 "$S3"
check "an unwritable directory keeps the old record" "$T/ro" \
    "$(cat "$S3/.unsloth-master-root" 2>/dev/null || printf '<missing>')"
check "the failure is reported"                      "yes" \
    "$(printf '%s' "$out_all" | grep -q 'could not write the master root record' && echo yes || echo no)"
check "and its staging file is cleaned up"           "0" \
    "$(find "$S3" -maxdepth 1 -name '.unsloth-master-root.*' | wc -l | tr -d ' ')"

# Same publish with room to write: it still has to actually record the new root.
S2="$T/ok/studio"
mkdir -p "$S2"
printf '%s\n' "$T/ok-old" > "$S2/.unsloth-master-root"
env -i HOME="$T/home" PATH="$PATH" STUDIO_HOME="$S2" UNSLOTH_ROOT="$T/ok" sh -c '
_publish() {
'"$block"'
    return 0
}
_PORTABLE_MARKER_PATH_1=""
_PORTABLE_MARKER_PRIOR_1=""
_PORTABLE_MARKER_PATH_3=""
_PORTABLE_MARKER_PRIOR_3=""
_publish
' _ >/dev/null 2>&1
check "a successful publish still records the root" "$T/ok" \
    "$(cat "$S2/.unsloth-master-root" 2>/dev/null || printf '<missing>')"
check "and leaves no temporary behind"              "0" \
    "$(find "$S2" -maxdepth 1 -name '.unsloth-master-root.*' | wc -l | tr -d ' ')"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
