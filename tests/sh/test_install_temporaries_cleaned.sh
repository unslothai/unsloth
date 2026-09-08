#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: every staging file this installer writes is removed when a signal ends the
# run.
#
# Three places stage a file and rename it into place: the master root record, the parent
# portable marker, and the generated shim. Each has an inline failure arm that removes its own
# temporary, and each arm is skipped when a signal diverts execution into the trap instead --
# which is the ordinary way an install ends, since Ctrl-C is how a user stops one.
#
# The shim's was the one not registered, so an interrupted portable install left
# .unsloth.shim.<pid> in ~/.local/bin: a directory the user has on PATH, where a successful
# retry never looks and nothing ever prunes it. Checked behaviourally rather than by grepping
# for the variable names, so a registration that is present but misspelled still fails.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockC="$(awk '
    /^_cleanup_install_temporaries\(\) \{$/ {g = 1}
    g {print}
    g && /^\}$/ {exit}
' "$INSTALL")"
case "$blockC" in
    *_cleanup_install_temporaries*) : ;;
    *) echo "FAIL: cleanup extraction broke"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
# Every slot the function reads, seeded with a real file, so a name it does not clear shows up
# as a survivor rather than as a missing variable under set -u.
printf '%s\n' 'set -u' "$blockC" \
    '_cleanup_install_temporaries' > "$SNIP"

slots="_EPR_TMP _PMP_TMP _SHIM_TMP"
# Discovered from the function itself, so a slot added later is covered without editing this
# list, and a slot silently dropped from the function is what the count check below catches.
found="$(printf '%s\n' "$blockC" | sed -n 's/.*\[ -n "\${\([A-Z_]*\):-}" \].*/\1/p' | sort -u)"
for s in $slots; do
    case " $(printf '%s' "$found" | tr '\n' ' ') " in
        *" $s "*) check "$s is registered for cleanup" yes yes ;;
        *) check "$s is registered for cleanup" yes no ;;
    esac
done

# Behavioural: seed each slot with a file that exists, run the cleanup, and every one is gone.
mkdir -p "$T/staging"
env_args=""
for s in $found; do
    : > "$T/staging/$s"
    env_args="$env_args $s=$T/staging/$s"
done
# shellcheck disable=SC2086
env -i PATH="$PATH" $env_args sh "$SNIP"
left="$(ls -1 "$T/staging" 2>/dev/null | wc -l | tr -d ' ')"
check "the cleanup removes every registered staging file" "0" "$left"

# A slot left empty must not turn into `rm -f ""` or a stray removal of the cwd.
: > "$T/staging/keep"
env -i PATH="$PATH" sh "$SNIP"
check "an unset slot removes nothing" present \
    "$([ -e "$T/staging/keep" ] && echo present || echo gone)"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
