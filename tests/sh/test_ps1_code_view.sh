#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
# shellcheck source=tests/sh/_ps1_source.sh
. "$SCRIPT_DIR/_ps1_source.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== line numbers survive the filter ==="
for f in "$SETUP_PS1" "$INSTALL_PS1"; do
    assert_eq "$(basename "$f") keeps its line count" \
        "$(wc -l < "$f")" "$(ps1_code "$f" | wc -l)"
done

echo "=== the shipped scripts ==="
# setup.ps1 exits in exactly one place; the emitted probe's `exit 1` is not its own.
assert_eq "setup.ps1 has one exit in its own code" \
    1 "$(ps1_code "$SETUP_PS1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"
assert_eq "the surviving exit is Exit-SetupFailure's" \
    1 "$(ps1_code "$SETUP_PS1" | grep -c '^[[:space:]]*exit \$Code$' || true)"
assert_eq "install.ps1's own '} finally {' blocks are counted, not the launcher's" \
    3 "$(ps1_code "$INSTALL_PS1" | grep -c '^    } finally {$' || true)"

echo "=== a redaction pattern is not a here-string opener ==="
# Both scripts have a `-replace ... '$1<redacted>@'` line whose raw text ends in @'.
for f in "$SETUP_PS1" "$INSTALL_PS1"; do
    _redact_line=$(grep -n "redacted" "$f" | head -1 | cut -d: -f1)
    assert_eq "$(basename "$f") keeps its redaction line as code" \
        1 "$(ps1_code "$f" | sed -n "${_redact_line}p" | grep -c 'redacted' || true)"
done

echo "=== an exit added to real code is seen ==="
cat > "$WORK/added-real.ps1" <<'PS1'
function Exit-SetupFailure {
    exit $Code
}
$probe = @'
if ($true) { exit 0 }
exit 1
'@
exit 2
PS1
assert_eq "a top-level exit outside the here-string is counted" \
    2 "$(ps1_code "$WORK/added-real.ps1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"

echo "=== an exit added inside a here-string is not ==="
cat > "$WORK/added-emitted.ps1" <<'PS1'
function Exit-SetupFailure {
    exit $Code
}
$probe = @'
exit 1
exit 3
'@
PS1
assert_eq "exits in an emitted script stay out of the count" \
    1 "$(ps1_code "$WORK/added-emitted.ps1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"

echo "=== the expandable form is handled too ==="
cat > "$WORK/expandable.ps1" <<'PS1'
$launcher = @"
exit 1
function Thing {
}
"@
exit 4
PS1
assert_eq "@\" ... \"@ bodies are blanked" \
    1 "$(ps1_code "$WORK/expandable.ps1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"

echo "=== the terminator must sit in column 0 ==="
# install.ps1 has an indented `"@echo off",` entry; a stripped comparison would close there.
cat > "$WORK/indented-terminator.ps1" <<'PS1'
$body = @'
line one
    "@echo off",
exit 1
'@
exit 5
PS1
assert_eq "an indented \"@ does not close a here-string" \
    1 "$(ps1_code "$WORK/indented-terminator.ps1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"

echo "=== a comment mentioning a delimiter is not one ==="
cat > "$WORK/comment-opener.ps1" <<'PS1'
# the terminator for a literal here-string is @'
exit 1
PS1
assert_eq "a trailing @' in a comment opens nothing" \
    1 "$(ps1_code "$WORK/comment-opener.ps1" | grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)"

summary
