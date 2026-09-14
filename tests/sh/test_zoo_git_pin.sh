#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _resolve_zoo_git_spec, which turns the --local
# unsloth-zoo overlay from "whatever the default branch holds at fetch time" into a
# requirement pinned to the commit that branch points at. git is stubbed, so no test
# here reaches github.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_resolve_zoo_git_spec()/,/^}/p' "$INSTALL_SH" > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

URL="unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
SHA="cb4a3100c1f34766e30f79229168fb2d91ea4123"

# A git stub on PATH, so the resolution is exercised without a network call.
# $STUB_OUT is what `git ls-remote` prints; $STUB_RC is its exit code. Its argv and
# the environment it was handed land in $STUB_LOG, which the guards are asserted on.
_STUB_DIR=$(mktemp -d)
cat > "$_STUB_DIR/git" <<'STUB'
#!/bin/sh
if [ -n "${STUB_LOG:-}" ]; then
    { echo "argv: $*"; echo "prompt=${GIT_TERMINAL_PROMPT-unset}"; } >> "$STUB_LOG"
fi
printf '%s' "$STUB_OUT"
exit "${STUB_RC:-0}"
STUB
chmod +x "$_STUB_DIR/git"
# A timeout stub that records it was used, then runs the real command: the probe has
# to be bounded, or a network that accepts and never answers hangs the installer.
cat > "$_STUB_DIR/timeout" <<'STUB'
#!/bin/sh
[ -n "${STUB_LOG:-}" ] && echo "bounded by timeout $1" >> "$STUB_LOG"
shift
exec "$@"
STUB
chmod +x "$_STUB_DIR/timeout"
STUB_LOG="$_STUB_DIR/calls.log"
export STUB_LOG
_PROMPT_BEFORE="${GIT_TERMINAL_PROMPT-unset}"
PATH="$_STUB_DIR:$PATH"
export PATH
STUDIO_LOCAL_INSTALL=true
STUB_RC=0
export STUB_OUT STUB_RC

echo "=== branch resolves to a commit ==="
STUB_OUT="$SHA	refs/heads/main
$SHA	refs/remotes/origin/main"
unset UNSLOTH_ZOO_REF
: > "$STUB_LOG"
_resolve_zoo_git_spec
assert_eq "main pinned to its commit" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_eq "label names the commit"    "main (cb4a3100c1f3)" "$_ZOO_REF_LABEL"

echo "=== the probe cannot stop and ask a human, and cannot wait forever ==="
# Nothing on this path needed credentials before the pin existed. A proxy answering
# 401 must not send git to a credential helper and leave the installer behind a
# prompt, and a network that accepts but never answers must not hang it either.
assert_contains "credential helpers disabled" "$(cat "$STUB_LOG")" "-c credential.helper= ls-remote"
assert_contains "GIT_TERMINAL_PROMPT=0"       "$(cat "$STUB_LOG")" "prompt=0"
assert_contains "the probe is bounded"        "$(cat "$STUB_LOG")" "bounded by timeout 20"
# Compared against what the caller had, not against "unset": a CI runner is allowed
# to export its own GIT_TERMINAL_PROMPT, and the point is that the probe changed nothing.
assert_eq "and the variable is left exactly as the caller had it" \
    "$_PROMPT_BEFORE" "${GIT_TERMINAL_PROMPT-unset}"

echo "=== a ref that merely ends in the name is not the ref ==="
# An ls-remote pattern matches the tail of a ref at slash boundaries, so `main` also
# matches refs/heads/archive/main, and that sorts first. Taking the first line pinned
# an unrelated history.
STUB_OUT="596da1d58adf69924b2977dc04f1d49a89cbfa60	refs/heads/archive/main
$SHA	refs/heads/main"
unset UNSLOTH_ZOO_REF
: > "$STUB_LOG"
_resolve_zoo_git_spec
assert_eq "archive/main is not mistaken for main" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_contains "the full ref names are what was asked for" "$(cat "$STUB_LOG")" \
    "refs/heads/main refs/tags/main refs/tags/main^{}"

echo "=== an annotated tag pins the commit, not the tag object ==="
UNSLOTH_ZOO_REF="v9"
export UNSLOTH_ZOO_REF
STUB_OUT="54700b3135d07a9bf78128b6c677fb909289e06d	refs/tags/v9
$SHA	refs/tags/v9^{}"
_resolve_zoo_git_spec
assert_eq "peeled to its commit" "$URL@$SHA" "$_ZOO_GIT_SPEC"

echo "=== a legal branch name is not quietly rewritten to main ==="
# `+` is legal in a branch name and inert inside a requirement; rejecting it installed
# main when a specific branch was asked for.
UNSLOTH_ZOO_REF="feature+cuda"
export UNSLOTH_ZOO_REF
STUB_OUT="$SHA	refs/heads/feature+cuda"
: > "$STUB_LOG"
_resolve_zoo_git_spec
assert_eq "feature+cuda survives validation" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_contains "and is what git was asked for" "$(cat "$STUB_LOG")" "refs/heads/feature+cuda"

echo "=== a ref a requirement cannot carry is refused out loud ==="
# uv reads `repo@release@2026` as revision "2026", so this ref cannot be expressed.
# Falling back to main is right; doing it silently is not.
_WARNING="$(UNSLOTH_ZOO_REF='release@2026' _resolve_zoo_git_spec 2>&1 >/dev/null)"
assert_contains "the refusal names the variable and the value" "$_WARNING" \
    "UNSLOTH_ZOO_REF='release@2026'"

echo "=== an explicit ref is honored and pinned ==="
UNSLOTH_ZOO_REF="v2026.5.4"
export UNSLOTH_ZOO_REF
STUB_OUT="$SHA	refs/tags/v2026.5.4"
_resolve_zoo_git_spec
assert_eq "tag pinned to its commit" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_eq "label names the tag"      "v2026.5.4 (cb4a3100c1f3)" "$_ZOO_REF_LABEL"

echo "=== unresolvable falls back to the ref as written ==="
# None of these may fail the install: an offline or git-less box installed fine
# before this pinning existed and still has to.
unset UNSLOTH_ZOO_REF
STUB_OUT=""
_resolve_zoo_git_spec
assert_eq "no matching ref -> main" "$URL@main" "$_ZOO_GIT_SPEC"
STUB_RC=128
STUB_OUT=""
_resolve_zoo_git_spec
assert_eq "remote unreachable -> main" "$URL@main" "$_ZOO_GIT_SPEC"
STUB_RC=0
STUB_OUT="not-a-sha	refs/heads/main"
_resolve_zoo_git_spec
assert_eq "unexpected output -> main" "$URL@main" "$_ZOO_GIT_SPEC"
assert_eq "label is the plain ref"    "main" "$_ZOO_REF_LABEL"

echo "=== non-local installs are untouched ==="
# The overlay only runs under --local; a normal install must not pay a network
# round trip for a spec it never uses.
STUDIO_LOCAL_INSTALL=false
STUB_OUT="$SHA	refs/heads/main"
_resolve_zoo_git_spec
assert_eq "no pin without --local" "$URL@main" "$_ZOO_GIT_SPEC"
STUDIO_LOCAL_INSTALL=true

echo "=== a malformed UNSLOTH_ZOO_REF never reaches the requirement ==="
STUB_OUT="$SHA	refs/heads/main"
for _bad in "main --index-url https://example.invalid/simple" \
            "main;curl evil" \
            "main#egg=unsloth-zoo" \
            "../../attacker/repo" \
            "-oProxyCommand=evil"; do
    UNSLOTH_ZOO_REF="$_bad"
    export UNSLOTH_ZOO_REF
    _resolve_zoo_git_spec
    assert_eq "rejected: $_bad" "$URL@$SHA" "$_ZOO_GIT_SPEC"
done
unset UNSLOTH_ZOO_REF

rm -rf "$_STUB_DIR"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
