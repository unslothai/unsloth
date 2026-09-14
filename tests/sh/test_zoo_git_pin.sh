#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _resolve_zoo_git_spec, which pins the --local
# unsloth-zoo overlay to a commit. git is stubbed: no test here reaches github.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_resolve_zoo_git_spec()/,/^}/p' "$INSTALL_SH" > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"

URL="unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
SHA="cb4a3100c1f34766e30f79229168fb2d91ea4123"

# $STUB_OUT is ls-remote's output, $STUB_RC its exit code, $STUB_LOG its argv + env.
_STUB_DIR=$(mktemp -d)
cat > "$_STUB_DIR/git" <<'STUB'
#!/bin/sh
if [ -n "${STUB_LOG:-}" ]; then
    {
        echo "argv: $*"
        echo "prompt=${GIT_TERMINAL_PROMPT-unset}"
        echo "askpass=[${GIT_ASKPASS-unset}] ssh_askpass=[${SSH_ASKPASS-unset}]"
    } >> "$STUB_LOG"
fi
printf '%s' "$STUB_OUT"
exit "${STUB_RC:-0}"
STUB
chmod +x "$_STUB_DIR/git"
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
assert_contains "credential helpers disabled" "$(cat "$STUB_LOG")" "-c credential.helper="
assert_contains "GIT_TERMINAL_PROMPT=0"       "$(cat "$STUB_LOG")" "prompt=0"
assert_contains "the probe is bounded"        "$(cat "$STUB_LOG")" "bounded by timeout 20"
# The only bound on a host with no `timeout`, which is stock macOS.
assert_contains "and bounded again by git itself" "$(cat "$STUB_LOG")" \
    "-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20"
assert_contains "askpass helpers cannot be reached" "$(cat "$STUB_LOG")" \
    "askpass=[] ssh_askpass=[]"
assert_contains "including one from a config file" "$(cat "$STUB_LOG")" "-c core.askPass="
# Against what the caller had, not "unset": a CI runner may export its own.
assert_eq "and the variable is left exactly as the caller had it" \
    "$_PROMPT_BEFORE" "${GIT_TERMINAL_PROMPT-unset}"

echo "=== a ref that merely ends in the name is not the ref ==="
# A pattern matches the ref tail at slash boundaries: `main` also matches
# refs/heads/archive/main, which sorts first, so line one pinned another history.
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
# `+` is legal and inert; rejecting it installed main when a branch was asked for.
UNSLOTH_ZOO_REF="feature+cuda"
export UNSLOTH_ZOO_REF
STUB_OUT="$SHA	refs/heads/feature+cuda"
: > "$STUB_LOG"
_resolve_zoo_git_spec
assert_eq "feature+cuda survives validation" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_contains "and is what git was asked for" "$(cat "$STUB_LOG")" "refs/heads/feature+cuda"

echo "=== a ref a requirement cannot carry is refused out loud ==="
_WARNING="$(UNSLOTH_ZOO_REF='release@2026' _resolve_zoo_git_spec 2>&1 >/dev/null)"
assert_contains "the refusal names the variable and the value" "$_WARNING" \
    "UNSLOTH_ZOO_REF='release@2026'"

echo "=== a fully qualified ref is asked for as given ==="
# refs/heads/release must not become refs/heads/refs/heads/release, which drops the pin.
UNSLOTH_ZOO_REF="refs/heads/release"
export UNSLOTH_ZOO_REF
STUB_OUT="$SHA	refs/heads/release"
: > "$STUB_LOG"
_resolve_zoo_git_spec
assert_eq "a qualified ref still pins" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_contains "asked for as written" "$(cat "$STUB_LOG")" \
    "refs/heads/release refs/heads/release^{}"

echo "=== the caller's own positional parameters survive ==="
# `set --` inside the resolver is function-local; install.sh still reads its own "$@".
_POSITIONAL_CHECK="$(STUDIO_LOCAL_INSTALL=true sh -c '. "$1"; set -- --no-torch --local; _resolve_zoo_git_spec; echo "$*"' _ "$_FUNC_FILE" 2>/dev/null | tail -n1)"
assert_eq "the caller still sees its own arguments" "--no-torch --local" "$_POSITIONAL_CHECK"

echo "=== an explicit ref is honored and pinned ==="
UNSLOTH_ZOO_REF="v2026.5.4"
export UNSLOTH_ZOO_REF
STUB_OUT="$SHA	refs/tags/v2026.5.4"
_resolve_zoo_git_spec
assert_eq "tag pinned to its commit" "$URL@$SHA" "$_ZOO_GIT_SPEC"
assert_eq "label names the tag"      "v2026.5.4 (cb4a3100c1f3)" "$_ZOO_REF_LABEL"

echo "=== unresolvable falls back to the ref as written ==="
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
rm -f "$_FUNC_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
