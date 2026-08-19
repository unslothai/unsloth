#!/usr/bin/env bash
# Regression test: setup.sh's reuse (NODE_SOURCE=system) path is strictly
# read-only. It runs no global npm install and sets no NPM_CONFIG_PREFIX, so
# reusing a good system Node never mutates the user's Node/npm/NVM. Only the
# isolated (bundled) path redirects npm into its own prefix and installs
# anything global (and even then -g lands in the isolated prefix). Extraction is
# anchored on setup.sh content, not line numbers, and self-validates so a
# refactor fails loudly here.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails+1)); }
pass() { printf '  PASS  %s\n' "$1"; }

# Arm 1: the NODE_SOURCE=system branch body (reuse a good system Node).
system_arm="$(awk '
    /^if \[ "\$NODE_SOURCE" = system \]; then/ {grab=1; next}
    /^elif \[ "\$NODE_SOURCE" = bundled \]; then/ {grab=0}
    grab {print}
' "$SETUP")"
# Arm 2: the NODE_SOURCE=bundled branch body (provision the isolated Node).
bundled_arm="$(awk '
    /^elif \[ "\$NODE_SOURCE" = bundled \]; then/ {grab=1; next}
    grab && /^else$/ {grab=0}
    grab {print}
' "$SETUP")"
# The optional-bun block (the only global install, gated on the bundled path).
bun_block="$(awk '
    /^if command -v bun &>\/dev\/null; then/ {grab=1}
    grab {print}
    grab && /^fi$/ {exit}
' "$SETUP")"

# Self-validate extraction so a setup.sh refactor cannot silently void the test.
[ -n "$system_arm" ] || { echo "FAIL: system arm extraction broke"; exit 1; }
case "$bundled_arm" in *'NPM_CONFIG_PREFIX="$NODE_DIR"'*) : ;; *) echo "FAIL: bundled arm extraction broke"; exit 1 ;; esac
case "$bun_block"   in *'npm install -g bun'*)            : ;; *) echo "FAIL: bun block extraction broke";  exit 1 ;; esac

# 1. system (reuse) arm performs no global npm install.
case "$system_arm" in *"npm install -g"*) fail "system arm runs no 'npm install -g'" ;; *) pass "system arm runs no 'npm install -g'" ;; esac
# 2. system (reuse) arm sets no npm prefix redirect (either casing of the var).
case "$system_arm" in *NPM_CONFIG_PREFIX*|*npm_config_prefix*) fail "system arm sets no NPM_CONFIG_PREFIX" ;; *) pass "system arm sets no NPM_CONFIG_PREFIX" ;; esac
# 3. system (reuse) arm does not rewrite PATH toward a managed Node dir.
case "$system_arm" in *"export PATH="*) fail "system arm does not rewrite PATH" ;; *) pass "system arm does not rewrite PATH" ;; esac
# 4. positive control: the bundled arm DOES pin the prefix (so 1-3 aren't vacuous).
case "$bundled_arm" in *'NPM_CONFIG_PREFIX="$NODE_DIR"'*) pass "bundled arm pins NPM_CONFIG_PREFIX to the isolated dir" ;; *) fail "bundled arm pins NPM_CONFIG_PREFIX to the isolated dir" ;; esac
# 5. the only global install (bun) is gated behind NODE_SOURCE=bundled.
guard_at=$(printf '%s\n' "$bun_block" | grep -n 'elif \[ "\$NODE_SOURCE" = bundled \]; then' | head -1 | cut -d: -f1)
bun_at=$(printf '%s\n' "$bun_block" | grep -n 'npm install -g bun' | head -1 | cut -d: -f1)
if [ -n "$guard_at" ] && [ -n "$bun_at" ] && [ "$guard_at" -lt "$bun_at" ]; then
    pass "global bun install gated behind NODE_SOURCE=bundled"
else
    fail "global bun install gated behind NODE_SOURCE=bundled"
fi

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
