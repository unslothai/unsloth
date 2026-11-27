#!/usr/bin/env bash
# setup.sh's system-Node reuse path is read-only: no global npm install, no NPM_CONFIG_PREFIX.
# Extraction is anchored on content and self-validates.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails+1)); }
pass() { printf '  PASS  %s\n' "$1"; }

system_arm="$(awk '
    /^if \[ "\$NODE_SOURCE" = system \]; then/ {grab=1; next}
    /^elif \[ "\$NODE_SOURCE" = bundled \]; then/ {grab=0}
    grab {print}
' "$SETUP")"
bundled_arm="$(awk '
    /^elif \[ "\$NODE_SOURCE" = bundled \]; then/ {grab=1; next}
    grab && /^else$/ {grab=0}
    grab {print}
' "$SETUP")"
bun_block="$(awk '
    /^if command -v bun &>\/dev\/null; then/ {grab=1}
    grab {print}
    grab && /^fi$/ {exit}
' "$SETUP")"

[ -n "$system_arm" ] || { echo "FAIL: system arm extraction broke"; exit 1; }
case "$bundled_arm" in *'NPM_CONFIG_PREFIX="$NODE_DIR"'*) : ;; *) echo "FAIL: bundled arm extraction broke"; exit 1 ;; esac
case "$bun_block"   in *'npm install -g bun'*)            : ;; *) echo "FAIL: bun block extraction broke";  exit 1 ;; esac

case "$system_arm" in *"npm install -g"*) fail "system arm runs no 'npm install -g'" ;; *) pass "system arm runs no 'npm install -g'" ;; esac
case "$system_arm" in *NPM_CONFIG_PREFIX*|*npm_config_prefix*) fail "system arm sets no NPM_CONFIG_PREFIX" ;; *) pass "system arm sets no NPM_CONFIG_PREFIX" ;; esac
case "$system_arm" in *"export PATH="*) fail "system arm does not rewrite PATH" ;; *) pass "system arm does not rewrite PATH" ;; esac
# Positive control so the negative checks above are not vacuous.
case "$bundled_arm" in *'NPM_CONFIG_PREFIX="$NODE_DIR"'*) pass "bundled arm pins NPM_CONFIG_PREFIX to the isolated dir" ;; *) fail "bundled arm pins NPM_CONFIG_PREFIX to the isolated dir" ;; esac
guard_at=$(printf '%s\n' "$bun_block" | grep -n 'elif \[ "\$NODE_SOURCE" = bundled \]; then' | head -1 | cut -d: -f1)
bun_at=$(printf '%s\n' "$bun_block" | grep -n 'npm install -g bun' | head -1 | cut -d: -f1)
if [ -n "$guard_at" ] && [ -n "$bun_at" ] && [ "$guard_at" -lt "$bun_at" ]; then
    pass "global bun install gated behind NODE_SOURCE=bundled"
else
    fail "global bun install gated behind NODE_SOURCE=bundled"
fi

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
