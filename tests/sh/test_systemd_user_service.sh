#!/bin/bash
# Unit tests for studio/systemd/install_user_service.sh (#9258).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/studio/systemd/install_user_service.sh"
TEMPLATE="$REPO_ROOT/studio/systemd/unsloth-studio.service.in"
PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

assert_contains() {
    if grep -qF -- "$3" "$2"; then ok "$1"; else bad "$1 (missing '$3' in $2)"; fi
}

echo "systemd install_user_service.sh"
[ -x "$INSTALL_SH" ] || chmod +x "$INSTALL_SH"
[ -f "$TEMPLATE" ] || { bad "template exists"; exit 1; }
ok "template and installer present"

_TMP="$(mktemp -d)"
_FAKE="$_TMP/unsloth"
printf '#!/bin/sh\nexit 0\n' > "$_FAKE"
chmod +x "$_FAKE"
mkdir -p "$_TMP/studio-home"

# Default bind must match `unsloth studio` (127.0.0.1), not expose every interface.
_UNIT_PATH=""
if _UNIT_PATH=$(XDG_CONFIG_HOME="$_TMP/config" env -u UNSLOTH_SYSTEMD_HOST -u UNSLOTH_SYSTEMD_PORT \
        bash "$INSTALL_SH" --unsloth-exe "$_FAKE" --studio-home "$_TMP/studio-home" --port 9090); then
    ok "writes unit without enable when systemd session absent or skipped"
else
    bad "writes unit without enable"
fi

if [ -f "$_UNIT_PATH" ]; then
    ok "returns unit path"
    assert_contains "managed marker" "$_UNIT_PATH" "unsloth-studio-managed-systemd"
    assert_contains "default host is loopback" "$_UNIT_PATH" "studio -H \"127.0.0.1\" -p 9090"
    assert_contains "exec stop" "$_UNIT_PATH" "studio stop"
    assert_contains "restart policy" "$_UNIT_PATH" "Restart=on-failure"
    assert_contains "studio home env" "$_UNIT_PATH" "UNSLOTH_STUDIO_HOME=$_TMP/studio-home"
    if grep -qF 'studio -H "0.0.0.0"' "$_UNIT_PATH"; then
        bad "default unit must not bind 0.0.0.0"
    else
        ok "default unit does not bind 0.0.0.0"
    fi
else
    bad "unit file created at returned path"
fi

# Explicit --host 0.0.0.0 remains the LAN opt-in (same as UNSLOTH_SYSTEMD_HOST).
_UNIT_PATH2=""
if _UNIT_PATH2=$(XDG_CONFIG_HOME="$_TMP/config2" bash "$INSTALL_SH" \
        --unsloth-exe "$_FAKE" --host 0.0.0.0 --port 9090); then
    ok "writes unit with explicit LAN host"
else
    bad "writes unit with explicit LAN host"
fi
if [ -f "$_UNIT_PATH2" ]; then
    assert_contains "explicit 0.0.0.0 opt-in" "$_UNIT_PATH2" "studio -H \"0.0.0.0\" -p 9090"
else
    bad "LAN unit file created"
fi

# UNSLOTH_SYSTEMD_HOST overrides the default when --host is omitted.
_UNIT_PATH3=""
if _UNIT_PATH3=$(XDG_CONFIG_HOME="$_TMP/config3" UNSLOTH_SYSTEMD_HOST=0.0.0.0 \
        bash "$INSTALL_SH" --unsloth-exe "$_FAKE" --port 8888); then
    ok "writes unit from UNSLOTH_SYSTEMD_HOST"
else
    bad "writes unit from UNSLOTH_SYSTEMD_HOST"
fi
if [ -f "$_UNIT_PATH3" ]; then
    assert_contains "env host opt-in" "$_UNIT_PATH3" "studio -H \"0.0.0.0\" -p 8888"
else
    bad "env-host unit file created"
fi

rm -rf "$_TMP"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
