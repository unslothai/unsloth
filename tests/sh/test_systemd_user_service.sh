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

_UNIT_PATH=""
if _UNIT_PATH=$(XDG_CONFIG_HOME="$_TMP/config" bash "$INSTALL_SH" \
        --unsloth-exe "$_FAKE" --studio-home "$_TMP/studio-home" --host 0.0.0.0 --port 9090); then
    ok "writes unit without enable when systemd session absent or skipped"
else
    bad "writes unit without enable"
fi

if [ -f "$_UNIT_PATH" ]; then
    ok "returns unit path"
    assert_contains "managed marker" "$_UNIT_PATH" "unsloth-studio-managed-systemd"
    assert_contains "exec start host/port" "$_UNIT_PATH" "studio -H \"0.0.0.0\" -p 9090"
    assert_contains "exec stop" "$_UNIT_PATH" "studio stop"
    assert_contains "restart policy" "$_UNIT_PATH" "Restart=on-failure"
    assert_contains "studio home env" "$_UNIT_PATH" "UNSLOTH_STUDIO_HOME=$_TMP/studio-home"
else
    bad "unit file created at returned path"
fi

rm -rf "$_TMP"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
