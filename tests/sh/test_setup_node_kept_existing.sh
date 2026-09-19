#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# install_node_prebuilt.py exits 0 when a failed update keeps a Node that still runs, and
# logs "keeping existing isolated Node". setup.sh must report that rather than a fresh
# install, and relay the takeown/icacls lines a denied rename printed (#9928), which it
# otherwise shows only on a non-zero exit. Runs the real exit-handling block, extracted by
# content anchors, against a stand-in installer.
set -uo pipefail

HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected-substring (or !forbidden-substring) actual
    case "$2" in
        '!'*)
            case "$3" in
                *"${2#!}"*) printf '  FAIL  %s : did not expect [%s] in:\n%s\n' "$1" "${2#!}" "$3"; fails=$((fails+1)) ;;
                *) printf '  PASS  %s\n' "$1" ;;
            esac ;;
        *)
            case "$3" in
                *"$2"*) printf '  PASS  %s\n' "$1" ;;
                *) printf '  FAIL  %s : expected [%s] in:\n%s\n' "$1" "$2" "$3"; fails=$((fails+1)) ;;
            esac ;;
    esac
}

# From the installer call to the log cleanup after the "already matches" check.
BLOCK="$(awk '
    /^    _NODE_LOG="\$\(mktemp\)"$/ {grab = 1}
    grab {print}
    grab && /^    rm -f "\$_NODE_LOG"$/ {exit}
' "$SETUP")"
case "$BLOCK" in
    *'install_node_prebuilt.py'*'_NODE_STATUS'*'already matches'*'rm -f "$_NODE_LOG"') : ;;
    *) echo "FAIL: the Node exit-handling block could not be extracted from setup.sh"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
cat > "$T/fake_installer" <<'EOF'
#!/bin/bash
cat "$FAKE_OUTPUT"
exit "$FAKE_STATUS"
EOF
chmod +x "$T/fake_installer"

run_block() { # exit-status installer-output
    printf '%s\n' "$2" > "$T/output"
    env -i PATH="$PATH" TMPDIR="$T" FAKE_OUTPUT="$T/output" FAKE_STATUS="$1" \
        FAKE_INSTALLER="$T/fake_installer" BLOCK="$BLOCK" \
        bash -c '
            set -euo pipefail
            C_OK=ok C_WARN=warn C_ERR=err C_DIM= C_RST=
            step() { printf "STEP %s | %s | %s\n" "$1" "$2" "${3:-ok}"; }
            substep() { printf "SUBSTEP %s\n" "$1"; }
            verbose_substep() { printf "VERBOSE %s\n" "$1"; }
            setup_fail() { printf "SETUP_FAIL %s | %s\n" "$1" "$2"; exit "$1"; }
            _is_verbose() { return 1; }
            _NODE_PY="$FAKE_INSTALLER"
            SCRIPT_DIR=/nonexistent
            NODE_DIR=/nonexistent/node
            eval "$BLOCK"
            echo "CONTINUED"
        ' 2>&1
}

DENIED='[node-prebuilt] rename still blocked (5) after 8 attempts
[node-prebuilt]   takeown /F "C:\Users\u\.unsloth\node" /R /D Y
[node-prebuilt]   icacls "C:\Users\u\.unsloth\node" /reset /T /C
[node-prebuilt] existing Node could not be replaced ([WinError 5] Access is denied); keeping existing isolated Node'

echo ""
echo "=== exit 0 after a denied rename kept the running Node ==="
out="$(run_block 0 "$DENIED")"
check "reported as a kept install, as a warning" "STEP node | update not applied, existing isolated Node kept | warn" "$out"
check "not llama.cpp's keep text, which setup counts as exactly two arms" "!existing prebuilt kept" "$out"
check "the takeown/icacls lines reach the user" '   | [node-prebuilt]   takeown /F' "$out"
check "setup carries on" "CONTINUED" "$out"
check "setup does not fail" "!SETUP_FAIL" "$out"

echo ""
echo "=== exit 0 after a failed download kept the running Node ==="
out="$(run_block 0 '[node-prebuilt] Node download failed (nodejs.org unreachable); keeping existing isolated Node')"
check "reported as a kept install, as a warning" "STEP node | update not applied, existing isolated Node kept | warn" "$out"
check "no repair lines, so the installer output is not relayed" "!   | " "$out"
check "setup carries on" "CONTINUED" "$out"

echo ""
echo "=== exit 0 after a fresh install ==="
out="$(run_block 0 '[node-prebuilt] installed isolated Node v24.18.0 (npm 11.x) at /x/node')"
check "no kept-install warning" "!existing isolated Node kept" "$out"
check "setup carries on" "CONTINUED" "$out"

echo ""
echo "=== a non-zero exit still fails setup, whatever the output says ==="
out="$(run_block 1 "$DENIED")"
check "exit 1 fails setup" "SETUP_FAIL 1" "$out"
check "a failed install is not reported as kept" "!existing isolated Node kept" "$out"
out="$(run_block 3 '[node-prebuilt] another install holds the lock')"
check "exit 3 fails setup with 3" "SETUP_FAIL 3" "$out"

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All checks passed"
