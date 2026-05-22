#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

echo "=== test_tauri_install_exit_order ==="

tauri_marker_line=$(grep -n "Tauri mode: done, skip shortcuts and auto-launch" "$INSTALL_SH" | head -n1 | cut -d: -f1)
tauri_exit_line=$(awk -v start="$tauri_marker_line" 'NR > start && /exit 0/ { print NR; exit }' "$INSTALL_SH")
tauri_done_line=$(awk -v start="$tauri_marker_line" 'NR > start && /tauri_log "DONE" ""/ { print NR; exit }' "$INSTALL_SH")
first_setup_check_line=$(grep -n 'if \[ "$_SETUP_EXIT" -ne 0 \]; then' "$INSTALL_SH" | head -n1 | cut -d: -f1)
setup_exit_line=$(awk -v start="$first_setup_check_line" 'NR > start && /exit "\$_SETUP_EXIT"/ { print NR; exit }' "$INSTALL_SH")
shortcut_line=$(grep -n 'create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"' "$INSTALL_SH" | head -n1 | cut -d: -f1)
shortcut_guard_line=$(
    awk -v stop="$first_setup_check_line" '
        NR >= stop { exit }
        /\[ "\$TAURI_MODE" != true \]/ {
            print NR
            exit
        }
    ' "$INSTALL_SH"
)
shortcut_guard_end_line=$(
    awk -v start="$shortcut_guard_line" -v shortcut="$shortcut_line" '
        NR <= start { next }
        NR <= shortcut { next }
        /^[[:space:]]*fi[[:space:]]*$/ {
            print NR
            exit
        }
    ' "$INSTALL_SH"
)
early_tauri_exit_line=$(
    awk -v setup="$first_setup_check_line" '
        NR >= setup { exit }
        /\[ "\$TAURI_MODE" = true \]/ {
            in_tauri = 1
            depth = 1
            next
        }
        in_tauri && /^[[:space:]]*if[[:space:]].*;[[:space:]]*then[[:space:]]*$/ {
            depth++
        }
        in_tauri && /exit[[:space:]]+0/ {
            print NR
            exit
        }
        in_tauri && /^[[:space:]]*fi[[:space:]]*$/ {
            depth--
            if (depth == 0) {
                in_tauri = 0
            }
        }
    ' "$INSTALL_SH"
)

if [ -z "$tauri_marker_line" ] || [ -z "$tauri_exit_line" ] || [ -z "$tauri_done_line" ] || [ -z "$first_setup_check_line" ] || [ -z "$setup_exit_line" ] || [ -z "$shortcut_line" ] || [ -z "$shortcut_guard_line" ] || [ -z "$shortcut_guard_end_line" ]; then
    echo "  FAIL: required install.sh markers not found"
    exit 1
fi

if [ -n "$early_tauri_exit_line" ]; then
    echo "  FAIL: Tauri success exit before setup failure check at line $early_tauri_exit_line"
    exit 1
fi

if [ "$shortcut_guard_line" -ge "$shortcut_line" ] || [ "$shortcut_line" -ge "$shortcut_guard_end_line" ]; then
    echo "  FAIL: shortcuts are not guarded by non-Tauri check"
    exit 1
fi

if [ "$shortcut_line" -ge "$first_setup_check_line" ]; then
    echo "  FAIL: shortcut line $shortcut_line is after setup failure check line $first_setup_check_line"
    exit 1
fi

if [ "$first_setup_check_line" -ge "$setup_exit_line" ]; then
    echo "  FAIL: setup failure check line $first_setup_check_line is after setup exit line $setup_exit_line"
    exit 1
fi

if [ "$setup_exit_line" -ge "$tauri_done_line" ]; then
    echo "  FAIL: setup failure exit line $setup_exit_line is after Tauri DONE line $tauri_done_line"
    exit 1
fi

if [ "$setup_exit_line" -ge "$tauri_exit_line" ]; then
    echo "  FAIL: setup failure check line $first_setup_check_line is after Tauri exit line $tauri_exit_line"
    exit 1
fi

echo "  PASS: non-Tauri shortcuts run before setup failure exit"
echo "  PASS: setup failure exits before Tauri success"
echo "  PASS: Tauri skips shortcuts"
