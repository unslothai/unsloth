#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# install.sh writes the WSL shortcut script under the Windows %TEMP% because a WSL UNC path is
# a remote script that RemoteSigned refuses. Runs the real block with cmd.exe/wslpath stubbed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

BLOCK=$(awk '
    /^        _css_win_temp=""$/ { grab = 1 }
    grab { print }
    grab && /_css_ps1_tmp=\$\(mktemp "\$_css_win_temp\/unsloth-shortcut-XXXXXX\.ps1"/ { print "        fi"; exit }
' "$INSTALL_SH")

echo "=== test_wsl_shortcut_script_location ==="

# Extraction failure is a FAIL, not an early exit, so the static half still runs.
HAVE_BLOCK=1
if ! printf '%s' "$BLOCK" | grep -q '_css_ps1_tmp=\$(mktemp'; then
    HAVE_BLOCK=0
    bad "could not extract the Windows-temp resolution block from install.sh.
        Expected a block starting with '        _css_win_temp=\"\"' and ending at the mktemp of
        \"\$_css_win_temp/unsloth-shortcut-XXXXXX.ps1\". Either that block was renamed or removed, or
        install.sh is back to writing the generated shortcut script into WSL's /tmp, which wslpath
        turns into a \\\\wsl.localhost UNC path that RemoteSigned refuses to load."
fi

STUBS=$(mktemp -d)
WINTEMP=$(mktemp -d)
cleanup() { rm -rf "$STUBS" "$WINTEMP"; }
trap cleanup EXIT

# Input-sensitive: only the exact string maps, so a misparsed path is visible.
# $1 = the only string that maps, $2 = where it maps to.
make_wslpath_stub() {
    cat > "$STUBS/wslpath" <<STUB
#!/bin/sh
[ "\$1" = "-u" ] || exit 1
if [ "\$2" = '$1' ]; then printf '%s' '$2'; else printf '%s' '$2/not-what-cmd-printed'; fi
STUB
    chmod +x "$STUBS/wslpath"
}

# $1 is what `echo %TEMP%` prints; $2, if given, is an AutoRun banner printed unless /d is passed.
make_cmd_stub() {
    cat > "$STUBS/cmd.exe" <<STUB
#!/bin/sh
# Real cmd.exe emits CRLF; the block strips it, so emit one here or the test would not cover that.
# It also runs the AutoRun command from HKCU\\Software\\Microsoft\\Command Processor first, on this
# same stdout, unless /d is passed. Clink sets one, so Cmder does, and so do corporate images.
_autorun=1
for _a in "\$@"; do [ "\$_a" = "/d" ] && _autorun=0; done
[ "\$_autorun" = 1 ] && [ -n '${2:-}' ] && printf '%s\r\n' '${2:-}'
printf '%s\r\n' '$1'
STUB
    chmod +x "$STUBS/cmd.exe"
    # Win32 strips trailing blanks from a path, so the block has to.
    make_wslpath_stub "$(printf '%s' "$1" | sed 's/[[:space:]]*$//')" "$WINTEMP"
}

run_block() {
    _css_win_temp=""
    _css_ps1_tmp=""
    ( PATH="$STUBS:$PATH"; eval "$BLOCK"; printf '%s\n%s\n' "$_css_win_temp" "$_css_ps1_tmp" )
}

if [ "$HAVE_BLOCK" -eq 1 ]; then

make_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
OUT=$(run_block)
RESOLVED=$(printf '%s' "$OUT" | sed -n 1p)
SCRIPT=$(printf '%s' "$OUT" | sed -n 2p)
assert_eq "resolved %TEMP% maps through wslpath" "$WINTEMP" "$RESOLVED"
[ -n "$SCRIPT" ] && ok "a script path was allocated" || bad "no script path was allocated"
case "$SCRIPT" in
    "$WINTEMP"/unsloth-shortcut-*.ps1) ok "script lives under the Windows temp" ;;
    *) bad "script is not under the Windows temp: $SCRIPT" ;;
esac
# Structural check, not a /tmp prefix check (which broke when TMPDIR=/tmp): the only allocation
# must be in the resolved Windows temp.
ALLOC=$(printf '%s' "$BLOCK" | grep -c 'mktemp' || true)
if [ "$ALLOC" -eq 1 ]; then
    ok "exactly one mktemp in the block, so there is no second path to fall back to"
else
    bad "the block contains $ALLOC mktemp calls; a second one is a fallback that can land in WSL's /tmp, which wslpath turns into a \\\\wsl.localhost UNC path that RemoteSigned refuses"
fi
assert_contains "the script is allocated inside the resolved Windows temp" \
    "$(printf '%s' "$BLOCK" | grep 'mktemp')" '$_css_win_temp/'
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# No cmd.exe: allocate nothing, and never fall back to a WSL path.
rm -f "$STUBS/cmd.exe"
OUT=$(run_block)
assert_eq "no interop: %TEMP% unresolved" "" "$(printf '%s' "$OUT" | sed -n 1p)"
assert_eq "no interop: no script allocated, so no UNC fallback" "" "$(printf '%s' "$OUT" | sed -n 2p)"

# Unset %TEMP% echoes back unexpanded.
make_cmd_stub '%TEMP%'
OUT=$(run_block)
assert_eq "unexpanded %TEMP% is rejected" "" "$(printf '%s' "$OUT" | sed -n 1p)"
assert_eq "unexpanded %TEMP% allocates nothing" "" "$(printf '%s' "$OUT" | sed -n 2p)"

make_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
make_wslpath_stub 'C:\Users\ci\AppData\Local\Temp' "$WINTEMP/definitely-absent"
OUT=$(run_block)
assert_eq "a %TEMP% that is not a directory is rejected" "" "$(printf '%s' "$OUT" | sed -n 1p)"

# cmd AutoRun (Clink, corporate images) prints before output unless /d is passed.
make_cmd_stub 'C:\Users\ci\AppData\Local\Temp' 'clink v1.6.20 is available.'
OUT=$(run_block)
assert_eq "an AutoRun banner does not corrupt %TEMP%" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
[ -n "$(printf '%s' "$OUT" | sed -n 2p)" ] \
    && ok "an AutoRun banner still allocates the script" \
    || bad "an AutoRun banner cost the user their shortcut"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

make_cmd_stub 'C:\Users\ci\AppData\Local\Temp   '
OUT=$(run_block)
assert_eq "trailing blanks are trimmed off %TEMP%" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

else
    echo "  SKIP: the four resolution cases need the block above; see the failure printed there"
fi

assert_contains "the %TEMP% probe disables AutoRun" \
    "$(grep 'cmd.exe .*echo .*%TEMP%' "$INSTALL_SH")" "cmd.exe /d /c"

# cmd expands %VAR% before parsing, so an unquoted value holding & splits the command.
assert_contains "the %TEMP% probe quotes the value inside cmd" \
    "$(grep 'cmd.exe .*echo .*%TEMP%' "$INSTALL_SH")" '"%TEMP%"'

LAUNCH=$(grep -n 'powershell.exe -NoProfile -ExecutionPolicy .* -File "\$_css_ps1_win"' "$INSTALL_SH" || true)
assert_contains "the shortcut launch uses RemoteSigned" "$LAUNCH" "-ExecutionPolicy RemoteSigned"
assert_not_contains "the shortcut launch does not relax the policy" "$LAUNCH" "-ExecutionPolicy Bypass"

# Anywhere in the file: a classifier matches the token whether or not the line runs.
POLICY_HITS=$(grep -c -- '-ExecutionPolicy Bypass' "$INSTALL_SH" || true)
assert_eq "install.sh relaxes the execution policy nowhere" "0" "$POLICY_HITS"

# No Add-Type: it runs csc.exe on PowerShell 5.1. SHChangeNotify goes through Python ctypes.
ADDTYPE_HITS=$(grep -c '^[[:space:]]*Add-Type' "$INSTALL_SH" || true)
assert_eq "install.sh compiles no C#" "0" "$ADDTYPE_HITS"
assert_not_contains "install.sh emits no P/Invoke stub" "$(cat "$INSTALL_SH")" "DefinePInvoke""Method"
assert_not_contains "install.sh defines no dynamic assembly" "$(cat "$INSTALL_SH")" "DefineDynamic""Assembly"

# The per-item notification is the only thing that refreshes a rewritten same-name .lnk.
BODY=$(cat "$INSTALL_SH")
assert_contains "per-item SHCNE_UPDATEITEM refresh kept" "$BODY" "f(0x2000,0x1005,p,None)"
assert_contains "global SHCNE_ASSOCCHANGED refresh kept" "$BODY" "f(0x8000000,0x1000,None,None)"

# Reproduces cmd's order: expand %VAR%, then parse metacharacters. & is legal in account names.
make_parsing_cmd_stub() {
    cat > "$STUBS/cmd.exe" <<STUB
#!/bin/sh
# Everything after /c, joined with spaces. cmd receives a command LINE, so the old unquoted
# `echo %TEMP%` form arrives as one line too; joining is what lets this stub judge both forms.
_cmd=""
_take=0
for _a in "\$@"; do
    if [ "\$_take" = 1 ]; then
        if [ -z "\$_cmd" ]; then _cmd="\$_a"; else _cmd="\$_cmd \$_a"; fi
    fi
    [ "\$_a" = "/c" ] && _take=1
done
# Expansion happens BEFORE parsing, exactly as cmd does it.
_cmd=\$(printf '%s' "\$_cmd" | sed "s/%TEMP%/\$(printf '%s' '$1' | sed 's/[&/\\\\]/\\\\&/g')/g")
# Now parse. An & inside double quotes is literal; outside, it separates commands.
_q=0; _first=""; _rest=""; _i=1
while [ "\$_i" -le "\${#_cmd}" ]; do
    _ch=\$(printf '%s' "\$_cmd" | cut -c"\$_i")
    if [ "\$_ch" = '"' ]; then _q=\$((1-_q))
    elif [ "\$_ch" = "&" ] && [ "\$_q" = 0 ]; then
        _rest=\$(printf '%s' "\$_cmd" | cut -c\$((_i+1))-); break
    fi
    _first="\$_first\$_ch"; _i=\$((_i+1))
done
# cmd's echo prints its argument VERBATIM, quotes included: `echo "x"` outputs "x" with the
# quotes. Stripping them here would hide whether install.sh strips them itself, which is the whole
# point of the quoting, and it hid the order in which install.sh trims trailing blanks.
_out=\$(printf '%s' "\$_first" | sed 's/^echo //')
printf '%s\r\n' "\$_out"
[ -n "\$_rest" ] && printf "'%s' is not recognized as an internal or external command\r\n" "\$_rest"
exit 0
STUB
    chmod +x "$STUBS/cmd.exe"
    make_wslpath_stub "$(printf '%s' "$1" | sed 's/[[:space:]]*$//')" "$WINTEMP"
}

make_parsing_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
OUT=$(run_block)
assert_eq "quoted expansion still resolves an ordinary path" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

make_parsing_cmd_stub 'C:\Users\A&B\AppData\Local\Temp'
OUT=$(run_block)
assert_eq "an ampersand in %TEMP% survives expansion" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
SCRIPT=$(printf '%s' "$OUT" | sed -n 2p)
[ -n "$SCRIPT" ] && ok "a script path was allocated despite the ampersand" || bad "the ampersand killed shortcut creation"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1


make_parsing_cmd_stub 'C:\Users\ci\AppData\Local\Temp   '
OUT=$(run_block)
assert_eq "trailing blanks are trimmed after the quotes come off" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# %TEMP% on a share is a remote script location. $1 = %TEMP%, $2 = %LOCALAPPDATA%.
make_three_var_cmd_stub() {
    cat > "$STUBS/cmd.exe" <<STUB
#!/bin/sh
printf '%s\r\n' '"$1"'
printf '%s\r\n' '"$2\\Temp"'
printf '%s\r\n' '"C:\\Windows\\Temp"'
exit 0
STUB
    chmod +x "$STUBS/cmd.exe"
    make_wslpath_stub "$2\\Temp" "$WINTEMP"
}

# %TEMP% on a UNC share must fall back to the local %LOCALAPPDATA%\Temp.
make_three_var_cmd_stub '\\fileserver.corp.example.com\profiles$\ci\Temp' 'C:\Users\ci\AppData\Local'
OUT=$(run_block)
assert_eq "a UNC %TEMP% falls through to a local directory" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
SCRIPT=$(printf '%s' "$OUT" | sed -n 2p)
[ -n "$SCRIPT" ] && ok "a script path was allocated despite a redirected %TEMP%" || bad "a UNC %TEMP% silently killed shortcut creation"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

make_three_var_cmd_stub '\\10.1.2.3\profiles\ci\Temp' 'C:\Users\ci\AppData\Local'
OUT=$(run_block)
assert_eq "an IP-literal UNC %TEMP% also falls through" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# A mapped drive is the same remote zone. $1 = %TEMP%, $2 = %LOCALAPPDATA%, $3 = `net use` output.
make_mapped_drive_stub() {
    cat > "$STUBS/cmd.exe" <<STUB
#!/bin/sh
for _a in "\$@"; do
    case "\$_a" in
        *"net use"*) printf '%s\r\n' '$3'; exit 0 ;;
    esac
done
printf '%s\r\n' '"$1"'
printf '%s\r\n' '"$2\\Temp"'
printf '%s\r\n' '"C:\\Windows\\Temp"'
exit 0
STUB
    chmod +x "$STUBS/cmd.exe"
    cat > "$STUBS/wslpath" <<STUB
#!/bin/sh
[ "\$1" = "-u" ] || exit 1
case "\$2" in
    '$1') printf '%s' '$NETTEMP' ;;
    "$2\\Temp") printf '%s' '$WINTEMP' ;;
    *) printf '%s' '$WINTEMP/not-what-cmd-printed' ;;
esac
STUB
    chmod +x "$STUBS/wslpath"
}

NETTEMP=$(mktemp -d)
trap 'rm -rf "$STUBS" "$WINTEMP" "$NETTEMP"' EXIT
make_mapped_drive_stub 'Z:\Temp' 'C:\Users\ci\AppData\Local' 'OK           Z:        \\fileserver\profiles    Microsoft Windows Network'
OUT=$(run_block)
assert_eq "a mapped network drive %TEMP% falls through to a local directory" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

make_mapped_drive_stub 'C:\Users\ci\AppData\Local\Temp' 'C:\Users\ci\AppData\Local' 'OK           Z:        \\fileserver\profiles    Microsoft Windows Network'
make_wslpath_stub 'C:\Users\ci\AppData\Local\Temp' "$WINTEMP"
OUT=$(run_block)
assert_eq "an unmapped local %TEMP% is still accepted" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# [automount] enabled=false leaves interop working but exposes no Windows drive, so the
# shortcut needs a stdin-fed launch path.
BODY_ALL=$(cat "$INSTALL_SH")
assert_contains "the generated script is captured once, not written straight to a file" \
    "$BODY_ALL" '_css_ps1_body=$(cat << WSLPS1_EOF'
assert_contains "there is a stdin launch for when no Windows directory is reachable" \
    "$BODY_ALL" 'powershell.exe -NoProfile -Command -'
# Execution policy applies to -File, not -Command, so this path needs no relaxation.
STDIN_LINE=$(printf '%s' "$BODY_ALL" | grep -- 'powershell.exe -NoProfile -Command -')
case "$STDIN_LINE" in
    *ExecutionPolicy*) bad "the stdin launch carries an execution policy flag it does not need: $STDIN_LINE" ;;
    *) ok "the stdin launch relaxes no execution policy" ;;
esac
# Fed from our pipe, never the installer's stdin, which in a piped install is the download.
case "$STDIN_LINE" in
    *printf*"|"*powershell.exe*) ok "the stdin launch is fed from its own pipe" ;;
    *) bad "the stdin launch does not pipe the body in: $STDIN_LINE" ;;
esac

# Must be the last thing the file does: `bad` returns success, so later failures would exit 0.
echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
