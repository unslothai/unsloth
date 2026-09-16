#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# install.sh writes a PowerShell script and runs it on the Windows side to create the WSL shortcut.
# Where that script lands decides whether the launch needs a relaxed execution policy: wslpath maps
# a WSL path to a \\wsl.localhost\<distro>\... UNC path, which PowerShell treats as a remote script
# and RemoteSigned refuses unsigned, so the launch used to relax the policy instead. Under the
# Windows %TEMP% the script is on a local volume, RemoteSigned loads it, and the relaxed policy is
# unnecessary.
#
# Exercises the real resolution block out of install.sh rather than a copy, with cmd.exe and wslpath
# stubbed, so the three outcomes that matter are covered on any host: resolved, unresolvable, and
# resolved-but-bogus. The static half then pins what the file must and must not contain.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

# From the first line of the resolution to the mktemp that consumes it. Tracking install.sh this way
# means a refactor that drops the guard fails here instead of silently going untested.
BLOCK=$(awk '
    /^        _css_win_temp=""$/ { grab = 1 }
    grab { print }
    grab && /_css_ps1_tmp=\$\(mktemp "\$_css_win_temp\/unsloth-shortcut-XXXXXX\.ps1"/ { print "        fi"; exit }
' "$INSTALL_SH")

echo "=== test_wsl_shortcut_script_location ==="

# Extraction failure is a FAIL, not an early exit, so the static half below still runs and the
# output names every problem at once instead of only the first. It is still fail-closed: FAIL is
# non-zero at the end, so a suite that cannot find what it measures never reports a pass.
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

# wslpath -u <windows path> -> our fake mount point.
#
# INPUT-SENSITIVE, and that is the whole point of the stub. A wslpath that returns the fake
# Windows temp whatever it is handed cannot tell "install.sh read the path correctly" from
# "install.sh read a banner and a path glued together", so every case below about WHAT cmd.exe
# printed would pass against a block that got it wrong. Only the exact string maps; anything
# else maps somewhere that does not exist, which is what the real wslpath does with a string
# that is not a path.
# $1 = the only string that maps, $2 = where it maps to.
make_wslpath_stub() {
    cat > "$STUBS/wslpath" <<STUB
#!/bin/sh
[ "\$1" = "-u" ] || exit 1
if [ "\$2" = '$1' ]; then printf '%s' '$2'; else printf '%s' '$2/not-what-cmd-printed'; fi
STUB
    chmod +x "$STUBS/wslpath"
}

# $1 is what `echo %TEMP%` prints, and is also the only string wslpath will map. $2, when given,
# is an AutoRun banner that precedes it on the SAME stdout unless cmd.exe is passed /d.
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
    # Trailing blanks are not part of the path: Win32 strips them, so the block has to.
    make_wslpath_stub "$(printf '%s' "$1" | sed 's/[[:space:]]*$//')" "$WINTEMP"
}

run_block() {
    _css_win_temp=""
    _css_ps1_tmp=""
    ( PATH="$STUBS:$PATH"; eval "$BLOCK"; printf '%s\n%s\n' "$_css_win_temp" "$_css_ps1_tmp" )
}

if [ "$HAVE_BLOCK" -eq 1 ]; then

# 1. Interop works: the script is created under the Windows temp, not in WSL's /tmp.
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
# Structural, not a path-prefix check. The first version of this asserted the script path did not
# start with /tmp, which is a proxy for "not in WSL's own temp" -- and a broken one: the harness
# creates its fake Windows temp with mktemp -d, so on any machine with TMPDIR=/tmp (which is every
# normal CI runner) the stub's own directory matched the pattern and the test failed while install.sh
# was behaving correctly. It passed locally only because this workspace sets TMPDIR elsewhere.
#
# What actually has to hold is that the only place the block can allocate the script is the resolved
# Windows temp, with no second mktemp anywhere to fall back to. That is true or false regardless of
# where anyone's temp directory happens to live.
ALLOC=$(printf '%s' "$BLOCK" | grep -c 'mktemp' || true)
if [ "$ALLOC" -eq 1 ]; then
    ok "exactly one mktemp in the block, so there is no second path to fall back to"
else
    bad "the block contains $ALLOC mktemp calls; a second one is a fallback that can land in WSL's /tmp, which wslpath turns into a \\\\wsl.localhost UNC path that RemoteSigned refuses"
fi
assert_contains "the script is allocated inside the resolved Windows temp" \
    "$(printf '%s' "$BLOCK" | grep 'mktemp')" '$_css_win_temp/'
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# 2. cmd.exe absent (interop disabled): nothing is allocated, so the caller takes its existing
#    "couldn't create the Windows shortcut" notice. It must NOT fall back to a WSL path, because
#    that is the case that needed the relaxed policy.
rm -f "$STUBS/cmd.exe"
OUT=$(run_block)
assert_eq "no interop: %TEMP% unresolved" "" "$(printf '%s' "$OUT" | sed -n 1p)"
assert_eq "no interop: no script allocated, so no UNC fallback" "" "$(printf '%s' "$OUT" | sed -n 2p)"

# 3. Interop present but %TEMP% unset: cmd.exe echoes the name back unexpanded. Reading that as a
#    directory would try to mktemp inside a literal "%TEMP%".
make_cmd_stub '%TEMP%'
OUT=$(run_block)
assert_eq "unexpanded %TEMP% is rejected" "" "$(printf '%s' "$OUT" | sed -n 1p)"
assert_eq "unexpanded %TEMP% allocates nothing" "" "$(printf '%s' "$OUT" | sed -n 2p)"

# 4. cmd.exe prints a path that does not exist: wslpath succeeds, the directory does not.
make_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
make_wslpath_stub 'C:\Users\ci\AppData\Local\Temp' "$WINTEMP/definitely-absent"
OUT=$(run_block)
assert_eq "a %TEMP% that is not a directory is rejected" "" "$(printf '%s' "$OUT" | sed -n 1p)"

# 5. cmd.exe runs an AutoRun command first, on the same stdout.
#
#    HKCU\Software\Microsoft\Command Processor\AutoRun runs before anything else unless /d is
#    passed, and `cmd /c` is not exempt. Clink sets one, so Cmder does, and so do plenty of
#    corporate images. Reading the whole stream glues the banner to the front of the path, which
#    wslpath then rejects, and every one of those users silently loses the shortcut they used to
#    get. The fix is /d, not parsing harder, so the stub only suppresses the banner for /d.
make_cmd_stub 'C:\Users\ci\AppData\Local\Temp' 'clink v1.6.20 is available.'
OUT=$(run_block)
assert_eq "an AutoRun banner does not corrupt %TEMP%" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
[ -n "$(printf '%s' "$OUT" | sed -n 2p)" ] \
    && ok "an AutoRun banner still allocates the script" \
    || bad "an AutoRun banner cost the user their shortcut"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# 6. Trailing blanks on the value. Win32 strips them from a path, [ -d ] does not, so a %TEMP%
#    set with one would otherwise resolve to a directory that appears not to exist.
make_cmd_stub 'C:\Users\ci\AppData\Local\Temp   '
OUT=$(run_block)
assert_eq "trailing blanks are trimmed off %TEMP%" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

else
    echo "  SKIP: the four resolution cases need the block above; see the failure printed there"
fi

# --- Static half: what the shipped file may and may not contain. -------------------------------

# /d on the cmd.exe that reads %TEMP%, so no AutoRun command gets to print onto the stdout this
# block is reading. Static as well as behavioural: the case above proves the block survives a
# banner, this proves it survives it the cheap way rather than by out-parsing whatever prints.
assert_contains "the %TEMP% probe disables AutoRun" \
    "$(grep 'cmd.exe .*echo .*%TEMP%' "$INSTALL_SH")" "cmd.exe /d /c"

# The value is expanded inside quotes. cmd expands before it parses, so an unquoted %TEMP% holding
# an & is split into two commands; the behavioural case above proves the block survives that, and
# this pins the mechanism so a rewrite cannot quietly drop the quoting and still pass on paths that
# happen to contain no metacharacter.
assert_contains "the %TEMP% probe quotes the value inside cmd" \
    "$(grep 'cmd.exe .*echo .*%TEMP%' "$INSTALL_SH")" '"%TEMP%"'

LAUNCH=$(grep -n 'powershell.exe -NoProfile -ExecutionPolicy .* -File "\$_css_ps1_win"' "$INSTALL_SH" || true)
assert_contains "the shortcut launch uses RemoteSigned" "$LAUNCH" "-ExecutionPolicy RemoteSigned"
assert_not_contains "the shortcut launch does not relax the policy" "$LAUNCH" "-ExecutionPolicy Bypass"

# Anywhere in the file, not just this launch: a relaxed policy is a token a classifier matches
# whether or not the line runs.
POLICY_HITS=$(grep -c -- '-ExecutionPolicy Bypass' "$INSTALL_SH" || true)
assert_eq "install.sh relaxes the execution policy nowhere" "0" "$POLICY_HITS"

# The generated script declares SHChangeNotify by emitting it. Add-Type -MemberDefinition writes C#
# to %TEMP% and runs csc.exe on PowerShell 5.1, and behavioural antivirus blocks the DLL that comes
# out; #10540 removed that from install.ps1 and this copy was missed for a release.
ADDTYPE_HITS=$(grep -c '^[[:space:]]*Add-Type' "$INSTALL_SH" || true)
assert_eq "install.sh compiles no C#" "0" "$ADDTYPE_HITS"
assert_contains "SHChangeNotify is emitted instead" "$(cat "$INSTALL_SH")" "DefinePInvokeMethod"

# Both notifications survive. The per-item one is the only thing that recovers a rewritten
# same-name .lnk; the global broadcast alone does not, so losing it would show as a stale icon.
BODY=$(cat "$INSTALL_SH")
assert_contains "per-item SHCNE_UPDATEITEM refresh kept" "$BODY" "SHChangeNotify(0x00002000, 0x0005"
assert_contains "global SHCNE_ASSOCCHANGED refresh kept" "$BODY" "SHChangeNotify(0x08000000, 0"

# A cmd.exe stub that reproduces the ORDER cmd works in: %VAR% is expanded first, and the result is
# then parsed for metacharacters. That is what makes an & inside the value dangerous, and the
# earlier stub could not show it because it echoed a fixed string without ever parsing a command.
# & is legal in a Windows account name, so "C:\Users\A&B\AppData\Local\Temp" is a real path.
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
    # Trimmed, like the other stub: Win32 strips trailing blanks from a path, so the block must hand
    # wslpath the trimmed value and this has to agree or the case could never pass.
    make_wslpath_stub "$(printf '%s' "$1" | sed 's/[[:space:]]*$//')" "$WINTEMP"
}

# A plain path through the faithful stub: the quoting must not corrupt the ordinary case.
make_parsing_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
OUT=$(run_block)
assert_eq "quoted expansion still resolves an ordinary path" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# The case this guards: an & in the Windows temp path. Unquoted, cmd splits the line, the echo
# prints a truncated path, wslpath is handed something that is not the temp directory and the whole
# shortcut is silently skipped.
make_parsing_cmd_stub 'C:\Users\A&B\AppData\Local\Temp'
OUT=$(run_block)
assert_eq "an ampersand in %TEMP% survives expansion" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
SCRIPT=$(printf '%s' "$OUT" | sed -n 2p)
[ -n "$SCRIPT" ] && ok "a script path was allocated despite the ampersand" || bad "the ampersand killed shortcut creation"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1


# Trailing blanks THROUGH the quoting path. The older trailing-blank case used the fixed-string
# stub, which emits the value unquoted, so it never saw that the trim now runs against a string
# whose last character is a closing quote. Win32 strips trailing spaces from a path and [ -d ] does
# not, so a %TEMP% of "C:\Temp   " has to come back as C:\Temp or the directory check clears it and
# shortcut creation is skipped.
make_parsing_cmd_stub 'C:\Users\ci\AppData\Local\Temp   '
OUT=$(run_block)
assert_eq "trailing blanks are trimmed after the quotes come off" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# %TEMP% redirected onto a share. A script written there is a REMOTE script and RemoteSigned
# refuses an unsigned one, so before the fallback these users silently stopped getting a shortcut
# they used to get. $1 is what %TEMP% expands to, $2 what %LOCALAPPDATA% does.
make_three_var_cmd_stub() {
    # Prints the three candidates in the order the block asks for them, each wrapped in quotes the
    # way cmd's echo emits them. The expand-then-parse behaviour is covered by the other stub; what
    # this one exercises is which candidate install.sh SELECTS.
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

# The regression this guards: %TEMP% on a UNC share must not kill shortcut creation. The fallback
# has to land on the local %LOCALAPPDATA%\Temp instead, and it must still be RemoteSigned.
make_three_var_cmd_stub '\\fileserver.corp.example.com\profiles$\ci\Temp' 'C:\Users\ci\AppData\Local'
OUT=$(run_block)
assert_eq "a UNC %TEMP% falls through to a local directory" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
SCRIPT=$(printf '%s' "$OUT" | sed -n 2p)
[ -n "$SCRIPT" ] && ok "a script path was allocated despite a redirected %TEMP%" || bad "a UNC %TEMP% silently killed shortcut creation"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# And an IP-literal UNC, which is the same zone and a different spelling.
make_three_var_cmd_stub '\\10.1.2.3\profiles\ci\Temp' 'C:\Users\ci\AppData\Local'
OUT=$(run_block)
assert_eq "an IP-literal UNC %TEMP% also falls through" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# A mapped network drive is the same share and the same remote zone as its UNC spelling, so a
# %TEMP% on Z: has to fall through too. $1 = %TEMP%, $2 = %LOCALAPPDATA%, $3 = the `net use` output.
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
    # Maps BOTH the mapped-drive path and the local one to real directories, so the mapped drive is
    # genuinely usable. Otherwise the old code would fall through for the wrong reason -- an
    # unmappable path -- and the case could never show the difference.
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

# And a LOCAL drive letter must not be rejected just because some other letter is mapped.
make_mapped_drive_stub 'C:\Users\ci\AppData\Local\Temp' 'C:\Users\ci\AppData\Local' 'OK           Z:        \\fileserver\profiles    Microsoft Windows Network'
make_wslpath_stub 'C:\Users\ci\AppData\Local\Temp' "$WINTEMP"
OUT=$(run_block)
assert_eq "an unmapped local %TEMP% is still accepted" "$WINTEMP" "$(printf '%s' "$OUT" | sed -n 1p)"
rm -f "$WINTEMP"/unsloth-shortcut-*.ps1

# ---------------------------------------------------------------------------
# The launch path, statically. [automount] enabled=false leaves interop working while exposing no
# Windows drive as a Linux directory, so no candidate can be allocated and the file-based launch
# cannot run at all. Before the move off /tmp those users still got a shortcut, so silently losing
# it is a regression rather than a pre-existing gap.
# ---------------------------------------------------------------------------
BODY_ALL=$(cat "$INSTALL_SH")
assert_contains "the generated script is captured once, not written straight to a file" \
    "$BODY_ALL" '_css_ps1_body=$(cat << WSLPS1_EOF'
assert_contains "there is a stdin launch for when no Windows directory is reachable" \
    "$BODY_ALL" 'powershell.exe -NoProfile -Command -'
# Execution policy applies to -File and not to -Command, so the fallback needs no relaxation and
# must not acquire one.
STDIN_LINE=$(printf '%s' "$BODY_ALL" | grep -- 'powershell.exe -NoProfile -Command -')
case "$STDIN_LINE" in
    *ExecutionPolicy*) bad "the stdin launch carries an execution policy flag it does not need: $STDIN_LINE" ;;
    *) ok "the stdin launch relaxes no execution policy" ;;
esac
# It must be fed from our own pipe, never from the installer's stdin, which under `curl | sh` is
# still the download (#7548).
case "$STDIN_LINE" in
    *printf*"|"*powershell.exe*) ok "the stdin launch is fed from its own pipe" ;;
    *) bad "the stdin launch does not pipe the body in: $STDIN_LINE" ;;
esac

# The one place the counters are read, and it has to be the LAST thing the file does. `bad` records
# a failure and returns success, so any assertion that runs after this check reports its failure and
# still leaves the script exiting 0.
echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
