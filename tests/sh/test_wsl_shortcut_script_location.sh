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

# wslpath -u <windows path> -> our fake mount point. Mirrors the real contract closely enough: the
# block only ever calls `wslpath -u` on the string cmd.exe printed.
cat > "$STUBS/wslpath" <<STUB
#!/bin/sh
[ "\$1" = "-u" ] || exit 1
printf '%s' "$WINTEMP"
STUB
chmod +x "$STUBS/wslpath"

make_cmd_stub() {
    cat > "$STUBS/cmd.exe" <<STUB
#!/bin/sh
# Real cmd.exe emits CRLF; the block strips it, so emit one here or the test would not cover that.
printf '%s\r\n' '$1'
STUB
    chmod +x "$STUBS/cmd.exe"
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
cat > "$STUBS/wslpath" <<STUB
#!/bin/sh
[ "\$1" = "-u" ] || exit 1
printf '%s' "$WINTEMP/definitely-absent"
STUB
chmod +x "$STUBS/wslpath"
make_cmd_stub 'C:\Users\ci\AppData\Local\Temp'
OUT=$(run_block)
assert_eq "a %TEMP% that is not a directory is rejected" "" "$(printf '%s' "$OUT" | sed -n 1p)"

else
    echo "  SKIP: the four resolution cases need the block above; see the failure printed there"
fi

# --- Static half: what the shipped file may and may not contain. -------------------------------

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

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
