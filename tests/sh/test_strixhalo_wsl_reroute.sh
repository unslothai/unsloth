#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for _maybe_reroute_strixhalo_to_2404() from install.sh.
#
# ROCm-on-WSL only targets Ubuntu 24.04: from a newer distro (e.g. 26.04) with a
# 24.04 distro present, re-run the install there and stop; otherwise leave the distro
# alone and let CPU-fallback print the `wsl --install` hint. Tested hermetically: the
# function is extracted from install.sh, its absolute paths rewritten to per-test
# fixtures, with a mock wsl.exe (no real WSL).
#
# Follows the extract-via-sed pattern of test_get_torch_index_url.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# All fixtures/temp files live under one root removed on exit, so a set -e abort
# can't leak dirs into $TMPDIR.
_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

assert_contains() {
    _label="$1"; _hay="$2"; _needle="$3"
    case "$_hay" in
        *"$_needle"*) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
        *) echo "  FAIL: $_label (missing '$_needle' in: $_hay)"; FAIL=$((FAIL + 1)) ;;
    esac
}

assert_absent() {
    _label="$1"; _hay="$2"; _needle="$3"
    case "$_hay" in
        *"$_needle"*) echo "  FAIL: $_label (unexpected '$_needle' in: $_hay)"; FAIL=$((FAIL + 1)) ;;
        *) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
    esac
}

# Extract the function and rewrite its hardcoded absolute paths to point at the
# fixture dir $1, plus stub substep()/colors. The mock wsl.exe lives in $1/bin.
build_func() {
    _fix="$1"
    _f=$(mktemp -p "$_TMP_ROOT")
    {
        echo 'substep() { printf "  %s\n" "$1"; }'
        echo 'C_WARN=""; C_OK=""'
        sed -n '/^_maybe_reroute_strixhalo_to_2404()/,/^}/p' "$INSTALL_SH"
    } | sed \
        -e "s#/dev/dxg#$_fix/dxg#g" \
        -e "s#/proc/cpuinfo#$_fix/cpuinfo#g" \
        -e "s#/opt/rocm/lib/librocdxg.so#$_fix/rocm-lib/librocdxg.so#g" \
        -e "s#/opt/rocm/lib64/librocdxg.so#$_fix/rocm-lib64/librocdxg.so#g" \
        -e "s#/etc/os-release#$_fix/os-release#g" \
        > "$_f"
    echo "$_f"
}

# make_fixture DXG CPU LIBROCDXG VER HAS2404
#   DXG/LIBROCDXG/HAS2404 = 1|0 ; CPU = strix|other ; VER = e.g. 26.04|24.04
make_fixture() {
    _d=$(mktemp -d -p "$_TMP_ROOT")
    mkdir -p "$_d/bin" "$_d/rocm-lib" "$_d/rocm-lib64"
    [ "$1" = 1 ] && : > "$_d/dxg"
    if [ "$2" = strix ]; then
        echo "model name : AMD Ryzen AI Max+ 395 w/ Radeon 8060S" > "$_d/cpuinfo"
    else
        echo "model name : Generic x86 CPU" > "$_d/cpuinfo"
    fi
    [ "$3" = 1 ] && : > "$_d/rocm-lib/librocdxg.so"
    printf 'VERSION_ID="%s"\n' "$4" > "$_d/os-release"
    if [ "$5" = 1 ]; then
        printf 'Ubuntu\nUbuntu-24.04\n' > "$_d/distros"
    else
        printf 'Ubuntu\n' > "$_d/distros"
    fi
    # Mock wsl.exe: `-l [-q]` lists distros; `-d <distro> -- <cmd...>` runs <cmd>
    # locally so the reroute command (echo __ROUTED__) is observable.
    cat > "$_d/bin/wsl.exe" <<MOCK
#!/bin/sh
case "\$1" in
    -l) cat "$_d/distros" ;;
    -d) _td="\$2"; shift; shift; shift; printf '__CMD__ -d %s -- %s\n' "\$_td" "\$*"; exec "\$@" ;;
esac
MOCK
    chmod +x "$_d/bin/wsl.exe"
    # Harmless curl stub so the default reroute (curl | sh) never hits the network.
    printf '#!/bin/sh\nexit 0\n' > "$_d/bin/curl"
    chmod +x "$_d/bin/curl"
    echo "$_d"
}

# run_func FIXTURE_DIR [extra VAR=val ...]
# Prints the function's stdout. "__ROUTED__" => the reroute exec ran;
# "__NOROUTE__" => the function returned without rerouting (exit 0 not hit).
run_func() {
    _fix="$1"; shift
    _func=$(build_func "$_fix")
    env PATH="$_fix/bin:$PATH" \
        OS=wsl SKIP_TORCH=false \
        UNSLOTH_SKIP_ROCM_WSL_SETUP=0 UNSLOTH_WSL_REROUTED=0 \
        UNSLOTH_WSL_REROUTE_CMD='echo __ROUTED__' \
        "$@" \
        bash -c ". '$_func'; _maybe_reroute_strixhalo_to_2404; echo SKIP_ROCM=\$UNSLOTH_SKIP_ROCM_WSL_SETUP; echo __NOROUTE__" 2>&1
    _rc=$?
    rm -f "$_func"
    return "$_rc"
}

echo "=== test_strixhalo_wsl_reroute ==="

# 1) 26.04, Strix Halo, no librocdxg, 24.04 exists -> ROUTE into Ubuntu-24.04
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d")
assert_contains "26.04 + existing 24.04 -> routes"      "$_out" "__ROUTED__"
assert_absent   "26.04 route stops current distro"      "$_out" "__NOROUTE__"
rm -rf "$_d"

# 2) Already on 24.04 -> NO route (correct distro, leave it alone)
_d=$(make_fixture 1 strix 0 24.04 1)
_out=$(run_func "$_d")
assert_contains "on 24.04 -> no route"                  "$_out" "__NOROUTE__"
assert_absent   "on 24.04 -> reroute not attempted"     "$_out" "__ROUTED__"
rm -rf "$_d"

# 3) librocdxg already present (working ROCm-on-WSL) -> NO route even on 26.04
_d=$(make_fixture 1 strix 1 26.04 1)
_out=$(run_func "$_d")
assert_contains "librocdxg present -> no route"         "$_out" "__NOROUTE__"
assert_absent   "librocdxg present -> not rerouted"     "$_out" "__ROUTED__"
rm -rf "$_d"

# 4) 26.04 but no Ubuntu-24.04 distro -> NO route (install-only-on-prompt path).
#    Unsupported distro with no reroute target must also skip the origin ROCm bootstrap.
_d=$(make_fixture 1 strix 0 26.04 0)
_out=$(run_func "$_d")
assert_contains "26.04 + no 24.04 distro -> no route"   "$_out" "__NOROUTE__"
assert_absent   "26.04 + no 24.04 -> not rerouted"      "$_out" "__ROUTED__"
assert_contains "26.04 + no 24.04 -> skip ROCm bootstrap" "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 5) No /dev/dxg (no WSL GPU passthrough) -> NO route
_d=$(make_fixture 0 strix 0 26.04 1)
_out=$(run_func "$_d")
assert_contains "no /dev/dxg -> no route"               "$_out" "__NOROUTE__"
rm -rf "$_d"

# 6) Non-Strix CPU -> NO route (don't reroute generic AMD/Intel WSL)
_d=$(make_fixture 1 other 0 26.04 1)
_out=$(run_func "$_d")
assert_contains "non-Strix CPU -> no route"             "$_out" "__NOROUTE__"
rm -rf "$_d"

# 7) Loop guard: UNSLOTH_WSL_REROUTED=1 -> NO route (the rerouted run must not re-route)
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_WSL_REROUTED=1)
assert_contains "REROUTED=1 loop guard -> no route"     "$_out" "__NOROUTE__"
assert_absent   "REROUTED=1 -> not rerouted again"      "$_out" "__ROUTED__"
rm -rf "$_d"

# 8) Opt-out: UNSLOTH_SKIP_ROCM_WSL_SETUP=1 -> NO route
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_SKIP_ROCM_WSL_SETUP=1)
assert_contains "SKIP_ROCM_WSL_SETUP=1 -> no route"     "$_out" "__NOROUTE__"
rm -rf "$_d"

# 9) Not WSL (OS=linux) -> NO route
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" OS=linux)
assert_contains "OS=linux -> no route"                  "$_out" "__NOROUTE__"
rm -rf "$_d"

# 10) SKIP_TORCH=true (GGUF-only) -> NO route (no GPU torch needed)
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" SKIP_TORCH=true)
assert_contains "SKIP_TORCH=true -> no route"           "$_out" "__NOROUTE__"
rm -rf "$_d"

# 11) Loop-guard payload: the reroute exports UNSLOTH_WSL_REROUTED=1 into the child
#     (so the nested install short-circuits gate 7). Verify it reaches wsl.exe -d.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_WSL_REROUTE_CMD='echo flag=[$UNSLOTH_WSL_REROUTED]')
assert_contains "reroute exports loop-guard flag"       "$_out" "flag=[1]"
rm -rf "$_d"

# 12) On Ubuntu 22.04 -> unsupported by the ROCm-on-WSL bootstrap (helper targets
#     24.04 only), so with a 24.04 distro present reroute the GPU install there.
_d=$(make_fixture 1 strix 0 22.04 1)
_out=$(run_func "$_d")
assert_contains "22.04 + existing 24.04 -> routes"      "$_out" "__ROUTED__"
assert_absent   "22.04 route stops current distro"      "$_out" "__NOROUTE__"
rm -rf "$_d"

# 13) --local install -> NO auto-reroute (a local checkout can't be replayed via
#     curl|sh); prints guidance and continues locally instead of a different install.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" STUDIO_LOCAL_INSTALL=true)
assert_contains "--local -> no auto-reroute"            "$_out" "__NOROUTE__"
assert_contains "--local -> prints re-run guidance"     "$_out" "re-run it from Ubuntu-24.04"
assert_absent   "--local -> reroute command not run"    "$_out" "__ROUTED__"
assert_contains "--local -> skip ROCm bootstrap"        "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 14) Custom UNSLOTH_STUDIO_HOME (env mode) is forwarded as an export into the reroute.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" _STUDIO_HOME_REDIRECT=env STUDIO_HOME=/custom/studio \
        UNSLOTH_WSL_REROUTE_CMD='echo home=[$UNSLOTH_STUDIO_HOME]')
assert_contains "custom STUDIO_HOME forwarded to reroute" "$_out" "home=[/custom/studio]"
rm -rf "$_d"

# 15) --package / --python are forwarded onto the default reroute command (curl|sh -s --).
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" PACKAGE_NAME=unsloth-zoo _USER_PYTHON=3.13 UNSLOTH_WSL_REROUTE_CMD=)
assert_contains "forwards --package to reroute"          "$_out" "--package 'unsloth-zoo'"
assert_contains "forwards --python to reroute"           "$_out" "--python '3.13'"
rm -rf "$_d"

# 16) --tauri is forwarded onto the default reroute command.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" PACKAGE_NAME=unsloth TAURI_MODE=true UNSLOTH_WSL_REROUTE_CMD=)
assert_contains "forwards --tauri to reroute"            "$_out" "--tauri"
rm -rf "$_d"

# 17) A FAILED reroute sets the ROCm-bootstrap skip guard so _maybe_bootstrap_rocm_wsl
#     does not later install ROCm into the unsupported origin distro.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_WSL_REROUTE_CMD='exit 1')
assert_contains "failed reroute -> sets ROCm-bootstrap skip guard" "$_out" "SKIP_ROCM=1"
assert_contains "failed reroute -> continues locally"             "$_out" "__NOROUTE__"
rm -rf "$_d"

# 18) A successful reroute (no failure) does NOT set the skip guard prematurely.
_d=$(make_fixture 1 strix 0 24.04 1)
_out=$(run_func "$_d")
assert_absent   "supported 24.04 -> skip guard not forced"        "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 19) No wsl.exe on an unsupported distro -> can't reach a 24.04 target, so stay
#     CPU-only AND set the skip guard (don't bootstrap ROCm into 26.04 etc.).
#     Drop the stub AND pin PATH to coreutils so a real host wsl.exe can't leak in.
_d=$(make_fixture 1 strix 0 26.04 1)
rm -f "$_d/bin/wsl.exe"
_out=$(run_func "$_d" PATH="$_d/bin:/usr/bin:/bin")
assert_contains "no wsl.exe -> no route"                          "$_out" "__NOROUTE__"
assert_absent   "no wsl.exe -> not rerouted"                      "$_out" "__ROUTED__"
assert_contains "no wsl.exe -> skip ROCm bootstrap"               "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 20) UNSLOTH_ROCM_WSL_AUTO=1 consent is forwarded as an export into the reroute so
#     the child auto-enables the GPU bootstrap instead of the desktop-app prompt.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_ROCM_WSL_AUTO=1 \
        UNSLOTH_WSL_REROUTE_CMD='echo auto=[$UNSLOTH_ROCM_WSL_AUTO]')
assert_contains "UNSLOTH_ROCM_WSL_AUTO forwarded to reroute"      "$_out" "auto=[1]"
rm -rf "$_d"

# 21) Both 24.04 and 22.04 installed -> target 24.04 (the only bootstrap-supported one).
_d=$(make_fixture 1 strix 0 26.04 1)
printf 'Ubuntu\nUbuntu-22.04\nUbuntu-24.04\n' > "$_d/distros"
_out=$(run_func "$_d")
assert_contains "both present -> routes"                          "$_out" "__ROUTED__"
assert_contains "both present -> targets 24.04"                   "$_out" "-d Ubuntu-24.04"
assert_absent   "both present -> does not target 22.04"           "$_out" "-d Ubuntu-22.04"
rm -rf "$_d"

# 22) Only 22.04 installed (no 24.04) -> no route. 22.04 isn't a bootstrap target,
#     so stay CPU-only and skip the origin ROCm bootstrap.
_d=$(make_fixture 1 strix 0 26.04 0)
printf 'Ubuntu\nUbuntu-22.04\n' > "$_d/distros"
_out=$(run_func "$_d")
assert_contains "only 22.04 -> no route"                          "$_out" "__NOROUTE__"
assert_absent   "only 22.04 -> not rerouted"                      "$_out" "__ROUTED__"
assert_contains "only 22.04 -> skip ROCm bootstrap"               "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 23) Neither 24.04 nor 22.04 present -> stay CPU-only and skip the ROCm bootstrap.
_d=$(make_fixture 1 strix 0 26.04 0)
printf 'Ubuntu\nUbuntu-20.04\n' > "$_d/distros"
_out=$(run_func "$_d")
assert_contains "no supported target -> no route"                 "$_out" "__NOROUTE__"
assert_contains "no supported target -> skip ROCm bootstrap"      "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 24) A custom distro that merely CONTAINS the name (Ubuntu-24.04-test) but has no
#     exact Ubuntu-24.04 must NOT be picked (substring match would fail wsl -d).
_d=$(make_fixture 1 strix 0 26.04 0)
printf 'Ubuntu\nUbuntu-24.04-test\n' > "$_d/distros"
_out=$(run_func "$_d")
assert_contains "substring-only distro -> no route"               "$_out" "__NOROUTE__"
assert_absent   "substring-only distro -> not rerouted"           "$_out" "__ROUTED__"
assert_contains "substring-only distro -> skip ROCm bootstrap"    "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

# 25) Exact Ubuntu-24.04 alongside a custom Ubuntu-24.04-test -> route to the exact one.
_d=$(make_fixture 1 strix 0 26.04 0)
printf 'Ubuntu\nUbuntu-24.04-test\nUbuntu-24.04\n' > "$_d/distros"
_out=$(run_func "$_d")
assert_contains "exact + custom -> routes"                        "$_out" "__ROUTED__"
assert_contains "exact + custom -> targets the exact 24.04"       "$_out" "-d Ubuntu-24.04 --"
rm -rf "$_d"

# 26) Tauri mode: a child exit 2 ([TAURI:NEED_SUDO]) must propagate so the desktop app
#     drives elevation for the target distro, not get masked as a CPU fallback here.
_d=$(make_fixture 1 strix 0 26.04 1)
_rc=0
_out=$(run_func "$_d" TAURI_MODE=true UNSLOTH_WSL_REROUTE_CMD='exit 2') || _rc=$?
if [ "$_rc" = "2" ]; then echo "  PASS: tauri child exit 2 -> reroute propagates exit 2"; PASS=$((PASS+1)); else echo "  FAIL: tauri exit 2 not propagated (rc=$_rc)"; FAIL=$((FAIL+1)); fi
assert_absent   "tauri exit 2 -> not a CPU fallback"              "$_out" "__NOROUTE__"
rm -rf "$_d"

# 27) The post-install autostart opt-out must reach the target distro, where the
#     final launch prompt is evaluated.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" _SKIP_AUTOSTART=true UNSLOTH_SKIP_AUTOSTART= \
        UNSLOTH_WSL_REROUTE_CMD='echo skip=[$UNSLOTH_SKIP_AUTOSTART]')
assert_contains "UNSLOTH_SKIP_AUTOSTART forwarded to reroute"    "$_out" "skip=[1]"
rm -rf "$_d"

# 28) Non-tauri mode: a child exit 2 is just a failure -> CPU fallback, not propagated.
_d=$(make_fixture 1 strix 0 26.04 1)
_rc=0
_out=$(run_func "$_d" UNSLOTH_WSL_REROUTE_CMD='exit 2') || _rc=$?
if [ "$_rc" = "0" ]; then echo "  PASS: non-tauri exit 2 -> not propagated"; PASS=$((PASS+1)); else echo "  FAIL: non-tauri exit 2 wrongly propagated (rc=$_rc)"; FAIL=$((FAIL+1)); fi
assert_contains "non-tauri child fail -> CPU fallback"            "$_out" "__NOROUTE__"
assert_contains "non-tauri child fail -> skip ROCm bootstrap"     "$_out" "SKIP_ROCM=1"
rm -rf "$_d"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
