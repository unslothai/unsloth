#!/bin/bash
# Unit tests for _maybe_reroute_strixhalo_to_2404() from install.sh.
#
# ROCm-on-WSL (the Strix Halo GPU runtime) only targets Ubuntu 24.04. When the
# installer runs in a newer default WSL distro (e.g. 26.04) but an Ubuntu-24.04
# distro already exists, it must re-run the install there and stop; otherwise it
# must leave the current distro alone and let the CPU-fallback path print the
# `wsl --install` guidance. This exercises that decision matrix hermetically:
# the function is extracted from install.sh and its hardcoded absolute paths are
# rewritten to per-test fixtures, with a mock wsl.exe on PATH (no real WSL).
#
# Follows the extract-via-sed pattern of test_get_torch_index_url.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

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
    _f=$(mktemp)
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
    _d=$(mktemp -d)
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
    -d) shift; shift; shift; exec "\$@" ;;
esac
MOCK
    chmod +x "$_d/bin/wsl.exe"
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
        bash -c ". '$_func'; _maybe_reroute_strixhalo_to_2404; echo __NOROUTE__" 2>&1
    rm -f "$_func"
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

# 4) 26.04 but no Ubuntu-24.04 distro -> NO route (install-only-on-prompt path)
_d=$(make_fixture 1 strix 0 26.04 0)
_out=$(run_func "$_d")
assert_contains "26.04 + no 24.04 distro -> no route"   "$_out" "__NOROUTE__"
assert_absent   "26.04 + no 24.04 -> not rerouted"      "$_out" "__ROUTED__"
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

# 11) Loop-guard payload check: the reroute exec exports UNSLOTH_WSL_REROUTED=1
#     into the child so the nested install short-circuits gate 7. Verify the
#     export is part of the command handed to wsl.exe -d.
_d=$(make_fixture 1 strix 0 26.04 1)
_out=$(run_func "$_d" UNSLOTH_WSL_REROUTE_CMD='echo flag=[$UNSLOTH_WSL_REROUTED]')
assert_contains "reroute exports loop-guard flag"       "$_out" "flag=[1]"
rm -rf "$_d"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
