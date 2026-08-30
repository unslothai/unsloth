#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# End-to-end routing assertions for the no-readable-version AMD reroute (issue #8731).
#
# Why this file exists, next to test_rocm_no_version_arch_route.sh: the reroute is
# TOP-LEVEL code in install.sh, not a function. get_torch_index_url runs inside a
# command substitution and cannot assign TORCH_INDEX_URL, so it deliberately returns
# */cpu and defers; the reroute that follows is what actually rewrites the index.
# A harness that sources functions with `sed -n '/^name()/,/^}/p'` therefore never
# executes the decision, and an assertion that greps install.sh for the reroute's
# own variable names passes just as happily when the reroute is deleted.
#
# This file splices the top-level block out by stable text and RUNS it, so every
# assertion below is the final TORCH_INDEX_URL an installer on that host would use.
# Deleting the reroute, or making _amd_agreed_index_family answer "no family",
# collapses the AMD rows back to */cpu and fails this suite.
#
# The splice deliberately runs PAST the reroute to the close of the whole index
# decision region: two further top-level blocks (the Strix gfx115x reroute and the
# gfx906 pin) can still rewrite TORCH_INDEX_URL, so stopping at the reroute would
# assert an index that is not the one the installer ends up with.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

[ -r "$INSTALL_SH" ] || { echo "  FAIL: no install.sh at $INSTALL_SH"; exit 1; }

_ROOT=$(mktemp -d)
cleanup() { rm -rf "$_ROOT"; }
trap cleanup EXIT
# The spliced block calls mktemp -d itself (the ROCm tag memo) and its cleanup trap
# is not part of the splice, so keep those under $_ROOT rather than in the real TMPDIR.
TMPDIR="$_ROOT/tmp"
export TMPDIR
mkdir -p "$TMPDIR"

# Every absolute host path the spliced code probes is rewritten into $_FAKE. This
# box may well have a real GPU of either vendor; without the rewrite a CI runner
# with an NVIDIA driver would send every AMD scenario down the CUDA path and the
# suite would pass while measuring nothing. The pre-flight below proves it took.
_FAKE="$_ROOT/fake"
_F_ROCM="$_FAKE/opt-rocm"
_F_PROCNV="$_FAKE/proc-driver-nvidia"
_F_NVSMI="$_FAKE/usr-bin-nvidia-smi"
_F_KFD="$_FAKE/dev-kfd"
_F_SYSKFD="$_FAKE/sys-class-kfd"
_F_PCI="$_FAKE/sys-bus-pci-devices"
_F_CPUINFO="$_FAKE/proc-cpuinfo"
_F_PROCVER="$_FAKE/proc-version"
_F_DXG="$_FAKE/dev-dxg"

_MOCK="$_ROOT/mock"
_TOOLS="$_ROOT/tools"
mkdir -p "$_MOCK" "$_TOOLS"
for _cmd in uname grep sed awk head tail tr ls sort cat cut wc mktemp rm mkdir chmod \
            dirname basename expr readlink id date find stat env timeout sleep sh bash \
            printf echo test true false touch cp ln; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS/$_cmd"
done

_redirect() {
    sed -e "s|/usr/bin/nvidia-smi|$_F_NVSMI|g" \
        -e "s|/proc/driver/nvidia|$_F_PROCNV|g" \
        -e "s|/opt/rocm|$_F_ROCM|g" \
        -e "s|/sys/class/kfd|$_F_SYSKFD|g" \
        -e "s|/sys/bus/pci/devices|$_F_PCI|g" \
        -e "s|/dev/kfd|$_F_KFD|g" \
        -e "s|/dev/dxg|$_F_DXG|g" \
        -e "s|/proc/cpuinfo|$_F_CPUINFO|g" \
        -e "s|/proc/version|$_F_PROCVER|g"
}

# ---------------------------------------------------------------------------
# Splice: every individually parseable top-level function, plus the block.
# ---------------------------------------------------------------------------
_FUNCS="$_ROOT/funcs.sh"
_BLOCK="$_ROOT/block.sh"

: > "$_ROOT/funcs.raw"
for _fn in $(grep -oE '^[A-Za-z_][A-Za-z0-9_]*\(\)[[:space:]]*\{[[:space:]]*$' "$INSTALL_SH" \
             | sed 's/().*//' | sort -u); do
    sed -n "/^$_fn()[[:space:]]*{[[:space:]]*\$/,/^}\$/p" "$INSTALL_SH" > "$_ROOT/.one"
    # A function whose body contains a heredoc with a bare `}` line gets truncated by
    # the range extraction and would be injected as a syntax error. Drop those rather
    # than poison the whole file; the required-function check below is what proves
    # none of them were on the reroute path.
    if bash -n "$_ROOT/.one" 2>/dev/null; then
        cat "$_ROOT/.one" >> "$_ROOT/funcs.raw"
        echo "" >> "$_ROOT/funcs.raw"
    fi
done
rm -f "$_ROOT/.one"
_redirect < "$_ROOT/funcs.raw" > "$_FUNCS"

awk '
    !started && (/^_ROCM_TAG_MEMO_DIR=\$\(mktemp/ || /^TORCH_INDEX_URL=\$\(get_torch_index_url\)$/) { started = 1 }
    started && /^fi  # _torch_index_pinned guard/ { print; exit }
    started { print }
' "$INSTALL_SH" > "$_ROOT/block.raw"
_redirect < "$_ROOT/block.raw" > "$_BLOCK"

# --- integrity: a silently empty or truncated splice must not look like a pass ---
_fatal() { echo "  FAIL: $1"; echo ""; echo "  passed: $PASS, failed: 1"; exit 1; }

for _fn in get_torch_index_url _has_amd_rocm_gpu _has_usable_nvidia_gpu \
           _probe_amd_gfx_arch _detect_rocm_version_tag _amd_arch_index_family_for_gfx \
           _amd_agreed_index_family _amd_sole_index_arch _infer_linux_amd_gfx_arch \
           _infer_amd_gfx_arch_from_gpu_name _amd_gpu_present_via_pci _run_bounded \
           _rocm_tag_from_amd_smi _rocm_tag_from_hipconfig _rocm_tag_from_rpm \
           _rocm_tag_from_dpkg _rocm_tag_from_version_file _highest_rocm_tag \
           _trim_index_path_slashes _cvd_hides_nvidia _ensure_rocm_probe_env; do
    grep -q "^$_fn()" "$_FUNCS" || \
        _fatal "install.sh no longer defines $_fn() at column 0 (splice would silently answer cpu)"
done
tail -1 "$_ROOT/block.raw" | grep -q '^fi  # _torch_index_pinned guard' || \
    _fatal "the top-level block splice never reached its end marker in install.sh"
grep -q 'TORCH_INDEX_URL=\$(get_torch_index_url)' "$_BLOCK" || \
    _fatal "the block splice lost the get_torch_index_url call"
grep -q 'UNSLOTH_AMD_ROCM_MIRROR' "$_BLOCK" || \
    _fatal "the block splice lost the AMD per-arch mirror"
bash -n "$_FUNCS" || _fatal "spliced functions do not parse"
bash -n "$_BLOCK" || _fatal "spliced block does not parse"
sh -n "$_FUNCS" || _fatal "spliced functions are not POSIX-parseable"
sh -n "$_BLOCK" || _fatal "spliced block is not POSIX-parseable"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Host fakery
# ---------------------------------------------------------------------------
reset_host() {
    rm -rf "$_FAKE" "$_MOCK"
    mkdir -p "$_FAKE" "$_MOCK" "$_F_PROCNV" "$_F_PCI" "$_F_ROCM"
    : > "$_F_CPUINFO"
    printf 'Linux version 7.1.5 (builder) #1 SMP\n' > "$_F_PROCVER"
}

# rocminfo names each agent twice, as the real tool does.
mock_rocminfo() {
    {
        echo "ROCk module is loaded"
        echo "Agent 1"
        echo "  Name:                    AMD EPYC CPU"
        echo "  Device Type:             CPU"
        _i=2
        for _a in "$@"; do
            echo "Agent $_i"
            echo "  Name:                    $_a"
            echo "  Marketing Name:          AMD Radeon Graphics"
            echo "  Device Type:             GPU"
            echo "  Name:                    $_a"
            _i=$((_i + 1))
        done
    } > "$_MOCK/.rocminfo.out"
    cat > "$_MOCK/rocminfo" <<'MOCK'
#!/bin/sh
cat "$(dirname "$0")/.rocminfo.out"
MOCK
    chmod +x "$_MOCK/rocminfo"
}

# $1 = the "ROCm version" field ("N/A" on the reported host), rest = gfx arches.
# Shaped like the real tool: `amd-smi list` shows BDF/UUID only, the arch comes from
# `amd-smi static --asic` TARGET_GRAPHICS_VERSION.
mock_amdsmi() {
    _ver="$1"; shift
    {
        _i=0
        for _a in "$@"; do
            echo "GPU: $_i"
            echo "    BDF: 0000:0$_i:00.0"
            echo "    UUID: 00000000-0000-0000-0000-00000000000$_i"
            _i=$((_i + 1))
        done
    } > "$_MOCK/.amdsmi.list"
    {
        for _a in "$@"; do
            echo "ASIC:"
            echo "    MARKET_NAME: AMD Radeon Graphics"
            echo "    TARGET_GRAPHICS_VERSION: $_a"
        done
    } > "$_MOCK/.amdsmi.asic"
    printf 'AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | ROCm version: %s | amdgpu version: 6.10.10\n' \
        "$_ver" > "$_MOCK/.amdsmi.version"
    cat > "$_MOCK/amd-smi" <<'MOCK'
#!/bin/sh
_d=$(dirname "$0")
case "${1:-}" in
    list) cat "$_d/.amdsmi.list" ;;
    static) cat "$_d/.amdsmi.asic" ;;
    *) cat "$_d/.amdsmi.version" ;;
esac
MOCK
    chmod +x "$_MOCK/amd-smi"
}

# hipconfig under $ROCM/bin, deliberately NOT on PATH (Fedora's layout).
# $1 = what it prints; "" = present but answers nothing, which is the reported host.
mock_hipconfig_offpath() {
    mkdir -p "$_F_ROCM/bin"
    cat > "$_F_ROCM/bin/hipconfig" <<MOCK
#!/bin/sh
printf '%s' '$1'
[ -n '$1' ] && echo
exit 0
MOCK
    chmod +x "$_F_ROCM/bin/hipconfig"
}

# No args = rpm present but knows no ROCm package (Fedora ships no rocm-core).
mock_rpm() {
    if [ $# -eq 2 ]; then
        printf '%s\n' "$1" > "$_MOCK/.rpm-pkg"; printf '%s\n' "$2" > "$_MOCK/.rpm-ver"
    else
        : > "$_MOCK/.rpm-pkg"; : > "$_MOCK/.rpm-ver"
    fi
    cat > "$_MOCK/rpm" <<'MOCK'
#!/bin/sh
_d=$(dirname "$0")
_known=$(cat "$_d/.rpm-pkg"); _ver=$(cat "$_d/.rpm-ver")
_hit=0; _skip=0
for _arg in "$@"; do
    if [ "$_skip" = 1 ]; then _skip=0; continue; fi
    case "$_arg" in
        --qf|--queryformat) _skip=1 ;;
        -*) : ;;
        *) if [ -n "$_known" ] && [ "$_arg" = "$_known" ]; then printf '%s\n' "$_ver"; _hit=1
           else printf 'package %s is not installed\n' "$_arg"; fi ;;
    esac
done
[ "$_hit" = 1 ] || exit 1
MOCK
    chmod +x "$_MOCK/rpm"
}

mock_dpkg() {   # $1 = version of an installed rocm-core
    printf '%s\n' "${1:-}" > "$_MOCK/.dpkg-ver"
    cat > "$_MOCK/dpkg-query" <<'MOCK'
#!/bin/sh
_v=$(cat "$(dirname "$0")/.dpkg-ver")
[ -n "$_v" ] || exit 1
printf 'install ok installed %s\n' "$_v"
MOCK
    chmod +x "$_MOCK/dpkg-query"
}

mock_nvidia_smi() {   # $1 = CUDA version, $2 = compute capability
    printf '%s\n' "$1" > "$_MOCK/.cuda-ver"
    printf '%s\n' "$2" > "$_MOCK/.compute-cap"
    cat > "$_MOCK/nvidia-smi" <<'MOCK'
#!/bin/sh
_d=$(dirname "$0")
for _a in "$@"; do
    case "$_a" in
        -L) echo "GPU 0: NVIDIA B200 (UUID: GPU-deadbeef)"; exit 0 ;;
        --query-gpu=compute_cap) cat "$_d/.compute-cap"; exit 0 ;;
    esac
done
printf 'NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: %s\n' "$(cat "$_d/.cuda-ver")"
MOCK
    chmod +x "$_MOCK/nvidia-smi"
    cp "$_MOCK/nvidia-smi" "$_F_NVSMI"; chmod +x "$_F_NVSMI"
    mkdir -p "$_F_PROCNV/gpus/0000:01:00.0"
}

mock_kfd() {   # AMD GPU visible through the KFD topology
    : > "$_F_KFD"
    mkdir -p "$_F_SYSKFD/kfd/topology/nodes/1"
    printf 'cpu_cores_count 0\nsimd_count 128\nvendor_id 4098\ndevice_id 29824\n' \
        > "$_F_SYSKFD/kfd/topology/nodes/1/properties"
}

mock_pci_amd_display() {
    mkdir -p "$_F_PCI/0000:03:00.0"
    echo 0x1002 > "$_F_PCI/0000:03:00.0/vendor"
    echo 0x030000 > "$_F_PCI/0000:03:00.0/class"
}

mock_lspci() {   # $1 = the display-controller name lspci reports
    printf '03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] %s [1002:7550]\n' \
        "$1" > "$_MOCK/.lspci.out"
    cat > "$_MOCK/lspci" <<'MOCK'
#!/bin/sh
cat "$(dirname "$0")/.lspci.out"
MOCK
    chmod +x "$_MOCK/lspci"
}

# ---------------------------------------------------------------------------
# Runner: run the spliced functions AND the spliced top-level block, then report
# the index the installer would actually have used.
# ---------------------------------------------------------------------------
_RUN_SHELL=bash
run_installer() {   # $1 = _ARCH
    cat > "$_ROOT/run.sh" <<EOF
set -e
unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL \\
      UNSLOTH_TORCH_INDEX_FAMILY UNSLOTH_AMD_ROCM_MIRROR UNSLOTH_PYTORCH_MIRROR \\
      ROCM_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
_ARCH=$1
_torch_index_pinned=false
SKIP_TORCH=false
TORCH_CONSTRAINT=""
TORCHVISION_CONSTRAINT=""
TORCHAUDIO_CONSTRAINT=""
. "$_FUNCS"
. "$_BLOCK"
printf 'INDEX=%s\n' "\$TORCH_INDEX_URL"
printf 'GFX=%s\n' "\${UNSLOTH_ROCM_GFX_ARCH:-}"
EOF
    PATH="$_MOCK:$_TOOLS" "$_RUN_SHELL" "$_ROOT/run.sh" 2>"$_ROOT/stderr.txt" || \
        printf 'INDEX=<installer exited %s>\n' "$?"
}

run_index() { run_installer "${1:-x86_64}" | sed -n 's/^INDEX=//p'; }
run_gfx()   { run_installer "${1:-x86_64}" | sed -n 's/^GFX=//p'; }

_BASE="https://download.pytorch.org/whl"
_AMD="https://repo.amd.com/rocm/whl"

echo "=== test_rocm_no_version_arch_route_e2e ==="

# ── 0. Pre-flight: the host redirect is load-bearing, so prove it took ──────
# A broken redirect looks exactly like a working one from the outside: every AMD
# scenario would quietly measure this machine's own GPU instead of the fake host.
# Negative and positive control, so neither "always false" nor "always true" passes.
reset_host
cat > "$_ROOT/nvcheck.sh" <<EOF
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
. "$_FUNCS"
if _has_usable_nvidia_gpu; then echo SEEN; else echo HIDDEN; fi
EOF
assert_eq "the real host's NVIDIA driver is redirected away" "HIDDEN" \
    "$(PATH="$_MOCK:$_TOOLS" bash "$_ROOT/nvcheck.sh" 2>/dev/null)"
mock_nvidia_smi "12.8" "9.0"
assert_eq "and a faked driver tree is still seen (the probe is reached)" "SEEN" \
    "$(PATH="$_MOCK:$_TOOLS" bash "$_ROOT/nvcheck.sh" 2>/dev/null)"
reset_host
cat > "$_ROOT/amdcheck.sh" <<EOF
. "$_FUNCS"
if _has_amd_rocm_gpu; then echo SEEN; else echo HIDDEN; fi
EOF
assert_eq "no AMD GPU is seen on a bare fake host" "HIDDEN" \
    "$(PATH="$_MOCK:$_TOOLS" bash "$_ROOT/amdcheck.sh" 2>/dev/null)"
mock_kfd; mock_pci_amd_display
assert_eq "and a faked KFD topology is seen" "SEEN" \
    "$(PATH="$_MOCK:$_TOOLS" bash "$_ROOT/amdcheck.sh" 2>/dev/null)"

# The reported host: Bazzite/Fedora, RX 9070 XT. rocminfo and amd-smi both read
# gfx1201, amd-smi reports no ROCm version, hipconfig lives under /opt/rocm rather
# than on PATH and answers nothing, and Fedora ships no rocm-core for rpm to find.
fedora_no_version_host() {   # $@ = gfx arches
    reset_host
    mock_rocminfo "$@"
    mock_amdsmi "N/A" "$@"
    mock_hipconfig_offpath ""
    mock_rpm
    mock_kfd
    mock_pci_amd_display
}

# ── 1. The reported host actually reaches the per-arch wheels ───────────────
# This is the assertion the suite was missing: not "install.sh mentions the reroute"
# but "a Fedora gfx1201 host with no readable version ends up at AMD's index".
fedora_no_version_host gfx1201
assert_eq "gfx1201 with no readable version routes to the per-arch index" \
    "$_AMD/gfx120X-all/" "$(run_index)"
fedora_no_version_host gfx1201
assert_eq "and the single named card is handed to setup.sh" \
    "gfx1201" "$(run_gfx)"

# ── 2. The same for an RDNA3 card, so the route is not one hardcoded family ─
fedora_no_version_host gfx1100
assert_eq "gfx1100 with no readable version routes to its own family" \
    "$_AMD/gfx110X-all/" "$(run_index)"

# ── 3. An arch with no per-arch index still needs a version ─────────────────
# gfx906 (MI50) is served only through a generic rocmX.Y leaf, so there is nothing
# to reroute to and CPU torch is the correct answer.
fedora_no_version_host gfx906
assert_eq "gfx906 with no readable version stays on cpu" "$_BASE/cpu" "$(run_index)"

# ── 4. A readable but unsupported version is a decision, not a detection miss ─
# Same gfx1201 card. ROCm 5.7 is below the floor, so the CPU fallback is deliberate
# and the reroute must not overturn it.
fedora_no_version_host gfx1201
mock_hipconfig_offpath "5.7.31921-0"
assert_eq "gfx1201 on unsupported ROCm 5.7 stays on cpu" "$_BASE/cpu" "$(run_index)"

# ── 5. Two AMD GPUs that want different wheels route to neither ─────────────
# An APU beside a discrete card enumerates in kernel order, so routing on whichever
# agent came first would put the wrong wheels on the box.
fedora_no_version_host gfx1201 gfx1036
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "gfx1201 beside gfx1036 stays on cpu" "$_BASE/cpu" "$(run_index)"

# ── 6. Two AMD GPUs in ONE family do route, without naming a card ───────────
# The wheels are right for both, so the index moves. UNSLOTH_ROCM_GFX_ARCH must stay
# unset: setup.sh makes a visibility-aware pick that naming one at random overrules.
fedora_no_version_host gfx1200 gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "a same-family pair routes to the shared family" \
    "$_AMD/gfx120X-all/" "$(run_index)"
fedora_no_version_host gfx1200 gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "a same-family pair names no single card" "" "$(run_gfx)"

# ── 7. The reroute must not reach past AMD hosts ────────────────────────────
# Stale ROCm packages on an NVIDIA box are common (a dual-boot leftover, a container
# image). The CUDA index has to survive them.
reset_host
mock_nvidia_smi "12.8" "9.0"
mock_hipconfig_offpath "6.4.43483-0"
mock_rpm rocm-core 6.4.1
mock_dpkg "6.4.1-1"
assert_eq "an NVIDIA host with stale ROCm packages keeps its CUDA index" \
    "$_BASE/cu128" "$(run_index)"

reset_host
mock_hipconfig_offpath "6.4.43483-0"
mock_rpm rocm-core 6.4.1
mock_dpkg "6.4.1-1"
assert_eq "a GPU-less host with stale ROCm packages stays on cpu" \
    "$_BASE/cpu" "$(run_index)"

# ── 8. ROCm wheels are x86_64-only ─────────────────────────────────────────
fedora_no_version_host gfx1201
assert_eq "the same host on aarch64 does not reroute" "$_BASE/cpu" "$(run_index aarch64)"

# ── 9. Shell parity ─────────────────────────────────────────────────────────
# install.sh is #!/bin/sh. bash is what CI runs these files with, so the block above
# proves nothing about dash, where an accidental bashism in the reroute would only
# surface on a Debian/Ubuntu host. Re-run the decision under /bin/sh as well.
_SH_REAL=$(command -v sh 2>/dev/null || true)
if [ -n "$_SH_REAL" ]; then
    _RUN_SHELL="$_SH_REAL"
    fedora_no_version_host gfx1201
    assert_eq "under /bin/sh the reported host still routes per-arch" \
        "$_AMD/gfx120X-all/" "$(run_index)"
    fedora_no_version_host gfx906
    assert_eq "under /bin/sh an unmapped arch still stays on cpu" \
        "$_BASE/cpu" "$(run_index)"
    _RUN_SHELL=bash
else
    echo "  SKIP: no /bin/sh to cross-check"
fi

echo ""
echo "  passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
