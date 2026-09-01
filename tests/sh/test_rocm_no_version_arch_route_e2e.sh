#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# End-to-end routing assertions for the no-readable-version AMD reroute (issue #8731). Separate from
# test_rocm_no_version_arch_route.sh because the reroute is TOP-LEVEL code: get_torch_index_url returns
# */cpu and defers, so sourcing functions never runs the decision and grepping for its variable names
# passes with the reroute deleted. The splice runs PAST it: Strix gfx115x and gfx906 still rewrite it.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

[ -r "$INSTALL_SH" ] || { echo "  FAIL: no install.sh at $INSTALL_SH"; exit 1; }

_ROOT=$(mktemp -d)
cleanup() { rm -rf "$_ROOT"; }
trap cleanup EXIT
# The spliced block runs its own mktemp -d (the ROCm tag memo) without a cleanup trap, so keep it under $_ROOT.
TMPDIR="$_ROOT/tmp"
export TMPDIR
mkdir -p "$TMPDIR"

# Host paths are redirected into $_FAKE, or a real GPU on this box gets measured and the suite passes regardless.
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

_FUNCS="$_ROOT/funcs.sh"
_BLOCK="$_ROOT/block.sh"

: > "$_ROOT/funcs.raw"
for _fn in $(grep -oE '^[A-Za-z_][A-Za-z0-9_]*\(\)[[:space:]]*\{[[:space:]]*$' "$INSTALL_SH" \
             | sed 's/().*//' | sort -u); do
    sed -n "/^$_fn()[[:space:]]*{[[:space:]]*\$/,/^}\$/p" "$INSTALL_SH" > "$_ROOT/.one"
    # A heredoc with a bare `}` truncates this extraction into a syntax error; the check below covers the loss.
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
printf 'RADEON=%s\n' "\${_amd_gpu_radeon:-}"
printf 'HSA=%s\n' "\${HSA_OVERRIDE_GFX_VERSION:-}"
EOF
    PATH="$_MOCK:$_TOOLS" "$_RUN_SHELL" "$_ROOT/run.sh" 2>"$_ROOT/stderr.txt" || \
        printf 'INDEX=<installer exited %s>\n' "$?"
}

run_index() { run_installer "${1:-x86_64}" | sed -n 's/^INDEX=//p'; }
run_gfx()   { run_installer "${1:-x86_64}" | sed -n 's/^GFX=//p'; }
run_radeon() { run_installer "${1:-x86_64}" | sed -n 's/^RADEON=//p'; }
run_hsa()   { run_installer "${1:-x86_64}" | sed -n 's/^HSA=//p'; }

_BASE="https://download.pytorch.org/whl"
_AMD="https://repo.amd.com/rocm/whl"

echo "=== test_rocm_no_version_arch_route_e2e ==="

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

fedora_no_version_host() {   # $@ = gfx arches
    reset_host
    mock_rocminfo "$@"
    mock_amdsmi "N/A" "$@"
    mock_hipconfig_offpath ""
    mock_rpm
    mock_kfd
    mock_pci_amd_display
}

fedora_no_version_host gfx1201
assert_eq "gfx1201 with no readable version routes to the per-arch index" \
    "$_AMD/gfx120X-all/" "$(run_index)"
fedora_no_version_host gfx1201
assert_eq "and the single named card is handed to setup.sh" \
    "gfx1201" "$(run_gfx)"

fedora_no_version_host gfx1100
assert_eq "gfx1100 with no readable version routes to its own family" \
    "$_AMD/gfx110X-all/" "$(run_index)"

# gfx906 (MI50) is served only through a generic rocmX.Y leaf, so there is nothing to reroute to.
fedora_no_version_host gfx906
assert_eq "gfx906 with no readable version stays on cpu" "$_BASE/cpu" "$(run_index)"

fedora_no_version_host gfx1201
mock_hipconfig_offpath "5.7.31921-0"
assert_eq "gfx1201 on unsupported ROCm 5.7 stays on cpu" "$_BASE/cpu" "$(run_index)"

fedora_no_version_host gfx1201 gfx1036
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "gfx1201 beside gfx1036 stays on cpu" "$_BASE/cpu" "$(run_index)"

# UNSLOTH_ROCM_GFX_ARCH must stay unset: setup.sh makes a visibility-aware pick that naming a card overrules.
fedora_no_version_host gfx1200 gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "a same-family pair routes to the shared family" \
    "$_AMD/gfx120X-all/" "$(run_index)"
fedora_no_version_host gfx1200 gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "a same-family pair names no single card" "" "$(run_gfx)"

# rocminfo calls every consumer card "Radeon", and the AMD per-arch mirror is
# repo.amd.com/ROCM/whl/gfx120X-all/, so branding the host off the whole URL sends the
# rerouted install down the repo.radeon.com branch: the summary reports wheels it never
# fetched and the Radeon path then warns it cannot read the ROCm version it does not need.
fedora_no_version_host gfx1201
mock_lspci "Navi 48 [Radeon RX 9070 XT]"
assert_eq "a per-arch reroute is not branded a Radeon repo install" "false" "$(run_radeon)"
fedora_no_version_host gfx1200 gfx1201
assert_eq "nor is the same-family reroute" "false" "$(run_radeon)"

# Real ROCm 7.x rocminfo for one gfx1201: the agent Name, an ISA triple carrying feature
# flags, and a SECOND ISA "gfx12-generic" (ROCm/ROCm#6110). One card must not read as two,
# and gfx12-generic must not read as an arch -- either way the sole-arch check rejects the
# host and a working 9070 XT falls back to cpu.
fedora_no_version_host gfx1201
{
    echo "ROCk module is loaded"
    echo "Agent 1"
    echo "  Name:                    AMD Ryzen 9 9950X"
    echo "  Device Type:             CPU"
    echo "Agent 2"
    echo "  Name:                    gfx1201"
    echo "  Marketing Name:          AMD Radeon RX 9070 XT"
    echo "  Device Type:             GPU"
    echo "  ISA Info:"
    echo "    ISA 1"
    echo "      Name:                    amdgcn-amd-amdhsa--gfx1201:sramecc+:xnack-"
    echo "    ISA 2"
    echo "      Name:                    amdgcn-amd-amdhsa--gfx12-generic"
} > "$_MOCK/.rocminfo.out"
assert_eq "a ROCm 7.x gfx1201 with two ISAs is still one card" \
    "$_AMD/gfx120X-all/" "$(run_index)"
assert_eq "and gfx12-generic does not count as a second arch" "gfx1201" "$(run_gfx)"

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

fedora_no_version_host gfx1201
assert_eq "the same host on aarch64 does not reroute" "$_BASE/cpu" "$(run_index aarch64)"

# install.sh is #!/bin/sh but CI runs these files with bash, so a bashism would surface only on a dash host.
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

# HSA_OVERRIDE_GFX_VERSION=11.0.0 is the standard Strix Halo workaround, so ROCr reports
# the spoofed gfx1100 on a physical gfx1151. The reroute derives BOTH the wheel family and
# the exported arch from that probe, so without the correction this host takes gfx110X-all
# wheels and hands setup.sh a gfx1100 to build llama.cpp for.
mock_strix_cpuinfo() { printf 'model name\t: AMD Ryzen AI Max+ 395 w/ Radeon 8060S\n' > "$_F_CPUINFO"; }
mock_kfd_gfx() {   # $1 = gfx_target_version, which the override cannot reach
    : > "$_F_KFD"
    mkdir -p "$_F_SYSKFD/kfd/topology/nodes/1"
    printf 'cpu_cores_count 0\nsimd_count 128\nvendor_id 4098\ndevice_id 29824\ngfx_target_version %s\n' \
        "$1" > "$_F_SYSKFD/kfd/topology/nodes/1/properties"
}

fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_gfx 110501
assert_eq "an HSA-spoofed Strix Halo routes to the arch it really is" \
    "$_AMD/gfx1151/" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_index)"
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_gfx 110501
assert_eq "and hands setup.sh the physical arch, not the spoof" \
    "gfx1151" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_gfx)"

# Controls. Without the override the same probe is the truth, and the correction must not
# fire on a real gfx1100 that merely sits in a Ryzen AI Max chassis.
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_gfx 110501
assert_eq "with no override the probed arch is taken at face value" \
    "$_AMD/gfx110X-all/" "$(run_index)"
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_gfx 110000
assert_eq "a real gfx1100 corroborated by the kernel keeps its own family" \
    "$_AMD/gfx110X-all/" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_index)"


# gfx1033 (Van Gogh) shares gfx103X-all with gfx1030-gfx1036, so a mixed 103X host is the
# one shape where _amd_agreed_index_family AGREES while _amd_sole_index_arch declines --
# and the shared-family arm then rewrote the cpu index the miscomputing gate had just
# chosen back to ROCm wheels, for a host containing the arch that must never receive them.
fedora_no_version_host gfx1033 gfx1030
assert_eq "a mixed gfx1033 + gfx1030 host does not reroute to the shared family" \
    "$_BASE/cpu" "$(run_index)"
fedora_no_version_host gfx1033 gfx1030
assert_eq "and names no arch for setup.sh to build" "" "$(run_gfx)"
# The same family without the bad arch still reroutes: this is a gfx1033 rule, not a
# "distrust shared families" rule.
fedora_no_version_host gfx1030 gfx1032
assert_eq "gfx1030 + gfx1032 still take the shared gfx103X-all index" \
    "$_AMD/gfx103X-all/" "$(run_index)"
# A lone Deck was already covered by the gate itself; assert it here too so the reroute
# cannot regrow its own path back to ROCm.
fedora_no_version_host gfx1033
assert_eq "a lone gfx1033 does not reroute" "$_BASE/cpu" "$(run_index)"

mock_kfd_two() {   # $1 $2 = gfx_target_version per KFD node
    : > "$_F_KFD"
    for _i in 1 2; do
        mkdir -p "$_F_SYSKFD/kfd/topology/nodes/$_i"
    done
    printf 'cpu_cores_count 0\nsimd_count 128\nvendor_id 4098\ngfx_target_version %s\n' \
        "$1" > "$_F_SYSKFD/kfd/topology/nodes/1/properties"
    printf 'cpu_cores_count 0\nsimd_count 128\nvendor_id 4098\ngfx_target_version %s\n' \
        "$2" > "$_F_SYSKFD/kfd/topology/nodes/2/properties"
}

# The override can collapse a mixed host into ONE reported arch, and a singleton probe then
# agrees on a family neither physical GPU can run. The spoof helper already declines here;
# an empty correction must not be read as "no spoof".
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_two 110501 120001
assert_eq "a spoof-collapsed mixed host does not reroute" \
    "$_BASE/cpu" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_index)"
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_two 110501 120001
assert_eq "and names no arch for setup.sh to build" \
    "" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_gfx)"

# Two nodes that agree are still a decline, because the shipped helper declines on any
# multi-node KFD. This asserts the conservative outcome rather than the ideal one:
# correcting here would mean changing the helper, which the Strix path shares.
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_two 110501 110501
assert_eq "two nodes the helper will not vouch for do not reroute either" \
    "$_BASE/cpu" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_index)"

# Native wheels plus a live override is the worst of both: the kernels are there and ROCr
# keeps reporting the arch they are not for. The rocm* leaf clears it; a gfx* leaf must too.
fedora_no_version_host gfx1100
mock_strix_cpuinfo; mock_kfd_gfx 110501
assert_eq "a corroborated spoof is cleared once native wheels are chosen" \
    "" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_hsa)"
fedora_no_version_host gfx1201
assert_eq "an override that corroborated nothing is left alone" \
    "11.0.0" "$(HSA_OVERRIDE_GFX_VERSION=11.0.0 run_hsa)"

echo ""
echo "  passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
