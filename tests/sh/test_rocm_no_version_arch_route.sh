#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression tests for readable-arch / unreadable-version AMD hosts (issue #8731).
#
# Bazzite (Fedora 44 atomic) with an RX 9070 XT reads gfx1201 from amd-smi but
# reports no ROCm version anywhere install.sh looked: amd-smi prints the arch and
# no version, /opt/rocm/.info/version is the AMD-installer layout, hipconfig is
# under /opt/rocm rather than on PATH, dpkg is Debian-only and Fedora ships no
# rocm-core. With no version the installer fell back to CPU torch on a card that
# works, and told the user to run pacman on an image that has no pacman.
#
# The version only picks between the generic rocmX.Y leaves. gfx1201 has its own
# repo.amd.com index, so the arch alone is enough. Pinned here: a mapped arch with
# no version defers to the per-arch reroute; a readable but UNSUPPORTED version is
# still a deliberate CPU fallback; an unmapped arch still needs a version; the two
# added version sources answer; and the no-version warning stops claiming ROCm is
# absent on a host that has it.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
_FAKE_ROCM_DIR=$(mktemp -d)
_FAKE_PROC_NV_DIR=$(mktemp -d)
{
    sed -n '/^_run_bounded()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_cvd_hides_nvidia()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_has_amd_rocm_gpu()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_has_usable_nvidia_gpu()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_ensure_rocm_probe_env()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_probe_amd_gfx_arch()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_amd_gpu_present_via_pci()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_infer_amd_gfx_arch_from_gpu_name()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_infer_linux_amd_gfx_arch()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_infer_unsupported_amd_gfx_arch_from_gpu_name()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_infer_linux_unsupported_amd_gfx_arch()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_amd_arch_index_family_for_gfx()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_trim_index_path_slashes()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_nvidia_cu126_verdict()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_cap_cuda_family_for_pre_turing()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_sdk_install_hint()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_tag_from_amd_smi()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_tag_from_version_file()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_tag_from_hipconfig()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_tag_from_dpkg()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_rocm_tag_from_rpm()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_highest_rocm_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_detect_rocm_version_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH"
} | sed -e "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
      -e "s|/proc/driver/nvidia|$_FAKE_PROC_NV_DIR|g" \
      -e "s|/opt/rocm|$_FAKE_ROCM_DIR|g" \
  > "$_FUNC_FILE"

for _fn in _rocm_sdk_install_hint _rocm_tag_from_hipconfig _rocm_tag_from_rpm \
           _detect_rocm_version_tag _amd_arch_index_family_for_gfx get_torch_index_url; do
    if ! grep -q "^$_fn()" "$_FUNC_FILE"; then
        echo "  FAIL: install.sh no longer defines $_fn() at column 0"
        exit 1
    fi
done

_TOOLS_DIR=$(mktemp -d)
for _cmd in uname grep sed head sh bash cat awk printf tr ls sort timeout sleep; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

cleanup() {
    rm -rf "$_FUNC_FILE" "$_FAKE_SMI_DIR" "$_FAKE_ROCM_DIR" "$_FAKE_PROC_NV_DIR" "$_TOOLS_DIR" "$_MOCK_DIR"
}
trap cleanup EXIT

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

assert_contains() {
    _label="$1"; _needle="$2"; _haystack="$3"
    case "$_haystack" in
        *"$_needle"*) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
        *) echo "  FAIL: $_label (no '$_needle' in: $_haystack)"; FAIL=$((FAIL + 1)) ;;
    esac
}

assert_lacks() {
    _label="$1"; _needle="$2"; _haystack="$3"
    case "$_haystack" in
        *"$_needle"*) echo "  FAIL: $_label (unexpected '$_needle' in: $_haystack)"; FAIL=$((FAIL + 1)) ;;
        *) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
    esac
}

_MOCK_DIR=$(mktemp -d)

# $1 = gfx arch amd-smi should name. No ROCm version field: that is the whole point
# of the reported host, and it is what `amd-smi static --asic` looks like there.
reset_host() {
    rm -rf "$_MOCK_DIR" "$_FAKE_ROCM_DIR/.info" "$_FAKE_ROCM_DIR/bin"
    _MOCK_DIR=$(mktemp -d)
    cat > "$_MOCK_DIR/amd-smi" <<MOCK
#!/bin/sh
case "\$1" in
    list) printf 'GPU: 0\\n  BDF: 0000:03:00.0\\n  NAME: $1\\n' ;;
    static) printf 'ASIC:\\n    TARGET_GRAPHICS_VERSION: $1\\n' ;;
    *) echo "AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0" ;;
esac
MOCK
    chmod +x "$_MOCK_DIR/amd-smi"
}

# hipconfig under \$ROCM/bin, deliberately NOT on PATH (Fedora's layout).
add_rocm_bin_hipconfig() {
    mkdir -p "$_FAKE_ROCM_DIR/bin"
    cat > "$_FAKE_ROCM_DIR/bin/hipconfig" <<MOCK
#!/bin/sh
echo "$1"
MOCK
    chmod +x "$_FAKE_ROCM_DIR/bin/hipconfig"
}

# $1 = package name rpm should know about, $2 = its version. Every other name misses.
# Renders one line PER package argument, in order, like real rpm: the version for a
# hit and "package X is not installed" on stdout for a miss. install.sh asks about
# all three names in one query (so a wedged rpm costs one timeout, not three), so a
# mock that answered only its last argument would hide which name actually matched.
add_rpm_package() {
    printf '%s\n' "$1" > "$_MOCK_DIR/.rpm-pkg"
    printf '%s\n' "$2" > "$_MOCK_DIR/.rpm-ver"
    cat > "$_MOCK_DIR/rpm" <<'MOCK'
#!/bin/sh
_d=${0%/*}
_known=$(cat "$_d/.rpm-pkg")
_ver=$(cat "$_d/.rpm-ver")
_hit=0
_skip=0
for _arg in "$@"; do
    if [ "$_skip" = 1 ]; then _skip=0; continue; fi
    case "$_arg" in
        --qf|--queryformat) _skip=1 ;;
        -*) : ;;
        *)
            if [ "$_arg" = "$_known" ]; then
                printf '%s\n' "$_ver"; _hit=1
            else
                printf 'package %s is not installed\n' "$_arg"
            fi
            ;;
    esac
done
[ "$_hit" = 1 ] || exit 1
MOCK
    chmod +x "$_MOCK_DIR/rpm"
}

run_index() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY ROCM_PATH
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
}

run_warnings() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY ROCM_PATH
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>&1 >/dev/null | tr '\n' ' '
}

run_version_tag() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset ROCM_PATH
         . '$_FUNC_FILE'; _detect_rocm_version_tag" 2>/dev/null
}

_BASE="https://download.pytorch.org/whl"

echo "=== test_rocm_no_version_arch_route ==="

# ── 1. The reported host: gfx1201, no version anywhere ──────────────────────
# The index stays */cpu here on purpose: get_torch_index_url runs in a command
# substitution and cannot set TORCH_INDEX_URL, so it defers and the reroute (parent
# shell, covered by the gate assertions below) installs the per-arch wheels.
reset_host gfx1201
assert_eq "no version + mapped arch still returns cpu from the function" \
    "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_contains "no version + mapped arch announces the per-arch route" \
    "routing to AMD per-arch wheels" "$_w"
assert_lacks "no version + mapped arch drops the 'no ROCm install' claim" \
    "no ROCm/HIP install was found" "$_w"
assert_lacks "no version + mapped arch does not suggest pacman" "pacman" "$_w"

# ── 2. A readable but unsupported version is still a deliberate CPU fallback ─
# Same card. ROCm 5.7 is a decision, not a detection miss, so it must NOT reroute.
reset_host gfx1201
add_rocm_bin_hipconfig "5.7.31921-0"
assert_eq "unsupported version keeps the cpu index" "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_contains "unsupported version says why" "require ROCm 6.0+" "$_w"
assert_lacks "unsupported version does not claim a per-arch route" \
    "routing to AMD per-arch wheels" "$_w"

# ── 3. A readable, supported version routes normally ────────────────────────
reset_host gfx1201
add_rocm_bin_hipconfig "6.4.43483-0"
assert_eq "supported version still picks the versioned leaf" \
    "$_BASE/rocm6.4" "$(run_index)"

# ── 4. An arch with no per-arch index still needs a version ─────────────────
# gfx906 (MI50) is served only through a generic rocmX.Y leaf, so with no version
# there is nothing to route to and CPU is correct.
reset_host gfx906
assert_eq "unmapped arch with no version stays cpu" "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_lacks "unmapped arch does not claim a per-arch route" \
    "routing to AMD per-arch wheels" "$_w"
assert_contains "unmapped arch names the sources it tried" "Version sources checked" "$_w"

# ── 5. hipconfig under \$ROCM/bin is found even when it is not on PATH ──────
reset_host gfx906
add_rocm_bin_hipconfig "6.2.41134-0"
assert_eq "hipconfig off PATH is read from ROCM_PATH/bin" "rocm6.2" "$(run_version_tag)"

# ── 6. Fedora's package names answer rpm ────────────────────────────────────
reset_host gfx906
add_rpm_package rocm-runtime "6.3.1"
assert_eq "rpm rocm-runtime answers where rocm-core does not" "rocm6.3" "$(run_version_tag)"
reset_host gfx906
add_rpm_package rocm-hip "7.0.2"
assert_eq "rpm rocm-hip answers too" "rocm7.0" "$(run_version_tag)"
reset_host gfx906
add_rpm_package rocm-core "6.1.0"
assert_eq "rpm rocm-core still answers" "rocm6.1" "$(run_version_tag)"

# ── 7. The warning stops claiming ROCm is absent when it is not ─────────────
# The reporter was told to install a package he already had, on a distro whose
# package manager was not the one named.
reset_host gfx906
mkdir -p "$_FAKE_ROCM_DIR/bin"
_w=$(run_warnings)
assert_contains "an existing ROCm tree is acknowledged" "so ROCm is likely installed" "$_w"
assert_lacks "an existing ROCm tree is not called missing" "Install the ROCm/HIP SDK" "$_w"

# ── 8. The reroute gate reads the no-version state ──────────────────────────
# The gate is top-level installer code, not a function, so assert on its text: the
# disjunct is what lets a readable-arch/no-version host reach the per-arch wheels.
_gate=$(grep -c '_amd_no_rocm_version_reroute' "$INSTALL_SH")
assert_eq "install.sh wires the no-version reroute state" "yes" \
    "$([ "$_gate" -ge 4 ] && echo yes || echo "no ($_gate refs)")"

echo ""
echo "  passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
