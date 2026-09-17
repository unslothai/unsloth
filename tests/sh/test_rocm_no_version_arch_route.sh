#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression tests for readable-arch / unreadable-version AMD hosts (issue #8731).
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
    echo
    sed -n '/^_amd_probe_arches()/,/^}/p' "$INSTALL_SH"
    echo
    sed -n '/^_amd_agreed_index_family()/,/^}/p' "$INSTALL_SH"
    echo
    sed -n '/^_amd_sole_index_arch()/,/^}/p' "$INSTALL_SH"
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

# $1 = gfx arch. Emitting NO ROCm version field is the fixture, not an oversight.
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

add_rocm_bin_hipconfig() {
    mkdir -p "$_FAKE_ROCM_DIR/bin"
    cat > "$_FAKE_ROCM_DIR/bin/hipconfig" <<MOCK
#!/bin/sh
echo "$1"
MOCK
    chmod +x "$_FAKE_ROCM_DIR/bin/hipconfig"
}

# install.sh queries all three package names in ONE rpm call, so the mock answers per argument: one that answered only the last would hide which name matched.
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

reset_host_mixed() {
    rm -rf "$_MOCK_DIR" "$_FAKE_ROCM_DIR/.info" "$_FAKE_ROCM_DIR/bin"
    _MOCK_DIR=$(mktemp -d)
    cat > "$_MOCK_DIR/amd-smi" <<MOCK
#!/bin/sh
case "\$1" in
    list) printf 'GPU: 0\n  BDF: 0000:03:00.0\n  NAME: $1\nGPU: 1\n  BDF: 0000:64:00.0\n  NAME: $2\n' ;;
    static) printf 'ASIC:\n    TARGET_GRAPHICS_VERSION: $1\nASIC:\n    TARGET_GRAPHICS_VERSION: $2\n' ;;
    *) echo "AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0" ;;
esac
MOCK
    chmod +x "$_MOCK_DIR/amd-smi"
}

run_sole_arch() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c         ". '$_FUNC_FILE'; _amd_sole_index_arch \"\$1\" || echo REJECTED" _ "$1" 2>/dev/null
}

run_family() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c         ". '$_FUNC_FILE'; _amd_agreed_index_family \"\$1\" || echo REJECTED" _ "$1" 2>/dev/null
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

# cpu is expected: get_torch_index_url runs in a command substitution, so it cannot set TORCH_INDEX_URL and defers to the parent-shell reroute.
reset_host gfx1201
assert_eq "no version + mapped arch still returns cpu from the function" \
    "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_contains "no version + mapped arch announces the per-arch route" \
    "routing to AMD per-arch wheels" "$_w"
assert_lacks "no version + mapped arch drops the 'no ROCm install' claim" \
    "no ROCm/HIP install was found" "$_w"
assert_lacks "no version + mapped arch does not suggest pacman" "pacman" "$_w"

reset_host gfx1201
add_rocm_bin_hipconfig "5.7.31921-0"
assert_eq "unsupported version keeps the cpu index" "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_contains "unsupported version says why" "require ROCm 6.0+" "$_w"
assert_lacks "unsupported version does not claim a per-arch route" \
    "routing to AMD per-arch wheels" "$_w"

reset_host gfx1201
add_rocm_bin_hipconfig "6.4.43483-0"
assert_eq "supported version still picks the versioned leaf" \
    "$_BASE/rocm6.4" "$(run_index)"

# gfx906 has no per-arch index, only the generic rocmX.Y leaves, so without a version there is nothing to route to.
reset_host gfx906
assert_eq "unmapped arch with no version stays cpu" "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_lacks "unmapped arch does not claim a per-arch route" \
    "routing to AMD per-arch wheels" "$_w"
assert_contains "unmapped arch names the sources it tried" "Version sources checked" "$_w"

reset_host gfx906
add_rocm_bin_hipconfig "6.2.41134-0"
assert_eq "hipconfig off PATH is read from ROCM_PATH/bin" "rocm6.2" "$(run_version_tag)"

reset_host gfx906
add_rpm_package rocm-runtime "6.3.1"
assert_eq "rpm rocm-runtime answers where rocm-core does not" "rocm6.3" "$(run_version_tag)"
reset_host gfx906
add_rpm_package rocm-hip "7.0.2"
assert_eq "rpm rocm-hip answers too" "rocm7.0" "$(run_version_tag)"
reset_host gfx906
add_rpm_package rocm-core "6.1.0"
assert_eq "rpm rocm-core still answers" "rocm6.1" "$(run_version_tag)"

reset_host gfx906
mkdir -p "$_FAKE_ROCM_DIR/bin"
_w=$(run_warnings)
assert_contains "an existing ROCm tree is acknowledged" "so ROCm is likely installed" "$_w"
assert_lacks "an existing ROCm tree is not called missing" "Install the ROCm/HIP SDK" "$_w"

# Top-level installer code, so this grep proves wiring, not behaviour: it passes even if the gate always decides "no".
# The routing itself is asserted in test_rocm_no_version_arch_route_e2e.sh, which splices that block out and RUNS it.
_gate=$(grep -c '_amd_no_rocm_version_reroute' "$INSTALL_SH")
assert_eq "install.sh wires the no-version reroute state" "yes" \
    "$([ "$_gate" -ge 4 ] && echo yes || echo "no ($_gate refs)")"

# Probes enumerate in kernel order, so taking the first token would put APU gfx1151 wheels on a 9070 XT listed behind it.
assert_eq "duplicate agents agree on a family"     "gfx120X-all" "$(run_family 'gfx1201
gfx1201')"
assert_eq "two cards in one family agree on it"     "gfx120X-all" "$(run_family 'gfx1201
gfx1200')"
assert_eq "an APU beside a discrete card agrees on nothing"     "REJECTED" "$(run_family 'gfx1151
gfx1201')"
assert_eq "an unmappable agent rejects the whole host"     "REJECTED" "$(run_family 'gfx1201
gfx906')"
assert_eq "an empty probe is rejected" "REJECTED" "$(run_family '')"

# setup.sh takes this concrete arch over its own visibility-aware pick, so it may only answer when there is one card to name.
assert_eq "duplicate agents are one device and name it"     "gfx1201" "$(run_sole_arch 'gfx1201
gfx1201')"
assert_eq "two cards in one family name neither"     "REJECTED" "$(run_sole_arch 'gfx1201
gfx1200')"
assert_eq "an APU beside a discrete card names neither"     "REJECTED" "$(run_sole_arch 'gfx1151
gfx1201')"

reset_host_mixed gfx1151 gfx1201
assert_eq "mixed-arch host with no version keeps the cpu index"     "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_lacks "mixed-arch host does not claim a per-arch route"     "routing to AMD per-arch wheels" "$_w"

reset_host_mixed gfx1201 gfx1200
assert_eq "same-family pair still defers to the reroute" "$_BASE/cpu" "$(run_index)"
_w=$(run_warnings)
assert_contains "same-family pair still announces the per-arch route"     "routing to AMD per-arch wheels" "$_w"
assert_contains "same-family pair names the family, not a card" "gfx120X-all" "$_w"

assert_eq "the export is guarded on a single named card" "yes"     "$(grep -q 'export UNSLOTH_ROCM_GFX_ARCH="\$_linux_inferred_gfx"' "$INSTALL_SH" &&        grep -c 'if \[ -n "\$_linux_inferred_gfx" \]; then' "$INSTALL_SH" >/dev/null &&        echo yes || echo no)"
assert_eq "the torch constraint is chosen off the family" "yes"     "$(grep -q 'gfx120X-all|gfx1151|gfx1150|gfx1152)' "$INSTALL_SH" && echo yes || echo no)"

# HIP reports gfx1201:sramecc+:xnack-, which no arm of the index case table matches; get_torch_index_url strips the suffix
# and then promises per-arch wheels, so anything downstream that does not strip it strands the host on CPU.
assert_eq "a gcnArchName suffix normalises to the bare arch" "gfx1201"     "$(run_sole_arch 'gfx1201:sramecc+:xnack-')"
assert_eq "the suffixed arch would not map on its own" "REJECTED"     "$(PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c ". '$_FUNC_FILE'; _amd_arch_index_family_for_gfx 'gfx1201:sramecc+:xnack-' || echo REJECTED" 2>/dev/null)"
assert_eq "the override reroute normalises before the case table" "yes"     "$(grep -q '_linux_inferred_gfx=\$(_amd_sole_index_arch "\$_linux_inferred_gfx")' "$INSTALL_SH" && echo yes || echo no)"

# The EXIT trap rm -rf's the memo path, so one inherited from the environment and never overwritten is an rm -rf on a caller-chosen path.
_reset_block=$(sed -n '/^# Clear inherited cleanup targets before installing traps\./,/^trap _on_install_exit EXIT/p' "$INSTALL_SH")
assert_contains "the memo dir is cleared before the traps" '_ROCM_TAG_MEMO_DIR=""' "$_reset_block"
assert_contains "the memo path is cleared before the traps" '_ROCM_TAG_MEMO=""' "$_reset_block"

reset_host gfx1201
mkdir -p "$_FAKE_ROCM_DIR/bin"
_COUNT_FILE="$_MOCK_DIR/hipconfig-calls"
cat > "$_FAKE_ROCM_DIR/bin/hipconfig" <<MOCK
#!/bin/sh
echo x >> "$_COUNT_FILE"
echo "6.4.43483-0"
MOCK
chmod +x "$_FAKE_ROCM_DIR/bin/hipconfig"
_memo_out=$(PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c     "unset ROCM_PATH
     _ROCM_TAG_MEMO='$_MOCK_DIR/memo'
     . '$_FUNC_FILE'
     _detect_rocm_version_tag
     _detect_rocm_version_tag" 2>/dev/null | tr '
' ' ')
assert_eq "both calls report the same tag" "rocm6.4 rocm6.4 " "$_memo_out"
assert_eq "the second call did not re-probe" "1"     "$(wc -l < "$_COUNT_FILE" | tr -d ' ')"

rm -f "$_MOCK_DIR/memo" "$_COUNT_FILE"
_nomemo_out=$(PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c     "unset ROCM_PATH _ROCM_TAG_MEMO
     . '$_FUNC_FILE'
     _detect_rocm_version_tag
     _detect_rocm_version_tag" 2>/dev/null | tr '
' ' ')
assert_eq "no memo path means no caching" "rocm6.4 rocm6.4 " "$_nomemo_out"
assert_eq "and both calls really probed" "2"     "$(wc -l < "$_COUNT_FILE" | tr -d ' ')"

echo ""
echo "  passed: $PASS, failed: $FAIL"
[ "$FAIL" -eq 0 ]
