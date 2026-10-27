#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Debian 13 ships hipconfig 5.7.x beside a 6.1.x runtime. Detection reads EVERY source and
# takes the highest; a genuine 5.x host still falls back to CPU.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
# Redirect the absolute ROCm/NVIDIA prefixes into empty dirs so the suite is hermetic.
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
    sed -n '/^_rocm_torch_explicitly_requested()/,/^}/p' "$INSTALL_SH"
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
    sed -n '/^_ROCM_BNB_GENERIC_FLOOR_TAG=/p' "$INSTALL_SH"
    sed -n '/^_rocm_bnb_compatible_generic_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_radeon_host_ver_not_older()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^get_radeon_wheel_url()/,/^}/p' "$INSTALL_SH"
} | sed -e "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
      -e "s|/proc/driver/nvidia|$_FAKE_PROC_NV_DIR|g" \
      -e "s|/opt/rocm|$_FAKE_ROCM_DIR|g" \
  > "$_FUNC_FILE"

# A renamed helper would otherwise fail every ROCm assertion as a plain "cpu".
for _fn in _rocm_tag_from_amd_smi _rocm_tag_from_version_file _rocm_tag_from_hipconfig \
           _rocm_tag_from_dpkg _rocm_tag_from_rpm _highest_rocm_tag \
           _detect_rocm_version_tag _rocm_bnb_compatible_generic_tag get_torch_index_url get_radeon_wheel_url \
           _radeon_host_ver_not_older; do
    if ! grep -q "^$_fn()" "$_FUNC_FILE"; then
        echo "  FAIL: install.sh no longer defines $_fn() at column 0"
        exit 1
    fi
done

# Minimal tool set so no unrelated host package answers. `timeout` and `sleep` must stay for
# the wedged-rpm case: _run_bounded looks up `timeout` on PATH.
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
        *"$_needle"*)
            echo "  PASS: $_label"
            PASS=$((PASS + 1))
            ;;
        *)
            echo "  FAIL: $_label (no '$_needle' in: $_haystack)"
            FAIL=$((FAIL + 1))
            ;;
    esac
}

_MOCK_DIR=$(mktemp -d)

reset_sources() {
    rm -rf "$_MOCK_DIR" "$_FAKE_ROCM_DIR/.info"
    _MOCK_DIR=$(mktemp -d)
    # Without a gfx name the ROCm branch bails before any version source. gfx1100 = RX 7900 XTX.
    cat > "$_MOCK_DIR/rocminfo" <<'MOCK'
#!/bin/sh
cat <<'ROCMINFO'
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
ROCMINFO
MOCK
    chmod +x "$_MOCK_DIR/rocminfo"
}

# $1 = version string printed by "hipconfig --version" (e.g. 5.7.31921-0)
add_hipconfig() {
    cat > "$_MOCK_DIR/hipconfig" <<MOCK
#!/bin/sh
echo "$1"
MOCK
    chmod +x "$_MOCK_DIR/hipconfig"
}

# $1 = contents of /opt/rocm/.info/version (e.g. 6.1.2-98)
add_version_file() {
    mkdir -p "$_FAKE_ROCM_DIR/.info"
    printf '%s\n' "$1" > "$_FAKE_ROCM_DIR/.info/version"
}

# $1 = ROCm version reported by "amd-smi version"
add_amd_smi() {
    cat > "$_MOCK_DIR/amd-smi" <<MOCK
#!/bin/sh
case "\$1" in
    list) printf 'GPU: 0\\n  BDF: 0000:03:00.0\\n  NAME: gfx1100\\n' ;;
    *) echo "AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | ROCm version: $1" ;;
esac
MOCK
    chmod +x "$_MOCK_DIR/amd-smi"
}

# Reproduces the WHOLE amd-smi line so a parser running past the field separator is caught.
add_amd_smi_line() {
    cat > "$_MOCK_DIR/amd-smi" <<MOCK
#!/bin/sh
case "\$1" in
    list) printf 'GPU: 0\\n  BDF: 0000:03:00.0\\n  NAME: gfx1100\\n' ;;
    *) echo "AMDSMI Tool: 24.7.1+b446d6c-dirty | AMDSMI Library version: 24.7.2.0 | ROCm version: $1 | amdgpu version: 6.10.10 | hsmp version: 2.2" ;;
esac
MOCK
    chmod +x "$_MOCK_DIR/amd-smi"
}

add_dpkg_packages() {
    printf '%s\n' "$@" > "$_MOCK_DIR/.dpkg-entries"
    cat > "$_MOCK_DIR/dpkg-query" <<'MOCK'
#!/bin/sh
_d=${0%/*}
_entries=$(cat "$_d/.dpkg-entries")
_fmt=''
_requested=''
_has_rocm_core=0
_has_hsa_runtime=0
while [ $# -gt 0 ]; do
    case "$1" in
        -f=*)           _fmt=${1#-f=} ;;
        --showformat=*) _fmt=${1#--showformat=} ;;
        -f|--showformat) shift; _fmt=$1 ;;
        -*)             : ;;
        rocm-core)
            _requested="$_requested $1"
            _has_rocm_core=1
            ;;
        libhsa-runtime64-1)
            _requested="$_requested $1"
            _has_hsa_runtime=1
            ;;
    esac
    shift
done
[ "$_has_rocm_core" -eq 1 ] && [ "$_has_hsa_runtime" -eq 1 ] || exit 1
[ -n "$_fmt" ] || _fmt='${Package}\t${Version}\n'
# Unknown fields render empty, which is what real dpkg-query does.
_emit() {
    _package=$1
    _status=$2
    _ver=$3
    # Status is "<want> <error-flag> <status>": removed but not purged reads
    # "deinstall ok config-files".
    case "$_status" in installed) _want=install ;; *) _want=deinstall ;; esac
    _out=$(printf '%s' "$_fmt" | sed \
        -e "s|\${Package}|$_package|g" \
        -e "s|\${Status}|$_want ok $_status|g" \
        -e "s|\${db:Status-Status}|$_status|g" \
        -e "s|\${db:Status-Want}|$_want|g" \
        -e "s|\${db:Status-Eflag}|ok|g" \
        -e "s|\${Version}|$_ver|g" \
        -e "s|\${[^}]*}||g")
    printf "$_out"
}
_missing=0
for _wanted in $_requested; do
    _found=0
    while IFS='|' read -r _package _status _ver; do
        [ "$_package" = "$_wanted" ] && _found=1
    done <<EOF
$_entries
EOF
    [ "$_found" -eq 1 ] || _missing=1
done
while IFS='|' read -r _package _status _ver; do
    case " $_requested " in
        *" $_package "*) _emit "$_package" "$_status" "$_ver" ;;
    esac
done <<EOF
$_entries
EOF
exit "$_missing"
MOCK
    chmod +x "$_MOCK_DIR/dpkg-query"
}

add_dpkg_rocm_core() {
    add_dpkg_packages "rocm-core|${2:-installed}|$1"
}

add_dpkg_hsa_runtime() {
    add_dpkg_packages "libhsa-runtime64-1|${2:-installed}|$1"
}

# $1 = rocm-core version as rpm reports it
add_rpm_rocm_core() {
    cat > "$_MOCK_DIR/rpm" <<MOCK
#!/bin/sh
for _a in "\$@"; do
    case "\$_a" in rocm-core) echo "$1"; exit 0 ;; esac
done
exit 1
MOCK
    chmod +x "$_MOCK_DIR/rpm"
}

# $1 = rocm-core, $2 = rocm-runtime. A partial upgrade can split them; neither outranks the other.
add_rpm_split_components() {
    cat > "$_MOCK_DIR/rpm" <<MOCK
#!/bin/sh
for _a in "\$@"; do
    case "\$_a" in
        rocm-core)    echo "$1" ;;
        rocm-runtime) echo "$2" ;;
    esac
done
exit 0
MOCK
    chmod +x "$_MOCK_DIR/rpm"
}

add_wedged_rpm() {
    # Stands in for `rpm -q` wedged on the rpmdb (stale BerkeleyDB locks, rpm 6.0.x vs dnf).
    # Sleeps rather than wedging so the suite stays killable.
    cat > "$_MOCK_DIR/rpm" <<MOCK
#!/bin/sh
sleep 30
MOCK
    chmod +x "$_MOCK_DIR/rpm"
}

# Outer bound: an unbounded probe fails in $1 seconds instead of hanging the suite.
run_index_outer_bounded() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" timeout "$1" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
}

run_index() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
}

run_warnings() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>&1 >/dev/null | tr '\n' ' '
}

run_radeon_url() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        'uname() { echo Linux; }; . "$1"; get_radeon_wheel_url "$2"' \
        _ "$_FUNC_FILE" "$1" 2>/dev/null
}

# Under set -e: with every source missing, detection must return empty AND succeed.
run_status_under_set_e() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "set -e
         unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url >/dev/null 2>&1; echo \$?" \
        2>/dev/null
}

_BASE="https://download.pytorch.org/whl"

echo "=== test_rocm_version_source_disagreement ==="

reset_sources
add_hipconfig "5.7.31921-0"
add_dpkg_hsa_runtime "1:6.1.2-1"
assert_eq "Debian 13 hipconfig 5.7 + HSA runtime 6.1 -> automatic rocm6.4 floor" "$_BASE/rocm6.4" "$(run_index)"
_warn=$(run_warnings)
case "$_warn" in
    *"require ROCm 6.0+"*) assert_eq "the same host emits no 6.0+ gate warning" "" "$_warn" ;;
    *) assert_eq "the same host emits no 6.0+ gate warning" "ok" "ok" ;;
esac
# The breadcrumb that makes a wrong-HIGH reading diagnosable from an install log.
assert_contains "disagreeing sources are named" "sources disagree (rocm5.7 rocm6.1)" "$_warn"
assert_contains "the winning reading is named" "using the highest, rocm6.1" "$_warn"
assert_eq "Radeon URL uses resolved Debian rocm6.1, not hipconfig 5.7" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/" "$(run_radeon_url rocm6.1)"

# rocm-core wins even when HSA reads higher and is emitted first.
reset_sources
add_dpkg_packages \
    "libhsa-runtime64-1|installed|6.4.3+dfsg-4" \
    "rocm-core|installed|1:6.1.2-2"
assert_eq "installed rocm-core outranks a HIGHER distro HSA reading -> automatic rocm6.4 floor" \
    "$_BASE/rocm6.4" "$(run_index)"
assert_eq "and the outranked HSA reading is not named as a disagreement" "" "$(run_warnings)"

reset_sources
add_dpkg_packages \
    "libhsa-runtime64-1|installed|5.7.1-2build1" \
    "rocm-core|installed|7.2.1.70201-81~24.04"
assert_eq "Ubuntu + AMD repo resolves rocm7.2" "$_BASE/rocm7.2" "$(run_index)"
assert_eq "Ubuntu + AMD repo warns about nothing" "" "$(run_warnings)"

reset_sources
add_hipconfig "5.7.31921-0"
add_dpkg_hsa_runtime "1:6.1.2-1"
assert_eq "no rocm-core, so the installed HSA runtime still votes -> automatic rocm6.4 floor" \
    "$_BASE/rocm6.4" "$(run_index)"

reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "6.1.2-98"
assert_eq "hipconfig 5.7 + version file 6.1 -> automatic rocm6.4 floor" "$_BASE/rocm6.4" "$(run_index)"

reset_sources
add_hipconfig "5.7.31921-0"
add_rpm_rocm_core "6.3.0"
assert_eq "hipconfig 5.7 + rocm-core 6.3 (rpm) -> automatic rocm6.4 floor" "$_BASE/rocm6.4" "$(run_index)"

reset_sources
add_amd_smi "6.1.0"
add_dpkg_rocm_core "6.4.1-1"
assert_eq "amd-smi 6.1 + rocm-core 6.4 -> rocm6.4 (highest, not first)" "$_BASE/rocm6.4" "$(run_index)"

reset_sources
add_amd_smi "6.4.0"
add_version_file "6.4.0-1"
add_hipconfig "6.4.43482-0"
assert_eq "all sources agree on 6.4 -> rocm6.4" "$_BASE/rocm6.4" "$(run_index)"
assert_eq "agreeing sources emit no disagreement breadcrumb" "" "$(run_warnings)"

# Overshoot installs wheels the runtime cannot load. `dpkg-query -W` still reports removed
# ("deinstall ok config-files") packages, so detection must require status "installed".
reset_sources
add_hipconfig "6.1.40093-0"
add_dpkg_rocm_core "1:7.0.0-1" config-files
assert_eq "config-files rocm-core 7.0 on a 6.1 host -> automatic rocm6.4 floor, not rocm7.0" \
    "$_BASE/rocm6.4" "$(run_index)"
assert_eq "the dead dpkg entry is not even named as a disagreement" "" "$(run_warnings)"

reset_sources
add_hipconfig "6.1.40093-0"
add_dpkg_hsa_runtime "1:7.0.0-1" config-files
assert_eq "config-files HSA runtime 7.0 on a 6.1 host -> automatic rocm6.4 floor, not rocm7.0" \
    "$_BASE/rocm6.4" "$(run_index)"
assert_eq "the dead HSA entry is not named as a disagreement" "" "$(run_warnings)"

# Only the dpkg status word differs from the config-files cases above, keeping them non-vacuous.
reset_sources
add_hipconfig "5.7.31921-0"
add_dpkg_rocm_core "1:6.1.2-1" installed
assert_eq "installed rocm-core 6.1 still beats hipconfig 5.7 -> automatic rocm6.4 floor" \
    "$_BASE/rocm6.4" "$(run_index)"

for _dead in config-files half-installed unpacked half-configured; do
    reset_sources
    add_hipconfig "6.1.40093-0"
    add_dpkg_rocm_core "1:7.2.0-1" "$_dead"
    assert_eq "dpkg state '$_dead' at 7.2 does not select wheels -> automatic rocm6.4 floor" \
        "$_BASE/rocm6.4" "$(run_index)"
done

# For the other four sources a HIGH reading is taken as truth, capped by tag normalisation.
for _pos in amd-smi version-file hipconfig rpm; do
    reset_sources
    add_hipconfig "6.1.40093-0"
    case "$_pos" in
        amd-smi)      add_amd_smi "9.9.0" ;;
        version-file) add_version_file "9.9.0-1" ;;
        hipconfig)    add_hipconfig "9.9.31921-0" ;;
        rpm)          add_rpm_rocm_core "9.9.0" ;;
    esac
    assert_eq "a 9.9 reading from $_pos is capped to the newest published leaf" \
        "$_BASE/rocm7.2" "$(run_index)"
    if [ "$_pos" != "hipconfig" ]; then
        assert_contains "a 9.9 reading from $_pos is recorded in the log" \
            "sources disagree (rocm6.1 rocm9.9) -- using the highest, rocm9.9" "$(run_warnings)"
    fi
done

# An INSTALLED high dpkg reading is the runtime, so the cap applies, not a rejection.
reset_sources
add_hipconfig "6.1.40093-0"
add_dpkg_rocm_core "1:9.9.0-1" installed
assert_eq "an installed 9.9 rocm-core is capped, not rejected" "$_BASE/rocm7.2" "$(run_index)"

reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "5.7.1-90"
assert_eq "genuine ROCm 5.7 everywhere -> cpu" "$_BASE/cpu" "$(run_index)"
_warn=$(run_warnings)
assert_contains "5.x host warns about the 6.0+ requirement" "require ROCm 6.0+" "$_warn"
assert_contains "5.x warning names the resolved tag" "ROCm rocm5.7 detected" "$_warn"
assert_contains "5.x warning says it took the highest reading" "HIGHEST version" "$_warn"

# Both overrides return before any GPU probing, so naming them in the warning is honest.
assert_contains "5.x warning names UNSLOTH_TORCH_INDEX_FAMILY" \
    "UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4" "$_warn"
assert_contains "5.x warning names UNSLOTH_TORCH_INDEX_URL" \
    "UNSLOTH_TORCH_INDEX_URL=" "$_warn"

reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "5.7.1-90"
_result=$(PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
    "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
     export UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4
     _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null)
assert_eq "the named override reaches this path -> rocm6.4" "$_BASE/rocm6.4" "$_result"

# Every source missing: warn, do not die under set -e. gfx1100 has its own index, so an
# unreadable version routes on the arch.
reset_sources
assert_eq "no version source at all -> cpu" "$_BASE/cpu" "$(run_index)"
assert_eq "no version source at all -> exit 0 under set -e" "0" "$(run_status_under_set_e)"
_warn=$(run_warnings)
assert_contains "no-version host still reaches an actionable warning" \
    "routing to AMD per-arch wheels" "$_warn"
assert_contains "no-version warning names the arch it routed on" "gfx1100" "$_warn"

reset_sources
add_amd_smi "N/A"
add_hipconfig "unknown"
add_version_file "not-a-version"
assert_eq "unparseable sources -> cpu" "$_BASE/cpu" "$(run_index)"
assert_eq "unparseable sources -> exit 0 under set -e" "0" "$(run_status_under_set_e)"
assert_contains "unparseable sources are treated as no version at all" \
    "routing to AMD per-arch wheels" "$(run_warnings)"

# Major 0 is garbage, not a version below every other.
reset_sources
add_hipconfig "0.0.0"
add_version_file "6.2.0-1"
assert_eq "major-0 source ignored, 6.2 wins -> automatic rocm6.4 floor" "$_BASE/rocm6.4" "$(run_index)"

# ── 12. Supported-tag normalisation and the automatic BNB floor ─────────────
# PyTorch publishes major.minor index leaves only, so patch levels normalise;
# automatic generic 6.0-6.3 hosts floor to rocm6.4, while 6.5+ clips to the
# last 6.x wheel set and 7.3+ caps to the latest known.
for _case in "6.0.2:rocm6.4" "6.1.3:rocm6.4" "6.2.4:rocm6.4" "6.3.1:rocm6.4" \
             "6.4.1:rocm6.4" "7.0.1:rocm7.0" "7.1.0:rocm7.1" "7.2.1:rocm7.2" \
             "6.5.0:rocm6.4" "6.9.0:rocm6.4" "7.3.0:rocm7.2" "8.0.0:rocm7.2"; do
    _ver="${_case%%:*}"
    _want="${_case##*:}"
    reset_sources
    add_amd_smi "$_ver"
    assert_eq "amd-smi $_ver -> $_want" "$_BASE/$_want" "$(run_index)"
done

# The amdgpu driver version follows the ROCm field on the same line; an unparseable field
# must yield nothing, or a fabricated reading outvotes a correct source.
for _case in "N/A:" "6.4.0:rocm6.4" "7.0.2:rocm7.0" ":"; do
    _field="${_case%%:*}"
    _want="${_case##*:}"
    reset_sources
    add_amd_smi_line "$_field"
    if [ -n "$_want" ]; then
        assert_eq "amd-smi full line, ROCm field '$_field' -> $_want" \
            "$_BASE/$_want" "$(run_index)"
    else
        assert_eq "amd-smi full line, ROCm field '$_field' -> no reading" \
            "$_BASE/cpu" "$(run_index)"
    fi
done

reset_sources
add_amd_smi_line "N/A"
add_version_file "6.1.3-42"
assert_eq "amd-smi N/A beside amdgpu 6.10 does not outvote a real 6.1" \
    "$_BASE/rocm6.4" "$(run_index)"

# rpm -q now always runs and can block forever on the rpmdb, so it is bounded and a
# timed-out source declines to answer.
reset_sources
add_version_file "6.4.0-1"
add_wedged_rpm
_t0=$(date +%s)
# `|| _res=""` so the outer bound firing is a FAIL, not a set -e abort.
_res=$(run_index_outer_bounded 20) || _res=""
_t1=$(date +%s)
# Empty means the outer bound fired, i.e. the rpm probe ran unbounded.
assert_eq "a wedged rpm does not stop the version file resolving the host" \
    "$_BASE/rocm6.4" "$_res"
if [ "$((_t1 - _t0))" -lt 20 ]; then
    assert_eq "the rpm probe is bounded, not left to block the installer" "ok" "ok"
else
    assert_eq "the rpm probe is bounded, not left to block the installer" \
        "under 20s" "$((_t1 - _t0))s (outer bound fired)"
fi

# A stale rocm-core beside a newer runtime must not decide the version.
reset_sources
add_rpm_split_components "5.7.1" "6.4.1"
assert_eq "a stale rocm-core beside a newer rocm-runtime resolves to the newer" \
    "$_BASE/rocm6.4" "$(run_index)"
reset_sources
add_rpm_split_components "6.4.1" "5.7.1"
assert_eq "and the ordering is by version, not by which name was queried first" \
    "$_BASE/rocm6.4" "$(run_index)"

reset_sources
add_version_file "6.5.0-1"
assert_eq "a ROCm 6.5 host clipped to the rocm6.4 leaf still gets rocm-rel-6.5.0" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.5.0/" "$(run_radeon_url rocm6.4)"

reset_sources
add_version_file "7.3.1-1"
assert_eq "a ROCm 7.3 host capped to the rocm7.2 leaf still gets rocm-rel-7.3.1" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.3.1/" "$(run_radeon_url rocm7.2)"

# The caller only falls back x.y.z -> x.y, so a leaf-derived x.y never reaches an x.y.z-only dir.
reset_sources
add_version_file "7.2.1-98"
assert_eq "a matching-family host still contributes its patch level" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/" "$(run_radeon_url rocm7.2)"

reset_sources
add_hipconfig "5.7.31921-0"
assert_eq "an older host probe never overrides the resolved leaf" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/" "$(run_radeon_url rocm6.1)"

reset_sources
assert_eq "with no readable host source the leaf is used verbatim" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/" "$(run_radeon_url rocm6.4)"
assert_eq "no host source and no leaf yields no Radeon URL" "" "$(run_radeon_url '')"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
