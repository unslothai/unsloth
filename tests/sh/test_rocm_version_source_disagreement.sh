#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression tests for ROCm version resolution in install.sh (issue #8402).
#
# Debian 13 (and Linux Mint on top of it) packages hipconfig at 5.7.x next to a
# 6.1.x rocminfo/HSA runtime. install.sh used to take the FIRST version source
# that answered, so it resolved rocm5.7, the "requires ROCm 6.0+" gate fired, and
# a working RX 7900 XTX (gfx1100) got CPU-only torch. Detection now reads EVERY
# source and takes the highest. Pinned here: a stale low source never shadows a
# higher one from any position; a genuine 5.x host still falls back to CPU; the
# sub-6.0 warning names the documented override; every source missing still warns
# without killing the installer (set -e); tag normalisation is unchanged.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
# The version sources, _ensure_rocm_probe_env's PATH append and the NVIDIA
# fallback all read absolute host paths, which would leak the test host's own
# ROCm or NVIDIA into every assertion. Redirect all three prefixes into empty
# temp dirs so the suite is hermetic on a ROCm, NVIDIA or bare box alike.
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
    sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH"
} | sed -e "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
      -e "s|/proc/driver/nvidia|$_FAKE_PROC_NV_DIR|g" \
      -e "s|/opt/rocm|$_FAKE_ROCM_DIR|g" \
  > "$_FUNC_FILE"

# Guard the extraction: a renamed helper would otherwise make every ROCm assertion
# below fail as a plain "cpu" with no hint why.
for _fn in _rocm_tag_from_amd_smi _rocm_tag_from_version_file _rocm_tag_from_hipconfig \
           _rocm_tag_from_dpkg _rocm_tag_from_rpm _highest_rocm_tag \
           _detect_rocm_version_tag get_torch_index_url; do
    if ! grep -q "^$_fn()" "$_FUNC_FILE"; then
        echo "  FAIL: install.sh no longer defines $_fn() at column 0"
        exit 1
    fi
done

# Minimal tool set: no amd-smi, hipconfig, dpkg-query, rpm or rocminfo unless a
# scenario supplies one, so an unrelated host package cannot answer a probe.
# `timeout` and `sleep` must stay in it for case 14: _run_bounded looks up
# `timeout` on PATH, so dropping it silently turns the bounded probe back into an
# unbounded one and that case passes for the wrong reason.
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

# ── Scenario builders ───────────────────────────────────────────────────────
# Every scenario starts from a clean mock dir and an empty fake /opt/rocm.
_MOCK_DIR=$(mktemp -d)

reset_sources() {
    rm -rf "$_MOCK_DIR" "$_FAKE_ROCM_DIR/.info"
    _MOCK_DIR=$(mktemp -d)
    # rocminfo is what makes this an AMD host at all: without a gfx name the ROCm
    # branch bails before any version source is consulted. gfx1100 = RX 7900 XTX,
    # the card in the report.
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

# $1 = the raw value of the "ROCm version:" field, e.g. "6.4.0" or "N/A". Unlike
# add_amd_smi this reproduces the WHOLE line, amdgpu driver version and all, so a
# parser that runs past the field separator is caught.
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

# $1 = rocm-core version as dpkg reports it (epoch prefixes allowed, e.g. 1:6.2.4-1)
# $2 = dpkg status word, default "installed". "config-files" is where `apt remove`
#      without `apt purge` leaves a package: still in /var/lib/dpkg/status, still
#      carrying its old version, and still reported by `dpkg-query -W`.
# The mock renders the showformat string it is given rather than a bare version, so
# these assertions test how install.sh ASKS dpkg, not just how it parses the answer.
add_dpkg_rocm_core() {
    printf '%s\n' "$1" > "$_MOCK_DIR/.dpkg-version"
    printf '%s\n' "${2:-installed}" > "$_MOCK_DIR/.dpkg-status"
    cat > "$_MOCK_DIR/dpkg-query" <<'MOCK'
#!/bin/sh
_d=${0%/*}
_ver=$(cat "$_d/.dpkg-version")
_status=$(cat "$_d/.dpkg-status")
# Status is "<want> <error-flag> <status>": removed but not purged reads "deinstall ok config-files".
case "$_status" in installed) _want=install ;; *) _want=deinstall ;; esac
_fmt=''
_found=''
while [ $# -gt 0 ]; do
    case "$1" in
        -f=*)           _fmt=${1#-f=} ;;
        --showformat=*) _fmt=${1#--showformat=} ;;
        -f|--showformat) shift; _fmt=$1 ;;
        -*)             : ;;
        rocm-core)      _found=1 ;;
    esac
    shift
done
[ -n "$_found" ] || exit 1
[ -n "$_fmt" ] || _fmt='${Package}\t${Version}\n'
# Unknown fields render empty, which is what real dpkg-query does.
_out=$(printf '%s' "$_fmt" | sed \
    -e "s|\${Package}|rocm-core|g" \
    -e "s|\${Status}|$_want ok $_status|g" \
    -e "s|\${db:Status-Status}|$_status|g" \
    -e "s|\${db:Status-Want}|$_want|g" \
    -e "s|\${db:Status-Eflag}|ok|g" \
    -e "s|\${Version}|$_ver|g" \
    -e "s|\${[^}]*}||g")
printf "$_out"
MOCK
    chmod +x "$_MOCK_DIR/dpkg-query"
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

add_wedged_rpm() {
    # Stands in for `rpm -q` wedged on the rpmdb (stale BerkeleyDB __db locks on
    # rpm < 4.16, i.e. RHEL 8 / SLES 15; the rpm 6.0.x deadlock against dnf).
    # Sleeps rather than really wedging so the suite stays hermetic and killable.
    cat > "$_MOCK_DIR/rpm" <<MOCK
#!/bin/sh
sleep 30
MOCK
    chmod +x "$_MOCK_DIR/rpm"
}

# run_index under an OUTER bound: an unbounded probe fails this in $1 seconds
# instead of hanging the suite.
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

# Same, but returns the warnings instead of the index, on one line.
run_warnings() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>&1 >/dev/null | tr '\n' ' '
}

# Same, under `set -e` like the real installer, reporting only the exit status:
# with every source missing, detection must return empty AND succeed or the
# installer dies before the actionable warning.
run_status_under_set_e() {
    PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
        "set -e
         unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
         _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url >/dev/null 2>&1; echo \$?" \
        2>/dev/null
}

_BASE="https://download.pytorch.org/whl"

echo "=== test_rocm_version_source_disagreement ==="

# ── 1. The reported host ────────────────────────────────────────────────────
# No /opt/rocm/.info/version, hipconfig from the distro's 5.7 packaging, rocm-core
# at 6.1 in dpkg. Before the fix hipconfig answered first and this returned cpu.
reset_sources
add_hipconfig "5.7.31921-0"
add_dpkg_rocm_core "1:6.1.2-1"
assert_eq "Debian 13 hipconfig 5.7 + rocm-core 6.1 -> rocm6.1" "$_BASE/rocm6.1" "$(run_index)"
_warn=$(run_warnings)
case "$_warn" in
    *"require ROCm 6.0+"*) assert_eq "the same host emits no 6.0+ gate warning" "" "$_warn" ;;
    *) assert_eq "the same host emits no 6.0+ gate warning" "ok" "ok" ;;
esac
# The breadcrumb that makes a wrong-HIGH reading diagnosable from an install log.
assert_contains "disagreeing sources are named" "sources disagree (rocm5.7 rocm6.1)" "$_warn"
assert_contains "the winning reading is named" "using the highest, rocm6.1" "$_warn"

# 2. Same shape from /opt/rocm/.info/version. This one already worked by position,
#    so it is here to pin that the rewrite did not lose it.
reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "6.1.2-98"
assert_eq "hipconfig 5.7 + version file 6.1 -> rocm6.1" "$_BASE/rocm6.1" "$(run_index)"

# 3. And from rpm, the last source in the order.
reset_sources
add_hipconfig "5.7.31921-0"
add_rpm_rocm_core "6.3.0"
assert_eq "hipconfig 5.7 + rocm-core 6.3 (rpm) -> rocm6.3" "$_BASE/rocm6.3" "$(run_index)"

# 4. Not only a floor problem: amd-smi answering first with a lower version must
#    not shadow a newer runtime either.
reset_sources
add_amd_smi "6.1.0"
add_dpkg_rocm_core "6.4.1-1"
assert_eq "amd-smi 6.1 + rocm-core 6.4 -> rocm6.4 (highest, not first)" "$_BASE/rocm6.4" "$(run_index)"

# 5. Agreement is unchanged: all sources on 6.4 still resolve 6.4.
reset_sources
add_amd_smi "6.4.0"
add_version_file "6.4.0-1"
add_hipconfig "6.4.43482-0"
assert_eq "all sources agree on 6.4 -> rocm6.4" "$_BASE/rocm6.4" "$(run_index)"
assert_eq "agreeing sources emit no disagreement breadcrumb" "" "$(run_warnings)"

# ── 5b. Overshoot: highest-wins must not let a DEAD source pick the wheels ──
# Undershoot lands on CPU wheels, which work; overshoot installs wheels the runtime
# cannot load. The one host state that can manufacture a wrong-HIGH reading is
# dpkg's: `dpkg-query -W` reports packages left in "deinstall ok config-files" by
# `apt remove` without a purge, still carrying the version they had, so a box that
# ran ROCm 7.0 and went back to 6.1 offers a 7.0 that outranks every honest source.
# Detection must require the status word "installed".
reset_sources
add_hipconfig "6.1.40093-0"
add_dpkg_rocm_core "1:7.0.0-1" config-files
assert_eq "config-files rocm-core 7.0 on a 6.1 host -> rocm6.1, not rocm7.0" \
    "$_BASE/rocm6.1" "$(run_index)"
assert_eq "the dead dpkg entry is not even named as a disagreement" "" "$(run_warnings)"

# 5c. The live entry still has to win, or the fix for the reported bug is gone.
#     This is what makes 5b non-vacuous: only the dpkg status word differs.
reset_sources
add_hipconfig "5.7.31921-0"
add_dpkg_rocm_core "1:6.1.2-1" installed
assert_eq "installed rocm-core 6.1 still beats hipconfig 5.7 -> rocm6.1" \
    "$_BASE/rocm6.1" "$(run_index)"

# 5d. The states an interrupted dpkg operation leaves behind are not "installed"
#     either, and none of them describes a runtime the GPU can use.
for _dead in config-files half-installed unpacked half-configured; do
    reset_sources
    add_hipconfig "6.1.40093-0"
    add_dpkg_rocm_core "1:7.2.0-1" "$_dead"
    assert_eq "dpkg state '$_dead' at 7.2 does not select wheels -> rocm6.1" \
        "$_BASE/rocm6.1" "$(run_index)"
done

# ── 5e. A stale-HIGH reading in each source position, one at a time ─────────
# dpkg's was the only source with a documented over-reporting state: rpm drops a
# version on erase, .info/version belongs to whichever tree /opt/rocm resolves to,
# and amd-smi and hipconfig report the userspace they were run from. For those four
# a HIGH reading is deliberately taken as truth, bounded by the tag normalisation
# (never an index leaf PyTorch does not publish) and auditable from the log.
for _pos in amd-smi version-file hipconfig rpm; do
    reset_sources
    add_hipconfig "6.1.40093-0"
    case "$_pos" in
        amd-smi)      add_amd_smi "9.9.0" ;;
        version-file) add_version_file "9.9.0-1" ;;
        hipconfig)    add_hipconfig "9.9.31921-0" ;;
        rpm)          add_rpm_rocm_core "9.9.0" ;;
    esac
    # Clipped to the newest leaf PyTorch actually publishes, never rocm9.9.
    assert_eq "a 9.9 reading from $_pos is capped to the newest published leaf" \
        "$_BASE/rocm7.2" "$(run_index)"
    if [ "$_pos" != "hipconfig" ]; then
        assert_contains "a 9.9 reading from $_pos is recorded in the log" \
            "sources disagree (rocm6.1 rocm9.9) -- using the highest, rocm9.9" "$(run_warnings)"
    fi
done

# 5f. dpkg in that same position: an INSTALLED high reading is the runtime, so the
#     cap applies rather than a rejection.
reset_sources
add_hipconfig "6.1.40093-0"
add_dpkg_rocm_core "1:9.9.0-1" installed
assert_eq "an installed 9.9 rocm-core is capped, not rejected" "$_BASE/rocm7.2" "$(run_index)"

# ── 6. A genuine ROCm 5.x host still falls back to CPU ──────────────────────
reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "5.7.1-90"
assert_eq "genuine ROCm 5.7 everywhere -> cpu" "$_BASE/cpu" "$(run_index)"
_warn=$(run_warnings)
assert_contains "5.x host warns about the 6.0+ requirement" "require ROCm 6.0+" "$_warn"
assert_contains "5.x warning names the resolved tag" "ROCm rocm5.7 detected" "$_warn"
assert_contains "5.x warning says it took the highest reading" "HIGHEST version" "$_warn"

# 7. Both overrides return early from get_torch_index_url, before any GPU probing,
#    so naming them in the warning is honest.
assert_contains "5.x warning names UNSLOTH_TORCH_INDEX_FAMILY" \
    "UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4" "$_warn"
assert_contains "5.x warning names UNSLOTH_TORCH_INDEX_URL" \
    "UNSLOTH_TORCH_INDEX_URL=" "$_warn"

# 8. And the override it names has to actually work on this same host.
reset_sources
add_hipconfig "5.7.31921-0"
add_version_file "5.7.1-90"
_result=$(PATH="$_MOCK_DIR:$_TOOLS_DIR" bash -c \
    "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
     export UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4
     _ARCH=x86_64; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null)
assert_eq "the named override reaches this path -> rocm6.4" "$_BASE/rocm6.4" "$_result"

# ── 9. Every source missing: warn, do not die ───────────────────────────────
# rocminfo alone, a fresh AMD host with no ROCm userspace. Under set -e the whole
# detection must still succeed.
# gfx1100 has its own repo.amd.com index, so since unslothai#8731 an unreadable
# version no longer settles for CPU here: the arch alone picks the wheels and the
# function defers to the reroute. The "no ROCm install" wording it used to print is
# now reserved for an arch with no per-arch index -- see
# tests/sh/test_rocm_no_version_arch_route.sh, which covers both halves.
reset_sources
assert_eq "no version source at all -> cpu" "$_BASE/cpu" "$(run_index)"
assert_eq "no version source at all -> exit 0 under set -e" "0" "$(run_status_under_set_e)"
_warn=$(run_warnings)
assert_contains "no-version host still reaches an actionable warning" \
    "routing to AMD per-arch wheels" "$_warn"
assert_contains "no-version warning names the arch it routed on" "gfx1100" "$_warn"

# 10. Sources present but every one of them unparseable: same contract.
reset_sources
add_amd_smi "N/A"
add_hipconfig "unknown"
add_version_file "not-a-version"
assert_eq "unparseable sources -> cpu" "$_BASE/cpu" "$(run_index)"
assert_eq "unparseable sources -> exit 0 under set -e" "0" "$(run_status_under_set_e)"
assert_contains "unparseable sources are treated as no version at all" \
    "routing to AMD per-arch wheels" "$(run_warnings)"

# 11. A source reporting major 0 is garbage, not a version below every other.
reset_sources
add_hipconfig "0.0.0"
add_version_file "6.2.0-1"
assert_eq "major-0 source ignored, 6.2 wins -> rocm6.2" "$_BASE/rocm6.2" "$(run_index)"

# ── 12. Supported-tag normalisation is unchanged ────────────────────────────
# PyTorch publishes major.minor index leaves only, so patch levels normalise;
# 6.5+ clips to the last 6.x wheel set and 7.3+ caps to the latest known.
for _case in "6.0.2:rocm6.0" "6.1.3:rocm6.1" "6.2.4:rocm6.2" "6.3.1:rocm6.3" \
             "6.4.1:rocm6.4" "7.0.1:rocm7.0" "7.1.0:rocm7.1" "7.2.1:rocm7.2" \
             "6.5.0:rocm6.4" "6.9.0:rocm6.4" "7.3.0:rocm7.2" "8.0.0:rocm7.2"; do
    _ver="${_case%%:*}"
    _want="${_case##*:}"
    reset_sources
    add_amd_smi "$_ver"
    assert_eq "amd-smi $_ver -> $_want" "$_BASE/$_want" "$(run_index)"
done

# ── 13. amd-smi reports one pipe-delimited line, and the ROCm field can be N/A ──
# The amdgpu driver version follows the ROCm one on the same line, so reading the
# field without stopping at the separator glued them together (N/A + amdgpu 6.10.10
# -> "rocm6.10"). Position used to hide that; under highest-wins a fabricated
# reading outvotes a correct source, so an unparseable field must yield nothing.
for _case in "N/A:" "6.4.0:rocm6.4" "7.0.2:rocm7.0" ":"; do
    _field="${_case%%:*}"
    _want="${_case##*:}"
    reset_sources
    add_amd_smi_line "$_field"
    if [ -n "$_want" ]; then
        assert_eq "amd-smi full line, ROCm field '$_field' -> $_want" \
            "$_BASE/$_want" "$(run_index)"
    else
        # Nothing else answers, so an unusable field must reach the CPU fallback.
        assert_eq "amd-smi full line, ROCm field '$_field' -> no reading" \
            "$_BASE/cpu" "$(run_index)"
    fi
done

# The driver version must never win the vote over a real, lower ROCm reading.
reset_sources
add_amd_smi_line "N/A"
add_version_file "6.1.3-42"
assert_eq "amd-smi N/A beside amdgpu 6.10 does not outvote a real 6.1" \
    "$_BASE/rocm6.1" "$(run_index)"

# ── 14. A wedged rpmdb must not hang the installer ─────────────────────────
# Highest-wins short-circuits nothing, so `rpm -q` now always runs where it used to
# be LAST in a first-answer chain and /opt/rocm/.info/version answered ahead of it
# on any normal RHEL/SLES install. Alone among the sources it can block forever on
# the rpmdb (stale BerkeleyDB __db locks on rpm < 4.16; the rpm 6.0.x read-lock
# deadlock against dnf), and an installer that hangs is worse than one that
# mis-detects, so this probe is bounded and a timed-out source declines to answer.
reset_sources
add_version_file "6.4.0-1"
add_wedged_rpm
_t0=$(date +%s)
# `|| _res=""` so the outer bound firing is a FAIL below rather than taking the
# suite down through set -e with no verdict printed.
_res=$(run_index_outer_bounded 20) || _res=""
_t1=$(date +%s)
# The wedged source must not take the answer down with it. Empty here means the
# outer bound fired, i.e. the rpm probe ran unbounded.
assert_eq "a wedged rpm does not stop the version file resolving the host" \
    "$_BASE/rocm6.4" "$_res"
if [ "$((_t1 - _t0))" -lt 20 ]; then
    assert_eq "the rpm probe is bounded, not left to block the installer" "ok" "ok"
else
    assert_eq "the rpm probe is bounded, not left to block the installer" \
        "under 20s" "$((_t1 - _t0))s (outer bound fired)"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
