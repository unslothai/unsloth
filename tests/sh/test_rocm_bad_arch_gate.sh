#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the "arch computes incorrectly under ROCm" gate in get_torch_index_url.
#
# History: the ROCm wheel index was keyed purely off the ROCm runtime VERSION, never off
# the GPU's gfx arch. gfx1033 (Van Gogh, the Steam Deck APU) is mapped to gfx103X-all in
# _GFX_TO_AMD_INDEX_ARCH, so a Deck with any ROCm runtime present was routed to ROCm
# wheels -- which install and then return wrong answers rather than refusing: training
# diverges to a negative MSE loss and then NaN, and torch.autograd.gradcheck fails in
# float64, reproduced on rocm7.1/torch 2.10, rocm7.2/torch 2.11 and AMD's own native
# gfx1033 build. Forward math matches CPU. See studio/ROCM_RDNA2_APU.md.
#
# The contract: gfx1033 routes to the cpu index; every other arch is untouched.
#
# Scope note, and the reason this is not gated on AMD's hardware support table: unsloth
# deliberately serves archs AMD does not list -- gfx906 (MI50) via the rocm6.3 legacy
# index, verified there with torch 2.7.0, and gfx1031-gfx1036 via gfx103X-all (#7277).
# A table-wide gate would drop support that is known to work, so this list holds only
# archs measured to be wrong.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

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

# The gate lives inline in get_torch_index_url on purpose (several harnesses extract that
# function alone; a helper they missed would be an undefined command whose negation sent
# every ROCm case to cpu). So exercise the real case block lifted from the function.
_FN_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE"' EXIT
awk '/# Archs measured to compute INCORRECTLY under ROCm/,/^        esac$/' "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q 'gfx1033' "$_FN_FILE"; then
    echo "FAIL: could not extract the gfx gate from get_torch_index_url in install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
# Build a real function whose BODY is the extracted block, then call it. Sourcing the
# block instead would not do: `return` in a dot-script unwinds the source, not the caller,
# so execution would fall through to the trailing echo and print both lines. Embedding it
# reproduces exactly how `return` behaves inside get_torch_index_url.
_GATE_FILE=$(mktemp)
{
    echo '_gate() {'
    cat "$_FN_FILE"
    echo '    echo rocm'
    echo '}'
} > "$_GATE_FILE"
trap 'rm -f "$_FN_FILE" "$_GATE_FILE"' EXIT

_SH="${BASH:-/bin/bash}"
_route() {  # gfx -> the cpu index url when the gate intercepts, else "rocm"
    "$_SH" -c "
        _base=https://download.pytorch.org/whl
        _amd_gfx_probe='$1'
        . '$_GATE_FILE'
        _gate
    " 2>/dev/null | tail -1
}

echo "=== The measured-bad arch routes to CPU ==="
assert_eq "gfx1033 -> cpu"                 "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route gfx1033)"
assert_eq "gfx1033 with feature suffix"    "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route 'gfx1033:xnack-')"
assert_eq "GFX1033 uppercase"              "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route GFX1033)"

echo "=== Everything else falls through to the ROCm path, unchanged ==="
# gfx906 and gfx1031-1036 are the deliberately-supported-beyond-AMD's-table archs; if
# this gate ever swallows them it has silently dropped verified support.
for _gfx in gfx906 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1036 \
            gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1103 \
            gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201; do
    assert_eq "$_gfx not intercepted" "rocm" "$(_route "$_gfx")"
done
assert_eq "empty probe not intercepted"   "rocm" "$(_route '')"
assert_eq "garbage not intercepted"       "rocm" "$(_route 'not-a-gfx')"
assert_eq "gfx10330 is not gfx1033"       "rocm" "$(_route gfx10330)"

echo "=== The runtime-less reroute honours the same gate ==="
# get_torch_index_url's gate is not enough on its own. The reroute below it fires when
# UNSLOTH_ROCM_GFX_ARCH is set, takes that value as the inferred arch, and rewrites a
# */cpu index to the arch family index -- so setting UNSLOTH_ROCM_GFX_ARCH=gfx1033 sent
# a Deck straight back to gfx103X-all, undoing the gate. The documented escape hatch is
# UNSLOTH_TORCH_INDEX_URL, which returns long before either point.
_REROUTE_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE"' EXIT
{
    awk '/^_amd_arch_index_family_for_gfx\(\)/,/^}/' "$INSTALL_SH"
    awk '/^_infer_linux_amd_gfx_arch\(\)/,/^}/' "$INSTALL_SH"
    echo '_reroute() {'
    awk '/_linux_inferred_gfx=\$\(_infer_linux_amd_gfx_arch/,/^            if \[ -n "\$_linux_inferred_gfx" \]; then$/' "$INSTALL_SH"
    echo '    _amd_arch_index_family_for_gfx "$_linux_inferred_gfx" || echo cpu'
    echo '    else echo cpu'   # gated out: no arch survives, so no reroute happens
    echo '    fi'
    echo '}'
} > "$_REROUTE_FILE"

_reroute_family() {  # UNSLOTH_ROCM_GFX_ARCH -> family index, or cpu when gated out
    "$_SH" -c "
        UNSLOTH_ROCM_GFX_ARCH='$1'; export UNSLOTH_ROCM_GFX_ARCH
        . '$_REROUTE_FILE'
        _reroute
    " 2>/dev/null | tail -1
}
assert_eq "gfx1033 override does not reroute"  "cpu"          "$(_reroute_family gfx1033)"
assert_eq "GFX1033 override does not reroute"  "cpu"          "$(_reroute_family GFX1033)"
assert_eq "gfx1032 override still reroutes"    "gfx103X-all"  "$(_reroute_family gfx1032)"
assert_eq "gfx1030 override still reroutes"    "gfx103X-all"  "$(_reroute_family gfx1030)"
assert_eq "gfx1151 override still reroutes"    "gfx1151"      "$(_reroute_family gfx1151)"

echo "=== The rejected override is not forwarded to llama.cpp either ==="
# Keeping PyTorch on CPU while leaving UNSLOTH_ROCM_GFX_ARCH exported splits the host in
# two: setup.sh copies the value into --rocm-gfx and _apply_host_overrides reads any
# forwarded gfx as proof of ROCm, which skips the AMD-without-ROCm Vulkan branch (112.8
# tok/s measured, vs 49.8 for the CPU bundle) and asks a Deck with no ROCm for a ROCm
# prebuilt or a HIP source build. A rejected arch must leave nothing behind; every other
# arch must still be handed on, or the runtime-less hosts this reroute exists for get AMD
# wheels for torch and a CPU llama.cpp.
_forwarded_gfx() {  # UNSLOTH_ROCM_GFX_ARCH -> what survives for setup.sh, or <unset>
    "$_SH" -c "
        UNSLOTH_ROCM_GFX_ARCH='$1'; export UNSLOTH_ROCM_GFX_ARCH
        . '$_REROUTE_FILE'
        _reroute >/dev/null 2>&1
        printf '%s' \"\${UNSLOTH_ROCM_GFX_ARCH:-<unset>}\"
    " 2>/dev/null | tail -1
}
assert_eq "gfx1033 override not forwarded"  "<unset>"  "$(_forwarded_gfx gfx1033)"
assert_eq "GFX1033 override not forwarded"  "<unset>"  "$(_forwarded_gfx GFX1033)"
assert_eq "gfx1030 override still forwarded" "gfx1030" "$(_forwarded_gfx gfx1030)"
assert_eq "gfx1151 override still forwarded" "gfx1151" "$(_forwarded_gfx gfx1151)"

echo "=== End to end: the REAL get_torch_index_url against a REAL rocminfo shape ==="
# Everything above feeds the gate a hand-built one-token _amd_gfx_probe, which is not
# what the probe produces. _probe_amd_gfx_arch keeps every `grep -oE` hit and rocminfo
# names each GPU agent TWICE -- once as the agent's own "Name: gfx1033" and once in its
# ISA Info block as "amdgcn-amd-amdhsa--gfx1033" -- so a single-GPU Steam Deck already
# yields "gfx1033\ngfx1033". A gate that compares the whole probe as one string matches
# neither, and the Deck falls through to the version-keyed ROCm index carrying exactly
# the wheels this gate exists to avoid. Drive the real function to catch that.
_E2E_DIR=$(mktemp -d)
_E2E_FUNCS="$_E2E_DIR/funcs.sh"
_FAKE_SMI_DIR=$(mktemp -d)
_FAKE_ROCM_DIR=$(mktemp -d)
_TOOLS_DIR=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE" "$_E2E_DIR" "$_FAKE_SMI_DIR" "$_FAKE_ROCM_DIR" "$_TOOLS_DIR"' EXIT

# Same extraction contract as tests/sh/test_get_torch_index_url.sh: miss a helper and
# the ROCm branch hits an undefined command and silently answers cpu, which would make
# these assertions pass for the wrong reason -- so the ROCm assertion below is the guard.
{
    for _fn in _run_bounded _cvd_hides_nvidia _has_amd_rocm_gpu _has_usable_nvidia_gpu \
               _ensure_rocm_probe_env _probe_amd_gfx_arch _amd_gpu_present_via_pci \
               _infer_amd_gfx_arch_from_gpu_name _infer_linux_amd_gfx_arch \
               _amd_arch_index_family_for_gfx _trim_index_path_slashes \
               _nvidia_cu126_verdict _cap_cuda_family_for_pre_turing \
               _rocm_tag_from_amd_smi _rocm_tag_from_version_file _rocm_tag_from_hipconfig \
               _rocm_tag_from_dpkg _rocm_tag_from_rpm _highest_rocm_tag \
               _detect_rocm_version_tag get_torch_index_url; do
        sed -n "/^$_fn()/,/^}/p" "$INSTALL_SH"
        echo ""
    done
} | sed -e "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
      -e "s|/opt/rocm|$_FAKE_ROCM_DIR|g" > "$_E2E_FUNCS"

for _cmd in uname grep sed head sh bash cat awk printf tr cut sort timeout; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

_make_rocminfo_host() {  # $1 = gfx arch -> a dir holding rocminfo + hipconfig mocks
    _mk_dir=$(mktemp -d)
    # Verbatim shape of a real single-GPU APU host: a CPU agent with no ISA, then the
    # GPU agent whose arch appears in BOTH its Name and its ISA Info Name.
    cat > "$_mk_dir/rocminfo" <<ROCMINFO
#!/bin/sh
cat <<'OUT'
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD Custom APU 0405
  Uuid:                    CPU-XX
  Marketing Name:          AMD Custom APU 0405
  Device Type:             CPU
  ISA Info:
    N/A
*******
Agent 2
*******
  Name:                    $1
  Uuid:                    GPU-XX
  Marketing Name:          AMD Custom GPU 0405
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--$1
      Machine Models:          HSA_MACHINE_MODEL_LARGE
*** Done ***
OUT
ROCMINFO
    # A readable ROCm 7.2 userspace, so a host that clears the gate really does reach
    # the version-keyed index rather than the "no ROCm version" cpu fallback.
    printf '#!/bin/sh\necho 7.2.0\n' > "$_mk_dir/hipconfig"
    chmod +x "$_mk_dir/rocminfo" "$_mk_dir/hipconfig"
    printf '%s' "$_mk_dir"
}

_index_for_rocminfo_host() {  # $1 = gfx arch -> the index get_torch_index_url picks
    _ifh_dir=$(_make_rocminfo_host "$1")
    PATH="$_ifh_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_ifh_dir"
}

# The probe really is multi-line -- assert that first, so a future single-hit probe
# does not turn the two assertions below into a vacuous pass.
_probe_lines=$( _pl_dir=$(_make_rocminfo_host gfx1033)
    PATH="$_pl_dir:$_TOOLS_DIR" "$_SH" -c "
        unset UNSLOTH_ROCM_GFX_ARCH; . '$_E2E_FUNCS'; _probe_amd_gfx_arch" 2>/dev/null \
        | grep -c gfx1033
    rm -rf "$_pl_dir" )
assert_eq "rocminfo yields more than one gfx token" "yes" \
    "$([ "${_probe_lines:-0}" -gt 1 ] && echo yes || echo no)"

assert_eq "gfx1033 rocminfo host -> cpu index" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_rocminfo_host gfx1033)"
# Negative arm: the gate must intercept ONLY the measured-bad arch. gfx1030 is the
# family neighbour that install.sh deliberately serves through gfx103X-all/ROCm.
assert_eq "gfx1030 rocminfo host -> rocm index" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_rocminfo_host gfx1030)"

echo "=== Structural: the gate precedes the version-keyed index selection ==="
_gate_line=$(grep -n 'Archs measured to compute INCORRECTLY under ROCm' "$INSTALL_SH" | head -1 | cut -d: -f1)
_idx_line=$(grep -n 'rocm7.2|rocm7.2.\*) echo "\$_base/rocm7.2"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "gate is before the rocm index case" "yes" \
    "$([ -n "$_gate_line" ] && [ -n "$_idx_line" ] && [ "$_gate_line" -lt "$_idx_line" ] && echo yes || echo no)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
