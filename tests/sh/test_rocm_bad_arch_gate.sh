#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# gfx1033 computes incorrectly under ROCm and routes to the cpu index; every other arch is
# untouched. gfx906 and gfx1031-gfx1036 are served deliberately.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
# The gate is inline on purpose: an extracted function missing a helper would send every
# ROCm case to cpu. So exercise the real case block.
_FN_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE"' EXIT
# An explicit end marker, not `fi`: a second `if` at the same indent would truncate.
awk '/# Archs measured to compute INCORRECTLY under ROCm/,/^        # end of the miscomputing-arch gate/' \
    "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q 'gfx1033' "$_FN_FILE"; then
    echo "FAIL: could not extract the gfx gate from get_torch_index_url in install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
# A real function, not a dot-script: `return` in a sourced file unwinds the source.
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
# gfx906 and gfx1031-1036 are supported beyond AMD's table.
for _gfx in gfx906 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1036 \
            gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1103 \
            gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201; do
    assert_eq "$_gfx not intercepted" "rocm" "$(_route "$_gfx")"
done
assert_eq "empty probe not intercepted"   "rocm" "$(_route '')"
assert_eq "garbage not intercepted"       "rocm" "$(_route 'not-a-gfx')"
assert_eq "gfx10330 is not gfx1033"       "rocm" "$(_route gfx10330)"

echo "=== A mixed host takes the cpu index too: presence, not selection ==="
# PRESENCE only: a healthy dGPU beside the APU no longer keeps the host on ROCm.
assert_eq "gfx1033 + gfx1100 -> cpu"   "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033
gfx1100')"
assert_eq "gfx1100 + gfx1033 -> cpu"   "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1100
gfx1033')"
# rocminfo names each agent twice, so a single-GPU Deck repeats the token.
assert_eq "repeated gfx1033 -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033
gfx1033')"
assert_eq "mixed case gfx1033 -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033:xnack-
GFX1033')"

echo "=== The runtime-less reroute honours the same gate ==="
# The reroute below the gate reads UNSLOTH_ROCM_GFX_ARCH and would rewrite */cpu back to
# a family index. The escape hatch is UNSLOTH_TORCH_INDEX_URL.
_REROUTE_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE"' EXIT
{
    awk '/^_amd_arch_index_family_for_gfx\(\)/,/^}/' "$INSTALL_SH"
    awk '/^_amd_probe_arches\(\)/,/^}/' "$INSTALL_SH"
    awk '/^_amd_sole_index_arch\(\)/,/^}/' "$INSTALL_SH"
    awk '/^_infer_linux_amd_gfx_arch\(\)/,/^}/' "$INSTALL_SH"
    echo '_reroute() {'
    awk '/_linux_inferred_gfx=\$\(_infer_linux_amd_gfx_arch/,/^            if \[ -n "\$_amd_family" \]; then$/' "$INSTALL_SH"
    echo '    echo "$_amd_family"'
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
# A rejected arch must not stay exported: setup.sh forwards it as --rocm-gfx, which skips
# the faster Vulkan branch.
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
# The real probe names each GPU agent twice, so drive the real function.
_E2E_DIR=$(mktemp -d)
_E2E_FUNCS="$_E2E_DIR/funcs.sh"
_FAKE_SMI_DIR=$(mktemp -d)
_FAKE_ROCM_DIR=$(mktemp -d)
_TOOLS_DIR=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE" "$_E2E_DIR" "$_FAKE_SMI_DIR" "$_FAKE_ROCM_DIR" "$_TOOLS_DIR"' EXIT

# Same extraction contract as test_get_torch_index_url.sh: a missed helper makes ROCm
# answer cpu and these pass wrongly. The ROCm assertion below guards it.
{
    for _fn in _run_bounded _cvd_hides_nvidia _has_amd_rocm_gpu _has_usable_nvidia_gpu \
               _ensure_rocm_probe_env _rocm_torch_explicitly_requested \
               _probe_amd_gfx_arch _amd_gfx_select_ordinals \
               _amd_gpu_present_via_pci \
               _infer_amd_gfx_arch_from_gpu_name _infer_linux_amd_gfx_arch \
               _amd_arch_index_family_for_gfx _trim_index_path_slashes \
               _nvidia_cu126_verdict _cap_cuda_family_for_pre_turing \
               _rocm_tag_from_amd_smi _rocm_tag_from_version_file _rocm_tag_from_hipconfig \
               _rocm_tag_from_dpkg _rocm_tag_from_rpm _highest_rocm_tag \
               _detect_rocm_version_tag _kfd_gfx_targets get_torch_index_url; do
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
    # Real single-GPU APU shape: a CPU agent with no ISA, then a GPU agent named twice.
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
    # A readable ROCm 7.2 so a host clearing the gate reaches the version-keyed index.
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

# Assert the probe is multi-line first, or the assertions below are vacuous.
_probe_lines=$( _pl_dir=$(_make_rocminfo_host gfx1033)
    PATH="$_pl_dir:$_TOOLS_DIR" "$_SH" -c "
        unset UNSLOTH_ROCM_GFX_ARCH; . '$_E2E_FUNCS'; _probe_amd_gfx_arch" 2>/dev/null \
        | grep -c gfx1033
    rm -rf "$_pl_dir" )
assert_eq "rocminfo yields more than one gfx token" "yes" \
    "$([ "${_probe_lines:-0}" -gt 1 ] && echo yes || echo no)"

assert_eq "gfx1033 rocminfo host -> cpu index" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_rocminfo_host gfx1033)"
# gfx1030 is the family neighbour served through gfx103X-all.
assert_eq "gfx1030 rocminfo host -> rocm index" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_rocminfo_host gfx1030)"

# Real two-agent shape: presence decides, and no mask can move it.
_make_two_agent_host() {
    _ta_apu=$(_make_rocminfo_host gfx1033)
    _ta_dgpu=$(_make_rocminfo_host gfx1100)
    _ta_dir=$(mktemp -d)
    cp "$_ta_apu/hipconfig" "$_ta_dir/hipconfig"
    printf '#!/bin/sh\n"%s/rocminfo"\n"%s/rocminfo"\n' "$_ta_apu" "$_ta_dgpu" > "$_ta_dir/rocminfo"
    chmod +x "$_ta_dir/rocminfo"
    printf '%s' "$_ta_dir"
}
_index_for_two_agent_host() {  # $1 = extra "VAR=value" env, or empty
    _ith_dir=$(_make_two_agent_host)
    PATH="$_ith_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES
        unset HIP_VISIBLE_DEVICES
        [ -z '$1' ] || export $1
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_ith_dir"
}
assert_eq "two-agent host (APU + dGPU) -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_two_agent_host '')"
assert_eq "two-agent host + ROCR=1 -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_two_agent_host ROCR_VISIBLE_DEVICES=1)"
assert_eq "two-agent host + HIP=1 -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_two_agent_host HIP_VISIBLE_DEVICES=1)"
assert_eq "two-agent host + UUID mask -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" \
    "$(_index_for_two_agent_host ROCR_VISIBLE_DEVICES=GPU-DEADBEEFDEADBEEF)"

echo "=== HSA_OVERRIDE_GFX_VERSION=10.3.0, the circulated Van Gogh workaround ==="
# ROCr applies HSA_OVERRIDE_GFX_VERSION in userland, so a real Deck's rocminfo answers gfx1030.
_make_spoofing_host() {  # -> a dir whose rocminfo honours HSA_OVERRIDE_GFX_VERSION
    _sp_real=$(_make_rocminfo_host gfx1033)
    _sp_spoofed=$(_make_rocminfo_host gfx1030)
    _sp_dir=$(mktemp -d)
    cp "$_sp_real/hipconfig" "$_sp_dir/hipconfig"
    cat > "$_sp_dir/rocminfo" <<SPOOF
#!/bin/sh
if [ -n "\${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
    exec "$_sp_spoofed/rocminfo"
fi
exec "$_sp_real/rocminfo"
SPOOF
    chmod +x "$_sp_dir/rocminfo"
    printf '%s' "$_sp_dir"
}

_index_for_spoofed_host() {  # $1 = HSA_OVERRIDE_GFX_VERSION ("" to leave it unset)
    _ish_dir=$(_make_spoofing_host)
    PATH="$_ish_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY
        if [ -n '$1' ]; then HSA_OVERRIDE_GFX_VERSION='$1'; export HSA_OVERRIDE_GFX_VERSION
        else unset HSA_OVERRIDE_GFX_VERSION; fi
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_ish_dir"
}

# Assert the mock really spoofs, or the gate assertion passes with nothing hidden.
_spoofed_probe=$( _sp_check=$(_make_spoofing_host)
    PATH="$_sp_check:$_TOOLS_DIR" "$_SH" -c "
        unset UNSLOTH_ROCM_GFX_ARCH
        HSA_OVERRIDE_GFX_VERSION=10.3.0; export HSA_OVERRIDE_GFX_VERSION
        . '$_E2E_FUNCS'; _probe_amd_gfx_arch | head -1" 2>/dev/null
    rm -rf "$_sp_check" )
assert_eq "the mock hides gfx1033 behind the override" "gfx1030" "$_spoofed_probe"

assert_eq "unspoofed Deck -> cpu index" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_spoofed_host '')"
assert_eq "spoofed Deck -> cpu index anyway" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_spoofed_host 10.3.0)"
# The re-probe is scoped to the gate: only gfx1033 silicon answers gfx1033 unspoofed.
assert_eq "spoofed gfx1030 host keeps rocm" \
    "https://download.pytorch.org/whl/rocm7.2" \
    "$(export HSA_OVERRIDE_GFX_VERSION=10.3.0; _index_for_rocminfo_host gfx1030)"

echo "=== KFD answers when the probe cannot, so a spoof cannot fill the gap ==="
# _amd_gfx_probe is collected with the override in force; amdkfd is the kernel's own table
# and no runtime variable reaches it.
_make_kfd_only_host() {  # rocminfo that answers ONLY while the override is set
    _ko_dir=$(mktemp -d)
    _ko_spoof=$(_make_rocminfo_host gfx1030)
    cp "$_ko_spoof/hipconfig" "$_ko_dir/hipconfig"
    cat > "$_ko_dir/rocminfo" <<KFDONLY
#!/bin/sh
if [ -n "\${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
    exec "$_ko_spoof/rocminfo"
fi
exit 1
KFDONLY
    chmod +x "$_ko_dir/rocminfo"
    printf '%s' "$_ko_dir"
}
_index_for_kfd_host() {  # $1 = gfx the kernel reports ("" for a KFD that says nothing)
    _ifk_dir=$(_make_kfd_only_host)
    _ifk_stub=$(mktemp -d)
    cat > "$_ifk_stub/kfd.sh" <<KSTUB
_kfd_gfx_targets() { [ -z '$1' ] || printf '%s\n' '$1'; }
KSTUB
    PATH="$_ifk_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        HSA_OVERRIDE_GFX_VERSION=10.3.0; export HSA_OVERRIDE_GFX_VERSION
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_ifk_stub/kfd.sh'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_ifk_dir" "$_ifk_stub"
}
assert_eq "KFD names gfx1033 behind the spoof -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_kfd_host gfx1033)"
# KFD naming a healthy card is believed too: a source, not a veto.
assert_eq "KFD names gfx1030 -> rocm" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_kfd_host gfx1030)"

# _has_amd_rocm_gpu and _kfd_gfx_targets read /dev/kfd by absolute path, so stub presence and
# silence KFD to make the simulated host the subject, not the runner's silicon.
_amd_host_no_kfd_stub() {  # -> a file to source AFTER funcs.sh
    _ahs_dir=$(mktemp -d)
    {
        printf '_has_amd_rocm_gpu() { return 0; }\n'
        printf '_kfd_gfx_targets() { :; }\n'
    } > "$_ahs_dir/host.sh"
    printf '%s' "$_ahs_dir/host.sh"
}

echo "=== An override nothing can verify is not evidence of a healthy arch ==="
# Older ROCr answering only under the override, no amd-smi, no KFD: unverifiable, not clean.
_index_unverifiable_override() {  # $1 = the env assignment to apply
    _iuo_dir=$(_make_kfd_only_host)
    _iuo_stub=$(_amd_host_no_kfd_stub)
    PATH="$_iuo_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        unset HSA_OVERRIDE_GFX_VERSION
        export $1
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_iuo_stub'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_iuo_dir" "$(dirname "$_iuo_stub")"
}
assert_eq "HSA override with no verifiable source -> cpu" \
    "https://download.pytorch.org/whl/cpu" \
    "$(_index_unverifiable_override HSA_OVERRIDE_GFX_VERSION=10.3.0)"
# UNSLOTH_ROCM_GFX_ARCH is a declared arch, not a spoof. Both reach the cpu index, so assert
# on which branch spoke.
_stderr_unverifiable_override() {  # $1 = env assignment -> stderr only
    _suo_dir=$(_make_kfd_only_host)
    _suo_stub=$(_amd_host_no_kfd_stub)
    PATH="$_suo_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        unset HSA_OVERRIDE_GFX_VERSION
        export $1
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_suo_stub'
        get_torch_index_url >/dev/null
    " 2>&1
    rm -rf "$_suo_dir" "$(dirname "$_suo_stub")"
}
assert_eq "the spoof refusal names HSA_OVERRIDE_GFX_VERSION" "yes" \
    "$(_stderr_unverifiable_override HSA_OVERRIDE_GFX_VERSION=10.3.0 \
       | grep -qF 'HSA_OVERRIDE_GFX_VERSION is set and this host cannot confirm' && echo yes || echo no)"
assert_eq "a declared arch on a tool-blind host is not a spoof" "yes" \
    "$(_stderr_unverifiable_override UNSLOTH_ROCM_GFX_ARCH=gfx1151 \
       | grep -qF 'cannot confirm its real arch' && echo no || echo yes)"
# Nothing is spoofed, so the declared gfx1151 is simply the arch.
assert_eq "a declared arch on a real AMD host keeps its rocm index" \
    "https://download.pytorch.org/whl/rocm7.2" \
    "$(_index_unverifiable_override UNSLOTH_ROCM_GFX_ARCH=gfx1151)"
# With no override, an empty physical read is just an unreadable host.
assert_eq "no override and no probe is not treated as a spoof" \
    "rocm" "$(_route '')"

echo "=== A declared arch must not answer for the silicon ==="
# UNSLOTH_ROCM_GFX_ARCH short-circuits _probe_amd_gfx_arch; "physical" mode skips it.
_index_for_declared_arch() {  # $1 = real silicon, $2 = UNSLOTH_ROCM_GFX_ARCH
    _ida_dir=$(_make_rocminfo_host "$1")
    PATH="$_ida_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
        unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        UNSLOTH_ROCM_GFX_ARCH='$2'; export UNSLOTH_ROCM_GFX_ARCH
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_ida_dir"
}

assert_eq "stale UNSLOTH_ROCM_GFX_ARCH=gfx1030 on a Deck -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_declared_arch gfx1033 gfx1030)"
assert_eq "declared gfx1030 on a real gfx1030 host keeps rocm" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_declared_arch gfx1030 gfx1030)"
_no_probe_index=$( _np=$(mktemp -d)
    _np_stub=$(_amd_host_no_kfd_stub)
    printf '#!/bin/sh\necho 7.2.0\n' > "$_np/hipconfig"; chmod +x "$_np/hipconfig"
    PATH="$_np:$_TOOLS_DIR" "$_SH" -c "
        unset UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY HSA_OVERRIDE_GFX_VERSION
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
        UNSLOTH_ROCM_GFX_ARCH=gfx1033; export UNSLOTH_ROCM_GFX_ARCH
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_np_stub'
        get_torch_index_url" 2>/dev/null | tail -1
    rm -rf "$_np" "$(dirname "$_np_stub")" )
assert_eq "declared gfx1033 with no probe tool -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$_no_probe_index"

echo "=== Structural: the gate precedes the version-keyed index selection ==="
_gate_line=$(grep -n 'Archs measured to compute INCORRECTLY under ROCm' "$INSTALL_SH" | head -1 | cut -d: -f1)
_idx_line=$(grep -n 'rocm7.2|rocm7.2.\*) echo "\$_base/rocm7.2"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "gate is before the rocm index case" "yes" \
    "$([ -n "$_gate_line" ] && [ -n "$_idx_line" ] && [ "$_gate_line" -lt "$_idx_line" ] && echo yes || echo no)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
