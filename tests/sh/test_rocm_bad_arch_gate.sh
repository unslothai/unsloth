#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the "arch computes incorrectly under ROCm" gate in get_torch_index_url.
#
# Contract: gfx1033 routes to the cpu index; every other arch is untouched. The index used
# to key off the ROCm runtime VERSION alone, so a Deck was routed to ROCm wheels that
# install and then compute wrong answers (studio/ROCM_RDNA2_APU.md). Not gated on AMD's
# support table, because unsloth deliberately serves gfx906 and gfx1031-gfx1036 (#7277).
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
# Delimited by an explicit end marker, not the block's closing `fi`: the gate has a second
# `if` at the same indent, so a structural end would truncate the extract.
awk '/# Archs measured to compute INCORRECTLY under ROCm/,/^        # end of the miscomputing-arch gate/' \
    "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q 'gfx1033' "$_FN_FILE"; then
    echo "FAIL: could not extract the gfx gate from get_torch_index_url in install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
# Build a real function whose BODY is the extracted block, then call it: `return` in a
# dot-script unwinds the source, not the caller, so sourcing would print both lines.
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

echo "=== A mixed host takes the cpu index too: presence, not selection ==="
# PRESENCE only: a healthy dGPU beside the APU no longer keeps the host on ROCm, because
# resolving which device the runtime selects needs per-device identity and mask layering,
# which _runtime_gfx_target() does and a token list cannot.
assert_eq "gfx1033 + gfx1100 -> cpu"   "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033
gfx1100')"
assert_eq "gfx1100 + gfx1033 -> cpu"   "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1100
gfx1033')"
# rocminfo names each agent twice, so the real single-GPU Deck probe repeats the token.
assert_eq "repeated gfx1033 -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033
gfx1033')"
assert_eq "mixed case gfx1033 -> cpu" \
    "https://download.pytorch.org/whl/cpu" "$(_route 'gfx1033:xnack-
GFX1033')"

echo "=== The runtime-less reroute honours the same gate ==="
# The gate is not enough on its own: the reroute below it takes UNSLOTH_ROCM_GFX_ARCH as
# the inferred arch and rewrites */cpu to the family index, undoing the gate. The
# documented escape hatch is UNSLOTH_TORCH_INDEX_URL, which returns long before either.
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
# Leaving UNSLOTH_ROCM_GFX_ARCH exported splits the host in two: setup.sh forwards it as
# --rocm-gfx and _apply_host_overrides reads any forwarded gfx as proof of ROCm, skipping
# the Vulkan branch (112.8 tok/s vs 49.8 CPU). A rejected arch must leave nothing behind;
# every other arch must still be handed on.
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
# Everything above feeds a hand-built one-token probe, which is not what the real one
# produces: rocminfo names each GPU agent TWICE (its "Name:" and its ISA Info line), so a
# single-GPU Deck already yields "gfx1033\ngfx1033" and a whole-string compare matches
# neither. Drive the real function to catch that.
_E2E_DIR=$(mktemp -d)
_E2E_FUNCS="$_E2E_DIR/funcs.sh"
_FAKE_SMI_DIR=$(mktemp -d)
_FAKE_ROCM_DIR=$(mktemp -d)
_TOOLS_DIR=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE" "$_E2E_DIR" "$_FAKE_SMI_DIR" "$_FAKE_ROCM_DIR" "$_TOOLS_DIR"' EXIT

# Same extraction contract as tests/sh/test_get_torch_index_url.sh: a missed helper makes
# the ROCm branch answer cpu, so these would pass for the wrong reason. The ROCm
# assertion below is the guard against that.
{
    for _fn in _run_bounded _cvd_hides_nvidia _has_amd_rocm_gpu _has_usable_nvidia_gpu \
               _ensure_rocm_probe_env _probe_amd_gfx_arch _amd_gfx_select_ordinals \
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
    # Real single-GPU APU shape: a CPU agent with no ISA, then the GPU agent whose arch
    # appears in BOTH its Name and its ISA Info Name.
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

# End to end on a REAL two-agent rocminfo shape: presence is what decides, and a mask
# cannot change it in either direction, because the gate never asks about selection.
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
# The verdict is mask-independent by construction, which is the property that ended the
# ordinal/UUID/probe-precedence chase: there is nothing left for a mask to distort.
assert_eq "two-agent host + ROCR=1 -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_two_agent_host ROCR_VISIBLE_DEVICES=1)"
assert_eq "two-agent host + HIP=1 -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" "$(_index_for_two_agent_host HIP_VISIBLE_DEVICES=1)"
assert_eq "two-agent host + UUID mask -> cpu (unchanged)" \
    "https://download.pytorch.org/whl/cpu" \
    "$(_index_for_two_agent_host ROCR_VISIBLE_DEVICES=GPU-DEADBEEFDEADBEEF)"

echo "=== HSA_OVERRIDE_GFX_VERSION=10.3.0, the circulated Van Gogh workaround ==="
# ROCr applies the override in USERLAND while building agent names, so with it set the
# real rocminfo on a Deck answers gfx1030 and the gate saw no bad token on the one host
# it exists for. This mock is that behaviour: one host, two answers.
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

# The mock really does spoof -- assert it first, or the gate assertion below passes
# because nothing was hidden from it.
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
# The re-probe is scoped to the gate, so a host that really is the reported arch keeps
# its ROCm index with the override set: only gfx1033 silicon answers gfx1033 unspoofed.
assert_eq "spoofed gfx1030 host keeps rocm" \
    "https://download.pytorch.org/whl/rocm7.2" \
    "$(export HSA_OVERRIDE_GFX_VERSION=10.3.0; _index_for_rocminfo_host gfx1030)"

echo "=== KFD answers when the probe cannot, so a spoof cannot fill the gap ==="
# The fallback chain ends at _amd_gfx_probe, collected WITH the override in force, so on
# a host whose ROCr cannot re-enumerate once it is stripped that fallback reports the
# spoofed arch. amdkfd is the kernel's own table and no runtime variable reaches it.
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
# KFD naming a healthy card is believed too: this is a source, not a veto.
assert_eq "KFD names gfx1030 -> rocm" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_kfd_host gfx1030)"

echo "=== An override nothing can verify is not evidence of a healthy arch ==="
# Older ROCr that answers only while the override is set, no amd-smi, no KFD: the chain
# used to end at the spoofed probe. Absence of evidence is not evidence of absence.
_index_unverifiable_override() {  # $1 = the env assignment to apply
    _iuo_dir=$(_make_kfd_only_host)
    _iuo_stub=$(mktemp -d)
    printf '_kfd_gfx_targets() { :; }\n' > "$_iuo_stub/kfd.sh"
    PATH="$_iuo_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        unset HSA_OVERRIDE_GFX_VERSION
        export $1
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_iuo_stub/kfd.sh'
        get_torch_index_url
    " 2>/dev/null | tail -1
    rm -rf "$_iuo_dir" "$_iuo_stub"
}
assert_eq "HSA override with no verifiable source -> cpu" \
    "https://download.pytorch.org/whl/cpu" \
    "$(_index_unverifiable_override HSA_OVERRIDE_GFX_VERSION=10.3.0)"
# UNSLOTH_ROCM_GFX_ARCH is a DECLARED arch, not a spoof: it renames nothing, and a
# tool-blind host is the runtime-less #7301 population the reroute serves. Treating it as
# unverifiable would strand every legitimate gfx1151 install on the cpu index.
# Both reach the cpu index, so the INDEX alone cannot tell the spoof refusal apart from
# the ordinary no-version deferral. Assert on which branch spoke.
_stderr_unverifiable_override() {  # $1 = env assignment -> stderr only
    _suo_dir=$(_make_kfd_only_host)
    _suo_stub=$(mktemp -d)
    printf '_kfd_gfx_targets() { :; }\n' > "$_suo_stub/kfd.sh"
    PATH="$_suo_dir:$_TOOLS_DIR" "$_SH" -c "
        unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_TORCH_INDEX_URL
        unset UNSLOTH_TORCH_INDEX_FAMILY ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        unset HSA_OVERRIDE_GFX_VERSION
        export $1
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        . '$_suo_stub/kfd.sh'
        get_torch_index_url >/dev/null
    " 2>&1
    rm -rf "$_suo_dir" "$_suo_stub"
}
assert_eq "the spoof refusal names HSA_OVERRIDE_GFX_VERSION" "yes" \
    "$(_stderr_unverifiable_override HSA_OVERRIDE_GFX_VERSION=10.3.0 \
       | grep -qF 'HSA_OVERRIDE_GFX_VERSION is set and this host cannot confirm' && echo yes || echo no)"
assert_eq "a declared arch on a tool-blind host is not a spoof" "yes" \
    "$(_stderr_unverifiable_override UNSLOTH_ROCM_GFX_ARCH=gfx1151 \
       | grep -qF 'cannot confirm its real arch' && echo no || echo yes)"
assert_eq "and it still reaches the cpu index for the reroute to pick up" \
    "https://download.pytorch.org/whl/cpu" \
    "$(_index_unverifiable_override UNSLOTH_ROCM_GFX_ARCH=gfx1151)"
# With NO override in force there is nothing being spoofed, so an empty physical read is
# just a host the probes cannot read, and the pre-existing routing is left alone.
assert_eq "no override and no probe is not treated as a spoof" \
    "rocm" "$(_route '')"

echo "=== A declared arch must not answer for the silicon ==="
# UNSLOTH_ROCM_GFX_ARCH short-circuits the top of _probe_amd_gfx_arch, so a stale gfx1030
# on a real Van Gogh answered the gate with a healthy arch. "physical" mode skips it.
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
# The override still routes an ordinary host: a real gfx1030 keeps ROCm either way.
assert_eq "declared gfx1030 on a real gfx1030 host keeps rocm" \
    "https://download.pytorch.org/whl/rocm7.2" "$(_index_for_declared_arch gfx1030 gfx1030)"
# And it is still honoured where nothing else can answer: no rocminfo, no amd-smi.
_no_probe_index=$( _np=$(mktemp -d)
    printf '#!/bin/sh\necho 7.2.0\n' > "$_np/hipconfig"; chmod +x "$_np/hipconfig"
    PATH="$_np:$_TOOLS_DIR" "$_SH" -c "
        unset UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY HSA_OVERRIDE_GFX_VERSION
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
        UNSLOTH_ROCM_GFX_ARCH=gfx1033; export UNSLOTH_ROCM_GFX_ARCH
        _ARCH=x86_64
        . '$_E2E_FUNCS'
        get_torch_index_url" 2>/dev/null | tail -1
    rm -rf "$_np" )
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
