#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# get_torch_index_url routes a Linux x86_64 host whose only GPU is an XPU-capable Intel part to
# the xpu index, as install.ps1 does on Windows; every other host keeps its index.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
ROOT="$SCRIPT_DIR/../.."
INSTALL_SH="${INSTALL_SH:-$ROOT/install.sh}"
HARDWARE_PY="$ROOT/studio/backend/utils/hardware/hardware.py"
STACK_PY="$ROOT/studio/install_python_stack.py"

_TMP=$(mktemp -d)
trap 'rm -rf "${_TMP:?}"' EXIT
_PCI="$_TMP/pci"
_ABSENT="$_TMP/absent"
_FUNC_FILE="$_TMP/funcs.sh"

# get_torch_index_url and the helpers it reaches, as tests/sh/test_get_torch_index_url.sh lists
# them, with the host paths the probes read redirected.
_FUNCS="_run_bounded _cvd_hides_nvidia _has_amd_rocm_gpu _has_usable_nvidia_gpu
    _rocm_torch_explicitly_requested _ensure_rocm_probe_env _probe_amd_gfx_arch
    _amd_gpu_present_via_pci _infer_amd_gfx_arch_from_gpu_name _infer_linux_amd_gfx_arch
    _amd_arch_index_family_for_gfx _amd_probe_arches _amd_agreed_index_family _amd_sole_index_arch
    _trim_index_path_slashes _nvidia_library_inventory _nvidia_driver_cuda_version
    _nvidia_cu126_verdict _cap_cuda_family_for_pre_turing _rocm_tag_from_amd_smi
    _rocm_tag_from_version_file _rocm_tag_from_hipconfig _rocm_tag_from_dpkg _rocm_tag_from_rpm
    _highest_rocm_tag _detect_rocm_version_tag _amd_hardware_corroborated _pci_devices_root
    _intel_xpu_gpu_id _intel_xpu_auto_gpu_id get_torch_index_url"
for _fn in $_FUNCS; do
    awk -v fn="$_fn" '
        !body && index($0, fn "()") == 1 { body = 1; print; if ($0 ~ /\}[[:space:]]*$/) body = 0; next }
        body { print; if ($0 ~ /^\}/) body = 0 }
    ' "$INSTALL_SH"
    echo
done | sed -e "s|/sys/bus/pci/devices|$_PCI|g" \
    -e "s|/usr/bin/nvidia-smi|$_ABSENT/nvidia-smi|g" \
    -e "s|/opt/rocm|$_ABSENT/rocm|g" \
    -e "s|/proc/driver/nvidia|$_ABSENT/proc-nvidia|g" \
    -e "s|/sys/class/kfd|$_ABSENT/kfd|g" \
    -e "s|/dev/kfd|$_ABSENT/dev-kfd|g" \
    -e "s|/dev/dxg|$_ABSENT/dxg|g" > "$_FUNC_FILE"

_TOOLS="$_TMP/tools"
mkdir -p "$_TOOLS"
for _cmd in grep sed head sh bash cat awk printf tr; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS/$_cmd"
done

# $1 = machine (x86_64 / aarch64).
make_uname() {
    cat > "$_TOOLS/uname" <<EOF
#!/bin/sh
case "\$1" in -m) echo $1 ;; *) echo Linux ;; esac
EOF
    chmod +x "$_TOOLS/uname"
}

# add_pci <slot> <vendor> <device> <class>
add_pci() {
    mkdir -p "$_PCI/$1"
    echo "$2" > "$_PCI/$1/vendor"
    echo "$3" > "$_PCI/$1/device"
    echo "$4" > "$_PCI/$1/class"
}

_NV="$_TMP/nv"
mkdir -p "$_NV"
cat > "$_NV/nvidia-smi" <<'EOF'
#!/bin/sh
case "$1" in
    -L) echo "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-fake)" ;;
    --query-gpu=compute_cap) echo 8.6 ;;
    *) echo "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.8 |" ;;
esac
EOF
chmod +x "$_NV/nvidia-smi"

# run <extra PATH dir or ""> <env assignments...>
run() {
    _extra="$1"; shift
    env -i HOME="$_TMP" PATH="${_extra:+$_extra:}$_TOOLS" "$@" \
        bash -c ". '$_FUNC_FILE'; _ARCH=x86_64; get_torch_index_url" 2>/dev/null
}

cell() {
    _label="$1"; _want="$2"; shift 2
    assert_eq "$_label" "https://download.pytorch.org/whl/$_want" "$(run "$@")"
}

echo "=== test_install_intel_xpu_auto_route ==="
make_uname x86_64

rm -rf "$_PCI"; mkdir -p "$_PCI"
cell "no GPU -> cpu" cpu ""

for _id in 0x56a0 0xe20b 0x64a0 0x7d51 0xb084 0xb087 0x0bd5 0x0bd0 0x0b69 0x0b6e; do
    rm -rf "$_PCI"; add_pci 0000:03:00.0 0x8086 "$_id" 0x030000
    cell "Intel $_id -> xpu" xpu ""
done

rm -rf "$_PCI"; add_pci 0000:00:02.0 0x8086 0x46a6 0x030000
cell "non-Arc Intel iGPU 0x46a6 -> cpu" cpu ""

# Cedar Trail (gma500) sits next to PVC; the old 0x0B69-0x0BE5 range took it.
for _id in 0x0be0 0x0be5; do
    rm -rf "$_PCI"; add_pci 0000:00:02.0 0x8086 "$_id" 0x030000
    cell "Cedar Trail $_id -> cpu" cpu ""
done

rm -rf "$_PCI"; add_pci 0000:00:1f.3 0x8086 0x56a0 0x040300
cell "Intel id on a non-display function -> cpu" cpu ""

rm -rf "$_PCI"; add_pci 0000:03:00.0 0x8086 0x56a0 0x030000; add_pci 0000:0b:00.0 0x1002 0x744c 0x030000
cell "Arc beside AMD silicon -> cpu (AMD reroute owns it)" cpu ""

rm -rf "$_PCI"; add_pci 0000:03:00.0 0x8086 0x56a0 0x030000
cell "Arc + NVIDIA -> cu128" cu128 "$_NV"
cell "Arc + family pin cpu -> cpu" cpu "" UNSLOTH_TORCH_INDEX_FAMILY=cpu
cell "Arc + opt-out -> cpu" cpu "" UNSLOTH_DISABLE_XPU_AUTO=1
cell "Arc + declared AMD arch -> cpu" cpu "" UNSLOTH_ROCM_GFX_ARCH=gfx1100
cell "Arc + emptied ZE_AFFINITY_MASK -> xpu (Level Zero reads it as unset)" xpu "" ZE_AFFINITY_MASK=
cell "Arc + ZE_AFFINITY_MASK=default -> xpu" xpu "" ZE_AFFINITY_MASK=default
cell "Arc + ZE_AFFINITY_MASK=-1 -> cpu" cpu "" ZE_AFFINITY_MASK=-1
cell "Arc + ZE_AFFINITY_MASK=0 -> cpu (indices need not follow PCI order)" cpu "" ZE_AFFINITY_MASK=0
cell "Arc + ZE_AFFINITY_MASK=0 + xpu pin -> xpu" xpu "" ZE_AFFINITY_MASK=0 UNSLOTH_TORCH_INDEX_FAMILY=xpu
_mask_info=$(env -i HOME="$_TMP" PATH="$_TOOLS" ZE_AFFINITY_MASK=0 bash -c ". '$_FUNC_FILE'; _ARCH=x86_64; get_torch_index_url" 2>&1 >/dev/null)
assert_contains "a set mask names the pin to use" "$_mask_info" "UNSLOTH_TORCH_INDEX_FAMILY=xpu"
cell "Arc + emptied mask + xpu pin -> xpu" xpu "" ZE_AFFINITY_MASK= UNSLOTH_TORCH_INDEX_FAMILY=xpu
cell "Arc + ONEAPI_DEVICE_SELECTOR -> cpu" cpu "" ONEAPI_DEVICE_SELECTOR=level_zero:0
cell "Arc + SYCL_DEVICE_FILTER -> cpu" cpu "" SYCL_DEVICE_FILTER=level_zero:gpu:0
cell "Arc + SYCL_DEVICE_ALLOWLIST -> cpu" cpu "" SYCL_DEVICE_ALLOWLIST=DeviceType:cpu
cell "Arc + ONEAPI_DEVICE_SELECTOR + xpu pin -> xpu" xpu "" ONEAPI_DEVICE_SELECTOR=level_zero:0 UNSLOTH_TORCH_INDEX_FAMILY=xpu
_sel_info=$(env -i HOME="$_TMP" PATH="$_TOOLS" ONEAPI_DEVICE_SELECTOR=level_zero:0 bash -c ". '$_FUNC_FILE'; _ARCH=x86_64; get_torch_index_url" 2>&1 >/dev/null)
assert_contains "a SYCL selector is named with the pin to use" "$_sel_info" "ONEAPI_DEVICE_SELECTOR is set -- skipping the Intel XPU auto route; set UNSLOTH_TORCH_INDEX_FAMILY=xpu"
_info=$(env -i HOME="$_TMP" PATH="$_TOOLS" bash -c ". '$_FUNC_FILE'; _ARCH=x86_64; get_torch_index_url" 2>&1 >/dev/null)
assert_contains "route prints the device and the opt-out" "$_info" "Intel GPU (0x56a0) detected"
assert_contains "route names UNSLOTH_DISABLE_XPU_AUTO" "$_info" "UNSLOTH_DISABLE_XPU_AUTO=1"

make_uname aarch64
cell "Arc on aarch64 -> cpu" cpu ""
make_uname x86_64

# The opt-out on an Arc host records a deliberate CPU backend, not install.sh's resolved answer.
_SRC_FILE="$_TMP/source.sh"
awk '/^# Derived from the index this script RESOLVED/ { on = 1 } on { print } on && /^    unset UNSLOTH_TORCH_BACKEND_SOURCE/ { last = 1; next } last && /^fi/ { exit }' \
    "$INSTALL_SH" > "$_SRC_FILE"
# source_of <env assignments...>: UNSLOTH_TORCH_BACKEND_SOURCE after the block for a resolved cpu.
source_of() {
    env -i HOME="$_TMP" PATH="$_TOOLS" "$@" bash -c ". '$_FUNC_FILE'; _ARCH=x86_64
        _torch_backend_was_stated=false; _torch_backend_stated_value=''; UNSLOTH_TORCH_BACKEND=cpu
        . '$_SRC_FILE'; printf '%s' \"\${UNSLOTH_TORCH_BACKEND_SOURCE:-unset}\"" 2>/dev/null
}
rm -rf "$_PCI"; add_pci 0000:03:00.0 0x8086 0x56a0 0x030000
assert_eq "Arc + opt-out records a deliberate cpu" "unset" "$(source_of UNSLOTH_DISABLE_XPU_AUTO=1)"
assert_eq "Arc + opt-out + SYCL selector still records a deliberate cpu" "unset" "$(source_of UNSLOTH_DISABLE_XPU_AUTO=1 ONEAPI_DEVICE_SELECTOR=level_zero:0)"
assert_eq "Arc without the opt-out keeps resolved" "resolved" "$(source_of)"
rm -rf "$_PCI"; add_pci 0000:00:02.0 0x8086 0x46a6 0x030000
assert_eq "opt-out without an XPU-capable GPU keeps resolved" "resolved" "$(source_of UNSLOTH_DISABLE_XPU_AUTO=1)"

# One allowlist in three places: hardware.py, install_python_stack.py (update-time revalidation) and
# the shell. The Python tables must be equal; the shell is probed on every id and bound neighbour.
if command -v python3 >/dev/null 2>&1 && grep -q "^_intel_xpu_gpu_id()" "$_FUNC_FILE"; then
    _pairs=$(python3 - "$HARDWARE_PY" "$STACK_PY" <<'PY'
import ast, sys
def tables(path):
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_INTEL_XPU_PCI_ID_RANGES", "_INTEL_XPU_PCI_IDS"):
                out[node.targets[0].id] = ast.literal_eval(node.value.args[0] if isinstance(node.value, ast.Call) else node.value)
    return out
vals = tables(sys.argv[1])
stack = tables(sys.argv[2])
if {k: set(v) for k, v in stack.items()} != {k: set(v) for k, v in vals.items()}:
    print("STACK_DIFFERS")
ids = set(vals["_INTEL_XPU_PCI_IDS"])
for lo, hi in vals["_INTEL_XPU_PCI_ID_RANGES"]:
    ids.update((lo, hi, (lo + hi) // 2))
probe = {i: True for i in ids}
for lo, hi in vals["_INTEL_XPU_PCI_ID_RANGES"]:
    for edge in (lo - 1, hi + 1):
        probe.setdefault(edge, any(a <= edge <= b for a, b in vals["_INTEL_XPU_PCI_ID_RANGES"]) or edge in vals["_INTEL_XPU_PCI_IDS"])
for i in sorted(probe):
    print(f"0x{i:04x} {'yes' if probe[i] else 'no'}")
PY
)
    _mismatch=""
    case "$_pairs" in *STACK_DIFFERS*) _mismatch=" install_python_stack.py" ;; esac
    _pairs=$(printf '%s\n' "$_pairs" | grep -v STACK_DIFFERS)
    while read -r _id _want; do
        rm -rf "$_PCI"; add_pci 0000:03:00.0 0x8086 "$_id" 0x030000
        _got=$(env -i PATH="$_TOOLS" bash -c ". '$_FUNC_FILE'; _intel_xpu_gpu_id >/dev/null && echo yes || echo no")
        [ "$_got" = "$_want" ] || _mismatch="$_mismatch $_id"
    done <<EOF
$_pairs
EOF
    assert_eq "shell and install_python_stack.py allowlists match hardware.py ($(printf '%s\n' "$_pairs" | wc -l | tr -d ' ') ids)" "" "$_mismatch"
fi

summary
