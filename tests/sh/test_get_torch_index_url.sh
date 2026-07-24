#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for get_torch_index_url() from install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract get_torch_index_url and its helper functions from install.sh.
# Also replace the hardcoded /usr/bin/nvidia-smi fallback with a
# controllable path so we can test the "no GPU" scenario on GPU machines.
_FUNC_FILE=$(mktemp)
_FAKE_SMI_DIR=$(mktemp -d)
{
    sed -n '/^_run_bounded()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_cvd_hides_nvidia()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_has_amd_rocm_gpu()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_has_usable_nvidia_gpu()/,/^}/p' "$INSTALL_SH"
    echo ""
    # ROCm gfx-arch probe helpers that get_torch_index_url / _has_amd_rocm_gpu
    # now call. These MUST stay in sync with install.sh: if get_torch_index_url
    # references a helper that is not extracted here, the ROCm branch hits an
    # undefined function, silently falls through to the CPU wheel index, and the
    # ROCm assertions below fail.
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
    echo ""
    sed -n '/^_trim_index_path_slashes()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^get_torch_index_url()/,/^}/p' "$INSTALL_SH"
} | sed "s|/usr/bin/nvidia-smi|$_FAKE_SMI_DIR/nvidia-smi-absent|g" \
  > "$_FUNC_FILE"

# Save system PATH so we always have basic tools (uname, grep, head, etc.)
_SYS_PATH="/usr/local/bin:/usr/bin:/bin"

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

# Helper: create a mock nvidia-smi that prints a given CUDA version string.
# Handles both default output (version header) and -L (GPU listing) so that
# _has_usable_nvidia_gpu sees a valid GPU.
make_mock_smi() {
    _dir=$(mktemp -d)
    cat > "$_dir/nvidia-smi" <<MOCK
#!/bin/sh
case "\$1" in
    -L)
        echo "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-fake-uuid)"
        ;;
    *)
        cat <<'SMI_OUT'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: $1     |
+-----------------------------------------------------------------------------------------+
SMI_OUT
        ;;
esac
MOCK
    chmod +x "$_dir/nvidia-smi"
    echo "$_dir"
}

# Helper: create a mock nvidia-smi that prints the new "CUDA UMD Version" header
# layout used by newer NVIDIA drivers (e.g. 610.x on Windows).  See issue #5812.
make_mock_smi_umd() {
    _dir=$(mktemp -d)
    cat > "$_dir/nvidia-smi" <<MOCK
#!/bin/sh
case "\$1" in
    -L)
        echo "GPU 0: NVIDIA GeForce RTX 5090 Laptop GPU (UUID: GPU-fake-uuid)"
        ;;
    *)
        cat <<'SMI_OUT'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.47                 KMD Version: 610.47        CUDA UMD Version: $1     |
+-----------------------------------------------------------------------------------------+
SMI_OUT
        ;;
esac
MOCK
    chmod +x "$_dir/nvidia-smi"
    echo "$_dir"
}

# Helper: create a mock amd-smi that prints a given ROCm version string
# Supports both "amd-smi version" and "amd-smi list" subcommands so that
# the GPU presence check (amd-smi list) also succeeds in tests.
make_mock_amd_smi() {
    _dir=$(mktemp -d)
    cat > "$_dir/amd-smi" <<MOCK
#!/bin/sh
case "\$1" in
    list)
        printf 'GPU: 0\\n  BDF: 0000:03:00.0\\n  NAME: gfx1100\\n'
        ;;
    *)
        cat <<AMD_OUT
AMDSMI Tool: 25.0.1+2b74356 | AMDSMI Library version: 25.0.1.0 | ROCm version: $1
AMD_OUT
        ;;
esac
MOCK
    chmod +x "$_dir/amd-smi"
    echo "$_dir"
}

# Build a minimal tools directory with symlinks to essential commands
# (uname, grep, head, etc.) but WITHOUT nvidia-smi or amd-smi.
_TOOLS_DIR=$(mktemp -d)
for _cmd in uname grep sed head sh bash cat awk printf tr; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

# Helper: run get_torch_index_url with a custom PATH
# $1 = directory with mock nvidia-smi (prepended to PATH), or "none" for no-GPU test
run_func() {
    _mock_dir="$1"
    # Default: strip CUDA_VISIBLE_DEVICES so the host environment cannot leak
    # in; a second argument sets it explicitly (hidden-GPU scenarios).
    if [ "$#" -ge 2 ]; then
        _cvd_setup="export CUDA_VISIBLE_DEVICES='$2'"
    else
        _cvd_setup="unset CUDA_VISIBLE_DEVICES"
    fi
    if [ "$_mock_dir" = "none" ]; then
        # Minimal PATH with only basic tools, no nvidia-smi anywhere
        PATH="$_TOOLS_DIR" bash -c "$_cvd_setup; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
    else
        # Put mock nvidia-smi dir first, then basic tools
        PATH="$_mock_dir:$_TOOLS_DIR" bash -c "$_cvd_setup; . '$_FUNC_FILE'; get_torch_index_url" 2>/dev/null
    fi
}

echo "=== test_get_torch_index_url ==="

# 1) No nvidia-smi available -> cpu
_result=$(run_func "none")
assert_eq "no nvidia-smi -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"

# 2) CUDA 12.6 -> cu126
_dir=$(make_mock_smi "12.6")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.6 -> cu126" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_dir"

# 3) CUDA 12.8 -> cu128
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.8 -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 4) CUDA 13.0 -> cu130
_dir=$(make_mock_smi "13.0")
_result=$(run_func "$_dir")
assert_eq "CUDA 13.0 -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 5) CUDA 12.4 -> cu124
_dir=$(make_mock_smi "12.4")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.4 -> cu124" "https://download.pytorch.org/whl/cu124" "$_result"
rm -rf "$_dir"

# 6) CUDA 11.8 -> cu118
_dir=$(make_mock_smi "11.8")
_result=$(run_func "$_dir")
assert_eq "CUDA 11.8 -> cu118" "https://download.pytorch.org/whl/cu118" "$_result"
rm -rf "$_dir"

# 7) CUDA 10.2 (too old) -> cpu
_dir=$(make_mock_smi "10.2")
_result=$(run_func "$_dir")
assert_eq "CUDA 10.2 -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 8) Unparseable nvidia-smi version but valid GPU listing -> cu126 default
_dir=$(mktemp -d)
cat > "$_dir/nvidia-smi" <<'MOCK'
#!/bin/sh
case "$1" in
    -L) echo "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-fake-uuid)" ;;
    *)  echo "something completely unexpected" ;;
esac
MOCK
chmod +x "$_dir/nvidia-smi"
_result=$(run_func "$_dir")
assert_eq "unparseable -> cu126" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_dir"

# 9) ROCm 6.3 (no nvidia-smi) -> rocm6.3
_dir=$(make_mock_amd_smi "6.3")
_result=$(run_func "$_dir")
assert_eq "ROCm 6.3 -> rocm6.3" "https://download.pytorch.org/whl/rocm6.3" "$_result"
rm -rf "$_dir"

# 10) ROCm 7.1 (no nvidia-smi) -> rocm7.1
_dir=$(make_mock_amd_smi "7.1")
_result=$(run_func "$_dir")
assert_eq "ROCm 7.1 -> rocm7.1" "https://download.pytorch.org/whl/rocm7.1" "$_result"
rm -rf "$_dir"

# 11) ROCm 7.2 (no nvidia-smi) -> rocm7.2
_dir=$(make_mock_amd_smi "7.2")
_result=$(run_func "$_dir")
assert_eq "ROCm 7.2 -> rocm7.2" "https://download.pytorch.org/whl/rocm7.2" "$_result"
rm -rf "$_dir"

# 12) Both nvidia-smi and amd-smi present -> CUDA takes precedence
_cuda_dir=$(make_mock_smi "12.6")
_amd_dir=$(make_mock_amd_smi "6.3")
_combined_dir=$(mktemp -d)
ln -sf "$_cuda_dir/nvidia-smi" "$_combined_dir/nvidia-smi"
ln -sf "$_amd_dir/amd-smi" "$_combined_dir/amd-smi"
_result=$(run_func "$_combined_dir")
assert_eq "CUDA+ROCm -> CUDA precedence" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_cuda_dir" "$_amd_dir" "$_combined_dir"

# 13) No nvidia-smi, no amd-smi -> cpu (duplicate of test 1, confirms ROCm didn't break it)
_result=$(run_func "none")
assert_eq "no GPU -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"

# 14) ROCm 6.1 (no nvidia-smi) -> rocm6.1
_dir=$(make_mock_amd_smi "6.1")
_result=$(run_func "$_dir")
assert_eq "ROCm 6.1 -> rocm6.1" "https://download.pytorch.org/whl/rocm6.1" "$_result"
rm -rf "$_dir"

# 15) ROCm 6.4 (no nvidia-smi) -> rocm6.4
_dir=$(make_mock_amd_smi "6.4")
_result=$(run_func "$_dir")
assert_eq "ROCm 6.4 -> rocm6.4" "https://download.pytorch.org/whl/rocm6.4" "$_result"
rm -rf "$_dir"

# 16) ROCm 7.0 (no nvidia-smi) -> rocm7.0
_dir=$(make_mock_amd_smi "7.0")
_result=$(run_func "$_dir")
assert_eq "ROCm 7.0 -> rocm7.0" "https://download.pytorch.org/whl/rocm7.0" "$_result"
rm -rf "$_dir"

# 17) ROCm 8.0 (future, no nvidia-smi) -> rocm7.2 (capped to latest known)
_dir=$(make_mock_amd_smi "8.0")
_result=$(run_func "$_dir")
assert_eq "ROCm 8.0 -> rocm7.2 (capped)" "https://download.pytorch.org/whl/rocm7.2" "$_result"
rm -rf "$_dir"

# 18) Malformed amd-smi output (empty version field) -> cpu
_dir=$(mktemp -d)
cat > "$_dir/amd-smi" <<'MOCK'
#!/bin/sh
echo "AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | ROCm version: "
MOCK
chmod +x "$_dir/amd-smi"
_result=$(run_func "$_dir")
assert_eq "empty amd-smi version -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 19) amd-smi with "N/A" version -> cpu
_dir=$(mktemp -d)
cat > "$_dir/amd-smi" <<'MOCK'
#!/bin/sh
echo "AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | ROCm version: N/A"
MOCK
chmod +x "$_dir/amd-smi"
_result=$(run_func "$_dir")
assert_eq "N/A amd-smi version -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 20) ROCm version with trailing text (e.g. "6.3.1-beta") -> rocm6.3
_dir=$(make_mock_amd_smi "6.3.1-beta")
_result=$(run_func "$_dir")
assert_eq "ROCm 6.3.1-beta -> rocm6.3" "https://download.pytorch.org/whl/rocm6.3" "$_result"
rm -rf "$_dir"

# 22) CUDA 12.6 still works after ROCm changes (regression check)
_dir=$(make_mock_smi "12.6")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.6 regression -> cu126" "https://download.pytorch.org/whl/cu126" "$_result"
rm -rf "$_dir"

# 23) CUDA 13.0 still works after ROCm changes (regression check)
_dir=$(make_mock_smi "13.0")
_result=$(run_func "$_dir")
assert_eq "CUDA 13.0 regression -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 24) CUDA 12.8 still works after ROCm changes (regression check)
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir")
assert_eq "CUDA 12.8 regression -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 25) UNSLOTH_PYTORCH_MIRROR overrides base URL (CUDA case)
_dir=$(make_mock_smi "12.6")
_result=$(UNSLOTH_PYTORCH_MIRROR="https://mirror.example.com/whl" run_func "$_dir")
assert_eq "mirror env + CUDA 12.6 -> mirror/cu126" "https://mirror.example.com/whl/cu126" "$_result"
rm -rf "$_dir"

# 26) UNSLOTH_PYTORCH_MIRROR overrides base URL (CPU case)
_result=$(UNSLOTH_PYTORCH_MIRROR="https://mirror.example.com/whl" run_func "none")
assert_eq "mirror env + no GPU -> mirror/cpu" "https://mirror.example.com/whl/cpu" "$_result"

# 27) Empty UNSLOTH_PYTORCH_MIRROR falls back to official URL
_result=$(UNSLOTH_PYTORCH_MIRROR="" run_func "none")
assert_eq "empty mirror env -> official/cpu" "https://download.pytorch.org/whl/cpu" "$_result"

# 28) Trailing slash in UNSLOTH_PYTORCH_MIRROR is stripped
_result=$(UNSLOTH_PYTORCH_MIRROR="https://mirror.example.com/whl/" run_func "none")
assert_eq "trailing slash stripped -> mirror/cpu" "https://mirror.example.com/whl/cpu" "$_result"

# 29) "CUDA UMD Version: 13.3" header (newer NVIDIA driver layout, issue #5812)
#     -> cu130, not the cu126 fallback.
_dir=$(make_mock_smi_umd "13.3")
_result=$(run_func "$_dir")
assert_eq "CUDA UMD Version 13.3 -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 30) "CUDA UMD Version: 12.8" header (newer layout on a 12.x driver) -> cu128
_dir=$(make_mock_smi_umd "12.8")
_result=$(run_func "$_dir")
assert_eq "CUDA UMD Version 12.8 -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 31) "CUDA UMD Version: 11.8" header (newer layout on an older driver) -> cu118
_dir=$(make_mock_smi_umd "11.8")
_result=$(run_func "$_dir")
assert_eq "CUDA UMD Version 11.8 -> cu118" "https://download.pytorch.org/whl/cu118" "$_result"
rm -rf "$_dir"

# 32) Driver-reported "CUDA Version: 13.3" (legacy header) -> cu130.
_dir=$(make_mock_smi "13.3")
_result=$(run_func "$_dir")
assert_eq "CUDA Version 13.3 -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 33) "CUDA Version: 13.7" -> cu130 (until a cu137 wheel index exists).
_dir=$(make_mock_smi "13.7")
_result=$(run_func "$_dir")
assert_eq "CUDA Version 13.7 -> cu130" "https://download.pytorch.org/whl/cu130" "$_result"
rm -rf "$_dir"

# 34) CUDA_VISIBLE_DEVICES="" hides the NVIDIA GPU -> cpu (no AMD present)
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir" "")
assert_eq "CVD='' hides NVIDIA -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 35) CUDA_VISIBLE_DEVICES=-1 hides the NVIDIA GPU -> cpu (no AMD present)
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir" "-1")
assert_eq "CVD=-1 hides NVIDIA -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# 36) Mixed AMD+NVIDIA host with NVIDIA hidden -> ROCm route is restored
_cuda_dir=$(make_mock_smi "12.6")
_amd_dir=$(make_mock_amd_smi "6.4")
_combined_dir=$(mktemp -d)
ln -sf "$_cuda_dir/nvidia-smi" "$_combined_dir/nvidia-smi"
ln -sf "$_amd_dir/amd-smi" "$_combined_dir/amd-smi"
_result=$(run_func "$_combined_dir" "-1")
assert_eq "CUDA+ROCm with CVD=-1 -> rocm6.4" "https://download.pytorch.org/whl/rocm6.4" "$_result"
rm -rf "$_cuda_dir" "$_amd_dir" "$_combined_dir"

# 37) CUDA_VISIBLE_DEVICES=0 (a visible device) must NOT hide the GPU
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir" "0")
assert_eq "CVD=0 keeps NVIDIA -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 38) Whitespace-padded "-1" still hides the GPU
_dir=$(make_mock_smi "12.8")
_result=$(run_func "$_dir" " -1 ")
assert_eq "CVD=' -1 ' hides NVIDIA -> cpu" "https://download.pytorch.org/whl/cpu" "$_result"
rm -rf "$_dir"

# --- explicit overrides (headless / container / CI; no GPU probing) ----------
# 39) UNSLOTH_TORCH_INDEX_FAMILY pins the family with no GPU present (not the cpu fallback).
_result=$(UNSLOTH_TORCH_INDEX_FAMILY="cu128" run_func "none")
assert_eq "family override (no GPU) -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"

# 40) Family override beats real detection: an nvidia-smi 12.6 host still gets cu128
#     (the Docker-build case -- builder sees the host driver but publishes a cu128 image).
_dir=$(make_mock_smi "12.6")
_result=$(UNSLOTH_TORCH_INDEX_FAMILY="cu128" run_func "$_dir")
assert_eq "family override beats detected 12.6 -> cu128" "https://download.pytorch.org/whl/cu128" "$_result"
rm -rf "$_dir"

# 41) UNSLOTH_TORCH_INDEX_URL is used verbatim and wins over detection.
_dir=$(make_mock_smi "12.6")
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu999" run_func "$_dir")
assert_eq "url override beats detection -> verbatim" "https://mirror.example.com/whl/cu999" "$_result"
rm -rf "$_dir"

# 42) Family override is appended to UNSLOTH_PYTORCH_MIRROR (mirror still honoured).
_result=$(UNSLOTH_PYTORCH_MIRROR="https://mirror.example.com/whl" UNSLOTH_TORCH_INDEX_FAMILY="cu128" run_func "none")
assert_eq "mirror + family override -> mirror/cu128" "https://mirror.example.com/whl/cu128" "$_result"

# 43) Trailing slash in UNSLOTH_TORCH_INDEX_URL is stripped.
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu128/" run_func "none")
assert_eq "url override trailing slash stripped" "https://mirror.example.com/whl/cu128" "$_result"

# 44) URL override takes precedence over family override.
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu130" UNSLOTH_TORCH_INDEX_FAMILY="cu128" run_func "none")
assert_eq "url override beats family override -> url" "https://mirror.example.com/whl/cu130" "$_result"

# 45) An empty override is ignored (falls through to normal detection).
_result=$(UNSLOTH_TORCH_INDEX_FAMILY="" UNSLOTH_TORCH_INDEX_URL="" run_func "none")
assert_eq "empty overrides ignored -> detected cpu" "https://download.pytorch.org/whl/cpu" "$_result"

# 46) ALL trailing slashes are stripped from a URL override (not just one).
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu128///" run_func "none")
assert_eq "url override double slash stripped" "https://mirror.example.com/whl/cu128" "$_result"

# 47) Leading and trailing slashes stripped from a family override.
_result=$(UNSLOTH_TORCH_INDEX_FAMILY="//cu128//" run_func "none")
assert_eq "family override slashes stripped" "https://download.pytorch.org/whl/cu128" "$_result"

# 48) A ?query token that ends in "/" is PRESERVED: only PATH slashes are trimmed, so a
# base64 token ending in "/" is not corrupted (path-only trim, not whole-URL rstrip).
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu128?token=ab12cd/" run_func "none")
assert_eq "url override preserves query token slash" "https://mirror.example.com/whl/cu128?token=ab12cd/" "$_result"

# 49) Double PATH slash before a query is collapsed while the query survives intact.
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu128//?token=ab12cd/" run_func "none")
assert_eq "url override path slash trimmed, query kept" "https://mirror.example.com/whl/cu128?token=ab12cd/" "$_result"

# 50) A #fragment ending in "/" is likewise preserved.
_result=$(UNSLOTH_TORCH_INDEX_URL="https://mirror.example.com/whl/cu128#anchor/" run_func "none")
assert_eq "url override preserves fragment slash" "https://mirror.example.com/whl/cu128#anchor/" "$_result"

rm -f "$_FUNC_FILE"
rm -rf "$_FAKE_SMI_DIR"
rm -rf "$_TOOLS_DIR"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
