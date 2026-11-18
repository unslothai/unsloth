"""setup.sh's llama.cpp source build follows ROCm torch on a mixed NVIDIA+AMD host, by the same
rule (install_llama_prebuilt.py --follows-rocm-torch) the prebuilt path applies."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STUDIO = REPO / "studio"
SETUP_SH = STUDIO / "setup.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "bash not available")


def _between(src, start, end):
    i = src.index(start)
    return src[i : src.index(end, i)]


def _fake_torch(root, rocm):
    pkg = root / "torch"
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text("")
    (pkg / "version.py").write_text(
        '__version__ = "2.9.0+rocm6.4"\nhip = "6.4.0"\n'
        if rocm
        else '__version__ = "2.9.0+cu128"\nhip = None\n'
    )


def _build_backend(
    tmp_path,
    *,
    nvidia,
    amd,
    torch = "cuda",
    hipcc = True,
    env = None,
):
    """Run the GPU probe gate and the source-build backend choice for real, with the probes
    pinned and a fake torch on PYTHONPATH; prints the backend the build would use."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    probe = _between(
        src,
        "# Skip AMD probes on a usable-NVIDIA host",
        '\nif [ "$_setup_nvidia_usable" = true ]; then\n    _setup_nv_banner_fields',
    )
    build = _between(src, "_select_nvcc() {", '_BUILD_DESC="building"')
    torch_root = tmp_path / "torch_root"
    _fake_torch(torch_root, torch == "rocm")
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f'SCRIPT_DIR="{STUDIO}"\n'
        f'python() {{ "{sys.executable}" "$@"; }}\n'
        f"_setup_has_physical_nvidia_gpu() {{ {str(nvidia).lower()}; }}\n"
        f"_setup_has_usable_nvidia_gpu() {{ {str(nvidia).lower()}; }}\n"
        f'command() {{ case "$2" in rocminfo) {str(amd).lower()} ;; nvcc) true ;; '
        f"hipcc) {str(hipcc).lower()} ;; *) return 1 ;; esac; }}\n"
        "_setup_run_smi() { :; }\n"
        "_setup_rocminfo_gpu_records() { echo 'gfx1100|AMD Radeon RX 7900 XTX'; }\n"
        "ls() { return 1; }\n"
        "_setup_amd_detected=false\n_setup_nvidia_usable=false\n_setup_nvidia_physical=false\n"
        '_setup_gfx_all=""\n_setup_amd_records=""\n_setup_amd_probe=""\n'
        f"{probe}\n"
        'GPU_BACKEND=""\nNVCC_PATH=""\nROCM_HIPCC=""\n'
        f"{build}\n"
        'echo "${GPU_BACKEND:-cpu}"\n'
    )
    run_env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(torch_root), **(env or {})}
    result = subprocess.run(
        ["bash", str(script)], capture_output = True, text = True, timeout = 60, env = run_env
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


FORCE = {"UNSLOTH_FORCE_ROCM_TORCH": "1"}


def test_mixed_host_with_force_builds_hip(tmp_path):
    assert _build_backend(tmp_path, nvidia = True, amd = True, env = FORCE) == "rocm"


def test_mixed_host_with_rocm_torch_installed_builds_hip(tmp_path):
    assert _build_backend(tmp_path, nvidia = True, amd = True, torch = "rocm") == "rocm"


@pytest.mark.parametrize("value", ["0", "", "no"])
def test_mixed_host_without_force_builds_cuda(tmp_path, value):
    env = {"UNSLOTH_FORCE_ROCM_TORCH": value} if value != "" else {}
    assert _build_backend(tmp_path, nvidia = True, amd = True, env = env) == "cuda"


@pytest.mark.parametrize(
    "extra",
    [
        {"UNSLOTH_LLAMA_CPP_BACKEND": "cuda"},
        {"UNSLOTH_LLAMA_CPP_BACKEND": "vulkan"},
        {"UNSLOTH_FORCE_VULKAN": "1"},
    ],
)
def test_an_explicit_backend_keeps_the_rocm_torch_rule_out(tmp_path, extra):
    assert _build_backend(tmp_path, nvidia = True, amd = True, env = {**FORCE, **extra}) == "cuda"


def test_auto_backend_still_follows_rocm_torch(tmp_path):
    env = {**FORCE, "UNSLOTH_LLAMA_CPP_BACKEND": "auto"}
    assert _build_backend(tmp_path, nvidia = True, amd = True, env = env) == "rocm"


def test_mixed_host_without_hipcc_falls_back_to_cuda_not_cpu(tmp_path):
    assert _build_backend(tmp_path, nvidia = True, amd = True, hipcc = False, env = FORCE) == "cuda"


@pytest.mark.parametrize(
    "nvidia, amd, expected",
    [
        (True, False, "cuda"),  # stale ROCm torch on an NVIDIA-only host keeps CUDA
        (False, True, "rocm"),
        (False, False, "cpu"),
    ],
)
@pytest.mark.parametrize("env", [{}, FORCE])
def test_single_vendor_hosts_are_unchanged(tmp_path, nvidia, amd, expected, env):
    assert _build_backend(tmp_path, nvidia = nvidia, amd = amd, env = env) == expected
