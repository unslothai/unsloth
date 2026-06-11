"""Tests for the GPU-detection follow-ups to PR 6174.

PR 6174 made NVIDIA take precedence and added a /proc/driver/nvidia/gpus
fallback in install.sh and studio/install_python_stack.py. These tests cover the
same hardening ported to the llama.cpp prebuilt installer
(studio/install_llama_prebuilt.py) and the Studio shell setup (studio/setup.sh):

  * detect_host() recognises NVIDIA via /proc/driver/nvidia/gpus when nvidia-smi
    is unavailable, and skips ROCm probing when NVIDIA is usable.
  * setup.sh routes through a timeout-bounded NVIDIA probe with a /proc fallback
    and only selects a CUDA/ROCm source build when the matching GPU is detected.

All tests use mocks or source-level assertions -- no GPU, network, or real
nvidia-smi/rocminfo invocation.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# Load studio/install_llama_prebuilt.py the same way the sibling suite does.
_MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt_followups", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
prebuilt_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prebuilt_mod
_SPEC.loader.exec_module(prebuilt_mod)

detect_host = prebuilt_mod.detect_host
_apply_host_overrides = prebuilt_mod._apply_host_overrides

SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"


def _make_run_capture(rocminfo_stdout: str = ""):
    """Return a fake run_capture: rocminfo reports rocminfo_stdout, everything
    else (nvidia-smi, amd-smi) returns empty so only the patched probes matter."""

    def _run_capture(cmd, *args, **kwargs):
        exe = str(cmd[0]) if cmd else ""
        result = MagicMock()
        if exe.endswith("rocminfo"):
            result.returncode = 0
            result.stdout = rocminfo_stdout
        else:
            result.returncode = 1
            result.stdout = ""
        result.stderr = ""
        return result

    return _run_capture


def _run_detect_host(
    *,
    machine: str = "x86_64",
    system: str = "Linux",
    which_map: dict | None = None,
    proc_dir_entries: list | None = None,
    rocminfo_stdout: str = "",
    env: dict | None = None,
):
    """Drive detect_host() against a fully synthetic host."""
    which_map = which_map or {}
    proc_dir_entries = proc_dir_entries if proc_dir_entries is not None else []

    real_isdir = prebuilt_mod.os.path.isdir
    real_listdir = prebuilt_mod.os.listdir
    proc_path = "/proc/driver/nvidia/gpus"

    def fake_isdir(p):
        if str(p) == proc_path:
            return bool(proc_dir_entries)
        return real_isdir(p)

    def fake_listdir(p):
        if str(p) == proc_path:
            if not proc_dir_entries:
                raise OSError("no such dir")
            return list(proc_dir_entries)
        return real_listdir(p)

    patches = [
        patch.object(prebuilt_mod.platform, "system", return_value=system),
        patch.object(prebuilt_mod.platform, "machine", return_value=machine),
        patch.object(prebuilt_mod.platform, "mac_ver", return_value=("", ("", "", ""), "")),
        patch.object(prebuilt_mod.shutil, "which", side_effect=lambda n: which_map.get(n)),
        patch.object(prebuilt_mod, "run_capture", side_effect=_make_run_capture(rocminfo_stdout)),
        patch.object(prebuilt_mod.os.path, "isdir", side_effect=fake_isdir),
        patch.object(prebuilt_mod.os, "listdir", side_effect=fake_listdir),
        patch.object(prebuilt_mod.os, "access", return_value=False),
        patch.dict(prebuilt_mod.os.environ, env or {}, clear=False),
    ]
    for p in patches:
        p.start()
    try:
        # Ensure CUDA_VISIBLE_DEVICES does not leak in from the test host unless
        # the scenario sets it explicitly.
        if env is None or "CUDA_VISIBLE_DEVICES" not in env:
            prebuilt_mod.os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return detect_host()
    finally:
        for p in patches:
            p.stop()


# ── install_llama_prebuilt.detect_host(): /proc NVIDIA fallback ──────────────


class TestDetectHostProcFallback:
    def test_proc_fallback_marks_physical_nvidia_when_smi_absent(self):
        """No nvidia-smi, but /proc/driver/nvidia/gpus is populated -> NVIDIA."""
        host = _run_detect_host(
            which_map={},  # nvidia-smi resolves to None
            proc_dir_entries=["0000:01:00.0"],
        )
        assert host.has_physical_nvidia is True

    def test_proc_fallback_has_usable_nvidia_when_devices_visible(self):
        """Default CUDA_VISIBLE_DEVICES (unset) -> visible tokens non-empty -> usable."""
        host = _run_detect_host(
            which_map={},
            proc_dir_entries=["0000:01:00.0"],
        )
        assert host.has_usable_nvidia is True

    def test_proc_fallback_not_usable_when_devices_hidden(self):
        """CUDA_VISIBLE_DEVICES='' hides all GPUs -> physical yes, usable no."""
        host = _run_detect_host(
            which_map={},
            proc_dir_entries=["0000:01:00.0"],
            env={"CUDA_VISIBLE_DEVICES": ""},
        )
        assert host.has_physical_nvidia is True
        assert host.has_usable_nvidia is False

    def test_empty_proc_dir_does_not_mark_nvidia(self):
        """A driver dir that exists but is empty must not assert a GPU."""
        host = _run_detect_host(which_map={}, proc_dir_entries=[])
        assert host.has_physical_nvidia is False

    def test_proc_fallback_is_linux_only(self):
        """The /proc fallback must not run on Windows (path is Linux-only)."""
        host = _run_detect_host(
            system="Windows",
            machine="amd64",
            which_map={},
            proc_dir_entries=["0000:01:00.0"],
        )
        assert host.has_physical_nvidia is False


# ── install_llama_prebuilt.detect_host(): NVIDIA precedence over ROCm ────────


class TestDetectHostNvidiaPrecedence:
    def test_rocm_probe_skipped_when_proc_nvidia_present(self):
        """rocminfo reports gfx1100, but a proc-detected NVIDIA GPU wins."""
        host = _run_detect_host(
            which_map={"rocminfo": "/usr/bin/rocminfo"},
            proc_dir_entries=["0000:01:00.0"],
            rocminfo_stdout="  Name:                    gfx1100\n",
        )
        assert host.has_usable_nvidia is True
        assert host.has_rocm is False

    def test_rocm_detected_when_no_nvidia(self):
        """With no NVIDIA signal at all, rocminfo gfx1100 -> has_rocm True."""
        host = _run_detect_host(
            which_map={"rocminfo": "/usr/bin/rocminfo"},
            proc_dir_entries=[],
            rocminfo_stdout="  Name:                    gfx1100\n",
        )
        assert host.has_usable_nvidia is False
        assert host.has_rocm is True


# ── _apply_host_overrides: forwarded --rocm-gfx / --has-rocm still win ───────


class TestOverridesStillWin:
    def test_forwarded_gfx_forces_rocm_on_non_nvidia_host(self):
        host = _run_detect_host(which_map={}, proc_dir_entries=[])
        assert host.has_rocm is False
        overridden = _apply_host_overrides(host, override_rocm_gfx="gfx1100")
        assert overridden.has_rocm is True
        assert overridden.rocm_gfx_target == "gfx1100"

    def test_override_has_rocm_forces_rocm(self):
        host = _run_detect_host(which_map={}, proc_dir_entries=[])
        overridden = _apply_host_overrides(host, override_has_rocm=True)
        assert overridden.has_rocm is True

    def test_force_cpu_drops_nvidia_attributes(self):
        host = _run_detect_host(which_map={}, proc_dir_entries=["0000:01:00.0"])
        assert host.has_usable_nvidia is True
        overridden = _apply_host_overrides(host, force_cpu=True)
        assert overridden.has_usable_nvidia is False
        assert overridden.has_physical_nvidia is False
        assert overridden.has_rocm is False


# ── setup.sh source-level guarantees ────────────────────────────────────────


class TestSetupShHardening:
    @pytest.fixture(scope="class")
    def setup_src(self) -> str:
        return SETUP_SH.read_text(encoding="utf-8")

    def test_has_usable_nvidia_helper_exists(self, setup_src):
        assert "_setup_has_usable_nvidia_gpu()" in setup_src

    def test_helper_uses_proc_fallback(self, setup_src):
        start = setup_src.find("_setup_has_usable_nvidia_gpu()")
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "/proc/driver/nvidia/gpus" in body, (
            "_setup_has_usable_nvidia_gpu must fall back to /proc/driver/nvidia/gpus"
        )

    def test_gpu_summary_uses_helper(self, setup_src):
        assert "if _setup_has_usable_nvidia_gpu; then" in setup_src

    def test_timeout_wrapper_exists(self, setup_src):
        start = setup_src.find("_setup_run_smi()")
        assert start >= 0, "_setup_run_smi timeout wrapper must exist"
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "timeout 10" in body
        assert "command -v timeout" in body

    def test_cuda_source_build_gated_on_usable_nvidia(self, setup_src):
        """The nvcc source-build search must be gated on _setup_nvidia_usable.

        The gate also honours CUDA_VISIBLE_DEVICES=-1 (deliberately hidden
        GPU), so assert on both conditions rather than the exact line.
        """
        anchor = setup_src.find('NVCC_PATH=""\n')
        assert anchor >= 0
        window = setup_src[anchor : anchor + 700]
        assert 'if [ "$_setup_nvidia_usable" = true ]' in window, (
            "CUDA toolkit search must require a usable NVIDIA GPU, not just nvcc"
        )
        assert '[ "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]' in window, (
            "CUDA toolkit search must respect a deliberately hidden GPU"
        )

    def test_rocm_source_build_gated_on_amd_detected(self, setup_src):
        """The hipcc source-build search must be gated on _setup_amd_detected."""
        anchor = setup_src.find('ROCM_HIPCC=""')
        assert anchor >= 0
        window = setup_src[anchor : anchor + 400]
        assert '[ "$_setup_amd_detected" = true ]' in window, (
            "ROCm toolkit search must require a detected AMD GPU, not just hipcc"
        )

    def test_compute_cap_probe_timeout_wrapped(self, setup_src):
        assert "_setup_run_smi nvidia-smi --query-gpu=compute_cap" in setup_src

    def test_driver_version_probe_timeout_wrapped(self, setup_src):
        start = setup_src.find("_cuda_driver_max_version()")
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "_setup_run_smi nvidia-smi" in body


# TEST: install.sh -- UNSLOTH_TORCH_BACKEND classified on the final path segment


class TestBackendExportLeafClassification:
    """A custom UNSLOTH_PYTORCH_MIRROR whose base path contains "rocm" or
    "gfx" must not mislabel a cu*/cpu index as ROCm; classification uses the
    final path segment of TORCH_INDEX_URL only."""

    @pytest.fixture(scope="class")
    def install_src(self) -> str:
        return (PACKAGE_ROOT / "install.sh").read_text(encoding="utf-8")

    def test_export_block_uses_leaf(self, install_src):
        anchor = install_src.find("_torch_index_leaf=")
        assert anchor >= 0, "backend export must classify on the final path segment"
        window = install_src[anchor : anchor + 500]
        assert 'export UNSLOTH_TORCH_BACKEND="rocm"' in window
        assert 'export UNSLOTH_TORCH_BACKEND="cpu"' in window
        assert 'export UNSLOTH_TORCH_BACKEND="cuda"' in window

    def test_leaf_classification_behaviour(self, tmp_path):
        import subprocess as sp

        script = tmp_path / "leaf.sh"
        src = (PACKAGE_ROOT / "install.sh").read_text(encoding="utf-8")
        anchor = src.find("_torch_index_leaf=")
        block = src[anchor : src.find("esac", anchor) + 4]
        # Drive the extracted block with adversarial mirror URLs.
        script.write_text(
            "#!/bin/sh\n"
            'TORCH_INDEX_URL="$1"\n' + block + "\n"
            'printf "%s" "$UNSLOTH_TORCH_BACKEND"\n'
        )
        cases = {
            "https://download.pytorch.org/whl/cu128": "cuda",
            "https://download.pytorch.org/whl/cpu": "cpu",
            "https://download.pytorch.org/whl/rocm6.4": "rocm",
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/": "rocm",
            "https://repo.amd.com/rocm/whl/gfx1151/": "rocm",
            "https://mirror.local/rocm-cache/cu128": "cuda",
            "https://mirror.local/gfx-cache/cpu": "cpu",
        }
        for url, expected in cases.items():
            out = sp.run(
                ["sh", str(script), url], capture_output=True, text=True, timeout=30
            ).stdout.strip()
            assert out == expected, f"{url} classified as {out!r}, expected {expected!r}"
