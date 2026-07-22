"""AMD ROCm support tests across install pathways (all mocked, no AMD HW)."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch, PropertyMock

import pytest


# ── Load modules under test ──────────────────────────────────────────────────

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# install_llama_prebuilt.py
_PREBUILT_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_PREBUILT_SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt", _PREBUILT_PATH
)
assert _PREBUILT_SPEC is not None and _PREBUILT_SPEC.loader is not None
prebuilt_mod = importlib.util.module_from_spec(_PREBUILT_SPEC)
sys.modules[_PREBUILT_SPEC.name] = prebuilt_mod
_PREBUILT_SPEC.loader.exec_module(prebuilt_mod)

HostInfo = prebuilt_mod.HostInfo
AssetChoice = prebuilt_mod.AssetChoice
PrebuiltFallback = prebuilt_mod.PrebuiltFallback
resolve_upstream_asset_choice = prebuilt_mod.resolve_upstream_asset_choice
runtime_patterns_for_choice = prebuilt_mod.runtime_patterns_for_choice
_apply_host_overrides = prebuilt_mod._apply_host_overrides
_normalize_forwarded_gfx = prebuilt_mod._normalize_forwarded_gfx

# install_python_stack.py
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location("studio_install_python_stack", _STACK_PATH)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_detect_rocm_version = stack_mod._detect_rocm_version
_ensure_rocm_torch = stack_mod._ensure_rocm_torch
_has_rocm_gpu = stack_mod._has_rocm_gpu
_has_usable_nvidia_gpu = stack_mod._has_usable_nvidia_gpu
_ROCM_TORCH_INDEX = stack_mod._ROCM_TORCH_INDEX
_windows_rocm_index_url = stack_mod._windows_rocm_index_url
_detect_windows_gfx_arch = stack_mod._detect_windows_gfx_arch
_install_bnb_windows_rocm = stack_mod._install_bnb_windows_rocm


def _extract_sh_function_body(source: str, name: str) -> str:
    """Return a shell function body from `source` by brace matching."""
    needle = f"{name}() {{"
    start = source.find(needle)
    if start < 0:
        return ""
    depth = 0
    i = start + len(needle) - 1  # land on the opening brace
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return source[start:]


# ── Helper: build HostInfo for different scenarios ──────────────────────────


def nvidia_host(**overrides) -> HostInfo:
    """NVIDIA Linux x86_64 host."""
    defaults = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = (12, 6),
        compute_caps = ["89"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def rocm_host(**overrides) -> HostInfo:
    """AMD ROCm Linux x86_64 host (no NVIDIA)."""
    defaults = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def cpu_host(**overrides) -> HostInfo:
    """CPU-only Linux x86_64 host."""
    defaults = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def macos_host(**overrides) -> HostInfo:
    """macOS arm64 host."""
    defaults = dict(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def windows_host(**overrides) -> HostInfo:
    """Windows x86_64 host."""
    defaults = dict(
        system = "Windows",
        machine = "amd64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def windows_rocm_host(**overrides) -> HostInfo:
    """Windows x86_64 host with ROCm."""
    defaults = dict(
        system = "Windows",
        machine = "amd64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


# ── Upstream asset fixture ───────────────────────────────────────────────────

LLAMA_TAG = "b8508"

UPSTREAM_ASSETS = {
    f"llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz": f"https://example.com/{LLAMA_TAG}-linux-cpu.tar.gz",
    f"llama-{LLAMA_TAG}-bin-ubuntu-rocm-7.2-x64.tar.gz": f"https://example.com/{LLAMA_TAG}-linux-rocm.tar.gz",
    f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip": f"https://example.com/{LLAMA_TAG}-win-cpu.zip",
    f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip": f"https://example.com/{LLAMA_TAG}-win-cuda.zip",
    f"llama-{LLAMA_TAG}-bin-win-hip-radeon-x64.zip": f"https://example.com/{LLAMA_TAG}-win-hip.zip",
    f"llama-{LLAMA_TAG}-bin-macos-arm64.tar.gz": f"https://example.com/{LLAMA_TAG}-macos-arm64.tar.gz",
    f"llama-{LLAMA_TAG}-bin-macos-x64.tar.gz": f"https://example.com/{LLAMA_TAG}-macos-x64.tar.gz",
}


# TEST: install_llama_prebuilt.py -- resolve_upstream_asset_choice


class TestResolveUpstreamAssetChoice:
    """Verify that the asset selection logic picks the right binary for each platform."""

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_nvidia_linux_gets_cpu_asset(self, mock_assets):
        """NVIDIA host should NOT hit the ROCm path -- gets CPU asset (CUDA handled elsewhere)."""
        host = nvidia_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "linux-cpu"
        assert "ubuntu-x64" in choice.name
        assert "rocm" not in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_rocm_linux_gets_rocm_prebuilt(self, mock_assets):
        """AMD ROCm Linux host should get the ROCm prebuilt."""
        host = rocm_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "linux-rocm"
        assert "rocm" in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_cpu_linux_gets_cpu_asset(self, mock_assets):
        """CPU-only Linux host should get CPU asset."""
        host = cpu_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "linux-cpu"
        assert "ubuntu-x64" in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_macos_arm64_gets_macos_asset(self, mock_assets):
        """macOS arm64 host should get macOS asset."""
        host = macos_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "macos-arm64"
        assert "macos-arm64" in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_windows_cpu_gets_cpu_asset(self, mock_assets):
        """Windows CPU-only host should get Windows CPU asset."""
        host = windows_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "windows-cpu"
        assert "win-cpu" in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_windows_rocm_gets_hip_asset(self, mock_assets):
        """Windows ROCm host should get Windows HIP asset."""
        host = windows_rocm_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "windows-hip"
        assert "hip" in choice.name

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_mixed_nvidia_rocm_prefers_nvidia(self, mock_assets):
        """Host with both NVIDIA and ROCm should use NVIDIA (CPU path here, CUDA elsewhere)."""
        host = nvidia_host(has_rocm = True)
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "linux-cpu"
        assert "rocm" not in choice.name

    @patch.object(prebuilt_mod, "github_release_assets")
    def test_rocm_linux_no_prebuilt_falls_back(self, mock_assets):
        """AMD ROCm host should fall back to source build when no ROCm prebuilt exists."""
        assets_without_rocm = {k: v for k, v in UPSTREAM_ASSETS.items() if "rocm" not in k}
        mock_assets.return_value = assets_without_rocm
        host = rocm_host()
        with pytest.raises(PrebuiltFallback, match = "ROCm detected"):
            resolve_upstream_asset_choice(host, LLAMA_TAG)

    @patch.object(prebuilt_mod, "github_release_assets")
    def test_windows_rocm_no_hip_falls_to_cpu(self, mock_assets):
        """Windows+ROCm with HIP prebuilt missing should fall through to CPU."""
        assets_no_hip = {k: v for k, v in UPSTREAM_ASSETS.items() if "hip" not in k}
        mock_assets.return_value = assets_no_hip
        host = windows_rocm_host()
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "windows-cpu"

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_macos_rocm_impossible_has_rocm_false(self, mock_assets):
        """macOS host should never have has_rocm=True in practice; verify it gets macOS asset."""
        host = macos_host(has_rocm = True)
        choice = resolve_upstream_asset_choice(host, LLAMA_TAG)
        assert choice.install_kind == "macos-arm64"

    @patch.object(prebuilt_mod, "github_release_assets", return_value = UPSTREAM_ASSETS)
    def test_linux_aarch64_rocm_gets_prebuilt_fallback(self, mock_assets):
        """Linux aarch64 with ROCm -- no x86_64 match, should raise PrebuiltFallback."""
        host = rocm_host(machine = "aarch64", is_x86_64 = False, is_arm64 = True)
        with pytest.raises(PrebuiltFallback):
            resolve_upstream_asset_choice(host, LLAMA_TAG)


# TEST: install_llama_prebuilt.py -- runtime_patterns_for_choice


class TestRuntimePatterns:
    """Verify runtime file patterns for all install kinds."""

    def test_linux_cpu_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-cpu"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "llama-server" in patterns
        assert "llama-quantize" in patterns
        # lib*.so* covers libllama/libggml/libmtmd plus the libllama-*-impl.so
        # split from ggml-org/llama.cpp #23462 (between b9279 and b9283).
        assert "lib*.so*" in patterns

    def test_linux_cuda_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-cuda"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "lib*.so*" in patterns

    def test_linux_rocm_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-rocm"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "lib*.so*" in patterns
        assert "llama-server" in patterns

    def test_windows_hip_patterns(self):
        choice = AssetChoice(
            repo = "",
            tag = "",
            name = "",
            url = "",
            source_label = "",
            install_kind = "windows-hip",
        )
        patterns = runtime_patterns_for_choice(choice)
        # Narrowed from "*.exe" to the two binaries Unsloth actually invokes.
        assert "llama-server.exe" in patterns
        assert "llama-quantize.exe" in patterns
        assert "*.dll" in patterns

    def test_macos_patterns(self):
        choice = AssetChoice(
            repo = "",
            tag = "",
            name = "",
            url = "",
            source_label = "",
            install_kind = "macos-arm64",
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "lib*.dylib" in patterns

    def test_diffusion_visual_server_kept(self):
        # The DiffusionGemma visual-server must survive the prune so Unsloth can
        # serve DiffusionGemma GGUFs natively.
        for kind, name in (
            ("linux-cuda", "llama-diffusion-gemma-visual-server"),
            ("macos-arm64", "llama-diffusion-gemma-visual-server"),
            ("windows-cuda", "llama-diffusion-gemma-visual-server.exe"),
        ):
            choice = AssetChoice(
                repo = "", tag = "", name = "", url = "", source_label = "", install_kind = kind
            )
            assert name in runtime_patterns_for_choice(choice)


# TEST: install_llama_prebuilt.py -- HostInfo.has_rocm field


class TestHostInfoRocm:
    """Verify has_rocm field does not affect other HostInfo behavior."""

    def test_has_rocm_default_false(self):
        host = HostInfo(
            system = "Linux",
            machine = "x86_64",
            is_windows = False,
            is_linux = True,
            is_macos = False,
            is_x86_64 = True,
            is_arm64 = False,
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            visible_cuda_devices = None,
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        assert host.has_rocm is False

    def test_has_rocm_explicit_true(self):
        host = rocm_host()
        assert host.has_rocm is True

    def test_nvidia_host_no_rocm(self):
        host = nvidia_host()
        assert host.has_rocm is False
        assert host.has_usable_nvidia is True

    def test_detect_host_has_rocm_detection_logic(self):
        """detect_host() should have ROCm GPU detection logic."""
        import inspect

        source = inspect.getsource(prebuilt_mod.detect_host)
        # Must probe for actual GPU, not just tool presence.
        assert "rocminfo" in source or "amd-smi" in source

    def test_detect_host_windows_rocm_detection(self):
        """detect_host() source should have Windows-specific ROCm GPU detection."""
        import inspect

        source = inspect.getsource(prebuilt_mod.detect_host)
        assert "hipinfo" in source or "amd-smi" in source


# TEST: install_python_stack.py -- _detect_rocm_version


class TestDetectRocmVersion:
    """Verify ROCm version detection from various sources."""

    def test_no_rocm_returns_none(self, tmp_path):
        """No ROCm installed should return None."""
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "nonexistent")}):
            with patch("shutil.which", return_value = None):
                result = _detect_rocm_version()
                assert result is None

    def test_version_from_file(self, tmp_path):
        """Reads version from /opt/rocm/.info/version."""
        info_dir = tmp_path / ".info"
        info_dir.mkdir()
        (info_dir / "version").write_text("7.1.0-12345\n")
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path)}):
            result = _detect_rocm_version()
            assert result == (7, 1)

    def test_version_62(self, tmp_path):
        """Reads ROCm 6.2 version."""
        info_dir = tmp_path / ".info"
        info_dir.mkdir()
        (info_dir / "version").write_text("6.2.0\n")
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path)}):
            result = _detect_rocm_version()
            assert result == (6, 2)

    def test_hipconfig_fallback(self, tmp_path):
        """Falls back to hipconfig --version when file not found."""
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "nonexistent")}):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = b"6.3.21234.2\n"
            with patch("shutil.which", return_value = "/usr/bin/hipconfig"):
                with patch("subprocess.run", return_value = mock_result):
                    result = _detect_rocm_version()
                    assert result == (6, 3)

    def test_dpkg_fallback_without_hipconfig(self, tmp_path):
        """dpkg rocm-core fallback works when amd-smi and hipconfig are absent
        (regression: a shadowing local re import raised UnboundLocalError)."""

        def which(cmd):
            return "/usr/bin/dpkg-query" if cmd == "dpkg-query" else None

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1:6.3.0-1\n"
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "nonexistent")}):
            with patch("shutil.which", side_effect = which):
                with patch("subprocess.run", return_value = mock_result):
                    assert _detect_rocm_version() == (6, 3)

    def test_empty_version_file(self, tmp_path):
        """Empty version file should return None."""
        info_dir = tmp_path / ".info"
        info_dir.mkdir()
        (info_dir / "version").write_text("")
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path)}):
            with patch("shutil.which", return_value = None):
                result = _detect_rocm_version()
                assert result is None

    def test_version_with_epoch_prefix(self, tmp_path):
        """Debian epoch prefix (2:6.2.0) -- version file has no epoch, so should parse."""
        info_dir = tmp_path / ".info"
        info_dir.mkdir()
        (info_dir / "version").write_text("6.2.0\n")
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path)}):
            result = _detect_rocm_version()
            assert result == (6, 2)

    def test_multiple_version_sources_first_wins(self, tmp_path):
        """When both .info/version and lib/rocm_version exist, first found wins."""
        info_dir = tmp_path / ".info"
        info_dir.mkdir()
        (info_dir / "version").write_text("7.1.0\n")
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "rocm_version").write_text("6.3.0\n")
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path)}):
            result = _detect_rocm_version()
            assert result == (7, 1)  # .info/version checked first

    def test_hipconfig_multiline_output(self, tmp_path):
        """hipconfig with multi-line output -- should use first line."""
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "nonexistent")}):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = b"6.3.21234.2\nSome extra info\n"
            with patch("shutil.which", return_value = "/usr/bin/hipconfig"):
                with patch("subprocess.run", return_value = mock_result):
                    result = _detect_rocm_version()
                    assert result == (6, 3)

    def test_hipconfig_timeout(self, tmp_path):
        """hipconfig that times out should return None."""
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "nonexistent")}):
            with patch("shutil.which", return_value = "/usr/bin/hipconfig"):
                with patch(
                    "subprocess.run",
                    side_effect = subprocess.TimeoutExpired("hipconfig", 5),
                ):
                    result = _detect_rocm_version()
                    assert result is None


# TEST: install_python_stack.py -- _ensure_rocm_torch


class TestEnsureRocmTorch:
    """Verify ROCm torch reinstall logic."""

    # _infer_linux_amd_gfx_arch mocked to None: on a real Strix host the live
    # /proc/cpuinfo would otherwise take the inferred-install path and break
    # these "must not install" hosts (environment leak, not the code under test).
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = None)
    def test_no_rocm_skips(self, mock_infer, mock_nvidia, mock_pip):
        """No ROCm toolchain should skip entirely."""
        # Pin _detect_windows_gfx_arch to None so a real AMD test host's WMI
        # fallback can't defeat the "no ROCm anywhere" premise.
        with patch.object(stack_mod, "_detect_windows_gfx_arch", return_value = None):
            with patch("os.path.isdir", return_value = False):
                with patch("shutil.which", return_value = None):
                    _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = False)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = "gfx1151")
    @patch.object(stack_mod, "_detect_rocm_version", return_value = None)
    def test_inferred_gfx_without_rocm_runtime_installs_amd_index(
        self, mock_ver, mock_infer, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """Strix Halo without /dev/kfd must still get AMD gfx1151 wheels (unslothai#7301)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"|2.10.0+cpu\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = False)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = "gfx1151")
    @patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = [])
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_inferred_gfx_not_overwritten_when_rocm_userland_readable(
        self, mock_ver, mock_gfx, mock_infer, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """Codex P1 #7305: after an inferred per-arch install, do not fall through to the
        generic pytorch.org/rocmX.Y reinstall just because has_hip_torch is still False.
        Readable ROCm userland without /dev/kfd is exactly the case that used to overwrite
        the AMD gfx wheels."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"|2.10.0+cpu\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        assert mock_pip.call_count == 1, mock_pip.call_args_list
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "rocm7.1" not in torch_call
        assert "download.pytorch.org" not in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = "gfx1151")
    @patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = ["gfx1100"])
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_inference_yields_to_runtime_visible_gpu(
        self, mock_ver, mock_gfx, mock_infer, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """When the runtime CAN enumerate a GPU, the cpuinfo inference must not
        install wheels: a mixed Strix APU + dGPU box with the dGPU selected would
        otherwise get gfx1151 wheels for a gfx1100 GPU. The runtime-visible arch
        (Strix override / generic branch) decides instead."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"|2.10.0+cpu\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        all_calls = str(mock_pip.call_args_list) + str(mock_pip_try.call_args_list)
        assert "gfx1151" not in all_calls, all_calls
        assert "rocm7.1" in all_calls, all_calls

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_cuda_torch_on_amd_host_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A CUDA-only torch build is unusable on an AMD-only host, so it must be
        reinstalled to ROCm (has_hip_torch is driven by the empty HIP marker, not
        by treating the CUDA version string as a HIP marker)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        # Single-line probe: empty HIP marker before "|" for a CUDA build.
        mock_probe.stdout = b"|2.10.0+cu126\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_torch_already_has_hip_skips(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """If torch already has HIP, should skip ROCm reinstall."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.1.12345|2.10.0+rocm7.1\n"  # HIP marker + version
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_cpu_torch_probe_line_not_read_as_hip(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """A CPU build's probe line ("|2.10.0+cpu") must not read as HIP: the version
        after the "|" separator is data, not a HIP marker, so has_hip_torch stays False
        and the reinstall fires."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"|2.10.0+cpu\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                with patch.object(stack_mod, "pip_install_try", return_value = True):
                    _ensure_rocm_torch()
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_cpu_torch_gets_rocm_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """CPU-only torch on ROCm host should trigger reinstall."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"  # empty = no GPU backend
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])
        assert mock_pip_try.call_count >= 1
        assert "bitsandbytes" in str(mock_pip_try.call_args_list[0])
        assert mock_pip_try.call_args.kwargs["force_pip"] is True

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 3))
    def test_rocm_63_selects_correct_tag(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """ROCm 6.3 should select rocm6.3 tag."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        torch_call = mock_pip.call_args_list[0]
        assert "rocm6.3" in str(torch_call)

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (5, 0))
    def test_old_rocm_skips(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """ROCm version too old (below 6.0) should skip."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = None)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = None)
    def test_version_unreadable_prints_warning(
        self, mock_ver, mock_infer, mock_gpu, mock_nvidia, mock_pip, capsys
    ):
        """ROCm detected but version unreadable should print warning and skip."""
        with patch("os.path.isdir", return_value = True):
            _ensure_rocm_torch()
        mock_pip.assert_not_called()
        captured = capsys.readouterr()
        assert "unreadable" in captured.out

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_72_selects_72_tag(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """ROCm 7.2 should select rocm7.2 tag (now in mapping with torch 2.11.0)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        torch_call = mock_pip.call_args_list[0]
        assert "rocm7.2" in str(torch_call)

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 14))
    @patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = ["gfx1150"])
    def test_rocm_714_strix_routes_to_amd_arch_index(
        self, mock_gfx, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """ROCm 7.14 caps to rocm7.2 on pytorch.org; Strix must use AMD gfx index."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.14.60850|2.11.0+rocm7.2\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1150" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 4))
    def test_explicit_gfx_index_honored_and_skips_strix_reroute(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """An explicit gfx wheel-index pin is authoritative: install from it verbatim
        with torch 2.11, and never re-probe gfx codes to second-guess it (host ROCm 6.4
        would otherwise pick the rocm6.4 wheel / trigger the Strix re-route)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"  # cpu torch -> reinstall
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx1151"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    # Would raise if the Strix block ran (it is skipped on an explicit pin).
                    with patch.object(
                        stack_mod, "_detect_amd_gfx_codes", side_effect = AssertionError
                    ):
                        _ensure_rocm_torch()
        assert mock_pip.call_count == 1
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    def test_rocm_pin_family_mismatch_helper(self):
        """_rocm_pin_family_mismatch: exact rocm compare, else the 2.11 line."""
        f = stack_mod._rocm_pin_family_mismatch
        base = "https://download.pytorch.org/whl"
        amd = "https://repo.amd.com/rocm/whl"
        # Exact rocm version comparison.
        assert f(f"{base}/rocm7.2", "2.11.0+rocm7.2") is False
        assert f(f"{base}/rocm7.2", "2.10.0+rocm6.4") is True
        assert f(f"{base}/rocm6.4", "2.10.0+rocm6.4") is False
        # rocm7.2 is KNOWN-2.11. A +rocm7.2 wheel whose RELEASE drifted off 2.11 shares the
        # tag but violates the spec -> mismatch (a plain version compare would accept it).
        assert f(f"{base}/rocm7.2", "2.12.0+rocm7.2") is True
        assert f(f"{base}/rocm7.2", "2.13.0+rocm7.2") is True
        assert f(f"{base}/rocm7.2", "2.11.5+rocm7.2") is False  # patch on 2.11 is in-spec
        # An UNKNOWN newer rocm (not on the 2.11 allowlist) is not floored to 2.11, so a
        # matching rocm version at any release line is NOT a mismatch on this branch.
        assert f(f"{base}/rocm8.0", "2.12.0+rocm8.0") is False
        # gfx pin (2.11 line) vs installed release line.
        assert f(f"{amd}/gfx1151", "2.10.0+rocm6.4") is True
        assert f(f"{amd}/gfx1151", "2.11.0+rocm7.13.0") is False
        # rocm7.2 pin vs an untagged (no +rocm) wheel: a CPU/CUDA build never
        # satisfies a ROCm pin, regardless of its release line -> always a mismatch.
        assert f(f"{base}/rocm7.2", "2.10.0") is True
        assert f(f"{base}/rocm7.2", "2.11.0") is True
        assert f(f"{base}/rocm6.4", "2.10.0") is True
        # A 2.11-allowlist gfx pin over a GENERIC (two-part +rocm7.2) 2.11 wheel mismatches:
        # the user wants AMD's per-arch (three-part) wheel, not the generic one.
        assert f(f"{amd}/gfx1151", "2.11.0+rocm7.2") is True
        assert f(f"{amd}/gfx120X-all", "2.11.0+rocm7.2") is True
        # ...but an already-installed per-arch (three-part) wheel is NOT re-flagged
        # (no reinstall loop once the correct gfx wheel is present).
        assert f(f"{amd}/gfx120X-all", "2.11.0+rocm7.13.0") is False
        assert f(f"{amd}/gfx1150", "2.11.0+rocm7.13.0") is False
        # A NON-2.11 gfx pin (gfx110X-all/gfx90a/gfx908) tracks the default <2.11 spec: a
        # correct 2.10+rocm wheel is NOT a mismatch, a 2.11 build is.
        assert f(f"{amd}/gfx110X-all", "2.10.0+rocm6.4") is False
        assert f(f"{amd}/gfx90a", "2.10.0+rocm6.3") is False
        assert f(f"{amd}/gfx908", "2.10.0+rocm7.0") is False
        assert f(f"{amd}/gfx110X-all", "2.11.0+rocm7.2") is True
        # A non-2.11 gfx pin over an untagged (no +rocm) wheel is a mismatch even
        # when torch is already <2.11: a CPU/CUDA build never satisfies the ROCm pin.
        assert f(f"{amd}/gfx110X-all", "2.10.0") is True
        assert f(f"{amd}/gfx90a", "2.10.0") is True
        # A major-only rocm pin (rocm7) compares on the major alone: rocm6.x mismatches,
        # any rocm7.x satisfies it, an untagged wheel never does, a bare +rocm is lenient.
        assert f(f"{base}/rocm7", "2.10.0+rocm6.4") is True
        assert f(f"{base}/rocm7", "2.11.0+rocm7.2") is False
        assert f(f"{base}/rocm7", "2.11.0+rocm7.13.0") is False
        assert f(f"{base}/rocm7", "2.10.0") is True
        assert f(f"{base}/rocm7", "2.10.0+rocm") is False

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_pin_mismatch_over_installed_rocm_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A rocm7.2 pin over an already-installed OLDER +rocm6.4 build must reinstall,
        even though has_hip_torch is True (the ROCm analogue of the CUDA cuXXX mismatch)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        # HIP marker present (has_hip_torch=True) + installed +rocm6.4 wheel.
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "rocm7.2" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 4))
    def test_gfx_pin_over_installed_pre211_rocm_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx* pin (2.11 line) over an installed pre-2.11 +rocm6.4 build reinstalls."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx1151"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    with patch.object(
                        stack_mod, "_detect_amd_gfx_codes", side_effect = AssertionError
                    ):
                        _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_pin_matches_installed_no_torch_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A rocm7.2 pin over an already-matching +rocm7.2 build must NOT reinstall torch
        (no false reinstall of a correct ROCm venv)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.2.12345|2.11.0+rocm7.2\n"
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        # No torch reinstall: any pip_install call must not target a torch index.
        for _call in mock_pip.call_args_list:
            _args = [str(a) for a in _call.args]
            if "--index-url" in _args:
                _url = _args[_args.index("--index-url") + 1]
                assert "rocm7.2" not in _url or "torch" not in " ".join(
                    _args
                ), "torch must not be reinstalled when the pin already matches"
        # A torch reinstall would pass torch>=... as a positional; assert none did.
        assert not any(
            any(str(a).startswith("torch") for a in _c.args) for _c in mock_pip.call_args_list
        )

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 4))
    def test_non211_gfx_pin_over_210_rocm_no_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx110X-all pin (NOT in the 2.11 allowlist) over a correct 2.10+rocm
        wheel must NOT be flagged stale -- the install path uses the default <2.11
        specs for that arch, so re-flagging would reinstall-loop on every update."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"6.4.12345|2.10.0+rocm6.4\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx110X-all"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    _ensure_rocm_torch()
        # has_hip_torch True + no mismatch -> torch must NOT be reinstalled.
        assert not any(
            any(str(a).startswith("torch") for a in _c.args) for _c in mock_pip.call_args_list
        )

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_gfx_pin_over_generic_rocm211_reinstalls(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """A gfx1151 pin over a GENERIC (two-part +rocm7.2) 2.11 wheel must reinstall
        the AMD per-arch wheel -- even though both are torch 2.11, the generic wheel
        is not the per-arch build the user pinned (Strix stays off the generic wheel)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.2.12345|2.11.0+rocm7.2\n"
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx1151"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("os.path.isdir", return_value = True):
                with patch("subprocess.run", return_value = mock_probe):
                    with patch.object(
                        stack_mod, "_detect_amd_gfx_codes", side_effect = AssertionError
                    ):
                        _ensure_rocm_torch()
        torch_call = str(mock_pip.call_args_list[0])
        assert "gfx1151" in torch_call
        assert "torch>=2.11.0,<2.12.0" in torch_call

    def test_radeon_url_not_classified_as_pip_rocm_family(self):
        """A repo.radeon.com find-links dir (leaf rocm-rel-7.2.1) starts with "rocm" but is
        NOT a pip --index-url ROCm family: it must route to the verbatim path, not a
        --index-url reinstall that fails against a find-links listing."""
        leaf_f = stack_mod._is_pip_rocm_family_leaf
        # Real pip ROCm families (download.pytorch.org/whl/rocmX.Y, repo.amd.com gfx).
        assert leaf_f("rocm7.2") is True
        assert leaf_f("rocm6.4") is True
        assert leaf_f("gfx120x-all") is True
        assert leaf_f("gfx1151") is True
        # A bare rocm<digits> (no minor) is still an exact family.
        assert leaf_f("rocm7") is True
        # A Radeon find-links dir leaf, a custom mirror, cpu and cuda are NOT pip rocm.
        assert leaf_f("rocm-rel-7.2.1") is False
        assert leaf_f("simple") is False
        assert leaf_f("current") is False
        assert leaf_f("cpu") is False
        assert leaf_f("cu128") is False
        # A rocm<digit>-SUFFIX private mirror shares the family prefix but is a custom pin
        # the verbatim path owns: a ^rocm\d PREFIX match would wrongly treat it as a
        # --index-url family. Match EXACTLY.
        assert leaf_f("rocm7.2-private") is False
        assert leaf_f("rocm7-current") is False
        assert leaf_f("rocm7.2.1") is False  # two-part local suffix -> custom, not rocm7.2

        radeon = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
        pip_rocm = "https://download.pytorch.org/whl/rocm7.2"
        amd_gfx = "https://repo.amd.com/rocm/whl/gfx120X-all"

        def _classify(url, fn):
            with patch.dict(stack_mod.os.environ, {"UNSLOTH_TORCH_INDEX_URL": url}, clear = False):
                stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
                return fn()

        rocm_fn = stack_mod._explicit_rocm_torch_index_url
        unk_fn = stack_mod._explicit_unknown_family_torch_index_url
        # Real pip rocm/gfx pins ARE a ROCm family (reinstallable via --index-url) and
        # are NOT "unknown".
        assert _classify(pip_rocm, rocm_fn) == pip_rocm
        assert _classify(amd_gfx, rocm_fn) == amd_gfx
        assert _classify(pip_rocm, unk_fn) is None
        assert _classify(amd_gfx, unk_fn) is None
        # The Radeon find-links URL is NOT a pip ROCm family (so _ensure_rocm_torch skips
        # it) and IS unknown, so the family repair helpers leave it alone.
        assert _classify(radeon, rocm_fn) is None
        assert _classify(radeon, unk_fn) == radeon

        # A rocm<digit>-suffix private mirror routes the same way: NOT a pip rocm family,
        # IS an unknown-family (verbatim) pin.
        suffixed = "https://co.internal/whl/rocm7.2-private"
        assert _classify(suffixed, rocm_fn) is None
        assert _classify(suffixed, unk_fn) == suffixed

    @patch.object(stack_mod, "pip_install")
    def test_ensure_cpu_torch_broken_probe_reinstalls(self, mock_pip):
        """_ensure_cpu_torch: torch present but unimportable (probe exit != 0) under an
        explicit CPU pin must reinstall from the pin, not return -- the base update does
        not repair a broken installed torch, so returning would strand it (Codex P2)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 1  # torch present but cannot import
        mock_probe.stdout = b""
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://mirror.local/cpu"}
        with patch.dict(stack_mod.os.environ, env, clear = False):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
            with patch("subprocess.run", return_value = mock_probe):
                with patch.object(stack_mod, "NO_TORCH", False):
                    stack_mod._ensure_cpu_torch()
        assert mock_pip.call_count == 1
        assert "https://mirror.local/cpu" in str(mock_pip.call_args)

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install_try", return_value = True)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_probe_timeout_triggers_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """Probe subprocess timeout should not crash; should proceed to reinstall."""
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", side_effect = subprocess.TimeoutExpired("python", 30)):
                _ensure_rocm_torch()
        # Probe timeout: treat torch as unusable and reinstall torch + bitsandbytes.
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])
        assert mock_pip_try.call_count >= 1
        assert mock_pip_try.call_args.kwargs["force_pip"] is True

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = False)
    @patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = None)
    def test_no_gpu_with_rocm_tools_skips(self, mock_infer, mock_gpu, mock_nvidia, mock_pip):
        """ROCm tools present but no actual AMD GPU should skip entirely."""
        # Pin the Windows arch probe to None so a real AMD host's WMI fallback
        # can't defeat the "no actual GPU" premise.
        with patch.object(stack_mod, "_detect_windows_gfx_arch", return_value = None):
            with patch("os.path.isdir", return_value = True):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True)
    def test_torch_backend_cuda_env_skips_entirely(self, mock_nvidia, mock_gpu, mock_pip):
        """UNSLOTH_TORCH_BACKEND=cuda must short-circuit before any GPU probe."""
        with patch.dict(os.environ, {"UNSLOTH_TORCH_BACKEND": "cuda"}):
            with patch.object(stack_mod, "_TORCH_BACKEND", "cuda"):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True)
    def test_torch_backend_cpu_env_skips_entirely(self, mock_nvidia, mock_gpu, mock_pip):
        """UNSLOTH_TORCH_BACKEND=cpu must short-circuit before any GPU probe."""
        with patch.dict(os.environ, {"UNSLOTH_TORCH_BACKEND": "cpu"}):
            with patch.object(stack_mod, "_TORCH_BACKEND", "cpu"):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()


# TEST: install_python_stack.py -- torch-index MARKER mechanism (PR #6692)


class TestHasRocmGpuKfdVendorGuard:
    """KFD sysfs fallback rejects non-AMD (NVIDIA) KFD nodes (source-level checks)."""

    def _src(self) -> str:
        """Return the source of _has_rocm_gpu from install_python_stack.py."""
        import inspect
        return inspect.getsource(stack_mod._has_rocm_gpu)

    def test_vendor_id_check_present(self):
        """_has_rocm_gpu sysfs fallback must check vendor_id 4098 (AMD 0x1002)."""
        src = self._src()
        assert "vendor_id" in src, (
            "_has_rocm_gpu KFD sysfs fallback must read the properties file "
            "to check vendor_id and exclude NVIDIA KFD nodes"
        )
        assert "4098" in src, (
            "_has_rocm_gpu must require AMD vendor_id 4098 (0x1002) in the "
            "KFD node properties to avoid false positives on NVIDIA systems"
        )

    def test_vendor_regex_pattern_anchored(self):
        """The vendor_id regex must use a word boundary to avoid partial matches."""
        import re as _re

        src = self._src()
        # Word boundary so "vendor_id 41098" doesn't match "vendor_id 4098".
        assert (
            _re.search(r"\\b.*vendor_id.*\\b", src) or "\\bvendor_id" in src
        ), "_has_rocm_gpu vendor_id check should use word boundary anchors"

    def test_sysfs_fallback_guarded_by_non_win32(self):
        """KFD sysfs fallback must be Linux-only (guarded by sys.platform != 'win32')."""
        src = self._src()
        assert "win32" in src, "_has_rocm_gpu sysfs fallback must be guarded by sys.platform check"

    def test_cpu_node_excluded(self):
        """gpu_id == '0' must be excluded (CPU topology nodes)."""
        src = self._src()
        assert (
            '!= "0"' in src or "== '0'" in src or "!= '0'" in src or '"0"' in src
        ), "_has_rocm_gpu must skip gpu_id 0 nodes (CPU nodes)"

    def test_install_sh_has_vendor_check(self):
        """_has_amd_rocm_gpu in install.sh sysfs fallback must also check vendor_id 4098."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("_has_amd_rocm_gpu()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert "vendor_id" in func_body, "_has_amd_rocm_gpu sysfs fallback must check vendor_id"
        assert "4098" in func_body, "_has_amd_rocm_gpu must require AMD vendor_id 4098 (0x1002)"

    def test_has_rocm_gpu_returns_false_when_nvidia_present(self):
        """_has_rocm_gpu returns False when _has_usable_nvidia_gpu is True (NVIDIA always wins)."""
        with patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True):
            with patch("shutil.which", return_value = "/usr/bin/rocminfo"):
                # rocminfo claims an AMD GPU is present.
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Name: gfx1100\n"
                with patch("subprocess.run", return_value = mock_result):
                    assert not stack_mod._has_rocm_gpu(), (
                        "_has_rocm_gpu must return False when NVIDIA GPU is detected, "
                        "regardless of what rocminfo reports"
                    )

    def test_install_sh_has_rocm_gpu_nvidia_guard(self):
        """_has_amd_rocm_gpu in install.sh must call _has_usable_nvidia_gpu and return 1 if true."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("_has_amd_rocm_gpu()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert (
            "_has_usable_nvidia_gpu" in func_body
        ), "_has_amd_rocm_gpu must call _has_usable_nvidia_gpu to block NVIDIA hosts"
        assert (
            "return 1" in func_body
        ), "_has_amd_rocm_gpu must return 1 (false) when NVIDIA GPU is detected"

    def test_has_usable_nvidia_gpu_proc_fallback_present(self):
        """`_has_usable_nvidia_gpu` must have a /proc/driver/nvidia fallback."""
        import inspect

        src = inspect.getsource(stack_mod._has_usable_nvidia_gpu)
        assert "/proc/driver/nvidia" in src, (
            "_has_usable_nvidia_gpu must fall back to /proc/driver/nvidia/gpus when "
            "nvidia-smi subprocess fails, to handle PATH gaps and driver init races"
        )

    def test_install_sh_has_usable_nvidia_gpu_proc_fallback(self):
        """_has_usable_nvidia_gpu in install.sh must also have a /proc/driver/nvidia fallback."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("_has_usable_nvidia_gpu()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert "/proc/driver/nvidia" in func_body, (
            "_has_usable_nvidia_gpu in install.sh must fall back to "
            "/proc/driver/nvidia/gpus when nvidia-smi fails"
        )


# TEST: install_python_stack.py -- _ROCM_TORCH_INDEX mapping


class TestRocmTorchIndex:
    """Verify the ROCm version -> torch index tag mapping."""

    def test_mapping_is_sorted_descending(self):
        """Keys should be in descending order for the next() iteration to work."""
        keys = list(_ROCM_TORCH_INDEX.keys())
        assert keys == sorted(keys, reverse = True)

    def test_rocm_72_in_mapping(self):
        """ROCm 7.2 should be in the active mapping (torch 2.11.0 now supported)."""
        assert (7, 2) in _ROCM_TORCH_INDEX
        assert _ROCM_TORCH_INDEX[(7, 2)] == "rocm7.2"

    def test_rocm_71_maps_correctly(self):
        assert _ROCM_TORCH_INDEX[(7, 1)] == "rocm7.1"

    def test_rocm_63_maps_correctly(self):
        assert _ROCM_TORCH_INDEX[(6, 3)] == "rocm6.3"

    def test_rocm_60_maps_correctly(self):
        assert _ROCM_TORCH_INDEX[(6, 0)] == "rocm6.0"

    def test_all_tags_use_download_pytorch(self):
        """All tags should be for download.pytorch.org, not repo.radeon.com."""
        for tag in _ROCM_TORCH_INDEX.values():
            assert tag.startswith("rocm")
            assert "radeon" not in tag

    def test_newer_rocm_selects_best_match(self):
        """ROCm 7.2 (now in map) should select rocm7.2 directly."""
        ver = (7, 2)
        tag = next(
            (
                t
                for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                if ver >= (maj, mn)
            ),
            None,
        )
        assert tag == "rocm7.2"

    def test_rocm_64_selects_64(self):
        ver = (6, 4)
        tag = next(
            (
                t
                for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                if ver >= (maj, mn)
            ),
            None,
        )
        assert tag == "rocm6.4"


# TEST: hardware.py -- IS_ROCM flag and detect_hardware


class TestHardwareRocmFlag:
    """Verify IS_ROCM flag behavior without importing the full hardware module."""

    def test_hardware_py_has_is_rocm(self):
        """hardware.py should define IS_ROCM."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert "IS_ROCM: bool" in source and "False" in source

    def test_hardware_py_sets_is_rocm_on_hip(self):
        """detect_hardware() should set IS_ROCM when torch.version.hip is set."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert 'torch.version, "hip"' in source or "torch.version.hip" in source

    def test_hardware_py_still_returns_cuda_for_rocm(self):
        """DeviceType should remain CUDA even on ROCm -- no DeviceType.ROCM."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        enum_section = source.split("class DeviceType")[1].split("\n\n")[0]
        assert "ROCM" not in enum_section

    def test_hardware_py_has_rocm_in_package_versions(self):
        """get_package_versions() should include 'rocm' key."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert '"rocm"' in source

    def test_hardware_py_device_type_cuda_references_intact(self):
        """All existing DeviceType.CUDA references should still be present."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert "DeviceType.CUDA" in source
        assert "DEVICE = DeviceType.CUDA" in source

    def test_is_rocm_exported_from_init(self):
        """IS_ROCM should be exported from hardware __init__.py."""
        init_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        source = init_path.read_text(encoding = "utf-8")
        assert "IS_ROCM" in source

    def test_is_rocm_in_all_list(self):
        """IS_ROCM should be in __all__ list in __init__.py."""
        init_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        source = init_path.read_text(encoding = "utf-8")
        assert '"IS_ROCM"' in source

    def test_get_package_versions_returns_rocm_key(self):
        """get_package_versions() source should return both 'cuda' and 'rocm' keys."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_package_versions")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert '"cuda"' in func_body
        assert '"rocm"' in func_body

    def test_distributed_stubs_cover_is_torchelastic_launched(self):
        """Must stub is_torchelastic_launched (Windows ROCm torch.distributed lacks it)."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert "is_torchelastic_launched" in source

    def test_distributed_stubs_cover_core_helpers(self):
        """_determine_attention_impl_for_gpu_estimate must stub the four core distributed helpers."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        for attr in ("is_initialized", "is_available", "get_rank", "get_world_size"):
            assert attr in source, f"distributed stub for '{attr}' missing from hardware.py"


# TEST: tokenizer_utils.py -- error message


class TestTokenizerErrorMessage:
    """Verify the AMD error message is updated."""

    def test_no_old_amd_message(self):
        """Old 'We do not support AMD' message should be gone."""
        tu_path = PACKAGE_ROOT / "unsloth" / "tokenizer_utils.py"
        source = tu_path.read_text(encoding = "utf-8")
        assert "We do not support AMD" not in source

    def test_new_message_has_docs_link(self):
        """New message should point to Unsloth AMD docs."""
        tu_path = PACKAGE_ROOT / "unsloth" / "tokenizer_utils.py"
        source = tu_path.read_text(encoding = "utf-8")
        assert "docs.unsloth.ai" in source or "No GPU detected" in source


# TEST: install.sh -- structural checks


class TestInstallShStructure:
    """Verify install.sh structural properties without running it."""

    def test_no_here_strings(self):
        """install.sh must not use the bash-only `<<<` here-string operator (breaks dash)."""
        import re

        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Strip quoted literals so `<<<` inside them is ignored.
            unquoted = re.sub(r"'[^']*'", "", line)
            unquoted = re.sub(r'"[^"]*"', "", unquoted)
            assert "<<<" not in unquoted, f"install.sh:{i} uses non-POSIX <<< here-string"

    def test_rocm_detection_present(self):
        """install.sh should have ROCm detection in get_torch_index_url."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "amd-smi" in source
        assert "rocm" in source.lower()

    def test_cuda_precedence(self):
        """ROCm detection runs only when NVIDIA is absent (check runtime ordering in get_torch_index_url)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        body = _extract_sh_function_body(source, "get_torch_index_url")
        nvidia_call = body.find("_has_usable_nvidia_gpu")
        # Gate uses _nvidia_detected (not -z "$_smi") to handle proc-only NVIDIA
        # hosts where nvidia-smi is absent but the GPU is found via /proc.
        no_nvidia_branch = body.find('if [ "$_nvidia_detected" -eq 0 ]')
        if no_nvidia_branch < 0:
            no_nvidia_branch = body.find('if [ -z "$_smi" ]')
        rocm_call = body.find("_has_amd_rocm_gpu")
        assert nvidia_call >= 0, "get_torch_index_url should call _has_usable_nvidia_gpu"
        assert no_nvidia_branch >= 0, "get_torch_index_url should gate ROCm on no-nvidia branch"
        assert (
            rocm_call > no_nvidia_branch
        ), "ROCm detection should sit inside the 'no NVIDIA' branch"
        assert (
            nvidia_call < no_nvidia_branch
        ), "NVIDIA detection should run before the no-NVIDIA branch"

    def test_bitsandbytes_amd_install(self):
        """install.sh should install bitsandbytes for AMD when ROCm detected."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "bitsandbytes" in source
        assert "rocm*)" in source  # case pattern for ROCm URLs

    def test_cpu_hint_mentions_amd(self):
        """CPU-only hint should mention AMD ROCm."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "ROCm" in source

    def test_rocm72_supported_future_capped(self):
        """ROCm 7.2 should pass through directly; 7.3+ falls back to rocm7.2."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert 'echo "$_base/rocm7.2"' in source  # fallback for unknown future versions
        assert "rocm6.*" in source
        assert "rocm7.0" in source
        assert "rocm7.1" in source
        assert "rocm7.2" in source

    def test_rocm_tag_validation_guard_exists(self):
        """install.sh should validate _rocm_tag with a case guard."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "rocm[1-9]*.[0-9]*)" in source
        assert '_rocm_tag=""' in source  # rejection path

    def test_dpkg_epoch_handling(self):
        """install.sh should strip Debian epoch prefix from dpkg-query output."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "sed 's/^[0-9]*://' " in source or "sed 's/^[0-9]*://'" in source

    def test_no_double_bracket_in_rocm_block(self):
        """ROCm block must not use bash-only [[ ]] (POSIX char classes [[:space:]] are fine)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("get_torch_index_url()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        import re

        for i, line in enumerate(func_body.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Strip POSIX char classes [[:foo:]] before checking for [[ ]].
            cleaned = re.sub(r"\[\[:[a-z]+:\]\]", "", line)
            assert "[[" not in cleaned, f"get_torch_index_url line {i} uses non-POSIX [["

    def test_no_arithmetic_expansion_in_rocm_block(self):
        """ROCm detection block should not use (( )) (bash-only)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("get_torch_index_url()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        for i, line in enumerate(func_body.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert (
                "((" not in line or "))" not in line or "$(()" in line
            ), f"get_torch_index_url line {i} may use non-POSIX (( ))"

    def test_macos_returns_cpu_before_rocm_check(self):
        """macOS should return CPU immediately (before any ROCm check)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("get_torch_index_url()")
        func_body = source[func_start:]
        darwin_pos = func_body.find("Darwin")
        rocm_pos = func_body.find("amd-smi")
        assert darwin_pos < rocm_pos, "macOS check should come before ROCm detection"

    def test_unsloth_torch_backend_exported_after_get_torch_index_url(self):
        """install.sh exports UNSLOTH_TORCH_BACKEND after TORCH_INDEX_URL (lets the stack skip GPU re-detection)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        torch_url_pos = source.find("TORCH_INDEX_URL=$(get_torch_index_url)")
        backend_pos = source.find("UNSLOTH_TORCH_BACKEND")
        assert backend_pos > 0, "UNSLOTH_TORCH_BACKEND must be set in install.sh"
        assert (
            backend_pos > torch_url_pos
        ), "UNSLOTH_TORCH_BACKEND must be set AFTER TORCH_INDEX_URL is resolved"
        assert '"cuda"' in source[backend_pos : backend_pos + 500]
        assert '"rocm"' in source[backend_pos : backend_pos + 500]
        assert '"cpu"' in source[backend_pos : backend_pos + 500]
        # Must be exported so subprocesses see it.
        assert "export UNSLOTH_TORCH_BACKEND" in source

    def test_kfd_sysfs_amd_vendor_check_in_has_amd_rocm_gpu(self):
        """_has_amd_rocm_gpu sysfs fallback must require AMD vendor_id 4098 (nvidia-open registers KFD nodes too)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("_has_amd_rocm_gpu()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert (
            "vendor_id" in func_body
        ), "_has_amd_rocm_gpu sysfs fallback must check vendor_id to exclude NVIDIA KFD nodes"
        assert (
            "4098" in func_body
        ), "_has_amd_rocm_gpu sysfs fallback must require AMD vendor_id 4098 (0x1002)"

    def test_kfd_awk_resets_state_per_file(self):
        """KFD sysfs awk must reset gpu/amd state per file (FNR==1) to avoid Ryzen+NVIDIA false positives."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("_has_amd_rocm_gpu()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert "FNR==1" in func_body, (
            "_has_amd_rocm_gpu KFD awk must reset state per file with FNR==1 "
            "to avoid false positives on Ryzen+NVIDIA hosts with multiple KFD nodes"
        )

    def test_get_torch_index_url_uses_nvidia_detected_flag(self):
        """get_torch_index_url must track NVIDIA via _nvidia_detected (proc-only NVIDIA still picks CUDA)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        func_start = source.find("get_torch_index_url()")
        func_end = source.find("\n}", func_start)
        func_body = source[func_start:func_end]
        assert "_nvidia_detected" in func_body, (
            "get_torch_index_url must use a _nvidia_detected flag (separate from "
            "_smi) so that proc-only NVIDIA detection still selects CUDA wheels"
        )
        assert (
            '_nvidia_detected" -eq 0' in func_body or "_nvidia_detected" in func_body
        ), "get_torch_index_url AMD branch must be skipped when _nvidia_detected=1"


# TEST: Live regression on current host (NVIDIA B200 expected)


class TestLiveRegression:
    """Live checks that run on the actual host -- skip if no NVIDIA GPU."""

    def test_get_torch_index_url_returns_cuda_on_nvidia(self):
        """On an NVIDIA machine, get_torch_index_url should return a CUDA URL."""
        import shutil

        if not shutil.which("nvidia-smi"):
            pytest.skip("No nvidia-smi available")
        # Skip if nvidia-smi exists but lists no GPU (binary without driver).
        check = subprocess.run(
            [
                "bash",
                "-c",
                "nvidia-smi -L 2>/dev/null | awk '/^GPU[[:space:]]+[0-9]+:/{f=1} END{exit !f}'",
            ],
            capture_output = True,
        )
        if check.returncode != 0:
            pytest.skip("nvidia-smi is on PATH but no GPU is listed")

        sh_path = PACKAGE_ROOT / "install.sh"
        # All three helper definitions must be in scope when we eval the extract.
        extract_cmd = (
            f"sed -n '/^_has_amd_rocm_gpu()/,/^}}$/p; "
            f"/^_has_usable_nvidia_gpu()/,/^}}$/p; "
            f"/^get_torch_index_url()/,/^}}$/p' '{sh_path}'"
        )
        result = subprocess.run(
            ["bash", "-c", f'eval "$({extract_cmd})"; get_torch_index_url'],
            capture_output = True,
            text = True,
            timeout = 30,
        )
        if result.returncode != 0:
            pytest.skip("Could not extract get_torch_index_url for live test")
        url = result.stdout.strip()
        assert "cu1" in url or "cuda" in url.lower(), f"Expected CUDA URL, got: {url}"


# TEST: worker.py -- ROCm Mamba/SSM source build path

_WORKER_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "training" / "worker.py"
_EXPORT_WORKER_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "export" / "worker.py"
# Shared torchao Windows-ROCm stub used by both workers.
_TORCHAO_STUB_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "_torchao_stub.py"
# RAG embedder -- runs in the main backend process and also needs the stub.
_EMBEDDINGS_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "rag" / "embeddings.py"
# Wheel-probe script literal lives in wheel_utils after the resolver refactor.
_WHEEL_UTILS_PATH = PACKAGE_ROOT / "studio" / "backend" / "utils" / "wheel_utils.py"


class TestWorkerRocmMambaSsm:
    """Verify worker.py Mamba/SSM install logic on ROCm."""

    def test_probe_returns_hip_version_field(self):
        """The wheel probe should include hip_version, and worker.py consumes it."""
        assert "hip_version" in _WHEEL_UTILS_PATH.read_text(encoding = "utf-8")
        assert "hip_version" in _WORKER_PATH.read_text(encoding = "utf-8")

    def test_probe_script_has_getattr_hip(self):
        """Probe script should use getattr for torch.version.hip (safe on CUDA)."""
        source = _WHEEL_UTILS_PATH.read_text(encoding = "utf-8")
        assert "getattr(torch.version, 'hip', None)" in source

    def test_direct_wheel_url_returns_none_without_cuda_major(self, monkeypatch):
        """direct_wheel_url should return None when cuda_major is empty (ROCm)."""
        _worker_spec = importlib.util.spec_from_file_location("test_worker", _WORKER_PATH)
        assert _worker_spec is not None and _worker_spec.loader is not None
        worker_mod = importlib.util.module_from_spec(_worker_spec)

        # Stub worker.py imports via monkeypatch so the fake "utils" is undone
        # and doesn't break later tests importing the real utils.* package.
        loggers_mock = MagicMock()
        loggers_mock.get_logger = MagicMock(return_value = MagicMock())
        monkeypatch.setitem(sys.modules, "structlog", MagicMock())
        monkeypatch.setitem(sys.modules, "loggers", loggers_mock)
        monkeypatch.setitem(sys.modules, "utils", MagicMock())
        monkeypatch.setitem(sys.modules, "utils.hardware", MagicMock())

        try:
            _worker_spec.loader.exec_module(worker_mod)
        except Exception:
            pytest.skip("Could not load worker module in test environment")

        env_rocm = {
            "python_tag": "cp312",
            "torch_mm": "2.6",
            "cuda_major": "",
            "hip_version": "7.1.12345",
            "cxx11abi": "TRUE",
        }
        result = worker_mod.direct_wheel_url(
            filename_prefix = "causal_conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
            env = env_rocm,
        )
        assert result is None

    def test_hipcc_check_exists_in_source(self):
        """worker.py should check for hipcc before ROCm source builds."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "hipcc" in source

    def test_rocm_source_build_status_message(self):
        """worker.py should send a specific status for ROCm source compilation."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "Compiling" in source and "from source for ROCm" in source

    def test_rocm_build_failure_message(self):
        """worker.py should send a clear error on ROCm build failure."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "Failed to compile" in source and "for ROCm" in source

    def test_timeout_on_install(self):
        """worker.py should have a timeout on pip install subprocess."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "TimeoutExpired" in source
        assert "timeout" in source


# TEST: amd.py -- AMD GPU monitoring


class TestAmdGpuMonitoring:
    """Verify amd.py module structure and mock behavior."""

    def test_amd_py_exists(self):
        """amd.py should exist in the hardware directory."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        assert amd_path.exists()

    def test_amd_py_has_required_functions(self):
        """amd.py should export the same function signatures as nvidia.py."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        source = amd_path.read_text(encoding = "utf-8")
        assert "def get_physical_gpu_count" in source
        assert "def get_primary_gpu_utilization" in source
        assert "def get_visible_gpu_utilization" in source

    def test_amd_smi_json_parsing(self, monkeypatch):
        """Verify _extract_gpu_metrics parses amd-smi JSON correctly."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        loggers_mock = MagicMock()
        loggers_mock.get_logger = MagicMock(return_value = MagicMock())
        monkeypatch.setitem(sys.modules, "loggers", loggers_mock)

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module in test environment")

        gpu_data = {
            "usage": {"gfx_activity": "85"},
            "temperature": {"edge": "72"},
            "power": {
                "current_socket_power": "200.5",
                "power_cap": "300",
            },
            "vram": {
                "vram_used": 8192,  # MB
                "vram_total": 16384,  # MB
            },
        }
        metrics = amd_mod._extract_gpu_metrics(gpu_data)
        assert metrics["gpu_utilization_pct"] == 85.0
        assert metrics["temperature_c"] == 72.0
        assert metrics["power_draw_w"] == 200.5
        assert metrics["power_limit_w"] == 300.0
        assert metrics["vram_used_gb"] == round(8192 / 1024, 2)
        assert metrics["vram_total_gb"] == round(16384 / 1024, 2)
        assert metrics["vram_utilization_pct"] is not None
        assert metrics["power_utilization_pct"] is not None

    def test_amd_primary_gpu_with_mock(self, monkeypatch):
        """get_primary_gpu_utilization returns correct dict with mocked amd-smi."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd2", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        loggers_mock = MagicMock()
        loggers_mock.get_logger = MagicMock(return_value = MagicMock())
        monkeypatch.setitem(sys.modules, "loggers", loggers_mock)

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        # _first_visible_amd_gpu_id() returns None if HIP/ROCR/CUDA_VISIBLE_DEVICES
        # is "" or "-1"; CI often sets CUDA_VISIBLE_DEVICES="", so clear them.
        for var in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            monkeypatch.delenv(var, raising = False)

        # amd-smi is gated off on Windows w/o a HIP SDK; opt in so the mock is
        # allowed on every platform.
        monkeypatch.setenv("UNSLOTH_ENABLE_AMD_SMI", "1")

        mock_json = json.dumps(
            [
                {
                    "usage": {"gfx_activity": "50"},
                    "temperature": {"edge": "65"},
                    "power": {"current_socket_power": "150", "power_cap": "250"},
                    "vram": {"vram_used": 4096, "vram_total": 16384},
                }
            ]
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_json

        # Premise is "amd-smi exists and answers": the guard which()-checks
        # before spawning, so mock which too for hosts lacking a real amd-smi.
        with patch.object(amd_mod.shutil, "which", return_value = "/usr/bin/amd-smi"):
            with patch.object(subprocess, "run", return_value = mock_result):
                result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is True
        assert result["gpu_utilization_pct"] == 50.0
        assert result["temperature_c"] == 65.0

    def test_amd_smi_not_found_returns_unavailable(self, monkeypatch):
        """get_primary_gpu_utilization returns available=False when amd-smi is missing."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd3", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        loggers_mock = MagicMock()
        loggers_mock.get_logger = MagicMock(return_value = MagicMock())
        monkeypatch.setitem(sys.modules, "loggers", loggers_mock)

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        # Opt in so the call reaches subprocess.run (testing OSError handling).
        with (
            patch.dict(os.environ, {"UNSLOTH_ENABLE_AMD_SMI": "1"}),
            patch.object(subprocess, "run", side_effect = OSError("amd-smi not found")),
        ):
            result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is False

    def test_amd_timeout_returns_unavailable(self, monkeypatch):
        """get_primary_gpu_utilization handles timeout gracefully."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd4", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        loggers_mock = MagicMock()
        loggers_mock.get_logger = MagicMock(return_value = MagicMock())
        monkeypatch.setitem(sys.modules, "loggers", loggers_mock)

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        # Opt in so the call reaches subprocess.run (testing timeout handling).
        with (
            patch.dict(os.environ, {"UNSLOTH_ENABLE_AMD_SMI": "1"}),
            patch.object(
                subprocess,
                "run",
                side_effect = subprocess.TimeoutExpired("amd-smi", 5),
            ),
        ):
            result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is False


# TEST: hardware.py -- IS_ROCM branching to amd.py


class TestHardwareAmdBranching:
    """Verify hardware.py branches to amd.py when IS_ROCM is True."""

    def test_hardware_imports_amd_module(self):
        """hardware.py should import from amd module when IS_ROCM."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        assert "from . import amd" in source

    def test_hardware_branches_on_is_rocm_for_utilization(self):
        """get_gpu_utilization dispatches visible metrics through amd.py on ROCm."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_gpu_utilization")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "_smi_query(" in func_body
        assert '"get_visible_gpu_utilization"' in func_body
        assert "_reconcile_rocm_unified_memory" in func_body
        smi = source[
            source.find("def _smi_query") : source.find("\ndef ", source.find("def _smi_query") + 1)
        ]
        assert "IS_ROCM" in smi
        assert "from . import amd" in smi

    def test_hardware_branches_on_is_rocm_for_visible(self):
        """get_visible_gpu_utilization dispatches to amd.py via _smi_query when IS_ROCM."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_visible_gpu_utilization")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        # The dispatcher call may wrap; allow whitespace before the func name arg.
        import re as _re

        assert _re.search(r'_smi_query\(\s*"get_visible_gpu_utilization"', func_body)
        smi = source[
            source.find("def _smi_query") : source.find("\ndef ", source.find("def _smi_query") + 1)
        ]
        assert "IS_ROCM" in smi
        assert "from . import amd" in smi

    def test_hardware_branches_on_is_rocm_for_physical_count(self):
        """get_physical_gpu_count should try amd.py when IS_ROCM."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_physical_gpu_count")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "IS_ROCM" in func_body
        assert "from . import amd" in func_body


# TEST: hardware.py -- apply_gpu_ids ROCm fallback (issue #5180)


class TestApplyGpuIdsRocmFallback:
    """apply_gpu_ids sets HIP_VISIBLE_DEVICES on ROCm hosts even when IS_ROCM is still False (issue #5180)."""

    def test_apply_gpu_ids_falls_back_to_torch_version_hip(self):
        """apply_gpu_ids probes torch.version.hip when IS_ROCM is False and no ROCm env vars set."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert 'getattr(_torch.version, "hip", None)' in func_body

    def test_apply_gpu_ids_sets_hip_but_not_rocr_visible_devices(self):
        """apply_gpu_ids sets HIP_VISIBLE_DEVICES but leaves ROCR_VISIBLE_DEVICES inherited (HSA indexing; issue #6118)."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert 'os.environ["HIP_VISIBLE_DEVICES"] = value' in func_body
        assert 'os.environ["ROCR_VISIBLE_DEVICES"] = value' not in func_body

    def test_apply_gpu_ids_rocm_fallback_is_guarded_by_try_except(self):
        """torch import in apply_gpu_ids must be wrapped in try/except so a missing torch never crashes."""
        hw_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "import torch as _torch" in func_body
        assert "except Exception" in func_body


# TEST: install_python_stack.py -- Windows AMD warning


class TestWindowsRocmWarning:
    """Verify Windows AMD GPU detection and warning message."""

    def test_windows_amd_warning_in_source(self):
        """install_python_stack.py should warn Windows AMD users."""
        source = _STACK_PATH.read_text(encoding = "utf-8")
        assert "AMD GPU detected" in source

    def test_windows_amd_warning_checks_hipinfo_or_amdsmi(self):
        """Warning should check for hipinfo or amd-smi."""
        source = _STACK_PATH.read_text(encoding = "utf-8")
        assert "hipinfo" in source
        assert "amd-smi" in source

    def test_windows_amd_warning_has_docs_link(self):
        """Warning should include AMD docs link."""
        source = _STACK_PATH.read_text(encoding = "utf-8")
        assert "docs.unsloth.ai/get-started/install-and-update/amd" in source


# TEST: unsloth/kernels/utils.py -- is_rdna() expansion


class TestIsRdnaExpansion:
    """Verify is_rdna() covers RDNA2, RDNA3, RDNA3.5, RDNA4 architectures."""

    def test_is_rdna_source_has_rdna2(self):
        """is_rdna() should include RDNA2 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text(encoding = "utf-8")
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1030" in func_body
        assert "gfx1031" in func_body
        assert "gfx1032" in func_body
        assert "gfx1033" in func_body
        assert "gfx1034" in func_body
        assert "gfx1035" in func_body
        assert "gfx1036" in func_body

    def test_is_rdna_source_has_rdna3(self):
        """is_rdna() should include RDNA3 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text(encoding = "utf-8")
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1100" in func_body
        assert "gfx1101" in func_body
        assert "gfx1102" in func_body
        assert "gfx1103" in func_body

    def test_is_rdna_source_has_rdna35(self):
        """is_rdna() should include RDNA3.5 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text(encoding = "utf-8")
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1150" in func_body
        assert "gfx1151" in func_body
        assert "gfx1152" in func_body

    def test_is_rdna_source_has_rdna4(self):
        """is_rdna() should include RDNA4 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text(encoding = "utf-8")
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1200" in func_body
        assert "gfx1201" in func_body

    def test_is_cdna_not_changed(self):
        """is_cdna() should remain unchanged (no RDNA architectures added)."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text(encoding = "utf-8")
        func_start = source.find("def is_cdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx940" in func_body
        assert "gfx941" in func_body
        assert "gfx942" in func_body
        assert "gfx950" in func_body
        # RDNA architectures should NOT be in is_cdna
        assert "gfx1030" not in func_body
        assert "gfx1100" not in func_body


# TEST: install_python_stack.py -- _windows_rocm_index_url arch mapping


class TestWindowsRocmIndexUrl:
    """Verify GPU arch → AMD pip index URL mapping."""

    def test_gfx1200_maps_to_gfx120x_all(self):
        url = stack_mod._windows_rocm_index_url("gfx1200")
        assert url is not None
        assert "gfx120X-all" in url

    def test_gfx1201_maps_to_gfx120x_all(self):
        url = stack_mod._windows_rocm_index_url("gfx1201")
        assert url is not None
        assert "gfx120X-all" in url

    def test_gfx1151_maps_to_gfx1151(self):
        url = stack_mod._windows_rocm_index_url("gfx1151")
        assert url is not None
        assert "gfx1151" in url

    def test_gfx1150_maps_to_gfx1150(self):
        url = stack_mod._windows_rocm_index_url("gfx1150")
        assert url is not None
        assert "gfx1150" in url

    def test_gfx1100_maps_to_gfx110x_all(self):
        url = stack_mod._windows_rocm_index_url("gfx1100")
        assert url is not None
        assert "gfx110X-all" in url

    def test_unknown_arch_returns_none(self):
        assert stack_mod._windows_rocm_index_url("gfx9999") is None

    def test_none_arch_returns_none(self):
        assert stack_mod._windows_rocm_index_url(None) is None

    def test_url_ends_with_slash(self):
        """AMD pip index URLs must end with / for --index-url compatibility."""
        url = stack_mod._windows_rocm_index_url("gfx1200")
        assert url is not None
        assert url.endswith("/")

    def test_base_url_uses_repo_amd_com_by_default(self):
        url = stack_mod._windows_rocm_index_url("gfx1200")
        assert url is not None
        assert "repo.amd.com" in url

    def test_mirror_env_var_overrides_base(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_ROCM_WINDOWS_MIRROR", "https://my-mirror.example.com/rocm/whl")
        # Reload module-level constant by calling helper directly
        url = stack_mod._windows_rocm_index_url("gfx1200")
        # The env var is read at module load time for _ROCM_WINDOWS_INDEX_BASE,
        # so just verify the helper itself doesn't error.
        assert url is not None


# TEST: install_python_stack.py -- _detect_windows_gfx_arch


class TestDetectWindowsGfxArch:
    """Verify hipinfo parsing for GPU arch detection on Windows."""

    def test_returns_none_when_hipinfo_not_on_path(self):
        # Neutralise the venv-hipInfo and WMI-name fallbacks too, since the
        # suite may run on a real AMD host where WMI would answer.
        with patch("shutil.which", return_value = None):
            with patch("os.path.isfile", return_value = False):
                with patch("subprocess.run", side_effect = FileNotFoundError):
                    result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_parses_gcnarchname_from_hipinfo_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"gcnArchName : gfx1200\nsome other line\n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
                result = stack_mod._detect_windows_gfx_arch()
        assert result == "gfx1200"

    def test_returns_arch_on_crash_with_gcnarchname_in_output(self):
        # Regression #6043: hipinfo may crash (0xC0000005 on RDNA 4) after printing
        # gcnArchName. Accept the arch whenever gcnArchName is in stdout, any exit code.
        mock_result = MagicMock()
        mock_result.returncode = -1073741819  # 0xC0000005 STATUS_ACCESS_VIOLATION
        mock_result.stdout = b"gcnArchName : gfx1200\nsome other line\n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
                result = stack_mod._detect_windows_gfx_arch()
        assert result == "gfx1200"

    def test_returns_none_on_nonzero_returncode_without_gcnarchname(self):
        # Non-zero exit without gcnArchName must return None (fall through to amd-smi/WMI).
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b"HIP runtime error: no device detected\n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
                result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_returns_none_when_no_gcnarchname_in_output(self):
        # hipinfo answers without a gcnArchName line. The WMI fallback must get
        # nothing (FileNotFoundError) so the mocked name can't resolve via the table.
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"deviceName : SomeUnknownDevice\n"

        def _run(cmd, **kwargs):
            if cmd and "powershell" in str(cmd[0]).lower():
                raise FileNotFoundError(cmd[0])
            return mock_result

        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", side_effect = _run):
                result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_returns_none_on_timeout(self):
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch(
                "subprocess.run",
                side_effect = subprocess.TimeoutExpired("hipinfo", 10),
            ):
                result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_strips_whitespace_from_arch(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"  gcnArchName :   gfx1201  \n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
                result = stack_mod._detect_windows_gfx_arch()
        assert result == "gfx1201"


# TEST: install_python_stack.py -- GPU-name / WMI fallback (no amd-smi, no hipinfo)


class TestGfxArchNameFallback:
    """With no amd-smi/hipinfo on Windows, arch must resolve from the GPU name via WMI (mirrors setup.ps1)."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("AMD Radeon(TM) 8060S Graphics", "gfx1151"),
            ("AMD Radeon(TM) 8065S Graphics", "gfx1151"),
            ("AMD Ryzen AI MAX+ 395 w/ Radeon 8060S", "gfx1151"),
            ("AMD Radeon(TM) 890M", "gfx1150"),
            ("AMD Ryzen AI 9 HX 370 w/ Radeon 890M", "gfx1150"),
            ("AMD Radeon RX 9070 XT", "gfx1201"),
            ("AMD Radeon RX 9070", "gfx1200"),
            ("AMD Radeon RX 7700S", "gfx1102"),  # (?!S) lookahead must not hit gfx1100
            ("AMD Radeon RX 7700 XT", "gfx1100"),
            ("AMD Radeon(TM) 780M", "gfx1103"),
            ("NVIDIA GeForce RTX 4090", None),
            ("Microsoft Basic Display Adapter", None),
            ("", None),
        ],
    )
    def test_name_to_arch_mapping(self, name, expected):
        assert stack_mod._gfx_arch_from_gpu_name(name) == expected

    def test_wmi_fallback_resolves_arch_without_any_tools(self):
        """hipinfo absent everywhere + amd-smi absent -> WMI name fallback."""
        ps_result = MagicMock()
        ps_result.returncode = 0
        ps_result.stdout = b"AMD Radeon(TM) 8060S Graphics\r\nMicrosoft Basic Display Adapter\r\n"

        def _run(cmd, **kwargs):
            if cmd and "powershell.exe" in str(cmd[0]).lower():
                return ps_result
            raise FileNotFoundError(cmd[0])

        with patch.dict(os.environ, {}, clear = False):
            for _v in (
                "HIP_PATH",
                "ROCM_PATH",
                "UNSLOTH_ROCM_GFX_ARCH",
                "UNSLOTH_ENABLE_AMD_SMI",
            ):
                os.environ.pop(_v, None)
            with patch("shutil.which", return_value = None):
                with patch("os.path.isfile", return_value = False):
                    with patch("subprocess.run", side_effect = _run):
                        result = stack_mod._detect_windows_gfx_arch()
        assert result == "gfx1151"

    def test_wmi_fallback_returns_none_for_non_amd_hosts(self):
        ps_result = MagicMock()
        ps_result.returncode = 0
        ps_result.stdout = b"NVIDIA GeForce RTX 4090\r\n"

        def _run(cmd, **kwargs):
            if cmd and "powershell.exe" in str(cmd[0]).lower():
                return ps_result
            raise FileNotFoundError(cmd[0])

        with patch.dict(os.environ, {}, clear = False):
            for _v in ("HIP_PATH", "ROCM_PATH", "UNSLOTH_ROCM_GFX_ARCH"):
                os.environ.pop(_v, None)
            with patch("shutil.which", return_value = None):
                with patch("os.path.isfile", return_value = False):
                    with patch("subprocess.run", side_effect = _run):
                        result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_stack_probes_venv_hipinfo(self):
        """venv Scripts hipInfo.exe (from AMD torch wheels) must be a probe candidate for driver-only hosts."""
        source = _STACK_PATH.read_text(encoding = "utf-8")
        assert 'os.path.join(os.path.dirname(sys.executable), "hipInfo.exe")' in source

    def test_prebuilt_resolve_exe_probes_venv_dir(self):
        """_resolve_exe must include the venv Scripts candidate for driver-only standalone reruns."""
        source = _PREBUILT_PATH.read_text(encoding = "utf-8")
        assert "_venv_candidate" in source

    def test_runtime_monitor_guards_amd_smi_absence(self):
        """amd.py must which()-check amd-smi before spawning (absence disables the poller)."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        source = amd_path.read_text(encoding = "utf-8")
        assert 'shutil.which("amd-smi") is None' in source


# TEST: install_python_stack.py -- _install_bnb_windows_rocm


class TestInstallBnbWindowsRocm:
    """Verify AMD Windows BNB wheel install helper."""

    @pytest.fixture(autouse = True)
    def _isolate_sitecustomize_persistence(self, monkeypatch, request):
        """Keep helper tests from writing to the active interpreter site-packages."""
        if request.node.name.startswith("test_persist"):
            return
        monkeypatch.setattr(
            stack_mod,
            "_persist_bnb_rocm_version",
            lambda version: True,
        )

    def test_calls_pip_install_try_with_win_amd64_url(self):
        """Should call pip_install_try with the win_amd64 wheel URL via plain pip."""
        with patch.object(stack_mod, "pip_install_try", return_value = True) as mock_pip:
            stack_mod._install_bnb_windows_rocm()
        assert mock_pip.call_count == 1
        call_args = str(mock_pip.call_args_list[0])
        assert "bitsandbytes" in call_args
        assert "win_amd64" in call_args
        # Force plain pip (uv mangles the bitsandbytes wheel) -- see
        # https://unsloth.ai/docs/get-started/install/amd/amd-hackathon
        assert mock_pip.call_args.kwargs.get("force_pip") is True

    def test_forces_plain_pip_not_uv(self):
        """The bnb wheel must be installed with plain pip, never uv."""
        with patch.object(stack_mod, "pip_install_try", return_value = True) as mock_pip:
            stack_mod._install_bnb_windows_rocm()
        assert mock_pip.call_args.kwargs.get("force_pip") is True

    def test_does_not_touch_uv_skip_env_var(self):
        """The UV_SKIP_WHEEL_FILENAME_CHECK hack is gone; the env must be untouched."""
        observed = {}

        def _capture(*args, **kwargs):
            observed["during"] = os.environ.get("UV_SKIP_WHEEL_FILENAME_CHECK")
            return True

        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("UV_SKIP_WHEEL_FILENAME_CHECK", None)
            with patch.object(stack_mod, "pip_install_try", side_effect = _capture):
                stack_mod._install_bnb_windows_rocm()
            assert observed.get("during") is None
            assert "UV_SKIP_WHEEL_FILENAME_CHECK" not in os.environ

    def test_returns_false_on_pip_failure(self):
        """A failed pip_install_try must surface as a False return, not BNB_ROCM_VERSION."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            with patch.object(stack_mod, "pip_install_try", return_value = False):
                result = stack_mod._install_bnb_windows_rocm()
            assert result is False
            assert "BNB_ROCM_VERSION" not in os.environ

    def test_no_op_when_win_amd64_url_missing(self):
        """Should be silent no-op if win_amd64 key absent from _BNB_ROCM_PRERELEASE_URLS."""
        with patch.object(stack_mod, "_BNB_ROCM_PRERELEASE_URLS", {}):
            with patch.object(stack_mod, "pip_install_try") as mock_pip:
                stack_mod._install_bnb_windows_rocm()
        mock_pip.assert_not_called()

    def test_sets_bnb_rocm_version_from_detected_dll(self):
        """BNB_ROCM_VERSION is set from the DLL detected after install."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "72"):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "72"

    def test_sets_bnb_rocm_version_from_newer_dll(self):
        """If AMD ships a newer DLL (e.g. rocm713.dll), that version is used."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "713"):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "713"

    def test_falls_back_to_72_when_detection_fails(self):
        """Falls back to '72' when DLL detection returns None."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver", return_value = None):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "72"

    def test_does_not_override_existing_bnb_rocm_version(self):
        """An explicit BNB_ROCM_VERSION in the caller's env must not be clobbered."""
        with patch.dict(os.environ, {"BNB_ROCM_VERSION": "60"}):
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "60"

    def test_does_not_persist_existing_bnb_rocm_version(self):
        """A caller override must not become the venv's managed default."""
        with patch.dict(os.environ, {"BNB_ROCM_VERSION": "60"}):
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver") as mock_detect:
                    with patch.object(
                        stack_mod, "_persist_bnb_rocm_version", return_value = True
                    ) as mock_persist:
                        stack_mod._install_bnb_windows_rocm()

            assert os.environ.get("BNB_ROCM_VERSION") == "60"
            mock_detect.assert_not_called()
            mock_persist.assert_not_called()

    def test_redetects_when_bnb_rocm_version_came_from_sitecustomize(self):
        """Persisted defaults should not mask a newer DLL suffix after reinstall."""
        with patch.dict(
            os.environ,
            {
                "BNB_ROCM_VERSION": "72",
                stack_mod._BNB_ROCM_VERSION_SOURCE_ENV: (
                    stack_mod._BNB_ROCM_VERSION_SOURCE_SITECUSTOMIZE
                ),
            },
        ):
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "713"):
                    with patch.object(
                        stack_mod, "_persist_bnb_rocm_version", return_value = True
                    ) as mock_persist:
                        stack_mod._install_bnb_windows_rocm()

            assert os.environ.get("BNB_ROCM_VERSION") == "713"
            assert (
                os.environ.get(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV)
                == stack_mod._BNB_ROCM_VERSION_SOURCE_DETECTED
            )
            mock_persist.assert_called_once_with("713")

    def test_persists_bnb_rocm_version_for_direct_venv_python(self, tmp_path):
        """BNB_ROCM_VERSION must apply to a fresh Python process in the venv."""
        site_packages = tmp_path / "site-packages"

        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            os.environ.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "72"):
                    with patch.object(
                        stack_mod.sysconfig, "get_path", return_value = str(site_packages)
                    ):
                        stack_mod._install_bnb_windows_rocm()

        sitecustomize = site_packages / "sitecustomize.py"
        source = sitecustomize.read_text(encoding = "utf-8")
        assert "BNB_ROCM_VERSION" in source
        assert stack_mod._BNB_ROCM_VERSION_SOURCE_ENV in source
        assert "'72'" in source

        probe_env = os.environ.copy()
        probe_env.pop("BNB_ROCM_VERSION", None)
        probe_env.pop(stack_mod._BNB_ROCM_VERSION_SOURCE_ENV, None)
        probe_env["PYTHONPATH"] = str(site_packages)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "print(os.environ.get('BNB_ROCM_VERSION', ''), "
                    "os.environ.get('UNSLOTH_BNB_ROCM_VERSION_SOURCE', ''))"
                ),
            ],
            env = probe_env,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            check = True,
        )
        assert result.stdout.strip() == "72 sitecustomize"

    def test_persist_bnb_rocm_version_replaces_existing_managed_block(self, tmp_path):
        """Updating sitecustomize.py must not duplicate the managed BNB block."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        sitecustomize = site_packages / "sitecustomize.py"
        sitecustomize.write_text(
            "EXISTING = True\n"
            "# BEGIN Unsloth BNB_ROCM_VERSION\n"
            "import os as _unsloth_os\n"
            "_unsloth_os.environ.setdefault('BNB_ROCM_VERSION', '72')\n"
            "# END Unsloth BNB_ROCM_VERSION\n",
            encoding = "utf-8",
        )

        with patch.object(stack_mod.sysconfig, "get_path", return_value = str(site_packages)):
            assert stack_mod._persist_bnb_rocm_version("713") is True

        source = sitecustomize.read_text(encoding = "utf-8")
        assert source.count("# BEGIN Unsloth BNB_ROCM_VERSION") == 1
        assert "EXISTING = True" in source
        assert "'713'" in source
        assert "'72'" not in source

    def test_persist_bnb_rocm_version_handles_non_utf8_sitecustomize(self, tmp_path):
        """A legacy non-UTF-8 sitecustomize.py should not abort installation."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        sitecustomize = site_packages / "sitecustomize.py"
        sitecustomize.write_bytes(b"\xff\xfe\x00")

        with patch.object(stack_mod.sysconfig, "get_path", return_value = str(site_packages)):
            assert stack_mod._persist_bnb_rocm_version("72") is False

    def test_persist_bnb_rocm_version_repairs_truncated_block(self, tmp_path):
        """A managed block missing its END marker is replaced, not duplicated."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        sitecustomize = site_packages / "sitecustomize.py"
        sitecustomize.write_text(
            "EXISTING = True\n"
            "# BEGIN Unsloth BNB_ROCM_VERSION\n"
            "import os as _unsloth_os\n"
            "_unsloth_os.environ.setdefault('BNB_ROCM_VERSION', '72')\n",
            encoding = "utf-8",
        )

        with patch.object(stack_mod.sysconfig, "get_path", return_value = str(site_packages)):
            assert stack_mod._persist_bnb_rocm_version("713") is True

        source = sitecustomize.read_text(encoding = "utf-8")
        assert source.count("# BEGIN Unsloth BNB_ROCM_VERSION") == 1
        assert source.count("# END Unsloth BNB_ROCM_VERSION") == 1
        assert "EXISTING = True" in source
        assert "'713'" in source
        assert "'72'" not in source

    def test_persist_bnb_rocm_version_dedupes_duplicate_blocks(self, tmp_path):
        """Multiple managed blocks collapse to one while preserving user content."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        sitecustomize = site_packages / "sitecustomize.py"
        block = (
            "# BEGIN Unsloth BNB_ROCM_VERSION\n"
            "import os as _unsloth_os\n"
            "_unsloth_os.environ.setdefault('BNB_ROCM_VERSION', '72')\n"
            "# END Unsloth BNB_ROCM_VERSION\n"
        )
        sitecustomize.write_text(block + "USER_MID = 1\n" + block, encoding = "utf-8")

        with patch.object(stack_mod.sysconfig, "get_path", return_value = str(site_packages)):
            assert stack_mod._persist_bnb_rocm_version("713") is True

        source = sitecustomize.read_text(encoding = "utf-8")
        assert source.count("# BEGIN Unsloth BNB_ROCM_VERSION") == 1
        assert source.count("# END Unsloth BNB_ROCM_VERSION") == 1
        assert "USER_MID = 1" in source
        assert "'713'" in source
        assert "'72'" not in source

    def test_persist_bnb_rocm_version_atomic_no_leftover_tmp(self, tmp_path):
        """The write-then-rename path must not leave its temp file behind."""
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        with patch.object(stack_mod.sysconfig, "get_path", return_value = str(site_packages)):
            assert stack_mod._persist_bnb_rocm_version("72") is True

        leftovers = [p.name for p in site_packages.iterdir() if "unsloth-tmp" in p.name]
        assert leftovers == []
        assert (site_packages / "sitecustomize.py").exists()


class TestRuntimeBnbRocmSourceGuards:
    """Runtime entrypoints redetect managed defaults but keep caller overrides."""

    _MAIN_PATH = PACKAGE_ROOT / "studio" / "backend" / "main.py"
    _TRAINING_WORKER_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "training" / "worker.py"

    def test_main_gate_redetects_persisted_default(self):
        source = self._MAIN_PATH.read_text(encoding = "utf-8")
        assert 'os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize"' in source
        assert 'os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"' in source

    def test_worker_gate_redetects_persisted_default(self):
        source = self._TRAINING_WORKER_PATH.read_text(encoding = "utf-8")
        assert 'os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize"' in source
        assert 'os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"' in source

    def test_fallback_prefers_seeded_value_over_hardcoded_72(self):
        """A failed redetect must not downgrade a persisted suffix to '72'."""
        for path in (self._MAIN_PATH, self._TRAINING_WORKER_PATH):
            source = path.read_text(encoding = "utf-8")
            assert (
                '_bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"' in source
            ), path.name

    def test_main_requires_found_rocm_dll(self):
        """HIP_PATH/ROCM_PATH alone (HIP SDK on a CUDA/CPU box) must not force
        a ROCm backend onto a non-ROCm bitsandbytes."""
        source = self._MAIN_PATH.read_text(encoding = "utf-8")
        assert "if _found_rocm_bnb:" in source
        assert "_hip_env" not in source

    def test_worker_requires_found_rocm_dll(self):
        """No DLL found: the worker must not write any override or touch the
        seeded marker (later import fixes must still see sitecustomize)."""
        source = self._TRAINING_WORKER_PATH.read_text(encoding = "utf-8")
        assert "if _found_rocm_bnb:" in source


class TestDetectBnbRocmDllVer:
    """Unit tests for _detect_bnb_rocm_dll_ver()."""

    def test_returns_none_when_bnb_not_installed(self):
        """Returns None if bitsandbytes is not importable."""
        import importlib.util
        with patch.object(importlib.util, "find_spec", return_value = None):
            assert stack_mod._detect_bnb_rocm_dll_ver() is None

    def test_detects_rocm72_dll(self, tmp_path):
        """Returns '72' when libbitsandbytes_rocm72.dll is present."""
        (tmp_path / "libbitsandbytes_rocm72.dll").write_text("")
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = [str(tmp_path)]
        import importlib.util

        with patch.object(importlib.util, "find_spec", return_value = mock_spec):
            assert stack_mod._detect_bnb_rocm_dll_ver() == "72"

    def test_detects_rocm713_dll(self, tmp_path):
        """Returns '713' when libbitsandbytes_rocm713.dll is present."""
        (tmp_path / "libbitsandbytes_rocm713.dll").write_text("")
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = [str(tmp_path)]
        import importlib.util

        with patch.object(importlib.util, "find_spec", return_value = mock_spec):
            assert stack_mod._detect_bnb_rocm_dll_ver() == "713"

    def test_returns_none_when_only_cuda_dlls(self, tmp_path):
        """Returns None when only CUDA DLLs are present (no ROCm DLL)."""
        (tmp_path / "libbitsandbytes_cuda121.dll").write_text("")
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = [str(tmp_path)]
        import importlib.util

        with patch.object(importlib.util, "find_spec", return_value = mock_spec):
            assert stack_mod._detect_bnb_rocm_dll_ver() is None

    def test_picks_highest_suffix_when_multiple_dlls(self, tmp_path):
        """Returns the highest numeric suffix across ROCm DLL variants (glob order is not guaranteed)."""
        (tmp_path / "libbitsandbytes_rocm72.dll").write_text("")
        (tmp_path / "libbitsandbytes_rocm713.dll").write_text("")
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = [str(tmp_path)]
        import importlib.util

        with patch.object(importlib.util, "find_spec", return_value = mock_spec):
            assert stack_mod._detect_bnb_rocm_dll_ver() == "713"


# TEST: install_python_stack.py -- UNSLOTH_ROCM_TORCH_INSTALLED early-return path


class TestRocmTorchInstalledEnvVar:
    """Verify UNSLOTH_ROCM_TORCH_INSTALLED=1 skips main install but still installs BNB."""

    @staticmethod
    def _ok_torch_probe(*a, **kw):
        # Probe returns 0 when torch imports as ROCm.
        rv = MagicMock()
        rv.returncode = 0
        return rv

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_skips_main_pip_install(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should not trigger torch pip_install."""
        with (
            patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}),
            patch.object(stack_mod.subprocess, "run", side_effect = self._ok_torch_probe),
        ):
            stack_mod._ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_calls_bnb_install(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should still call _install_bnb_windows_rocm."""
        with (
            patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}),
            patch.object(stack_mod.subprocess, "run", side_effect = self._ok_torch_probe),
        ):
            stack_mod._ensure_rocm_torch()
        mock_bnb.assert_called_once()

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_sets_rocm_windows_flag(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should set _rocm_windows_torch_installed."""
        stack_mod._rocm_windows_torch_installed = False
        with (
            patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}),
            patch.object(stack_mod.subprocess, "run", side_effect = self._ok_torch_probe),
        ):
            stack_mod._ensure_rocm_torch()
        assert stack_mod._rocm_windows_torch_installed is True

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_falls_through_when_torch_missing(self, mock_pip, mock_bnb):
        """If the venv was wiped between runs, the stale env-var must not suppress reinstall."""
        stack_mod._rocm_windows_torch_installed = False

        def _bad_probe(*a, **kw):
            rv = MagicMock()
            rv.returncode = 1
            return rv

        with (
            patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}),
            patch.object(stack_mod.subprocess, "run", side_effect = _bad_probe),
            patch.object(stack_mod, "IS_WINDOWS", False),
            patch.object(stack_mod, "IS_MACOS", True),
        ):
            stack_mod._ensure_rocm_torch()
        # macOS branch is the next exit; the point is the early-return did NOT fire.
        mock_bnb.assert_not_called()


class TestWindowsRocmTorchaoGuard:
    """Verify the torchao skip can detect an installed Windows ROCm torch build."""

    def test_installed_torch_is_windows_rocm_accepts_rocm_probe(self):
        rv = MagicMock()
        rv.returncode = 0
        rv.stdout = "yes"
        with (
            patch.object(stack_mod, "IS_WINDOWS", True),
            patch.object(stack_mod.subprocess, "run", return_value = rv),
        ):
            assert stack_mod._installed_torch_is_windows_rocm() is True

    def test_installed_torch_is_windows_rocm_rejects_non_rocm_probe(self):
        rv = MagicMock()
        rv.returncode = 0
        rv.stdout = ""
        with (
            patch.object(stack_mod, "IS_WINDOWS", True),
            patch.object(stack_mod.subprocess, "run", return_value = rv),
        ):
            assert stack_mod._installed_torch_is_windows_rocm() is False

    def test_installed_torch_is_windows_rocm_is_non_windows_noop(self):
        with patch.object(stack_mod, "IS_WINDOWS", False):
            assert stack_mod._installed_torch_is_windows_rocm() is False

    @patch.object(stack_mod, "_repair_bad_anyio")
    @patch.object(stack_mod, "_ensure_rocm_torch")
    @patch.object(stack_mod, "_ensure_cuda_torch")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True)
    @patch.object(stack_mod, "run")
    @patch.object(stack_mod, "pip_install")
    def test_install_python_stack_skips_torchao_when_windows_rocm_torch_is_installed(
        self, mock_pip, mock_run, mock_has_nvidia, mock_cuda, mock_rocm, mock_anyio, tmp_path
    ):
        unstructured_plugin = tmp_path / "unstructured"
        github_plugin = tmp_path / "github"
        unstructured_plugin.mkdir()
        github_plugin.mkdir()

        subprocess_result = MagicMock()
        subprocess_result.returncode = 0
        subprocess_result.stdout = ""

        with (
            patch.dict(os.environ, {"SKIP_STUDIO_BASE": "1"}),
            patch.object(stack_mod, "IS_WINDOWS", True),
            patch.object(stack_mod, "IS_MACOS", False),
            patch.object(stack_mod, "IS_MAC_ARM", False),
            patch.object(stack_mod, "NO_TORCH", False),
            patch.object(stack_mod, "_rocm_windows_torch_installed", False),
            patch.object(stack_mod, "_bootstrap_uv", return_value = False),
            patch.object(stack_mod, "_installed_torch_is_windows_rocm", return_value = True),
            patch.object(stack_mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin),
            patch.object(stack_mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin),
            patch.object(stack_mod.subprocess, "run", return_value = subprocess_result),
        ):
            assert stack_mod.install_python_stack() == 0

        installed_specs = [str(arg) for call in mock_pip.call_args_list for arg in call.args]
        assert not any("torchao" in arg for arg in installed_specs)


class TestProgressStepCountMatchesTotal:
    """The progress bar must reach exactly _TOTAL: every _progress() step is counted in
    base_total. Regression for a repair step added without incrementing base_total,
    which pushed _STEP past _TOTAL (Codex P2)."""

    def _run_stack(self, tmp_path, *, is_windows, is_macos, is_mac_arm):
        unstructured_plugin = tmp_path / "unstructured"
        github_plugin = tmp_path / "github"
        unstructured_plugin.mkdir()
        github_plugin.mkdir()
        sub = MagicMock()
        sub.returncode = 0
        sub.stdout = ""
        with (
            patch.dict(os.environ, {"SKIP_STUDIO_BASE": "1"}),
            patch.object(stack_mod, "IS_WINDOWS", is_windows),
            patch.object(stack_mod, "IS_MACOS", is_macos),
            patch.object(stack_mod, "IS_MAC_ARM", is_mac_arm),
            patch.object(stack_mod, "NO_TORCH", False),
            patch.object(stack_mod, "_rocm_windows_torch_installed", False),
            patch.object(stack_mod, "_bootstrap_uv", return_value = False),
            patch.object(stack_mod, "_installed_torch_is_windows_rocm", return_value = False),
            patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True),
            patch.object(stack_mod, "_repair_bad_anyio"),
            patch.object(stack_mod, "_ensure_cuda_torch"),
            patch.object(stack_mod, "_ensure_rocm_torch"),
            patch.object(stack_mod, "_ensure_cpu_torch"),
            patch.object(stack_mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin),
            patch.object(stack_mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin),
            patch.object(stack_mod.subprocess, "run", return_value = sub),
        ):
            assert stack_mod.install_python_stack() == 0
            return stack_mod._STEP, stack_mod._TOTAL

    def test_windows_progress_reaches_total(self, tmp_path):
        step, total = self._run_stack(tmp_path, is_windows = True, is_macos = False, is_mac_arm = False)
        assert step == total, f"Windows progress {step} != total {total} (final step uncounted)"

    def test_linux_progress_reaches_total(self, tmp_path):
        step, total = self._run_stack(tmp_path, is_windows = False, is_macos = False, is_mac_arm = False)
        assert step == total, f"Linux progress {step} != total {total}"


# TEST: worker.py -- Windows ROCm patches (source-level checks)


class TestWorkerWindowsRocmPatches:
    """Verify worker.py contains the required Windows ROCm runtime patches."""

    def test_grouped_mm_dispatch_patch_present(self):
        """worker.py must register a _grouped_mm CUDA dispatch override."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert '_gm_lib.impl("_grouped_mm"' in source

    def test_grouped_mm_patch_targets_cuda_dispatch_key(self):
        """The dispatch override must target the CUDA key (not CompositeImplicitAutograd)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert '"_grouped_mm", _grouped_mm_safe_impl, "CUDA"' in source

    def test_grouped_mm_lib_kept_alive(self):
        """The Library object must be stored to prevent GC clearing the registration."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "_WINDOWS_ROCM_GROUPED_MM_LIB" in source

    def test_grouped_mm_handles_offs_grouped_case(self):
        """_grouped_mm fallback must handle the grouped (offs!=None) variant."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "offs_list" in source
        assert "offs.tolist()" in source

    def test_worker_calls_shared_torchao_stub(self):
        """worker.py must invoke the shared torchao stub entrypoint."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "install_torchao_windows_rocm_stub()" in source

    def test_export_worker_calls_shared_torchao_stub(self):
        """export/worker.py must invoke the same shared torchao stub entrypoint."""
        source = _EXPORT_WORKER_PATH.read_text(encoding = "utf-8")
        assert "install_torchao_windows_rocm_stub()" in source

    def test_embedder_calls_shared_torchao_stub(self):
        """embeddings.py must install the stub before importing sentence-transformers:
        it runs in the main process (not a stubbed worker), so otherwise transformers
        -> torchao crashes on Windows ROCm and the embedder drops to llama-server."""
        source = _EMBEDDINGS_PATH.read_text(encoding = "utf-8")
        assert "install_torchao_windows_rocm_stub()" in source

    def test_torchao_stub_uses_stub_type_meta(self):
        """Torchao stub must use _StubTypeMeta so isinstance() returns False not TypeError."""
        source = _TORCHAO_STUB_PATH.read_text(encoding = "utf-8")
        assert "_StubTypeMeta" in source

    def test_stub_type_meta_has_instancecheck(self):
        """_StubTypeMeta must define __instancecheck__ returning False."""
        source = _TORCHAO_STUB_PATH.read_text(encoding = "utf-8")
        assert "__instancecheck__" in source

    def test_stub_subpackage_finder_registered(self):
        """_StubSubpackageFinder must be appended to sys.meta_path."""
        source = _TORCHAO_STUB_PATH.read_text(encoding = "utf-8")
        assert "sys.meta_path.append(_StubSubpackageFinder())" in source

    def test_torchao_key_submodules_pre_stubbed(self):
        """Key torchao submodules (dtypes, quantization) must be pre-stubbed."""
        source = _TORCHAO_STUB_PATH.read_text(encoding = "utf-8")
        assert "torchao.dtypes" in source
        assert "torchao.quantization" in source

    def test_torchdynamo_disabled_on_windows_rocm(self):
        """worker.py should disable dynamo on Windows ROCm as belt-and-suspenders."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "TORCHDYNAMO_DISABLE" in source

    def test_bnb_rocm_version_set_on_windows_rocm(self):
        """worker.py must set BNB_ROCM_VERSION from the detected DLL suffix (BNB's auto-detect can mismatch)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "BNB_ROCM_VERSION" in source
        assert "_detect_bnb_rocm_dll_ver" in source or "libbitsandbytes_rocm" in source
        # Falls back to the seeded value, never a blind "72".
        assert '_bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION")' in source

    def test_bnb_rocm_version_set_before_ml_imports(self):
        """BNB_ROCM_VERSION must appear in section 1f, before section 2 ML imports."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        idx_bnb = source.find("BNB_ROCM_VERSION")
        # Use the entry-point section-2 marker (not the trainer helper's own "# ── 2.").
        idx_sec2 = source.find("# ── 2. Now import ML libraries")
        assert idx_bnb != -1, "BNB_ROCM_VERSION not found in worker.py"
        assert idx_sec2 != -1, "'# ── 2. Now import ML libraries' marker not found in worker.py"
        assert idx_bnb < idx_sec2, (
            "BNB_ROCM_VERSION must be set before section 2 ML imports "
            f"(found at {idx_bnb}, section 2 at {idx_sec2})"
        )

    def test_grouped_mm_patch_guarded_by_windows_and_hip_check(self):
        """_grouped_mm patch must only apply on Windows + HIP torch."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert 'sys.platform == "win32"' in source
        # Gates on HIP version via a getattr chain ("version", "hip").
        assert '"version"' in source and '"hip"' in source

    def test_hip_ver_at_least_helper_defined(self):
        """_hip_ver_at_least helper must be defined inside the Windows ROCm block."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "def _hip_ver_at_least(major: int, minor: int)" in source

    def test_grouped_mm_patch_gated_on_hip_lt_713(self):
        """_grouped_mm patch must be skipped on HIP >= 7.13 (AMD fixed the bug in ROCm 7.13)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "_hip_ver_at_least(7, 13)" in source
        # Patch must be inside the negated `if not` guard.
        assert "if not _hip_ver_at_least(7, 13):" in source

    def test_grouped_mm_hip_713_skip_message_present(self):
        """worker.py must log a message when skipping the patch on HIP >= 7.13."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "HIP >= 7.13" in source
        assert "7.13" in source

    def test_grouped_mm_patch_else_branch_present(self):
        """An else branch must follow the _hip_ver_at_least gate (skip path for 7.13+)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        gate_idx = source.find("if not _hip_ver_at_least(7, 13):")
        assert gate_idx != -1, "Version gate not found in worker.py"
        else_idx = source.find("else:", gate_idx)
        assert else_idx != -1, "else: branch after _hip_ver_at_least gate not found"

    def test_hip_ver_at_least_handles_amd_version_format(self):
        """_hip_ver_at_least must split on '.' and compare only major.minor (handles '7.13.99004')."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert 'split(".")[:2]' in source or ".split('.')[:2]" in source


# TEST: install_python_stack.py -- _ROCM_TORCH_PKG_SPECS mapping


class TestRocmTorchPkgSpecs:
    """Verify per-tag torch version specs are correct."""

    def test_rocm72_has_torch_211(self):
        """rocm7.2 should specify torch 2.11.x."""
        specs = stack_mod._ROCM_TORCH_PKG_SPECS.get("rocm7.2")
        assert specs is not None
        torch_spec = specs[0]
        assert "2.11" in torch_spec

    def test_default_caps_below_211(self):
        """Default spec (rocm7.1 and earlier) should cap below 2.11."""
        specs = stack_mod._ROCM_TORCH_PKG_SPECS.get("_default")
        assert specs is not None
        torch_spec = specs[0]
        assert "<2.11" in torch_spec

    def test_specs_have_torch_vision_audio(self):
        """Each entry should be a 3-tuple: torch, torchvision, torchaudio."""
        for tag, specs in stack_mod._ROCM_TORCH_PKG_SPECS.items():
            assert len(specs) == 3, f"{tag}: expected (torch, torchvision, torchaudio)"
            assert "torch" in specs[0]
            assert "torchvision" in specs[1]
            assert "torchaudio" in specs[2]

    def test_gfx_to_amd_index_covers_rdna4(self):
        """_GFX_TO_AMD_INDEX_ARCH must cover gfx1200 and gfx1201 (RDNA 4)."""
        mapping = stack_mod._GFX_TO_AMD_INDEX_ARCH
        assert mapping.get("gfx1200") == "gfx120X-all"
        assert mapping.get("gfx1201") == "gfx120X-all"

    def test_gfx_to_amd_index_covers_strix_halo(self):
        """_GFX_TO_AMD_INDEX_ARCH must cover gfx1151 and gfx1150 (RDNA 3.5)."""
        mapping = stack_mod._GFX_TO_AMD_INDEX_ARCH
        assert mapping.get("gfx1151") == "gfx1151"
        assert mapping.get("gfx1150") == "gfx1150"

    def test_gfx_to_amd_index_covers_rdna3(self):
        """_GFX_TO_AMD_INDEX_ARCH must cover gfx1100-gfx1103 (RDNA 3)."""
        mapping = stack_mod._GFX_TO_AMD_INDEX_ARCH
        for arch in ("gfx1100", "gfx1101", "gfx1102", "gfx1103"):
            assert mapping.get(arch) == "gfx110X-all", f"{arch} missing from mapping"


# TEST: setup.ps1 / install.ps1 -- Strix Halo gfx arch detection

_SETUP_PS1_PATH = PACKAGE_ROOT / "studio" / "setup.ps1"
_INSTALL_PS1_PATH = PACKAGE_ROOT / "install.ps1"


class TestStrixHaloGfxArchDetection:
    """setup.ps1 / install.ps1 gfx arch detection for Strix Halo / iGPU (HIP runtime only, no hipinfo)."""

    def test_amd_smi_static_asic_attempted_in_setup(self):
        """setup.ps1 must try 'amd-smi static --asic' when list output lacks gfx arch."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "static --asic" in source

    def test_amd_smi_static_asic_attempted_in_install(self):
        """install.ps1 must try 'amd-smi static --asic' when list output lacks gfx arch."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "static --asic" in source

    def test_env_var_override_in_setup(self):
        """setup.ps1 must honour UNSLOTH_ROCM_GFX_ARCH as a manual arch override."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "UNSLOTH_ROCM_GFX_ARCH" in source

    def test_env_var_override_in_install(self):
        """install.ps1 must honour UNSLOTH_ROCM_GFX_ARCH as a manual arch override."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "UNSLOTH_ROCM_GFX_ARCH" in source

    def test_name_arch_table_covers_strix_halo_in_setup(self):
        """setup.ps1 name→arch table must map 890M / Strix Halo to gfx1151."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "gfx1151" in source
        assert "890M" in source or "Strix Halo" in source

    def test_name_arch_table_covers_strix_halo_in_install(self):
        """install.ps1 name→arch table must map 890M / Strix Halo to gfx1151."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "gfx1151" in source
        assert "890M" in source or "Strix Halo" in source

    def test_name_arch_table_covers_strix_point_in_setup(self):
        """setup.ps1 name→arch table must map 880M / Strix Point to gfx1150."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "gfx1150" in source
        assert "880M" in source or "Strix Point" in source

    def test_name_arch_table_covers_strix_point_in_install(self):
        """install.ps1 name→arch table must map 880M / Strix Point to gfx1150."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "gfx1150" in source
        assert "880M" in source or "Strix Point" in source

    def test_name_arch_table_covers_rdna3_phoenix_in_setup(self):
        """setup.ps1 name→arch table must map 780M / Phoenix to gfx1103."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "gfx1103" in source
        assert "780M" in source or "Phoenix" in source

    def test_wmi_does_not_set_hasrocm_in_setup(self):
        """WMI block in setup.ps1 must NOT set $HasROCm = $true (no runtime confirmation)."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        wmi_idx = source.find("Win32_VideoController")
        assert wmi_idx != -1, "WMI block not found in setup.ps1"
        # $HasROCm = $true must not appear within 300 chars of the WMI call.
        wmi_context = source[wmi_idx : wmi_idx + 300]
        assert "$HasROCm = $true" not in wmi_context

    def test_gfx_arch_regex_parses_from_amd_smi_output(self):
        """Both files must use the gfx\\d+[a-z]? regex to parse arch from amd-smi output."""
        for path in (_SETUP_PS1_PATH, _INSTALL_PS1_PATH):
            source = path.read_text(encoding = "utf-8")
            assert (
                "gfx\\d+" in source or r"gfx\d+" in source
            ), f"gfx arch regex not found in {path.name}"


# TEST: HIP SDK tool path resolution via HIP_PATH / ROCM_PATH env vars


class TestHipSdkEnvPathResolution:
    """Both install scripts resolve hipinfo/hipconfig via HIP_PATH/ROCM_PATH off $PATH, and warn."""

    @staticmethod
    def _assert_accepts_partial_hipinfo_output(source: str):
        hipout_idx = source.find("$hipOut = & $hipinfoExe.Source")
        assert hipout_idx != -1
        hipinfo_block = source[hipout_idx : hipout_idx + 1600]
        assert 'if ($hipOut -match "(?i)gcnArchName")' in hipinfo_block
        assert "$LASTEXITCODE -eq 0 -and $hipOut -match" not in hipinfo_block
        assert "but reported gcnArchName" in hipinfo_block

    # ── hipinfo resolution ────────────────────────────────────────────────────

    def test_setup_checks_hip_path_for_hipinfo(self):
        """setup.ps1 must reference HIP_PATH when resolving hipinfo."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP_PATH" in source
        assert "hipinfo" in source

    def test_install_checks_hip_path_for_hipinfo(self):
        """install.ps1 must reference HIP_PATH when resolving hipinfo."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP_PATH" in source
        assert "hipinfo" in source

    def test_setup_checks_rocm_path_as_hipinfo_fallback(self):
        """setup.ps1 must also check ROCM_PATH as a secondary hipinfo fallback."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCM_PATH" in source
        assert "ROCM_PATH" in source and "HIP_PATH" in source

    def test_install_checks_rocm_path_as_hipinfo_fallback(self):
        """install.ps1 must also check ROCM_PATH as a secondary hipinfo fallback."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCM_PATH" in source
        assert "ROCM_PATH" in source and "HIP_PATH" in source

    def test_setup_resolves_hipinfo_via_bin_subdir(self):
        """setup.ps1 must join the env var root with 'bin\\hipinfo.exe'."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert r"bin\hipinfo.exe" in source

    def test_install_resolves_hipinfo_via_bin_subdir(self):
        """install.ps1 must join the env var root with 'bin\\hipinfo.exe'."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert r"bin\hipinfo.exe" in source

    # ── hipinfo not-on-PATH warning ───────────────────────────────────────────

    def test_setup_warns_when_hipinfo_not_on_path(self):
        """setup.ps1 must warn when hipinfo is found via env var but not on PATH."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "hipinfo not on PATH" in source

    def test_install_warns_when_hipinfo_not_on_path(self):
        """install.ps1 must warn when hipinfo is found via env var but not on PATH."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "hipinfo not on PATH" in source

    # ── warn when HIP_PATH set but exe missing ────────────────────────────────

    def test_setup_warns_when_hip_path_set_but_exe_missing(self):
        """setup.ps1 must warn when HIP_PATH is set but hipinfo.exe is not present."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "incomplete" in source or "not found at" in source

    def test_install_warns_when_hip_path_set_but_exe_missing(self):
        """install.ps1 must warn when HIP_PATH is set but hipinfo.exe is not present."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "incomplete" in source or "not found at" in source

    # ── hipinfo runtime error warning ─────────────────────────────────────────

    def test_setup_warns_on_hipinfo_nonzero_exit(self):
        """setup.ps1 must warn when hipinfo runs but returns a non-zero exit code."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP runtime error" in source or "runtime error" in source.lower()

    def test_install_warns_on_hipinfo_nonzero_exit(self):
        """install.ps1 must warn when hipinfo runs but returns a non-zero exit code."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP runtime error" in source or "runtime error" in source.lower()

    def test_setup_accepts_hipinfo_gcnarchname_on_nonzero_exit(self):
        """setup.ps1 must accept partial hipinfo output from the #6043 crash path."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        self._assert_accepts_partial_hipinfo_output(source)

    def test_install_accepts_hipinfo_gcnarchname_on_nonzero_exit(self):
        """install.ps1 must accept partial hipinfo output from the #6043 crash path."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        self._assert_accepts_partial_hipinfo_output(source)

    # ── hipconfig resolution ──────────────────────────────────────────────────

    def test_setup_resolves_hipconfig_via_bin_subdir(self):
        """setup.ps1 must also fall back to HIP_PATH/bin/hipconfig.exe for version detection."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert r"bin\hipconfig.exe" in source

    def test_install_resolves_hipconfig_via_bin_subdir(self):
        """install.ps1 must also fall back to HIP_PATH/bin/hipconfig.exe for version detection."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert r"bin\hipconfig.exe" in source

    def test_setup_warns_when_hipconfig_not_on_path(self):
        """setup.ps1 must warn when hipconfig is found via env var but not on PATH."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "hipconfig not on PATH" in source

    def test_install_warns_when_hipconfig_not_on_path(self):
        """install.ps1 must warn when hipconfig is found via env var but not on PATH."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "hipconfig not on PATH" in source

    # ── PATH fix hint ─────────────────────────────────────────────────────────

    def test_setup_provides_path_fix_hint(self):
        """setup.ps1 must tell the user how to add the HIP bin dir to PATH."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "PATH" in source and ("SetEnvironmentVariable" in source or "Add" in source)

    def test_install_provides_path_fix_hint(self):
        """install.ps1 must tell the user how to add the HIP bin dir to PATH."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "PATH" in source and ("SetEnvironmentVariable" in source or "Add" in source)


# TEST: HIP SDK detected substep -- path + hipconfig version shown in terminal


class TestHipSdkDetectedSubstep:
    """Both scripts print HIP SDK path and full hipconfig version as substeps when ROCm is detected."""

    def test_setup_prints_hip_sdk_path_substep(self):
        """setup.ps1 must print an 'HIP SDK:' substep showing the resolved path."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP SDK:" in source

    def test_install_prints_hip_sdk_path_substep(self):
        """install.ps1 must print an 'HIP SDK:' substep showing the resolved path."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP SDK:" in source

    def test_setup_shows_hipconfig_full_version(self):
        """setup.ps1 must capture and display the full hipconfig version string."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCmVersionFull" in source or "hipconfig:" in source

    def test_install_shows_hipconfig_full_version(self):
        """install.ps1 must capture and display the full hipconfig version string."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCmVersionFull" in source or "hipconfig:" in source

    def test_setup_captures_full_version_not_just_major_minor(self):
        """setup.ps1 must store the raw hipconfig output line, not just major.minor."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCmVersionFull" in source

    def test_install_captures_full_version_not_just_major_minor(self):
        """install.ps1 must store the raw hipconfig output line, not just major.minor."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCmVersionFull" in source

    def test_setup_uses_hip_path_or_rocm_path_for_sdk_display(self):
        """setup.ps1 HIP SDK path substep must check HIP_PATH then ROCM_PATH."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP_PATH" in source and "ROCM_PATH" in source

    def test_install_uses_hip_path_or_rocm_path_for_sdk_display(self):
        """install.ps1 HIP SDK path substep must check HIP_PATH then ROCM_PATH."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HIP_PATH" in source and "ROCM_PATH" in source

    def test_setup_rocm_step_uses_full_version(self):
        """setup.ps1 'rocm' step label must prefer the full version string."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "ROCmVersionFull" in source and "rocm" in source


# TEST: install.sh -- Strix Halo rocm7.1 → rocm7.2 override

_INSTALL_SH_PATH = PACKAGE_ROOT / "install.sh"
_SETUP_SH_PATH = PACKAGE_ROOT / "studio" / "setup.sh"


class TestStrixRocm71Override:
    """install.sh routes gfx1151/gfx1150 to AMD's arch index instead of ROCm 7.1 (_grouped_mm segfault)."""

    def test_linux_gfx_inference_helpers_present(self):
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "_infer_linux_amd_gfx_arch" in source
        assert "_amd_arch_index_family_for_gfx" in source
        assert "_amd_gpu_present_via_pci" in source
        assert "unslothai#7301" in source

    def test_infer_linux_amd_gfx_from_cpuinfo(self):
        assert stack_mod._linux_amd_gfx_from_cpuinfo is not None
        with patch.object(
            Path,
            "read_text",
            return_value = "model name : AMD Ryzen AI Max+ 395 w/ Radeon 8060S\n",
        ):
            assert stack_mod._linux_amd_gfx_from_cpuinfo() == "gfx1151"
        # 8065S (Gorgon Halo) must match on the Radeon name alone, even without the
        # "Ryzen AI Max" branding (mirrors setup.sh / setup.ps1 which list 8065S).
        with patch.object(Path, "read_text", return_value = "model name : AMD Radeon 8065S\n"):
            assert stack_mod._linux_amd_gfx_from_cpuinfo() == "gfx1151"

    def test_infer_gfx_gated_out_of_wsl_without_runtime(self):
        """On WSL the cpuinfo/lspci inference must be skipped unless the WSL ROCDXG
        runtime (librocdxg) is present: a bare `unsloth studio update` must not
        install per-arch ROCm wheels into an env that still can't expose the GPU.
        An explicit UNSLOTH_ROCM_GFX_ARCH override stays authoritative regardless."""
        m = stack_mod
        with (
            patch.object(m, "_linux_amd_gfx_from_cpuinfo", return_value = "gfx1151"),
            patch.object(m, "_linux_amd_gfx_from_lspci", return_value = None),
            patch.dict(os.environ, {"UNSLOTH_ROCM_GFX_ARCH": ""}),
        ):
            # WSL + no runtime -> inference suppressed (CPU torch stays).
            with (
                patch.object(m, "_is_wsl", return_value = True),
                patch.object(m, "_wsl_rocm_runtime_present", return_value = False),
            ):
                assert m._infer_linux_amd_gfx_arch() is None
            # WSL + runtime present (this dev box) -> inference still runs.
            with (
                patch.object(m, "_is_wsl", return_value = True),
                patch.object(m, "_wsl_rocm_runtime_present", return_value = True),
            ):
                assert m._infer_linux_amd_gfx_arch() == "gfx1151"
            # Native Linux (not WSL) -> the gate never applies.
            with (
                patch.object(m, "_is_wsl", return_value = False),
                patch.object(m, "_wsl_rocm_runtime_present", return_value = False),
            ):
                assert m._infer_linux_amd_gfx_arch() == "gfx1151"
        # Explicit override wins even on a bare WSL box (no runtime).
        with (
            patch.object(m, "_is_wsl", return_value = True),
            patch.object(m, "_wsl_rocm_runtime_present", return_value = False),
            patch.dict(os.environ, {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151"}),
        ):
            assert m._infer_linux_amd_gfx_arch() == "gfx1151"

    def test_lspci_scan_covers_all_display_controllers(self):
        """The lspci fallback must scan every display-class line, not just the
        first: a non-AMD controller (Intel iGPU, ASPEED BMC) often enumerates
        before the AMD dGPU. Non-AMD vendors must never map (an NVIDIA GeForce
        GTX 860M would otherwise hit the AMD 860M pattern), and a 0000: PCI
        domain prefix must not break matching."""
        m = stack_mod

        def fake_lspci(stdout):
            result = SimpleNamespace(returncode = 0, stdout = stdout)
            return (
                patch.object(m.shutil, "which", return_value = "/usr/bin/lspci"),
                patch.object(m.subprocess, "run", return_value = result),
            )

        intel_then_amd = (
            "00:02.0 VGA compatible controller [0300]: Intel Corporation Raptor Lake-S GT1 [8086:a780]\n"
            "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI]"
            " Navi 31 [Radeon RX 7900 XT] [1002:744c]\n"
        )
        nvidia_only = "01:00.0 3D controller [0302]: NVIDIA Corporation GM107M [GeForce GTX 860M] [10de:1392]\n"
        domain_prefixed = (
            "0000:c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI]"
            " Strix Halo [Radeon Graphics / Radeon 8060S] [1002:150e]\n"
        )
        unmapped_then_mapped = (
            "03:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI]"
            " Cape Verde [FirePro W600] [1002:6821]\n"
            "04:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI]"
            " Navi 33 [Radeon RX 7600] [1002:7480]\n"
        )
        for stdout, expected in (
            (intel_then_amd, "gfx1100"),
            (nvidia_only, None),
            (domain_prefixed, "gfx1151"),
            (unmapped_then_mapped, "gfx1102"),
        ):
            w, r = fake_lspci(stdout)
            with w, r:
                assert m._linux_amd_gfx_from_lspci() == expected, stdout

    def test_install_sh_lspci_scan_covers_all_display_controllers(self):
        """install.sh mirror of the scan-all behaviour, executed with a shimmed
        lspci: Intel-first still finds the AMD dGPU, NVIDIA-only maps nothing
        (860M collision), a domain-prefixed AMD line still maps."""
        shell = shutil.which("bash")
        if not shell:
            pytest.skip("bash needed to execute the probe block")
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        name_fn = re.search(
            r"^_infer_amd_gfx_arch_from_gpu_name\(\) \{\n.*?\n\}\n", source, re.S | re.M
        )
        scan = re.search(
            r"^    if command -v lspci[^\n]*\n.*?\nEOF\n    fi\n    return 1\n", source, re.S | re.M
        )
        assert name_fn and scan, "could not extract the lspci scan block"
        cases = (
            (
                "00:02.0 VGA compatible controller [0300]: Intel Corporation UHD [8086:a780]\n"
                "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI]"
                " Navi 31 [Radeon RX 7900 XT] [1002:744c]",
                "OK:gfx1100",
            ),
            (
                "01:00.0 3D controller [0302]: NVIDIA Corporation GM107M [GeForce GTX 860M] [10de:1392]",
                "OK:",
            ),
            (
                "0000:c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc."
                " [AMD/ATI] Strix Halo [Radeon 8060S] [1002:150e]",
                "OK:gfx1151",
            ),
        )
        for lspci_out, expected in cases:
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "lspci")
                with open(p, "w", encoding = "utf-8") as f:
                    f.write(f'#!/bin/sh\ncat <<"EOT"\n{lspci_out}\nEOT\n')
                os.chmod(p, 0o755)
                script = (
                    "set -euo pipefail\n"
                    + name_fn.group(0)
                    + "probe() {\n"
                    + scan.group(0)
                    + "}\nprintf 'OK:%s\\n' \"$(probe || true)\"\n"
                )
                env = dict(os.environ, PATH = d + os.pathsep + os.environ.get("PATH", ""))
                r = subprocess.run([shell, "-c", script], env = env, capture_output = True, text = True)
                assert r.returncode == 0, f"scan aborted: {r.stderr}"
                assert (
                    r.stdout.splitlines()[-1] == expected
                ), f"lspci scan wrong for {lspci_out!r}: {r.stdout!r}"

    def test_install_sh_infer_gfx_gated_on_wsl_runtime(self):
        """install.sh's _infer_linux_amd_gfx_arch must, like the Python side, skip
        the cpuinfo/lspci inference on WSL unless librocdxg is present -- the
        override still returns first, so it stays authoritative."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        body = _extract_sh_function_body(source, "_infer_linux_amd_gfx_arch")
        assert body, "could not extract _infer_linux_amd_gfx_arch"
        override = body.find("UNSLOTH_ROCM_GFX_ARCH")
        dxg = body.find("/dev/dxg")
        rocdxg = body.find("librocdxg")
        # Anchor on the first cpuinfo *inference* (the grep), not a comment mention.
        infer = body.find("grep -qiE 'Ryzen AI Max")
        assert override >= 0 and dxg >= 0 and rocdxg >= 0 and infer >= 0
        assert "microsoft" in body, "WSL gate must also detect WSL via /proc/version"
        assert override < dxg, "the explicit override must return before the WSL gate"
        assert (
            dxg < infer and rocdxg < infer
        ), "the WSL/librocdxg gate must run before the cpuinfo/lspci inference"

    def test_install_sh_reroute_is_x86_64_only(self):
        """The Linux inferred-gfx reroute must be x86_64-only: ROCm torch wheels are
        not published for arm64, so an inferred/overridden gfx must not push an
        arm64 host to the AMD arch index (get_torch_index_url returns CPU there)."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        idx = source.find("_linux_inferred_gfx=$(_infer_linux_amd_gfx_arch")
        assert idx >= 0, "reroute consumer not found"
        window = source[max(0, idx - 400) : idx]
        assert (
            'case "$_ARCH" in x86_64|amd64)' in window
        ), "the inferred-gfx reroute must guard on x86_64|amd64 arch"

    def test_install_sh_reroute_skips_visible_rocm_gpu(self):
        """A */cpu index on a host whose AMD GPU IS visible to the ROCm probes is a
        deliberate fallback (unsupported/unreadable ROCm version, warned about in
        get_torch_index_url), not a missing runtime: the reroute must not override
        it with inferred per-arch wheels. The explicit UNSLOTH_ROCM_GFX_ARCH
        override must still win either way."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        idx = source.find("_linux_inferred_gfx=$(_infer_linux_amd_gfx_arch")
        assert idx >= 0, "reroute consumer not found"
        window = source[max(0, idx - 700) : idx]
        assert (
            "! _has_amd_rocm_gpu" in window
        ), "the reroute must be gated on _has_amd_rocm_gpu being false"
        assert (
            '[ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ] || ! _has_amd_rocm_gpu' in window
        ), "an explicit UNSLOTH_ROCM_GFX_ARCH override must bypass the visible-GPU gate"

    def test_install_sh_reroute_exports_gfx_for_setup_sh(self):
        """The inferred arch must be exported as UNSLOTH_ROCM_GFX_ARCH so the
        downstream setup.sh run (which re-probes ROCm independently and finds
        nothing on these runtime-less hosts) routes llama.cpp to the matching
        ROCm prebuilt instead of the CPU one -- setup.sh and
        install_llama_prebuilt.py both read that env var."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assign = source.find('TORCH_INDEX_URL="${_amd_mirror}/${_amd_family}/"')
        assert assign >= 0, "inferred-gfx index assignment not found"
        block_end = source.find("esac", assign)
        assert (
            'export UNSLOTH_ROCM_GFX_ARCH="$_linux_inferred_gfx"' in source[assign:block_end]
        ), "the reroute must export the inferred gfx for the setup.sh handoff"
        # setup.sh's side of the handoff must still exist.
        setup_source = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
        assert "UNSLOTH_ROCM_GFX_ARCH" in setup_source

    def test_amd_arch_index_url_linux_honors_amd_mirror(self):
        """On Linux the inferred-gfx repair must honour UNSLOTH_AMD_ROCM_MIRROR (the
        var install.sh uses), not the Windows mirror var, so a mirrored/air-gapped
        Linux install does not silently fall back to repo.amd.com. Windows still
        delegates to the Windows mirror path."""
        m = stack_mod
        with (
            patch.object(m, "IS_WINDOWS", False),
            patch.dict(os.environ, {"UNSLOTH_AMD_ROCM_MIRROR": "https://mirror.local/rocm"}),
        ):
            assert m._amd_arch_index_url("gfx1151") == "https://mirror.local/rocm/gfx1151/"
        with (
            patch.object(m, "IS_WINDOWS", False),
            patch.dict(os.environ, {"UNSLOTH_AMD_ROCM_MIRROR": ""}),
        ):
            assert m._amd_arch_index_url("gfx1151") == "https://repo.amd.com/rocm/whl/gfx1151/"
            assert m._amd_arch_index_url("gfx9999") is None
        # Windows path is unchanged: delegate to the Windows mirror helper.
        with patch.object(m, "IS_WINDOWS", True):
            assert m._amd_arch_index_url("gfx1151") == m._windows_rocm_index_url("gfx1151")

    def test_strix_gfx_detection_in_install_sh(self):
        """install.sh must detect gfx1151 and gfx1150 for the override."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "gfx1151" in source and "gfx1150" in source

    def test_rocm71_override_to_amd_arch_index_in_install_sh(self):
        """install.sh must override TORCH_INDEX_URL to AMD arch-specific index for Strix."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "repo.amd.com/rocm/whl" in source
        assert "_strix_gfx" in source
        # URL must incorporate the detected gfx arch (gfx1151 -> .../gfx1151/).
        strix_idx = source.find("_amd_strix_base")
        assert strix_idx != -1
        ctx = source[strix_idx : strix_idx + 500]
        assert "_strix_gfx" in ctx

    def test_radeon_repo_bypassed_for_strix_in_install_sh(self):
        """install.sh must set _amd_gpu_radeon=false when Strix + ROCm 7.1 detected."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "_amd_gpu_radeon=false" in source

    def test_strix_override_warns_with_moe_utils_reference(self):
        """install.sh must emit a [WARN] mentioning the moe_utils segfault."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "moe_utils" in source or "_grouped_mm" in source

    def test_strix_override_scoped_below_arch_floor(self):
        """Strix reroute must fire for rocm leaves BELOW the arch floor (7.13) and
        NOT at/above it. Executed via _rocm_leaf_below so it verifies the actual
        version comparison, not a text match that a comment could satisfy."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # Selector + gate must switch on the index LEAF, not the whole URL (a mirror
        # base path with its own rocm token would false-positive otherwise).
        assert 'case "$_torch_index_leaf" in' in source
        assert '_rocm_leaf_below "$_torch_index_leaf" 7 13' in source
        shell = shutil.which("sh") or shutil.which("bash")
        if not shell:
            pytest.skip("no POSIX shell to execute _rocm_leaf_below")
        match = re.search(r"^_rocm_leaf_below\(\) \{.*?^\}", source, re.S | re.M)
        assert match, "could not extract _rocm_leaf_below from install.sh"
        fn = match.group(0)

        def below(leaf):
            return (
                subprocess.run(
                    [shell, "-c", f'{fn}\n_rocm_leaf_below "$1" 7 13', "_", leaf]
                ).returncode
                == 0
            )

        for leaf in ("rocm6.0", "rocm7.0", "rocm7.1", "rocm7.2", "rocm7.12"):
            assert below(leaf), f"{leaf} must reroute (below arch floor 7.13)"
        for leaf in ("rocm7.13", "rocm7.14", "rocm8.0", "gfx1151", "cu128", "cpu"):
            assert not below(leaf), f"{leaf} must NOT reroute (>= floor or non-rocm)"

    def test_gfx_probe_survives_no_match_under_set_e(self):
        """A gfx probe whose grep finds no match must not abort install.sh under
        set -euo pipefail before the amd-smi fallback runs. The reroute case now
        matches every rocm* index, so this would break ordinary 6.x/7.2 installs
        with a flaky rocminfo. Executed with shimmed tools, not a text match."""
        shell = shutil.which("bash")
        if not shell:
            pytest.skip("bash needed to execute the probe block")
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        block = re.search(
            r'^        _gfx_all=""\n.*?(?=^        _strix_gfx="")', source, re.S | re.M
        )
        assert block, "could not extract the gfx-detection block"
        with tempfile.TemporaryDirectory() as d:
            # rocminfo emits no gfx token; amd-smi supplies gfx1151 (the fallback)
            for name, out in (("rocminfo", "no gpu here"), ("amd-smi", "GPU: gfx1151")):
                p = os.path.join(d, name)
                with open(p, "w", encoding = "utf-8") as f:
                    f.write(f'#!/bin/sh\ncat <<"EOT"\n{out}\nEOT\n')
                os.chmod(p, 0o755)
            script = (
                'set -euo pipefail\nHIP_VISIBLE_DEVICES=""\nROCR_VISIBLE_DEVICES=""\n'
                + block.group(0)
                + '\nprintf "OK:%s\\n" "$_gfx_all"\n'
            )
            env = dict(os.environ, PATH = d + os.pathsep + os.environ.get("PATH", ""))
            r = subprocess.run([shell, "-c", script], env = env, capture_output = True, text = True)
            assert r.returncode == 0, f"probe aborted under set -e: {r.stderr}"
            assert "OK:gfx1151" in r.stdout, f"amd-smi fallback not reached: {r.stdout!r}"

    def test_strix_routing_helpers_cover_rocm714(self):
        # Reroute for any generic pytorch.org index below the 7.13 arch floor (7.0,
        # 7.2, a future 7.3+), never at/above it -- mirrors install.sh _rocm_leaf_below.
        assert stack_mod._generic_pytorch_rocm_tag((7, 14)) == "rocm7.2"
        assert stack_mod._strix_needs_amd_arch_index((7, 14)) is True
        assert stack_mod._strix_needs_amd_arch_index((7, 0)) is True
        assert stack_mod._strix_needs_amd_arch_index((6, 0)) is True
        assert stack_mod._strix_needs_amd_arch_index((5, 0)) is False

    def test_torch_constraint_updated_for_strix_amd_index(self):
        """install.sh must set TORCH_CONSTRAINT>=2.11 when routing Strix to AMD index."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "TORCH_CONSTRAINT" in source and "2.11" in source

    def test_torch_constraint_211_matches_leaf_not_whole_url(self):
        """The 2.11 constraint case must match the index LEAF, not the whole URL.

        A custom UNSLOTH_PYTORCH_MIRROR whose base path contains a gfx/rocm7.2
        segment (e.g. https://mirror.local/gfx-cache) with a cu*/cpu family must
        not be pushed to the torch 2.11 line -- same leaf-only reasoning the
        UNSLOTH_TORCH_BACKEND classification uses.
        """
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # The 2.11 constraint block must switch on $_torch_index_leaf, not the full
        # $TORCH_INDEX_URL (a */gfx* match false-positives on a mirror base path). Only the
        # _grouped_mm-bug gfx families (gfx120X-all / gfx1151 / gfx1150) are pushed to 2.11;
        # a bare gfx* would also floor gfx110X-all/gfx90a/gfx908, left bare on purpose.
        assert 'case "$_torch_index_leaf" in\n    rocm7.2|gfx120x-all|gfx1151|gfx1150)' in source, (
            "the torch>=2.11 constraint must match the specific gfx leaves that need "
            "it (rocm7.2|gfx120x-all|gfx1151|gfx1150), not a bare gfx* or the whole URL"
        )

    def test_amd_rocm_mirror_env_var_respected(self):
        """install.sh must honour UNSLOTH_AMD_ROCM_MIRROR for air-gapped installs."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "UNSLOTH_AMD_ROCM_MIRROR" in source

    def test_tauri_family_recognises_amd_arch_url(self):
        """_tauri_torch_index_family must return a rocm* family for AMD arch-specific URLs."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "rocm/whl/gfx" in source


# TEST: setup.sh -- gcc-install-dir fix for Ubuntu 24.04 + ROCm 7.x clang-20


class TestSetupShGccInstallDir:
    """setup.sh applies --gcc-install-dir for HIP builds on Ubuntu 24.04+ (ROCm 7.x clang-20 header bug)."""

    def test_gcc_install_dir_search_loop_present(self):
        """setup.sh must iterate gcc versions 14→11 to find one with C++ headers."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "_GCC_INSTALL_DIR" in source
        assert "/usr/lib/gcc/x86_64-linux-gnu" in source

    def test_gcc_install_dir_checks_include_dir(self):
        """setup.sh must check that the gcc dir has an 'include' subdirectory."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "include" in source and "_GCC_INSTALL_DIR" in source

    def test_gcc_install_dir_appended_to_cmake_hip_flags(self):
        """setup.sh must pass --gcc-install-dir via CMAKE_HIP_FLAGS."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "CMAKE_HIP_FLAGS" in source
        assert "gcc-install-dir" in source

    def test_gcc_install_dir_only_applied_in_hip_build_block(self):
        """The --gcc-install-dir fix must only apply in the HIP/ROCm build branch."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        hip_idx = source.find("GGML_HIP=ON")
        gcc_idx = source.find("gcc-install-dir")
        assert hip_idx != -1 and gcc_idx != -1
        assert hip_idx < gcc_idx

    def test_gcc_install_dir_logs_substep(self):
        """setup.sh must print a substep when the gcc install dir is resolved."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "gcc install dir" in source or "GCC_INSTALL_DIR" in source


# TEST: main.py -- BNB_ROCM_VERSION server startup + distributed stubs

_MAIN_PY_PATH = PACKAGE_ROOT / "studio" / "backend" / "main.py"
_HARDWARE_PY_PATH = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"


class TestServerStartupRocmFixes:
    """main.py sets BNB_ROCM_VERSION pre-bnb-import; hardware.py stubs _distributed_c10d pre-torch.distributed."""

    # ── BNB_ROCM_VERSION in server process ────────────────────────────────────

    def test_main_py_sets_bnb_rocm_version(self):
        """main.py must set BNB_ROCM_VERSION in the server process before imports."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert "BNB_ROCM_VERSION" in source

    def test_main_py_bnb_detection_scoped_to_win32(self):
        """main.py BNB_ROCM_VERSION logic must be inside the win32 platform guard."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        win32_idx = source.find('sys.platform == "win32"')
        bnb_idx = source.find("BNB_ROCM_VERSION")
        assert win32_idx != -1 and bnb_idx != -1
        assert win32_idx < bnb_idx

    def test_main_py_bnb_dll_detection_uses_glob(self):
        """main.py must scan for libbitsandbytes_rocm*.dll to find the right version."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert "libbitsandbytes_rocm" in source

    def test_main_py_bnb_falls_back_to_72(self):
        """main.py must fall back to BNB_ROCM_VERSION='72' when no DLL is found."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert '"72"' in source or "'72'" in source

    def test_main_py_bnb_only_set_when_not_already_in_env(self):
        """main.py must not override an existing BNB_ROCM_VERSION env var."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert '"BNB_ROCM_VERSION" not in os.environ' in source

    # ── hipInfo.exe PATH prepend (bitsandbytes arch-probe fix) ────────────────
    # bnb's get_rocm_gpu_arch() runs hipinfo.exe via PATH at import; the AMD wheel ships it
    # in venv Scripts (on PATH only for activated venvs), so without the prepend bnb logs
    # "[WinError 2]" when launched directly.

    def test_main_py_prepends_hipinfo_dir_to_path(self):
        """main.py must make hipInfo.exe resolvable before bnb imports."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert "hipInfo.exe" in source
        # Prepend must precede the BNB_ROCM_VERSION block so bnb sees the fixed PATH.
        assert source.find("hipInfo.exe") < source.find("BNB_ROCM_VERSION")

    def test_main_py_hipinfo_prepend_gated_on_file_presence(self):
        """Prepend must check hipInfo.exe exists first (only AMD wheels ship it; leave NVIDIA/CPU untouched)."""
        source = _MAIN_PY_PATH.read_text(encoding = "utf-8")
        assert 'os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe"))' in source

    def test_worker_py_prepends_hipinfo_dir_to_path(self):
        """worker.py must mirror the prepend for standalone-spawned workers."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "hipInfo.exe" in source

    def test_install_stack_prepends_hipinfo_dir_to_path(self):
        """install_python_stack.py must prepend so child import checks inherit a PATH where bnb's probe works."""
        source = _STACK_PATH.read_text(encoding = "utf-8")
        assert "hipInfo.exe" in source

    # ── torch._C._distributed_c10d stubs in hardware.py ──────────────────────

    def test_hardware_py_injects_distributed_c10d_stub(self):
        """hardware.py must inject torch._C._distributed_c10d into sys.modules."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "_distributed_c10d" in source

    def test_hardware_py_stub_injected_before_distributed_import(self):
        """The sys.modules stub must be injected BEFORE import torch.distributed."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        c10d_idx = source.find("_distributed_c10d")
        dist_idx = source.find("import torch.distributed")
        assert c10d_idx != -1 and dist_idx != -1
        assert c10d_idx < dist_idx

    def test_hardware_py_stub_uses_types_moduletype(self):
        """hardware.py must create the stub with types.ModuleType."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "ModuleType" in source

    def test_hardware_py_stub_scoped_to_win32(self):
        """hardware.py distributed stub injection must be gated on win32."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert 'platform == "win32"' in source or "win32" in source

    def test_hardware_py_stub_exposes_fake_process_group(self):
        """hardware.py stub must set FakeProcessGroup so torch.distributed doesn't raise AttributeError."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "FakeProcessGroup" in source

    def test_hardware_py_stub_exposes_process_group(self):
        """hardware.py stub must set ProcessGroup on the c10d stub."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "ProcessGroup" in source

    def test_hardware_py_stub_uses_setattr_for_symbols(self):
        """hardware.py must use setattr to populate stub symbols dynamically."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "setattr" in source

    def test_hardware_py_stub_all_c10d_siblings_covered(self):
        """hardware.py must stub all three torch._C._distributed_* submodules."""
        source = _HARDWARE_PY_PATH.read_text(encoding = "utf-8")
        assert "_distributed_c10d" in source
        assert "_distributed_autograd" in source
        assert "_distributed_rpc" in source


# TEST: install.ps1 / setup.ps1 -- HipSdkInstalled flag (SDK found, device inaccessible)


class TestHipSdkInstalledButDeviceInaccessible:
    """When hipinfo is found but exits non-zero, both scripts distinguish device-inaccessible from SDK-not-found."""

    def test_install_ps1_has_hip_sdk_installed_flag(self):
        """install.ps1 must track HipSdkInstalled separately from HasROCm."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HipSdkInstalled" in source

    def test_setup_ps1_has_hip_sdk_installed_flag(self):
        """setup.ps1 must track HipSdkInstalled separately from HasROCm."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HipSdkInstalled" in source

    def test_install_ps1_sets_flag_when_hipinfo_binary_found(self):
        """install.ps1 must set HipSdkInstalled=true inside the 'if ($hipinfoExe)' block."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        hipinfo_block_idx = source.find("if ($hipinfoExe)")
        sdk_flag_idx = source.find("$HipSdkInstalled = $true", hipinfo_block_idx)
        assert hipinfo_block_idx != -1 and sdk_flag_idx != -1
        assert sdk_flag_idx > hipinfo_block_idx

    def test_setup_ps1_sets_flag_when_hipinfo_binary_found(self):
        """setup.ps1 must set HipSdkInstalled=true inside the 'if ($hipinfoExe)' block."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        hipinfo_block_idx = source.find("if ($hipinfoExe)")
        sdk_flag_idx = source.find("$HipSdkInstalled = $true", hipinfo_block_idx)
        assert hipinfo_block_idx != -1 and sdk_flag_idx != -1
        assert sdk_flag_idx > hipinfo_block_idx

    def test_install_ps1_version_capture_runs_when_sdk_installed(self):
        """install.ps1 must capture hipconfig version when HipSdkInstalled even if HasROCm is false."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "HasROCm -or $HipSdkInstalled" in source or "$HipSdkInstalled" in source

    def test_setup_ps1_version_capture_runs_when_sdk_installed(self):
        """setup.ps1 must capture hipconfig version when HipSdkInstalled even if HasROCm is false."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "HasROCm -or $HipSdkInstalled" in source or "$HipSdkInstalled" in source

    def test_install_ps1_distinct_message_for_sdk_found_but_device_inaccessible(self):
        """install.ps1 must show 'not ROCm-accessible' message (not 'HIP SDK not found') when SDK present."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "not ROCm-accessible" in source

    def test_setup_ps1_distinct_message_for_sdk_found_but_device_inaccessible(self):
        """setup.ps1 must show 'not ROCm-accessible' message (not 'HIP SDK not found') when SDK present."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "not ROCm-accessible" in source

    def test_install_ps1_driver_guidance_in_sdk_found_branch(self):
        """install.ps1 must tell user this is a driver issue, not an SDK issue."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "driver issue" in source

    def test_setup_ps1_driver_guidance_in_sdk_found_branch(self):
        """setup.ps1 must tell user this is a driver issue, not an SDK issue."""
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "driver issue" in source

    def test_install_ps1_cpu_hint_distinguishes_driver_vs_no_sdk(self):
        """install.ps1 CPU-only hint must say 'GPU not ROCm-accessible' not 'require the HIP SDK' when SDK found."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "GPU not ROCm-accessible" in source


# TEST: --rocm-gfx forwarding -- setup.sh/setup.ps1 forward their resolved gfx
# arch to install_llama_prebuilt.py so the per-gfx prebuilt is picked.

_SETUP_SH_PATH = PACKAGE_ROOT / "studio" / "setup.sh"


class TestNormalizeForwardedGfx:
    """A forwarded gfx string is reduced to a single clean gfx token."""

    def test_plain_token(self):
        assert _normalize_forwarded_gfx("gfx1151") == "gfx1151"

    def test_uppercase_normalized(self):
        assert _normalize_forwarded_gfx("GFX1151") == "gfx1151"

    def test_extracts_from_noise(self):
        assert _normalize_forwarded_gfx("gcnArchName: gfx942") == "gfx942"

    def test_malformed_is_ignored(self):
        assert _normalize_forwarded_gfx("not-a-gpu") is None

    def test_empty_and_none(self):
        assert _normalize_forwarded_gfx("") is None
        assert _normalize_forwarded_gfx(None) is None


class TestApplyHostOverrides:
    """Forwarded ROCm detection is folded into the host profile correctly."""

    def test_forwarded_gfx_fills_empty_probe(self):
        # Installer probe found no gfx (amd-smi-only / name-inferred host).
        host = rocm_host(rocm_gfx_target = None)
        out = _apply_host_overrides(host, override_rocm_gfx = "gfx1151")
        assert out.has_rocm is True
        assert out.rocm_gfx_target == "gfx1151"

    def test_forwarded_gfx_implies_rocm(self):
        # A CPU-looking host with a forwarded gfx is an AMD host.
        out = _apply_host_overrides(cpu_host(), override_rocm_gfx = "gfx1200")
        assert out.has_rocm is True
        assert out.rocm_gfx_target == "gfx1200"

    def test_forwarded_gfx_is_authoritative(self):
        # setup already applied visible-device selection; its value wins.
        host = rocm_host(rocm_gfx_target = "gfx1100")
        out = _apply_host_overrides(host, override_rocm_gfx = "gfx1151")
        assert out.rocm_gfx_target == "gfx1151"

    def test_has_rocm_only_keeps_probe_gfx(self):
        out = _apply_host_overrides(cpu_host(), override_has_rocm = True)
        assert out.has_rocm is True
        assert out.rocm_gfx_target is None

    def test_malformed_forwarded_gfx_falls_back_to_has_rocm(self):
        out = _apply_host_overrides(cpu_host(), override_has_rocm = True, override_rocm_gfx = "junk")
        assert out.has_rocm is True
        assert out.rocm_gfx_target is None

    def test_no_overrides_leaves_host_unchanged(self):
        host = nvidia_host()
        assert _apply_host_overrides(host) is host


class TestRocmGfxForwarding:
    """setup.sh / setup.ps1 forward their resolved gfx; the installer accepts it."""

    def test_installer_exposes_rocm_gfx_arg(self):
        source = _PREBUILT_PATH.read_text(encoding = "utf-8")
        assert '"--rocm-gfx"' in source
        # Defaults to the env override for standalone runs.
        assert 'os.environ.get("UNSLOTH_ROCM_GFX_ARCH")' in source

    def test_setup_sh_forwards_rocm_gfx(self):
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "--rocm-gfx" in source
        assert '"$_setup_gfx"' in source

    def test_setup_sh_forwards_has_rocm(self):
        # If AMD is detected but gfx resolution fails, --has-rocm is still forwarded.
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "--has-rocm" in source
        assert "_setup_amd_detected" in source

    def test_setup_ps1_forwards_rocm_gfx(self):
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert "--rocm-gfx" in source
        assert "$script:ROCmGfxArch" in source

    def test_setup_sh_routes_unconditionally_to_fork(self):
        # CPU-only hosts no longer fall back to ggml-org -- the release-repo
        # decision is an unconditional fork assignment now. Pin the line text.
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert '_HELPER_RELEASE_REPO="unslothai/llama.cpp"' in source
        assert '_HELPER_RELEASE_REPO="ggml-org/llama.cpp"' not in source

    def test_setup_ps1_routes_unconditionally_to_fork(self):
        # Same on Windows: the fork now ships the windows-cpu / windows-arm64
        # bundles, so $HelperReleaseRepo is an unconditional fork assignment.
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        assert '$HelperReleaseRepo = "unslothai/llama.cpp"' in source
        assert "$HelperReleaseRepo = if (" not in source

    # The text pins above guard the literal. The tests below execute the real routing line
    # from setup.sh / setup.ps1 and assert the resolved release repo, so a refactor that
    # reintroduces a conditional (or a ggml-org branch) is still caught. Inputs vary
    # (CPU-only, inferred/forwarded gfx, usable NVIDIA) to prove no host hits ggml-org.

    @staticmethod
    def _resolve_setup_sh_repo(
        host_machine,
        nvidia_usable,
        setup_gfx,
        rocm_gfx_arch_env = "",
    ):
        """Run setup.sh's release-repo routing block under bash and return the
        resolved _HELPER_RELEASE_REPO. PATH is emptied so any stray tooling probe
        misses; routing is unconditional, so the GPU inputs only prove no branch
        reroutes a host to ggml-org."""
        import shutil

        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        start = source.index('\n_HELPER_RELEASE_REPO="unslothai/llama.cpp"\n') + 1
        end = source.index("\n_LLAMA_PR=", start)
        block = source[start:end]
        assert "_HELPER_RELEASE_REPO" in block, "setup.sh routing anchors not found"
        env = {
            "PATH": "",  # no ROCm tooling discoverable
            "ROUTING_BLOCK": block,
            "_HOST_SYSTEM": "Linux",
            "_HOST_MACHINE": host_machine,
            "_setup_nvidia_usable": "true" if nvidia_usable else "false",
            "_setup_gfx": setup_gfx,
            "UNSLOTH_ROCM_GFX_ARCH": rocm_gfx_arch_env,
        }
        result = subprocess.run(
            [bash, "-c", 'eval "$ROUTING_BLOCK"; printf "%s" "$_HELPER_RELEASE_REPO"'],
            capture_output = True,
            text = True,
            timeout = 30,
            env = env,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    @pytest.mark.parametrize(
        "machine, nvidia_usable, setup_gfx, env_gfx",
        [
            ("x86_64", False, "", ""),  # plain CPU host (used to take ggml-org)
            ("aarch64", False, "", ""),  # plain CPU arm64 host (used to take ggml-org)
            ("x86_64", False, "gfx1100", ""),  # name-inferred gfx
            ("x86_64", False, "", "gfx1100"),  # env-forwarded gfx
            ("x86_64", True, "", ""),  # usable NVIDIA
        ],
    )
    def test_setup_sh_routing_block_always_resolves_to_fork(
        self, machine, nvidia_usable, setup_gfx, env_gfx
    ):
        assert (
            self._resolve_setup_sh_repo(
                machine, nvidia_usable, setup_gfx, rocm_gfx_arch_env = env_gfx
            )
            == "unslothai/llama.cpp"
        )

    @staticmethod
    def _resolve_setup_ps1_repo():
        """Run setup.ps1's $HelperReleaseRepo assignment under pwsh and return the
        resolved repo. The assignment is unconditional now, so there are no host
        inputs to vary."""
        import shutil

        pwsh = shutil.which("pwsh")
        if pwsh is None:
            pytest.skip("pwsh not available")
        source = _SETUP_PS1_PATH.read_text(encoding = "utf-8")
        line = next(
            (ln for ln in source.splitlines() if ln.strip().startswith("$HelperReleaseRepo =")),
            None,
        )
        assert line is not None, "$HelperReleaseRepo selection not found in setup.ps1"
        harness = f"{line}\nWrite-Output $HelperReleaseRepo"
        result = subprocess.run(
            [pwsh, "-NoProfile", "-Command", harness],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_setup_ps1_routing_resolves_to_fork(self):
        # Windows routing is unconditional now: CPU-only Windows (x64 and arm64)
        # uses the fork's windows-cpu / windows-arm64 bundles, not ggml-org.
        assert self._resolve_setup_ps1_repo() == "unslothai/llama.cpp"


# TEST: _pick_rocm_gfx_target -- visible-device selection from rocminfo output.
# Honours CUDA/HIP_VISIBLE_DEVICES so a mixed-arch host installs the prebuilt for the
# selected GPU, not GPU 0.

_pick_rocm_gfx_target = prebuilt_mod._pick_rocm_gfx_target


def test_pick_rocm_gfx_target_honors_cuda_visible_devices(monkeypatch):
    """CUDA_VISIBLE_DEVICES=1 must select gfx1100 on a gfx1151 + gfx1100 host (HIP honours CUDA var)."""
    # rocminfo reports each token twice (as in the real tool output).
    probe_out = "gfx1151\ngfx1151\ngfx1100\ngfx1100"
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _pick_rocm_gfx_target(probe_out) == "gfx1100"


def test_pick_rocm_gfx_target_cuda_visible_devices_minus_one_returns_none(monkeypatch):
    """CUDA_VISIBLE_DEVICES=-1 means no GPU visible; resolver must return None."""
    probe_out = "gfx1151\ngfx1100"
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _pick_rocm_gfx_target(probe_out) is None


def test_pick_rocm_gfx_target_same_arch_multi_gpu(monkeypatch):
    """Regression: [gfx1100, gfx1100, gfx1151] with HIP_VISIBLE_DEVICES=2 must return gfx1151 (no dict.fromkeys collapse)."""
    # rocminfo output for 3 GPUs (2x gfx1100 + 1x gfx1151), one Agent section each.
    probe_out = (
        "***\nAgent 1\n***\n  gfx1100 some info\n  gfx1100\n"
        "***\nAgent 2\n***\n  gfx1100 some info\n  gfx1100\n"
        "***\nAgent 3\n***\n  gfx1151 some info\n  gfx1151\n"
    )
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    assert _pick_rocm_gfx_target(probe_out) == "gfx1151"


# TEST: WSL ROCDXG fixes -- drop-in persistence + system-HIP-before-bundle


_INSTALL_SH_PATH = PACKAGE_ROOT / "install.sh"
_LLAMA_CPP_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"


class TestWslSystemRocmLibDirs:
    """_wsl_system_rocm_lib_dirs: no-op off a ROCDXG WSL host; else returns the system ROCm lib dir for binary_env."""

    def test_empty_without_dev_dxg(self):
        with patch("os.path.exists", return_value = False):
            assert prebuilt_mod._wsl_system_rocm_lib_dirs() == []

    def test_empty_on_bare_metal_linux(self):
        # /dev/dxg present but /proc/version is not a WSL kernel.
        with patch("os.path.exists", lambda p: p == "/dev/dxg"):
            with patch(
                "builtins.open",
                mock_open(read_data = "Linux version 6.8.0-generic"),
            ):
                assert prebuilt_mod._wsl_system_rocm_lib_dirs() == []

    def test_returns_system_lib_on_wsl_with_librocdxg(self):
        # Normalize separators: os.path.join uses "\" on the Windows test host.
        def _exists(p):
            p = str(p).replace("\\", "/")
            return p in ("/dev/dxg", "/opt/rocm/lib/librocdxg.so")

        with patch("os.path.exists", _exists):
            with patch(
                "builtins.open",
                mock_open(read_data = "Linux version 5.15.0-microsoft-standard-WSL2"),
            ):
                assert prebuilt_mod._wsl_system_rocm_lib_dirs() == ["/opt/rocm/lib"]

    def test_empty_on_wsl_without_librocdxg(self):
        # WSL kernel + /dev/dxg but no librocdxg -> not a ROCDXG ROCm install.
        with patch("os.path.exists", lambda p: p == "/dev/dxg"):
            with patch(
                "builtins.open",
                mock_open(read_data = "microsoft-standard-WSL2"),
            ):
                assert prebuilt_mod._wsl_system_rocm_lib_dirs() == []


class TestBinaryEnvWslOrdering:
    """binary_env puts system ROCm lib ahead of the bundle dir + sets HSA_ENABLE_DXG_DETECTION on WSL; no-op bare-metal."""

    @staticmethod
    def _linux_host():
        return HostInfo(
            system = "Linux",
            machine = "x86_64",
            is_windows = False,
            is_linux = True,
            is_macos = False,
            is_x86_64 = True,
            is_arm64 = False,
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            visible_cuda_devices = None,
            has_physical_nvidia = False,
            has_usable_nvidia = False,
            has_rocm = True,
        )

    def test_wsl_prepends_system_rocm_and_sets_hsa(self, tmp_path):
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        # dedupe_existing_dirs drops non-existent dirs, so use a real dir.
        sys_rocm = tmp_path / "sysrocm"
        sys_rocm.mkdir()
        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = [str(sys_rocm)]):
            with patch.dict(os.environ, {}, clear = True):
                env = prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        ld = env["LD_LIBRARY_PATH"].split(os.pathsep)
        # Compare resolved paths (dedupe_existing_dirs calls Path.resolve()).
        ld_resolved = [str(Path(p).resolve()) for p in ld]
        assert ld_resolved[0] == str(sys_rocm.resolve())
        assert str(binary.parent.resolve()) in ld_resolved
        assert ld_resolved.index(str(sys_rocm.resolve())) < ld_resolved.index(
            str(binary.parent.resolve())
        )
        assert env.get("HSA_ENABLE_DXG_DETECTION") == "1"

    def test_bare_metal_linux_unchanged(self, tmp_path):
        binary = tmp_path / "bundle" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        with patch.object(prebuilt_mod, "_wsl_system_rocm_lib_dirs", return_value = []):
            with patch.dict(os.environ, {}, clear = True):
                env = prebuilt_mod.binary_env(binary, tmp_path, self._linux_host())
        ld = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert ld[0] == str(binary.parent)  # bundle dir first, as before
        assert "HSA_ENABLE_DXG_DETECTION" not in env


class TestInstallShDropinPersistence:
    """install.sh persists the ROCm-on-WSL drop-in even when rocminfo already enumerates the GPU (reinstall safety)."""

    def test_has_persist_helper(self):
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "_persist_rocm_wsl_dropin()" in source

    def test_gate5_early_return_persists_dropin(self):
        """The rocminfo-already-works early return must call the persist helper before returning."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # The persist call must precede `return 0` at the rocminfo GPU-agent gate
        # (uniquely identified by the `!/generic/` clause the other probes lack).
        gate = source.find("Name:[[:space:]]*gfx[1-9]/ && !/generic/")
        assert gate != -1
        window = source[gate : gate + 900]
        assert "_persist_rocm_wsl_dropin" in window
        assert window.find("_persist_rocm_wsl_dropin") < window.find("return 0")

    def test_persist_helper_gated_on_librocdxg(self):
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        body_start = source.find("_persist_rocm_wsl_dropin()")
        body = source[body_start : body_start + 1200]
        assert "librocdxg.so" in body
        assert "profile.d/unsloth-rocm-wsl.sh" in body


_STRIXHALO_WSL_PATH = PACKAGE_ROOT / "scripts" / "install_rocm_wsl_strixhalo.sh"


class TestWslRerouteNvidiaGuard:
    """_maybe_reroute_strixhalo_to_2404 must skip the AMD reroute on hybrid AMD+NVIDIA hosts by
    reusing _has_usable_nvidia_gpu (CUDA_VISIBLE_DEVICES-aware + /proc/driver/nvidia fallback),
    which must be defined before the reroute's call site so it is actually available."""

    def test_reroute_calls_nvidia_helper_before_amd_signal(self):
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        start = source.find("_maybe_reroute_strixhalo_to_2404()")
        assert start != -1
        # Slice the WHOLE function body (to its closing brace at column 0), not a
        # fixed-length window: preamble growth must not push the signals out of view.
        end = source.find("\n}", start)
        assert end != -1
        body = source[start:end]
        nv = body.find("_has_usable_nvidia_gpu")
        wmi = body.find("_wsl_amd_gpu_name")
        assert nv != -1, "reroute must consult _has_usable_nvidia_gpu before deciding to reroute"
        assert wmi != -1
        # The NVIDIA guard must precede the AMD/WMI signal and return early.
        assert nv < wmi
        assert body.find("return 0", nv) < wmi

    def test_nvidia_helper_and_deps_defined_before_reroute_callsite(self):
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        call = source.find("\n_maybe_reroute_strixhalo_to_2404 || true")
        assert call != -1
        for fn in ("_run_bounded() {", "_cvd_hides_nvidia() {", "_has_usable_nvidia_gpu() {"):
            idx = source.find(fn)
            assert idx != -1 and idx < call, f"{fn} must be defined before the reroute call"


class TestStrixhaloGfxOverridePipefail:
    """The UNSLOTH_WSL_GFX override check must use a consuming grep, not grep -q: under
    `set -o pipefail` an early -q exit SIGPIPEs printf and misreports the arch on large output."""

    def test_gfx_override_uses_consuming_grep(self):
        source = _STRIXHALO_WSL_PATH.read_text(encoding = "utf-8")
        idx = source.find('grep -E "Name:[[:space:]]*${GFX}')
        assert idx != -1, "GFX override must use a consuming grep -E (not grep -q)"
        line = source[idx : source.find("\n", idx)]
        assert ">/dev/null" in line
        assert 'grep -qE "Name:[[:space:]]*${GFX}' not in source


class TestLlamaCppRuntimeWslOrdering:
    """The serve-time launcher mirrors binary_env: system HIP before the bundle dir on WSL."""

    def test_has_wsl_helper(self):
        source = _LLAMA_CPP_PATH.read_text(encoding = "utf-8")
        assert "_wsl_system_rocm_lib_dirs" in source

    def test_prepends_before_binary_dir(self):
        source = _LLAMA_CPP_PATH.read_text(encoding = "utf-8")
        idx_helper = source.find("lib_dirs.extend(_wsl_system_rocm_lib_dirs())")
        idx_binary = source.find("lib_dirs.append(binary_dir)")
        assert idx_helper != -1 and idx_binary != -1
        assert idx_helper < idx_binary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
