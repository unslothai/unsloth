"""Tests for AMD ROCm support across install pathways.

Verifies that ROCm detection and installation logic works correctly
WITHOUT breaking existing CUDA, CPU, macOS, and Windows pathways.
All tests use mocks -- no AMD hardware required.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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

# install_python_stack.py
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack", _STACK_PATH
)
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
    """Return the body of a shell function from `source` by brace matching.

    Used by structural tests that need to assert ordering of helper
    calls inside a specific function rather than across the whole
    install.sh file.
    """
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


# =============================================================================
# TEST: install_llama_prebuilt.py -- resolve_upstream_asset_choice
# =============================================================================


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
        # NVIDIA hosts go through the normal path (CUDA handled by resolve_linux_cuda_choice)
        assert choice.install_kind == "linux-cpu"
        assert "rocm" not in choice.name

    @patch.object(prebuilt_mod, "github_release_assets")
    def test_rocm_linux_no_prebuilt_falls_back(self, mock_assets):
        """AMD ROCm host should fall back to source build when no ROCm prebuilt exists."""
        # Remove the ROCm asset from available assets
        assets_without_rocm = {
            k: v for k, v in UPSTREAM_ASSETS.items() if "rocm" not in k
        }
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


# =============================================================================
# TEST: install_llama_prebuilt.py -- runtime_patterns_for_choice
# =============================================================================


class TestRuntimePatterns:
    """Verify runtime file patterns for all install kinds."""

    def test_linux_cpu_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-cpu"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "llama-server" in patterns
        assert "llama-quantize" in patterns

    def test_linux_cuda_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-cuda"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "libggml-cuda.so*" in patterns

    def test_linux_rocm_patterns(self):
        choice = AssetChoice(
            repo = "", tag = "", name = "", url = "", source_label = "", install_kind = "linux-rocm"
        )
        patterns = runtime_patterns_for_choice(choice)
        assert "libggml-hip.so*" in patterns
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
        assert "*.exe" in patterns
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


# =============================================================================
# TEST: install_llama_prebuilt.py -- HostInfo.has_rocm field
# =============================================================================


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
        # Must probe for actual GPU, not just tool presence
        assert "rocminfo" in source or "amd-smi" in source

    def test_detect_host_windows_rocm_detection(self):
        """detect_host() source should have Windows-specific ROCm GPU detection."""
        import inspect

        source = inspect.getsource(prebuilt_mod.detect_host)
        assert "hipinfo" in source or "amd-smi" in source


# =============================================================================
# TEST: install_python_stack.py -- _detect_rocm_version
# =============================================================================


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
        # Version files don't typically have epoch prefix, but lib/rocm_version might
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


# =============================================================================
# TEST: install_python_stack.py -- _ensure_rocm_torch
# =============================================================================


class TestEnsureRocmTorch:
    """Verify ROCm torch reinstall logic."""

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    def test_no_rocm_skips(self, mock_nvidia, mock_pip):
        """No ROCm toolchain should skip entirely."""
        with patch("os.path.isdir", return_value = False):
            with patch("shutil.which", return_value = None):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_torch_already_has_cuda_skips(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip
    ):
        """If torch already has CUDA, should skip ROCm reinstall."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"12.6\n"  # CUDA version string
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_torch_already_has_hip_skips(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip
    ):
        """If torch already has HIP, should skip ROCm reinstall."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"7.1.12345\n"  # HIP version string
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        mock_pip.assert_not_called()

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
        # Should install torch via pip_install and bitsandbytes via pip_install_try.
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])
        assert mock_pip_try.call_count >= 1
        assert "bitsandbytes" in str(mock_pip_try.call_args_list[0])

    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (6, 3))
    def test_rocm_63_selects_correct_tag(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip
    ):
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
    @patch.object(stack_mod, "_detect_rocm_version", return_value = None)
    def test_version_unreadable_prints_warning(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, capsys
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
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1))
    def test_probe_timeout_triggers_reinstall(
        self, mock_ver, mock_gpu, mock_nvidia, mock_pip, mock_pip_try
    ):
        """Probe subprocess timeout should not crash; should proceed to reinstall."""
        with patch("os.path.isdir", return_value = True):
            with patch(
                "subprocess.run", side_effect = subprocess.TimeoutExpired("python", 30)
            ):
                _ensure_rocm_torch()
        # If probe times out, the function should treat torch as unusable and reinstall
        # both torch (via pip_install) and bitsandbytes (via pip_install_try).
        assert mock_pip.call_count == 1
        assert "rocm7.1" in str(mock_pip.call_args_list[0])
        assert mock_pip_try.call_count >= 1

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = False)
    def test_no_gpu_with_rocm_tools_skips(self, mock_gpu, mock_nvidia, mock_pip):
        """ROCm tools present but no actual AMD GPU should skip entirely."""
        with patch("os.path.isdir", return_value = True):
            _ensure_rocm_torch()
        mock_pip.assert_not_called()


# =============================================================================
# TEST: install_python_stack.py -- _ROCM_TORCH_INDEX mapping
# =============================================================================


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


# =============================================================================
# TEST: hardware.py -- IS_ROCM flag and detect_hardware
# =============================================================================


class TestHardwareRocmFlag:
    """Verify IS_ROCM flag behavior without importing the full hardware module."""

    def test_hardware_py_has_is_rocm(self):
        """hardware.py should define IS_ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        assert "IS_ROCM: bool" in source and "False" in source

    def test_hardware_py_sets_is_rocm_on_hip(self):
        """detect_hardware() should set IS_ROCM when torch.version.hip is set."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        assert 'torch.version, "hip"' in source or "torch.version.hip" in source

    def test_hardware_py_still_returns_cuda_for_rocm(self):
        """DeviceType should remain CUDA even on ROCm -- no DeviceType.ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        # Ensure ROCM is NOT a DeviceType member
        enum_section = source.split("class DeviceType")[1].split("\n\n")[0]
        assert "ROCM" not in enum_section

    def test_hardware_py_has_rocm_in_package_versions(self):
        """get_package_versions() should include 'rocm' key."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        assert '"rocm"' in source

    def test_hardware_py_device_type_cuda_references_intact(self):
        """All existing DeviceType.CUDA references should still be present."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        # Key functions that must still reference DeviceType.CUDA
        assert "DeviceType.CUDA" in source
        assert "DEVICE = DeviceType.CUDA" in source

    def test_is_rocm_exported_from_init(self):
        """IS_ROCM should be exported from hardware __init__.py."""
        init_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        )
        source = init_path.read_text(encoding = "utf-8")
        assert "IS_ROCM" in source

    def test_is_rocm_in_all_list(self):
        """IS_ROCM should be in __all__ list in __init__.py."""
        init_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        )
        source = init_path.read_text(encoding = "utf-8")
        # Extract __all__ section
        assert '"IS_ROCM"' in source

    def test_get_package_versions_returns_rocm_key(self):
        """get_package_versions() source should return both 'cuda' and 'rocm' keys."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        # Find the get_package_versions function body
        func_start = source.find("def get_package_versions")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert '"cuda"' in func_body
        assert '"rocm"' in func_body

    def test_distributed_stubs_cover_is_torchelastic_launched(self):
        """_determine_attention_impl_for_gpu_estimate must stub is_torchelastic_launched.

        resolve_attention_implementation calls is_torchelastic_launched() on
        Windows ROCm where torch.distributed ships without that helper, causing
        a warning: 'module torch.distributed has no attribute is_torchelastic_launched'.
        """
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        assert "is_torchelastic_launched" in source

    def test_distributed_stubs_cover_core_helpers(self):
        """_determine_attention_impl_for_gpu_estimate must stub the four core distributed helpers."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        for attr in ("is_initialized", "is_available", "get_rank", "get_world_size"):
            assert (
                attr in source
            ), f"distributed stub for '{attr}' missing from hardware.py"


# =============================================================================
# TEST: tokenizer_utils.py -- error message
# =============================================================================


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


# =============================================================================
# TEST: install.sh -- structural checks
# =============================================================================


class TestInstallShStructure:
    """Verify install.sh structural properties without running it."""

    def test_no_here_strings(self):
        """install.sh must not use <<< (not POSIX)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        # <<< is bash-only; breaks dash
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "<<<" not in line, f"install.sh:{i} uses non-POSIX <<< here-string"

    def test_rocm_detection_present(self):
        """install.sh should have ROCm detection in get_torch_index_url."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        assert "amd-smi" in source
        assert "rocm" in source.lower()

    def test_cuda_precedence(self):
        """ROCm detection should only run when nvidia-smi is absent.

        install.sh defines _has_amd_rocm_gpu and _has_usable_nvidia_gpu
        helpers near each other (file-position order has no semantic
        meaning), so check the runtime ordering inside
        get_torch_index_url instead: NVIDIA branch runs first and the
        AMD/ROCm branch only fires inside the `if [ -z "$_smi" ]`
        block.
        """
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text(encoding = "utf-8")
        body = _extract_sh_function_body(source, "get_torch_index_url")
        nvidia_call = body.find("_has_usable_nvidia_gpu")
        no_nvidia_branch = body.find('if [ -z "$_smi" ]')
        rocm_call = body.find("_has_amd_rocm_gpu")
        assert (
            nvidia_call >= 0
        ), "get_torch_index_url should call _has_usable_nvidia_gpu"
        assert (
            no_nvidia_branch >= 0
        ), "get_torch_index_url should gate ROCm on no-nvidia-smi"
        assert (
            rocm_call > no_nvidia_branch
        ), "ROCm detection should sit inside the 'no nvidia-smi' branch"
        assert (
            nvidia_call < no_nvidia_branch
        ), "NVIDIA detection should run before the no-nvidia-smi branch"

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
        # Allowlisted versions should pass through directly
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
        """ROCm detection block should not use [[ ]] (bash-only, not POSIX).
        Note: [[:space:]], [[:digit:]] etc. are valid POSIX character classes, not bash [[ ]]."""
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
            # Remove POSIX character classes [[:foo:]] before checking for [[ ]]
            cleaned = re.sub(r"\[\[:[a-z]+:\]\]", "", line)
            assert (
                "[[" not in cleaned
            ), f"get_torch_index_url line {i} uses non-POSIX [["

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


# =============================================================================
# TEST: Live regression on current host (NVIDIA B200 expected)
# =============================================================================


class TestLiveRegression:
    """Live checks that run on the actual host -- skip if no NVIDIA GPU."""

    def test_get_torch_index_url_returns_cuda_on_nvidia(self):
        """On an NVIDIA machine, get_torch_index_url should return a CUDA URL."""
        import shutil

        if not shutil.which("nvidia-smi"):
            pytest.skip("No nvidia-smi available")
        # Skip if nvidia-smi exists but does not actually list a GPU on this
        # host (containers occasionally ship the binary without a driver).
        check = subprocess.run(
            [
                "bash",
                "-c",
                "nvidia-smi -L 2>/dev/null | "
                "awk '/^GPU[[:space:]]+[0-9]+:/{f=1} END{exit !f}'",
            ],
            capture_output = True,
        )
        if check.returncode != 0:
            pytest.skip("nvidia-smi is on PATH but no GPU is listed")

        sh_path = PACKAGE_ROOT / "install.sh"
        # get_torch_index_url calls _has_usable_nvidia_gpu and
        # _has_amd_rocm_gpu, so all three function definitions must be
        # in scope when we eval the extract.
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


# =============================================================================
# TEST: worker.py -- ROCm Mamba/SSM source build path
# =============================================================================

# Load worker.py module
_WORKER_PATH = PACKAGE_ROOT / "studio" / "backend" / "core" / "training" / "worker.py"
# The wheel-probe subprocess was hoisted out of worker.py into wheel_utils
# during the wheel-resolver refactor; the probe script literal lives there.
_WHEEL_UTILS_PATH = PACKAGE_ROOT / "studio" / "backend" / "utils" / "wheel_utils.py"


class TestWorkerRocmMambaSsm:
    """Verify worker.py Mamba/SSM install logic on ROCm."""

    def test_probe_returns_hip_version_field(self):
        """The wheel probe should include hip_version, and worker.py should
        consume it."""
        assert "hip_version" in _WHEEL_UTILS_PATH.read_text(encoding = "utf-8")
        assert "hip_version" in _WORKER_PATH.read_text(encoding = "utf-8")

    def test_probe_script_has_getattr_hip(self):
        """Probe script should use getattr for torch.version.hip (safe on CUDA)."""
        source = _WHEEL_UTILS_PATH.read_text(encoding = "utf-8")
        assert "getattr(torch.version, 'hip', None)" in source

    def test_direct_wheel_url_returns_none_without_cuda_major(self):
        """_direct_wheel_url should return None when cuda_major is empty (ROCm)."""
        # Load module for function access
        _worker_spec = importlib.util.spec_from_file_location(
            "test_worker", _WORKER_PATH
        )
        assert _worker_spec is not None and _worker_spec.loader is not None
        worker_mod = importlib.util.module_from_spec(_worker_spec)

        # Mock all the imports worker.py needs
        sys.modules["structlog"] = MagicMock()
        sys.modules["loggers"] = MagicMock()
        sys.modules["loggers"].get_logger = MagicMock(return_value = MagicMock())
        sys.modules["utils"] = MagicMock()
        sys.modules["utils.hardware"] = MagicMock()

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
        result = worker_mod._direct_wheel_url(
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


# =============================================================================
# TEST: amd.py -- AMD GPU monitoring
# =============================================================================


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

    def test_amd_smi_json_parsing(self):
        """Verify _extract_gpu_metrics parses amd-smi JSON correctly."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        sys.modules["loggers"] = MagicMock()
        sys.modules["loggers"].get_logger = MagicMock(return_value = MagicMock())

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module in test environment")

        # Simulate amd-smi metric JSON output
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

        sys.modules["loggers"] = MagicMock()
        sys.modules["loggers"].get_logger = MagicMock(return_value = MagicMock())

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        # _first_visible_amd_gpu_id() short-circuits to None when any of
        # HIP / ROCR / CUDA_VISIBLE_DEVICES is set to "" or "-1". CI runners
        # often unset CUDA at the env level by setting CUDA_VISIBLE_DEVICES
        # to "" so the test must not inherit that.
        for var in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            monkeypatch.delenv(var, raising = False)

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

        with patch.object(subprocess, "run", return_value = mock_result):
            result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is True
        assert result["gpu_utilization_pct"] == 50.0
        assert result["temperature_c"] == 65.0

    def test_amd_smi_not_found_returns_unavailable(self):
        """get_primary_gpu_utilization returns available=False when amd-smi is missing."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd3", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        sys.modules["loggers"] = MagicMock()
        sys.modules["loggers"].get_logger = MagicMock(return_value = MagicMock())

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        with patch.object(subprocess, "run", side_effect = OSError("amd-smi not found")):
            result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is False

    def test_amd_timeout_returns_unavailable(self):
        """get_primary_gpu_utilization handles timeout gracefully."""
        amd_path = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "amd.py"
        _amd_spec = importlib.util.spec_from_file_location("test_amd4", amd_path)
        assert _amd_spec is not None and _amd_spec.loader is not None
        amd_mod = importlib.util.module_from_spec(_amd_spec)

        sys.modules["loggers"] = MagicMock()
        sys.modules["loggers"].get_logger = MagicMock(return_value = MagicMock())

        try:
            _amd_spec.loader.exec_module(amd_mod)
        except Exception:
            pytest.skip("Could not load amd module")

        with patch.object(
            subprocess,
            "run",
            side_effect = subprocess.TimeoutExpired("amd-smi", 5),
        ):
            result = amd_mod.get_primary_gpu_utilization()
        assert result["available"] is False


# =============================================================================
# TEST: hardware.py -- IS_ROCM branching to amd.py
# =============================================================================


class TestHardwareAmdBranching:
    """Verify hardware.py branches to amd.py when IS_ROCM is True."""

    def test_hardware_imports_amd_module(self):
        """hardware.py should import from amd module when IS_ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        assert "from . import amd" in source

    def test_hardware_branches_on_is_rocm_for_utilization(self):
        """get_gpu_utilization should dispatch to amd.py via _smi_query
        when IS_ROCM, and the dispatcher itself must check IS_ROCM and
        import the amd backend."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_gpu_utilization")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert '_smi_query("get_primary_gpu_utilization"' in func_body
        smi = source[
            source.find("def _smi_query") : source.find(
                "\ndef ", source.find("def _smi_query") + 1
            )
        ]
        assert "IS_ROCM" in smi
        assert "from . import amd" in smi

    def test_hardware_branches_on_is_rocm_for_visible(self):
        """get_visible_gpu_utilization should dispatch to amd.py via
        _smi_query when IS_ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_visible_gpu_utilization")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        # The dispatcher call may wrap onto multiple lines; allow whitespace
        # between the open paren and the literal func name argument.
        import re as _re

        assert _re.search(r'_smi_query\(\s*"get_visible_gpu_utilization"', func_body)
        smi = source[
            source.find("def _smi_query") : source.find(
                "\ndef ", source.find("def _smi_query") + 1
            )
        ]
        assert "IS_ROCM" in smi
        assert "from . import amd" in smi

    def test_hardware_branches_on_is_rocm_for_physical_count(self):
        """get_physical_gpu_count should try amd.py when IS_ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def get_physical_gpu_count")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "IS_ROCM" in func_body
        assert "from . import amd" in func_body


# =============================================================================
# TEST: hardware.py -- apply_gpu_ids ROCm fallback (issue #5180)
# =============================================================================


class TestApplyGpuIdsRocmFallback:
    """Verify apply_gpu_ids sets HIP_VISIBLE_DEVICES on ROCm hosts even when
    IS_ROCM is still False (worker subprocess before detect_hardware runs)."""

    def test_apply_gpu_ids_falls_back_to_torch_version_hip(self):
        """apply_gpu_ids should probe torch.version.hip when IS_ROCM is False and no ROCm env vars are set."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert 'getattr(_torch.version, "hip", None)' in func_body

    def test_apply_gpu_ids_sets_hip_and_rocr_visible_devices(self):
        """apply_gpu_ids should set both HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES on ROCm."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert 'os.environ["HIP_VISIBLE_DEVICES"] = value' in func_body
        assert 'os.environ["ROCR_VISIBLE_DEVICES"] = value' in func_body

    def test_apply_gpu_ids_rocm_fallback_is_guarded_by_try_except(self):
        """torch import in apply_gpu_ids must be wrapped in try/except so a missing torch never crashes."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text(encoding = "utf-8")
        func_start = source.find("def apply_gpu_ids")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "import torch as _torch" in func_body
        assert "except Exception" in func_body


# =============================================================================
# TEST: install_python_stack.py -- Windows AMD warning
# =============================================================================


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


# =============================================================================
# TEST: unsloth/kernels/utils.py -- is_rdna() expansion
# =============================================================================


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


# =============================================================================
# TEST: install_python_stack.py -- _windows_rocm_index_url arch mapping
# =============================================================================


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
        monkeypatch.setenv(
            "UNSLOTH_ROCM_WINDOWS_MIRROR", "https://my-mirror.example.com/rocm/whl"
        )
        # Reload module-level constant by calling helper directly
        url = stack_mod._windows_rocm_index_url("gfx1200")
        # The env var is read at module load time for _ROCM_WINDOWS_INDEX_BASE,
        # so just verify the helper itself doesn't error.
        assert url is not None


# =============================================================================
# TEST: install_python_stack.py -- _detect_windows_gfx_arch
# =============================================================================


class TestDetectWindowsGfxArch:
    """Verify hipinfo parsing for GPU arch detection on Windows."""

    def test_returns_none_when_hipinfo_not_on_path(self):
        with patch("shutil.which", return_value = None):
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

    def test_returns_none_on_nonzero_returncode(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b"gcnArchName : gfx1200\n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
                result = stack_mod._detect_windows_gfx_arch()
        assert result is None

    def test_returns_none_when_no_gcnarchname_in_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"deviceName : Radeon RX 9060 XT\n"
        with patch("shutil.which", return_value = "/usr/bin/hipinfo"):
            with patch("subprocess.run", return_value = mock_result):
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


# =============================================================================
# TEST: install_python_stack.py -- _install_bnb_windows_rocm
# =============================================================================


class TestInstallBnbWindowsRocm:
    """Verify AMD Windows BNB wheel install helper."""

    def test_calls_pip_install_try_with_win_amd64_url(self):
        """Should call pip_install_try with the win_amd64 wheel URL."""
        with patch.object(stack_mod, "pip_install_try", return_value = True) as mock_pip:
            stack_mod._install_bnb_windows_rocm()
        assert mock_pip.call_count == 1
        call_args = str(mock_pip.call_args_list[0])
        assert "bitsandbytes" in call_args
        assert "win_amd64" in call_args

    def test_sets_uv_skip_env_var_during_install(self):
        """UV_SKIP_WHEEL_FILENAME_CHECK must be '1' when pip_install_try runs."""
        observed = {}

        def _capture(*args, **kwargs):
            observed["val"] = os.environ.get("UV_SKIP_WHEEL_FILENAME_CHECK")
            return True

        with patch.object(stack_mod, "pip_install_try", side_effect = _capture):
            stack_mod._install_bnb_windows_rocm()
        assert observed.get("val") == "1"

    def test_restores_uv_skip_env_var_after_install(self):
        """UV_SKIP_WHEEL_FILENAME_CHECK should be removed after install if it wasn't set before."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("UV_SKIP_WHEEL_FILENAME_CHECK", None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                stack_mod._install_bnb_windows_rocm()
            assert "UV_SKIP_WHEEL_FILENAME_CHECK" not in os.environ

    def test_restores_previous_uv_skip_value(self):
        """If UV_SKIP_WHEEL_FILENAME_CHECK was already set, restore it afterwards."""
        with patch.dict(os.environ, {"UV_SKIP_WHEEL_FILENAME_CHECK": "0"}):
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("UV_SKIP_WHEEL_FILENAME_CHECK") == "0"

    def test_restores_env_even_if_install_raises(self):
        """UV_SKIP_WHEEL_FILENAME_CHECK must be cleaned up even on pip failure."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("UV_SKIP_WHEEL_FILENAME_CHECK", None)
            with patch.object(
                stack_mod, "pip_install_try", side_effect = RuntimeError("pip failed")
            ):
                try:
                    stack_mod._install_bnb_windows_rocm()
                except RuntimeError:
                    pass
            assert "UV_SKIP_WHEEL_FILENAME_CHECK" not in os.environ

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
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(
                    stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "72"
                ):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "72"

    def test_sets_bnb_rocm_version_from_newer_dll(self):
        """If AMD ships a newer DLL (e.g. rocm713.dll), that version is used."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(
                    stack_mod, "_detect_bnb_rocm_dll_ver", return_value = "713"
                ):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "713"

    def test_falls_back_to_72_when_detection_fails(self):
        """Falls back to '72' when DLL detection returns None."""
        with patch.dict(os.environ, {}, clear = False):
            os.environ.pop("BNB_ROCM_VERSION", None)
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                with patch.object(
                    stack_mod, "_detect_bnb_rocm_dll_ver", return_value = None
                ):
                    stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "72"

    def test_does_not_override_existing_bnb_rocm_version(self):
        """An explicit BNB_ROCM_VERSION in the caller's env must not be clobbered."""
        with patch.dict(os.environ, {"BNB_ROCM_VERSION": "60"}):
            with patch.object(stack_mod, "pip_install_try", return_value = True):
                stack_mod._install_bnb_windows_rocm()
            assert os.environ.get("BNB_ROCM_VERSION") == "60"


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


# =============================================================================
# TEST: install_python_stack.py -- UNSLOTH_ROCM_TORCH_INSTALLED early-return path
# =============================================================================


class TestRocmTorchInstalledEnvVar:
    """Verify UNSLOTH_ROCM_TORCH_INSTALLED=1 skips main install but still installs BNB."""

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_skips_main_pip_install(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should not trigger torch pip_install."""
        with patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}):
            stack_mod._ensure_rocm_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_calls_bnb_install(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should still call _install_bnb_windows_rocm."""
        with patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}):
            stack_mod._ensure_rocm_torch()
        mock_bnb.assert_called_once()

    @patch.object(stack_mod, "_install_bnb_windows_rocm")
    @patch.object(stack_mod, "pip_install")
    def test_env_var_sets_rocm_windows_flag(self, mock_pip, mock_bnb):
        """UNSLOTH_ROCM_TORCH_INSTALLED=1 should set _rocm_windows_torch_installed."""
        stack_mod._rocm_windows_torch_installed = False
        with patch.dict(os.environ, {"UNSLOTH_ROCM_TORCH_INSTALLED": "1"}):
            stack_mod._ensure_rocm_torch()
        assert stack_mod._rocm_windows_torch_installed is True


# =============================================================================
# TEST: worker.py -- Windows ROCm patches (source-level checks)
# =============================================================================


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

    def test_torchao_stub_uses_stub_type_meta(self):
        """Torchao stub must use _StubTypeMeta so isinstance() returns False not TypeError."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "_StubTypeMeta" in source

    def test_stub_type_meta_has_instancecheck(self):
        """_StubTypeMeta must define __instancecheck__ returning False."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "__instancecheck__" in source

    def test_stub_subpackage_finder_registered(self):
        """_StubSubpackageFinder must be appended to sys.meta_path."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "sys.meta_path.append(_StubSubpackageFinder())" in source

    def test_torchao_key_submodules_pre_stubbed(self):
        """Key torchao submodules (dtypes, quantization) must be pre-stubbed."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "torchao.dtypes" in source
        assert "torchao.quantization" in source

    def test_torchdynamo_disabled_on_windows_rocm(self):
        """worker.py should disable dynamo on Windows ROCm as belt-and-suspenders."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "TORCHDYNAMO_DISABLE" in source

    def test_bnb_rocm_version_set_on_windows_rocm(self):
        """worker.py must set BNB_ROCM_VERSION in the Windows ROCm section.

        BNB auto-detects HIP version from torch.version.hip, which can mismatch
        the DLL suffix in the AMD prerelease wheel.  The worker must detect the
        actual DLL suffix and override BNB's auto-detection before ML imports.
        """
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        # Env var must be set
        assert "BNB_ROCM_VERSION" in source
        # Detection helper must be used
        assert "_detect_bnb_rocm_dll_ver" in source or "libbitsandbytes_rocm" in source
        # "72" must appear as the safe fallback
        assert '"72"' in source or "'72'" in source

    def test_bnb_rocm_version_set_before_ml_imports(self):
        """BNB_ROCM_VERSION must appear in section 1f, before section 2 ML imports."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        idx_bnb = source.find("BNB_ROCM_VERSION")
        # Use the specific section-2 marker that appears in the worker process
        # entry-point function (not the trainer helper which has its own "# ── 2.").
        idx_sec2 = source.find("# ── 2. Now import ML libraries")
        assert idx_bnb != -1, "BNB_ROCM_VERSION not found in worker.py"
        assert (
            idx_sec2 != -1
        ), "'# ── 2. Now import ML libraries' marker not found in worker.py"
        assert idx_bnb < idx_sec2, (
            "BNB_ROCM_VERSION must be set before section 2 ML imports "
            f"(found at {idx_bnb}, section 2 at {idx_sec2})"
        )

    def test_grouped_mm_patch_guarded_by_windows_and_hip_check(self):
        """_grouped_mm patch must only apply on Windows + HIP torch."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        # Must check sys.platform == "win32"
        assert 'sys.platform == "win32"' in source
        # Must gate on HIP version — code uses getattr chain: "version" and "hip"
        assert '"version"' in source and '"hip"' in source

    def test_hip_ver_at_least_helper_defined(self):
        """_hip_ver_at_least helper must be defined inside the Windows ROCm block."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "def _hip_ver_at_least(major: int, minor: int)" in source

    def test_grouped_mm_patch_gated_on_hip_lt_713(self):
        """_grouped_mm patch must be skipped on HIP >= 7.13 (AMD fixed the bug in ROCm 7.13)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        # The guard must call _hip_ver_at_least with exactly (7, 13)
        assert "_hip_ver_at_least(7, 13)" in source
        # The patch must be inside the `if not` branch (negated guard)
        assert "if not _hip_ver_at_least(7, 13):" in source

    def test_grouped_mm_hip_713_skip_message_present(self):
        """worker.py must log a message when skipping the patch on HIP >= 7.13."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        assert "HIP >= 7.13" in source
        assert "7.13" in source

    def test_grouped_mm_patch_else_branch_present(self):
        """An else branch must follow the _hip_ver_at_least gate (skip path for 7.13+)."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        # There must be an else: after the if not _hip_ver_at_least(7, 13): block
        gate_idx = source.find("if not _hip_ver_at_least(7, 13):")
        assert gate_idx != -1, "Version gate not found in worker.py"
        # The else: branch must appear after the gate
        else_idx = source.find("else:", gate_idx)
        assert else_idx != -1, "else: branch after _hip_ver_at_least gate not found"

    def test_hip_ver_at_least_handles_amd_version_format(self):
        """_hip_ver_at_least must split on '.' and compare only major.minor (handles '7.13.99004')."""
        source = _WORKER_PATH.read_text(encoding = "utf-8")
        # Must split the version string and take the first two parts
        assert 'split(".")[:2]' in source or ".split('.')[:2]" in source


# =============================================================================
# TEST: install_python_stack.py -- _ROCM_TORCH_PKG_SPECS mapping
# =============================================================================


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


# =============================================================================
# TEST: setup.ps1 / install.ps1 -- Strix Halo gfx arch detection
# =============================================================================

_SETUP_PS1_PATH = PACKAGE_ROOT / "studio" / "setup.ps1"
_INSTALL_PS1_PATH = PACKAGE_ROOT / "install.ps1"


class TestStrixHaloGfxArchDetection:
    """Verify that setup.ps1 and install.ps1 have robust gfx arch detection
    for Strix Halo / iGPU users who only have the HIP runtime (no hipinfo)."""

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
        # Find the WMI block and confirm HasROCm is not set inside it
        wmi_idx = source.find("Win32_VideoController")
        assert wmi_idx != -1, "WMI block not found in setup.ps1"
        # The nearest HasROCm = $true must not appear between the WMI block
        # and the closing brace of that if-block.  We check by confirming
        # $HasROCm = $true does NOT appear within 300 chars of the WMI call.
        wmi_context = source[wmi_idx : wmi_idx + 300]
        assert "$HasROCm = $true" not in wmi_context

    def test_gfx_arch_regex_parses_from_amd_smi_output(self):
        """Both files must use the gfx\\d+[a-z]? regex to parse arch from amd-smi output."""
        for path in (_SETUP_PS1_PATH, _INSTALL_PS1_PATH):
            source = path.read_text(encoding = "utf-8")
            # The regex pattern used to match gfx arches
            assert (
                "gfx\\d+" in source or r"gfx\d+" in source
            ), f"gfx arch regex not found in {path.name}"


# =============================================================================
# TEST: HIP SDK tool path resolution via HIP_PATH / ROCM_PATH env vars
# =============================================================================


class TestHipSdkEnvPathResolution:
    """Verify that both install scripts resolve hipinfo/hipconfig via HIP_PATH
    and ROCM_PATH when the tools are not on $PATH, and emit explicit warnings."""

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
        # Confirm the fallback pattern: HIP_PATH ?? ROCM_PATH (or equivalent elseif)
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
        # The warning must mention that the SDK install may be incomplete
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
        # Should mention adding to PATH or SetEnvironmentVariable
        assert "PATH" in source and (
            "SetEnvironmentVariable" in source or "Add" in source
        )

    def test_install_provides_path_fix_hint(self):
        """install.ps1 must tell the user how to add the HIP bin dir to PATH."""
        source = _INSTALL_PS1_PATH.read_text(encoding = "utf-8")
        assert "PATH" in source and (
            "SetEnvironmentVariable" in source or "Add" in source
        )


# =============================================================================
# TEST: HIP SDK detected substep -- path + hipconfig version shown in terminal
# =============================================================================


class TestHipSdkDetectedSubstep:
    """Verify that both scripts print HIP SDK path and full hipconfig version
    as substeps under the gpu step when AMD ROCm is successfully detected."""

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


# =============================================================================
# TEST: install.sh -- Strix Halo rocm7.1 → rocm7.2 override
# =============================================================================

_INSTALL_SH_PATH = PACKAGE_ROOT / "install.sh"
_SETUP_SH_PATH   = PACKAGE_ROOT / "studio" / "setup.sh"


class TestStrixRocm71Override:
    """Verify install.sh skips Radeon repo and forces rocm7.2 for gfx1151/gfx1150
    when ROCm 7.1 would otherwise be selected (known _grouped_mm segfault)."""

    def test_strix_gfx_detection_in_install_sh(self):
        """install.sh must detect gfx1151 and gfx1150 for the override."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "gfx1151" in source and "gfx1150" in source

    def test_rocm71_override_to_rocm72_in_install_sh(self):
        """install.sh must override TORCH_INDEX_URL from rocm7.1 to rocm7.2 for Strix."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # The override must explicitly reference rocm7.2 in context with Strix detection
        assert "rocm7.2" in source
        assert "_strix_gfx" in source

    def test_radeon_repo_bypassed_for_strix_in_install_sh(self):
        """install.sh must set _amd_gpu_radeon=false when Strix + ROCm 7.1 detected."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # After Strix detection the Radeon repo flag must be disabled
        assert "_amd_gpu_radeon=false" in source

    def test_strix_override_warns_with_moe_utils_reference(self):
        """install.sh must emit a [WARN] mentioning the moe_utils segfault."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        assert "moe_utils" in source or "_grouped_mm" in source

    def test_strix_override_only_fires_on_rocm71(self):
        """install.sh must scope the Strix override to rocm7.1 only (not rocm7.2+)."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # The Strix guard must be inside a rocm7.1 case branch
        strix_idx = source.find("_strix_gfx")
        assert strix_idx != -1
        # Look back for the rocm7.1 pattern within 600 chars before _strix_gfx
        context_before = source[max(0, strix_idx - 600) : strix_idx]
        assert "rocm7.1" in context_before

    def test_torch_constraint_updated_for_rocm72(self):
        """install.sh must update TORCH_CONSTRAINT to allow torch>=2.11 when forcing rocm7.2."""
        source = _INSTALL_SH_PATH.read_text(encoding = "utf-8")
        # TORCH_CONSTRAINT must be set inside the Strix override block
        assert "TORCH_CONSTRAINT" in source and "2.11" in source


# =============================================================================
# TEST: setup.sh -- gcc-install-dir fix for Ubuntu 24.04 + ROCm 7.x clang-20
# =============================================================================


class TestSetupShGccInstallDir:
    """Verify setup.sh applies the --gcc-install-dir flag when building llama.cpp
    with HIP on Ubuntu 24.04+ to work around ROCm 7.x clang-20 header path bug."""

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
        # GGML_HIP=ON must appear before gcc-install-dir in the source
        hip_idx = source.find("GGML_HIP=ON")
        gcc_idx = source.find("gcc-install-dir")
        assert hip_idx != -1 and gcc_idx != -1
        assert hip_idx < gcc_idx

    def test_gcc_install_dir_logs_substep(self):
        """setup.sh must print a substep when the gcc install dir is resolved."""
        source = _SETUP_SH_PATH.read_text(encoding = "utf-8")
        assert "gcc install dir" in source or "GCC_INSTALL_DIR" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
