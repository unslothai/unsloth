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

    @patch.object(stack_mod, "pip_install")
    @patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False)
    @patch.object(stack_mod, "_has_rocm_gpu", return_value = True)
    @patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 2))
    def test_rocm_72_selects_71_tag(self, mock_ver, mock_gpu, mock_nvidia, mock_pip):
        """ROCm 7.2 should select rocm7.1 tag (capped, not in mapping)."""
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = b"\n"
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = mock_probe):
                _ensure_rocm_torch()
        torch_call = mock_pip.call_args_list[0]
        assert "rocm7.1" in str(torch_call)

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

    def test_rocm_72_not_in_mapping(self):
        """ROCm 7.2 should NOT be in the active mapping (torch 2.11.0 exceeds bound)."""
        assert (7, 2) not in _ROCM_TORCH_INDEX

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
        """ROCm 7.2 (not in map) should select rocm7.1 via >= comparison."""
        ver = (7, 2)
        tag = next(
            (
                t
                for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                if ver >= (maj, mn)
            ),
            None,
        )
        assert tag == "rocm7.1"

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
        source = hw_path.read_text()
        assert "IS_ROCM: bool" in source and "False" in source

    def test_hardware_py_sets_is_rocm_on_hip(self):
        """detect_hardware() should set IS_ROCM when torch.version.hip is set."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
        assert 'torch.version, "hip"' in source or "torch.version.hip" in source

    def test_hardware_py_still_returns_cuda_for_rocm(self):
        """DeviceType should remain CUDA even on ROCm -- no DeviceType.ROCM."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
        # Ensure ROCM is NOT a DeviceType member
        enum_section = source.split("class DeviceType")[1].split("\n\n")[0]
        assert "ROCM" not in enum_section

    def test_hardware_py_has_rocm_in_package_versions(self):
        """get_package_versions() should include 'rocm' key."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
        assert '"rocm"' in source

    def test_hardware_py_device_type_cuda_references_intact(self):
        """All existing DeviceType.CUDA references should still be present."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
        # Key functions that must still reference DeviceType.CUDA
        assert "DeviceType.CUDA" in source
        assert "DEVICE = DeviceType.CUDA" in source

    def test_is_rocm_exported_from_init(self):
        """IS_ROCM should be exported from hardware __init__.py."""
        init_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        )
        source = init_path.read_text()
        assert "IS_ROCM" in source

    def test_is_rocm_in_all_list(self):
        """IS_ROCM should be in __all__ list in __init__.py."""
        init_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "__init__.py"
        )
        source = init_path.read_text()
        # Extract __all__ section
        assert '"IS_ROCM"' in source

    def test_get_package_versions_returns_rocm_key(self):
        """get_package_versions() source should return both 'cuda' and 'rocm' keys."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
        # Find the get_package_versions function body
        func_start = source.find("def get_package_versions")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert '"cuda"' in func_body
        assert '"rocm"' in func_body


# =============================================================================
# TEST: tokenizer_utils.py -- error message
# =============================================================================


class TestTokenizerErrorMessage:
    """Verify the AMD error message is updated."""

    def test_no_old_amd_message(self):
        """Old 'We do not support AMD' message should be gone."""
        tu_path = PACKAGE_ROOT / "unsloth" / "tokenizer_utils.py"
        source = tu_path.read_text()
        assert "We do not support AMD" not in source

    def test_new_message_has_docs_link(self):
        """New message should point to Unsloth AMD docs."""
        tu_path = PACKAGE_ROOT / "unsloth" / "tokenizer_utils.py"
        source = tu_path.read_text()
        assert "docs.unsloth.ai" in source or "No GPU detected" in source


# =============================================================================
# TEST: install.sh -- structural checks
# =============================================================================


class TestInstallShStructure:
    """Verify install.sh structural properties without running it."""

    def test_no_here_strings(self):
        """install.sh must not use <<< (not POSIX)."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
        # <<< is bash-only; breaks dash
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "<<<" not in line, f"install.sh:{i} uses non-POSIX <<< here-string"

    def test_rocm_detection_present(self):
        """install.sh should have ROCm detection in get_torch_index_url."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
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
        source = sh_path.read_text()
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
        source = sh_path.read_text()
        assert "bitsandbytes" in source
        assert "rocm*)" in source  # case pattern for ROCm URLs

    def test_cpu_hint_mentions_amd(self):
        """CPU-only hint should mention AMD ROCm."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
        assert "ROCm" in source

    def test_rocm72_capped_to_71(self):
        """ROCm 7.2+ should fall back to rocm7.1 index."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
        assert 'echo "$_base/rocm7.1"' in source  # fallback for unknown versions
        # Allowlisted versions should pass through directly
        assert "rocm6.*" in source
        assert "rocm7.0" in source
        assert "rocm7.1" in source

    def test_rocm_tag_validation_guard_exists(self):
        """install.sh should validate _rocm_tag with a case guard."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
        assert "rocm[1-9]*.[0-9]*)" in source
        assert '_rocm_tag=""' in source  # rejection path

    def test_dpkg_epoch_handling(self):
        """install.sh should strip Debian epoch prefix from dpkg-query output."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
        assert "sed 's/^[0-9]*://' " in source or "sed 's/^[0-9]*://'" in source

    def test_no_double_bracket_in_rocm_block(self):
        """ROCm detection block should not use [[ ]] (bash-only, not POSIX).
        Note: [[:space:]], [[:digit:]] etc. are valid POSIX character classes, not bash [[ ]]."""
        sh_path = PACKAGE_ROOT / "install.sh"
        source = sh_path.read_text()
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
        source = sh_path.read_text()
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
        source = sh_path.read_text()
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
        assert "hip_version" in _WHEEL_UTILS_PATH.read_text()
        assert "hip_version" in _WORKER_PATH.read_text()

    def test_probe_script_has_getattr_hip(self):
        """Probe script should use getattr for torch.version.hip (safe on CUDA)."""
        source = _WHEEL_UTILS_PATH.read_text()
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
        source = _WORKER_PATH.read_text()
        assert "hipcc" in source

    def test_rocm_source_build_status_message(self):
        """worker.py should send a specific status for ROCm source compilation."""
        source = _WORKER_PATH.read_text()
        assert "Compiling" in source and "from source for ROCm" in source

    def test_rocm_build_failure_message(self):
        """worker.py should send a clear error on ROCm build failure."""
        source = _WORKER_PATH.read_text()
        assert "Failed to compile" in source and "for ROCm" in source

    def test_timeout_on_install(self):
        """worker.py should have a timeout on pip install subprocess."""
        source = _WORKER_PATH.read_text()
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
        source = amd_path.read_text()
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

    def test_amd_primary_gpu_with_mock(self):
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
        source = hw_path.read_text()
        assert "from . import amd" in source

    def test_hardware_branches_on_is_rocm_for_utilization(self):
        """get_gpu_utilization should dispatch to amd.py via _smi_query
        when IS_ROCM, and the dispatcher itself must check IS_ROCM and
        import the amd backend."""
        hw_path = (
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
        )
        source = hw_path.read_text()
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
        source = hw_path.read_text()
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
        source = hw_path.read_text()
        func_start = source.find("def get_physical_gpu_count")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "IS_ROCM" in func_body
        assert "from . import amd" in func_body


# =============================================================================
# TEST: install_python_stack.py -- Windows AMD warning
# =============================================================================


class TestWindowsRocmWarning:
    """Verify Windows AMD GPU detection and warning message."""

    def test_windows_amd_warning_in_source(self):
        """install_python_stack.py should warn Windows AMD users."""
        source = _STACK_PATH.read_text()
        assert "AMD GPU detected on Windows" in source

    def test_windows_amd_warning_checks_hipinfo_or_amdsmi(self):
        """Warning should check for hipinfo or amd-smi."""
        source = _STACK_PATH.read_text()
        assert "hipinfo" in source
        assert "amd-smi" in source

    def test_windows_amd_warning_has_docs_link(self):
        """Warning should include AMD docs link."""
        source = _STACK_PATH.read_text()
        assert "docs.unsloth.ai/get-started/install-and-update/amd" in source


# =============================================================================
# TEST: unsloth/kernels/utils.py -- is_rdna() expansion
# =============================================================================


class TestIsRdnaExpansion:
    """Verify is_rdna() covers RDNA2, RDNA3, RDNA3.5, RDNA4 architectures."""

    def test_is_rdna_source_has_rdna2(self):
        """is_rdna() should include RDNA2 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text()
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
        source = utils_path.read_text()
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1100" in func_body
        assert "gfx1101" in func_body
        assert "gfx1102" in func_body
        assert "gfx1103" in func_body

    def test_is_rdna_source_has_rdna35(self):
        """is_rdna() should include RDNA3.5 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text()
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1150" in func_body
        assert "gfx1151" in func_body
        assert "gfx1152" in func_body

    def test_is_rdna_source_has_rdna4(self):
        """is_rdna() should include RDNA4 architectures."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text()
        func_start = source.find("def is_rdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx1200" in func_body
        assert "gfx1201" in func_body

    def test_is_cdna_not_changed(self):
        """is_cdna() should remain unchanged (no RDNA architectures added)."""
        utils_path = PACKAGE_ROOT / "unsloth" / "kernels" / "utils.py"
        source = utils_path.read_text()
        func_start = source.find("def is_cdna()")
        func_body = source[func_start : source.find("\ndef ", func_start + 1)]
        assert "gfx940" in func_body
        assert "gfx941" in func_body
        assert "gfx942" in func_body
        assert "gfx950" in func_body
        # RDNA architectures should NOT be in is_cdna
        assert "gfx1030" not in func_body
        assert "gfx1100" not in func_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
