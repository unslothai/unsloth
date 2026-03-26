"""Tests for binary selection logic in install_llama_prebuilt.py.

Covers: normalize_compute_cap, normalize_compute_caps, parse_cuda_visible_devices,
supports_explicit_visible_device_matching, select_visible_gpu_rows,
compatible_linux_runtime_lines, pick_windows_cuda_runtime,
compatible_windows_runtime_lines, runtime_line_from_cuda_version,
apply_approved_hashes, linux_cuda_choice_from_release, windows_cuda_attempts,
resolve_upstream_asset_choice.

No GPU, no network, no torch required -- all I/O is monkeypatched.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

HostInfo = INSTALL_LLAMA_PREBUILT.HostInfo
AssetChoice = INSTALL_LLAMA_PREBUILT.AssetChoice
PublishedLlamaArtifact = INSTALL_LLAMA_PREBUILT.PublishedLlamaArtifact
PublishedReleaseBundle = INSTALL_LLAMA_PREBUILT.PublishedReleaseBundle
ApprovedArtifactHash = INSTALL_LLAMA_PREBUILT.ApprovedArtifactHash
ApprovedReleaseChecksums = INSTALL_LLAMA_PREBUILT.ApprovedReleaseChecksums
PrebuiltFallback = INSTALL_LLAMA_PREBUILT.PrebuiltFallback
LinuxCudaSelection = INSTALL_LLAMA_PREBUILT.LinuxCudaSelection
UPSTREAM_REPO = INSTALL_LLAMA_PREBUILT.UPSTREAM_REPO

normalize_compute_cap = INSTALL_LLAMA_PREBUILT.normalize_compute_cap
normalize_compute_caps = INSTALL_LLAMA_PREBUILT.normalize_compute_caps
parse_cuda_visible_devices = INSTALL_LLAMA_PREBUILT.parse_cuda_visible_devices
supports_explicit_visible_device_matching = (
    INSTALL_LLAMA_PREBUILT.supports_explicit_visible_device_matching
)
select_visible_gpu_rows = INSTALL_LLAMA_PREBUILT.select_visible_gpu_rows
compatible_linux_runtime_lines = INSTALL_LLAMA_PREBUILT.compatible_linux_runtime_lines
pick_windows_cuda_runtime = INSTALL_LLAMA_PREBUILT.pick_windows_cuda_runtime
compatible_windows_runtime_lines = (
    INSTALL_LLAMA_PREBUILT.compatible_windows_runtime_lines
)
runtime_line_from_cuda_version = INSTALL_LLAMA_PREBUILT.runtime_line_from_cuda_version
apply_approved_hashes = INSTALL_LLAMA_PREBUILT.apply_approved_hashes
linux_cuda_choice_from_release = INSTALL_LLAMA_PREBUILT.linux_cuda_choice_from_release
windows_cuda_attempts = INSTALL_LLAMA_PREBUILT.windows_cuda_attempts
resolve_upstream_asset_choice = INSTALL_LLAMA_PREBUILT.resolve_upstream_asset_choice


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_host(**overrides):
    system = overrides.pop("system", "Linux")
    machine = overrides.pop("machine", "x86_64")
    defaults = dict(
        system = system,
        machine = machine,
        is_linux = system == "Linux",
        is_windows = system == "Windows",
        is_macos = system == "Darwin",
        is_x86_64 = machine.lower() in {"x86_64", "amd64"},
        is_arm64 = machine.lower() in {"arm64", "aarch64"},
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = (12, 8),
        compute_caps = ["86"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


def make_artifact(asset_name, **overrides):
    defaults = dict(
        asset_name = asset_name,
        install_kind = "linux-cuda",
        runtime_line = "cuda12",
        coverage_class = "targeted",
        supported_sms = ["75", "80", "86", "89", "90"],
        min_sm = 75,
        max_sm = 90,
        bundle_profile = "cuda12-newer",
        rank = 100,
    )
    defaults.update(overrides)
    return PublishedLlamaArtifact(**defaults)


def make_release(artifacts, **overrides):
    defaults = dict(
        repo = "unslothai/llama.cpp",
        release_tag = "v1.0",
        upstream_tag = "b8508",
        assets = {a.asset_name: f"https://example.com/{a.asset_name}" for a in artifacts},
        manifest_asset_name = "llama-prebuilt-manifest.json",
        artifacts = artifacts,
        selection_log = [],
    )
    defaults.update(overrides)
    return PublishedReleaseBundle(**defaults)


def make_checksums(asset_names):
    return ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "v1.0",
        upstream_tag = "b8508",
        source_commit = None,
        artifacts = {
            name: ApprovedArtifactHash(
                asset_name = name,
                sha256 = "a" * 64,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            )
            for name in asset_names
        },
    )


def mock_linux_runtime(monkeypatch, lines):
    dirs = {line: ["/usr/lib/stub"] for line in lines}
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "detected_linux_runtime_lines",
        lambda: (list(lines), dict(dirs)),
    )


def mock_windows_runtime(monkeypatch, lines):
    dirs = {line: ["C:\\Windows\\System32"] for line in lines}
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "detected_windows_runtime_lines",
        lambda: (list(lines), dict(dirs)),
    )


# ===========================================================================
# A. normalize_compute_cap
# ===========================================================================


class TestNormalizeComputeCap:
    def test_dotted_86(self):
        assert normalize_compute_cap("8.6") == "86"

    def test_dotted_leading_zero(self):
        assert normalize_compute_cap("07.05") == "75"

    def test_already_normalized(self):
        assert normalize_compute_cap("75") == "75"

    def test_int_input(self):
        assert normalize_compute_cap(86) == "86"

    def test_empty_string(self):
        assert normalize_compute_cap("") is None

    def test_whitespace(self):
        assert normalize_compute_cap("  ") is None

    def test_non_numeric(self):
        assert normalize_compute_cap("x.y") is None

    def test_triple_part(self):
        assert normalize_compute_cap("8.6.0") is None

    def test_zero_minor(self):
        assert normalize_compute_cap("9.0") == "90"


# ===========================================================================
# B. normalize_compute_caps
# ===========================================================================


class TestNormalizeComputeCaps:
    def test_deduplication(self):
        assert normalize_compute_caps(["8.6", "86", "8.6"]) == ["86"]

    def test_numeric_sort(self):
        assert normalize_compute_caps(["9.0", "7.5", "8.6"]) == ["75", "86", "90"]

    def test_drops_invalid(self):
        assert normalize_compute_caps(["8.6", "bad", "", "7.5"]) == ["75", "86"]

    def test_empty_input(self):
        assert normalize_compute_caps([]) == []


# ===========================================================================
# C. parse_cuda_visible_devices
# ===========================================================================


class TestParseCudaVisibleDevices:
    def test_none(self):
        assert parse_cuda_visible_devices(None) is None

    def test_empty(self):
        assert parse_cuda_visible_devices("") == []

    def test_minus_one(self):
        assert parse_cuda_visible_devices("-1") == []

    def test_single(self):
        assert parse_cuda_visible_devices("0") == ["0"]

    def test_multi(self):
        assert parse_cuda_visible_devices("0,1,2") == ["0", "1", "2"]

    def test_whitespace_stripped(self):
        assert parse_cuda_visible_devices(" 0 , 1 ") == ["0", "1"]


# ===========================================================================
# D. supports_explicit_visible_device_matching
# ===========================================================================


class TestSupportsExplicitVisibleDeviceMatching:
    def test_all_digits(self):
        assert supports_explicit_visible_device_matching(["0", "1", "2"]) is True

    def test_gpu_prefix(self):
        assert supports_explicit_visible_device_matching(["GPU-abc123"]) is True

    def test_none(self):
        assert supports_explicit_visible_device_matching(None) is False

    def test_empty(self):
        assert supports_explicit_visible_device_matching([]) is False

    def test_mixed_invalid(self):
        assert supports_explicit_visible_device_matching(["0", "MIG-device"]) is False


# ===========================================================================
# E. select_visible_gpu_rows
# ===========================================================================


class TestSelectVisibleGpuRows:
    ROWS = [
        ("0", "GPU-aaa", "8.6"),
        ("1", "GPU-bbb", "7.5"),
        ("2", "GPU-ccc", "8.9"),
    ]

    def test_none_returns_all(self):
        assert select_visible_gpu_rows(self.ROWS, None) == list(self.ROWS)

    def test_empty_returns_empty(self):
        assert select_visible_gpu_rows(self.ROWS, []) == []

    def test_filter_by_index(self):
        result = select_visible_gpu_rows(self.ROWS, ["0", "2"])
        assert result == [("0", "GPU-aaa", "8.6"), ("2", "GPU-ccc", "8.9")]

    def test_filter_by_uuid_case_insensitive(self):
        result = select_visible_gpu_rows(self.ROWS, ["gpu-bbb"])
        assert result == [("1", "GPU-bbb", "7.5")]

    def test_dedup_same_device(self):
        result = select_visible_gpu_rows(self.ROWS, ["0", "0"])
        assert result == [("0", "GPU-aaa", "8.6")]

    def test_missing_token(self):
        result = select_visible_gpu_rows(self.ROWS, ["99"])
        assert result == []


# ===========================================================================
# F. compatible_linux_runtime_lines
# ===========================================================================


class TestCompatibleLinuxRuntimeLines:
    def test_no_driver(self):
        host = make_host(driver_cuda_version = None)
        assert compatible_linux_runtime_lines(host) == []

    def test_driver_11_8(self):
        host = make_host(driver_cuda_version = (11, 8))
        assert compatible_linux_runtime_lines(host) == []

    def test_driver_12_4(self):
        host = make_host(driver_cuda_version = (12, 4))
        assert compatible_linux_runtime_lines(host) == ["cuda12"]

    def test_driver_13_0(self):
        host = make_host(driver_cuda_version = (13, 0))
        assert compatible_linux_runtime_lines(host) == ["cuda13", "cuda12"]


# ===========================================================================
# G. pick_windows_cuda_runtime + compatible_windows_runtime_lines
# ===========================================================================


class TestPickWindowsCudaRuntime:
    def test_no_driver(self):
        host = make_host(driver_cuda_version = None)
        assert pick_windows_cuda_runtime(host) is None

    def test_below_threshold(self):
        host = make_host(driver_cuda_version = (12, 3))
        assert pick_windows_cuda_runtime(host) is None

    def test_driver_12_4(self):
        host = make_host(driver_cuda_version = (12, 4))
        assert pick_windows_cuda_runtime(host) == "12.4"

    def test_driver_13_1(self):
        host = make_host(driver_cuda_version = (13, 1))
        assert pick_windows_cuda_runtime(host) == "13.1"


class TestCompatibleWindowsRuntimeLines:
    def test_no_driver(self):
        host = make_host(driver_cuda_version = None)
        assert compatible_windows_runtime_lines(host) == []

    def test_driver_12_4(self):
        host = make_host(driver_cuda_version = (12, 4))
        assert compatible_windows_runtime_lines(host) == ["cuda12"]

    def test_driver_13_1(self):
        host = make_host(driver_cuda_version = (13, 1))
        assert compatible_windows_runtime_lines(host) == ["cuda13", "cuda12"]


# ===========================================================================
# H. runtime_line_from_cuda_version
# ===========================================================================


class TestRuntimeLineFromCudaVersion:
    def test_cuda_12(self):
        assert runtime_line_from_cuda_version("12.6") == "cuda12"

    def test_cuda_13(self):
        assert runtime_line_from_cuda_version("13.0") == "cuda13"

    def test_cuda_11(self):
        assert runtime_line_from_cuda_version("11.8") is None

    def test_none(self):
        assert runtime_line_from_cuda_version(None) is None

    def test_empty(self):
        assert runtime_line_from_cuda_version("") is None


# ===========================================================================
# I. apply_approved_hashes
# ===========================================================================


class TestApplyApprovedHashes:
    def _choice(self, name):
        return AssetChoice(
            repo = "test",
            tag = "v1",
            name = name,
            url = f"https://x/{name}",
            source_label = "test",
        )

    def test_both_approved(self):
        c1, c2 = self._choice("a.tar.gz"), self._choice("b.tar.gz")
        checksums = make_checksums(["a.tar.gz", "b.tar.gz"])
        result = apply_approved_hashes([c1, c2], checksums)
        assert len(result) == 2
        assert all(c.expected_sha256 == "a" * 64 for c in result)

    def test_one_approved(self):
        c1, c2 = self._choice("a.tar.gz"), self._choice("missing.tar.gz")
        checksums = make_checksums(["a.tar.gz"])
        result = apply_approved_hashes([c1, c2], checksums)
        assert len(result) == 1
        assert result[0].name == "a.tar.gz"

    def test_none_approved(self):
        c1 = self._choice("missing.tar.gz")
        checksums = make_checksums(["other.tar.gz"])
        with pytest.raises(PrebuiltFallback, match = "approved checksum"):
            apply_approved_hashes([c1], checksums)

    def test_empty_input(self):
        checksums = make_checksums(["a.tar.gz"])
        with pytest.raises(PrebuiltFallback, match = "approved checksum"):
            apply_approved_hashes([], checksums)


# ===========================================================================
# J. linux_cuda_choice_from_release -- core selection
# ===========================================================================


class TestLinuxCudaChoiceFromRelease:
    # --- Runtime line resolution ---

    def test_no_runtime_lines_detected(self, monkeypatch):
        mock_linux_runtime(monkeypatch, [])
        host = make_host(driver_cuda_version = (12, 8))
        art = make_artifact("bundle-cuda12.tar.gz")
        release = make_release([art])
        assert linux_cuda_choice_from_release(host, release) is None

    def test_detected_lines_incompatible_with_driver(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(driver_cuda_version = (12, 4))
        art = make_artifact("bundle-cuda13.tar.gz", runtime_line = "cuda13")
        release = make_release([art])
        assert linux_cuda_choice_from_release(host, release) is None

    def test_driver_13_only_cuda12_detected(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(driver_cuda_version = (13, 0))
        art = make_artifact("bundle-cuda12.tar.gz", runtime_line = "cuda12")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.runtime_line == "cuda12"

    def test_preferred_runtime_line_reorders(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (13, 0))
        art12 = make_artifact("bundle-cuda12.tar.gz", runtime_line = "cuda12")
        art13 = make_artifact("bundle-cuda13.tar.gz", runtime_line = "cuda13")
        release = make_release([art12, art13])
        result = linux_cuda_choice_from_release(
            host, release, preferred_runtime_line = "cuda12"
        )
        assert result is not None
        assert result.primary.runtime_line == "cuda12"

    def test_preferred_runtime_line_unavailable(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(driver_cuda_version = (12, 8))
        art = make_artifact("bundle-cuda12.tar.gz", runtime_line = "cuda12")
        release = make_release([art])
        result = linux_cuda_choice_from_release(
            host, release, preferred_runtime_line = "cuda13"
        )
        assert result is not None
        assert result.primary.runtime_line == "cuda12"
        log_entries = result.selection_log
        assert any("unavailable_on_host" in entry for entry in log_entries)

    # --- SM matching ---

    def test_exact_sm_match(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact(
            "bundle.tar.gz", supported_sms = ["75", "86", "89"], min_sm = 75, max_sm = 89
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "bundle.tar.gz"

    def test_sm_not_in_supported_sms(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact(
            "bundle.tar.gz", supported_sms = ["75", "80", "89"], min_sm = 75, max_sm = 89
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_sm_outside_min_range(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["50"])
        art = make_artifact(
            "bundle.tar.gz", supported_sms = ["50", "75", "86"], min_sm = 75, max_sm = 90
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_sm_outside_max_range(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["100"])
        art = make_artifact(
            "bundle.tar.gz", supported_sms = ["100", "75", "86"], min_sm = 75, max_sm = 90
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_very_old_sm(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["50"])
        art = make_artifact("bundle.tar.gz", min_sm = 75, max_sm = 90)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_very_new_sm(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["100"])
        art = make_artifact("bundle.tar.gz", min_sm = 75, max_sm = 90)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    # --- Unknown compute caps (empty list) ---

    def test_unknown_caps_only_portable(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = [])
        targeted = make_artifact("targeted.tar.gz", coverage_class = "targeted")
        portable = make_artifact("portable.tar.gz", coverage_class = "portable")
        release = make_release([targeted, portable])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "portable.tar.gz"

    def test_unknown_caps_no_portable(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = [])
        targeted = make_artifact("targeted.tar.gz", coverage_class = "targeted")
        release = make_release([targeted])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    # --- Multi-GPU ---

    def test_multi_gpu_all_covered(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["75", "89"])
        art = make_artifact(
            "bundle.tar.gz",
            supported_sms = ["75", "80", "86", "89", "90"],
            min_sm = 75,
            max_sm = 90,
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None

    def test_multi_gpu_not_all_covered(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["50", "89"])
        art = make_artifact(
            "bundle.tar.gz", supported_sms = ["75", "89"], min_sm = 75, max_sm = 89
        )
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    # --- Artifact selection priority ---

    def test_narrowest_sm_range_wins(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        wide = make_artifact(
            "wide.tar.gz",
            supported_sms = ["75", "86", "90"],
            min_sm = 75,
            max_sm = 90,
            rank = 100,
        )
        narrow = make_artifact(
            "narrow.tar.gz",
            supported_sms = ["80", "86", "89"],
            min_sm = 80,
            max_sm = 89,
            rank = 100,
        )
        release = make_release([wide, narrow])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "narrow.tar.gz"

    def test_range_tie_lower_rank_wins(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        high = make_artifact(
            "high.tar.gz",
            supported_sms = ["75", "86", "90"],
            min_sm = 75,
            max_sm = 90,
            rank = 200,
        )
        low = make_artifact(
            "low.tar.gz",
            supported_sms = ["75", "86", "90"],
            min_sm = 75,
            max_sm = 90,
            rank = 50,
        )
        release = make_release([high, low])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "low.tar.gz"

    def test_targeted_preferred_portable_fallback(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        targeted = make_artifact("targeted.tar.gz", coverage_class = "targeted", rank = 100)
        portable = make_artifact("portable.tar.gz", coverage_class = "portable", rank = 100)
        release = make_release([targeted, portable])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "targeted.tar.gz"
        assert len(result.attempts) == 2
        assert result.attempts[1].name == "portable.tar.gz"

    # --- Edge cases ---

    def test_asset_missing_from_release_assets(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz")
        release = make_release([art], assets = {})
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_artifact_empty_supported_sms(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", supported_sms = [])
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_artifact_missing_min_sm(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", min_sm = None, max_sm = 90)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_artifact_missing_max_sm(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", min_sm = 75, max_sm = None)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_no_linux_cuda_artifacts(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", install_kind = "windows-cuda")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_empty_artifacts_list(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        release = make_release([])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None


# ===========================================================================
# K. windows_cuda_attempts
# ===========================================================================


class TestWindowsCudaAttempts:
    TAG = "b8508"

    def _upstream(self, *runtime_versions):
        assets = {}
        for rv in runtime_versions:
            name = f"llama-{self.TAG}-bin-win-cuda-{rv}-x64.zip"
            assets[name] = f"https://example.com/{name}"
        return assets

    def test_driver_12_4_no_dlls_fallback(self, monkeypatch):
        mock_windows_runtime(monkeypatch, [])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (12, 4))
        assets = self._upstream("12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 1
        assert result[0].runtime_line == "cuda12"

    def test_driver_13_1_both_dlls(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 2
        assert result[0].runtime_line == "cuda13"
        assert result[1].runtime_line == "cuda12"

    def test_preferred_reorders(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, "cuda12")
        assert len(result) == 2
        assert result[0].runtime_line == "cuda12"

    def test_preferred_unavailable(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (12, 4))
        assets = self._upstream("12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, "cuda13")
        assert len(result) == 1
        assert result[0].runtime_line == "cuda12"

    def test_detected_incompatible_with_driver(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (12, 4))
        assets = self._upstream("12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 1
        assert result[0].runtime_line == "cuda12"

    def test_driver_too_old(self, monkeypatch):
        mock_windows_runtime(monkeypatch, [])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (11, 8))
        assets = self._upstream("12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result == []

    def test_asset_missing_from_upstream(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (12, 4))
        result = windows_cuda_attempts(host, self.TAG, {}, None)
        assert result == []

    def test_both_assets_present(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 2


# ===========================================================================
# L. resolve_upstream_asset_choice -- platform routing
# ===========================================================================


class TestResolveUpstreamAssetChoice:
    TAG = "b8508"

    def _mock_github_assets(self, monkeypatch, assets):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: assets,
        )

    def test_linux_x86_64_cpu(self, monkeypatch):
        name = f"llama-{self.TAG}-bin-ubuntu-x64.tar.gz"
        self._mock_github_assets(monkeypatch, {name: f"https://x/{name}"})
        host = make_host(
            has_usable_nvidia = False, nvidia_smi = None, has_physical_nvidia = False
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "linux-cpu"
        assert result.name == name

    def test_linux_cpu_missing(self, monkeypatch):
        self._mock_github_assets(monkeypatch, {})
        host = make_host(
            has_usable_nvidia = False, nvidia_smi = None, has_physical_nvidia = False
        )
        with pytest.raises(PrebuiltFallback, match = "Linux CPU"):
            resolve_upstream_asset_choice(host, self.TAG)

    def test_windows_x86_64_cpu(self, monkeypatch):
        name = f"llama-{self.TAG}-bin-win-cpu-x64.zip"
        self._mock_github_assets(monkeypatch, {name: f"https://x/{name}"})
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            has_usable_nvidia = False,
            nvidia_smi = None,
            has_physical_nvidia = False,
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "windows-cpu"
        assert result.name == name

    def test_windows_cpu_missing(self, monkeypatch):
        self._mock_github_assets(monkeypatch, {})
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            has_usable_nvidia = False,
            nvidia_smi = None,
            has_physical_nvidia = False,
        )
        with pytest.raises(PrebuiltFallback, match = "Windows CPU"):
            resolve_upstream_asset_choice(host, self.TAG)

    def test_macos_arm64(self, monkeypatch):
        name = f"llama-{self.TAG}-bin-macos-arm64.tar.gz"
        self._mock_github_assets(monkeypatch, {name: f"https://x/{name}"})
        host = make_host(
            system = "Darwin",
            machine = "arm64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "macos-arm64"
        assert result.name == name

    def test_macos_arm64_missing(self, monkeypatch):
        self._mock_github_assets(monkeypatch, {})
        host = make_host(
            system = "Darwin",
            machine = "arm64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        with pytest.raises(PrebuiltFallback, match = "macOS arm64"):
            resolve_upstream_asset_choice(host, self.TAG)

    def test_macos_x86_64(self, monkeypatch):
        name = f"llama-{self.TAG}-bin-macos-x64.tar.gz"
        self._mock_github_assets(monkeypatch, {name: f"https://x/{name}"})
        host = make_host(
            system = "Darwin",
            machine = "x86_64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "macos-x64"
        assert result.name == name

    def test_linux_aarch64(self, monkeypatch):
        self._mock_github_assets(monkeypatch, {})
        host = make_host(
            system = "Linux",
            machine = "aarch64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        with pytest.raises(
            PrebuiltFallback, match = "no prebuilt policy exists for Linux aarch64"
        ):
            resolve_upstream_asset_choice(host, self.TAG)

    def test_windows_usable_nvidia_delegates(self, monkeypatch):
        cuda_name = f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip"
        self._mock_github_assets(monkeypatch, {cuda_name: f"https://x/{cuda_name}"})
        mock_windows_runtime(monkeypatch, ["cuda12"])
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_windows_cuda_choices",
            lambda host, tag, assets: [
                AssetChoice(
                    repo = UPSTREAM_REPO,
                    tag = tag,
                    name = cuda_name,
                    url = f"https://x/{cuda_name}",
                    source_label = "upstream",
                    install_kind = "windows-cuda",
                    runtime_line = "cuda12",
                )
            ],
        )
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (12, 4),
            has_usable_nvidia = True,
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "windows-cuda"
        assert result.name == cuda_name
