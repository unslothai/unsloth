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
parse_direct_linux_release_bundle = (
    INSTALL_LLAMA_PREBUILT.parse_direct_linux_release_bundle
)
windows_cuda_attempts = INSTALL_LLAMA_PREBUILT.windows_cuda_attempts
resolve_upstream_asset_choice = INSTALL_LLAMA_PREBUILT.resolve_upstream_asset_choice
resolve_requested_install_tag = INSTALL_LLAMA_PREBUILT.resolve_requested_install_tag
resolve_install_attempts = INSTALL_LLAMA_PREBUILT.resolve_install_attempts
resolve_install_release_plans = INSTALL_LLAMA_PREBUILT.resolve_install_release_plans
resolve_published_release = INSTALL_LLAMA_PREBUILT.resolve_published_release
resolve_source_build_plan = INSTALL_LLAMA_PREBUILT.resolve_source_build_plan
validated_checksums_for_bundle = INSTALL_LLAMA_PREBUILT.validated_checksums_for_bundle
parse_approved_release_checksums = (
    INSTALL_LLAMA_PREBUILT.parse_approved_release_checksums
)
published_release_matches_request = (
    INSTALL_LLAMA_PREBUILT.published_release_matches_request
)
exact_source_archive_logical_name = (
    INSTALL_LLAMA_PREBUILT.exact_source_archive_logical_name
)
source_archive_logical_name = INSTALL_LLAMA_PREBUILT.source_archive_logical_name
windows_cuda_upstream_asset_names = (
    INSTALL_LLAMA_PREBUILT.windows_cuda_upstream_asset_names
)
env_int = INSTALL_LLAMA_PREBUILT.env_int
direct_upstream_release_plan = INSTALL_LLAMA_PREBUILT.direct_upstream_release_plan
_pinned_windows_cuda_fallback = INSTALL_LLAMA_PREBUILT._pinned_windows_cuda_fallback
CudaRuntimePreference = INSTALL_LLAMA_PREBUILT.CudaRuntimePreference
published_windows_cuda_attempts = INSTALL_LLAMA_PREBUILT.published_windows_cuda_attempts
_windows_cuda_attempt_covers_blackwell = (
    INSTALL_LLAMA_PREBUILT._windows_cuda_attempt_covers_blackwell
)
resolve_release_asset_choice = INSTALL_LLAMA_PREBUILT.resolve_release_asset_choice
pinned_macos_release_tag = INSTALL_LLAMA_PREBUILT.pinned_macos_release_tag
resolve_simple_install_release_plans = (
    INSTALL_LLAMA_PREBUILT.resolve_simple_install_release_plans
)


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
        source_repo = None,
        source_repo_url = None,
        source_ref_kind = None,
        requested_source_ref = None,
        resolved_source_ref = None,
        source_commit = None,
        source_commit_short = None,
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
        source_repo = None,
        source_repo_url = None,
        source_ref_kind = None,
        requested_source_ref = None,
        resolved_source_ref = None,
        source_commit = None,
        source_commit_short = None,
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


def make_checksums_with_source(
    asset_names,
    *,
    release_tag = "v1.0",
    upstream_tag = "b8508",
    source_repo = None,
    source_repo_url = None,
    source_ref_kind = None,
    requested_source_ref = None,
    resolved_source_ref = None,
    source_commit = None,
):
    artifacts = {
        **{
            name: ApprovedArtifactHash(
                asset_name = name,
                sha256 = "a" * 64,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            )
            for name in asset_names
        },
        source_archive_logical_name(upstream_tag): ApprovedArtifactHash(
            asset_name = source_archive_logical_name(upstream_tag),
            sha256 = "b" * 64,
            repo = "ggml-org/llama.cpp",
            kind = "upstream-source",
        ),
    }
    normalized_source_commit = (
        source_commit.lower() if isinstance(source_commit, str) else None
    )
    if normalized_source_commit:
        artifacts[exact_source_archive_logical_name(normalized_source_commit)] = (
            ApprovedArtifactHash(
                asset_name = exact_source_archive_logical_name(normalized_source_commit),
                sha256 = "c" * 64,
                repo = source_repo or "example/custom-llama.cpp",
                kind = "exact-source",
            )
        )
    return ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        source_repo = source_repo,
        source_repo_url = source_repo_url,
        source_ref_kind = source_ref_kind,
        requested_source_ref = requested_source_ref,
        resolved_source_ref = resolved_source_ref,
        source_commit = normalized_source_commit,
        source_commit_short = normalized_source_commit[:7]
        if normalized_source_commit
        else None,
        artifacts = artifacts,
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

    def test_future_major_derives_lines(self):
        # A future major (14.x) offers cuda14 first, then older majors.
        host = make_host(driver_cuda_version = (14, 0))
        assert compatible_linux_runtime_lines(host) == ["cuda14", "cuda13", "cuda12"]


class TestParseDirectLinuxReleaseBundle:
    def _release(self, *targets):
        names = [f"app-bTEST-linux-x64-{t}.tar.gz" for t in targets]
        return {
            "tag_name": "bTEST",
            "assets": [
                {"name": n, "browser_download_url": "https://x/" + n} for n in names
            ],
        }

    def _cuda_artifact(self, bundle):
        return [a for a in bundle.artifacts if a.install_kind == "linux-cuda"][0]

    def test_parses_known_cuda13_bundle(self):
        bundle = parse_direct_linux_release_bundle(
            "unslothai/llama.cpp", self._release("cuda13-newer")
        )
        assert bundle is not None
        assert self._cuda_artifact(bundle).runtime_line == "cuda13"

    def test_parses_future_cuda_major_with_forward_profile(self):
        # A future major name parses and inherits the newest known major's
        # coverage for the same class as a forward default.
        bundle = parse_direct_linux_release_bundle(
            "unslothai/llama.cpp", self._release("cuda14-newer")
        )
        assert bundle is not None
        art = self._cuda_artifact(bundle)
        assert art.runtime_line == "cuda14"
        assert art.coverage_class == "newer"
        assert art.max_sm == 120  # inherited from cuda13-newer


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

    def test_driver_13_0_uses_cuda13_line(self):
        host = make_host(driver_cuda_version = (13, 0))
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

    def test_driver_13_0_uses_cuda13_line(self):
        host = make_host(driver_cuda_version = (13, 0))
        assert compatible_windows_runtime_lines(host) == ["cuda13", "cuda12"]

    def test_future_major_derives_lines(self):
        host = make_host(driver_cuda_version = (14, 0))
        assert compatible_windows_runtime_lines(host) == ["cuda14", "cuda13", "cuda12"]


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

    def test_upstream_asset_can_match_compatibility_tag_name(self):
        choice = AssetChoice(
            repo = UPSTREAM_REPO,
            tag = "main",
            name = "llama-main-bin-macos-arm64.tar.gz",
            url = "https://x/llama-main-bin-macos-arm64.tar.gz",
            source_label = "upstream",
        )
        checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "r1",
            upstream_tag = "b9000",
            artifacts = {
                "llama-b9000-bin-macos-arm64.tar.gz": ApprovedArtifactHash(
                    asset_name = "llama-b9000-bin-macos-arm64.tar.gz",
                    sha256 = "a" * 64,
                    repo = UPSTREAM_REPO,
                    kind = "macos-arm64-upstream",
                )
            },
        )

        result = apply_approved_hashes([choice], checksums)
        assert result[0].expected_sha256 == "a" * 64

    def test_windows_cuda_legacy_choice_can_match_current_upstream_name(self):
        choice = AssetChoice(
            repo = UPSTREAM_REPO,
            tag = "b9000",
            name = "llama-b9000-bin-win-cuda-13.1-x64.zip",
            url = "https://x/llama-b9000-bin-win-cuda-13.1-x64.zip",
            source_label = "upstream",
        )
        checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "r1",
            upstream_tag = "b9000",
            artifacts = {
                "cudart-llama-bin-win-cuda-13.1-x64.zip": ApprovedArtifactHash(
                    asset_name = "cudart-llama-bin-win-cuda-13.1-x64.zip",
                    sha256 = "b" * 64,
                    repo = UPSTREAM_REPO,
                    kind = "windows-cuda-upstream",
                )
            },
        )

        result = apply_approved_hashes([choice], checksums)
        assert result[0].expected_sha256 == "b" * 64

    def test_windows_cuda_current_choice_can_match_legacy_compatibility_name(self):
        choice = AssetChoice(
            repo = UPSTREAM_REPO,
            tag = "main",
            name = "cudart-llama-bin-win-cuda-13.1-x64.zip",
            url = "https://x/cudart-llama-bin-win-cuda-13.1-x64.zip",
            source_label = "upstream",
        )
        checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "r1",
            upstream_tag = "b9000",
            artifacts = {
                "llama-b9000-bin-win-cuda-13.1-x64.zip": ApprovedArtifactHash(
                    asset_name = "llama-b9000-bin-win-cuda-13.1-x64.zip",
                    sha256 = "c" * 64,
                    repo = UPSTREAM_REPO,
                    kind = "windows-cuda-upstream",
                )
            },
        )

        result = apply_approved_hashes([choice], checksums)
        assert result[0].expected_sha256 == "c" * 64

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
# J. published release resolution
# ===========================================================================


class TestPublishedReleaseResolution:
    def test_latest_skips_invalid_release_and_uses_next_valid(self, monkeypatch):
        invalid = make_release([], release_tag = "v2.0", upstream_tag = "b9000")
        valid = make_release([], release_tag = "v1.0", upstream_tag = "b8999")

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([invalid, valid]),
        )

        def fake_load(repo, release_tag):
            if release_tag == "v2.0":
                raise PrebuiltFallback("checksum asset missing")
            return make_checksums_with_source(
                [], release_tag = "v1.0", upstream_tag = "b8999"
            )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            fake_load,
        )

        resolved = resolve_published_release("latest", "unslothai/llama.cpp")
        assert resolved.bundle.release_tag == "v1.0"
        assert resolved.bundle.upstream_tag == "b8999"
        assert resolved.checksums.release_tag == "v1.0"

    def test_concrete_tag_matches_manifest_upstream_tag(self, monkeypatch):
        release = make_release([], release_tag = "release-b8508", upstream_tag = "b8508")
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([release]),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: make_checksums_with_source(
                [],
                release_tag = release_tag,
                upstream_tag = "b8508",
            ),
        )

        assert (
            resolve_requested_install_tag("b8508", "", "unslothai/llama.cpp") == "b8508"
        )

    def test_concrete_tag_without_matching_release_raises(self, monkeypatch):
        release = make_release([], release_tag = "release-b9000", upstream_tag = "b9000")
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([release]),
        )

        with pytest.raises(PrebuiltFallback, match = "matched upstream tag b8508"):
            resolve_requested_install_tag("b8508", "", "unslothai/llama.cpp")

    def test_pinned_release_must_match_requested_upstream_tag(self, monkeypatch):
        bundle = make_release(
            [], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000"
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "pinned_published_release_bundle",
            lambda repo, release_tag: bundle,
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: make_checksums_with_source(
                [],
                release_tag = release_tag,
                upstream_tag = "b9000",
            ),
        )

        with pytest.raises(PrebuiltFallback, match = "but requested b8508"):
            resolve_requested_install_tag(
                "b8508",
                "llama-prebuilt-latest",
                "unslothai/llama.cpp",
            )

    def test_request_matches_requested_source_ref(self, monkeypatch):
        release = make_release(
            [],
            release_tag = "release-main",
            upstream_tag = "b9000",
            requested_source_ref = "main",
            resolved_source_ref = "refs/heads/main",
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([release]),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: make_checksums_with_source(
                [],
                release_tag = release_tag,
                upstream_tag = "b9000",
                requested_source_ref = "main",
                resolved_source_ref = "refs/heads/main",
            ),
        )

        resolved = resolve_published_release("main", "unslothai/llama.cpp")
        assert resolved.bundle.release_tag == "release-main"

    def test_request_matches_source_commit(self, monkeypatch):
        commit = "a" * 40
        release = make_release(
            [],
            release_tag = "release-commit",
            upstream_tag = "b9000",
            source_commit = commit,
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_published_release_bundles",
            lambda repo, published_release_tag = "": iter([release]),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: make_checksums_with_source(
                [],
                release_tag = release_tag,
                upstream_tag = "b9000",
                source_commit = commit,
            ),
        )

        resolved = resolve_published_release(commit, "unslothai/llama.cpp")
        assert resolved.bundle.release_tag == "release-commit"


class TestSourceBuildPlanResolution:
    def test_matches_request_by_non_tag_provenance(self):
        bundle = make_release(
            [],
            requested_source_ref = "main",
            resolved_source_ref = "refs/heads/main",
            source_commit = "a" * 40,
        )
        assert published_release_matches_request(bundle, "main") is True
        assert published_release_matches_request(bundle, "refs/heads/main") is True
        assert published_release_matches_request(bundle, "a" * 12) is True
        assert published_release_matches_request(bundle, "a" * 40) is True

    def test_matches_pull_ref_aliases(self):
        bundle = make_release(
            [],
            requested_source_ref = "refs/pull/123/head",
            resolved_source_ref = "pull/123/head",
        )
        assert published_release_matches_request(bundle, "refs/pull/123/head") is True
        assert published_release_matches_request(bundle, "pull/123/head") is True

    def test_prefers_exact_source_commit_when_available(self, monkeypatch):
        commit = "a" * 40
        resolved = INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
            bundle = make_release(
                [],
                release_tag = "release-main",
                upstream_tag = "b9000",
                source_repo = "example/custom-llama.cpp",
                source_repo_url = "https://github.com/example/custom-llama.cpp",
                source_ref_kind = "branch",
                requested_source_ref = "main",
                resolved_source_ref = "refs/heads/main",
                source_commit = commit,
            ),
            checksums = make_checksums_with_source(
                [],
                release_tag = "release-main",
                upstream_tag = "b9000",
                source_repo = "example/custom-llama.cpp",
                source_repo_url = "https://github.com/example/custom-llama.cpp",
                source_ref_kind = "branch",
                requested_source_ref = "main",
                resolved_source_ref = "refs/heads/main",
                source_commit = commit,
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_published_release",
            lambda requested_tag, published_repo, published_release_tag = "": resolved,
        )

        plan = resolve_source_build_plan("main", "unslothai/llama.cpp")
        assert plan.source_url == "https://github.com/example/custom-llama.cpp"
        assert plan.source_ref_kind == "commit"
        assert plan.source_ref == commit
        assert plan.compatibility_upstream_tag == "b9000"

    def test_uses_branch_provenance_without_exact_source_hash(self, monkeypatch):
        resolved = INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
            bundle = make_release(
                [],
                release_tag = "release-main",
                upstream_tag = "b9000",
                source_repo = "example/custom-llama.cpp",
                source_repo_url = "https://github.com/example/custom-llama.cpp",
                source_ref_kind = "branch",
                requested_source_ref = "main",
                resolved_source_ref = "main",
            ),
            checksums = make_checksums_with_source(
                [],
                release_tag = "release-main",
                upstream_tag = "b9000",
                source_repo = "example/custom-llama.cpp",
                source_repo_url = "https://github.com/example/custom-llama.cpp",
                source_ref_kind = "branch",
                requested_source_ref = "main",
                resolved_source_ref = "main",
                source_commit = None,
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_published_release",
            lambda requested_tag, published_repo, published_release_tag = "": resolved,
        )

        plan = resolve_source_build_plan("main", "unslothai/llama.cpp")
        assert plan.source_url == "https://github.com/example/custom-llama.cpp"
        assert plan.source_ref_kind == "branch"
        assert plan.source_ref == "main"
        assert plan.compatibility_upstream_tag == "b9000"

    def test_direct_main_request_without_published_release_uses_branch_kind(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_published_release",
            lambda requested_tag, published_repo, published_release_tag = "": (
                _ for _ in ()
            ).throw(PrebuiltFallback("missing")),
        )

        plan = resolve_source_build_plan("main", "unslothai/llama.cpp")
        assert plan.source_url == "https://github.com/ggml-org/llama.cpp"
        assert plan.source_ref_kind == "branch"
        assert plan.source_ref == "main"


class TestParseApprovedReleaseChecksums:
    def test_rejects_wrong_component(self):
        with pytest.raises(RuntimeError, match = "did not describe llama.cpp"):
            parse_approved_release_checksums(
                "repo/test",
                "r1",
                {
                    "schema_version": 1,
                    "component": "other",
                    "release_tag": "r1",
                    "upstream_tag": "b8508",
                    "artifacts": {},
                },
            )

    def test_rejects_mismatched_release_tag(self):
        with pytest.raises(RuntimeError, match = "did not match pinned release tag"):
            parse_approved_release_checksums(
                "repo/test",
                "r1",
                {
                    "schema_version": 1,
                    "component": "llama.cpp",
                    "release_tag": "r2",
                    "upstream_tag": "b8508",
                    "artifacts": {},
                },
            )

    def test_rejects_bad_sha256(self):
        with pytest.raises(RuntimeError, match = "valid sha256"):
            parse_approved_release_checksums(
                "repo/test",
                "r1",
                {
                    "schema_version": 1,
                    "component": "llama.cpp",
                    "release_tag": "r1",
                    "upstream_tag": "b8508",
                    "artifacts": {
                        "asset.tar.gz": {
                            "sha256": "bad-digest",
                        }
                    },
                },
            )

    def test_rejects_unsupported_schema_version(self):
        with pytest.raises(RuntimeError, match = "schema_version=2 is unsupported"):
            parse_approved_release_checksums(
                "repo/test",
                "r1",
                {
                    "schema_version": 2,
                    "component": "llama.cpp",
                    "release_tag": "r1",
                    "upstream_tag": "b8508",
                    "artifacts": {},
                },
            )


class TestValidatedChecksumsForBundle:
    def test_rejects_manifest_checksum_mismatch(self, monkeypatch):
        bundle = make_release([], release_tag = "r1", upstream_tag = "b8508")
        bundle.manifest_sha256 = "a" * 64
        checksums = make_checksums_with_source(
            [], release_tag = "r1", upstream_tag = "b8508"
        )
        checksums.artifacts[bundle.manifest_asset_name] = ApprovedArtifactHash(
            asset_name = bundle.manifest_asset_name,
            sha256 = "b" * 64,
            repo = "unslothai/llama.cpp",
            kind = "published-manifest",
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: checksums,
        )

        with pytest.raises(PrebuiltFallback, match = "manifest checksum"):
            validated_checksums_for_bundle("unslothai/llama.cpp", bundle)


# ===========================================================================
# K. linux_cuda_choice_from_release -- core selection
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

    def test_arm64_host_selects_linux_arm64_cuda_kind(self, monkeypatch):
        # An arm64 CUDA host (DGX Spark / Grace Hopper) selects the
        # linux-arm64-cuda bundle and ignores the x64 linux-cuda one.
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(
            machine = "aarch64",
            driver_cuda_version = (13, 0),
            compute_caps = ["90"],
        )
        arm = make_artifact(
            "app-b9457-linux-arm64-cuda13-portable.tar.gz",
            install_kind = "linux-arm64-cuda",
            runtime_line = "cuda13",
            coverage_class = "portable",
            supported_sms = ["90", "100", "120", "121"],
            min_sm = 90,
            max_sm = 121,
            bundle_profile = "cuda13-portable",
        )
        x64 = make_artifact(
            "app-b9457-linux-x64-cuda13-portable.tar.gz",
            install_kind = "linux-cuda",
            runtime_line = "cuda13",
        )
        release = make_release([arm, x64])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.install_kind == "linux-arm64-cuda"
        assert (
            result.primary.name == "app-b9457-linux-arm64-cuda13-portable.tar.gz"
        )

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


def make_profile_artifact(asset_name, profile_name, **overrides):
    profile = INSTALL_LLAMA_PREBUILT.DIRECT_LINUX_BUNDLE_PROFILES[profile_name]
    defaults = dict(
        runtime_line = profile["runtime_line"],
        coverage_class = profile["coverage_class"],
        supported_sms = [str(value) for value in profile["supported_sms"]],
        min_sm = int(profile["min_sm"]),
        max_sm = int(profile["max_sm"]),
        bundle_profile = profile_name,
        rank = int(profile["rank"]),
    )
    defaults.update(overrides)
    return make_artifact(asset_name, **defaults)


class TestBlackwellUltraSm103Coverage:
    """sm_103 (B300 / GB300) runs on the bundled base compute_100 PTX via JIT."""

    def test_profiles_list_sm103_wherever_sm100_is_shipped(self):
        for (
            name,
            profile,
        ) in INSTALL_LLAMA_PREBUILT.DIRECT_LINUX_BUNDLE_PROFILES.items():
            sms = {str(value) for value in profile["supported_sms"]}
            if "100" in sms:
                assert "103" in sms, name
            else:
                assert "103" not in sms, name

    def test_b300_selects_cuda13_newer_prebuilt(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(compute_caps = ["103"], driver_cuda_version = (13, 0))
        art = make_profile_artifact("cuda13-newer.tar.gz", "cuda13-newer")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "cuda13-newer.tar.gz"

    def test_b300_selects_cuda12_newer_prebuilt(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["103"], driver_cuda_version = (12, 8))
        art = make_profile_artifact("cuda12-newer.tar.gz", "cuda12-newer")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "cuda12-newer.tar.gz"

    def test_b300_reported_as_decimal_normalizes_and_matches(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(compute_caps = ["10.3"], driver_cuda_version = (13, 0))
        art = make_profile_artifact("cuda13-portable.tar.gz", "cuda13-portable")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None

    def test_b300_falls_back_to_portable_when_only_portable_present(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(compute_caps = ["103"], driver_cuda_version = (13, 0))
        art = make_profile_artifact("cuda13-portable.tar.gz", "cuda13-portable")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "cuda13-portable.tar.gz"

    def test_older_bundle_still_rejects_b300(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda13"])
        host = make_host(compute_caps = ["103"], driver_cuda_version = (13, 0))
        art = make_profile_artifact("cuda13-older.tar.gz", "cuda13-older")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None


# ===========================================================================
# L. resolve_install_attempts
# ===========================================================================


class TestResolveInstallAttempts:
    def test_windows_cuda_prefers_published_asset_from_selected_release(
        self, monkeypatch
    ):
        host = make_host(system = "Windows", machine = "AMD64")
        host.driver_cuda_version = (12, 4)
        mock_windows_runtime(monkeypatch, ["cuda12"])
        asset_name = "llama-b9000-bin-win-cuda-12.4-x64.zip"
        release = make_release(
            [
                make_artifact(
                    asset_name,
                    install_kind = "windows-cuda",
                    runtime_line = "cuda12",
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9000",
            assets = {asset_name: f"https://published.example/{asset_name}"},
        )
        checksums = make_checksums_with_source(
            [asset_name],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: (_ for _ in ()).throw(
                AssertionError(
                    "published Windows CUDA choice should not query upstream"
                )
            ),
        )

        requested_tag, resolved_tag, attempts, approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert requested_tag == "latest"
        assert resolved_tag == "b9000"
        assert attempts[0].name == asset_name
        assert attempts[0].source_label == "published"
        assert attempts[0].expected_sha256 == "a" * 64
        assert approved.release_tag == "llama-prebuilt-latest"

    def test_windows_cuda_uses_selected_release_upstream_tag(self, monkeypatch):
        host = make_host(system = "Windows", machine = "AMD64")
        host.driver_cuda_version = (12, 4)
        mock_windows_runtime(monkeypatch, ["cuda12"])
        release = make_release(
            [], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000"
        )
        checksums = make_checksums_with_source(
            ["llama-b9000-bin-win-cuda-12.4-x64.zip"],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: {
                f"llama-{tag}-bin-win-cuda-12.4-x64.zip": f"https://example.com/llama-{tag}-bin-win-cuda-12.4-x64.zip"
            },
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_windows_cuda_choices",
            lambda host, tag, assets: [
                AssetChoice(
                    repo = UPSTREAM_REPO,
                    tag = tag,
                    name = f"llama-{tag}-bin-win-cuda-12.4-x64.zip",
                    url = assets[f"llama-{tag}-bin-win-cuda-12.4-x64.zip"],
                    source_label = "upstream",
                    install_kind = "windows-cuda",
                    runtime_line = "cuda12",
                )
            ],
        )

        requested_tag, resolved_tag, attempts, approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert requested_tag == "latest"
        assert resolved_tag == "b9000"
        assert attempts[0].name == "llama-b9000-bin-win-cuda-12.4-x64.zip"
        assert attempts[0].expected_sha256 == "a" * 64
        assert approved.release_tag == "llama-prebuilt-latest"

    def test_linux_cpu_uses_same_tag_upstream_asset(self, monkeypatch):
        host = make_host(
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        release = make_release(
            [], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000"
        )
        checksums = make_checksums_with_source(
            ["llama-b9000-bin-ubuntu-x64.tar.gz"],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: {
                f"llama-{tag}-bin-ubuntu-x64.tar.gz": f"https://example.com/llama-{tag}-bin-ubuntu-x64.tar.gz"
            },
        )

        _requested_tag, resolved_tag, attempts, _approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert resolved_tag == "b9000"
        assert attempts[0].name == "llama-b9000-bin-ubuntu-x64.tar.gz"
        assert attempts[0].source_label == "upstream"
        assert attempts[0].expected_sha256 == "a" * 64

    def test_linux_cuda_does_not_fall_back_to_upstream_cpu(self, monkeypatch):
        host = make_host(system = "Linux", machine = "x86_64", compute_caps = ["86"])
        release = make_release(
            [], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000"
        )
        checksums = make_checksums_with_source(
            [],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        mock_linux_runtime(monkeypatch, ["cuda12"])

        with pytest.raises(
            PrebuiltFallback, match = "no compatible published Linux CUDA bundle"
        ):
            resolve_install_attempts("latest", host, "unslothai/llama.cpp", "")

    def test_windows_cpu_prefers_published_asset(self, monkeypatch):
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        asset_name = "llama-b9000-bin-win-cpu-x64.zip"
        release = make_release(
            [
                make_artifact(
                    asset_name,
                    install_kind = "windows-cpu",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9000",
            assets = {asset_name: f"https://published.example/{asset_name}"},
        )
        checksums = make_checksums_with_source(
            [asset_name],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: (_ for _ in ()).throw(
                AssertionError("published Windows CPU choice should not query upstream")
            ),
        )

        _requested_tag, resolved_tag, attempts, _approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert resolved_tag == "b9000"
        assert attempts[0].name == asset_name
        assert attempts[0].source_label == "published"

    def test_macos_prefers_published_asset(self, monkeypatch):
        host = make_host(
            system = "Darwin",
            machine = "arm64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        asset_name = "llama-b9000-bin-macos-arm64.tar.gz"
        release = make_release(
            [
                make_artifact(
                    asset_name,
                    install_kind = "macos-arm64",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9000",
            assets = {asset_name: f"https://published.example/{asset_name}"},
        )
        checksums = make_checksums_with_source(
            [asset_name],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: (_ for _ in ()).throw(
                AssertionError("published macOS choice should not query upstream")
            ),
        )

        _requested_tag, resolved_tag, attempts, _approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert resolved_tag == "b9000"
        assert attempts[0].name == asset_name
        assert attempts[0].source_label == "published"

    def test_windows_cpu_missing_checksum_rejects_install(self, monkeypatch):
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        published_name = "llama-b9000-bin-win-cpu-x64.zip"
        release = make_release(
            [
                make_artifact(
                    published_name,
                    install_kind = "windows-cpu",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9000",
            assets = {published_name: f"https://published.example/{published_name}"},
        )
        checksums = make_checksums_with_source(
            [],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: {
                f"llama-{tag}-bin-win-cpu-x64.zip": f"https://upstream.example/llama-{tag}-bin-win-cpu-x64.zip"
            },
        )

        with pytest.raises(
            PrebuiltFallback,
            match = "approved checksum asset did not contain the selected prebuilt archive",
        ):
            resolve_install_attempts(
                "latest",
                host,
                "unslothai/llama.cpp",
                "",
            )


class TestResolveInstallReleasePlans:
    def test_latest_collects_multiple_older_release_plans_up_to_limit(
        self, monkeypatch
    ):
        host = make_host(
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        releases = [
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r3", upstream_tag = "b9003"),
                checksums = make_checksums_with_source(
                    ["llama-b9003-bin-ubuntu-x64.tar.gz"],
                    release_tag = "r3",
                    upstream_tag = "b9003",
                ),
            ),
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r2", upstream_tag = "b9002"),
                checksums = make_checksums_with_source(
                    ["llama-b9002-bin-ubuntu-x64.tar.gz"],
                    release_tag = "r2",
                    upstream_tag = "b9002",
                ),
            ),
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r1", upstream_tag = "b9001"),
                checksums = make_checksums_with_source(
                    ["llama-b9001-bin-ubuntu-x64.tar.gz"],
                    release_tag = "r1",
                    upstream_tag = "b9001",
                ),
            ),
        ]

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                releases
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: {
                f"llama-{tag}-bin-ubuntu-x64.tar.gz": f"https://example.com/llama-{tag}-bin-ubuntu-x64.tar.gz"
            },
        )

        requested_tag, plans = resolve_install_release_plans(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
            max_release_fallbacks = 2,
        )

        assert requested_tag == "latest"
        assert [plan.release_tag for plan in plans] == ["r3", "r2"]
        assert [plan.llama_tag for plan in plans] == ["b9003", "b9002"]

    def test_latest_skips_non_installable_release_and_keeps_searching(
        self, monkeypatch
    ):
        host = make_host(
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        releases = [
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r2", upstream_tag = "b9002"),
                checksums = make_checksums_with_source(
                    [],
                    release_tag = "r2",
                    upstream_tag = "b9002",
                ),
            ),
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r1", upstream_tag = "b9001"),
                checksums = make_checksums_with_source(
                    ["llama-b9001-bin-ubuntu-x64.tar.gz"],
                    release_tag = "r1",
                    upstream_tag = "b9001",
                ),
            ),
        ]

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "": iter(
                releases
            ),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "github_release_assets",
            lambda repo, tag: (
                {}
                if tag == "b9002"
                else {
                    f"llama-{tag}-bin-ubuntu-x64.tar.gz": f"https://example.com/llama-{tag}-bin-ubuntu-x64.tar.gz"
                }
            ),
        )

        _requested_tag, plans = resolve_install_release_plans(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
            max_release_fallbacks = 2,
        )

        assert len(plans) == 1
        assert plans[0].release_tag == "r1"
        assert plans[0].llama_tag == "b9001"

    def test_malformed_release_fallback_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_MAX_PREBUILT_RELEASE_FALLBACKS", "not-an-int")
        assert (
            env_int("UNSLOTH_LLAMA_MAX_PREBUILT_RELEASE_FALLBACKS", 3, minimum = 1) == 3
        )

    def test_import_with_malformed_release_fallback_env_does_not_crash(
        self, monkeypatch
    ):
        monkeypatch.setenv("UNSLOTH_LLAMA_MAX_PREBUILT_RELEASE_FALLBACKS", "bad-value")
        spec = importlib.util.spec_from_file_location(
            "studio_install_llama_prebuilt_env_reload",
            MODULE_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            assert module.DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS == 2
        finally:
            sys.modules.pop(spec.name, None)


# ===========================================================================
# N. windows_cuda_attempts
# ===========================================================================


class TestWindowsCudaAttempts:
    TAG = "b8508"

    def _upstream(self, *runtime_versions, current_names: bool = False):
        assets = {}
        for rv in runtime_versions:
            if current_names:
                name = f"cudart-llama-bin-win-cuda-{rv}-x64.zip"
            else:
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

    def test_driver_below_published_minor_is_gated_to_cuda12(self, monkeypatch):
        # A 13.0 driver cannot run a 13.1 build (forward minor), so it is gated
        # out of cuda13 and falls back to the cuda12 build it can run, even when
        # only the cuda13 runtime libs are detected.
        mock_windows_runtime(monkeypatch, ["cuda13"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 0))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda12"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip"

    def test_driver_at_published_minor_selects_cuda13(self, monkeypatch):
        # A 13.1 driver matches the published 13.1 build exactly.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda13"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.1-x64.zip"

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

    def test_current_upstream_names_are_supported(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4", current_names = True)
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 2
        assert result[0].name == "cudart-llama-bin-win-cuda-13.1-x64.zip"
        assert result[1].name == "cudart-llama-bin-win-cuda-12.4-x64.zip"

    def test_cudart_runtime_archive_is_paired(self, monkeypatch):
        # #5106: cudart bundle must surface on runtime_url so
        # install_from_archives downloads it.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = {
            f"llama-{self.TAG}-bin-win-cuda-13.1-x64.zip": f"https://example.com/llama-{self.TAG}-bin-win-cuda-13.1-x64.zip",
            "cudart-llama-bin-win-cuda-13.1-x64.zip": "https://example.com/cudart-llama-bin-win-cuda-13.1-x64.zip",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip": f"https://example.com/llama-{self.TAG}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip": "https://example.com/cudart-llama-bin-win-cuda-12.4-x64.zip",
        }
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 2
        # cuda13 first (host driver supports 13.1)
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.1-x64.zip"
        assert result[0].runtime_name == "cudart-llama-bin-win-cuda-13.1-x64.zip"
        assert result[0].runtime_url == (
            "https://example.com/cudart-llama-bin-win-cuda-13.1-x64.zip"
        )
        # cuda12 second
        assert result[1].name == f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip"
        assert result[1].runtime_name == "cudart-llama-bin-win-cuda-12.4-x64.zip"

    def test_no_runtime_archive_when_cudart_absent(self, monkeypatch):
        # Older releases without the cudart split must still install.
        mock_windows_runtime(monkeypatch, ["cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (12, 4))
        assets = {
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip": f"https://example.com/llama-{self.TAG}-bin-win-cuda-12.4-x64.zip",
        }
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 1
        assert result[0].runtime_url is None
        assert result[0].runtime_name is None

    def test_cudart_only_assets_do_not_self_pair(self, monkeypatch):
        # Legacy cudart-only naming path must not self-pair.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.1", "12.4", current_names = True)
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert len(result) == 2
        for attempt in result:
            assert attempt.runtime_url is None
            assert attempt.runtime_name is None

    def test_tracks_upstream_cuda13_minor_bump(self, monkeypatch):
        # ggml-org bumped the published Windows cuda13 build 13.1 -> 13.3; the
        # selector must follow it instead of the old hardcoded 13.1 (#5861).
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 3))
        assets = self._upstream("13.3", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda13"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"

    def test_cuda13_minor_bump_pairs_matching_cudart(self, monkeypatch):
        # The paired cudart bundle must track the same bumped minor.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 3))
        assets = {
            f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip": "https://example.com/llama-13.3",
            "cudart-llama-bin-win-cuda-13.3-x64.zip": "https://example.com/cudart-13.3",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip": "https://example.com/llama-12.4",
            "cudart-llama-bin-win-cuda-12.4-x64.zip": "https://example.com/cudart-12.4",
        }
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"
        assert result[0].runtime_name == "cudart-llama-bin-win-cuda-13.3-x64.zip"

    def test_driver_below_published_minor_does_not_get_newer_build(self, monkeypatch):
        # ggml-org ships only cuda-13.3; a 13.1 driver cannot run it (forward
        # minor), so it is gated to the cuda-12.4 build instead of an
        # unguaranteed 13.3. A 13.3 driver still gets 13.3 (see other tests).
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 1))
        assets = self._upstream("13.3", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda12"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip"

    def test_tracks_future_cuda13_minor(self, monkeypatch):
        # A later within-major bump (13.4) is tracked the same as 13.3.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 4))
        assets = self._upstream("13.4", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.4-x64.zip"

    def test_new_cuda_major_selected_when_published(self, monkeypatch):
        # A new CUDA major (14.x) driver picks the published cuda14 build.
        mock_windows_runtime(monkeypatch, ["cuda14", "cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (14, 0))
        assets = self._upstream("14.0", "13.3", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda14"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-14.0-x64.zip"

    def test_new_cuda_major_degrades_to_published_cuda13(self, monkeypatch):
        # A 14.x driver with no cuda14 build runs the newest published cuda13
        # build via backward compatibility.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (14, 0))
        assets = self._upstream("13.3", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"


# ===========================================================================
# N.1b. _pinned_windows_cuda_fallback -- pinned b9360 cuda-13.1 Blackwell fallback
# ===========================================================================


class TestPinnedBlackwellCudaFallback:
    """A Blackwell host on a 13.0/13.1/13.2 driver, gated off the in-release 13.3
    build, gets the pinned immutable b9360 cuda-13.1 GPU build instead of the
    CPU-only cuda-12.4 drop. The pin is dormant for everyone else."""

    TAG = "b8508"

    def _win_host(self, driver, caps):
        return make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = driver,
            compute_caps = caps,
        )

    def test_pin_offered_for_driver_13_1_blackwell(self):
        pin = _pinned_windows_cuda_fallback(self._win_host((13, 1), ["120"]), [])
        assert pin is not None
        assert pin.tag == "b9360"
        assert pin.runtime_line == "cuda13"
        assert pin.name == "llama-b9360-bin-win-cuda-13.1-x64.zip"
        assert pin.runtime_name == "cudart-llama-bin-win-cuda-13.1-x64.zip"
        assert pin.url.endswith("/b9360/llama-b9360-bin-win-cuda-13.1-x64.zip")
        assert pin.runtime_url.endswith("/b9360/cudart-llama-bin-win-cuda-13.1-x64.zip")
        assert pin.install_kind == "windows-cuda"
        assert pin.expected_sha256 and len(pin.expected_sha256) == 64
        assert pin.runtime_sha256 and len(pin.runtime_sha256) == 64

    def test_pin_offered_for_driver_13_2(self):
        assert (
            _pinned_windows_cuda_fallback(self._win_host((13, 2), ["120"]), [])
            is not None
        )

    def test_pin_offered_for_sm121_variant(self):
        # sm_121 is Blackwell-family and also needs toolkit >= 12.8.
        assert (
            _pinned_windows_cuda_fallback(self._win_host((13, 1), ["121"]), [])
            is not None
        )

    def test_pin_uses_max_of_multi_gpu_caps(self):
        assert (
            _pinned_windows_cuda_fallback(self._win_host((13, 1), ["86", "120"]), [])
            is not None
        )

    @pytest.mark.parametrize("sm", ["89", "90", "100"])
    def test_pin_not_offered_to_non_blackwell(self, sm):
        # Ada/Hopper run the cuda-12.4 build fine; the pin must not fire.
        assert _pinned_windows_cuda_fallback(self._win_host((13, 1), [sm]), []) is None

    def test_pin_offered_for_driver_13_0(self):
        # b9360 is native sm_120a SASS (no JIT) and ships a cuda-13.1 cudart,
        # both of which run on a 13.0 r580+ driver via CUDA minor-version
        # compatibility. 13.0 is the mainstream Blackwell branch, so it must fire.
        assert (
            _pinned_windows_cuda_fallback(self._win_host((13, 0), ["120"]), [])
            is not None
        )

    def test_pin_not_offered_below_floor(self):
        # 12.x predates Blackwell entirely; the pin stays dormant below 13.0.
        assert (
            _pinned_windows_cuda_fallback(self._win_host((12, 9), ["120"]), []) is None
        )

    def test_pin_not_offered_without_driver(self):
        assert _pinned_windows_cuda_fallback(self._win_host(None, ["120"]), []) is None

    def test_pin_not_offered_on_linux(self):
        host = make_host(
            system = "Linux",
            machine = "x86_64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        assert _pinned_windows_cuda_fallback(host, []) is None

    def test_pin_dormant_when_cuda13_attempt_present(self, monkeypatch):
        # A runnable in-release cuda13 build makes the pin unnecessary.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = self._win_host((13, 1), ["120"])
        assets = {
            f"llama-{self.TAG}-bin-win-cuda-13.1-x64.zip": "https://example.com/13.1",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip": "https://example.com/12.4",
        }
        existing = windows_cuda_attempts(host, self.TAG, assets, None)
        assert any(a.runtime_line == "cuda13" for a in existing)
        assert _pinned_windows_cuda_fallback(host, existing) is None

    def _win_cuda_attempt(self, minor):
        major = minor.split(".")[0]
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = self.TAG,
            name = f"llama-{self.TAG}-bin-win-cuda-{minor}-x64.zip",
            url = "https://example.com/x",
            source_label = "upstream",
            install_kind = "windows-cuda",
            runtime_line = f"cuda{major}",
        )

    def test_pin_dormant_when_runnable_cuda14_present(self, monkeypatch):
        # A future Blackwell host with an in-release cuda14 build (no cuda13)
        # must not get the older b9360 13.1 pin ahead of the runnable cuda14.
        mock_windows_runtime(monkeypatch, ["cuda14", "cuda12"])
        host = self._win_host((14, 0), ["120"])
        assets = {
            f"llama-{self.TAG}-bin-win-cuda-14.0-x64.zip": "https://example.com/14.0",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip": "https://example.com/12.4",
        }
        existing = windows_cuda_attempts(host, self.TAG, assets, None)
        assert any(a.runtime_line == "cuda14" for a in existing)
        assert _pinned_windows_cuda_fallback(host, existing) is None

    def test_pin_dormant_when_runnable_cuda12_8_present(self):
        # A cuda-12.8 build also covers Blackwell, so the pin defers to it.
        host = self._win_host((13, 1), ["120"])
        existing = [self._win_cuda_attempt("12.8")]
        assert _pinned_windows_cuda_fallback(host, existing) is None

    def test_pin_fires_when_only_cuda12_4_present(self):
        # cuda-12.4 does not cover Blackwell, so the pin still fires.
        host = self._win_host((13, 1), ["120"])
        existing = [self._win_cuda_attempt("12.4")]
        assert _pinned_windows_cuda_fallback(host, existing) is not None

    @pytest.mark.parametrize(
        "minor, covers",
        [
            ("12.4", False),
            ("12.8", True),
            ("13.1", True),
            ("13.3", True),
            ("14.0", True),
        ],
    )
    def test_attempt_covers_blackwell(self, minor, covers):
        assert (
            _windows_cuda_attempt_covers_blackwell(self._win_cuda_attempt(minor))
            is covers
        )

    def test_attempt_covers_blackwell_ignores_non_cuda_kind(self):
        cpu = AssetChoice(
            repo = UPSTREAM_REPO,
            tag = self.TAG,
            name = f"llama-{self.TAG}-bin-win-cpu-x64.zip",
            url = "https://example.com/x",
            source_label = "upstream",
            install_kind = "windows-cpu",
        )
        assert _windows_cuda_attempt_covers_blackwell(cpu) is False

    def _app_attempt(self, profile, runtime_line, max_sm):
        # The fork's app-named windows-cuda bundle: no toolkit minor in the name,
        # SM coverage declared directly (as published_windows_cuda_attempts sets it).
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = self.TAG,
            name = f"app-{self.TAG}-windows-x64-{runtime_line}-{profile}.zip",
            url = "https://example.com/x",
            source_label = "published",
            install_kind = "windows-cuda",
            runtime_line = runtime_line,
            coverage_class = "newer" if profile == "newer" else profile,
            max_sm = max_sm,
            min_sm = 80,
            supported_sms = ["120"] if max_sm >= 120 else ["86", "89"],
        )

    @pytest.mark.parametrize(
        "profile, runtime_line, max_sm, covers",
        [
            ("newer", "cuda13", 120, True),   # native Blackwell build
            ("newer", "cuda12", 120, True),   # 12.8 toolkit app bundle reaches sm120
            ("older", "cuda12", 89, False),   # 12.4 toolkit app bundle stops at Ada
        ],
    )
    def test_attempt_covers_blackwell_app_bundle(
        self, profile, runtime_line, max_sm, covers
    ):
        # App-named bundles carry no toolkit minor; coverage is read from max_sm.
        attempt = self._app_attempt(profile, runtime_line, max_sm)
        assert _windows_cuda_attempt_covers_blackwell(attempt) is covers

    def test_pin_dormant_when_app_bundle_covers_blackwell(self):
        # Regression: the fork's app-named cuda13 bundle covers Blackwell, so the
        # b9360 pin must retire instead of being prepended ahead of the native
        # in-release build (previously the coverage check only matched legacy
        # -bin-win-cuda-X.Y-x64.zip names, so the pin never went dormant).
        host = self._win_host((13, 1), ["120"])
        existing = [self._app_attempt("newer", "cuda13", 120)]
        assert _pinned_windows_cuda_fallback(host, existing) is None


# ===========================================================================
# N.1c. direct_upstream_release_plan -- pinned Blackwell fallback ordering
# ===========================================================================


class TestDirectUpstreamBlackwellPin:
    """End to end: the pin lands ahead of cuda-12.4 on the simple/upstream path
    a Blackwell Windows host actually uses, and stays absent once a runnable
    in-release cuda13 build exists."""

    TAG = "b9365"

    def _release(self):
        names = [
            f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            f"llama-{self.TAG}-bin-win-cpu-x64.zip",
        ]
        return {
            "tag_name": self.TAG,
            "assets": [
                {"name": n, "browser_download_url": f"https://example.com/{n}"}
                for n in names
            ],
        }

    def _no_torch(self, monkeypatch):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "detect_torch_cuda_runtime_preference",
            lambda host: CudaRuntimePreference(runtime_line = None, selection_log = []),
        )

    def test_blackwell_13_1_prepends_pin(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        plan = direct_upstream_release_plan(
            self._release(), host, UPSTREAM_REPO, "latest"
        )
        order = [(a.tag, a.runtime_line or a.install_kind) for a in plan.attempts]
        assert order == [
            ("b9360", "cuda13"),
            (self.TAG, "cuda12"),
            (self.TAG, "windows-cpu"),
        ]
        assert plan.attempts[0].name == "llama-b9360-bin-win-cuda-13.1-x64.zip"
        # Direct/upstream path stays unverified-by-manifest (no approved hashes).
        assert plan.approved_checksums.artifacts == {}

    def test_blackwell_13_3_no_pin(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 3),
            compute_caps = ["120"],
        )
        plan = direct_upstream_release_plan(
            self._release(), host, UPSTREAM_REPO, "latest"
        )
        assert "b9360" not in [a.tag for a in plan.attempts]
        assert plan.attempts[0].tag == self.TAG
        assert plan.attempts[0].runtime_line == "cuda13"
        assert plan.attempts[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"


# ===========================================================================
# N.1d. published_windows_cuda_attempts -- version-dynamic ordering seed
# ===========================================================================


class TestPublishedWindowsCudaAttemptsDynamicMajor:
    """The published-path ordering seed is derived from the release's real
    published minors, so a future CUDA major published here is selectable
    instead of being hidden by a hardcoded cuda12/cuda13 seed."""

    TAG = "b8508"

    def _win_cuda_artifact(self, minor, runtime_line):
        return make_artifact(
            f"llama-{self.TAG}-bin-win-cuda-{minor}-x64.zip",
            install_kind = "windows-cuda",
            runtime_line = runtime_line,
            supported_sms = ["75", "80", "86", "89", "90", "100", "120"],
            max_sm = 120,
        )

    def _release(self, minors_lines):
        artifacts = [self._win_cuda_artifact(m, line) for m, line in minors_lines]
        return make_release(artifacts, upstream_tag = self.TAG)

    def test_future_cuda14_published_is_selected(self, monkeypatch):
        # With the dynamic seed a 14.x driver reaches a published cuda14 build;
        # the old hardcoded cuda12/cuda13 seed would never order it (the cuda14
        # line would be skipped for want of a 14.x asset in the seed).
        mock_windows_runtime(monkeypatch, ["cuda14", "cuda13", "cuda12"])
        release = self._release(
            [("14.0", "cuda14"), ("13.3", "cuda13"), ("12.4", "cuda12")]
        )
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (14, 0),
            compute_caps = ["120"],
        )
        result = published_windows_cuda_attempts(host, release, None)
        assert result[0].runtime_line == "cuda14"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-14.0-x64.zip"

    def test_cuda13_minor_selected_for_13_3_driver(self, monkeypatch):
        # Existing behavior unchanged: a 13.3 driver gets the real 13.3 build.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 3),
            compute_caps = ["120"],
        )
        result = published_windows_cuda_attempts(host, release, None)
        assert result[0].runtime_line == "cuda13"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"

    def test_below_minor_driver_gated_to_cuda12(self, monkeypatch):
        # A 13.1 driver is gated off a published 13.3 and falls to cuda12.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        result = published_windows_cuda_attempts(host, release, None)
        assert result[0].runtime_line == "cuda12"


# ===========================================================================
# N.1e. resolve_release_asset_choice -- pin on the published install path
# ===========================================================================


class TestResolveReleaseAssetChoicePin:
    """The published (non --simple-policy) install path reaches the same b9360
    Blackwell pin as the simple path, with its verified hash threaded."""

    TAG = "b8508"

    def _release(self, minors_lines):
        artifacts = [
            make_artifact(
                f"llama-{self.TAG}-bin-win-cuda-{minor}-x64.zip",
                install_kind = "windows-cuda",
                runtime_line = line,
                supported_sms = ["75", "80", "86", "89", "90", "100", "120"],
                max_sm = 120,
            )
            for minor, line in minors_lines
        ]
        assets = {}
        for minor, _line in minors_lines:
            assets[f"llama-{self.TAG}-bin-win-cuda-{minor}-x64.zip"] = (
                f"https://example.com/llama-{minor}"
            )
            assets[f"cudart-llama-bin-win-cuda-{minor}-x64.zip"] = (
                f"https://example.com/cudart-{minor}"
            )
        return make_release(artifacts, upstream_tag = self.TAG, assets = assets)

    def _checksums(self, minors):
        names = []
        for minor in minors:
            names.append(f"llama-{self.TAG}-bin-win-cuda-{minor}-x64.zip")
            names.append(f"cudart-llama-bin-win-cuda-{minor}-x64.zip")
        return make_checksums(names)

    def _no_torch(self, monkeypatch):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "detect_torch_cuda_runtime_preference",
            lambda host: CudaRuntimePreference(runtime_line = None, selection_log = []),
        )

    def test_pin_applied_on_published_path_for_13_1(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        checksums = self._checksums(["12.4"])  # 13.3 gated off for a 13.1 driver
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        result = resolve_release_asset_choice(host, self.TAG, release, checksums)
        assert result[0].tag == "b9360"
        assert result[0].name == "llama-b9360-bin-win-cuda-13.1-x64.zip"
        # apply_approved_hashes threaded the pin's verified hash from the
        # augmented checksums (the pin survives the approved-hash gate).
        assert result[0].expected_sha256 and len(result[0].expected_sha256) == 64
        assert result[0].runtime_sha256 and len(result[0].runtime_sha256) == 64
        assert any(a.runtime_line == "cuda12" for a in result)

    def test_pin_dormant_on_published_path_for_13_3(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        checksums = self._checksums(["13.3", "12.4"])
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 3),
            compute_caps = ["120"],
        )
        result = resolve_release_asset_choice(host, self.TAG, release, checksums)
        assert "b9360" not in [a.tag for a in result]
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"

    def test_pin_not_applied_for_non_blackwell(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        checksums = self._checksums(["12.4"])
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["89"],
        )
        result = resolve_release_asset_choice(host, self.TAG, release, checksums)
        assert "b9360" not in [a.tag for a in result]


class TestPublishedWindowsCudaAppBundleSmSelection:
    """app-named windows-cuda bundles carry no minor in the filename, so the
    driver-minor gate is skipped. Selection must instead filter by SM coverage,
    or every host gets the lowest-rank "older" bundle regardless of its GPU."""

    TAG = "b9457"

    def _app(self, klass, supported, min_sm, max_sm, rank):
        return make_artifact(
            f"app-{self.TAG}-windows-x64-cuda12-{klass}.zip",
            install_kind = "windows-cuda",
            runtime_line = "cuda12",
            coverage_class = klass,
            supported_sms = supported,
            min_sm = min_sm,
            max_sm = max_sm,
            bundle_profile = f"cuda12-{klass}",
            rank = rank,
        )

    def test_blackwell_sm120_skips_older_bundle(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda12"])
        older = self._app("older", ["70", "75", "80", "86", "89"], 70, 89, 10)
        newer = self._app("newer", ["86", "89", "90", "100", "120"], 86, 120, 20)
        portable = self._app(
            "portable", ["70", "75", "80", "86", "89", "90", "100", "120"], 70, 120, 30
        )
        release = make_release([older, newer, portable], upstream_tag = self.TAG)
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (12, 8),
            compute_caps = ["120"],
        )
        result = published_windows_cuda_attempts(host, release, None)
        assert result, "expected a windows-cuda attempt for an sm120 host"
        # The lowest-rank "older" bundle (max_sm 89) must not be chosen, and the
        # tightest covering bundle is cuda12-newer (range 86-120).
        assert result[0].name == f"app-{self.TAG}-windows-x64-cuda12-newer.zip"

    def _line(self, line, klass, rank):
        return make_artifact(
            f"app-{self.TAG}-windows-x64-{line}-{klass}.zip",
            install_kind = "windows-cuda",
            runtime_line = line,
            coverage_class = klass,
            supported_sms = ["86", "89", "90", "100", "120"],
            min_sm = 86,
            max_sm = 120,
            bundle_profile = f"{line}-{klass}",
            rank = rank,
        )

    def test_cuda13_reachable_on_driver_13_0(self, monkeypatch):
        # app-named cuda13 bundles must be reachable on a 13.0 driver. The old
        # synthetic '13.1' minor gate dropped the whole cuda13 line (13.1 > 13.0),
        # so a cu13 host fell to cuda12. cuda13 is gated at the major level now.
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        release = make_release(
            [self._line("cuda12", "newer", 20), self._line("cuda13", "newer", 50)],
            upstream_tag = self.TAG,
        )
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 0),
            compute_caps = ["120"],
        )
        result = published_windows_cuda_attempts(host, release, "cuda13")
        assert result
        assert result[0].runtime_line == "cuda13"
        assert result[0].name == f"app-{self.TAG}-windows-x64-cuda13-newer.zip"


class TestPublishedRocmGfxSelection:
    """Published ROCm bundles are matched by the host's detected gfx family, not
    by rank -- rank ties would alphabetically hand every AMD GPU the gfx103X
    bundle (e.g. a gfx1151 Strix Halo host)."""

    GFX = ["gfx103X", "gfx110X", "gfx120X", "gfx1150", "gfx1151"]
    MEMBERS = {
        "gfx103X": ["gfx1030", "gfx1031", "gfx1032", "gfx1034"],
        "gfx110X": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
        "gfx120X": ["gfx1200", "gfx1201"],
        "gfx1150": ["gfx1150"],
        "gfx1151": ["gfx1151"],
    }

    def _release(self, install_kind, prefix):
        artifacts = [
            make_artifact(
                f"{prefix}-{gfx}.{'zip' if 'windows' in install_kind else 'tar.gz'}",
                install_kind = install_kind,
                runtime_line = None,
                coverage_class = None,
                supported_sms = [],
                min_sm = None,
                max_sm = None,
                bundle_profile = None,
                rank = 1000,
                gfx_target = gfx,
                mapped_targets = self.MEMBERS[gfx],
            )
            for gfx in self.GFX
        ]
        return make_release(artifacts, upstream_tag = "b9457")

    def _host(self, gfx):
        return make_host(
            machine = "x86_64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
            has_rocm = True,
            rocm_gfx_target = gfx,
        )

    def test_gfx1100_selects_gfx110X_family(self):
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        choice = INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
            release, self._host("gfx1100"), "linux-rocm"
        )
        assert choice is not None
        assert choice.name == "app-b9457-linux-x64-rocm-gfx110X.tar.gz"

    def test_gfx1151_strix_halo_not_handed_gfx103X(self):
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        choice = INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
            release, self._host("gfx1151"), "linux-rocm"
        )
        assert choice is not None
        assert choice.name == "app-b9457-linux-x64-rocm-gfx1151.tar.gz"

    def test_windows_rocm_gfx_match(self):
        release = self._release("windows-rocm", "app-b9457-windows-x64-rocm")
        choice = INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
            release, self._host("gfx1201"), "windows-rocm"
        )
        assert choice is not None
        assert choice.name == "app-b9457-windows-x64-rocm-gfx120X.zip"

    def test_uncovered_gpu_returns_none(self):
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        assert (
            INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
                release, self._host("gfx900"), "linux-rocm"
            )
            is None
        )

    def test_in_prefix_but_unbuilt_arch_returns_none(self):
        # gfx1033 shares the gfx103 prefix but is not in any bundle's
        # mapped_targets, so it must fall back to source, not be served gfx103X.
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        for unbuilt in ("gfx1033", "gfx1035", "gfx1104", "gfx1202"):
            assert (
                INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
                    release, self._host(unbuilt), "linux-rocm"
                )
                is None
            ), unbuilt


class TestPublishedMacosForkSelection:
    """macOS now routes to the fork (setup.sh), which ships
    llama-<tag>-bin-macos-<arch>.tar.gz with pinned deployment targets, selected
    by install_kind."""

    def _release(self):
        arts = [
            make_artifact(
                "llama-b9457-bin-macos-arm64.tar.gz",
                install_kind = "macos-arm64",
                runtime_line = None,
                coverage_class = None,
                supported_sms = [],
                min_sm = None,
                max_sm = None,
                bundle_profile = "macos-metal-arm64",
                rank = 50,
            ),
            make_artifact(
                "llama-b9457-bin-macos-x64.tar.gz",
                install_kind = "macos-x64",
                runtime_line = None,
                coverage_class = None,
                supported_sms = [],
                min_sm = None,
                max_sm = None,
                bundle_profile = "macos-cpu-x64",
                rank = 50,
            ),
        ]
        return make_release(arts, upstream_tag = "b9457")

    def test_macos_arm64_selects_fork_bundle(self):
        choice = INSTALL_LLAMA_PREBUILT.published_asset_choice_for_kind(
            self._release(), "macos-arm64"
        )
        assert choice is not None
        assert choice.name == "llama-b9457-bin-macos-arm64.tar.gz"
        assert choice.install_kind == "macos-arm64"

    def test_macos_x64_selects_fork_bundle(self):
        choice = INSTALL_LLAMA_PREBUILT.published_asset_choice_for_kind(
            self._release(), "macos-x64"
        )
        assert choice is not None
        assert choice.name == "llama-b9457-bin-macos-x64.tar.gz"


# ===========================================================================
# N.1. apply_approved_hashes -- runtime archive checksum threading
# ===========================================================================


class TestApplyApprovedHashesRuntimePair:
    """Runtime archive must inherit a manifest hash, or be dropped."""

    TAG = "b8508"

    def _runtime_paired_attempt(self) -> AssetChoice:
        return AssetChoice(
            repo = "unslothai/llama.cpp",
            tag = self.TAG,
            name = f"llama-{self.TAG}-bin-win-cuda-13.1-x64.zip",
            url = f"https://x/llama-{self.TAG}-bin-win-cuda-13.1-x64.zip",
            source_label = "published",
            install_kind = "windows-cuda",
            runtime_line = "cuda13",
            runtime_name = "cudart-llama-bin-win-cuda-13.1-x64.zip",
            runtime_url = "https://x/cudart-llama-bin-win-cuda-13.1-x64.zip",
        )

    def test_runtime_hash_threaded_when_present(self):
        attempt = self._runtime_paired_attempt()
        checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = self.TAG,
            upstream_tag = self.TAG,
            artifacts = {
                attempt.name: ApprovedArtifactHash(
                    asset_name = attempt.name,
                    sha256 = "0" * 64,
                    repo = "unslothai/llama.cpp",
                    kind = "windows-cuda",
                ),
                "cudart-llama-bin-win-cuda-13.1-x64.zip": ApprovedArtifactHash(
                    asset_name = "cudart-llama-bin-win-cuda-13.1-x64.zip",
                    sha256 = "1" * 64,
                    repo = "unslothai/llama.cpp",
                    kind = "windows-cuda",
                ),
            },
        )
        result = apply_approved_hashes([attempt], checksums)
        assert len(result) == 1
        assert result[0].expected_sha256 == "0" * 64
        assert result[0].runtime_sha256 == "1" * 64
        assert result[0].runtime_name == "cudart-llama-bin-win-cuda-13.1-x64.zip"

    def test_runtime_pair_dropped_when_hash_missing(self):
        # Drop the pair rather than install an unverified runtime.
        attempt = self._runtime_paired_attempt()
        checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = self.TAG,
            upstream_tag = self.TAG,
            artifacts = {
                attempt.name: ApprovedArtifactHash(
                    asset_name = attempt.name,
                    sha256 = "0" * 64,
                    repo = "unslothai/llama.cpp",
                    kind = "windows-cuda",
                ),
            },
        )
        result = apply_approved_hashes([attempt], checksums)
        assert len(result) == 1
        assert result[0].expected_sha256 == "0" * 64
        assert result[0].runtime_url is None
        assert result[0].runtime_name is None
        assert result[0].runtime_sha256 is None


# ===========================================================================
# O. resolve_upstream_asset_choice -- platform routing
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


# ===========================================================================
# N.2. Deterministic macOS prebuilt pin (b9415)
# ===========================================================================


def _macos_host(machine = "arm64", version = (15, 5)):
    return make_host(
        system = "Darwin",
        machine = machine,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        macos_version = version,
    )


class TestPinnedMacosReleaseTag:
    """pinned_macos_release_tag: pin b9415 only for ggml-org upstream macOS hosts
    below macOS 26; latest (None) for 26+, unknown version, the fork, non-macOS."""

    def test_arm64_sequoia_pins_b9415(self):
        host = _macos_host("arm64", (15, 5))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_arm64_sonoma_pins_b9415(self):
        host = _macos_host("arm64", (14, 7))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_x64_ventura_13_3_pins_b9415(self):
        # b9415's Intel slice is minos 13.3, so 13.3 Intel hosts still load it.
        host = _macos_host("x86_64", (13, 3))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_tahoe_26_0_takes_latest(self):
        host = _macos_host("arm64", (26, 0))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None

    def test_tahoe_26_1_takes_latest(self):
        host = _macos_host("arm64", (26, 1))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None

    def test_unknown_version_takes_latest(self):
        host = _macos_host("arm64", None)
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None

    def test_fork_repo_is_dormant(self):
        # The unslothai/llama.cpp fork publishes its own minos-13.3 prebuilts.
        host = _macos_host("arm64", (15, 5))
        fork = INSTALL_LLAMA_PREBUILT.DEFAULT_PUBLISHED_REPO
        assert pinned_macos_release_tag(host, fork) is None

    def test_non_macos_host_is_dormant(self):
        host = make_host(system = "Linux", machine = "x86_64")
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None


class TestResolveSimpleMacosPin:
    """End to end on the simple/upstream path macOS actually uses: a pre-26 host
    deterministically resolves b9415 (no walk-back); a macOS 26 host takes the
    latest release. Mirrors how setup.sh routes Darwin to ggml-org/llama.cpp."""

    TAGS = ["b9442", "b9430", "b9428", "b9415"]  # newest-first feed

    def _feed(self, monkeypatch):
        calls = []

        def _release(tag):
            name = f"llama-{tag}-bin-macos-arm64.tar.gz"
            return {
                "tag_name": tag,
                "assets": [
                    {
                        "name": name,
                        "browser_download_url": f"https://example.com/{name}",
                    }
                ],
            }

        def fake_iter(repo, published_release_tag = "", requested_tag = ""):
            calls.append((repo, published_release_tag, requested_tag))
            # Emulate the real iterator: a specific tag yields only that release.
            if requested_tag and requested_tag != "latest":
                yield _release(requested_tag)
                return
            for tag in self.TAGS:
                yield _release(tag)

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT, "iter_release_payloads_by_time", fake_iter
        )
        return calls

    def test_pre26_host_pins_b9415_without_walkback(self, monkeypatch):
        calls = self._feed(monkeypatch)
        host = _macos_host("arm64", (15, 5))
        requested_tag, plans = resolve_simple_install_release_plans(
            "latest", host, "ggml-org/llama.cpp", ""
        )
        assert requested_tag == "b9415"
        assert len(plans) == 1
        assert plans[0].release_tag == "b9415"
        assert plans[0].llama_tag == "b9415"
        assert plans[0].attempts[0].install_kind == "macos-arm64"
        assert plans[0].attempts[0].name == "llama-b9415-bin-macos-arm64.tar.gz"
        # The pin overrode the requested tag before any release was fetched.
        assert calls[0][2] == "b9415"
        # Simple/upstream path stays unverified-by-manifest, exactly as before.
        assert plans[0].approved_checksums.artifacts == {}

    def test_tahoe_host_takes_latest_release(self, monkeypatch):
        calls = self._feed(monkeypatch)
        host = _macos_host("arm64", (26, 0))
        requested_tag, plans = resolve_simple_install_release_plans(
            "latest", host, "ggml-org/llama.cpp", ""
        )
        assert requested_tag == "latest"
        assert plans[0].release_tag == "b9442"
        # No pin: the iterator was asked for latest, not a specific tag.
        assert calls[0][2] == "latest"


# ===========================================================================
# Linux arm64 + GPU must not install the x64-only fork bundle
# ===========================================================================


class TestLinuxArm64ForkFallsBackToSource:
    """The fork now ships linux-arm64-cuda bundles (GH200/GB200/DGX Spark). An
    arm64 Linux host on the fork no longer hard-fails on the simple path; it
    delegates to the manifest-aware resolver, which selects the arm64 CUDA
    bundle (or falls back to source only if none matches)."""

    def test_arm64_nvidia_fork_delegates_to_manifest_resolver(self, monkeypatch):
        # arm64 fork hosts are no longer blocked up front; the simple resolver
        # hands them to the manifest-aware resolver instead.
        called = {}

        def _full(llama_tag, host, repo, tag, **_kw):
            called["args"] = (host.machine, repo)
            return "b9457", ["plan"]

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT, "resolve_install_release_plans", _full
        )
        host = make_host(system = "Linux", machine = "aarch64")
        tag, plans = resolve_simple_install_release_plans(
            "latest", host, "unslothai/llama.cpp", ""
        )
        assert called.get("args") == ("aarch64", "unslothai/llama.cpp")
        assert plans == ["plan"]

    def test_x86_64_fork_is_not_blocked_by_the_arch_guard(self, monkeypatch):
        # x64 host must pass the guard and reach the iterator (here empty, so it
        # raises the generic message, not the arch one).
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_release_payloads_by_time",
            lambda *_a, **_k: iter(()),
        )
        host = make_host(system = "Linux", machine = "x86_64")
        with pytest.raises(PrebuiltFallback) as exc:
            resolve_simple_install_release_plans(
                "latest", host, "unslothai/llama.cpp", ""
            )
        assert "linux-x64 prebuilts" not in str(exc.value)

    def test_arm64_cpu_on_ggml_org_is_not_blocked(self, monkeypatch):
        # CPU-only arm64 routes to ggml-org (not the fork), so the guard must not
        # fire; it reaches the iterator (empty here -> generic message).
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_release_payloads_by_time",
            lambda *_a, **_k: iter(()),
        )
        host = make_host(
            system = "Linux",
            machine = "aarch64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        with pytest.raises(PrebuiltFallback) as exc:
            resolve_simple_install_release_plans(
                "latest", host, "ggml-org/llama.cpp", ""
            )
        assert "linux-x64 prebuilts" not in str(exc.value)


# ===========================================================================
# arm64 Linux GPU: CPU prebuilt fallback after a failed source build (--cpu-fallback)
# ===========================================================================


class TestCpuFallback:
    """--cpu-fallback drops GPU attributes so the CPU prebuilt for the host's
    OS/arch is selected, letting an arm64 GPU host install ggml-org's arm64 CPU
    build as a last resort when its source build produced no binary."""

    _SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"

    def _arm64_nvidia(self):
        return make_host(
            system = "Linux",
            machine = "aarch64",
            driver_cuda_version = (13, 0),
            compute_caps = ["90"],
            has_physical_nvidia = True,
            has_usable_nvidia = True,
        )

    def test_force_cpu_drops_gpu_attrs_before_planning(self, monkeypatch, tmp_path):
        captured = {}

        def _capture(llama_tag, host, *a, **k):
            captured["host"] = host
            raise PrebuiltFallback("stop after capture")

        monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", self._arm64_nvidia)
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_simple_install_release_plans",
            _capture,
        )
        # install_prebuilt exits EXIT_FALLBACK on PrebuiltFallback; we only care
        # about the host it handed to the resolver before that.
        with pytest.raises(SystemExit):
            INSTALL_LLAMA_PREBUILT.install_prebuilt(
                install_dir = tmp_path / "llama",
                llama_tag = "latest",
                published_repo = "ggml-org/llama.cpp",
                published_release_tag = "",
                simple_policy = True,
                force_cpu = True,
            )
        host = captured["host"]
        assert host.has_usable_nvidia is False
        assert host.has_physical_nvidia is False
        assert host.has_rocm is False
        # Arch is preserved so the arm64 CPU bundle (not x64) is chosen.
        assert host.is_arm64 is True

    def test_cpu_forced_arm64_selects_ubuntu_arm64(self):
        tag = "b9444"
        release = {
            "tag_name": tag,
            "assets": [
                {
                    "name": f"llama-{tag}-bin-ubuntu-arm64.tar.gz",
                    "browser_download_url": f"https://x/llama-{tag}-bin-ubuntu-arm64.tar.gz",
                },
                {
                    "name": f"llama-{tag}-bin-ubuntu-x64.tar.gz",
                    "browser_download_url": f"https://x/llama-{tag}-bin-ubuntu-x64.tar.gz",
                },
            ],
        }
        # A GPU arm64 host cannot pick the CPU arm64 bundle on its own.
        with pytest.raises(PrebuiltFallback):
            direct_upstream_release_plan(
                release, self._arm64_nvidia(), "ggml-org/llama.cpp", "latest"
            )
        # force_cpu drops the GPU attributes, so the CPU arm64 bundle is selected.
        cpu_host = make_host(
            system = "Linux",
            machine = "aarch64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        plan = direct_upstream_release_plan(
            release, cpu_host, "ggml-org/llama.cpp", "latest"
        )
        assert plan.attempts[0].install_kind == "linux-arm64"
        assert plan.attempts[0].name == f"llama-{tag}-bin-ubuntu-arm64.tar.gz"

    def test_setup_sh_has_arm64_cpu_prebuilt_fallback(self):
        source = self._SETUP_SH.read_text(encoding = "utf-8")
        assert "--cpu-fallback" in source
        # Fallback targets ggml-org (the only repo with an arm64 Linux build) and
        # is gated on a degraded source build for arm64.
        assert "ggml-org/llama.cpp" in source
        assert "_LLAMA_CPP_DEGRADED" in source
