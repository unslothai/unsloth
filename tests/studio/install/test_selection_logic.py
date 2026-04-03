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
