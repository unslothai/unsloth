"""Binary selection logic in install_llama_prebuilt.py; all I/O monkeypatched."""

import importlib.util
import os
import socket
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
RUN_MODULE_PATH = PACKAGE_ROOT / "studio" / "backend" / "run.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
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

pick_windows_cuda_runtime = INSTALL_LLAMA_PREBUILT.pick_windows_cuda_runtime
compatible_windows_runtime_lines = INSTALL_LLAMA_PREBUILT.compatible_windows_runtime_lines
apply_approved_hashes = INSTALL_LLAMA_PREBUILT.apply_approved_hashes
linux_cuda_choice_from_release = INSTALL_LLAMA_PREBUILT.linux_cuda_choice_from_release
windows_cuda_attempts = INSTALL_LLAMA_PREBUILT.windows_cuda_attempts
resolve_upstream_asset_choice = INSTALL_LLAMA_PREBUILT.resolve_upstream_asset_choice
resolve_requested_install_tag = INSTALL_LLAMA_PREBUILT.resolve_requested_install_tag
resolve_install_attempts = INSTALL_LLAMA_PREBUILT.resolve_install_attempts
_fork_manifest_release_plans = INSTALL_LLAMA_PREBUILT._fork_manifest_release_plans
resolve_published_release = INSTALL_LLAMA_PREBUILT.resolve_published_release
resolve_source_build_plan = INSTALL_LLAMA_PREBUILT.resolve_source_build_plan
validated_checksums_for_bundle = INSTALL_LLAMA_PREBUILT.validated_checksums_for_bundle
parse_approved_release_checksums = INSTALL_LLAMA_PREBUILT.parse_approved_release_checksums
published_release_matches_request = INSTALL_LLAMA_PREBUILT.published_release_matches_request
exact_source_archive_logical_name = INSTALL_LLAMA_PREBUILT.exact_source_archive_logical_name
source_archive_logical_name = INSTALL_LLAMA_PREBUILT.source_archive_logical_name
windows_cuda_upstream_asset_names = INSTALL_LLAMA_PREBUILT.windows_cuda_upstream_asset_names
env_int = INSTALL_LLAMA_PREBUILT.env_int
direct_upstream_release_plan = INSTALL_LLAMA_PREBUILT.direct_upstream_release_plan
CudaRuntimePreference = INSTALL_LLAMA_PREBUILT.CudaRuntimePreference
published_windows_cuda_attempts = INSTALL_LLAMA_PREBUILT.published_windows_cuda_attempts
_windows_cuda_attempt_covers_blackwell = (
    INSTALL_LLAMA_PREBUILT._windows_cuda_attempt_covers_blackwell
)
resolve_release_asset_choice = INSTALL_LLAMA_PREBUILT.resolve_release_asset_choice
pinned_macos_release_tag = INSTALL_LLAMA_PREBUILT.pinned_macos_release_tag
resolve_simple_install_release_plans = INSTALL_LLAMA_PREBUILT.resolve_simple_install_release_plans


@pytest.fixture(autouse = True)
def _disable_download_host_fast_path(monkeypatch):
    # This module exercises the GitHub API enumeration and asset selection against
    # mocked releases; keep the download-host fast path (real CDN) out of the way.
    # test_download_host_resolve.py covers the fast path itself.
    monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")


def load_studio_run_module(monkeypatch):
    logger = types.SimpleNamespace(
        debug = lambda *a, **k: None,
        info = lambda *a, **k: None,
        warning = lambda *a, **k: None,
    )
    loggers = types.ModuleType("loggers")
    loggers.get_logger = lambda name: logger
    monkeypatch.setitem(sys.modules, "loggers", loggers)

    startup_banner = types.ModuleType("startup_banner")
    startup_banner.print_studio_access_banner = lambda **k: None
    startup_banner.print_studio_stop_hint = lambda: None
    startup_banner.stdout_supports_color = lambda: False
    monkeypatch.setitem(sys.modules, "startup_banner", startup_banner)

    paths = types.ModuleType("utils.paths")
    paths.__path__ = []
    storage_roots = types.ModuleType("utils.paths.storage_roots")
    storage_roots.studio_root = lambda: PACKAGE_ROOT / ".studio-test-root"
    paths.storage_roots = storage_roots
    monkeypatch.setitem(sys.modules, "utils.paths", paths)
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage_roots)
    monkeypatch.syspath_prepend(str(RUN_MODULE_PATH.parent))

    spec = importlib.util.spec_from_file_location(
        "studio_backend_run_warning_test", RUN_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Helper factories


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
    normalized_source_commit = source_commit.lower() if isinstance(source_commit, str) else None
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
        source_commit_short = normalized_source_commit[:7] if normalized_source_commit else None,
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
# Unsloth run.py localhost warning
# ===========================================================================


class TestStudioLocalhostIpv6Warning:
    def _prepare_loopback(self, run_module, monkeypatch):
        # Unsloth confirmed answering on the IPv4 loopback.
        monkeypatch.setattr(
            run_module,
            "_working_local_url",
            lambda port: f"http://127.0.0.1:{port}",
        )

    def _set_getaddrinfo(self, monkeypatch, entries):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: entries)

    @staticmethod
    def _ipv4(port = 8888):
        return (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))

    @staticmethod
    def _ipv6(port = 8888):
        return (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0))

    # -- _localhost_ipv6_mismatch_url --

    def test_ipv4_localhost_does_not_warn(self, monkeypatch):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [self._ipv4()])

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) is None

    def test_dual_stack_localhost_does_not_warn(self, monkeypatch):
        # Dual-stack localhost: browsers fall back to IPv4 when ::1 refuses, so no warning.
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [self._ipv6(), self._ipv4()])

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) is None

    def test_ipv6_only_localhost_returns_ipv4_url(self, monkeypatch, capsys):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [self._ipv6()])
        monkeypatch.setattr(run_module, "_stdout_color_ok", lambda: False)

        local_url = run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888)
        assert local_url == "http://127.0.0.1:8888"

        run_module._print_localhost_ipv6_mismatch_warning(local_url, 8888)
        captured = capsys.readouterr()
        assert "localhost resolves to IPv6 (::1)" in captured.out
        assert "http://127.0.0.1:8888" in captured.out
        assert "http://localhost:8888" in captured.out

    def test_ipv6_listener_does_not_suppress_warning(self, monkeypatch):
        # A process on ::1 is NOT Unsloth (binds 127.0.0.1 only), so the warning must
        # still fire -- that is exactly when http://localhost opens the wrong service.
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [self._ipv6()])
        monkeypatch.setattr(
            run_module,
            "_local_port_open",
            lambda host, port, timeout = 1.0: True,
        )

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) == "http://127.0.0.1:8888"

    @pytest.mark.parametrize("host", ["0.0.0.0", "::"])
    def test_network_bind_suppresses_warning(self, monkeypatch, host):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [self._ipv6()])

        assert run_module._localhost_ipv6_mismatch_url(host, 8888) is None

    @pytest.mark.parametrize("port", [0, -1])
    def test_non_positive_port_returns_none(self, monkeypatch, port):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", port) is None

    def test_ipv4_not_answering_suppresses_warning(self, monkeypatch):
        # Unsloth not confirmed on 127.0.0.1 -> no warning.
        run_module = load_studio_run_module(monkeypatch)
        monkeypatch.setattr(run_module, "_working_local_url", lambda port: None)
        self._set_getaddrinfo(monkeypatch, [self._ipv6()])

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) is None

    def test_empty_resolver_result_suppresses_warning(self, monkeypatch):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)
        self._set_getaddrinfo(monkeypatch, [])

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) is None

    def test_resolver_error_suppresses_warning(self, monkeypatch):
        run_module = load_studio_run_module(monkeypatch)
        self._prepare_loopback(run_module, monkeypatch)

        def _raise(*_a, **_k):
            raise socket.gaierror("resolver unavailable")

        monkeypatch.setattr(socket, "getaddrinfo", _raise)

        assert run_module._localhost_ipv6_mismatch_url("127.0.0.1", 8888) is None

    # -- _emit_startup_output (banner / warning wiring) --

    def _wire_recorders(self, run_module, monkeypatch):
        calls = {"banner": [], "warning": [], "stop_hint": 0, "reachability": []}
        monkeypatch.setattr(
            run_module,
            "print_studio_access_banner",
            lambda **kwargs: calls["banner"].append(kwargs),
        )
        monkeypatch.setattr(
            run_module,
            "_print_localhost_ipv6_mismatch_warning",
            lambda local_url, port: calls["warning"].append((local_url, port)),
        )
        monkeypatch.setattr(
            run_module,
            "print_studio_stop_hint",
            lambda: calls.__setitem__("stop_hint", calls["stop_hint"] + 1),
        )
        monkeypatch.setattr(
            run_module,
            "_verify_global_reachability",
            lambda display_host, port: calls["reachability"].append((display_host, port)),
        )
        return calls

    def test_emit_startup_output_wires_mismatch_warning(self, monkeypatch):
        run_module = load_studio_run_module(monkeypatch)
        calls = self._wire_recorders(run_module, monkeypatch)
        monkeypatch.setattr(
            run_module,
            "_localhost_ipv6_mismatch_url",
            lambda host, port: "http://127.0.0.1:8888",
        )

        run_module._emit_startup_output("127.0.0.1", 8888, "127.0.0.1")

        assert calls["banner"][0]["include_stop_hint"] is False
        assert calls["warning"] == [("http://127.0.0.1:8888", 8888)]
        assert calls["stop_hint"] == 1
        assert calls["reachability"] == []

    def test_emit_startup_output_plain_localhost(self, monkeypatch):
        run_module = load_studio_run_module(monkeypatch)
        calls = self._wire_recorders(run_module, monkeypatch)
        monkeypatch.setattr(run_module, "_localhost_ipv6_mismatch_url", lambda host, port: None)

        run_module._emit_startup_output("127.0.0.1", 8888, "127.0.0.1")

        # The stop hint no longer rides inside the banner: it is printed once at the
        # end so it stays the final line after the tool-policy notice.
        assert calls["banner"][0]["include_stop_hint"] is False
        assert calls["warning"] == []
        assert calls["stop_hint"] == 1
        assert calls["reachability"] == []

    @pytest.mark.parametrize("host", ["0.0.0.0", "::"])
    def test_emit_startup_output_wildcard_runs_reachability(self, monkeypatch, host):
        run_module = load_studio_run_module(monkeypatch)
        calls = self._wire_recorders(run_module, monkeypatch)
        monkeypatch.setattr(run_module, "_localhost_ipv6_mismatch_url", lambda h, port: None)

        run_module._emit_startup_output(host, 8888, "203.0.113.5")

        assert calls["banner"][0]["include_stop_hint"] is False
        assert calls["warning"] == []
        assert calls["reachability"] == [("203.0.113.5", 8888)]
        assert calls["stop_hint"] == 1


# ===========================================================================
# A-F, H. Component-independent GPU/token helpers: the behavior tables live in
# tests/studio/install/test_prebuilt_core.py (they are pure prebuilt_core
# functions). This pin proves the installer still re-exports them from core, so
# the master tables keep covering the names this module and its callers use.
# ===========================================================================


def test_core_helper_aliases_bound_to_prebuilt_core():
    import prebuilt_core as _core
    for name in (
        "normalize_compute_cap",
        "normalize_compute_caps",
        "parse_cuda_visible_devices",
        "supports_explicit_visible_device_matching",
        "select_visible_gpu_rows",
        "compatible_linux_runtime_lines",
        "runtime_line_from_cuda_version",
    ):
        assert getattr(INSTALL_LLAMA_PREBUILT, name) is getattr(_core, name), name


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

    @pytest.mark.parametrize("minor", [0, 1, 2, 3])
    def test_cuda12_runs_on_any_12_x_driver(self, minor):
        # Regression: Windows previously gated cuda12 below a 12.4 driver, but minor-version
        # compat runs toolkit-12.8 bundles on any 12.x driver, same as Linux.
        host = make_host(driver_cuda_version = (12, minor))
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
            return make_checksums_with_source([], release_tag = "v1.0", upstream_tag = "b8999")

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

        assert resolve_requested_install_tag("b8508", "", "unslothai/llama.cpp") == "b8508"

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
        bundle = make_release([], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000")
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
                source_repo = "example/custom-llama.cpp",
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

    def test_direct_main_request_without_published_release_uses_branch_kind(self, monkeypatch):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "resolve_published_release",
            lambda requested_tag, published_repo, published_release_tag = "": (_ for _ in ()).throw(
                PrebuiltFallback("missing")
            ),
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
        checksums = make_checksums_with_source([], release_tag = "r1", upstream_tag = "b8508")
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

    def test_rejects_exact_source_without_repo(self, monkeypatch):
        # An exact source archive with no repo to clone from would silently fall back
        # to upstream source at the tag, so validation must fail closed.
        bundle = make_release([], release_tag = "r1", upstream_tag = "b8508")
        checksums = make_checksums_with_source(
            [], release_tag = "r1", upstream_tag = "b8508", source_commit = "a" * 40
        )  # exact source archive, but no source_repo
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: checksums,
        )

        with pytest.raises(PrebuiltFallback, match = "exact source archive"):
            validated_checksums_for_bundle("unslothai/llama.cpp", bundle)

    def test_accepts_exact_source_when_only_bundle_has_repo(self, monkeypatch):
        # The source repo can live only in the manifest bundle, not the checksums;
        # validation must accept the bundle's repo rather than failing closed.
        bundle = make_release(
            [],
            release_tag = "r1",
            upstream_tag = "b8508",
            source_repo = "ggml-org/llama.cpp",
            source_repo_url = "https://github.com/ggml-org/llama.cpp",
        )
        checksums = make_checksums_with_source(
            [], release_tag = "r1", upstream_tag = "b8508", source_commit = "a" * 40
        )  # exact source archive, repo only on the bundle
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "load_approved_release_checksums",
            lambda repo, release_tag: checksums,
        )

        assert validated_checksums_for_bundle("unslothai/llama.cpp", bundle) is checksums
        plan = INSTALL_LLAMA_PREBUILT.source_build_plan_for_release(
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(bundle = bundle, checksums = checksums)
        )
        assert plan.source_url == "https://github.com/ggml-org/llama.cpp"
        assert plan.source_ref_kind == "commit"
        assert plan.source_ref == "a" * 40


# ===========================================================================
# K. linux_cuda_choice_from_release -- core selection
# ===========================================================================


class TestLinuxCudaChoiceFromRelease:
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
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda12"

    def test_preferred_runtime_line_unavailable(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(driver_cuda_version = (12, 8))
        art = make_artifact("bundle-cuda12.tar.gz", runtime_line = "cuda12")
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda13")
        assert result is not None
        assert result.primary.runtime_line == "cuda12"
        log_entries = result.selection_log
        assert any("unavailable_on_host" in entry for entry in log_entries)

    def test_blackwell_prefers_cuda13_over_torch_cuda12(self, monkeypatch):
        # Both lines sm_120-capable: cuda13 wins over torch's cuda12.
        mock_linux_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (13, 0), compute_caps = ["120"])
        art12 = make_artifact(
            "bundle-cuda12-newer.tar.gz",
            runtime_line = "cuda12",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        art13 = make_artifact(
            "bundle-cuda13-newer.tar.gz",
            runtime_line = "cuda13",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        release = make_release([art12, art13])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda13"
        assert any("blackwell_runtime_override" in entry for entry in result.selection_log)

    def test_blackwell_skips_incapable_cuda13_line(self, monkeypatch):
        # cuda13 line can't cover sm_120 (only an -older bundle): stay on native cuda12.
        mock_linux_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (13, 0), compute_caps = ["120"])
        art13_older = make_artifact(
            "bundle-cuda13-older.tar.gz",
            runtime_line = "cuda13",
            supported_sms = ["75", "86", "89"],
            min_sm = 75,
            max_sm = 89,
        )
        art12_newer = make_artifact(
            "bundle-cuda12-newer.tar.gz",
            runtime_line = "cuda12",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        release = make_release([art13_older, art12_newer])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda12"
        assert result.primary.name == "bundle-cuda12-newer.tar.gz"

    def test_blackwell_cuda13_unavailable_uses_cuda12(self, monkeypatch):
        # cuda13 runtime libs absent: override must not force an undetected line.
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(driver_cuda_version = (13, 0), compute_caps = ["120"])
        art12 = make_artifact(
            "bundle-cuda12-newer.tar.gz",
            runtime_line = "cuda12",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        release = make_release([art12])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda12"

    def test_non_blackwell_keeps_torch_preference(self, monkeypatch):
        # Non-Blackwell: torch preference untouched, no override.
        mock_linux_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (13, 0), compute_caps = ["86"])
        art12 = make_artifact(
            "bundle-cuda12.tar.gz",
            runtime_line = "cuda12",
            supported_sms = ["86"],
            min_sm = 86,
            max_sm = 86,
        )
        art13 = make_artifact(
            "bundle-cuda13.tar.gz",
            runtime_line = "cuda13",
            supported_sms = ["86"],
            min_sm = 86,
            max_sm = 86,
        )
        release = make_release([art12, art13])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda12"
        assert not any("blackwell_runtime_override" in entry for entry in result.selection_log)

    def test_blackwell_ignores_malformed_runtime_line(self, monkeypatch):
        # A malformed runtime_line must be skipped, not crash the major sort.
        mock_linux_runtime(monkeypatch, ["cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (13, 0), compute_caps = ["120"])
        bad = make_artifact(
            "bundle-cudaX.tar.gz",
            runtime_line = "cudaX",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        art13 = make_artifact(
            "bundle-cuda13-newer.tar.gz",
            runtime_line = "cuda13",
            supported_sms = ["86", "120"],
            min_sm = 86,
            max_sm = 120,
        )
        release = make_release([bad, art13])
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda13"

    def test_blackwell_prefers_cuda14_over_lower_majors(self, monkeypatch):
        # The highest sm_120-capable CUDA major wins.
        mock_linux_runtime(monkeypatch, ["cuda14", "cuda13", "cuda12"])
        host = make_host(driver_cuda_version = (14, 0), compute_caps = ["120"])
        arts = [
            make_artifact(
                f"bundle-{rtl}.tar.gz",
                runtime_line = rtl,
                supported_sms = ["86", "120"],
                min_sm = 86,
                max_sm = 120,
            )
            for rtl in ("cuda12", "cuda13", "cuda14")
        ]
        release = make_release(arts)
        result = linux_cuda_choice_from_release(host, release, preferred_runtime_line = "cuda12")
        assert result is not None
        assert result.primary.runtime_line == "cuda14"

    def test_arm64_host_selects_linux_arm64_cuda_kind(self, monkeypatch):
        # arm64 CUDA host selects the linux-arm64-cuda bundle, not the x64 one.
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
        assert result.primary.name == "app-b9457-linux-arm64-cuda13-portable.tar.gz"

    # --- SM matching ---

    def test_exact_sm_match(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", supported_sms = ["75", "86", "89"], min_sm = 75, max_sm = 89)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is not None
        assert result.primary.name == "bundle.tar.gz"

    def test_sm_not_in_supported_sms(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["86"])
        art = make_artifact("bundle.tar.gz", supported_sms = ["75", "80", "89"], min_sm = 75, max_sm = 89)
        release = make_release([art])
        result = linux_cuda_choice_from_release(host, release)
        assert result is None

    def test_sm_outside_min_range(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(compute_caps = ["50"])
        art = make_artifact("bundle.tar.gz", supported_sms = ["50", "75", "86"], min_sm = 75, max_sm = 90)
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
        art = make_artifact("bundle.tar.gz", supported_sms = ["75", "89"], min_sm = 75, max_sm = 89)
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
    def test_windows_cuda_prefers_published_asset_from_selected_release(self, monkeypatch):
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
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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
                AssertionError("published Windows CUDA choice should not query upstream")
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
        release = make_release([], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000")
        checksums = make_checksums_with_source(
            ["llama-b9000-bin-win-cuda-12.4-x64.zip"],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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

    def test_linux_cpu_fork_without_bundle_raises_no_upstream_fallback(self, monkeypatch):
        # A CPU-only Linux host on the fork never falls back to the ggml-org CPU
        # asset. CPU-only Linux now routes to the fork, but if a release manifest
        # happens to ship no CPU bundle the resolver raises rather than quietly
        # reaching for an upstream asset.
        host = make_host(
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        release = make_release([], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000")
        checksums = make_checksums_with_source(
            [],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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
                AssertionError("fork CPU host must not query upstream assets")
            ),
        )

        with pytest.raises(PrebuiltFallback, match = "no compatible Linux prebuilt asset was found"):
            resolve_install_attempts("latest", host, "unslothai/llama.cpp", "")

    def test_linux_cuda_does_not_fall_back_to_upstream_cpu(self, monkeypatch):
        host = make_host(system = "Linux", machine = "x86_64", compute_caps = ["86"])
        release = make_release([], release_tag = "llama-prebuilt-latest", upstream_tag = "b9000")
        checksums = make_checksums_with_source(
            [],
            release_tag = release.release_tag,
            upstream_tag = "b9000",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )
        mock_linux_runtime(monkeypatch, ["cuda12"])

        with pytest.raises(PrebuiltFallback, match = "no compatible Linux prebuilt asset was found"):
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
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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

    @pytest.mark.parametrize(
        "system, machine, asset_name, install_kind, bundle_profile",
        [
            # CPU-only Linux x64 -> fork linux-cpu (was ggml-org ubuntu-x64).
            ("Linux", "x86_64", "app-b9625-linux-x64-cpu.tar.gz", "linux-cpu", "linux-cpu-x64"),
            # CPU-only Linux arm64 -> fork linux-arm64 (was ggml-org ubuntu-arm64).
            (
                "Linux",
                "aarch64",
                "app-b9625-linux-arm64-cpu.tar.gz",
                "linux-arm64",
                "linux-cpu-arm64",
            ),
            # CPU-only Windows arm64 -> fork windows-arm64 (was ggml-org win-cpu-arm64).
            (
                "Windows",
                "arm64",
                "app-b9625-windows-arm64-cpu.zip",
                "windows-arm64",
                "windows-cpu-arm64",
            ),
        ],
    )
    def test_cpu_host_prefers_published_fork_asset(
        self, monkeypatch, system, machine, asset_name, install_kind, bundle_profile
    ):
        # CPU-only hosts now select the fork's CPU bundle from the manifest and
        # must never query ggml-org upstream assets. Windows x64 CPU is covered
        # separately by test_windows_cpu_prefers_published_asset.
        host = make_host(
            system = system,
            machine = machine,
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        release = make_release(
            [
                make_artifact(
                    asset_name,
                    install_kind = install_kind,
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = bundle_profile,
                    rank = 1000,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9625",
            assets = {asset_name: f"https://published.example/{asset_name}"},
        )
        checksums = make_checksums_with_source(
            [asset_name],
            release_tag = release.release_tag,
            upstream_tag = "b9625",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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
                AssertionError("fork CPU host must not query upstream assets")
            ),
        )

        _requested_tag, resolved_tag, attempts, _approved = resolve_install_attempts(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
        )

        assert resolved_tag == "b9625"
        assert attempts[0].name == asset_name
        assert attempts[0].install_kind == install_kind
        assert attempts[0].source_label == "published"

    def test_cpu_only_unsupported_arch_source_builds(self, monkeypatch):
        # A CPU-only Linux host that is neither x86_64 nor arm64 (ppc64le,
        # riscv64, s390x) has no compatible CPU bundle. It must source-build, not
        # receive the x86_64 linux-cpu binary (the Linux preflight checks libs,
        # not ELF arch, so a wrong-arch binary would slip through).
        host = make_host(
            machine = "ppc64le",
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            nvidia_smi = None,
        )
        assert not host.is_x86_64 and not host.is_arm64
        x64_asset = "app-b9625-linux-x64-cpu.tar.gz"
        release = make_release(
            [
                make_artifact(
                    x64_asset,
                    install_kind = "linux-cpu",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = "linux-cpu-x64",
                    rank = 1000,
                )
            ],
            release_tag = "llama-prebuilt-latest",
            upstream_tag = "b9625",
            assets = {x64_asset: f"https://published.example/{x64_asset}"},
        )
        checksums = make_checksums_with_source(
            [x64_asset],
            release_tag = release.release_tag,
            upstream_tag = "b9625",
        )

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
                [
                    INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                        bundle = release,
                        checksums = checksums,
                    )
                ]
            ),
        )

        with pytest.raises(PrebuiltFallback, match = "no compatible Linux prebuilt asset was found"):
            resolve_install_attempts("latest", host, "unslothai/llama.cpp", "")

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
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
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
    def _cuda_bundle(self, asset_name, release_tag, upstream_tag):
        # Fork CUDA bundle covering the default NVIDIA host (sm 86, cuda12), so each
        # release yields a plan via linux_cuda_choice_from_release.
        art = make_artifact(
            asset_name,
            install_kind = "linux-cuda",
            runtime_line = "cuda12",
            coverage_class = "portable",
            supported_sms = ["75", "80", "86", "89", "90"],
            min_sm = 75,
            max_sm = 90,
        )
        return INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
            bundle = make_release([art], release_tag = release_tag, upstream_tag = upstream_tag),
            checksums = make_checksums_with_source(
                [asset_name],
                release_tag = release_tag,
                upstream_tag = upstream_tag,
            ),
        )

    def test_latest_collects_multiple_older_release_plans_up_to_limit(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(system = "Linux", machine = "x86_64", compute_caps = ["86"])
        releases = [
            self._cuda_bundle("app-b9003-linux-x64-cuda12.tar.gz", "r3", "b9003"),
            self._cuda_bundle("app-b9002-linux-x64-cuda12.tar.gz", "r2", "b9002"),
            self._cuda_bundle("app-b9001-linux-x64-cuda12.tar.gz", "r1", "b9001"),
        ]

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
                releases
            ),
        )

        requested_tag, plans = _fork_manifest_release_plans(
            "latest",
            host,
            "unslothai/llama.cpp",
            "",
            max_release_fallbacks = 2,
        )

        assert requested_tag == "latest"
        assert [plan.release_tag for plan in plans] == ["r3", "r2"]
        assert [plan.llama_tag for plan in plans] == ["b9003", "b9002"]

    def test_latest_skips_non_installable_release_and_keeps_searching(self, monkeypatch):
        mock_linux_runtime(monkeypatch, ["cuda12"])
        host = make_host(system = "Linux", machine = "x86_64", compute_caps = ["86"])
        releases = [
            # r2 ships no fork bundle: yields no plan, is skipped.
            INSTALL_LLAMA_PREBUILT.ResolvedPublishedRelease(
                bundle = make_release([], release_tag = "r2", upstream_tag = "b9002"),
                checksums = make_checksums_with_source(
                    [],
                    release_tag = "r2",
                    upstream_tag = "b9002",
                ),
            ),
            self._cuda_bundle("app-b9001-linux-x64-cuda12.tar.gz", "r1", "b9001"),
        ]

        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "iter_resolved_published_releases",
            lambda requested_tag, published_repo, published_release_tag = "", **_kwargs: iter(
                releases
            ),
        )

        _requested_tag, plans = _fork_manifest_release_plans(
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
        assert env_int("UNSLOTH_LLAMA_MAX_PREBUILT_RELEASE_FALLBACKS", 3, minimum = 1) == 3

    def test_import_with_malformed_release_fallback_env_does_not_crash(self, monkeypatch):
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

    def _upstream(
        self,
        *runtime_versions,
        current_names: bool = False,
    ):
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
        # A 13.0 driver can't run a 13.1 build (forward minor), so it is gated off
        # cuda13 to the cuda12 build, even when only cuda13 runtime libs are detected.
        mock_windows_runtime(monkeypatch, ["cuda13"])
        host = make_host(system = "Windows", machine = "AMD64", driver_cuda_version = (13, 0))
        assets = self._upstream("13.1", "12.4")
        result = windows_cuda_attempts(host, self.TAG, assets, None)
        assert result[0].runtime_line == "cuda12"
        assert result[0].name == f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip"

    def test_driver_at_published_minor_selects_cuda13(self, monkeypatch):
        # A 13.1 driver matches the published 13.1 build.
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
        # #5106: cudart bundle must surface on runtime_url so install_from_archives downloads it.
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
        # #5861: selector must follow the published cuda13 bump 13.1 -> 13.3, not hardcode 13.1.
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
        # Only cuda-13.3 published; a 13.1 driver can't run it (forward minor), so it is
        # gated to cuda-12.4. A 13.3 driver still gets 13.3 (other tests).
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
# N.1b. _windows_cuda_attempt_covers_blackwell -- Blackwell coverage classifier
# ===========================================================================


class TestWindowsCudaAttemptCoversBlackwell:
    """A windows-cuda attempt covers Blackwell only when its toolkit minor (from the asset name) is >= 12.8, or, for app-named bundles without a toolkit minor, its declared max_sm reaches sm_120."""

    TAG = "b8508"

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
        assert _windows_cuda_attempt_covers_blackwell(self._win_cuda_attempt(minor)) is covers

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
        # Fork app-named windows-cuda bundle: no toolkit minor in the name, SM coverage declared directly.
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
            ("newer", "cuda13", 120, True),  # native Blackwell build
            ("newer", "cuda12", 120, True),  # 12.8 toolkit app bundle reaches sm120
            ("older", "cuda12", 89, False),  # 12.4 toolkit app bundle stops at Ada
        ],
    )
    def test_attempt_covers_blackwell_app_bundle(self, profile, runtime_line, max_sm, covers):
        # App-named bundles carry no toolkit minor; coverage is read from max_sm.
        attempt = self._app_attempt(profile, runtime_line, max_sm)
        assert _windows_cuda_attempt_covers_blackwell(attempt) is covers


# ===========================================================================
# N.1c. direct_upstream_release_plan -- Blackwell windows-cuda fallback ordering
# ===========================================================================


class TestDirectUpstreamBlackwellPin:
    """On the simple/upstream path a Blackwell host drops the sm_120-incapable cuda-12.4 attempt; with no pinned fallback it falls through to the windows-cpu build, while an in-release cuda-13.3 build is taken when present."""

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
                {"name": n, "browser_download_url": f"https://example.com/{n}"} for n in names
            ],
        }

    def _no_torch(self, monkeypatch):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "detect_torch_cuda_runtime_preference",
            lambda host: CudaRuntimePreference(runtime_line = None, selection_log = []),
        )

    def test_blackwell_13_1_falls_to_cpu(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        plan = direct_upstream_release_plan(self._release(), host, UPSTREAM_REPO, "latest")
        order = [(a.tag, a.runtime_line or a.install_kind) for a in plan.attempts]
        # cuda-12.4 (no sm_120) is dropped entirely on Blackwell, and there is no pinned
        # b9360 fallback anymore, so the host falls through to the windows-cpu build.
        assert order == [(self.TAG, "windows-cpu")]
        assert "b9360" not in [a.tag for a in plan.attempts]
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
        plan = direct_upstream_release_plan(self._release(), host, UPSTREAM_REPO, "latest")
        assert "b9360" not in [a.tag for a in plan.attempts]
        assert plan.attempts[0].tag == self.TAG
        assert plan.attempts[0].runtime_line == "cuda13"
        assert plan.attempts[0].name == f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip"


# N.1c2. Blackwell never falls to a non-sm_120 windows-cuda attempt


class TestBlackwellCuda124Exclusion:
    """A Blackwell host must never have a windows-cuda attempt that can't offload sm_120 anywhere in its chain."""

    def _bw_host(self):
        return make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )

    def _upstream_cuda(
        self,
        minor,
        tag = "b9365",
    ):
        return AssetChoice(
            repo = "ggml-org/llama.cpp",
            tag = tag,
            name = f"llama-{tag}-bin-win-cuda-{minor}-x64.zip",
            url = f"https://example.com/{minor}",
            source_label = "upstream",
            install_kind = "windows-cuda",
            runtime_line = "cuda" + minor.split(".")[0],
        )

    def test_drops_124_keeps_133_on_blackwell(self):
        kept = INSTALL_LLAMA_PREBUILT._drop_blackwell_incapable_windows_cuda(
            self._bw_host(),
            [self._upstream_cuda("13.3"), self._upstream_cuda("12.4")],
        )
        assert [a.name for a in kept] == ["llama-b9365-bin-win-cuda-13.3-x64.zip"]

    def test_keeps_manifest_cuda12_bundle_with_sm120(self):
        # Published cuda12 app bundles are toolkit-12.8 builds with sm_120; manifest SM
        # metadata must keep them on Blackwell.
        bundle = AssetChoice(
            repo = "unslothai/llama.cpp",
            tag = "b9585",
            name = "app-b9585-windows-x64-cuda12-portable.zip",
            url = "https://example.com/app",
            source_label = "published",
            install_kind = "windows-cuda",
            runtime_line = "cuda12",
            supported_sms = ["70", "120"],
            max_sm = 120,
        )
        kept = INSTALL_LLAMA_PREBUILT._drop_blackwell_incapable_windows_cuda(
            self._bw_host(), [bundle]
        )
        assert kept == [bundle]
        assert _windows_cuda_attempt_covers_blackwell(bundle)

    def test_manifest_bundle_without_sm120_dropped(self):
        bundle = AssetChoice(
            repo = "unslothai/llama.cpp",
            tag = "b9585",
            name = "app-b9585-windows-x64-cuda12-older.zip",
            url = "https://example.com/app",
            source_label = "published",
            install_kind = "windows-cuda",
            runtime_line = "cuda12",
            supported_sms = ["70", "75", "80"],
            max_sm = 80,
        )
        assert (
            INSTALL_LLAMA_PREBUILT._drop_blackwell_incapable_windows_cuda(self._bw_host(), [bundle])
            == []
        )

    def test_non_blackwell_host_unfiltered(self):
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (12, 9),
            compute_caps = ["89"],
        )
        attempts = [self._upstream_cuda("12.4")]
        assert (
            INSTALL_LLAMA_PREBUILT._drop_blackwell_incapable_windows_cuda(host, attempts)
            == attempts
        )

    def test_non_cuda_attempts_pass_through(self):
        cpu = AssetChoice(
            repo = "ggml-org/llama.cpp",
            tag = "b9365",
            name = "llama-b9365-bin-win-cpu-x64.zip",
            url = "https://example.com/cpu",
            source_label = "upstream",
            install_kind = "windows-cpu",
        )
        kept = INSTALL_LLAMA_PREBUILT._drop_blackwell_incapable_windows_cuda(
            self._bw_host(), [self._upstream_cuda("12.4"), cpu]
        )
        assert kept == [cpu]


class TestLinuxPublishedAttemptsNvidiaCpuGate:
    """Live fork-manifest path: an NVIDIA host whose CUDA selection finds nothing gets an empty attempt list (source-builds with CUDA), not the manifest CPU bundle. CPU-only hosts still get the CPU bundle."""

    def _cpu_only_bundle(self):
        return make_release(
            [
                make_artifact(
                    "app-b8508-linux-x64-cpu.tar.gz",
                    install_kind = "linux-cpu",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                    rank = 1000,
                ),
            ]
        )

    def test_nvidia_host_without_cuda_line_gets_no_cpu_attempt(self, monkeypatch):
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "detect_torch_cuda_runtime_preference",
            lambda host: CudaRuntimePreference(runtime_line = None, selection_log = []),
        )
        monkeypatch.setattr(
            INSTALL_LLAMA_PREBUILT,
            "detected_linux_runtime_lines",
            lambda: (["cuda13"], {"cuda13": ["/usr/local/cuda/lib64"]}),
        )
        host = make_host(driver_cuda_version = (13, 1), compute_caps = ["100"])
        attempts = INSTALL_LLAMA_PREBUILT._linux_published_attempts(host, self._cpu_only_bundle())
        assert attempts == []

    def test_cpu_host_gets_cpu_attempt(self):
        host = make_host(
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        attempts = INSTALL_LLAMA_PREBUILT._linux_published_attempts(host, self._cpu_only_bundle())
        assert [a.install_kind for a in attempts] == ["linux-cpu"]


# ===========================================================================
# N.1d. published_windows_cuda_attempts -- version-dynamic ordering seed
# ===========================================================================


class TestPublishedWindowsCudaAttemptsDynamicMajor:
    """The ordering seed is derived from the release's published minors, so a future CUDA major is selectable, not hidden by a hardcoded cuda12/cuda13 seed."""

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
        # The dynamic seed lets a 14.x driver reach a published cuda14 build; the old
        # hardcoded cuda12/cuda13 seed would never order it.
        mock_windows_runtime(monkeypatch, ["cuda14", "cuda13", "cuda12"])
        release = self._release([("14.0", "cuda14"), ("13.3", "cuda13"), ("12.4", "cuda12")])
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
        # A 13.3 driver gets the real 13.3 build.
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
        # A 13.1 driver is gated off published 13.3 and falls to cuda12.
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
    """The manifest install path drops sm_120-incapable windows-cuda attempts on Blackwell exactly like the filename path; with no pinned fallback a 13.1 host left with only cuda-12.4 has no usable CUDA attempt and walks back, while a 13.3 host keeps its in-release cuda-13.3 build."""

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

    def test_no_cuda_attempt_on_published_path_for_13_1(self, monkeypatch):
        mock_windows_runtime(monkeypatch, ["cuda13", "cuda12"])
        self._no_torch(monkeypatch)
        # After the published Blackwell filter drops every attempt, the resolver
        # walks back to the upstream release; stub that fetch so the unit test stays
        # offline (a live GitHub call is blocked by the security scanner) and the
        # walk-back deterministically finds no usable CUDA build -> PrebuiltFallback.
        monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "github_release_assets", lambda repo, tag: {})
        release = self._release([("13.3", "cuda13"), ("12.4", "cuda12")])
        checksums = self._checksums(["12.4"])  # 13.3 gated off for a 13.1 driver
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        # The sm_120-incapable upstream cuda-12.4 zip is dropped on Blackwell, and there
        # is no pinned b9360 fallback anymore, so no usable windows-cuda attempt remains
        # and the manifest resolver walks back instead of selecting cuda-12.4.
        with pytest.raises(PrebuiltFallback):
            resolve_release_asset_choice(host, self.TAG, release, checksums)

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
    """app-named windows-cuda bundles carry no minor, so the driver-minor gate is skipped; selection must filter by SM coverage instead of handing every host the lowest-rank "older" bundle."""

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
        # Not the lowest-rank "older" bundle (max_sm 89); the tightest covering bundle
        # is cuda12-newer (range 86-120).
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
        # Regression: a synthetic '13.1' minor gate dropped the whole cuda13 line on a
        # 13.0 driver; cuda13 app bundles are gated at the major level now.
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

    def test_app_bundle_offered_when_no_runtime_dll_detected(self, monkeypatch):
        # torch bundles cudart in torch/lib, which DLL probing misses, so detection
        # returns nothing; the app bundle ships its own runtime, so selection must
        # fall back to the driver-derived order rather than drop to the upstream build.
        mock_windows_runtime(monkeypatch, [])
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
        assert result, "torch-only host must still get the fork app bundle"
        assert result[0].name == f"app-{self.TAG}-windows-x64-cuda13-newer.zip"


class TestPublishedRocmGfxSelection:
    """Published ROCm bundles match by detected gfx family, not rank -- rank ties would alphabetically hand every AMD GPU the gfx103X bundle."""

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
        # gfx1033 shares the gfx103 prefix but isn't in any bundle's mapped_targets, so
        # it must fall back to source, not be served gfx103X.
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        for unbuilt in ("gfx1033", "gfx1035", "gfx1104", "gfx1202"):
            assert (
                INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
                    release, self._host(unbuilt), "linux-rocm"
                )
                is None
            ), unbuilt

    def test_family_token_matches_family_bundle(self):
        # The update path forwards a family token (gfx110X, lowercased to gfx110x), not
        # a concrete arch; it must still select the family bundle, not source-build.
        release = self._release("linux-rocm", "app-b9457-linux-x64-rocm")
        for token in ("gfx110X", "gfx110x"):
            choice = INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
                release, self._host(token), "linux-rocm"
            )
            assert choice is not None, token
            assert choice.name == "app-b9457-linux-x64-rocm-gfx110X.tar.gz", token

    def test_windows_family_token_matches_family_bundle(self):
        # The Windows update path forwards the same family token (gfx120X), so the
        # family-label match must cover windows-rocm too.
        release = self._release("windows-rocm", "app-b9457-windows-x64-rocm")
        choice = INSTALL_LLAMA_PREBUILT.published_rocm_choice_for_host(
            release, self._host("gfx120x"), "windows-rocm"
        )
        assert choice is not None
        assert choice.name == "app-b9457-windows-x64-rocm-gfx120X.zip"


class TestPublishedMacosForkSelection:
    """macOS routes to the fork's llama-<tag>-bin-macos-<arch>.tar.gz, selected by install_kind."""

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
    """Runtime archive inherits a manifest hash, or is dropped."""

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
        host = make_host(has_usable_nvidia = False, nvidia_smi = None, has_physical_nvidia = False)
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "linux-cpu"
        assert result.name == name

    def test_linux_cpu_missing(self, monkeypatch):
        self._mock_github_assets(monkeypatch, {})
        host = make_host(has_usable_nvidia = False, nvidia_smi = None, has_physical_nvidia = False)
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

    def test_windows_blackwell_drops_incapable_cuda_to_cpu(self, monkeypatch):
        # A Blackwell host on a 13.1 driver gets cuda-13.3 gated off and only the
        # sm_120-incapable cuda-12.4 build left; it must fall through to the CPU
        # bundle rather than be handed a GPU build it cannot offload.
        names = [
            f"llama-{self.TAG}-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip",
            f"llama-{self.TAG}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            f"llama-{self.TAG}-bin-win-cpu-x64.zip",
        ]
        self._mock_github_assets(monkeypatch, {n: f"https://x/{n}" for n in names})
        host = make_host(
            system = "Windows",
            machine = "AMD64",
            has_usable_nvidia = True,
            driver_cuda_version = (13, 1),
            compute_caps = ["120"],
        )
        result = resolve_upstream_asset_choice(host, self.TAG)
        assert result.install_kind == "windows-cpu"
        assert result.name == f"llama-{self.TAG}-bin-win-cpu-x64.zip"

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
        with pytest.raises(PrebuiltFallback, match = "no prebuilt policy exists for Linux aarch64"):
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
    def test_arm64_sequoia_pins_b9415(self):
        host = _macos_host("arm64", (15, 5))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_arm64_sonoma_pins_b9415(self):
        host = _macos_host("arm64", (14, 7))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_x64_ventura_13_3_pins_b9415(self):
        host = _macos_host("x86_64", (13, 3))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) == "b9415"

    def test_tahoe_takes_latest(self):
        host = _macos_host("arm64", (26, 0))
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None

    def test_unknown_version_takes_latest(self):
        host = _macos_host("arm64", None)
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None

    def test_fork_repo_is_dormant(self):
        host = _macos_host("arm64", (15, 5))
        fork = INSTALL_LLAMA_PREBUILT.DEFAULT_PUBLISHED_REPO
        assert pinned_macos_release_tag(host, fork) is None

    def test_non_macos_host_is_dormant(self):
        host = make_host(system = "Linux", machine = "x86_64")
        assert pinned_macos_release_tag(host, UPSTREAM_REPO) is None


class TestResolveSimpleMacosPin:
    """Pre-26 upstream macOS resolves b9415; macOS 26 keeps latest."""

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

        def fake_iter(
            repo,
            published_release_tag = "",
            requested_tag = "",
        ):
            calls.append((repo, published_release_tag, requested_tag))
            # Real iterator: a specific tag yields only that release.
            if requested_tag and requested_tag != "latest":
                yield _release(requested_tag)
                return
            for tag in self.TAGS:
                yield _release(tag)

        monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "iter_release_payloads_by_time", fake_iter)
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
        assert calls[0][2] == "b9415"
        assert plans[0].approved_checksums.artifacts == {}

    def test_tahoe_host_takes_latest_release(self, monkeypatch):
        calls = self._feed(monkeypatch)
        host = _macos_host("arm64", (26, 0))
        requested_tag, plans = resolve_simple_install_release_plans(
            "latest", host, "ggml-org/llama.cpp", ""
        )
        assert requested_tag == "latest"
        assert plans[0].release_tag == "b9442"
        # No pin: the iterator was asked for latest.
        assert calls[0][2] == "latest"


# ===========================================================================
# Linux arm64 + GPU must not install the x64-only fork bundle
# ===========================================================================


class TestLinuxArm64ForkFallsBackToSource:
    """The fork ships linux-arm64-cuda bundles; an arm64 Linux fork host delegates to the manifest-aware resolver (arm64 CUDA bundle, or source if none matches) instead of hard-failing."""

    def test_arm64_nvidia_fork_delegates_to_manifest_resolver(self, monkeypatch):
        # arm64 fork hosts are no longer blocked up front; routed to the manifest resolver.
        called = {}

        def _full(llama_tag, host, repo, tag, **_kw):
            called["args"] = (host.machine, repo)
            return "b9457", ["plan"]

        monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_fork_manifest_release_plans", _full)
        host = make_host(system = "Linux", machine = "aarch64")
        tag, plans = resolve_simple_install_release_plans("latest", host, "unslothai/llama.cpp", "")
        assert called.get("args") == ("aarch64", "unslothai/llama.cpp")
        assert plans == ["plan"]

    def test_x86_64_fork_delegates_to_manifest_resolver(self, monkeypatch):
        # The old linux-x64 arch guard is gone: x64 fork hosts route to the manifest
        # resolver like every other fork host, not a separate filename-parsing path.
        called = {}

        def _full(llama_tag, host, repo, tag, **_kw):
            called["args"] = (host.machine, repo)
            return "b9457", ["plan"]

        monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_fork_manifest_release_plans", _full)
        host = make_host(system = "Linux", machine = "x86_64")
        tag, plans = resolve_simple_install_release_plans("latest", host, "unslothai/llama.cpp", "")
        assert called.get("args") == ("x86_64", "unslothai/llama.cpp")
        assert plans == ["plan"]

    def test_arm64_cpu_on_ggml_org_is_not_blocked(self, monkeypatch):
        # ggml-org is reachable only via an explicit --published-repo override now,
        # but the guard must still not fire on arm64 there; it reaches the iterator
        # (empty here -> generic message).
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
            resolve_simple_install_release_plans("latest", host, "ggml-org/llama.cpp", "")
        assert "linux-x64 prebuilts" not in str(exc.value)


# ===========================================================================
# arm64 Linux GPU: CPU prebuilt fallback after a failed source build (--cpu-fallback)
# ===========================================================================


class TestCpuFallback:
    """--cpu-fallback drops GPU attributes so the host's OS/arch CPU prebuilt is selected, letting an arm64 GPU host install the fork's arm64 CPU bundle when its source build produced no binary."""

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
        # We only care about the host handed to the resolver before the fallback exit.
        with pytest.raises(SystemExit):
            INSTALL_LLAMA_PREBUILT.install_prebuilt(
                install_dir = tmp_path / "llama",
                llama_tag = "latest",
                published_repo = "ggml-org/llama.cpp",
                published_release_tag = "",
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
        # A GPU arm64 host can't pick the CPU arm64 bundle on its own.
        with pytest.raises(PrebuiltFallback):
            direct_upstream_release_plan(
                release, self._arm64_nvidia(), "ggml-org/llama.cpp", "latest"
            )
        # force_cpu drops GPU attributes, so the CPU arm64 bundle is selected.
        cpu_host = make_host(
            system = "Linux",
            machine = "aarch64",
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        plan = direct_upstream_release_plan(release, cpu_host, "ggml-org/llama.cpp", "latest")
        assert plan.attempts[0].install_kind == "linux-arm64"
        assert plan.attempts[0].name == f"llama-{tag}-bin-ubuntu-arm64.tar.gz"

    def test_setup_sh_has_arm64_cpu_prebuilt_fallback(self):
        source = self._SETUP_SH.read_text(encoding = "utf-8")
        # The arm64 GPU last-resort CPU fallback now pulls the fork's arm64 CPU
        # bundle (app-<tag>-linux-arm64-cpu.tar.gz), not ggml-org's, and is gated
        # on a degraded source build for arm64.
        start = source.index("_ARM64_CPU_CMD=(")
        end = source.index(")", start)
        block = source[start:end]
        assert "--cpu-fallback" in block
        assert '--published-repo "unslothai/llama.cpp"' in block
        assert '--published-repo "ggml-org/llama.cpp"' not in block
        assert "_LLAMA_CPP_DEGRADED" in source


# ===========================================================================
# setup.sh / setup.ps1: CUDA toolkit newer than driver diagnostics
# ===========================================================================


@pytest.mark.skipif(sys.platform == "win32", reason = "bash-only Unsloth installer tests")
class TestCudaDriverToolkitMismatchMessage:
    _SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
    _SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"

    def _setup_sh_cuda_helper_fragment(self):
        source = self._SETUP_SH.read_text(encoding = "utf-8")
        start = source.index("_nvcc_meets_llama_minimum()")
        end = source.index("print_llama_error_log()")
        return source[start:end]

    def _run_bash(
        self,
        script,
        *,
        env = None,
    ):
        proc = subprocess.run(
            ["/bin/bash", "-c", script],
            check = True,
            text = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            env = {**os.environ, **(env or {})},
        )
        return proc.stdout + proc.stderr

    def _fake_nvidia_smi(self, tmp_path, output):
        mock_bin = tmp_path / "bin"
        mock_bin.mkdir()
        nvidia_smi = mock_bin / "nvidia-smi"
        nvidia_smi.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                cat <<'OUT'
                {output}
                OUT
                """
            ),
            encoding = "utf-8",
        )
        nvidia_smi.chmod(0o755)
        return mock_bin

    def test_setup_sh_same_major_minor_mismatch_is_accepted(self, tmp_path):
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.1 |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _driver="$(_cuda_driver_max_version)"
            if _cuda_toolkit_major_gt_driver "13.3" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "13.3" "$_driver"
            else
                printf 'compatible:%s\\n' "$_driver"
            fi
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert "compatible:13.1" in output
        assert "major-version mismatch" not in output

    def test_setup_sh_driver_major_too_old_message_names_major_mismatch(self, tmp_path):
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 570.95   Driver Version: 570.95   CUDA Version: 12.9 |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _driver="$(_cuda_driver_max_version)"
            if _cuda_toolkit_major_gt_driver "13.3" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "13.3" "$_driver"
            fi
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert (
            "CUDA Toolkit 13.3 is a major-version mismatch: toolkit major 13 "
            "exceeds driver CUDA major 12 (12.9)."
        ) in output
        assert (
            "Update the NVIDIA GPU driver to run CUDA Toolkit 13.3, or install "
            "a CUDA 12.x toolkit."
        ) in output
        assert "prebuilt CUDA bundle" in output

    def test_setup_sh_happy_path_does_not_print_mismatch(self, tmp_path):
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.3 |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _driver="$(_cuda_driver_max_version)"
            if _cuda_toolkit_major_gt_driver "13.3" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "13.3" "$_driver"
            else
                printf 'compatible:%s\\n' "$_driver"
            fi
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert "compatible:13.3" in output
        assert "Unsloth supports CUDA Toolkit" not in output

    def test_setup_sh_skips_check_without_nvidia_smi(self, tmp_path):
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _driver="$(_cuda_driver_max_version)"
            if [ -n "$_driver" ] && _cuda_toolkit_major_gt_driver "13.3" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "13.3" "$_driver"
            else
                printf 'skipped\\n'
            fi
            """
        )
        output = self._run_bash(script, env = {"PATH": str(empty_bin)})
        assert "skipped" in output
        assert "Unsloth supports CUDA Toolkit" not in output

    def test_setup_sh_unparsable_nvidia_smi_output_falls_back(self, tmp_path):
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: N/A |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _driver="$(_cuda_driver_max_version)"
            if [ -n "$_driver" ] && _cuda_toolkit_major_gt_driver "13.3" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "13.3" "$_driver"
            else
                printf 'fallback:%s\\n' "${{_driver:-generic}}"
            fi
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert "fallback:generic" in output
        assert "Unsloth supports CUDA Toolkit" not in output

    def test_setup_sh_cuda_version_gt_compares_numerically(self):
        # 13.9 vs 13.10 is where a lexical compare goes wrong (9 > 1).
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            {self._setup_sh_cuda_helper_fragment()}
            for pair in "13.9 13.10" "13.10 13.9" "14.0 13.9" "13.3 13.3"; do
                if _cuda_version_gt $pair; then r=gt; else r=le; fi
                printf '%s -> %s\\n' "$pair" "$r"
            done
            """
        )
        output = self._run_bash(script)
        assert "13.9 13.10 -> le" in output
        assert "13.10 13.9 -> gt" in output
        assert "14.0 13.9 -> gt" in output
        assert "13.3 13.3 -> le" in output

    def test_setup_ps1_mirrors_driver_mismatch_guidance(self):
        source = self._SETUP_PS1.read_text(encoding = "utf-8")
        assert "Write-CudaDriverToolkitMismatch" in source
        assert (
            "CUDA Toolkit $ToolkitVersion is a major-version mismatch: toolkit "
            "major $toolkitMajor exceeds driver CUDA major $driverMajor"
        ) in source
        assert (
            "Update the NVIDIA GPU driver to run CUDA Toolkit $ToolkitVersion, "
            "or install a CUDA $driverMajor.x toolkit." in source
        )
        assert (
            "Or let Unsloth use the prebuilt CUDA bundle; it does not need the local toolkit."
        ) in source
        assert (
            "Write-CudaDriverToolkitMismatch -ToolkitVersion $IncompatibleToolkit "
            "-DriverMaxCuda $DriverMaxCuda"
        ) in source

    def _fake_nvcc(self, tmp_path, release):
        mock_bin = tmp_path / f"nvcc-bin-{release.replace('.', '-')}"
        mock_bin.mkdir()
        nvcc = mock_bin / "nvcc"
        nvcc.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                printf '%s\\n' 'Cuda compilation tools, release {release}, V{release}.0'
                """
            ),
            encoding = "utf-8",
        )
        nvcc.chmod(0o755)
        return nvcc

    def test_setup_sh_major_mismatch_uses_newest_compatible_detected_toolkit(self, tmp_path):
        blocked_nvcc = self._fake_nvcc(tmp_path, "13.3")
        older_nvcc = self._fake_nvcc(tmp_path, "12.6")
        compatible_nvcc = self._fake_nvcc(tmp_path, "12.8")
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _cuda_nvcc_candidate_paths() {{
                printf '%s\\n' "{blocked_nvcc}" "{older_nvcc}" "{compatible_nvcc}"
            }}
            NVCC_PATH="{blocked_nvcc}"
            GPU_BACKEND="cuda"
            _BUILD_DESC="building"
            _NVCC_CHECK="$(_nvcc_meets_llama_minimum "$NVCC_PATH")"
            _NVCC_STATUS="$(printf '%s\\n' "$_NVCC_CHECK" | sed -n '1p')"
            _NVCC_VER="$(printf '%s\\n' "$_NVCC_CHECK" | sed -n '2p')"
            _DRIVER_MAX_CUDA="12.9"
            _CUDA_TOOLKIT_ALLOWED=true
            if [ "$_NVCC_STATUS" != "too_old" ] && \
               [ -n "$_NVCC_VER" ] && \
               _cuda_toolkit_major_gt_driver "$_NVCC_VER" "$_DRIVER_MAX_CUDA"; then
                _BLOCKED_NVCC_VER="$_NVCC_VER"
                if _ALT_NVCC_CHECK="$(_cuda_find_compatible_nvcc_for_driver "$_DRIVER_MAX_CUDA" "$NVCC_PATH")"; then
                    NVCC_PATH="$(printf '%s\\n' "$_ALT_NVCC_CHECK" | sed -n '1p')"
                    _NVCC_VER="$(printf '%s\\n' "$_ALT_NVCC_CHECK" | sed -n '2p')"
                    GPU_BACKEND="cuda"
                    substep "CUDA Toolkit $_BLOCKED_NVCC_VER is a major-version mismatch with driver CUDA $_DRIVER_MAX_CUDA; using compatible CUDA Toolkit $_NVCC_VER at $NVCC_PATH."
                else
                    NVCC_PATH=""
                    GPU_BACKEND=""
                    _BUILD_DESC="building (CPU, CUDA toolkit major > driver)"
                    _CUDA_TOOLKIT_ALLOWED=false
                fi
            fi
            printf 'NVCC_PATH=%s\\n' "$NVCC_PATH"
            printf 'NVCC_VER=%s\\n' "$_NVCC_VER"
            printf 'GPU_BACKEND=%s\\n' "$GPU_BACKEND"
            printf 'BUILD_DESC=%s\\n' "$_BUILD_DESC"
            printf 'ALLOWED=%s\\n' "$_CUDA_TOOLKIT_ALLOWED"
            """
        )
        output = self._run_bash(script)
        assert f"NVCC_PATH={compatible_nvcc}" in output
        assert "NVCC_VER=12.8" in output
        assert "GPU_BACKEND=cuda" in output
        assert "ALLOWED=true" in output
        assert "CPU" not in output

    def test_setup_sh_parses_cuda_umd_version_variant(self, tmp_path):
        # Newer drivers report "CUDA UMD Version"; the helper must read it too.
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 610.00   Driver Version: 610.00   CUDA UMD Version: 13.0 |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            {self._setup_sh_cuda_helper_fragment()}
            printf '%s' "$(_cuda_driver_max_version)"
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert output.strip() == "13.0"

    def test_setup_sh_empty_toolkit_version_skips_mismatch(self, tmp_path):
        # The real guard requires a non-empty nvcc version before warning.
        mock_bin = self._fake_nvidia_smi(
            tmp_path,
            "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.0 |",
        )
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            _nvcc=""
            _driver="$(_cuda_driver_max_version)"
            if [ -n "$_nvcc" ] && [ -n "$_driver" ] && _cuda_toolkit_major_gt_driver "$_nvcc" "$_driver"; then
                _print_cuda_driver_toolkit_mismatch "$_nvcc" "$_driver"
            else
                printf 'skipped\\n'
            fi
            """
        )
        output = self._run_bash(
            script,
            env = {"PATH": f"{mock_bin}:{os.environ.get('PATH', '')}"},
        )
        assert "skipped" in output
        assert "Unsloth supports CUDA Toolkit" not in output

    def test_setup_sh_nvcc_below_minimum_is_too_old(self, tmp_path):
        # CUDA toolkit < 12.4 short-circuits to the too_old branch (no mismatch).
        nvcc = self._fake_nvcc(tmp_path, "12.0")
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            {self._setup_sh_cuda_helper_fragment()}
            _nvcc_meets_llama_minimum "{nvcc}"
            """
        )
        output = self._run_bash(script)
        assert output.splitlines()[0] == "too_old"
        assert "12.0" in output

    def test_setup_sh_compatible_finder_rejects_too_old_only_candidate(self, tmp_path):
        # Only candidate is below the llama minimum: finder must fail (CPU fallback), not pick 12.0.
        blocked_nvcc = self._fake_nvcc(tmp_path, "13.3")
        too_old_nvcc = self._fake_nvcc(tmp_path, "12.0")
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            {self._setup_sh_cuda_helper_fragment()}
            _cuda_nvcc_candidate_paths() {{
                printf '%s\\n' "{blocked_nvcc}" "{too_old_nvcc}"
            }}
            if _ALT="$(_cuda_find_compatible_nvcc_for_driver "12.9" "{blocked_nvcc}")"; then
                printf 'FOUND:%s\\n' "$_ALT"
            else
                printf 'NONE\\n'
            fi
            """
        )
        output = self._run_bash(script)
        assert "NONE" in output
        assert "FOUND" not in output

    def _cuda_build_decision_output(self, *, nvcc_path, driver):
        # Mirror setup.sh's source-build decision: keep toolkit, switch, or degrade to CPU.
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            C_WARN=
            substep() {{ printf '%s\\n' "$1"; }}
            {self._setup_sh_cuda_helper_fragment()}
            NVCC_PATH="{nvcc_path}"
            GPU_BACKEND="cuda"
            _DRIVER_MAX_CUDA="{driver}"
            _NVCC_CHECK="$(_nvcc_meets_llama_minimum "$NVCC_PATH")"
            _NVCC_STATUS="$(printf '%s\\n' "$_NVCC_CHECK" | sed -n '1p')"
            _NVCC_VER="$(printf '%s\\n' "$_NVCC_CHECK" | sed -n '2p')"
            _CUDA_TOOLKIT_ALLOWED=true
            if [ "$_NVCC_STATUS" = "too_old" ]; then
                NVCC_PATH=""; GPU_BACKEND=""; _CUDA_TOOLKIT_ALLOWED=false
            elif [ -n "$_NVCC_VER" ] && [ -n "$_DRIVER_MAX_CUDA" ] && _cuda_toolkit_major_gt_driver "$_NVCC_VER" "$_DRIVER_MAX_CUDA"; then
                if _ALT="$(_cuda_find_compatible_nvcc_for_driver "$_DRIVER_MAX_CUDA" "$NVCC_PATH")"; then
                    NVCC_PATH="$(printf '%s\\n' "$_ALT" | sed -n '1p')"
                    _NVCC_VER="$(printf '%s\\n' "$_ALT" | sed -n '2p')"
                else
                    NVCC_PATH=""; GPU_BACKEND=""; _CUDA_TOOLKIT_ALLOWED=false
                fi
            fi
            printf 'NVCC_PATH=%s\\n' "$NVCC_PATH"
            printf 'NVCC_VER=%s\\n' "$_NVCC_VER"
            printf 'GPU_BACKEND=%s\\n' "$GPU_BACKEND"
            printf 'ALLOWED=%s\\n' "$_CUDA_TOOLKIT_ALLOWED"
            """
        )
        return self._run_bash(script)

    def test_setup_sh_same_major_newer_minor_keeps_original_toolkit(self, tmp_path):
        # Same-major newer-minor (13.3 vs driver 13.0): build CUDA with it, never fall back.
        toolkit = self._fake_nvcc(tmp_path, "13.3")
        output = self._cuda_build_decision_output(nvcc_path = toolkit, driver = "13.0")
        assert f"NVCC_PATH={toolkit}" in output
        assert "NVCC_VER=13.3" in output
        assert "GPU_BACKEND=cuda" in output
        assert "ALLOWED=true" in output

    def test_setup_sh_missing_driver_version_still_enables_cuda(self, tmp_path):
        # No driver CUDA version from nvidia-smi: keep CUDA enabled (pre-fix behavior), not CPU.
        toolkit = self._fake_nvcc(tmp_path, "13.3")
        output = self._cuda_build_decision_output(nvcc_path = toolkit, driver = "")
        assert f"NVCC_PATH={toolkit}" in output
        assert "GPU_BACKEND=cuda" in output
        assert "ALLOWED=true" in output

    def test_setup_sh_compatible_finder_rejects_newer_major_only_candidate(self, tmp_path):
        # Only alternative is still newer-major than the driver: finder must fail, not pick it.
        blocked_nvcc = self._fake_nvcc(tmp_path, "13.3")
        other_newer_nvcc = self._fake_nvcc(tmp_path, "13.1")
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            {self._setup_sh_cuda_helper_fragment()}
            _cuda_nvcc_candidate_paths() {{
                printf '%s\\n' "{blocked_nvcc}" "{other_newer_nvcc}"
            }}
            if _ALT="$(_cuda_find_compatible_nvcc_for_driver "12.9" "{blocked_nvcc}")"; then
                printf 'FOUND:%s\\n' "$_ALT"
            else
                printf 'NONE\\n'
            fi
            """
        )
        output = self._run_bash(script)
        assert "NONE" in output
        assert "FOUND" not in output


class TestExactSourceAssetUrl:
    """exact_source_asset_url resolves the published source-commit release asset
    for mix builds even when the manifest omits the top-level repo/release_tag.

    A mix build's merge commit is never pushed, so its codeload/archive URLs
    404; the ``llama.cpp-source-commit-<sha>.tar.gz`` asset is the only durable
    copy. If the asset URL resolves empty, hydration falls through to the 404-ing
    commit archive and the whole prebuilt install fails to a source build.
    """

    COMMIT = "c4fca6de" + "a" * 32  # 40-char sha
    INSTALL_TAG = "b9616-mix-17e50db"

    def _artifact(self, *, repo):
        name = exact_source_archive_logical_name(self.COMMIT)
        return ApprovedArtifactHash(
            asset_name = name,
            sha256 = "c" * 64,
            repo = repo,
            kind = "exact-source",
        )

    def _checksums(self, *, repo, release_tag):
        return ApprovedReleaseChecksums(
            repo = repo,
            release_tag = release_tag,
            upstream_tag = "b9616",
            source_repo = "unslothai/llama.cpp",
            source_commit = self.COMMIT,
            artifacts = {},
        )

    def _expected(self, repo, tag):
        return (
            f"https://github.com/{repo}/releases/download/"
            f"{tag}/{exact_source_archive_logical_name(self.COMMIT)}"
        )

    def test_uses_manifest_repo_and_tag_when_present(self):
        checksums = self._checksums(repo = "unslothai/llama.cpp", release_tag = self.INSTALL_TAG)
        url = INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
            checksums, "unslothai/llama.cpp", self._artifact(repo = None), True, "ignored-tag"
        )
        assert url == self._expected("unslothai/llama.cpp", self.INSTALL_TAG)

    def test_falls_back_to_install_tag_when_manifest_tag_missing(self):
        # Regression: an empty manifest release_tag must not drop the asset URL.
        # Before the fix this returned None and hydration 404'd on the merge commit.
        checksums = self._checksums(repo = "unslothai/llama.cpp", release_tag = "")
        url = INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
            checksums, "unslothai/llama.cpp", self._artifact(repo = None), True, self.INSTALL_TAG
        )
        assert url == self._expected("unslothai/llama.cpp", self.INSTALL_TAG)

    def test_falls_back_to_source_repo_when_manifest_repo_missing(self):
        checksums = self._checksums(repo = "", release_tag = self.INSTALL_TAG)
        url = INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
            checksums, "unslothai/llama.cpp", self._artifact(repo = None), True, self.INSTALL_TAG
        )
        assert url == self._expected("unslothai/llama.cpp", self.INSTALL_TAG)

    def test_prefers_artifact_repo_over_manifest_repo(self):
        checksums = self._checksums(repo = "unslothai/checksums-only", release_tag = self.INSTALL_TAG)
        url = INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
            checksums,
            "unslothai/llama.cpp",
            self._artifact(repo = "unslothai/llama.cpp"),
            True,
            self.INSTALL_TAG,
        )
        assert url == self._expected("unslothai/llama.cpp", self.INSTALL_TAG)

    def test_returns_none_for_non_exact_source(self):
        checksums = self._checksums(repo = "unslothai/llama.cpp", release_tag = self.INSTALL_TAG)
        assert (
            INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
                checksums, UPSTREAM_REPO, None, False, self.INSTALL_TAG
            )
            is None
        )

    def test_returns_none_without_source_archive(self):
        checksums = self._checksums(repo = "unslothai/llama.cpp", release_tag = self.INSTALL_TAG)
        assert (
            INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
                checksums, "unslothai/llama.cpp", None, True, self.INSTALL_TAG
            )
            is None
        )

    def test_resolves_through_real_parser_chain(self):
        # The cases above hand-build ApprovedReleaseChecksums; this exercises the
        # production path they bypass: parse_approved_release_checksums ->
        # preferred_source_archive -> exact_source_asset_url, so a regression in the
        # parser/selection wiring can't pass while only the helper unit tests stay green.
        payload = {
            "schema_version": 1,
            "component": "llama.cpp",
            "release_tag": self.INSTALL_TAG,
            "upstream_tag": "b9616",
            "source_repo": "unslothai/llama.cpp",
            "source_commit": self.COMMIT,
            "artifacts": {
                exact_source_archive_logical_name(self.COMMIT): {
                    "sha256": "c" * 64,
                    "kind": "exact-source",
                },
            },
        }
        checksums = INSTALL_LLAMA_PREBUILT.parse_approved_release_checksums(
            "unslothai/llama.cpp", self.INSTALL_TAG, payload
        )
        source_repo, _source_ref, source_archive, exact_source = (
            INSTALL_LLAMA_PREBUILT.preferred_source_archive(checksums, "b9616")
        )
        assert exact_source is True
        assert source_archive is not None
        url = INSTALL_LLAMA_PREBUILT.exact_source_asset_url(
            checksums, source_repo, source_archive, exact_source, self.INSTALL_TAG
        )
        assert url == self._expected("unslothai/llama.cpp", self.INSTALL_TAG)
