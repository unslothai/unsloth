# SPDX-License-Identifier: AGPL-3.0-only
# Logic tests for studio/install_whisper_prebuilt.py -- the prebuilt whisper-server installer.
# No network/GPU: release resolution is injected and archives are built on disk in a tmp dir.

import importlib.util
import io
import json
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_whisper_prebuilt.py"
# The installer imports install_llama_prebuilt (same directory); make studio/
# importable so that resolves under spec-based loading.
_STUDIO_DIR = str(MODULE_PATH.parent)
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)
SPEC = importlib.util.spec_from_file_location("studio_install_whisper_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

HostInfo = M.HostInfo
PrebuiltFallback = M.PrebuiltFallback
BusyInstallConflict = M.BusyInstallConflict

RELEASE_TAG = "v1.9.1-unsloth.1"
UPSTREAM_TAG = "v1.9.1"
SOURCE_COMMIT = "0" * 40
STUDIO_PROTOCOL = "inference/multipart-v1"


def _host(
    whisper_os: str,
    whisper_arch: str,
    *,
    has_usable_nvidia: bool = False,
    has_rocm: bool = False,
    rocm_gfx: str | None = None,
    compute_caps: tuple[str, ...] = (),
    driver_cuda_version: tuple[int, int] | None = None,
    torch_runtime_line: str | None = None,
    macos_version: tuple[int, int] | None = None,
) -> HostInfo:
    ext = ".zip" if whisper_os == "windows" else ".tar.gz"
    return HostInfo(
        system = {"linux": "Linux", "macos": "Darwin", "windows": "Windows"}[whisper_os],
        machine = whisper_arch,
        whisper_os = whisper_os,
        whisper_arch = whisper_arch,
        archive_ext = ext,
        is_windows = whisper_os == "windows",
        is_macos = whisper_os == "macos",
        is_apple_silicon = whisper_os == "macos" and whisper_arch == "arm64",
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
        rocm_gfx = rocm_gfx,
        compute_caps = compute_caps,
        driver_cuda_version = driver_cuda_version,
        torch_runtime_line = torch_runtime_line,
        macos_version = macos_version,
    )


def _artifact(os_: str, arch: str, backend: str, asset: str, sha256: str, **extra) -> dict:
    art = {
        "os": os_,
        "arch": arch,
        "backend": backend,
        "asset": asset,
        "sha256": sha256,
        "runtime_line": extra.get("runtime_line"),
    }
    art.update(extra)
    return art


def _manifest(
    artifacts: list[dict],
    *,
    component: str = "whisper.cpp",
    schema_version: int = 1,
) -> dict:
    return {
        "schema_version": schema_version,
        "component": component,
        "studio_protocol": STUDIO_PROTOCOL,
        "upstream_tag": UPSTREAM_TAG,
        "source_commit": SOURCE_COMMIT,
        "artifacts": artifacts,
    }


@pytest.fixture(autouse = True)
def _default_detected_runtime(monkeypatch):
    """CUDA selection intersects the driver lines with an on-disk runtime scan.
    Default that scan to "both majors present" so the SM/driver matrix is
    deterministic on any test host; on-disk-gating tests override it."""
    monkeypatch.setattr(
        M,
        "detected_cuda_runtime_lines",
        lambda *, is_windows: ["cuda13", "cuda12"],
    )


# ── Host detection (probes come from install_llama_prebuilt.detect_host) ──
def _llama_host(
    system: str,
    machine: str,
    *,
    has_usable_nvidia: bool = False,
    compute_caps: list[str] | None = None,
    driver_cuda_version: tuple[int, int] | None = None,
    has_rocm: bool = False,
    rocm_gfx: str | None = None,
    macos_version: tuple[int, int] | None = None,
):
    """A fake install_llama_prebuilt HostInfo, as llama_detect_host would return."""
    lowered = machine.lower()
    return M.llama.HostInfo(
        system = system,
        machine = machine,
        is_windows = system == "Windows",
        is_linux = system == "Linux",
        is_macos = system == "Darwin",
        is_x86_64 = lowered in {"x86_64", "amd64"},
        is_arm64 = lowered in {"arm64", "aarch64"},
        nvidia_smi = None,
        driver_cuda_version = driver_cuda_version,
        compute_caps = list(compute_caps or []),
        visible_cuda_devices = None,
        has_physical_nvidia = has_usable_nvidia,
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
        rocm_gfx_target = rocm_gfx,
        macos_version = macos_version,
    )


@pytest.mark.parametrize(
    "system,machine,exp_os,exp_arch,exp_ext",
    [
        ("Linux", "x86_64", "linux", "x64", ".tar.gz"),
        ("Linux", "aarch64", "linux", "arm64", ".tar.gz"),
        ("Darwin", "x86_64", "macos", "x64", ".tar.gz"),
        ("Darwin", "arm64", "macos", "arm64", ".tar.gz"),
        ("Windows", "AMD64", "windows", "x64", ".zip"),
        ("Windows", "ARM64", "windows", "arm64", ".zip"),
    ],
)
def test_detect_host(monkeypatch, system, machine, exp_os, exp_arch, exp_ext):
    monkeypatch.setattr(M, "llama_detect_host", lambda: _llama_host(system, machine))
    host = M.detect_host()
    assert (host.whisper_os, host.whisper_arch, host.archive_ext) == (exp_os, exp_arch, exp_ext)
    assert host.is_windows == (exp_os == "windows")
    assert host.is_apple_silicon == (exp_os == "macos" and exp_arch == "arm64")


def test_detect_host_populates_nvidia_caps(monkeypatch):
    monkeypatch.setattr(
        M,
        "llama_detect_host",
        lambda: _llama_host(
            "Linux",
            "x86_64",
            has_usable_nvidia = True,
            compute_caps = ["100"],
            driver_cuda_version = (13, 0),
        ),
    )
    monkeypatch.setattr(
        M,
        "detect_torch_cuda_runtime_preference",
        lambda base: types.SimpleNamespace(runtime_line = "cuda13"),
    )
    host = M.detect_host()
    assert host.has_usable_nvidia is True
    assert host.compute_caps == ("100",)
    assert host.driver_cuda_version == (13, 0)
    assert host.torch_runtime_line == "cuda13"


def test_detect_host_maps_rocm_fields(monkeypatch):
    monkeypatch.setattr(
        M,
        "llama_detect_host",
        lambda: _llama_host("Linux", "x86_64", has_rocm = True, rocm_gfx = "gfx1100"),
    )
    host = M.detect_host()
    assert host.has_rocm is True
    assert host.rocm_gfx == "gfx1100"
    assert host.has_usable_nvidia is False


@pytest.mark.parametrize(
    "system,machine",
    [("Plan9", "x86_64"), ("Linux", "sparc64"), ("Linux", "armv7l")],
)
def test_detect_host_unsupported(monkeypatch, system, machine):
    monkeypatch.setattr(M, "llama_detect_host", lambda: _llama_host(system, machine))
    with pytest.raises(PrebuiltFallback):
        M.detect_host()


# ── Backend override + auto-detect ──
def test_apply_host_overrides_clears_cuda_fields():
    base = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
        torch_runtime_line = "cuda13",
    )
    rocm = M.apply_host_overrides(base, has_rocm = True, rocm_gfx = "gfx1030")
    assert rocm.has_rocm and rocm.rocm_gfx == "gfx1030" and not rocm.has_usable_nvidia
    # A forced-ROCm host must not carry stray CUDA detection into selection.
    assert rocm.compute_caps == () and rocm.driver_cuda_version is None
    forced = M.apply_host_overrides(base, force_cpu = True)
    assert not forced.has_usable_nvidia and not forced.has_rocm and not forced.is_apple_silicon
    assert forced.compute_caps == () and forced.driver_cuda_version is None


# ── Asset naming (pure) ──
def test_whisper_asset_name():
    assert (
        M.whisper_asset_name(RELEASE_TAG, _host("linux", "x64"), "cpu")
        == "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"
    )
    assert (
        M.whisper_asset_name(RELEASE_TAG, _host("macos", "arm64"), "metal")
        == "whisper-v1.9.1-unsloth.1-macos-arm64-metal.tar.gz"
    )
    assert (
        M.whisper_asset_name(RELEASE_TAG, _host("windows", "x64"), "cuda12")
        == "whisper-v1.9.1-unsloth.1-windows-x64-cuda12.zip"
    )


# ── Install layout ──
def test_whisper_server_path_layout():
    nix = _host("linux", "x64")
    win = _host("windows", "x64")
    assert M.whisper_server_path(Path("/w"), nix) == Path("/w/build/bin/whisper-server")
    assert M.whisper_server_path(Path("/w"), win) == Path("/w/build/bin/Release/whisper-server.exe")


# ── Manifest parse + basic selection (wiring pin; the rejection matrix, the
# CUDA/ROCm coverage matrices, extraction guards, resolver payload shape and
# macOS min_os gating are asserted against the real whisper descriptor in
# tests/studio/install/test_prebuilt_core.py) ──
def test_parse_manifest_ok_and_basic_selection():
    cpu_asset = "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"
    manifest = M.parse_manifest(_manifest([_artifact("linux", "x64", "cpu", cpu_asset, "a" * 64)]))
    assert manifest["component"] == "whisper.cpp"
    assert manifest["studio_protocol"] == STUDIO_PROTOCOL
    host = _host("linux", "x64")
    # CPU/Metal need no per-GPU coverage: first os/arch/backend match / None.
    assert M.select_artifact(manifest, host, "cpu")["asset"] == cpu_asset
    assert M.select_artifact(manifest, host, "metal") is None


# ── Coverage-aware CUDA selection (shared with install_llama_prebuilt.py) ──
# The seven real linux-x64 CUDA profiles + their SM coverage, from the live
# release manifest. select_artifact must match the host's compute caps + driver.
_CUDA_PROFILES = {
    "cuda12-legacy": ("cuda12", "legacy", ["50", "52", "60", "61"]),
    "cuda12-older": ("cuda12", "older", ["70", "75", "80", "86", "89"]),
    "cuda12-newer": ("cuda12", "newer", ["86", "89", "90", "100", "103", "120"]),
    "cuda12-portable": (
        "cuda12",
        "portable",
        ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
    ),
    "cuda13-older": ("cuda13", "older", ["75", "80", "86", "89"]),
    "cuda13-newer": ("cuda13", "newer", ["86", "89", "90", "100", "103", "120"]),
    "cuda13-portable": ("cuda13", "portable", ["75", "80", "86", "89", "90", "100", "103", "120"]),
}


def _cuda_manifest() -> dict:
    artifacts = [_artifact("linux", "x64", "cpu", "whisper-linux-x64-cpu.tar.gz", "a" * 64)]
    for name, (line, cls, sms) in _CUDA_PROFILES.items():
        artifacts.append(
            _artifact(
                "linux",
                "x64",
                "cuda",
                f"whisper-linux-x64-{name}.tar.gz",
                "b" * 64,
                runtime_line = line,
                coverage_class = cls,
                supported_sms = sms,
                min_sm = int(sms[0]),
                max_sm = int(sms[-1]),
            )
        )
    return M.parse_manifest(_manifest(artifacts))


# ── Traversal-safe extraction ──
def _add_file(
    tar: tarfile.TarFile,
    name: str,
    data: bytes,
    mode: int = 0o644,
):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tar.addfile(info, io.BytesIO(data))


def test_extract_tar_gz_preserves_exec_and_libs(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, "whisper-server", b"#!/bin/sh\necho ok\n", mode = 0o755)
        _add_file(tar, "libwhisper.so", b"ELF-ish")
    dest = tmp_path / "out"
    M.extract_archive(archive, dest)
    assert (dest / "whisper-server").stat().st_mode & 0o111
    assert (dest / "libwhisper.so").is_file()


# ── Fixture archive + full staging/activate install ──
def _build_cpu_bundle(tmp_path: Path, host: HostInfo) -> tuple[Path, str, str]:
    """Build a fake CPU bundle archive with a dummy server + lib; return (path, name, sha256)."""
    asset = M.whisper_asset_name(RELEASE_TAG, host, "cpu")
    archive = tmp_path / asset
    server_name = M.server_binary_name(host)
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, server_name, b"#!/bin/sh\necho whisper\n", mode = 0o755)
        _add_file(tar, "libwhisper.so", b"dummy-libwhisper")
        _add_file(tar, "libggml-base.so", b"dummy-libggml")
    return archive, asset, M.sha256_file(archive)


def _install_env(
    monkeypatch,
    tmp_path,
    host,
    *,
    asset,
    sha256,
    backend_in_manifest = "cpu",
):
    """Wire monkeypatches so install_prebuilt runs fully offline against a local
    archive, injecting the release bundle + checksum index (the checksum trust
    model) via the single fetch_release_for_install seam."""
    manifest = M.parse_manifest(
        _manifest(
            [_artifact(host.whisper_os, host.whisper_arch, backend_in_manifest, asset, sha256)]
        )
    )
    bundle = M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {asset: f"https://example.invalid/{asset}"},
    )
    checksums = {asset: sha256}
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (bundle, checksums),
    )
    return bundle


def test_install_produces_colocated_layout_and_marker(tmp_path, monkeypatch):
    host = _host("linux", "x64")
    archive, asset, sha256 = _build_cpu_bundle(tmp_path, host)

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = sha256)

    install_dir = tmp_path / "whisper.cpp"
    rc = M.install_prebuilt(install_dir, backend = "cpu")
    assert rc == M.EXIT_SUCCESS

    server = install_dir / "build" / "bin" / "whisper-server"
    assert server.is_file()
    assert server.stat().st_mode & 0o111  # +x set on unix
    # every shared lib from the archive is co-located next to the server
    assert (install_dir / "build" / "bin" / "libwhisper.so").is_file()
    assert (install_dir / "build" / "bin" / "libggml-base.so").is_file()
    # marker written at the component root
    marker = json.loads((install_dir / M.METADATA_FILENAME).read_text())
    assert marker["component"] == "whisper.cpp"
    assert marker["release_tag"] == RELEASE_TAG
    assert marker["backend"] == "cpu"
    assert marker["studio_protocol"] == STUDIO_PROTOCOL
    assert marker["install_fingerprint"]


def test_install_is_idempotent(tmp_path, monkeypatch):
    host = _host("linux", "x64")
    archive, asset, sha256 = _build_cpu_bundle(tmp_path, host)

    calls = {"n": 0}

    def fake_download(url, destination):
        calls["n"] += 1
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = sha256)

    install_dir = tmp_path / "whisper.cpp"
    assert M.install_prebuilt(install_dir, backend = "cpu") == M.EXIT_SUCCESS
    assert calls["n"] == 1
    # Second run: marker + binary already match -> "already matches", no download.
    assert M.install_prebuilt(install_dir, backend = "cpu") == M.EXIT_SUCCESS
    assert calls["n"] == 1


def test_install_sha_mismatch_then_retry_fails_closed(tmp_path, monkeypatch):
    host = _host("linux", "x64")
    archive, asset, _real_sha = _build_cpu_bundle(tmp_path, host)
    wrong_sha = "f" * 64  # the checksum index claims a different sha than the archive

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = wrong_sha)

    install_dir = tmp_path / "whisper.cpp"
    with pytest.raises(PrebuiltFallback):
        M.install_prebuilt(install_dir, backend = "cpu")
    # A failed verify never activates a binary.
    assert not (install_dir / "build" / "bin" / "whisper-server").exists()


def test_install_cpu_fallback_when_accel_absent(tmp_path, monkeypatch):
    host = _host("linux", "x64", has_usable_nvidia = True, driver_cuda_version = (13, 0))
    archive, asset, sha256 = _build_cpu_bundle(tmp_path, host)

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    # Manifest only carries a cpu artifact; auto-detect wants cuda -> falls back to cpu.
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = sha256, backend_in_manifest = "cpu")

    install_dir = tmp_path / "whisper.cpp"
    assert M.install_prebuilt(install_dir, backend = "auto") == M.EXIT_SUCCESS
    marker = json.loads((install_dir / M.METADATA_FILENAME).read_text())
    assert marker["backend"] == "cpu"


# ── Busy lock -> exit 3 ──
def test_busy_lock_maps_to_exit_busy(tmp_path, monkeypatch):
    host = _host("linux", "x64")
    archive, asset, sha256 = _build_cpu_bundle(tmp_path, host)
    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", lambda u, d: None)
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = sha256)

    from contextlib import contextmanager

    @contextmanager
    def busy_lock(_path):
        raise BusyInstallConflict("held by another process")
        yield  # pragma: no cover

    monkeypatch.setattr(M, "install_lock", busy_lock)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_BUSY


# ── Resolver JSON shape ──
def test_resolve_mode_keeps_stdout_json_only(monkeypatch, capsys):
    # Even in a CUDA host where selection emits diagnostics, --resolve-prebuilt
    # must keep stdout to exactly the JSON line (setup.sh / whisper_cpp_update.py
    # parse it); the cuda_selection log noise belongs on stderr.
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    manifest = _cuda_manifest()
    bundle = M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {
            a["asset"]: f"https://example.invalid/{a['asset']}" for a in manifest["artifacts"]
        },
    )
    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(
        M, "fetch_release_for_install", lambda repo, *, published_release_tag = None: (bundle, {})
    )
    # A prior install test may have left the module flag True; the resolver must
    # force it back to stderr regardless.
    monkeypatch.setattr(M, "_LOG_TO_STDOUT", True, raising = False)

    rc = M.main(["--resolve-prebuilt", "--output-format", "json"])
    assert rc == M.EXIT_SUCCESS
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())  # exactly one JSON line, parseable
    assert payload["asset"] == "whisper-linux-x64-cuda13-newer.tar.gz"
    assert "[whisper-prebuilt]" not in captured.out  # no log noise on stdout
    assert "cuda_selection:" in captured.err  # diagnostics routed to stderr


def test_main_maps_prebuilt_fallback_to_exit_fallback(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise PrebuiltFallback("no prebuilt")

    monkeypatch.setattr(M, "install_prebuilt", boom)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_FALLBACK


def test_main_maps_unexpected_error_to_exit_error(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(M, "install_prebuilt", boom)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_ERROR


def test_resolve_mode_unexpected_error_reports_unavailable(monkeypatch, capsys):
    # An unexpected failure inside the probe maps to prebuilt_available=False, not
    # a traceback, so the caller falls back cleanly.
    host = _host("linux", "x64")
    monkeypatch.setattr(M, "detect_host", lambda: host)

    def boom(repo, *, published_release_tag = None):
        raise RuntimeError("network down")

    monkeypatch.setattr(M, "fetch_release_for_install", boom)
    rc = M.main(["--resolve-prebuilt", "--output-format", "json"])
    assert rc == M.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"prebuilt_available": False, "repo": "unslothai/whisper.cpp"}


# ── --rocm-gfx / --has-rocm overrides (llama parity) ──
def test_rocm_gfx_override_implies_has_rocm():
    # --rocm-gfx alone (no --has-rocm) must enable ROCm and clear NVIDIA, else the
    # host stays on its CUDA/CPU path and never picks the ROCm bundle.
    base = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
        torch_runtime_line = "cuda13",
    )
    out = M.apply_host_overrides(base, rocm_gfx = "gfx1100")
    assert out.has_rocm is True
    assert out.rocm_gfx == "gfx1100"
    assert out.has_usable_nvidia is False
    assert out.compute_caps == () and out.driver_cuda_version is None
    assert M.auto_detect_backend(out) == "rocm"


# ── On-disk CUDA runtime detection (exact SONAMEs over llama's dir scan) ──
def _isolate_runtime_dirs(monkeypatch):
    """Neutralise every runtime-dir source except CUDA_RUNTIME_LIB_DIR so the scan
    sees only the test's temp dir (not the host's real CUDA install). The dir
    sources are llama module globals resolved at call time."""
    monkeypatch.setattr(M.llama, "python_runtime_dirs", lambda: [])
    monkeypatch.setattr(M.llama, "ldconfig_runtime_dirs", lambda required: [])
    monkeypatch.setattr(M.llama, "glob_paths", lambda *patterns: [])
    for var in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        monkeypatch.delenv(var, raising = False)


def test_detected_linux_runtime_lines_matches_only_present_major(tmp_path, monkeypatch):
    libdir = tmp_path / "cuda13"
    libdir.mkdir()
    (libdir / "libcudart.so.13").write_text("x")
    (libdir / "libcublas.so.13").write_text("x")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate_runtime_dirs(monkeypatch)
    assert M.detected_linux_runtime_lines() == ["cuda13"]  # not cuda12/14/...


def test_detected_linux_runtime_lines_requires_both_libs(tmp_path, monkeypatch):
    # libcudart present but libcublas missing -> the line is NOT usable.
    libdir = tmp_path / "partial"
    libdir.mkdir()
    (libdir / "libcudart.so.13").write_text("x")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate_runtime_dirs(monkeypatch)
    assert M.detected_linux_runtime_lines() == []


def test_detected_linux_runtime_lines_requires_soname_not_versioned_only(tmp_path, monkeypatch):
    # Only the fully-versioned files (libcudart.so.13.0.88) with no SONAME symlink
    # (libcudart.so.13): the dynamic linker resolves the SONAME, so this is NOT
    # loadable and must not be reported. A `{lib}*` glob would wrongly match here.
    libdir = tmp_path / "versioned_only"
    libdir.mkdir()
    (libdir / "libcudart.so.13.0.88").write_text("x")
    (libdir / "libcublas.so.13.0.88").write_text("x")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate_runtime_dirs(monkeypatch)
    assert M.detected_linux_runtime_lines() == []


def test_detected_linux_runtime_lines_accepts_soname_symlink(tmp_path, monkeypatch):
    # SONAME symlink present (points at the versioned file) -> loadable -> detected.
    libdir = tmp_path / "with_symlink"
    libdir.mkdir()
    (libdir / "libcudart.so.13.0.88").write_text("x")
    (libdir / "libcublas.so.13.0.88").write_text("x")
    (libdir / "libcudart.so.13").symlink_to(libdir / "libcudart.so.13.0.88")
    (libdir / "libcublas.so.13").symlink_to(libdir / "libcublas.so.13.0.88")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate_runtime_dirs(monkeypatch)
    assert M.detected_linux_runtime_lines() == ["cuda13"]


def test_detected_linux_runtime_lines_empty_when_nothing_present(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(empty))
    _isolate_runtime_dirs(monkeypatch)
    assert M.detected_linux_runtime_lines() == []
