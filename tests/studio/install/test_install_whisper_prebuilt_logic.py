# SPDX-License-Identifier: AGPL-3.0-only
# Logic tests for studio/install_whisper_prebuilt.py -- the prebuilt whisper-server installer.
# No network/GPU: release resolution is injected and archives are built on disk in a tmp dir.

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_whisper_prebuilt.py"
# The installer imports install_llama_prebuilt (same directory);
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
ReleaseCompatibilityError = M.ReleaseCompatibilityError
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


def test_whisper_server_path_layout():
    nix = _host("linux", "x64")
    win = _host("windows", "x64")
    assert M.whisper_server_path(Path("/w"), nix) == Path("/w/build/bin/whisper-server")
    assert M.whisper_server_path(Path("/w"), win) == Path("/w/build/bin/Release/whisper-server.exe")


def test_parse_manifest_ok_and_basic_selection():
    cpu_asset = "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"
    manifest = M.parse_manifest(_manifest([_artifact("linux", "x64", "cpu", cpu_asset, "a" * 64)]))
    assert manifest["component"] == "whisper.cpp"
    assert manifest["studio_protocol"] == STUDIO_PROTOCOL
    host = _host("linux", "x64")
    # The pinned pre-slim escape hatch:
    assert M.select_artifact(manifest, host, "cpu")["asset"] == cpu_asset
    assert M.select_artifact(manifest, host, "metal") is None


def test_select_artifact_never_picks_fat_gpu_bundles():
    # backend selects nothing (the core then retries with cpu), never the fat
    # A pinned pre-slim release's fat GPU bundles are dead shapes:
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact("linux", "x64", "cpu", "whisper-linux-x64-cpu.tar.gz", "a" * 64),
                _artifact("linux", "x64", "cuda", "whisper-linux-x64-cuda13.tar.gz", "b" * 64),
                _artifact("linux", "x64", "rocm", "whisper-linux-x64-rocm.tar.gz", "c" * 64),
            ]
        )
    )
    cuda_host = _host("linux", "x64", has_usable_nvidia = True)
    assert M.select_artifact(manifest, cuda_host, "cuda") is None
    rocm_host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1100")
    assert M.select_artifact(manifest, rocm_host, "rocm") is None
    artifact, backend, used_fallback = M.select_artifact_with_fallback(manifest, cuda_host, "cuda")
    assert artifact["asset"] == "whisper-linux-x64-cpu.tar.gz"
    assert backend == "cpu" and used_fallback is True


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
    if sys.platform != "win32":
        assert server.stat().st_mode & 0o111
    assert (install_dir / "build" / "bin" / "libwhisper.so").is_file()
    assert (install_dir / "build" / "bin" / "libggml-base.so").is_file()
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


def test_resolve_mode_keeps_stdout_json_only(tmp_path, monkeypatch, capsys):
    # Even when the slim pairing emits diagnostics, --resolve-prebuilt must keep stdout to exactly the JSON line
    # (setup.sh / whisper_cpp_update.py parse it);
    # the slim_selection log noise belongs on stderr.
    host = _cuda_host()
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")
    )
    manifest = _slim_manifest()
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
    # A prior install test may have left the module flag True;
    monkeypatch.setattr(M, "_LOG_TO_STDOUT", True, raising = False)

    rc = M.main(["--resolve-prebuilt", "--output-format", "json"])
    assert rc == M.EXIT_SUCCESS
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["asset"] == SLIM_ASSET
    assert "[whisper-prebuilt]" not in captured.out
    assert "slim_selection:" in captured.err


def test_main_maps_prebuilt_fallback_to_exit_error(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise PrebuiltFallback("no prebuilt")

    monkeypatch.setattr(M, "install_prebuilt", boom)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_ERROR


def test_main_forwards_requested_whisper_tags(tmp_path, monkeypatch):
    seen = {}

    def install(*args, **kwargs):
        seen.update(kwargs)
        return M.EXIT_SUCCESS

    monkeypatch.setattr(M, "install_prebuilt", install)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--whisper-tag", "v1.9.0"])
    assert rc == M.EXIT_SUCCESS
    assert seen["whisper_tag"] == "v1.9.0"

    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))
    monkeypatch.setattr(
        M,
        "resolve_prebuilt",
        lambda host, **kwargs: (
            seen.update(kwargs) or {"prebuilt_available": False, "repo": "unslothai/whisper.cpp"}
        ),
    )
    assert M.main(["--resolve-prebuilt", "v1.8.0", "--output-format", "json"]) == 0
    assert seen["whisper_tag"] == "v1.8.0"


def test_main_reserves_exit_2_for_release_incompatibility(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise ReleaseCompatibilityError("slim bundle requires llama.cpp b2; installed b1")

    monkeypatch.setattr(M, "install_prebuilt", boom)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_INCOMPATIBLE


def test_main_maps_unexpected_error_to_exit_error(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(M, "install_prebuilt", boom)
    rc = M.main(["--install-dir", str(tmp_path / "whisper.cpp"), "--backend", "cpu"])
    assert rc == M.EXIT_ERROR


def test_resolve_mode_unexpected_error_reports_unavailable(monkeypatch, capsys):
    # An unexpected failure inside the probe maps to prebuilt_available=False, not a traceback, so the caller falls
    host = _host("linux", "x64")
    monkeypatch.setattr(M, "detect_host", lambda: host)

    def boom(repo, *, published_release_tag = None):
        raise RuntimeError("network down")

    monkeypatch.setattr(M, "fetch_release_for_install", boom)
    rc = M.main(["--resolve-prebuilt", "--output-format", "json"])
    assert rc == M.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {
        "prebuilt_available": False,
        "repo": "unslothai/whisper.cpp",
        "unavailable_reason": "unresolved",
    }


def test_resolve_mode_separates_incompatible_from_unresolved(monkeypatch, capsys):
    """The probe's negative answer must say WHY, as the install path already does
    (exit 2 against exit 1). Flattened, a network blip reads as a pairing gap."""
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))

    def incompatible(repo, *, published_release_tag = None):
        raise ReleaseCompatibilityError("slim bundle requires llama.cpp b2; installed b1")

    monkeypatch.setattr(M, "fetch_release_for_install", incompatible)
    assert M.main(["--resolve-prebuilt", "--output-format", "json"]) == M.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["prebuilt_available"] is False
    assert payload["unavailable_reason"] == "incompatible"


def test_resolve_mode_reports_an_ordinary_fallback_as_unresolved(monkeypatch, capsys):
    """An unsupported platform, a malformed manifest, a checksum failure:
    _slim_release_incompatibility excludes those, so they are not a pairing gap."""
    monkeypatch.setattr(M, "detect_host", lambda: _host("linux", "x64"))

    def fallback(repo, *, published_release_tag = None):
        raise M.PrebuiltFallback("manifest omitted an 'artifacts' list")

    monkeypatch.setattr(M, "fetch_release_for_install", fallback)
    assert M.main(["--resolve-prebuilt", "--output-format", "json"]) == M.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["unavailable_reason"] == "unresolved"


def test_rocm_gfx_override_implies_has_rocm():
    # host stays on its CUDA/CPU path and never picks the ROCm bundle.
    # rocm-gfx alone (no --has-rocm) must enable ROCm and clear NVIDIA, else the host stays on its CUDA/CPU path and
    base = _host("linux", "x64", has_usable_nvidia = True)
    out = M.apply_host_overrides(base, rocm_gfx = "gfx1100")
    assert out.has_rocm is True
    assert out.rocm_gfx == "gfx1100"
    assert out.has_usable_nvidia is False
    assert M.auto_detect_backend(out) == "rocm"


SLIM_LLAMA_TAG = "b10069-mix-fb3d4ca"
SLIM_ASSET = "whisper-v1.9.1-unsloth.1-linux-x64-slim.tar.gz"
CPU_ASSET = "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"


def _fake_llama_bin(
    tmp_path: Path,
    *,
    backend_module: str | None = "libggml-cuda.so",
    sonames: tuple[str, ...] = ("libggml.so.0", "libggml-base.so.0"),
) -> Path:
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    for name in sonames:
        (bin_dir / name).write_bytes(b"ggml-old-" + name.encode())
    if backend_module:
        (bin_dir / backend_module).write_bytes(b"ggml-backend-old")
    (bin_dir / "libggml-cpu-x64.so").write_bytes(b"ggml-cpu-old")
    (bin_dir / "libllama.so").write_bytes(b"not-ggml")  # must never be linked
    return bin_dir


def _slim_artifact(**extra) -> dict:
    art = {
        "os": "linux",
        "arch": "x64",
        "backend": "slim",
        "asset": SLIM_ASSET,
        "sha256": "c" * 64,
        "install_kind": "slim",
        "requires_llama_tag": SLIM_LLAMA_TAG,
        "requires_ggml_version": "0.17.0",
        "requires_ggml_sonames": ["libggml.so.0", "libggml-base.so.0"],
        "min_os": None,
    }
    art.update(extra)
    return art


def _slim_manifest(slim_extra: dict | None = None) -> dict:
    """The transition-release shape: a slim bundle beside the published fat CPU
    bundle (the pinned pre-slim escape hatch)."""
    artifacts = [
        _artifact("linux", "x64", "cpu", CPU_ASSET, "a" * 64),
        _slim_artifact(**(slim_extra or {})),
    ]
    return M.parse_manifest(_manifest(artifacts))


def _cuda_host() -> HostInfo:
    return _host("linux", "x64", has_usable_nvidia = True)


def test_installed_llama_runtime_reads_marker(tmp_path):
    root = tmp_path / "llama.cpp"
    bin_dir = root / "build" / "bin"
    if sys.platform == "win32":
        bin_dir = bin_dir / "Release"
    bin_dir.mkdir(parents = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": SLIM_LLAMA_TAG, "bundle_profile": "cuda13-newer"})
    )
    assert M.llama.installed_llama_runtime(root) == (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")


@pytest.mark.parametrize(
    "prepare",
    [
        lambda root: None,
        lambda root: (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{}"),
        lambda root: (root / "UNSLOTH_PREBUILT_INFO.json").write_text("not json"),
    ],
)
def test_installed_llama_runtime_rejects_incomplete_installs(tmp_path, prepare):
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents = True)
    prepare(root)
    assert M.llama.installed_llama_runtime(root) is None


def test_installed_llama_runtime_requires_bin_dir(tmp_path):
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps({"release_tag": SLIM_LLAMA_TAG}))
    assert M.llama.installed_llama_runtime(root) is None  # marker but no build/bin


def test_slim_selected_when_all_pairing_checks_pass(tmp_path, monkeypatch):
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")
    )
    artifact, backend, used_fallback = M.select_artifact_with_fallback(
        _slim_manifest(), _cuda_host(), "cuda"
    )
    assert artifact["asset"] == SLIM_ASSET
    assert backend == "cuda" and used_fallback is False


@pytest.mark.parametrize(
    "runtime,slim_extra",
    [
        (lambda bin_dir: None, None),
        # Installed llama tag does not match requires_llama_tag.
        (lambda bin_dir: (bin_dir, "b99999-mix-0000000", "cuda13-newer"), None),
        # A required soname is missing from the llama bin dir.
        (
            lambda bin_dir: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer"),
            {"requires_ggml_sonames": ["libggml.so.0", "libggml-base.so.0", "libggml-extra.so.9"]},
        ),
        # Manifest omits the soname contract entirely.
        (lambda bin_dir: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer"), {"requires_ggml_sonames": []}),
    ],
)
def test_slim_pairing_failure_falls_back_to_pinned_cpu(tmp_path, monkeypatch, runtime, slim_extra):
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: runtime(bin_dir))
    artifact, backend, used_fallback = M.select_artifact_with_fallback(
        _slim_manifest(slim_extra), _cuda_host(), "cuda"
    )
    assert artifact["asset"] == CPU_ASSET
    assert backend == "cpu" and used_fallback is True


def test_slim_missing_accel_module_rides_the_cpu_module(tmp_path, monkeypatch):
    # Sonames present but no libggml-cuda.so in the llama bin dir:
    bin_dir = _fake_llama_bin(tmp_path, backend_module = None)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")
    )
    artifact, backend, used_fallback = M.select_artifact_with_fallback(
        _slim_manifest(), _cuda_host(), "cuda"
    )
    assert artifact["asset"] == SLIM_ASSET
    assert backend == "cpu" and used_fallback is True


def test_slim_selected_for_cpu_backend_on_linux(tmp_path, monkeypatch):
    # With the fat per-accelerator chain gone, a failed pairing degrades to the one legacy shape:
    # Slim-only releases must serve cpu too:
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")
    )
    artifact, backend, _fb = M.select_artifact_with_fallback(
        _slim_manifest(), _host("linux", "x64"), "cpu"
    )
    assert artifact["asset"] == SLIM_ASSET and backend == "cpu"


def test_slim_cpu_requires_a_cpu_module(tmp_path, monkeypatch):
    # Sonames present but no libggml-cpu* variant in the llama bin dir -> fat cpu.
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents = True)
    for name in ("libggml.so.0", "libggml-base.so.0"):
        (bin_dir / name).write_bytes(b"ggml")
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "linux-cpu")
    )
    artifact, backend, _fb = M.select_artifact_with_fallback(
        _slim_manifest(), _host("linux", "x64"), "cpu"
    )
    assert artifact["asset"] == CPU_ASSET and backend == "cpu"


WIN_SLIM_ASSET = "whisper-v1.9.1-unsloth.1-windows-x64-slim.zip"


def _windows_slim_manifest(*, requires_ggml_sonames: list[str]) -> dict:
    return M.parse_manifest(
        _manifest(
            [
                _slim_artifact(
                    os = "windows",
                    arch = "x64",
                    asset = WIN_SLIM_ASSET,
                    requires_ggml_sonames = requires_ggml_sonames,
                )
            ]
        )
    )


def test_slim_selected_for_cpu_backend_on_windows(tmp_path, monkeypatch):
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin" / "Release"
    bin_dir.mkdir(parents = True)
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu-haswell.dll"):
        (bin_dir / name).write_bytes(b"ggml")
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "windows-cpu")
    )
    manifest = _windows_slim_manifest(requires_ggml_sonames = ["ggml.dll", "ggml-base.dll"])
    artifact, backend, _fb = M.select_artifact_with_fallback(
        manifest, _host("windows", "x64"), "cpu"
    )
    assert artifact["asset"] == WIN_SLIM_ASSET and backend == "cpu"


WIN_LIBOMP_SONAMES = ["ggml.dll", "ggml-base.dll", "libomp140.x86_64.dll"]

WIN_GPU_BUNDLES = [
    ("rocm", "ggml-hip.dll", {"has_rocm": True, "rocm_gfx": "gfx1150"}),
    ("cuda", "ggml-cuda.dll", {"has_usable_nvidia": True}),
    ("vulkan", "ggml-vulkan.dll", {}),
    # The cpu backend on a GPU bundle: the exemption keys on what the paired bin dir holds, not on the request, and that
    # bundle's ggml-cpu.dll has no libomp to find either, so this must pair outright rather than via a cpu retry.
    ("cpu", "ggml-hip.dll", {}),
]


@pytest.mark.parametrize("backend, module, host_kwargs", WIN_GPU_BUNDLES)
def test_windows_gpu_slim_does_not_require_cpu_only_libomp(
    tmp_path, monkeypatch, backend, module, host_kwargs
):
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin" / "Release"
    bin_dir.mkdir(parents = True)
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", module):
        (bin_dir / name).write_bytes(b"ggml")
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, ""))
    manifest = _windows_slim_manifest(requires_ggml_sonames = WIN_LIBOMP_SONAMES)

    artifact, chosen, used_fallback = M.select_artifact_with_fallback(
        manifest, _host("windows", "x64", **host_kwargs), backend
    )

    assert artifact["asset"] == WIN_SLIM_ASSET
    assert chosen == backend and used_fallback is False


# "" is the upstream-sourced install: install_llama_prebuilt builds those AssetChoices without a bundle_profile, so a
# real cpu bundle reports no profile.
@pytest.mark.parametrize("profile", ["windows-cpu-x64", ""])
def test_windows_cpu_bundle_still_requires_manifest_libomp(tmp_path, monkeypatch, profile):
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin" / "Release"
    bin_dir.mkdir(parents = True)
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu-haswell.dll"):
        (bin_dir / name).write_bytes(b"ggml")
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, profile))
    artifact = _windows_slim_manifest(requires_ggml_sonames = WIN_LIBOMP_SONAMES)["artifacts"][0]

    assert M.slim_pairing_for_artifact(artifact, _host("windows", "x64"), "cpu") is None

    (bin_dir / "libomp140.x86_64.dll").write_bytes(b"omp")
    assert M.slim_pairing_for_artifact(artifact, _host("windows", "x64"), "cpu") is not None


def test_linux_slim_still_requires_manifest_libomp(tmp_path, monkeypatch):
    # The exemption is Windows only:
    bin_dir = _fake_llama_bin(tmp_path, backend_module = "libggml-cuda.so")
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "linux-cuda")
    )
    artifact = _slim_artifact(
        requires_ggml_sonames = ["libggml.so.0", "libggml-base.so.0", "libomp.so.5"],
    )

    assert M.slim_pairing_for_artifact(artifact, _host("linux", "x64"), "cuda") is None


@pytest.mark.parametrize(
    "missing_name",
    ["ggml.dll", "ggml-base.dll", "ggml-hip.dll"],
)
def test_windows_rocm_slim_still_requires_ggml_runtime(tmp_path, monkeypatch, missing_name):
    # The shared Windows manifest lists the OpenMP runtime only the cpu llama bundle ships;
    # The empty bundle_profile is what the published rocm artifacts carry.
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin" / "Release"
    bin_dir.mkdir(parents = True)
    for name in {"ggml.dll", "ggml-base.dll", "ggml-hip.dll"} - {missing_name}:
        (bin_dir / name).write_bytes(b"ggml")
    # ggml backend module, so their presence must not stand in for it.
    # The HIP DLLs every published ROCm bundle also ships:
    for name in ("amdhip64_7.dll", "hipblas.dll", "libhipblaslt.dll"):
        (bin_dir / name).write_bytes(b"runtime")
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, ""))
    artifact = _windows_slim_manifest(
        requires_ggml_sonames = [
            "ggml.dll",
            "ggml-base.dll",
            "libomp140.x86_64.dll",
        ]
    )["artifacts"][0]

    assert (
        M.slim_pairing_for_artifact(
            artifact,
            _host("windows", "x64", has_rocm = True, rocm_gfx = "gfx1150"),
            "rocm",
        )
        is None
    )


def test_windows_rocm_slim_rejects_decoy_hip_dlls_alone(tmp_path, monkeypatch):
    # DLL must fail the pairing rather than install a whisper that cannot load.
    # Same loss, but with libomp present so the soname gate cannot do the
    # A cpu bundle's ggml really does import libomp, so a runtime that lost the DLL must fail the pairing rather than
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin" / "Release"
    bin_dir.mkdir(parents = True)
    for name in (
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "libomp140.x86_64.dll",
        "amdhip64_7.dll",
        "hipblas.dll",
        "libhipblaslt.dll",
    ):
        (bin_dir / name).write_bytes(b"x")
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, ""))
    artifact = _windows_slim_manifest(requires_ggml_sonames = WIN_LIBOMP_SONAMES)["artifacts"][0]

    assert (
        M.slim_pairing_for_artifact(
            artifact,
            _host("windows", "x64", has_rocm = True, rocm_gfx = "gfx1150"),
            "rocm",
        )
        is None
    )


MAC_SLIM_ASSET = "whisper-v1.9.1-unsloth.1-macos-arm64-slim.tar.gz"


def _fake_llama_bin_macos(tmp_path: Path) -> Path:
    """The dylib names the live llama macos bundle ships (core + cpu + metal)."""
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    for name in (
        "libggml.dylib",
        "libggml-base.dylib",
        "libggml-cpu.dylib",
        "libggml-metal.dylib",
        "libggml-blas.dylib",
    ):
        (bin_dir / name).write_bytes(b"ggml-" + name.encode())
    (bin_dir / "libllama.dylib").write_bytes(b"not-ggml")
    return bin_dir


def _macos_slim_manifest(arch: str = "arm64") -> dict:
    return M.parse_manifest(
        _manifest(
            [
                _slim_artifact(
                    os = "macos",
                    arch = arch,
                    asset = MAC_SLIM_ASSET,
                    requires_ggml_sonames = ["libggml.dylib", "libggml-base.dylib"],
                )
            ]
        )
    )


@pytest.mark.parametrize("backend", ["metal", "cpu"])
def test_slim_selected_for_metal_and_cpu_on_macos(tmp_path, monkeypatch, backend):
    bin_dir = _fake_llama_bin_macos(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "macos-metal-arm64")
    )
    host = _host("macos", "arm64", macos_version = (14, 0))
    artifact, effective, _fb = M.select_artifact_with_fallback(
        _macos_slim_manifest(), host, backend
    )
    assert artifact["asset"] == MAC_SLIM_ASSET
    assert effective == backend  # the accel identity, never "slim"


def test_macos_auto_backends_route_to_slim(tmp_path, monkeypatch):
    bin_dir = _fake_llama_bin_macos(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "macos-metal-arm64")
    )
    silicon = _host("macos", "arm64", macos_version = (14, 0))
    assert M.resolve_backend(silicon, "auto", cpu_fallback = False) == "metal"
    artifact, backend, _fb = M.select_artifact_with_fallback(
        _macos_slim_manifest(), silicon, "metal"
    )
    assert artifact["asset"] == MAC_SLIM_ASSET and backend == "metal"

    intel = _host("macos", "x64", macos_version = (14, 0))
    assert M.resolve_backend(intel, "auto", cpu_fallback = False) == "cpu"
    artifact, backend, _fb = M.select_artifact_with_fallback(
        _macos_slim_manifest(arch = "x64"), intel, "cpu"
    )
    assert artifact["asset"] == MAC_SLIM_ASSET and backend == "cpu"


def test_slim_metal_requires_the_metal_module(tmp_path, monkeypatch):
    # A macos llama runtime without libggml-metal*.dylib cannot back metal; the
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents = True)
    for name in ("libggml.dylib", "libggml-base.dylib", "libggml-cpu.dylib"):
        (bin_dir / name).write_bytes(b"ggml")
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "macos-cpu-x64")
    )
    host = _host("macos", "arm64", macos_version = (14, 0))
    artifact, backend, used_fallback = M.select_artifact_with_fallback(
        _macos_slim_manifest(), host, "metal"
    )
    assert artifact["asset"] == MAC_SLIM_ASSET
    assert backend == "cpu" and used_fallback is True


def test_slim_only_release_pairing_failure_is_actionable(tmp_path, monkeypatch):
    # A slim-only release with no llama install is an operational failure, not the narrowly handled installed-version
    lines: list[str] = []
    monkeypatch.setattr(M, "log", lines.append)
    monkeypatch.setattr(M, "installed_llama_runtime", lambda: None)
    manifest = M.parse_manifest(_manifest([_slim_artifact()]))
    with pytest.raises(PrebuiltFallback):
        M.select_artifact_with_fallback(manifest, _cuda_host(), "cuda")
    assert any(
        f"slim bundle requires llama.cpp {SLIM_LLAMA_TAG}; "
        "install or update llama.cpp first" in line
        for line in lines
    )


def test_slim_release_tag_skew_has_distinct_compatibility_error(tmp_path, monkeypatch):
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M,
        "installed_llama_runtime",
        lambda: (bin_dir, "b10068-mix-old", "cuda13-newer"),
    )
    manifest = M.parse_manifest(_manifest([_slim_artifact()]))
    with pytest.raises(ReleaseCompatibilityError, match = SLIM_LLAMA_TAG):
        M.select_artifact_with_fallback(manifest, _cuda_host(), "cuda")


# A newer llama build that keeps the same ggml commit as SLIM_LLAMA_TAG.
NEWER_LLAMA_TAG = "b10079-mix-fb3d4ca"


@pytest.mark.parametrize(
    "installed,required,pairs",
    [
        (SLIM_LLAMA_TAG, SLIM_LLAMA_TAG, True),
        (NEWER_LLAMA_TAG, SLIM_LLAMA_TAG, True),
        ("b10069-mix-0000000", SLIM_LLAMA_TAG, False),
        (SLIM_LLAMA_TAG, None, False),
        ("b10069", "b10069", True),
        ("b10070", "b10069", False),
    ],
)
def test_llama_runtime_pairs_falls_back_to_mix_suffix(installed, required, pairs):
    assert M.llama_runtime_pairs(installed, required) is pairs


LLAMA_REPO = "unslothai/llama.cpp"
# A publisher that is NOT the default, so a lookup that dropped installed_repo and fell back to unslothai/llama.cpp
FORK_REPO = "acme-fork/llama.cpp"


@pytest.fixture(autouse = True)
def _offline_published_tree_lookup(monkeypatch):
    """Keep llama_runtime_pairs offline and its per-tag cache test-local.

    A missing tree sends published_llama_ggml_tree to the llama release, so
    without this the pairing assertions below would depend on live release
    assets and on this machine's own llama marker. The cache is module level
    and outlives monkeypatch, so clear it too. Tests wanting the lookup stub
    _download_host_json_once themselves, which wins because this fixture is
    applied first."""
    M._PUBLISHED_GGML_TREE_CACHE.clear()

    def unreachable(url):
        raise OSError(f"offline test tried to fetch {url}")

    monkeypatch.setattr(M, "_download_host_json_once", unreachable)
    # No marker to read, so installed_llama_tree_repo() stays None unless a test points it somewhere;
    monkeypatch.setattr(M.llama, "default_managed_llama_dir", lambda: Path("/nonexistent-llama"))
    yield
    M._PUBLISHED_GGML_TREE_CACHE.clear()


# The autouse fixture above replaces _download_host_json_once with an offline guard, so a test exercising the real
_REAL_DOWNLOAD_HOST_JSON_ONCE = M._download_host_json_once


# Real releases: same -mix-2c8b9c1 suffix, but genuinely different ggml trees.
SUFFIX_SHARED_A = "b10173-mix-2c8b9c1"
SUFFIX_SHARED_B = "b10181-mix-2c8b9c1"
TREE_A = "8f3c6e197debb027f500df9f76e710e137f9fe68"
TREE_B = "e96ffb0e063f66952b0c54796a74755b6041c867"


@pytest.mark.parametrize(
    "installed,required,installed_tree,required_tree,pairs",
    [
        # The bug this fixes: a shared suffix is not a shared ggml.
        (SUFFIX_SHARED_A, SUFFIX_SHARED_B, TREE_A, TREE_B, False),
        # Different suffixes, same ggml:
        ("b10241-mix-89aa77b", "b10225-mix-345e1e3", TREE_A, TREE_A, True),
        # A missing tree falls back to the suffix only when the release does not publish one either;
        (SUFFIX_SHARED_A, SUFFIX_SHARED_B, None, TREE_B, True),
        (SUFFIX_SHARED_A, SUFFIX_SHARED_B, TREE_A, None, True),
        (SUFFIX_SHARED_A, SUFFIX_SHARED_B, "", "", True),
        # An exact tag pairs before trees are consulted.
        (SUFFIX_SHARED_A, SUFFIX_SHARED_A, TREE_A, TREE_B, True),
    ],
)
def test_llama_runtime_pairs_prefers_ggml_tree(
    installed, required, installed_tree, required_tree, pairs
):
    assert (
        M.llama_runtime_pairs(
            installed,
            required,
            installed_ggml_tree = installed_tree,
            required_ggml_tree = required_tree,
        )
        is pairs
    )


def test_llama_runtime_pairs_reads_the_published_tree_when_the_marker_has_none(monkeypatch):
    # The case the fallback got wrong:
    trees = {SUFFIX_SHARED_A: TREE_A, SUFFIX_SHARED_B: TREE_B}
    fetched = []

    def fake(url):
        fetched.append(url)
        return {"ggml_tree": trees[next(t for t in trees if t in url)]}

    monkeypatch.setattr(M, "_download_host_json_once", fake)
    assert (
        M.llama_runtime_pairs(SUFFIX_SHARED_A, SUFFIX_SHARED_B, installed_repo = FORK_REPO) is False
    )
    assert len(fetched) == 2
    # The installed tag is read from the repo that published it.
    # FORK_REPO is not the default publisher, so a lookup ignoring installed_repo would build a unslothai/llama.cpp URL
    # here and this would catch it.
    assert any(f"/{FORK_REPO}/releases/download/{SUFFIX_SHARED_A}/" in url for url in fetched)


def test_published_ggml_tree_is_fetched_once_per_tag(monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        return {"ggml_tree": TREE_A}

    monkeypatch.setattr(M, "_download_host_json_once", fake)
    assert M.published_llama_ggml_tree(SUFFIX_SHARED_A) == TREE_A
    assert M.published_llama_ggml_tree(SUFFIX_SHARED_A) == TREE_A
    assert calls == [
        f"https://github.com/unslothai/llama.cpp/releases/download/"
        f"{SUFFIX_SHARED_A}/llama-prebuilt-manifest.json"
    ]


def test_the_manifest_probe_still_authenticates(monkeypatch):
    """download_bytes falls back to auth_headers(url) only when headers are ABSENT, so a bare
    User-Agent silently dropped auth. A private published repo then 404s and the caller falls
    back to the -mix- suffix compare this probe exists to replace; an anonymous huggingface.co
    fetch shares the per-IP limit that 429s CI fleets."""
    seen = {}

    def fake_download_bytes(
        url,
        timeout = None,
        attempts = None,
        headers = None,
        **kwargs,
    ):
        seen["headers"] = headers
        seen["attempts"] = attempts
        return b'{"ggml_tree": "abc"}'

    monkeypatch.setenv("GH_TOKEN", "gh-secret")
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    monkeypatch.setattr(M, "download_bytes", fake_download_bytes)

    assert _REAL_DOWNLOAD_HOST_JSON_ONCE("https://github.com/o/r/releases/download/t/m.json") == {
        "ggml_tree": "abc"
    }
    assert seen["headers"].get("Authorization") == "Bearer gh-secret"
    # The single-attempt policy is the ONLY thing this wrapper overrides.
    assert seen["attempts"] == 1


@pytest.mark.parametrize(
    "payload",
    [{}, {"ggml_tree": ""}, {"ggml_tree": None}, {"ggml_tree": ["x"]}, ["not", "a", "dict"]],
)
def test_published_ggml_tree_ignores_a_manifest_without_a_usable_tree(monkeypatch, payload):
    monkeypatch.setattr(M, "_download_host_json_once", lambda url: payload)
    assert M.published_llama_ggml_tree(SUFFIX_SHARED_A) is None


def test_published_ggml_tree_survives_an_unreachable_release(monkeypatch):
    def boom(url):
        raise OSError("network down")

    monkeypatch.setattr(M, "_download_host_json_once", boom)
    assert M.published_llama_ggml_tree(SUFFIX_SHARED_A) is None
    assert (
        M.llama_runtime_pairs(SUFFIX_SHARED_A, SUFFIX_SHARED_B, installed_repo = LLAMA_REPO) is True
    )


def test_published_tree_is_not_consulted_when_both_trees_are_known(monkeypatch):
    # Fail closed to the old comparison rather than stranding a valid install.
    def boom(url):
        raise AssertionError(f"should not fetch {url}")

    monkeypatch.setattr(M, "_download_host_json_once", boom)
    assert (
        M.llama_runtime_pairs(
            SUFFIX_SHARED_A,
            SUFFIX_SHARED_B,
            installed_ggml_tree = TREE_A,
            required_ggml_tree = TREE_B,
        )
        is False
    )
    assert M.llama_runtime_pairs(SUFFIX_SHARED_A, SUFFIX_SHARED_A) is True


@pytest.mark.parametrize(
    "marker,expected",
    [
        ({"published_repo": LLAMA_REPO}, LLAMA_REPO),
        ({"published_repo": LLAMA_REPO, "binary_repo": LLAMA_REPO}, LLAMA_REPO),
        # Binaries from another repo:
        ({"published_repo": LLAMA_REPO, "binary_repo": "ggml-org/llama.cpp"}, None),
        ({"binary_repo": LLAMA_REPO}, None),
        ({}, None),
    ],
)
def test_installed_llama_tree_repo_follows_the_marker(monkeypatch, tmp_path, marker, expected):
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps({"release_tag": "b1", **marker}))
    monkeypatch.setattr(M.llama, "default_managed_llama_dir", lambda: root)
    assert M.installed_llama_tree_repo() == expected


def test_pairing_does_not_infer_a_tree_for_non_fork_binaries(monkeypatch):
    # Without a repo the installed tag's tree is never fetched, so an upstream
    # built runtime cannot be paired by a tree that does not describe it.
    # A raising stub cannot police this: published_llama_ggml_tree() catches
    # The case the fallback got wrong:
    trees = {SUFFIX_SHARED_A: TREE_A, SUFFIX_SHARED_B: TREE_B}
    fetched = []

    def record(url):
        fetched.append(url)
        return {"ggml_tree": trees[next(t for t in trees if t in url)]}

    monkeypatch.setattr(M, "_download_host_json_once", record)
    assert M.llama_runtime_pairs(SUFFIX_SHARED_A, SUFFIX_SHARED_B, installed_repo = None) is True
    # the required tag because a lone required tree cannot decide the pairing.
    # Nothing at all is probed:
    assert fetched == []


def test_installed_llama_ggml_tree_reads_marker(tmp_path):
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": SLIM_LLAMA_TAG, "ggml_tree": TREE_A})
    )
    assert M.llama.installed_llama_ggml_tree(root) == TREE_A


@pytest.mark.parametrize(
    "prepare",
    [
        lambda root: None,
        lambda root: (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{}"),
        lambda root: (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps({"ggml_tree": ""})
        ),
        lambda root: (root / "UNSLOTH_PREBUILT_INFO.json").write_text("not json"),
    ],
)
def test_installed_llama_ggml_tree_absent_is_none(tmp_path, prepare):
    root = tmp_path / "llama.cpp"
    root.mkdir()
    prepare(root)
    assert M.llama.installed_llama_ggml_tree(root) is None


def test_slim_pairs_across_llama_build_bump_with_same_ggml(tmp_path, monkeypatch):
    # The live failure:
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, NEWER_LLAMA_TAG, "cuda13-newer")
    )
    artifact, backend, used_fallback = M.select_artifact_with_fallback(
        _slim_manifest(), _cuda_host(), "cuda"
    )
    assert artifact["asset"] == SLIM_ASSET
    assert backend == "cuda" and used_fallback is False


def test_slim_build_bump_same_ggml_is_not_a_compatibility_error(tmp_path, monkeypatch):
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, NEWER_LLAMA_TAG, "cuda13-newer")
    )
    assert M._slim_release_incompatibility(_slim_manifest(), _cuda_host()) is None


def test_link_ggml_runtime_hardlinks_every_ggml_library(tmp_path):
    bin_dir = _fake_llama_bin(tmp_path)
    whisper_bin = tmp_path / "whisper.cpp" / "build" / "bin"
    linked = M.link_ggml_runtime(bin_dir, whisper_bin)
    assert linked == [
        "libggml-base.so.0",
        "libggml-cpu-x64.so",
        "libggml-cuda.so",
        "libggml.so.0",
    ]
    for name in ("libggml.so.0", "libggml-base.so.0", "libggml-cuda.so", "libggml-cpu-x64.so"):
        source, dest = bin_dir / name, whisper_bin / name
        assert dest.is_file() and not dest.is_symlink()
        assert dest.stat().st_ino == source.stat().st_ino
    assert not (whisper_bin / "libllama.so").exists()


def test_link_ggml_runtime_copy_fallback(tmp_path, monkeypatch):
    def no_link(src, dst, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(M.os, "link", no_link)
    # and must still select rather than degrade to CPU or report unavailable.
    # A same-ggml build bump must not surface as a release incompatibility (the
    # The live failure: the llama installer advances to a newer build that keeps the same ggml commit, so the slim
    bin_dir = _fake_llama_bin(tmp_path)
    whisper_bin = tmp_path / "whisper.cpp" / "build" / "bin"
    assert len(M.link_ggml_runtime(bin_dir, whisper_bin)) == 4
    dest = whisper_bin / "libggml.so.0"
    assert dest.read_bytes() == (bin_dir / "libggml.so.0").read_bytes()
    assert dest.stat().st_nlink == 1


def test_link_ggml_runtime_hardlinks_dylibs(tmp_path):
    # macOS wiring: the libggml* glob must take the .dylib names too.
    bin_dir = _fake_llama_bin_macos(tmp_path)
    whisper_bin = tmp_path / "whisper.cpp" / "build" / "bin"
    linked = M.link_ggml_runtime(bin_dir, whisper_bin)
    assert linked == [
        "libggml-base.dylib",
        "libggml-blas.dylib",
        "libggml-cpu.dylib",
        "libggml-metal.dylib",
        "libggml.dylib",
    ]
    for name in linked:
        source, dest = bin_dir / name, whisper_bin / name
        assert dest.is_file() and not dest.is_symlink()
        assert dest.stat().st_ino == source.stat().st_ino
    assert not (whisper_bin / "libllama.dylib").exists()


def test_link_ggml_runtime_fails_closed_on_empty_runtime(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PrebuiltFallback):
        M.link_ggml_runtime(empty, tmp_path / "whisper_bin")


def test_link_ggml_runtime_wires_windows_libomp(tmp_path):
    # llama's clang-built windows-arm64 ggml-base.dll imports libomp140.aarch64.dll (bundled, not a system DLL): the
    # wiring must place it next to whisper-server.exe or the loader dies with DLL_NOT_FOUND.
    bin_dir = tmp_path / "llama_bin"
    bin_dir.mkdir()
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "libomp140.aarch64.dll"):
        (bin_dir / name).write_bytes(b"x")
    (bin_dir / "llama.dll").write_bytes(b"x")  # never wired
    whisper_bin = tmp_path / "whisper_bin"
    linked = M.link_ggml_runtime(bin_dir, whisper_bin)
    assert linked == ["ggml-base.dll", "ggml-cpu.dll", "ggml.dll", "libomp140.aarch64.dll"]
    assert not (whisper_bin / "llama.dll").exists()


def test_link_ggml_runtime_wires_linux_libomp(tmp_path):
    # llama's clang-built linux-arm64 libggml-base.so NEEDS libomp.so.5, bundled and never on the host:
    bin_dir = tmp_path / "llama_bin"
    bin_dir.mkdir()
    for name in ("libggml.so.0", "libggml-base.so.0", "libggml-cpu-armv8.0_1.so", "libomp.so.5"):
        (bin_dir / name).write_bytes(b"x")
    (bin_dir / "libllama.so").write_bytes(b"x")  # never wired
    whisper_bin = tmp_path / "whisper_bin"
    linked = M.link_ggml_runtime(bin_dir, whisper_bin)
    assert linked == [
        "libggml-base.so.0",
        "libggml-cpu-armv8.0_1.so",
        "libggml.so.0",
        "libomp.so.5",
    ]
    assert (whisper_bin / "libomp.so.5").is_file()
    assert not (whisper_bin / "libllama.so").exists()


def test_link_ggml_runtime_linux_libomp_alone_is_not_a_pairing(tmp_path):
    bin_dir = tmp_path / "llama_bin"
    bin_dir.mkdir()
    (bin_dir / "libomp.so.5").write_bytes(b"x")
    with pytest.raises(PrebuiltFallback):
        M.link_ggml_runtime(bin_dir, tmp_path / "whisper_bin")


def test_rocm_runtime_wires_complete_windows_dll_overlay(tmp_path):
    # Same fail-closed rule as the Windows case:
    bin_dir = tmp_path / "llama_bin"
    bin_dir.mkdir()
    dlls = {
        "ggml.dll",
        "ggml-base.dll",
        "ggml-hip.dll",
        "amdhip64.dll",
        "rocblas.dll",
        "hipblaslt.dll",
        "hsa-runtime64.dll",
        # The llama Windows ROCm archive can carry transitive DLLs whose names do not contain hip/roc/amd.
        "runtime-support.dll",
    }
    for name in dlls:
        (bin_dir / name).write_bytes(name.encode())
    (bin_dir / "not-a-runtime.txt").write_text("ignored")

    whisper_bin = tmp_path / "whisper_bin"
    linked = M.link_ggml_runtime(bin_dir, whisper_bin, backend = "rocm")
    linked_dirs = M.link_runtime_directories(
        bin_dir,
        whisper_bin,
        backend = "rocm",
        host = _host("windows", "x64"),
    )

    assert set(linked) == dlls
    assert linked_dirs == []
    assert all((whisper_bin / name).is_file() for name in dlls)
    assert not (whisper_bin / "not-a-runtime.txt").exists()


def test_assemble_does_not_copy_packaging_marker_beside_server(tmp_path):
    host = _host("linux", "x64")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "whisper-server").write_text("server")
    (bundle / M.METADATA_FILENAME).write_text('{"backend":"slim"}')
    staged = tmp_path / "staged"
    M.assemble_install_tree(bundle, staged, host)
    assert not (staged / "build" / "bin" / M.METADATA_FILENAME).exists()


def test_rocm_runtime_wires_packaged_dependency_closure_and_catalogs(tmp_path, monkeypatch):
    llama_bin = _fake_llama_bin(tmp_path, backend_module = "libggml-hip.so")
    for name in ("libamdhip64.so.6", "libhipblas.so.2", "libcustomrocm.so.1"):
        (llama_bin / name).write_bytes(name.encode())
    for directory in M.SLIM_ROCM_RUNTIME_DIRS:
        catalog = llama_bin / directory / "library" / "gfx1100"
        catalog.mkdir(parents = True)
        (catalog / "kernel.dat").write_bytes(directory.encode())

    dependencies = {
        "libggml-hip.so": {"libamdhip64.so.6", "libcustomrocm.so.1"},
        "libamdhip64.so.6": {"libhipblas.so.2"},
    }
    monkeypatch.setattr(M, "_elf_needed", lambda path: dependencies.get(path.name, set()))
    whisper_bin = tmp_path / "whisper-bin"
    linked = M.link_ggml_runtime(llama_bin, whisper_bin, backend = "rocm")
    linked_dirs = M.link_runtime_directories(
        llama_bin,
        whisper_bin,
        backend = "rocm",
        host = _host("linux", "x64"),
    )

    assert "libcustomrocm.so.1" in linked
    assert "libhipblas.so.2" in linked
    assert linked_dirs == ["hipblaslt", "rocblas"]
    for directory in linked_dirs:
        linked_catalog = whisper_bin / directory / "library" / "gfx1100" / "kernel.dat"
        source_catalog = llama_bin / directory / "library" / "gfx1100" / "kernel.dat"
        assert linked_catalog.stat().st_ino == source_catalog.stat().st_ino


def test_rocm_runtime_requires_the_rocblas_kernel_catalog(tmp_path):
    # rocblas stays mandatory:
    llama_bin = _fake_llama_bin(tmp_path, backend_module = "libggml-hip.so")
    (llama_bin / "hipblaslt").mkdir()
    (llama_bin / "hipblaslt" / "kernel.dat").write_bytes(b"kernel")
    with pytest.raises(PrebuiltFallback, match = "rocblas"):
        M.link_runtime_directories(
            llama_bin,
            tmp_path / "whisper-bin",
            backend = "rocm",
            host = _host("linux", "x64"),
        )


def test_rocm_runtime_rejects_an_empty_rocblas_kernel_catalog(tmp_path):
    llama_bin = _fake_llama_bin(tmp_path, backend_module = "libggml-hip.so")
    (llama_bin / "rocblas" / "library").mkdir(parents = True)
    with pytest.raises(PrebuiltFallback, match = "empty rocblas"):
        M.link_runtime_directories(
            llama_bin,
            tmp_path / "whisper-bin",
            backend = "rocm",
            host = _host("linux", "x64"),
        )


def _gfx103x_llama_bin(tmp_path: Path) -> Path:
    """The published linux-x64-rocm-gfx103X layout from #8364: librocblas plus
    its Tensile catalog, and libhipblaslt.so.1 with no hipblaslt/ catalog at all.
    hipBLASLt builds no kernels for gfx1030 (RDNA2), so nothing is missing here;
    llama.cpp installs and runs inference on exactly this tree."""
    llama_bin = _fake_llama_bin(tmp_path, backend_module = "libggml-hip.so")
    for name in ("libamdhip64.so.7", "libhipblas.so.3", "librocblas.so.5", "libhipblaslt.so.1"):
        (llama_bin / name).write_bytes(name.encode())
    catalog = llama_bin / "rocblas" / "library"
    catalog.mkdir(parents = True)
    (catalog / "TensileLibrary_Type_HH_Contraction_gfx1030.dat").write_bytes(b"kernel")
    return llama_bin


def test_rocm_runtime_wires_rocblas_when_hipblaslt_ships_no_kernels(tmp_path):
    # #8364:
    llama_bin = _gfx103x_llama_bin(tmp_path)
    whisper_bin = tmp_path / "whisper-bin"

    linked_dirs = M.link_runtime_directories(
        llama_bin,
        whisper_bin,
        backend = "rocm",
        host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1030"),
    )

    assert linked_dirs == ["rocblas"]
    catalog = whisper_bin / "rocblas" / "library" / "TensileLibrary_Type_HH_Contraction_gfx1030.dat"
    assert catalog.is_file()
    assert catalog.stat().st_ino == (llama_bin / "rocblas" / "library" / catalog.name).stat().st_ino
    assert not (whisper_bin / "hipblaslt").exists()


def test_rocm_runtime_treats_an_empty_hipblaslt_catalog_as_absent(tmp_path):
    # 8364: RX 6800 (gfx1030) on linux x64.
    llama_bin = _gfx103x_llama_bin(tmp_path)
    (llama_bin / "hipblaslt" / "library").mkdir(parents = True)
    whisper_bin = tmp_path / "whisper-bin"

    assert M.link_runtime_directories(
        llama_bin,
        whisper_bin,
        backend = "rocm",
        host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1030"),
    ) == ["rocblas"]
    # No empty stand-in directory:
    assert not (whisper_bin / "hipblaslt").exists()


@pytest.mark.parametrize(
    "whisper_os, backend",
    [
        ("linux", "cpu"),
        ("linux", "cuda"),
        ("linux", "vulkan"),
        ("macos", "cpu"),
        ("macos", "metal"),
        ("windows", "cpu"),
        ("windows", "cuda"),
        ("windows", "vulkan"),
        ("windows", "rocm"),
    ],
)
def test_runtime_directories_are_a_linux_rocm_concern_only(tmp_path, whisper_os, backend):
    # Non-regression for the backends #8364 must not touch: no catalog is looked
    # for, so a bundle without one is never rejected over it.
    llama_bin = _fake_llama_bin(tmp_path, backend_module = None)
    assert (
        M.link_runtime_directories(
            llama_bin,
            tmp_path / f"whisper-bin-{whisper_os}-{backend}",
            backend = backend,
            host = _host(whisper_os, "x64"),
        )
        == []
    )


def test_rocm_runtime_catalog_copy_fallback(tmp_path, monkeypatch):
    # rocblas stays mandatory: libggml-hip.so lists librocblas.so.5 in its ELF NEEDED (checked on the published b10342
    llama_bin = _fake_llama_bin(tmp_path, backend_module = "libggml-hip.so")
    for directory in M.SLIM_ROCM_RUNTIME_DIRS:
        catalog = llama_bin / directory
        catalog.mkdir()
        (catalog / "kernel.dat").write_bytes(directory.encode())
    monkeypatch.setattr(M.os, "link", lambda *args: (_ for _ in ()).throw(OSError("xdev")))
    whisper_bin = tmp_path / "whisper-bin"
    M.link_runtime_directories(
        llama_bin,
        whisper_bin,
        backend = "rocm",
        host = _host("linux", "x64"),
    )
    for directory in M.SLIM_ROCM_RUNTIME_DIRS:
        assert (whisper_bin / directory / "kernel.dat").read_bytes() == directory.encode()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason = "os.access(X_OK) is always true on Windows, so the guard is POSIX only",
)
def test_existing_install_requires_executable_server(tmp_path, monkeypatch):
    # A marker-matching install with a non-executable server must reinstall: the sidecar refuses it via os.access(X_OK),
    # so "already matches" would otherwise leave dictation permanently broken.
    host = _host("linux", "x64")
    selection = object()
    monkeypatch.setattr(M.core, "existing_install_matches", lambda *a: True)
    server = tmp_path / "build" / "bin" / "whisper-server"
    server.parent.mkdir(parents = True)
    server.write_text("bin")
    monkeypatch.setattr(M, "installed_server_path", lambda d, h: server)
    monkeypatch.setattr(M, "load_prebuilt_metadata", lambda d: {})
    server.chmod(0o644)
    assert M.existing_install_matches(tmp_path, host, selection) is False
    server.chmod(0o755)
    assert M.existing_install_matches(tmp_path, host, selection) is True


def test_existing_slim_install_requires_wired_libraries(tmp_path, monkeypatch):
    # A slim install whose hardlinked ggml files vanished (llama dir deleted) must reinstall so update re-wires instead
    host = _host("linux", "x64")
    monkeypatch.setattr(M.core, "existing_install_matches", lambda *a: True)
    server = tmp_path / "build" / "bin" / "whisper-server"
    server.parent.mkdir(parents = True)
    server.write_text("bin")
    server.chmod(0o755)
    monkeypatch.setattr(M, "installed_server_path", lambda d, h: server)
    marker = {
        "install_kind": "slim",
        "runtime_wiring_version": M.SLIM_RUNTIME_WIRING_VERSION,
        "linked_libraries": ["libggml.so.0"],
    }
    monkeypatch.setattr(M, "load_prebuilt_metadata", lambda d: marker)
    marker.pop("runtime_wiring_version")
    assert M.existing_install_matches(tmp_path, host, object()) is False
    marker["runtime_wiring_version"] = M.SLIM_RUNTIME_WIRING_VERSION
    assert M.existing_install_matches(tmp_path, host, object()) is False
    (server.parent / "libggml.so.0").write_text("lib")
    assert M.existing_install_matches(tmp_path, host, object()) is True
    marker.update(backend = "rocm", linked_runtime_directories = [])
    assert M.existing_install_matches(tmp_path, _host("windows", "x64"), object()) is True


@pytest.mark.parametrize(
    "runtime_dirs, current",
    [
        (["hipblaslt", "rocblas"], True),  # every target hipBLASLt has kernels for gfx1030 and friends:
        (["rocblas"], True),  # gfx1030 and friends: #8364
        ([], False),  # rocblas is load-bearing, never optional
        (["hipblaslt"], False),
        (["rocblas", "unexpected"], False),
        ("rocblas", False),
    ],
)
def test_existing_rocm_install_accepts_the_catalogs_the_target_has(
    tmp_path, monkeypatch, runtime_dirs, current
):
    # #8364:
    host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1030")
    monkeypatch.setattr(M.core, "existing_install_matches", lambda *a: True)
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server = bin_dir / "whisper-server"
    server.write_text("bin")
    server.chmod(0o755)
    (bin_dir / "libggml.so.0").write_text("lib")
    for name in ("hipblaslt", "rocblas", "unexpected"):
        (bin_dir / name).mkdir()
        (bin_dir / name / "kernel.dat").write_text("kernel")
    monkeypatch.setattr(M, "installed_server_path", lambda d, h: server)
    monkeypatch.setattr(
        M,
        "load_prebuilt_metadata",
        lambda d: {
            "install_kind": "slim",
            "backend": "rocm",
            "runtime_wiring_version": M.SLIM_RUNTIME_WIRING_VERSION,
            "linked_libraries": ["libggml.so.0"],
            "linked_runtime_directories": runtime_dirs,
        },
    )

    assert M.existing_install_matches(tmp_path, host, object()) is current


@pytest.mark.parametrize("empty", ["hipblaslt", "rocblas"])
def test_existing_rocm_install_reinstalls_over_an_empty_catalog_on_disk(
    tmp_path, monkeypatch, empty
):
    # Relaxing the marker check to membership must not relax the on-disk check:
    host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1030")
    monkeypatch.setattr(M.core, "existing_install_matches", lambda *a: True)
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server = bin_dir / "whisper-server"
    server.write_text("bin")
    server.chmod(0o755)
    (bin_dir / "libggml.so.0").write_text("lib")
    for name in ("hipblaslt", "rocblas"):
        (bin_dir / name / "library").mkdir(parents = True)
        if name != empty:
            (bin_dir / name / "library" / "kernel.dat").write_text("kernel")
    monkeypatch.setattr(M, "installed_server_path", lambda d, h: server)
    monkeypatch.setattr(
        M,
        "load_prebuilt_metadata",
        lambda d: {
            "install_kind": "slim",
            "backend": "rocm",
            "runtime_wiring_version": M.SLIM_RUNTIME_WIRING_VERSION,
            "linked_libraries": ["libggml.so.0"],
            "linked_runtime_directories": ["hipblaslt", "rocblas"],
        },
    )

    assert M.existing_install_matches(tmp_path, host, object()) is False


def test_link_ggml_runtime_libomp_alone_is_not_a_pairing(tmp_path):
    # A libomp without any ggml library is not a usable llama runtime.
    bin_dir = tmp_path / "llama_bin"
    bin_dir.mkdir()
    (bin_dir / "libomp140.aarch64.dll").write_bytes(b"x")
    with pytest.raises(PrebuiltFallback):
        M.link_ggml_runtime(bin_dir, tmp_path / "whisper_bin")


def _build_slim_bundle(tmp_path: Path, host: HostInfo) -> tuple[Path, str]:
    """Slim archive per the CI contract: whisper-server + libwhisper only."""
    archive = tmp_path / SLIM_ASSET
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, M.server_binary_name(host), b"#!/bin/sh\necho whisper\n", mode = 0o755)
        _add_file(tar, "libwhisper.so.1", b"dummy-libwhisper")
    return archive, M.sha256_file(archive)


def _slim_install_env(monkeypatch, tmp_path, host, *, sha256: str) -> Path:
    llama_bin = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M,
        "installed_llama_runtime",
        lambda install_dir = None: (llama_bin, SLIM_LLAMA_TAG, "cuda13-newer"),
    )
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact("linux", "x64", "cpu", CPU_ASSET, "a" * 64),
                _slim_artifact(sha256 = sha256),
            ]
        )
    )
    bundle = M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {SLIM_ASSET: f"https://example.invalid/{SLIM_ASSET}"},
    )
    checksums = {SLIM_ASSET: sha256, CPU_ASSET: "a" * 64}
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (bundle, checksums),
    )
    return llama_bin


def test_slim_install_wires_links_and_marker(tmp_path, monkeypatch):
    host = _cuda_host()
    archive, sha256 = _build_slim_bundle(tmp_path, host)

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    llama_bin = _slim_install_env(monkeypatch, tmp_path, host, sha256 = sha256)

    install_dir = tmp_path / "whisper.cpp"
    assert M.install_prebuilt(install_dir, backend = "cuda") == M.EXIT_SUCCESS

    whisper_bin = install_dir / "build" / "bin"
    assert (whisper_bin / "whisper-server").is_file()
    assert (whisper_bin / "libwhisper.so.1").is_file()
    for name in ("libggml.so.0", "libggml-base.so.0", "libggml-cuda.so", "libggml-cpu-x64.so"):
        dest = whisper_bin / name
        assert dest.is_file() and not dest.is_symlink()
        assert dest.stat().st_ino == (llama_bin / name).stat().st_ino

    marker = json.loads((install_dir / M.METADATA_FILENAME).read_text())
    assert marker["backend"] == "cuda"
    assert marker["asset"] == SLIM_ASSET
    assert marker["install_kind"] == "slim"
    assert marker["paired_llama_tag"] == SLIM_LLAMA_TAG
    assert marker["linked_from"] == str(llama_bin)
    # The wired filenames land in the marker;
    assert marker["linked_libraries"] == [
        "libggml-base.so.0",
        "libggml-cpu-x64.so",
        "libggml-cuda.so",
        "libggml.so.0",
    ]
    assert marker["runtime_wiring_version"] == M.SLIM_RUNTIME_WIRING_VERSION
    assert marker["linked_runtime_directories"] == []


def test_slim_rocm_install_pairs_a_runtime_without_a_hipblaslt_catalog(tmp_path, monkeypatch):
    # End to end over the #8364 tree: the install must complete and the marker
    # 8364: a gfx1030 install wires rocblas alone and is complete, so "already matches" must hold for it, while a
    host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1030")
    archive, sha256 = _build_slim_bundle(tmp_path, host)
    llama_bin = _gfx103x_llama_bin(tmp_path)

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    monkeypatch.setattr(M, "_elf_needed", lambda path: set())
    monkeypatch.setattr(
        M,
        "installed_llama_runtime",
        lambda install_dir = None: (llama_bin, SLIM_LLAMA_TAG, "rocm-gfx103X"),
    )
    manifest = M.parse_manifest(_manifest([_slim_artifact(sha256 = sha256)]))
    bundle = M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {SLIM_ASSET: f"https://example.invalid/{SLIM_ASSET}"},
    )
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (bundle, {SLIM_ASSET: sha256}),
    )

    install_dir = tmp_path / "whisper.cpp"
    assert M.install_prebuilt(install_dir, backend = "rocm") == M.EXIT_SUCCESS

    whisper_bin = install_dir / "build" / "bin"
    assert (whisper_bin / "libggml-hip.so").is_file()
    assert (whisper_bin / "rocblas" / "library").is_dir()
    assert not (whisper_bin / "hipblaslt").exists()
    marker = json.loads((install_dir / M.METADATA_FILENAME).read_text())
    assert marker["backend"] == "rocm"
    assert marker["install_kind"] == "slim"
    assert marker["linked_runtime_directories"] == ["rocblas"]
    assert marker["runtime_wiring_version"] == M.SLIM_RUNTIME_WIRING_VERSION
    # A second run is a no-op: the wiring the target can have is the wiring it has.
    assert M.install_prebuilt(install_dir, backend = "rocm") == M.EXIT_SUCCESS


def test_slim_links_survive_a_llama_dir_swap(tmp_path, monkeypatch):
    # The whole point of hardlinks:
    host = _cuda_host()
    archive, sha256 = _build_slim_bundle(tmp_path, host)

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(M, "download_file", fake_download)
    llama_bin = _slim_install_env(monkeypatch, tmp_path, host, sha256 = sha256)

    install_dir = tmp_path / "whisper.cpp"
    assert M.install_prebuilt(install_dir, backend = "cuda") == M.EXIT_SUCCESS
    whisper_lib = install_dir / "build" / "bin" / "libggml.so.0"
    old_inode = whisper_lib.stat().st_ino
    old_content = whisper_lib.read_bytes()

    for path in llama_bin.iterdir():
        path.unlink()
    (llama_bin / "libggml.so.0").write_bytes(b"ggml-NEW")

    assert whisper_lib.stat().st_ino == old_inode
    assert whisper_lib.read_bytes() == old_content
    assert whisper_lib.stat().st_nlink == 1  # the llama side is gone; ours remains


# The exact resolver key set shipped before slim existed;
# install_kind is the one additive field and must stay the only difference.
_LEGACY_RESOLVER_KEYS = {
    "prebuilt_available",
    "repo",
    "release_tag",
    "upstream_tag",
    "backend",
    "requested_backend",
    "cpu_fallback",
    "asset",
    "os",
    "arch",
    "runtime_line",
}


def _resolver_payload(monkeypatch, capsys, manifest, host) -> dict:
    bundle = M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {},
    )
    monkeypatch.setattr(M, "detect_host", lambda: host)
    monkeypatch.setattr(
        M, "fetch_release_for_install", lambda repo, *, published_release_tag = None: (bundle, {})
    )
    assert M.main(["--resolve-prebuilt", "--output-format", "json"]) == M.EXIT_SUCCESS
    return json.loads(capsys.readouterr().out.strip())


def test_resolver_reports_install_kind_fat_additively(monkeypatch, capsys):
    manifest = M.parse_manifest(_manifest([_artifact("linux", "x64", "cpu", CPU_ASSET, "a" * 64)]))
    payload = _resolver_payload(monkeypatch, capsys, manifest, _host("linux", "x64"))
    assert set(payload) == _LEGACY_RESOLVER_KEYS | {"install_kind"}
    assert payload["install_kind"] == "fat"
    assert payload["asset"] == CPU_ASSET


def _release_for_selection(release_tag: str, upstream_tag: str, artifact: dict) -> M.ReleaseBundle:
    payload = _manifest([artifact])
    payload["upstream_tag"] = upstream_tag
    return M.ReleaseBundle(
        repo = "unslothai/whisper.cpp",
        release_tag = release_tag,
        manifest = M.parse_manifest(payload),
        asset_urls = {},
    )


def test_whisper_tag_selects_matching_published_upstream(monkeypatch):
    latest_asset = "latest-linux-x64-cpu.tar.gz"
    pinned_asset = "pinned-linux-x64-cpu.tar.gz"
    latest = _release_for_selection(
        "v1.9.2-unsloth.1",
        "v1.9.2",
        _artifact("linux", "x64", "cpu", latest_asset, "a" * 64),
    )
    pinned = _release_for_selection(
        "v1.9.0-unsloth.3",
        "v1.9.0",
        _artifact("linux", "x64", "cpu", pinned_asset, "b" * 64),
    )
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an upstream pin consulted the unrelated newest release")
        ),
    )
    monkeypatch.setattr(
        M,
        "_published_release_tags",
        lambda repo: [latest.release_tag, pinned.release_tag],
    )
    monkeypatch.setattr(
        M,
        "_fetch_release_candidate",
        lambda repo, tag: latest if tag == latest.release_tag else pinned,
    )
    monkeypatch.setattr(M, "fetch_release_checksums", lambda bundle: {pinned_asset: "b" * 64})

    payload = M.resolve_prebuilt(
        _host("linux", "x64"),
        published_repo = "unslothai/whisper.cpp",
        published_release_tag = None,
        whisper_tag = "v1.9.0",
        backend = "cpu",
        cpu_fallback = False,
    )
    assert payload["prebuilt_available"] is True
    assert payload["release_tag"] == pinned.release_tag
    assert payload["upstream_tag"] == "v1.9.0"


def test_macos_walks_back_to_newest_compatible_release(monkeypatch):
    latest_asset = "latest-macos-arm64-cpu.tar.gz"
    compatible_asset = "compatible-macos-arm64-cpu.tar.gz"
    latest = _release_for_selection(
        "v1.9.2-unsloth.1",
        "v1.9.2",
        _artifact(
            "macos",
            "arm64",
            "cpu",
            latest_asset,
            "a" * 64,
            min_os = "macos-15.0",
        ),
    )
    compatible = _release_for_selection(
        "v1.9.1-unsloth.1",
        "v1.9.1",
        _artifact(
            "macos",
            "arm64",
            "cpu",
            compatible_asset,
            "b" * 64,
            min_os = "macos-13.0",
        ),
    )
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (latest, {latest_asset: "a" * 64}),
    )
    monkeypatch.setattr(
        M,
        "_published_release_tags",
        lambda repo: [latest.release_tag, compatible.release_tag],
    )
    monkeypatch.setattr(M, "_fetch_release_candidate", lambda repo, tag: compatible)
    monkeypatch.setattr(
        M,
        "fetch_release_checksums",
        lambda bundle: {compatible_asset: "b" * 64},
    )

    payload = M.resolve_prebuilt(
        _host("macos", "arm64", macos_version = (14, 7)),
        published_repo = "unslothai/whisper.cpp",
        published_release_tag = None,
        backend = "cpu",
        cpu_fallback = False,
    )
    assert payload["prebuilt_available"] is True
    assert payload["release_tag"] == compatible.release_tag


def test_macos_walkback_never_masks_checksum_failure(monkeypatch):
    asset = "latest-macos-arm64-cpu.tar.gz"
    latest = _release_for_selection(
        "v1.9.2-unsloth.1",
        "v1.9.2",
        _artifact("macos", "arm64", "cpu", asset, "a" * 64, min_os = "macos-13.0"),
    )
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (latest, {asset: "b" * 64}),
    )
    monkeypatch.setattr(
        M,
        "_published_release_tags",
        lambda repo: (_ for _ in ()).throw(AssertionError("integrity failure walked back")),
    )

    with pytest.raises(PrebuiltFallback, match = "disagrees"):
        M._release_plan_for_host(
            _host("macos", "arm64", macos_version = (14, 7)),
            published_repo = "unslothai/whisper.cpp",
            published_release_tag = None,
            whisper_tag = "latest",
            requested_backend = "cpu",
        )


def test_resolver_reports_install_kind_slim_when_paired(tmp_path, monkeypatch, capsys):
    bin_dir = _fake_llama_bin(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "cuda13-newer")
    )
    payload = _resolver_payload(monkeypatch, capsys, _slim_manifest(), _cuda_host())
    assert set(payload) == _LEGACY_RESOLVER_KEYS | {"install_kind"}
    assert payload["install_kind"] == "slim"
    assert payload["asset"] == SLIM_ASSET
    assert payload["backend"] == "cuda"


def test_resolver_reports_metal_slim_on_macos(tmp_path, monkeypatch, capsys):
    # Same contract shape on macs:
    bin_dir = _fake_llama_bin_macos(tmp_path)
    monkeypatch.setattr(
        M, "installed_llama_runtime", lambda: (bin_dir, SLIM_LLAMA_TAG, "macos-metal-arm64")
    )
    host = _host("macos", "arm64", macos_version = (14, 0))
    payload = _resolver_payload(monkeypatch, capsys, _macos_slim_manifest(), host)
    assert set(payload) == _LEGACY_RESOLVER_KEYS | {"install_kind"}
    assert payload["install_kind"] == "slim"
    assert payload["asset"] == MAC_SLIM_ASSET
    assert payload["backend"] == "metal"
    assert payload["requested_backend"] == "metal"
