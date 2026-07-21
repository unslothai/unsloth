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
# The installer imports the shared prebuilt core via `from backend...`; make
# studio/ importable so that resolves under spec-based loading.
_STUDIO_DIR = str(MODULE_PATH.parent)
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)
SPEC = importlib.util.spec_from_file_location("studio_install_whisper_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

from backend.utils.prebuilt.hosts import NvidiaCaps  # noqa: E402

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
        M._runtime_libs,
        "detected_cuda_runtime_lines",
        lambda *, is_windows: ["cuda13", "cuda12"],
    )


# ── Host detection ──
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
    monkeypatch.setattr(M.platform, "system", lambda: system)
    monkeypatch.setattr(M.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        M,
        "detect_nvidia_caps",
        lambda **_kw: NvidiaCaps(
            has_usable_nvidia = False, compute_caps = [], driver_cuda_version = None
        ),
    )
    monkeypatch.setattr(M, "_detect_rocm_gfx", lambda: (False, None))
    host = M.detect_host()
    assert (host.whisper_os, host.whisper_arch, host.archive_ext) == (exp_os, exp_arch, exp_ext)
    assert host.is_windows == (exp_os == "windows")
    assert host.is_apple_silicon == (exp_os == "macos" and exp_arch == "arm64")


def test_detect_host_populates_nvidia_caps(monkeypatch):
    monkeypatch.setattr(M.platform, "system", lambda: "Linux")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        M,
        "detect_nvidia_caps",
        lambda **_kw: NvidiaCaps(
            has_usable_nvidia = True, compute_caps = ["10.0"], driver_cuda_version = (13, 0)
        ),
    )
    monkeypatch.setattr(M, "detect_torch_cuda_runtime_line", lambda: "cuda13")
    host = M.detect_host()
    assert host.has_usable_nvidia is True
    assert host.compute_caps == ("10.0",)
    assert host.driver_cuda_version == (13, 0)
    assert host.torch_runtime_line == "cuda13"


@pytest.mark.parametrize(
    "system,machine",
    [("Plan9", "x86_64"), ("Linux", "sparc64"), ("Linux", "armv7l")],
)
def test_detect_host_unsupported(monkeypatch, system, machine):
    monkeypatch.setattr(M.platform, "system", lambda: system)
    monkeypatch.setattr(M.platform, "machine", lambda: machine)
    with pytest.raises(PrebuiltFallback):
        M.detect_host()


def test_detect_rocm_gfx_wsl_env_and_skips_cpu_agent(monkeypatch):
    # rocminfo lists the CPU agent (gfx000) before the real GPU (gfx1100); the
    # probe must skip gfx000 and pick gfx1100, and pass HSA_ENABLE_DXG_DETECTION
    # so a WSL /dev/dxg host enumerates at all.
    rocminfo_out = "  Name:  gfx000\n  Marketing: CPU\n  Name:  gfx1100\n"
    captured = {}
    # The gfx picker honors *_VISIBLE_DEVICES; clear them so this asserts the
    # default (first GPU) path deterministically regardless of the CI env.
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)

    def fake_run(cmd, *a, **kw):
        captured["env"] = kw.get("env")
        return types.SimpleNamespace(returncode = 0, stdout = rocminfo_out, stderr = "")

    monkeypatch.setattr(
        M.shutil, "which", lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None
    )
    monkeypatch.setattr(M.subprocess, "run", fake_run)
    has_rocm, gfx = M._detect_rocm_gfx()
    assert (has_rocm, gfx) == (True, "gfx1100")
    assert captured["env"]["HSA_ENABLE_DXG_DETECTION"] == "1"


def test_detect_rocm_gfx_cpu_only_returns_false(monkeypatch):
    # Only the CPU agent + a generic ISA line: no real GPU -> not ROCm.
    out = "  Name:  gfx000\n  Name:  gfx11-generic\n"
    monkeypatch.setattr(
        M.shutil, "which", lambda name: "/usr/bin/rocminfo" if name == "rocminfo" else None
    )
    monkeypatch.setattr(
        M.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(returncode = 0, stdout = out, stderr = ""),
    )
    assert M._detect_rocm_gfx() == (False, None)


# ── Backend override + auto-detect ──
def test_auto_detect_backend_prefers_metal_on_apple_silicon():
    assert M.auto_detect_backend(_host("macos", "arm64")) == "metal"
    # Intel mac has no Metal bundle in the P0 matrix -> cpu.
    assert M.auto_detect_backend(_host("macos", "x64")) == "cpu"


def test_auto_detect_backend_nvidia_then_rocm_then_cpu():
    assert M.auto_detect_backend(_host("linux", "x64", has_usable_nvidia = True)) == "cuda"
    assert M.auto_detect_backend(_host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1100")) == "rocm"
    assert M.auto_detect_backend(_host("linux", "x64")) == "cpu"


def test_resolve_backend_override_and_cpu_fallback():
    host = _host("linux", "x64", has_usable_nvidia = True)
    assert M.resolve_backend(host, "auto", cpu_fallback = False) == "cuda"
    assert M.resolve_backend(host, "vulkan", cpu_fallback = False) == "vulkan"
    # --cpu-fallback wins over both detection and an explicit backend.
    assert M.resolve_backend(host, "cuda", cpu_fallback = True) == "cpu"
    with pytest.raises(PrebuiltFallback):
        M.resolve_backend(host, "tpu", cpu_fallback = False)


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


# ── Manifest parse + rejection ──
def test_parse_manifest_ok_and_basic_selection():
    cpu_asset = "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"
    manifest = M.parse_manifest(_manifest([_artifact("linux", "x64", "cpu", cpu_asset, "a" * 64)]))
    assert manifest["component"] == "whisper.cpp"
    assert manifest["studio_protocol"] == STUDIO_PROTOCOL
    host = _host("linux", "x64")
    # CPU/Metal need no per-GPU coverage: first os/arch/backend match / None.
    assert M.select_artifact(manifest, host, "cpu")["asset"] == cpu_asset
    assert M.select_artifact(manifest, host, "metal") is None


def test_parse_manifest_rejects_wrong_component():
    with pytest.raises(PrebuiltFallback):
        M.parse_manifest(_manifest([], component = "llama.cpp"))


def test_parse_manifest_rejects_unknown_schema():
    with pytest.raises(PrebuiltFallback):
        M.parse_manifest(_manifest([], schema_version = 999))


def test_parse_manifest_rejects_non_object_and_missing_artifacts():
    with pytest.raises(PrebuiltFallback):
        M.parse_manifest(["not", "a", "dict"])
    with pytest.raises(PrebuiltFallback):
        M.parse_manifest({"schema_version": 1, "component": "whisper.cpp"})


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


@pytest.mark.parametrize(
    "caps,driver,expected",
    [
        (["10.0"], (13, 0), "cuda13-newer"),  # B200 sm_100: legacy/older rejected
        (["10.0"], (12, 8), "cuda12-newer"),  # B200 on a CUDA-12 driver
        (["9.0"], (13, 0), "cuda13-newer"),  # H100 sm_90
        (["8.9"], (13, 0), "cuda13-older"),  # 4090 sm_89: tightest covering (75-89)
        (["8.0"], (13, 0), "cuda13-older"),  # A100 sm_80
        (["7.5"], (13, 0), "cuda13-older"),  # T4 sm_75
        (["7.5"], (12, 4), "cuda12-older"),  # T4 on a CUDA-12 driver
        ([], (13, 0), "cuda13-portable"),  # unknown caps -> portable only
    ],
)
def test_cuda_selection_matrix(caps, driver, expected):
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = tuple(caps),
        driver_cuda_version = driver,
    )
    artifact = M.select_artifact(_cuda_manifest(), host, "cuda")
    assert artifact is not None
    assert artifact["asset"] == f"whisper-linux-x64-{expected}.tar.gz"


def test_cuda_multi_gpu_requires_single_covering_artifact():
    # T4 (75) + B200 (100): only a portable bundle covers both.
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("7.5", "10.0"),
        driver_cuda_version = (13, 0),
    )
    assert (
        M.select_artifact(_cuda_manifest(), host, "cuda")["asset"]
        == "whisper-linux-x64-cuda13-portable.tar.gz"
    )


def test_cuda_on_disk_runtime_gates_the_line(monkeypatch):
    # B200 under a CUDA-13 driver, but only cuda12 runtime libs on disk (e.g.
    # torch-cuda12): the bundles don't ship libcudart/libcublas, so the cuda13
    # line is unusable and selection must fall to cuda12-newer.
    monkeypatch.setattr(
        M._runtime_libs, "detected_cuda_runtime_lines", lambda *, is_windows: ["cuda12"]
    )
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    assert (
        M.select_artifact(_cuda_manifest(), host, "cuda")["asset"]
        == "whisper-linux-x64-cuda12-newer.tar.gz"
    )


def test_cuda_no_on_disk_runtime_returns_none(monkeypatch):
    # No CUDA runtime libraries anywhere -> no usable line -> None (CPU fallback).
    monkeypatch.setattr(M._runtime_libs, "detected_cuda_runtime_lines", lambda *, is_windows: [])
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    assert M.select_artifact(_cuda_manifest(), host, "cuda") is None


def test_cuda_selection_stable_under_manifest_shuffle():
    import random

    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = (13, 0),
    )
    manifest = _cuda_manifest()
    picks = set()
    for seed in range(8):
        shuffled = dict(manifest)
        arts = list(manifest["artifacts"])
        random.Random(seed).shuffle(arts)
        shuffled["artifacts"] = arts
        picks.add(M.select_artifact(shuffled, host, "cuda")["asset"])
    assert picks == {"whisper-linux-x64-cuda13-newer.tar.gz"}


def test_cuda_no_driver_version_returns_none_then_cpu_fallback():
    # nvidia-smi present but no parseable CUDA version -> no CUDA line -> None,
    # and select_artifact_with_cpu_fallback then serves the CPU bundle.
    host = _host(
        "linux",
        "x64",
        has_usable_nvidia = True,
        compute_caps = ("10.0",),
        driver_cuda_version = None,
    )
    manifest = _cuda_manifest()
    assert M.select_artifact(manifest, host, "cuda") is None
    artifact, backend, used_cpu = M.select_artifact_with_cpu_fallback(manifest, host, "cuda")
    assert backend == "cpu" and used_cpu is True


# ── Coverage-aware ROCm selection ──
def _rocm_manifest() -> dict:
    artifacts = [_artifact("linux", "x64", "cpu", "whisper-linux-x64-cpu.tar.gz", "a" * 64)]
    families = [
        ("gfx103X", ["gfx1030", "gfx1031", "gfx1032"]),
        ("gfx110X", ["gfx1100", "gfx1101", "gfx1102"]),
        ("gfx120X", ["gfx1200", "gfx1201"]),
        ("gfx1151", ["gfx1151"]),
        ("gfx90a", ["gfx90a"]),
        ("gfx908", ["gfx908"]),
    ]
    for label, mapped in families:
        artifacts.append(
            _artifact(
                "linux",
                "x64",
                "rocm",
                f"whisper-linux-x64-rocm-{label}.tar.gz",
                "b" * 64,
                gfx_target = label,
                mapped_targets = mapped,
            )
        )
    return M.parse_manifest(_manifest(artifacts))


@pytest.mark.parametrize(
    "gfx,expected",
    [
        ("gfx1030", "gfx103X"),
        ("gfx1100", "gfx110X"),
        ("gfx1201", "gfx120X"),
        ("gfx1151", "gfx1151"),
        ("gfx90a", "gfx90a"),
        ("gfx908", "gfx908"),
        ("gfx110X", "gfx110X"),  # family token forwarded by the updater
    ],
)
def test_rocm_family_and_exact_matching(gfx, expected):
    host = _host("linux", "x64", has_rocm = True, rocm_gfx = gfx)
    assert (
        M.select_artifact(_rocm_manifest(), host, "rocm")["asset"]
        == f"whisper-linux-x64-rocm-{expected}.tar.gz"
    )


def test_rocm_unbuilt_arch_in_family_returns_none():
    # gfx1033 is in the gfx103 family but not a built mapped_target -> no bundle.
    host = _host("linux", "x64", has_rocm = True, rocm_gfx = "gfx1033")
    assert M.select_artifact(_rocm_manifest(), host, "rocm") is None


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


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, "../escape.txt", b"pwn")
    with pytest.raises(PrebuiltFallback):
        M.extract_archive(archive, tmp_path / "out")


def test_extract_rejects_absolute_member(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/passwd", b"pwn")
    with pytest.raises(PrebuiltFallback):
        M.extract_archive(archive, tmp_path / "out")


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
def test_resolve_prebuilt_reports_available(tmp_path, monkeypatch):
    host = _host("linux", "x64")
    asset = M.whisper_asset_name(RELEASE_TAG, host, "cpu")
    _install_env(monkeypatch, tmp_path, host, asset = asset, sha256 = "a" * 64)
    payload = M.resolve_prebuilt(
        host,
        published_repo = "unslothai/whisper.cpp",
        published_release_tag = None,
        backend = "cpu",
        cpu_fallback = False,
    )
    assert payload["prebuilt_available"] is True
    assert payload["repo"] == "unslothai/whisper.cpp"
    assert payload["release_tag"] == RELEASE_TAG
    assert payload["backend"] == "cpu"
    assert payload["asset"] == asset
    assert payload["os"] == "linux" and payload["arch"] == "x64"


def test_resolve_prebuilt_reports_unavailable(monkeypatch):
    host = _host("linux", "x64")

    def boom(repo, *, published_release_tag = None):
        raise PrebuiltFallback("no release")

    monkeypatch.setattr(M, "fetch_release_for_install", boom)
    payload = M.resolve_prebuilt(
        host,
        published_repo = "unslothai/whisper.cpp",
        published_release_tag = None,
        backend = "cpu",
        cpu_fallback = False,
    )
    assert payload == {"prebuilt_available": False, "repo": "unslothai/whisper.cpp"}


def test_emit_resolver_output_json_and_plain(capsys):
    payload = {"prebuilt_available": True, "asset": "whisper-x-linux-x64-cpu.tar.gz"}
    M.emit_resolver_output(payload, output_format = "json")
    assert json.loads(capsys.readouterr().out.strip())["asset"] == payload["asset"]
    M.emit_resolver_output(payload, output_format = "plain")
    assert capsys.readouterr().out.strip() == payload["asset"]


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


# ── macOS min_os enforcement ──
def test_macos_min_os_filters_incompatible_metal():
    host = _host("macos", "arm64", macos_version = (14, 0))
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact(
                    "macos",
                    "arm64",
                    "metal",
                    "whisper-macos-arm64-metal-new.tar.gz",
                    "b" * 64,
                    min_os = "macos-15.0",
                ),
                _artifact(
                    "macos",
                    "arm64",
                    "metal",
                    "whisper-macos-arm64-metal.tar.gz",
                    "a" * 64,
                    min_os = "macos-13.0",
                ),
            ]
        )
    )
    # host 14.0 can't load the 15.0 bundle; the 13.0 bundle is picked instead.
    assert M.select_artifact(manifest, host, "metal")["asset"] == "whisper-macos-arm64-metal.tar.gz"


def test_macos_min_os_excludes_all_when_host_too_old():
    host = _host("macos", "arm64", macos_version = (13, 0))
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact(
                    "macos",
                    "arm64",
                    "metal",
                    "whisper-macos-arm64-metal-new.tar.gz",
                    "b" * 64,
                    min_os = "macos-15.0",
                )
            ]
        )
    )
    assert M.select_artifact(manifest, host, "metal") is None


def test_macos_min_os_unknown_host_version_keeps_artifact():
    # Unknown host macOS version -> defer to runtime validation, don't reject.
    host = _host("macos", "arm64", macos_version = None)
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact(
                    "macos",
                    "arm64",
                    "metal",
                    "whisper-macos-arm64-metal-new.tar.gz",
                    "b" * 64,
                    min_os = "macos-15.0",
                )
            ]
        )
    )
    assert (
        M.select_artifact(manifest, host, "metal")["asset"]
        == "whisper-macos-arm64-metal-new.tar.gz"
    )


def test_macos_min_os_accepts_bare_version_format():
    # A bare "14.0" (no 'macos-' prefix) must still parse, for forward-compat.
    host = _host("macos", "arm64", macos_version = (13, 0))
    manifest = M.parse_manifest(
        _manifest(
            [
                _artifact(
                    "macos",
                    "arm64",
                    "metal",
                    "whisper-macos-arm64-metal.tar.gz",
                    "a" * 64,
                    min_os = "14.0",
                )
            ]
        )
    )
    assert M.select_artifact(manifest, host, "metal") is None  # 13.0 < 14.0


def test_macos_min_os_ok_helper_handles_prefix_and_bare():
    host14 = _host("macos", "arm64", macos_version = (14, 0))
    # The live manifest format is 'macos-<ver>'; the prefix must be stripped.
    assert M._macos_min_os_ok(host14, "macos-14.0") is True
    assert M._macos_min_os_ok(host14, "macos-15.0") is False
    assert M._macos_min_os_ok(host14, "13.3") is True  # bare also parses
    assert M._macos_min_os_ok(host14, None) is True  # unknown -> defer
    assert M._macos_min_os_ok(_host("macos", "arm64"), "macos-15.0") is True  # host ver unknown


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


# ── existing_install_matches contract ──
def test_existing_install_matches_requires_marker_and_binary(tmp_path):
    host = _host("linux", "x64")
    selection = M.InstallSelection(
        published_repo = "unslothai/whisper.cpp",
        release_tag = RELEASE_TAG,
        upstream_tag = UPSTREAM_TAG,
        source_commit = SOURCE_COMMIT,
        asset = "whisper-x-linux-x64-cpu.tar.gz",
        asset_sha256 = "a" * 64,
        backend = "cpu",
        runtime_line = None,
        coverage = {},
        studio_protocol = STUDIO_PROTOCOL,
    )
    install_dir = tmp_path / "whisper.cpp"
    assert M.existing_install_matches(install_dir, host, selection) is False  # no marker
    # marker present but binary missing -> not a match
    server = M.whisper_server_path(install_dir, host)
    server.parent.mkdir(parents = True, exist_ok = True)
    M.write_prebuilt_metadata(install_dir, selection)
    assert M.existing_install_matches(install_dir, host, selection) is False
    server.write_text("#!/bin/sh\n")
    assert M.existing_install_matches(install_dir, host, selection) is True
    # a different backend has a different fingerprint -> not a match
    other = M.InstallSelection(**{**selection.__dict__, "backend": "cuda"})
    assert M.existing_install_matches(install_dir, host, other) is False
