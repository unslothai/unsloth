#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform whisper.cpp (whisper-server) prebuilt installer for Unsloth Studio.

Downloads a per-platform whisper.cpp bundle published by the Unsloth fork
(``unslothai/whisper.cpp``) into an isolated ``<UNSLOTH_HOME>/whisper.cpp``,
never a system whisper.cpp. The canonical install target matches the
sidecar/build-script contract in ``stt_ggml_sidecar.py``:

    <install-dir>/build/bin/whisper-server              (Unix)
    <install-dir>/build/bin/Release/whisper-server.exe  (Windows)

Bundles are dynamically linked: ``whisper-server`` needs ``libwhisper``,
``libggml*`` and any GPU backend libraries. The binaries carry ``RUNPATH=$ORIGIN``
so the installer co-locates every shared library from the archive into the same
``build/bin`` directory and marks the server executable on Unix.

Archives are verified against the release's own ``whisper-prebuilt-sha256.json``
checksum index (same model as ``install_llama_prebuilt.py``). An asset absent
from the index, or a release without it, fails closed (setup reports the prebuilt
unavailable and local dictation uses Transformers STT). Being a same-origin
checksum it proves integrity, not authenticity; pair the release with GitHub
artifact attestations for provenance.

Release resolution prefers the download host (``github.com/<repo>/releases/...``),
fetching the manifest + checksum index with zero ``api.github.com`` calls (that
API is rate-limited to 60 req/hour unauthenticated); the GitHub API is only a
fallback on a 404 / malformed asset.

The whole component-agnostic flow (verified downloads, safe extraction, the
install lock, release resolution, checksum-index handling, the install driver,
the resolve probe) lives in ``prebuilt_core.py`` and is shared with
``install_llama_prebuilt.py``. This module keeps only the whisper specifics: the
host mapping, install-tree layout, slim pairing with the installed llama.cpp ggml
runtime, the pinned-release CPU escape hatch, the marker filename, and the CLI.
Every retained public name is a thin wrapper tests can monkeypatch so the core
observes the patch.

Selection is slim-only for current releases (v1.9.1-unsloth.2+): every published
bundle is a slim (ggml-less) whisper paired to the llama.cpp prebuilt, which
provides all ggml backends. The per-accelerator fat selection chain was deleted
with the fat bundles; the one legacy shape still honored is the published fat CPU
bundle of an explicitly pinned pre-slim release.

Mirrors ``install_node_prebuilt.py`` / ``install_llama_prebuilt.py``. Exit codes:
0 success (or already current), 1 error, 2 incompatible paired release, 3 busy. A re-run
that already matches logs "already matches" and returns 0 without downloading
(the scripts grep it).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Put studio/ on sys.path so install_llama_prebuilt / prebuilt_core resolve
# whether run as a script (from any cwd) or imported by the tests.
_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import install_llama_prebuilt as llama  # noqa: E402
import prebuilt_core as core  # noqa: E402

# Shared machinery re-exported as module globals; the wrappers below resolve
# these by bare name at call time so tests can monkeypatch them here.
PrebuiltFallback = core.PrebuiltFallback
BusyInstallConflict = core.BusyInstallConflict
auth_headers = llama.auth_headers
download_bytes = llama.download_bytes
download_file = llama.download_file
fetch_json = llama.fetch_json
sha256_file = core.sha256_file
_URL_OPENER = core._URL_OPENER
install_lock = core.install_lock
install_lock_path = core.install_lock_path
parse_macos_version = core.parse_macos_version
release_asset_map = core.release_asset_map
release_asset_download_url = core.release_asset_download_url
compute_install_fingerprint = core.compute_install_fingerprint
artifact_coverage = core.artifact_coverage
emit_resolver_output = core.emit_resolver_output
_valid_sha256 = core.valid_sha256
_swap_into_place = core.swap_into_place
InstallSelection = core.InstallSelection
ReleaseBundle = core.ReleaseBundle
llama_detect_host = llama.detect_host
installed_llama_runtime = llama.installed_llama_runtime

# Late-binding seam for prebuilt_core: name lookups hit this module's globals
# first (so monkeypatches apply), then the core defaults.
_OPS = core.ModuleOps(globals())


# Resolver mode keeps stdout to the JSON payload only (setup.sh parses it);
# main() flips this on for the install path so setup surfaces progress.
_LOG_TO_STDOUT = False


class ReleaseCompatibilityError(PrebuiltFallback):
    """A valid slim release cannot pair with this host's installed runtime."""


def log(message: str) -> None:
    print(f"[whisper-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr)


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INCOMPATIBLE = 2
EXIT_BUSY = 3

COMPONENT = "whisper.cpp"
SCHEMA_VERSION = 1
USER_AGENT = "unsloth-studio-whisper-prebuilt"

DEFAULT_PUBLISHED_REPO = "unslothai/whisper.cpp"
# Release assets published by the fork's prebuilt CI.
MANIFEST_ASSET_NAME = "whisper-prebuilt-manifest.json"
SHA256_ASSET_NAME = "whisper-prebuilt-sha256.json"

METADATA_FILENAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# Backends the installer can select: accelerator identities for the slim pairing
# (which llama ggml module must exist) and the marker. Only "cpu" can match a fat
# artifact (the pinned pre-slim escape hatch).
SUPPORTED_BACKENDS = ("cpu", "cuda", "metal", "vulkan", "rocm")

# Fallback the core applies on a GPU-selection miss: retry with cpu so the slim
# bundle can pair via the llama cpu modules, or -- on a pinned pre-slim release --
# the published fat CPU bundle installs. A miss is an install error unless the
# valid slim release specifically requires a different paired llama release.
FALLBACK_BACKEND = "cpu"

# Backends whose slim (ggml-less) whisper bundle can ride the installed llama.cpp
# prebuilt's ggml runtime. All are eligible: the next whisper release ships only
# slim bundles, so cpu and metal pair like the GPUs do.
SLIM_ELIGIBLE_BACKENDS = ("cpu", "cuda", "metal", "rocm", "vulkan")

# ggml backend module the llama bin dir must provide per accelerator, keyed by
# asset os token, following llama's bundle layout / health globs. A backend with
# no glob for the host os (metal off macOS) is never slim there. llama's x64 cpu
# bundles ship per-arch libggml-cpu-<variant> modules, macOS a single
# libggml-cpu; the cpu globs cover both.
SLIM_BACKEND_MODULE_GLOBS = {
    "cpu": {
        "linux": "libggml-cpu*.so*",
        "windows": "ggml-cpu*.dll",
        "macos": "libggml-cpu*.dylib",
    },
    "cuda": {"linux": "libggml-cuda.so*", "windows": "ggml-cuda.dll"},
    "metal": {"macos": "libggml-metal*.dylib"},
    "rocm": {"linux": "libggml-hip.so*", "windows": "*hip*.dll"},
    "vulkan": {"linux": "libggml-vulkan.so*", "windows": "ggml-vulkan.dll"},
}

# Everything the slim wiring mirrors from the llama bin dir: the core ggml
# sonames plus every dlopen'd backend module (CPU variants included). libggml*
# matches .so and .dylib alike; ggml*.dll covers Windows. libomp*.dll rides along
# because llama's clang-built windows-arm64 ggml-base.dll imports
# libomp140.aarch64.dll, shipped in the llama bundle but never a system DLL, so
# the loader needs it next to whisper-server.exe (MSVC x64 uses System32
# vcomp140.dll; Linux ggml needs system libgomp.so.1, which llama already requires).
SLIM_GGML_LIBRARY_GLOBS = ("libggml*", "ggml*.dll", "libomp*.dll")
SLIM_ROCM_LIBRARY_GLOBS = (
    "libamd*.so*",
    "libhip*.so*",
    "libhsa*.so*",
    "libroc*.so*",
)
SLIM_ROCM_RUNTIME_DIRS = ("hipblaslt", "rocblas")
SLIM_RUNTIME_WIRING_VERSION = 2

INSTALL_STAGING_ROOT_NAME = ".staging"

# Master switch for the staged runtime smoke test (start whisper-server + a tiny
# transcription) before the atomic activate. Off by default so a bundle whose GPU
# forward pass JIT-compiles kernels does not stall every install; the check and
# its CPU-asset retry stay intact -- set True to re-enable.
_RUN_STAGED_PREBUILT_VALIDATION = False


# ── Host detection (probes shared with install_llama_prebuilt) ──
@dataclass(frozen = True)
class HostInfo:
    system: str  # platform.system()
    machine: str  # lowered platform.machine()
    whisper_os: str  # asset token: linux | macos | windows
    whisper_arch: str  # asset token: x64 | arm64
    archive_ext: str  # .tar.gz | .zip
    is_windows: bool
    is_macos: bool
    is_apple_silicon: bool
    has_usable_nvidia: bool = False
    has_rocm: bool = False
    rocm_gfx: str | None = None
    # (major, minor) from platform.mac_ver(); None off macOS or if unparseable.
    # Enforces a macOS artifact's min_os so we never pick a bundle that cannot
    # load on this OS version.
    macos_version: tuple[int, int] | None = None


def host_from_llama(base: Any) -> HostInfo:
    """Map install_llama_prebuilt's detected host onto the whisper asset tokens.
    Raises PrebuiltFallback on an unsupported OS/arch."""
    system = base.system
    machine = base.machine.lower()
    if system == "Linux":
        whisper_os = "linux"
    elif system == "Darwin":
        whisper_os = "macos"
    elif system == "Windows":
        whisper_os = "windows"
    else:
        raise PrebuiltFallback(f"unsupported operating system for whisper.cpp prebuilt: {system}")

    if machine in {"x86_64", "amd64", "x64"}:
        whisper_arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        whisper_arch = "arm64"
    else:
        raise PrebuiltFallback(f"unsupported CPU architecture for whisper.cpp prebuilt: {machine}")

    return HostInfo(
        system = system,
        machine = machine,
        whisper_os = whisper_os,
        whisper_arch = whisper_arch,
        archive_ext = ".zip" if base.is_windows else ".tar.gz",
        is_windows = base.is_windows,
        is_macos = base.is_macos,
        is_apple_silicon = base.is_macos and whisper_arch == "arm64",
        has_usable_nvidia = base.has_usable_nvidia,
        has_rocm = base.has_rocm,
        rocm_gfx = base.rocm_gfx_target,
        macos_version = base.macos_version,
    )


def detect_host() -> HostInfo:
    return host_from_llama(llama_detect_host())


def apply_host_overrides(
    host: HostInfo,
    *,
    has_rocm: bool = False,
    rocm_gfx: str | None = None,
    force_cpu: bool = False,
) -> HostInfo:
    """Apply CLI overrides (setup forwards hardware hints) onto a detected host."""
    if force_cpu:
        return replace(
            host,
            has_usable_nvidia = False,
            has_rocm = False,
            rocm_gfx = None,
            is_apple_silicon = False,
        )
    updates: dict[str, Any] = {}
    if has_rocm or rocm_gfx:
        # --rocm-gfx implies --has-rocm (llama parity); otherwise the host stays on
        # its CUDA/CPU path and never picks the ROCm bundle. Drop CUDA detection too.
        updates["has_rocm"] = True
        updates["has_usable_nvidia"] = False
    if rocm_gfx:
        updates["rocm_gfx"] = rocm_gfx
    return replace(host, **updates) if updates else host


def host_platform_tokens(host: HostInfo) -> tuple[str, str]:
    """Canonical (os, arch) asset tokens the core uses for matching/reporting."""
    return host.whisper_os, host.whisper_arch


# ── Backend selection ──
def auto_detect_backend(host: HostInfo) -> str:
    return core.auto_detect_backend(_OPS, host)


def resolve_backend(host: HostInfo, requested: str | None, *, cpu_fallback: bool) -> str:
    return core.resolve_backend(_OPS, host, requested, cpu_fallback = cpu_fallback)


# ── Asset naming (pure, unit tested) ──
def whisper_asset_name(release_tag: str, host: HostInfo, accel: str) -> str:
    """e.g. whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz.

    `accel` is the asset-filename accelerator token (cpu, metal, cuda12, ...),
    finer-grained than the coarse backend. An install's authoritative asset name
    comes from the release manifest; this constructor is for defaults/diagnostics.
    """
    tag = release_tag if release_tag.startswith("v") else f"v{release_tag}"
    return f"whisper-{tag}-{host.whisper_os}-{host.whisper_arch}-{accel}{host.archive_ext}"


# ── Manifest (release-side artifact catalogue; generic parser in the core) ──
def validate_schema_version(payload: dict[str, Any], *, label: str) -> None:
    core.validate_schema_version(
        payload, label = label, schema_version = SCHEMA_VERSION, error = PrebuiltFallback
    )


def parse_manifest(payload: Any, *, label: str = MANIFEST_ASSET_NAME) -> dict[str, Any]:
    return core.parse_manifest(_OPS, payload, label = label)


def _macos_min_os_ok(host: HostInfo, min_os: Any) -> bool:
    return core.macos_min_os_ok(_OPS, host, min_os)


def macos_min_os_ok(host: HostInfo, min_os: Any) -> bool:
    return _macos_min_os_ok(host, min_os)


def _artifacts_for_host(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> list[dict[str, Any]]:
    return core.artifacts_for_host(_OPS, manifest, host, backend)


def artifacts_for_host(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> list[dict[str, Any]]:
    return _artifacts_for_host(manifest, host, backend)


# ── Slim selection (paired with the installed llama.cpp ggml runtime) ──
def slim_pairing_for_artifact(
    artifact: dict[str, Any], host: HostInfo, backend: str
) -> tuple[Path, str] | None:
    """(llama bin dir, llama release tag) when the installed llama runtime can
    back this slim artifact, else None. Each failed gate logs why; the caller
    then retries via CPU or reports the prebuilt unavailable."""
    asset = artifact.get("asset")
    runtime = installed_llama_runtime()
    if runtime is None:
        log(f"slim_selection: {asset} skipped: no managed llama.cpp prebuilt install")
        return None
    llama_bin_dir, llama_tag, _profile = runtime
    requires_tag = artifact.get("requires_llama_tag")
    if not isinstance(requires_tag, str) or requires_tag != llama_tag:
        log(
            f"slim_selection: {asset} skipped: installed llama tag {llama_tag!r} "
            f"!= required {requires_tag!r}"
        )
        return None
    sonames = artifact.get("requires_ggml_sonames")
    if not isinstance(sonames, list) or not sonames:
        log(f"slim_selection: {asset} skipped: manifest lists no requires_ggml_sonames")
        return None
    missing = [str(name) for name in sonames if not (llama_bin_dir / str(name)).is_file()]
    if missing:
        log(f"slim_selection: {asset} skipped: llama runtime missing {', '.join(missing)}")
        return None
    module_glob = SLIM_BACKEND_MODULE_GLOBS.get(backend, {}).get(host.whisper_os)
    if module_glob is None:
        log(f"slim_selection: {asset} skipped: no {backend} ggml module on {host.whisper_os}")
        return None
    if not any(path.is_file() for path in llama_bin_dir.glob(module_glob)):
        log(f"slim_selection: {asset} skipped: llama runtime has no {module_glob} module")
        return None
    log(f"slim_selection: selected {asset} paired with llama {llama_tag} at {llama_bin_dir}")
    return llama_bin_dir, llama_tag


def select_slim_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """The paired slim artifact for any backend (cpu and metal included), or None.
    Slim assets carry backend "slim" so the CPU escape hatch's fat os/arch/backend
    matching never sees them. When the release ships slim candidates but none
    pair, log the actionable reason: on a slim-only release nothing else selects,
    setup reports the prebuilt unavailable, and local dictation stays on
    Transformers STT."""
    if backend not in SLIM_ELIGIBLE_BACKENDS:
        return None
    os_token, arch_token = host_platform_tokens(host)
    candidates = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("os") == os_token
        and artifact.get("arch") == arch_token
        and artifact.get("backend") == "slim"
        and artifact.get("install_kind") == "slim"
        and (not host.is_macos or _macos_min_os_ok(host, artifact.get("min_os")))
    ]
    for artifact in candidates:
        if slim_pairing_for_artifact(artifact, host, backend) is not None:
            return artifact
    if candidates:
        required_tag = candidates[0].get("requires_llama_tag")
        log(f"slim bundle requires llama.cpp {required_tag}; install or update llama.cpp first")
    return None


def select_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """Slim-only: every backend resolves to the paired slim artifact. The one
    legacy shape still honored is the published fat CPU bundle of a pinned pre-slim
    release, reached by a GPU-backend miss through the core's CPU fallback retry.
    No other fat artifact is ever selected."""
    slim = select_slim_artifact(manifest, host, backend)
    if slim is not None:
        return slim
    if backend != "cpu":
        return None
    candidates = _artifacts_for_host(manifest, host, "cpu")
    return candidates[0] if candidates else None


def select_artifact_with_cpu_fallback(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> tuple[dict[str, Any], str, bool]:
    try:
        return core.select_artifact_with_fallback(_OPS, manifest, host, backend)
    except PrebuiltFallback as exc:
        reason = _slim_release_incompatibility(manifest, host)
        if reason is not None:
            raise ReleaseCompatibilityError(reason) from exc
        raise


def _slim_release_incompatibility(manifest: dict[str, Any], host: HostInfo) -> str | None:
    """Explain a valid platform slim release that this runtime cannot pair.

    This intentionally excludes missing files, malformed manifests, checksum
    failures, and unsupported platforms. Those are operational errors and must
    never be converted into the update path's kept-existing-runtime success.
    """
    os_token, arch_token = host_platform_tokens(host)
    candidates = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("os") == os_token
        and artifact.get("arch") == arch_token
        and artifact.get("backend") == "slim"
        and artifact.get("install_kind") == "slim"
    ]
    if not candidates:
        return None
    os_compatible = [
        artifact
        for artifact in candidates
        if not host.is_macos or _macos_min_os_ok(host, artifact.get("min_os"))
    ]
    if not os_compatible:
        required = candidates[0].get("min_os")
        return f"slim bundle requires macOS {required}; this host is older"
    runtime = installed_llama_runtime()
    if runtime is None:
        return None
    installed_tag = runtime[1]
    required_tags = {
        artifact.get("requires_llama_tag")
        for artifact in os_compatible
        if isinstance(artifact.get("requires_llama_tag"), str)
    }
    if required_tags and installed_tag not in required_tags:
        required_tag = sorted(required_tags)[0]
        return (
            f"slim bundle requires llama.cpp {required_tag}; "
            f"installed llama.cpp is {installed_tag}"
        )
    return None


def select_artifact_with_fallback(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> tuple[dict[str, Any], str, bool]:
    return select_artifact_with_cpu_fallback(manifest, host, backend)


# ── Release checksum index (trust anchor: the release's own sha256 asset) ──
def parse_release_checksums(repo: str, release_tag: str, payload: Any) -> dict[str, str]:
    return core.parse_release_checksums(_OPS, repo, release_tag, payload)


def fetch_release_checksums(bundle: ReleaseBundle) -> dict[str, str]:
    return core.fetch_release_checksums(_OPS, bundle)


def expected_sha256_for(
    checksums: dict[str, str],
    asset_name: str,
    *,
    manifest_sha256: str | None = None,
) -> str:
    return core.expected_sha256_for(_OPS, checksums, asset_name, manifest_sha256 = manifest_sha256)


# ── Verified download (retries once on checksum mismatch) ──
def download_file_verified(
    url: str, destination: Path, *, expected_sha256: str, label: str
) -> None:
    core.download_file_verified_strict(
        _OPS, url, destination, expected_sha256 = expected_sha256, label = label
    )


# ── GitHub release resolution ──
def github_release(repo: str, tag: str) -> dict[str, Any]:
    return core.github_release(_OPS, repo, tag, error = PrebuiltFallback)


def fetch_release_bundle(repo: str, release_tag: str) -> ReleaseBundle:
    return core.fetch_release_bundle(_OPS, repo, release_tag)


def asset_download_url(bundle: ReleaseBundle, asset_name: str) -> str:
    return core.asset_download_url(_OPS, bundle, asset_name)


# ── Download-host fast path (resolve + fetch the JSON assets with no GitHub API) ──
def _download_host_latest_release_tag(repo: str) -> str | None:
    return core.download_host_latest_release_tag(_OPS, repo)


def _download_host_json(url: str) -> Any:
    return core.fetch_download_host_json(_OPS, url)


def _resolve_release_via_download_host(
    repo: str, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]] | None:
    return core.resolve_release_via_download_host(_OPS, repo, published_release_tag)


# ── Archive extraction (core's guarded extractor + tar exec-bit restore) ──
def extract_archive(archive_path: Path, destination: Path) -> None:
    """Shared guarded extraction, then restore tar exec bits (the extractor writes
    plain files; whisper-server must stay executable)."""
    core.extract_archive(archive_path, destination)
    core.restore_tar_exec_bits(archive_path, destination)


# ── Install layout ──
def server_binary_name(host: HostInfo) -> str:
    return "whisper-server.exe" if host.is_windows else "whisper-server"


def runtime_bin_dir(install_dir: Path, host: HostInfo) -> Path:
    """Canonical directory holding whisper-server + its co-located libs."""
    if host.is_windows:
        return install_dir / "build" / "bin" / "Release"
    return install_dir / "build" / "bin"


def whisper_server_path(install_dir: Path, host: HostInfo) -> Path:
    return runtime_bin_dir(install_dir, host) / server_binary_name(host)


def installed_server_path(install_dir: Path, host: HostInfo) -> Path:
    return whisper_server_path(install_dir, host)


def _locate_server_in_tree(root: Path, host: HostInfo) -> Path:
    return core.locate_server_in_tree(_OPS, root, host)


def locate_server_in_tree(root: Path, host: HostInfo) -> Path:
    return _locate_server_in_tree(root, host)


def _assemble_install_tree(bundle_root: Path, staged_root: Path, host: HostInfo) -> Path:
    """Lay out staged_root as a full install: build/bin/<server + libs>.

    Everything beside the server in the archive (shared libs, backend kernel
    subdirs, license/build-info) is co-located into the canonical bin dir so the
    server's RUNPATH=$ORIGIN resolves its libs.
    """
    bin_dir = runtime_bin_dir(staged_root, host)
    bin_dir.mkdir(parents = True, exist_ok = True)
    for entry in sorted(bundle_root.iterdir()):
        if entry.name == METADATA_FILENAME:
            continue
        dest = bin_dir / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, dest, symlinks = True)
        else:
            shutil.copy2(entry, dest, follow_symlinks = False)
    server = bin_dir / server_binary_name(host)
    if not server.exists():
        raise PrebuiltFallback("staged install is missing the whisper-server binary")
    if not host.is_windows:
        os.chmod(server, server.stat().st_mode | 0o111)
    return server


def assemble_install_tree(bundle_root: Path, staged_root: Path, host: HostInfo) -> Path:
    return _assemble_install_tree(bundle_root, staged_root, host)


def _validate_staged_server(staged_root: Path, host: HostInfo) -> None:
    """Optional pre-activate smoke test. Gated off by default (see the switch)."""
    if not _RUN_STAGED_PREBUILT_VALIDATION:
        return
    server = whisper_server_path(staged_root, host)
    env = os.environ.copy()
    bin_dir = str(runtime_bin_dir(staged_root, host))
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = bin_dir + (os.pathsep + env[var] if env.get(var) else "")
    try:
        result = subprocess.run(
            [str(server), "--help"],
            capture_output = True,
            text = True,
            timeout = 60,
            env = env,
            **llama.windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrebuiltFallback(f"staged whisper-server failed to launch: {exc}") from exc
    if result.returncode != 0:
        raise PrebuiltFallback(
            f"staged whisper-server --help exited {result.returncode}: {result.stderr.strip()}"
        )


def validate_staged_server(staged_root: Path, host: HostInfo) -> None:
    _validate_staged_server(staged_root, host)


# ── Slim install wiring (llama ggml runtime into the whisper bin dir) ──
def _elf_needed(path: Path) -> set[str] | None:
    """Read ELF DT_NEEDED names when a standard inspection tool is available."""
    commands = (("readelf", "-d"), ("objdump", "-p"))
    for command in commands:
        try:
            result = subprocess.run(
                [*command, str(path)], capture_output = True, text = True, timeout = 10
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        needed: set[str] = set()
        for line in result.stdout.splitlines():
            match = re.search(r"\(NEEDED\).*\[([^]]+)\]", line)
            if match:
                needed.add(match.group(1))
                continue
            match = re.match(r"\s*NEEDED\s+(\S+)", line)
            if match:
                needed.add(match.group(1))
        return needed
    return None


def _runtime_library_sources(llama_bin_dir: Path, backend: str | None) -> list[Path]:
    patterns = list(SLIM_GGML_LIBRARY_GLOBS)
    sources = {
        path for pattern in patterns for path in llama_bin_dir.glob(pattern) if path.is_file()
    }
    if backend == "rocm":
        # Windows llama HIP/ROCm prebuilts use a flat *.dll runtime overlay.
        # Mirror that complete no-SDK closure, not only filenames containing
        # ggml, hip, or roc: transitive runtime DLLs can have unrelated names.
        windows_dlls = {path for path in llama_bin_dir.glob("*.dll") if path.is_file()}
        if windows_dlls:
            sources.update(windows_dlls)
            return sorted(sources)

        # Linux can inspect DT_NEEDED and copy the exact packaged closure. Fall
        # back to the ROCm globs only when standard ELF tools are unavailable.
        by_name = {path.name: path for path in llama_bin_dir.iterdir() if path.is_file()}
        pending = list(sources)
        inspected_hip = False
        system_libraries = {
            "libc.so.6",
            "libdl.so.2",
            "libgcc_s.so.1",
            "libm.so.6",
            "libpthread.so.0",
            "librt.so.1",
            "libstdc++.so.6",
        }
        while pending:
            source = pending.pop()
            needed = _elf_needed(source)
            if needed is None:
                continue
            if source.name.startswith("libggml-hip"):
                inspected_hip = True
            for name in needed - system_libraries:
                dependency = by_name.get(name)
                if dependency is not None and dependency not in sources:
                    sources.add(dependency)
                    pending.append(dependency)
        if not inspected_hip:
            sources.update(
                path
                for pattern in SLIM_ROCM_LIBRARY_GLOBS
                for path in llama_bin_dir.glob(pattern)
                if path.is_file()
            )
    return sorted(sources)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_ggml_runtime(
    llama_bin_dir: Path,
    whisper_bin_dir: Path,
    *,
    backend: str | None = None,
) -> list[str]:
    """Hardlink every ggml library from the llama runtime into the whisper bin
    dir; returns the wired filenames (sorted) for the marker / sidecar launch guard.

    Hardlinks (not symlinks) on purpose: when the llama updater atomically swaps
    its install dir, the old inodes stay alive under these links, so a
    not-yet-updated whisper keeps running the exact ggml build it was installed
    against. Falls back to a copy where linking is unsupported or crosses devices.
    Re-run on every install/update so the links track the current pairing.
    """
    sources = _runtime_library_sources(llama_bin_dir, backend)
    if not any(source.name.startswith(("libggml", "ggml")) for source in sources):
        raise PrebuiltFallback(
            f"no ggml libraries found in {llama_bin_dir} to pair the slim whisper install"
        )
    whisper_bin_dir.mkdir(parents = True, exist_ok = True)
    for source in sources:
        # Follows a libggml.so.0 -> libggml.so.0.x symlink to its inode, so every
        # created name is a real hardlink surviving a dir swap.
        _link_or_copy(source, whisper_bin_dir / source.name)
    return [source.name for source in sources]


def link_runtime_directories(
    llama_bin_dir: Path, whisper_bin_dir: Path, *, backend: str, host: HostInfo
) -> list[str]:
    """Mirror GPU kernel catalogs that packaged ROCm libraries load at runtime."""
    # Windows ROCm prebuilts are a flat DLL overlay. The hipblaslt/rocblas
    # catalog directories are part of the Linux ROCm bundle layout only.
    if backend != "rocm" or host.is_windows:
        return []
    linked: list[str] = []
    for name in SLIM_ROCM_RUNTIME_DIRS:
        source_root = llama_bin_dir / name
        if not source_root.is_dir():
            raise PrebuiltFallback(f"paired ROCm runtime is missing its {name} kernel catalog")
        files = [path for path in source_root.rglob("*") if path.is_file()]
        if not files:
            raise PrebuiltFallback(f"paired ROCm runtime has an empty {name} kernel catalog")
        for source in files:
            _link_or_copy(source, whisper_bin_dir / name / source.relative_to(source_root))
        linked.append(name)
    return linked


def prepare_runtime_payload(staged_root: Path, host: HostInfo, selection: Any) -> Any:
    """Slim installs wire the paired llama ggml runtime into the staged bin dir
    before validation so the activated tree is self-contained; the returned
    selection carries the wired filenames for the marker. Fat installs need
    nothing (the archive carries its ggml) and return None."""
    if getattr(selection, "install_kind", None) != "slim" or not selection.linked_from:
        return None
    source = Path(selection.linked_from)
    destination = runtime_bin_dir(staged_root, host)
    linked = link_ggml_runtime(source, destination, backend = selection.backend)
    linked_dirs = link_runtime_directories(
        source,
        destination,
        backend = selection.backend,
        host = host,
    )
    log(f"slim install: hardlinked {len(linked)} ggml libraries from {selection.linked_from}")
    return replace(
        selection,
        linked_libraries = tuple(linked),
        runtime_wiring_version = SLIM_RUNTIME_WIRING_VERSION,
        linked_runtime_directories = tuple(linked_dirs),
    )


# ── Metadata / marker ──
def metadata_path(install_dir: Path) -> Path:
    return install_dir / METADATA_FILENAME


def selection_from_artifact(
    *,
    published_repo: str,
    release_tag: str,
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    backend: str,
    asset_sha256: str,
) -> InstallSelection:
    selection = core.selection_from_artifact(
        _OPS,
        published_repo = published_repo,
        release_tag = release_tag,
        manifest = manifest,
        artifact = artifact,
        backend = backend,
        asset_sha256 = asset_sha256,
    )
    if artifact.get("install_kind") != "slim":
        return selection
    # A slim selection carries its pairing so the install wiring and marker know
    # which llama runtime provides the ggml libraries.
    runtime = installed_llama_runtime()
    if runtime is None or runtime[1] != artifact.get("requires_llama_tag"):
        raise PrebuiltFallback(
            "the paired llama.cpp runtime changed underneath the slim whisper selection"
        )
    llama_bin_dir, llama_tag, _profile = runtime
    return replace(
        selection,
        install_kind = "slim",
        paired_llama_tag = llama_tag,
        linked_from = str(llama_bin_dir),
    )


def write_prebuilt_metadata(install_dir: Path, selection: InstallSelection) -> None:
    core.write_prebuilt_metadata(_OPS, install_dir, selection)


def load_whisper_prebuilt_metadata(install_dir: Path) -> dict[str, Any] | None:
    return core.load_prebuilt_metadata(_OPS, install_dir)


def load_prebuilt_metadata(install_dir: Path) -> dict[str, Any] | None:
    return load_whisper_prebuilt_metadata(install_dir)


def existing_install_matches(
    install_dir: Path, host: HostInfo, selection: InstallSelection
) -> bool:
    """Core fingerprint match, hardened for repairability: a marker-matching
    install is only "current" when the server is executable (the sidecar refuses
    non-executable binaries, so setup must repair not skip) and, for slim
    installs, every wired ggml library is still present (else a deleted/moved
    llama dir leaves dictation broken while update reports up to date)."""
    if not core.existing_install_matches(_OPS, install_dir, host, selection):
        return False
    server = installed_server_path(install_dir, host)
    if not host.is_windows and not os.access(server, os.X_OK):
        log(f"existing install at {install_dir} has a non-executable server; reinstalling")
        return False
    marker = load_prebuilt_metadata(install_dir) or {}
    if marker.get("install_kind") == "slim":
        bin_dir = server.parent
        if marker.get("runtime_wiring_version") != SLIM_RUNTIME_WIRING_VERSION:
            log(f"existing slim install at {install_dir} has stale runtime wiring; reinstalling")
            return False
        linked_libraries = marker.get("linked_libraries")
        if (
            not isinstance(linked_libraries, list)
            or not linked_libraries
            or not all(
                isinstance(name, str) and name and Path(name).name == name
                for name in linked_libraries
            )
        ):
            log(f"existing slim install at {install_dir} has invalid runtime wiring; reinstalling")
            return False
        missing = [name for name in linked_libraries if not (bin_dir / name).is_file()]
        if missing:
            log(
                f"existing slim install at {install_dir} is missing wired ggml "
                f"libraries ({', '.join(missing[:4])}); reinstalling"
            )
            return False
        runtime_dirs = marker.get("linked_runtime_directories")
        if (
            marker.get("backend") == "rocm"
            and not host.is_windows
            and (
                not isinstance(runtime_dirs, list)
                or set(runtime_dirs) != set(SLIM_ROCM_RUNTIME_DIRS)
            )
        ):
            log(f"existing ROCm install at {install_dir} lacks kernel catalogs; reinstalling")
            return False
        missing_dirs = [
            name
            for name in runtime_dirs or []
            if not isinstance(name, str)
            or not (bin_dir / name).is_dir()
            or not any(path.is_file() for path in (bin_dir / name).rglob("*"))
        ]
        if missing_dirs:
            log(
                f"existing slim install at {install_dir} is missing runtime directories "
                f"({', '.join(str(name) for name in missing_dirs[:4])}); reinstalling"
            )
            return False
    return True


# ── Orchestration ──
def resolve_newest_release_tag(repo: str) -> str:
    return core.resolve_newest_release_tag(_OPS, repo)


def resolve_release_tag(published_repo: str, *, published_release_tag: str | None) -> str:
    return core.resolve_release_tag(
        _OPS, published_repo, published_release_tag = published_release_tag
    )


def fetch_release_for_install(
    repo: str, *, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]]:
    return core.fetch_release_for_install(_OPS, repo, published_release_tag = published_release_tag)


@dataclass(frozen = True)
class WhisperReleasePlan:
    bundle: ReleaseBundle
    selection: InstallSelection | None
    artifact: dict[str, Any]
    resolved_backend: str
    used_fallback: bool


def _normalized_upstream_tag(value: str) -> str:
    value = value.strip()
    return value[1:] if value[:1].lower() == "v" else value


def _bundle_matches_whisper_tag(bundle: ReleaseBundle, whisper_tag: str) -> bool:
    requested = whisper_tag.strip()
    if not requested or requested.lower() == "latest":
        return True
    upstream = bundle.manifest.get("upstream_tag")
    return isinstance(upstream, str) and _normalized_upstream_tag(
        upstream
    ) == _normalized_upstream_tag(requested)


def _published_release_tags(repo: str) -> list[str]:
    """Published release tags in newest-first order for compatibility search."""
    payload = fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    if not isinstance(payload, list):
        raise PrebuiltFallback(f"unexpected releases payload for {repo}")
    releases = [
        release
        for release in payload
        if isinstance(release, dict)
        and not release.get("draft")
        and not release.get("prerelease")
        and isinstance(release.get("tag_name"), str)
        and release.get("tag_name")
    ]
    releases.sort(key = lambda release: release.get("published_at") or "", reverse = True)
    return [str(release["tag_name"]) for release in releases]


def _fetch_release_candidate(repo: str, release_tag: str) -> ReleaseBundle:
    """Fetch one candidate manifest by its deterministic download-host URL."""
    manifest_url = release_asset_download_url(repo, release_tag, MANIFEST_ASSET_NAME)
    try:
        payload = _download_host_json(manifest_url)
        manifest = parse_manifest(payload, label = f"{MANIFEST_ASSET_NAME} in {repo}@{release_tag}")
    except Exception as exc:
        raise PrebuiltFallback(
            f"could not read {MANIFEST_ASSET_NAME} from {repo}@{release_tag}: {exc}"
        ) from exc
    asset_names = {MANIFEST_ASSET_NAME, SHA256_ASSET_NAME}
    asset_names.update(
        str(artifact["asset"])
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("asset")
    )
    asset_urls = {name: release_asset_download_url(repo, release_tag, name) for name in asset_names}
    return ReleaseBundle(
        repo = repo,
        release_tag = release_tag,
        manifest = manifest,
        asset_urls = asset_urls,
    )


def _plan_bundle(
    host: HostInfo,
    bundle: ReleaseBundle,
    checksums: dict[str, str],
    *,
    published_repo: str,
    requested_backend: str,
    verify_checksums: bool,
) -> WhisperReleasePlan:
    artifact, resolved_backend, used_fallback = select_artifact_with_cpu_fallback(
        bundle.manifest, host, requested_backend
    )
    selection = (
        plan_selection(
            host,
            bundle,
            published_repo = published_repo,
            backend = requested_backend,
            checksums = checksums,
        )
        if verify_checksums
        else None
    )
    return WhisperReleasePlan(
        bundle = bundle,
        selection = selection,
        artifact = artifact,
        resolved_backend = resolved_backend,
        used_fallback = used_fallback,
    )


def _release_plan_for_host(
    host: HostInfo,
    *,
    published_repo: str,
    published_release_tag: str | None,
    whisper_tag: str,
    requested_backend: str,
    verify_checksums: bool = True,
) -> WhisperReleasePlan:
    """Select the requested or newest host-compatible published release.

    Explicit published release pins never walk. An upstream ``whisper_tag``
    searches published manifests for that exact upstream version. An unpinned
    macOS install may walk older published releases only when the newest
    manifest has no host-compatible artifact. Checksum and archive failures are
    outside this search and remain hard failures.
    """
    requested_specific_tag = whisper_tag.strip().lower() not in ("", "latest")
    first_bundle: ReleaseBundle | None = None
    first_error: PrebuiltFallback | None = None
    # An upstream version pin searches manifests directly. It must not depend
    # on the unrelated newest release or its checksum index being healthy.
    if not requested_specific_tag or published_release_tag:
        first_bundle, first_checksums = fetch_release_for_install(
            published_repo, published_release_tag = published_release_tag
        )
        if not _bundle_matches_whisper_tag(first_bundle, whisper_tag):
            first_error = PrebuiltFallback(
                f"{published_repo}@{first_bundle.release_tag} targets whisper.cpp "
                f"{first_bundle.manifest.get('upstream_tag')}, not requested {whisper_tag}"
            )
        else:
            try:
                select_artifact_with_cpu_fallback(first_bundle.manifest, host, requested_backend)
            except PrebuiltFallback as exc:
                first_error = exc
            else:
                # Integrity validation happens only after compatibility succeeds.
                # Its failures are never eligible for release walkback.
                return _plan_bundle(
                    host,
                    first_bundle,
                    first_checksums,
                    published_repo = published_repo,
                    requested_backend = requested_backend,
                    verify_checksums = verify_checksums,
                )

    if published_release_tag:
        assert first_error is not None
        raise first_error

    if not requested_specific_tag and not host.is_macos:
        assert first_error is not None
        raise first_error

    for release_tag in _published_release_tags(published_repo):
        if first_bundle is not None and release_tag == first_bundle.release_tag:
            continue
        try:
            bundle = _fetch_release_candidate(published_repo, release_tag)
        except PrebuiltFallback:
            continue
        if not _bundle_matches_whisper_tag(bundle, whisper_tag):
            continue
        try:
            # Establish host compatibility before fetching or trusting this
            # candidate's checksum index. Once selected, integrity failures must
            # stop the install rather than silently downgrade again.
            select_artifact_with_cpu_fallback(bundle.manifest, host, requested_backend)
        except PrebuiltFallback:
            continue
        checksums = fetch_release_checksums(bundle) if verify_checksums else {}
        plan = _plan_bundle(
            host,
            bundle,
            checksums,
            published_repo = published_repo,
            requested_backend = requested_backend,
            verify_checksums = verify_checksums,
        )
        log(
            f"selected compatible published release {bundle.release_tag} "
            f"(upstream {bundle.manifest.get('upstream_tag')})"
        )
        return plan

    if requested_specific_tag:
        raise PrebuiltFallback(
            f"no published {COMPONENT} release for upstream tag {whisper_tag} supports this host"
        )
    assert first_error is not None
    raise first_error


def _install_from_bundle(
    install_dir: Path, host: HostInfo, bundle: ReleaseBundle, selection: InstallSelection
) -> None:
    core.install_from_bundle(_OPS, install_dir, host, bundle, selection)


def plan_selection(
    host: HostInfo,
    bundle: ReleaseBundle,
    *,
    published_repo: str,
    backend: str,
    checksums: dict[str, str],
) -> InstallSelection:
    return core.plan_selection(
        _OPS,
        host,
        bundle,
        published_repo = published_repo,
        backend = backend,
        checksums = checksums,
    )


def install_prebuilt(
    install_dir: Path,
    *,
    whisper_tag: str = "latest",
    published_repo: str = DEFAULT_PUBLISHED_REPO,
    published_release_tag: str | None = None,
    backend: str | None = "auto",
    has_rocm: bool = False,
    rocm_gfx: str | None = None,
    cpu_fallback: bool = False,
    force: bool = False,
) -> int:
    host = apply_host_overrides(
        detect_host(), has_rocm = has_rocm, rocm_gfx = rocm_gfx, force_cpu = cpu_fallback
    )
    requested_backend = resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    os_token, arch_token = host_platform_tokens(host)
    log(
        f"target {COMPONENT} from {published_repo} "
        f"({os_token}-{arch_token}, backend {requested_backend})"
    )
    plan = _release_plan_for_host(
        host,
        published_repo = published_repo,
        published_release_tag = published_release_tag,
        whisper_tag = whisper_tag,
        requested_backend = requested_backend,
    )
    if plan.selection is None:  # pragma: no cover - install plans always verify
        raise PrebuiltFallback("install plan did not validate its checksum entry")
    return core.install_selected_prebuilt(
        _OPS,
        install_dir,
        host = host,
        bundle = plan.bundle,
        selection = plan.selection,
        force = force,
    )


def resolver_payload_extra(artifact: dict[str, Any]) -> dict[str, Any]:
    """Additive --resolve-prebuilt field: whether the selected asset installs
    slim (paired with the llama ggml runtime) or fat (self-contained)."""
    return {"install_kind": "slim" if artifact.get("install_kind") == "slim" else "fat"}


def resolve_prebuilt(
    host: HostInfo,
    *,
    published_repo: str,
    published_release_tag: str | None,
    whisper_tag: str = "latest",
    backend: str | None,
    cpu_fallback: bool,
) -> dict[str, Any]:
    requested_backend = resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    try:
        plan = _release_plan_for_host(
            host,
            published_repo = published_repo,
            published_release_tag = published_release_tag,
            whisper_tag = whisper_tag,
            requested_backend = requested_backend,
            verify_checksums = False,
        )
    except PrebuiltFallback:
        return {"prebuilt_available": False, "repo": published_repo}
    os_token, arch_token = host_platform_tokens(host)
    payload = {
        "prebuilt_available": True,
        "repo": published_repo,
        "release_tag": plan.bundle.release_tag,
        "upstream_tag": plan.bundle.manifest.get("upstream_tag"),
        "backend": plan.resolved_backend,
        "requested_backend": requested_backend,
        "cpu_fallback": plan.used_fallback,
        "asset": str(plan.artifact.get("asset")),
        "os": os_token,
        "arch": arch_token,
        "runtime_line": plan.artifact.get("runtime_line"),
    }
    payload.update(resolver_payload_extra(plan.artifact))
    return payload


# The whisper component descriptor: the declarative form of everything above,
# for descriptor-driven consumers (shared tests, future tooling). The shipped CLI
# runs through this module's wrappers so monkeypatch seams stay intact.
DESCRIPTOR = core.ComponentDescriptor(
    component = COMPONENT,
    log_prefix = "whisper-prebuilt",
    published_repo = DEFAULT_PUBLISHED_REPO,
    manifest_asset_name = MANIFEST_ASSET_NAME,
    sha256_asset_name = SHA256_ASSET_NAME,
    metadata_filename = METADATA_FILENAME,
    user_agent = USER_AGENT,
    supported_backends = SUPPORTED_BACKENDS,
    schema_version = SCHEMA_VERSION,
    fallback_backend = FALLBACK_BACKEND,
    detect_host = detect_host,
    host_platform_tokens = host_platform_tokens,
    server_binary_name = server_binary_name,
    runtime_bin_dir = runtime_bin_dir,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "Install a prebuilt whisper.cpp (whisper-server) for Unsloth Studio"
    )
    parser.add_argument(
        "--install-dir",
        default = None,
        help = (
            "managed whisper.cpp directory, e.g. <UNSLOTH_HOME>/whisper.cpp. Required for an "
            "install; omit it only with --resolve-prebuilt (a read-only probe)."
        ),
    )
    parser.add_argument(
        "--whisper-tag",
        default = os.environ.get("UNSLOTH_WHISPER_TAG", "latest"),
        help = "upstream whisper.cpp tag hint (default 'latest' or $UNSLOTH_WHISPER_TAG)",
    )
    parser.add_argument(
        "--published-repo",
        default = DEFAULT_PUBLISHED_REPO,
        help = f"GitHub repo publishing the prebuilt releases (default {DEFAULT_PUBLISHED_REPO})",
    )
    parser.add_argument(
        "--published-release-tag",
        default = os.environ.get("UNSLOTH_WHISPER_RELEASE_TAG") or None,
        help = "explicit release tag to install (default: the newest published release)",
    )
    parser.add_argument(
        "--backend",
        default = os.environ.get("UNSLOTH_WHISPER_BACKEND", "auto"),
        choices = ("auto", *SUPPORTED_BACKENDS),
        help = "accelerator backend; 'auto' detects from hardware",
    )
    parser.add_argument("--has-rocm", action = "store_true", help = "treat this host as ROCm-capable")
    parser.add_argument("--rocm-gfx", default = None, help = "ROCm gfx target override, e.g. gfx1100")
    parser.add_argument(
        "--cpu-fallback", action = "store_true", help = "force the CPU asset regardless of hardware"
    )
    parser.add_argument(
        "--resolve-prebuilt",
        nargs = "?",
        const = "latest",
        default = None,
        help = "report whether a prebuilt exists for this host without downloading",
    )
    parser.add_argument(
        "--output-format",
        choices = ("plain", "json"),
        default = "plain",
        help = "resolver output format (default plain)",
    )
    parser.add_argument(
        "--force", action = "store_true", help = "reinstall even if the install already matches"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global _LOG_TO_STDOUT
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.resolve_prebuilt is not None:
        # Read-only probe: stdout is only the JSON/asset line (setup.sh and
        # whisper_cpp_update.py parse it), so force diagnostics to stderr and map
        # any failure to "not available" rather than a traceback.
        _LOG_TO_STDOUT = False
        try:
            host = apply_host_overrides(
                detect_host(),
                has_rocm = args.has_rocm,
                rocm_gfx = args.rocm_gfx,
                force_cpu = args.cpu_fallback,
            )
            payload = resolve_prebuilt(
                host,
                published_repo = args.published_repo,
                published_release_tag = args.published_release_tag,
                whisper_tag = args.resolve_prebuilt,
                backend = args.backend,
                cpu_fallback = args.cpu_fallback,
            )
        except PrebuiltFallback:
            payload = {"prebuilt_available": False, "repo": args.published_repo}
        except Exception as exc:  # noqa: BLE001 - probe must never crash the caller
            log(f"resolve failed: {exc}")
            payload = {"prebuilt_available": False, "repo": args.published_repo}
        emit_resolver_output(payload, output_format = args.output_format)
        return EXIT_SUCCESS

    # Install path: progress logs go to stdout so setup surfaces them.
    _LOG_TO_STDOUT = True

    if not args.install_dir:
        parser.error("--install-dir is required unless --resolve-prebuilt is used")
    install_dir = Path(args.install_dir).expanduser().resolve()
    try:
        return install_prebuilt(
            install_dir,
            whisper_tag = args.whisper_tag,
            published_repo = args.published_repo,
            published_release_tag = args.published_release_tag,
            backend = args.backend,
            has_rocm = args.has_rocm,
            rocm_gfx = args.rocm_gfx,
            cpu_fallback = args.cpu_fallback,
            force = args.force,
        )
    except BusyInstallConflict as exc:
        log(str(exc))
        return EXIT_BUSY
    except ReleaseCompatibilityError as exc:
        log(f"incompatible release: {exc}")
        return EXIT_INCOMPATIBLE
    except PrebuiltFallback as exc:
        log(f"prebuilt install failed: {exc}")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        log(f"unexpected error: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
