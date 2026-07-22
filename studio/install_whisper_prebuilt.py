#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform whisper.cpp (whisper-server) prebuilt installer for Unsloth Studio.

Downloads a per-platform whisper.cpp bundle published by the Unsloth fork
(``unslothai/whisper.cpp``) into an isolated ``<UNSLOTH_HOME>/whisper.cpp`` and
never touches a system whisper.cpp. The canonical install target matches the
sidecar/build-script contract in ``stt_ggml_sidecar.py``:

    <install-dir>/build/bin/whisper-server              (Unix)
    <install-dir>/build/bin/Release/whisper-server.exe  (Windows)

Prebuilt bundles are dynamically linked: ``whisper-server`` needs ``libwhisper``,
``libggml*`` and any GPU backend libraries. The binaries carry ``RUNPATH=$ORIGIN``
so the installer co-locates every shared library from the archive into the same
``build/bin`` directory as the server, and marks the server executable on Unix.

Archives are verified against the release's own ``whisper-prebuilt-sha256.json``
checksum index, fetched from the same GitHub release (the same model as
``install_llama_prebuilt.py``). An asset absent from that index, or a release that
does not publish it, fails closed to a source build. This is a same-origin
checksum -- it proves integrity, not authenticity -- so pair the release with
GitHub artifact attestations for provenance.

Release resolution prefers the download host (``github.com/<repo>/releases/...``),
so the manifest + checksum index are fetched with **zero ``api.github.com`` calls**
(that API is rate-limited to 60 req/hour unauthenticated); the GitHub API is only
used as a fallback on a 404 / malformed asset.

The whole component-agnostic flow -- verified downloads with retries and
token-safe redirects, safe archive extraction, the install lock, release
resolution, checksum-index handling, coverage-aware CUDA/ROCm selection, the
install driver and the resolve probe -- lives in ``prebuilt_core.py`` (same
directory) and is shared with ``install_llama_prebuilt.py``. This module keeps
only the whisper specifics: the host mapping (whisper asset tokens over llama's
hardware probes), the install-tree layout, the CPU fallback policy, the marker
filename, and the CLI. Every retained public name is a thin wrapper so tests
can monkeypatch it on this module and the core observes the patch.

Mirrors ``install_node_prebuilt.py`` / ``install_llama_prebuilt.py`` so the setup
scripts drive it the same way. Exit codes: 0 success (or already current), 1 error,
2 source fallback, 3 busy. A re-run that already matches logs "already matches"
and returns 0 without downloading (the scripts grep it).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Put studio/ on sys.path so install_llama_prebuilt / prebuilt_core resolve
# whether this file is run as a script (from any cwd) or imported by the tests.
_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import install_llama_prebuilt as llama  # noqa: E402
import prebuilt_core as core  # noqa: E402

# Shared machinery re-exported as module globals. The wrappers below resolve
# these by bare name at call time so tests can monkeypatch them on this module.
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
detect_torch_cuda_runtime_preference = llama.detect_torch_cuda_runtime_preference
linux_runtime_dirs_for_required_libraries = llama.linux_runtime_dirs_for_required_libraries
detected_windows_runtime_lines = llama.detected_windows_runtime_lines

# Late-binding seam handed to prebuilt_core: name lookups hit this module's
# globals first (so monkeypatches apply), then the core defaults.
_OPS = core.ModuleOps(globals())


# Resolver mode keeps stdout to the JSON payload only (setup.sh parses it);
# main() flips this on the install path so setup surfaces progress.
_LOG_TO_STDOUT = False


def log(message: str) -> None:
    print(f"[whisper-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr)


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_FALLBACK = 2
EXIT_BUSY = 3

COMPONENT = "whisper.cpp"
SCHEMA_VERSION = 1
USER_AGENT = "unsloth-studio-whisper-prebuilt"

DEFAULT_PUBLISHED_REPO = "unslothai/whisper.cpp"
# Release assets published by the fork's prebuilt CI.
MANIFEST_ASSET_NAME = "whisper-prebuilt-manifest.json"
SHA256_ASSET_NAME = "whisper-prebuilt-sha256.json"

METADATA_FILENAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# Backends the installer knows how to select. Asset filenames carry a finer
# accelerator token (e.g. cuda12); selection matches the manifest artifact's
# coarse `backend` field so a new CUDA toolkit needs no code change here.
SUPPORTED_BACKENDS = ("cpu", "cuda", "metal", "vulkan", "rocm")

# Fallback policy the core applies on a GPU-selection miss: whisper releases
# always publish a CPU bundle, so retry with it instead of a source build.
FALLBACK_BACKEND = "cpu"

# Backends whose slim (ggml-less) whisper bundle can ride the ggml runtime of
# the installed llama.cpp prebuilt. cpu and metal always install fat.
SLIM_ELIGIBLE_BACKENDS = ("cuda", "rocm", "vulkan")

# ggml backend module the llama bin dir must provide per accelerator, as
# (linux glob, windows glob) following llama's bundle layout / health globs.
SLIM_BACKEND_MODULE_GLOBS = {
    "cuda": ("libggml-cuda.so*", "ggml-cuda.dll"),
    "rocm": ("libggml-hip.so*", "*hip*.dll"),
    "vulkan": ("libggml-vulkan.so*", "ggml-vulkan.dll"),
}

# Everything the slim wiring must mirror from the llama bin dir: the core ggml
# sonames plus every dlopen'd backend module (CPU variants included).
SLIM_GGML_LIBRARY_GLOBS = ("libggml*", "ggml*.dll")

INSTALL_STAGING_ROOT_NAME = ".staging"

# Master switch for the staged runtime smoke test (start whisper-server + a tiny
# transcription) before the atomic activate. Disabled by default so a bundle whose
# GPU forward pass JIT-compiles kernels does not stall every install; the check and
# its CPU-asset retry are kept intact -- set this True to re-enable them.
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
    # NVIDIA GPU compute capabilities (SM strings) and the driver's CUDA version,
    # used to pick an SM-appropriate CUDA bundle. Empty / None on non-NVIDIA
    # hosts. `torch_runtime_line` ('cuda12'/'cuda13') is torch's build
    # preference, a tie-break on non-Blackwell hosts.
    compute_caps: tuple[str, ...] = ()
    driver_cuda_version: tuple[int, int] | None = None
    torch_runtime_line: str | None = None
    # (major, minor) from platform.mac_ver(); None off macOS or if unparseable.
    # Used to enforce a macOS artifact's min_os so we never pick a bundle that
    # cannot load on this OS version.
    macos_version: tuple[int, int] | None = None


def host_from_llama(base: Any) -> HostInfo:
    """Map install_llama_prebuilt's detected host (NVIDIA caps + driver honoring
    CUDA_VISIBLE_DEVICES, ROCm gfx incl. WSL, macOS version) onto the whisper
    asset tokens. Raises PrebuiltFallback on an unsupported OS/arch."""
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

    # Self-guarded: returns None unless the host has usable NVIDIA + torch CUDA.
    torch_runtime_line = detect_torch_cuda_runtime_preference(base).runtime_line

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
        compute_caps = tuple(base.compute_caps),
        driver_cuda_version = base.driver_cuda_version,
        torch_runtime_line = torch_runtime_line,
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
            compute_caps = (),
            driver_cuda_version = None,
            torch_runtime_line = None,
        )
    updates: dict[str, Any] = {}
    if has_rocm or rocm_gfx:
        # --rocm-gfx implies --has-rocm (llama parity): otherwise the host stays on
        # its CUDA/CPU path and never picks the ROCm bundle. Drop CUDA detection too.
        updates["has_rocm"] = True
        updates["has_usable_nvidia"] = False
        updates["compute_caps"] = ()
        updates["driver_cuda_version"] = None
        updates["torch_runtime_line"] = None
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
    which can be finer-grained than the coarse backend. The authoritative asset
    name for an install comes from the release manifest; this constructor is used
    for default names and diagnostics.
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


# ── On-disk CUDA runtime detection (exact SONAMEs; llama's dir scan) ──
def detected_linux_runtime_lines() -> list[str]:
    """`cuda<major>` lines whose libcudart + libcublas SONAMEs are on disk, newest
    first. Exact-name matching: the loader resolves the SONAME (libcudart.so.13),
    so a versioned-only file without that symlink is not loadable and a `{lib}*`
    glob would overreport."""
    return core.detected_linux_runtime_lines_exact(_OPS)


def detected_linux_runtime_lines_exact() -> list[str]:
    return detected_linux_runtime_lines()


def detected_cuda_runtime_lines(*, is_windows: bool) -> list[str]:
    """Platform-appropriate on-disk CUDA runtime-line detection (newest first)."""
    if is_windows:
        return detected_windows_runtime_lines()[0]
    return detected_linux_runtime_lines()


# ── Coverage-aware selection (core primitives over the whisper manifest) ──
def _select_cuda_artifact(
    candidates: list[dict[str, Any]], host: HostInfo, log_lines: list[str]
) -> dict[str, Any] | None:
    return core.select_cuda_artifact(_OPS, candidates, host, log_lines)


def select_cuda_artifact(
    candidates: list[dict[str, Any]], host: HostInfo, log_lines: list[str]
) -> dict[str, Any] | None:
    return _select_cuda_artifact(candidates, host, log_lines)


def _select_rocm_artifact(
    candidates: list[dict[str, Any]], host_gfx: str | None, log_lines: list[str]
) -> dict[str, Any] | None:
    return core.select_rocm_artifact(_OPS, candidates, host_gfx, log_lines)


def select_rocm_artifact(
    candidates: list[dict[str, Any]], host_gfx: str | None, log_lines: list[str]
) -> dict[str, Any] | None:
    return _select_rocm_artifact(candidates, host_gfx, log_lines)


def slim_pairing_for_artifact(
    artifact: dict[str, Any], host: HostInfo, backend: str
) -> tuple[Path, str] | None:
    """(llama bin dir, llama release tag) when the installed llama runtime can
    back this slim artifact, else None. Each failed gate logs why and the caller
    falls through to the fat selection chain unchanged."""
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
    linux_glob, windows_glob = SLIM_BACKEND_MODULE_GLOBS[backend]
    module_glob = windows_glob if host.is_windows else linux_glob
    if not any(path.is_file() for path in llama_bin_dir.glob(module_glob)):
        log(f"slim_selection: {asset} skipped: llama runtime has no {module_glob} module")
        return None
    log(f"slim_selection: selected {asset} paired with llama {llama_tag} at {llama_bin_dir}")
    return llama_bin_dir, llama_tag


def select_slim_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """The paired slim artifact for a GPU backend, or None. macOS stays fat; slim
    assets carry backend "slim" so the fat os/arch/backend matching never sees
    them and the fat chain is untouched by construction."""
    if backend not in SLIM_ELIGIBLE_BACKENDS or host.is_macos:
        return None
    os_token, arch_token = host_platform_tokens(host)
    for artifact in manifest.get("artifacts", []):
        if (
            artifact.get("os") == os_token
            and artifact.get("arch") == arch_token
            and artifact.get("backend") == "slim"
            and artifact.get("install_kind") == "slim"
            and slim_pairing_for_artifact(artifact, host, backend) is not None
        ):
            return artifact
    return None


def select_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """Slim-first: a GPU backend prefers the paired slim artifact when the
    installed llama runtime satisfies its requirements; any miss falls through
    to the shared fat selection chain unchanged."""
    slim = select_slim_artifact(manifest, host, backend)
    if slim is not None:
        return slim
    return core.select_artifact(_OPS, manifest, host, backend)


def select_artifact_with_cpu_fallback(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> tuple[dict[str, Any], str, bool]:
    return core.select_artifact_with_fallback(_OPS, manifest, host, backend)


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
    """The shared traversal/symlink-guarded extraction, then restore tar exec bits
    (the extractor writes plain files; whisper-server must stay executable)."""
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
    """Lay out staged_root as a full install: build/bin/<server + all libs>.

    Everything sitting beside the server in the archive (shared libraries, backend
    kernel subdirs, license/build-info) is co-located into the canonical bin dir so
    the server's RUNPATH=$ORIGIN resolves its libs.
    """
    bin_dir = runtime_bin_dir(staged_root, host)
    bin_dir.mkdir(parents = True, exist_ok = True)
    for entry in sorted(bundle_root.iterdir()):
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
def link_ggml_runtime(llama_bin_dir: Path, whisper_bin_dir: Path) -> int:
    """Hardlink every ggml library from the llama runtime into the whisper bin
    dir; returns how many were wired.

    Hardlinks (not symlinks) on purpose: when the llama updater atomically swaps
    its install dir, the old inodes stay alive under these links, so a
    not-yet-updated whisper keeps running the exact ggml build it was installed
    against -- the version-skew window closes by construction. Falls back to a
    copy where linking is unsupported or crosses devices. Re-run on every
    install/update so the links always track the current pairing.
    """
    sources = sorted(
        {
            path
            for pattern in SLIM_GGML_LIBRARY_GLOBS
            for path in llama_bin_dir.glob(pattern)
            if path.is_file()
        }
    )
    if not sources:
        raise PrebuiltFallback(
            f"no ggml libraries found in {llama_bin_dir} to pair the slim whisper install"
        )
    whisper_bin_dir.mkdir(parents = True, exist_ok = True)
    for source in sources:
        dest = whisper_bin_dir / source.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            # Follows a libggml.so.0 -> libggml.so.0.x symlink to its inode, so
            # every created name is a real hardlink that survives a dir swap.
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
    return len(sources)


def prepare_runtime_payload(staged_root: Path, host: HostInfo, selection: Any) -> None:
    """Slim installs wire the paired llama ggml runtime into the staged bin dir
    before validation, so the activated tree is self-contained. Fat installs
    need nothing (the archive already carries its ggml)."""
    if getattr(selection, "install_kind", None) != "slim" or not selection.linked_from:
        return
    linked = link_ggml_runtime(Path(selection.linked_from), runtime_bin_dir(staged_root, host))
    log(f"slim install: hardlinked {linked} ggml libraries from {selection.linked_from}")


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
    # A slim selection carries its pairing so the install wiring and the marker
    # know exactly which llama runtime provides the ggml libraries.
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
    return core.existing_install_matches(_OPS, install_dir, host, selection)


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
    return core.install_prebuilt(
        _OPS,
        install_dir,
        published_repo = published_repo,
        published_release_tag = published_release_tag,
        backend = backend,
        cpu_fallback = cpu_fallback,
        force = force,
        host = host,
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
    backend: str | None,
    cpu_fallback: bool,
) -> dict[str, Any]:
    return core.resolve_prebuilt(
        _OPS,
        host,
        published_repo = published_repo,
        published_release_tag = published_release_tag,
        backend = backend,
        cpu_fallback = cpu_fallback,
    )


# The whisper component descriptor: the declarative form of everything above,
# for descriptor-driven consumers (shared tests, future tooling). The shipped
# CLI runs through this module's wrappers so monkeypatch seams stay intact.
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
        # any failure to "not available" instead of a traceback.
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
    except PrebuiltFallback as exc:
        log(f"prebuilt unavailable: {exc}")
        return EXIT_FALLBACK
    except Exception as exc:  # noqa: BLE001
        log(f"unexpected error: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
