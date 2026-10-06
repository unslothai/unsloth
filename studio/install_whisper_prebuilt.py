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
(the scripts grep it). A release lookup that could not answer at all also returns 0 when
an intact, previously validated install is on disk, logging "update unavailable, existing
prebuilt kept; keeping the existing complete install" -- llama.cpp's wording for its own
identical outcome, and the substring the setup scripts grep to report it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

# Put studio/ on sys.path so install_llama_prebuilt / prebuilt_core resolve whether run as a script
# from any cwd or imported by the tests.
_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import install_llama_prebuilt as llama  # noqa: E402
import prebuilt_core as core  # noqa: E402

# Shared machinery re-exported as module globals; the wrappers below resolve these by bare name at
# call time so tests can monkeypatch them here.
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
installed_llama_ggml_tree = llama.installed_llama_ggml_tree

# Late-binding seam for prebuilt_core: name lookups hit this module's globals first, so
# monkeypatches apply, then the core defaults.
_OPS = core.ModuleOps(globals())


# Resolver mode keeps stdout to the JSON payload only (setup.sh parses it); main() flips this on
# for the install path so setup surfaces progress.
_LOG_TO_STDOUT = False


# Raised by prebuilt_core when a fetched release cannot be trusted; re-exported so this
# module's callers can name it without importing core.
ReleaseIntegrityError = core.ReleaseIntegrityError


class ReleaseCompatibilityError(PrebuiltFallback):
    """A valid slim release cannot pair with this host's installed runtime."""


def log(message: str) -> None:
    print(f"[whisper-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr)


def log_lines(lines: Iterable[str]) -> None:
    for line in lines:
        log(line)


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

# Backends the installer can select: accelerator identities for the slim pairing (which llama ggml
# module must exist) and the marker. Only "cpu" can match a fat artifact, the pinned escape hatch.
SUPPORTED_BACKENDS = ("cpu", "cuda", "metal", "vulkan", "rocm")

# Fallback on a GPU-selection miss: retry with cpu so the slim bundle can pair via the llama cpu
# modules, or the published fat CPU bundle installs on a pinned pre-slim release.
FALLBACK_BACKEND = "cpu"

# Backends whose slim (ggml-less) whisper bundle can ride the installed llama.cpp prebuilt's ggml
# runtime. All are eligible: the next whisper release ships only slim bundles.
SLIM_ELIGIBLE_BACKENDS = ("cpu", "cuda", "metal", "rocm", "vulkan")

# ggml backend module the llama bin dir must provide per accelerator.
# llama's x64 cpu bundles ship per-arch libggml-cpu-<variant> modules and macOS a single libggml-cpu, so the cpu globs
# cover both; a backend with no glob for the host os (metal off macOS) is never slim there.
SLIM_BACKEND_MODULE_GLOBS = {
    "cpu": {
        "linux": "libggml-cpu*.so*",
        "windows": "ggml-cpu*.dll",
        "macos": "libggml-cpu*.dylib",
    },
    "cuda": {"linux": "libggml-cuda.so*", "windows": "ggml-cuda.dll"},
    "metal": {"macos": "libggml-metal*.dylib"},
    # Windows rocm must name the ggml module: the bundles also ship amdhip64
    "rocm": {"linux": "libggml-hip.so*", "windows": "ggml-hip*.dll"},
    "vulkan": {"linux": "libggml-vulkan.so*", "windows": "ggml-vulkan.dll"},
}

# Everything the slim wiring mirrors from the llama bin dir: core ggml sonames plus every dlopen'd
# backend module. libomp* rides along because llama's clang-built slices link ggml against the LLVM
# OpenMP runtime and ship it in the bundle, so the loader can never find it on the host; Linux x64
# and macOS ship none, so the globs match nothing there.
SLIM_GGML_LIBRARY_GLOBS = (
    "libggml*",
    "ggml*.dll",
    "libomp*.dll",
    "libomp*.so*",
    "libomp*.dylib",
)
SLIM_ROCM_LIBRARY_GLOBS = (
    "libamd*.so*",
    "libhip*.so*",
    "libhsa*.so*",
    "libroc*.so*",
)
# Kernel catalogs a linux ROCm llama bundle can ship, mirrored into the whisper bin dir so the
# packaged libraries find their Tensile kernels beside them.
SLIM_ROCM_RUNTIME_DIRS = ("hipblaslt", "rocblas")
# Of those, the ones a paired runtime must actually carry.
SLIM_ROCM_REQUIRED_RUNTIME_DIRS = ("rocblas",)
# 3: libomp*.so*/dylib joined the wiring.
SLIM_RUNTIME_WIRING_VERSION = 3

INSTALL_STAGING_ROOT_NAME = ".staging"

# Master switch for the staged runtime smoke test, off by default so a bundle whose GPU forward
# pass JIT-compiles kernels does not stall every install. The check and its CPU-asset retry are
# intact: set True to re-enable.
_RUN_STAGED_PREBUILT_VALIDATION = False


# ── Host detection (probes shared with install_llama_prebuilt) ──
@dataclass(frozen = True)
class HostInfo:
    system: str
    machine: str
    whisper_os: str
    whisper_arch: str  # asset token: x64 | arm64
    archive_ext: str
    is_windows: bool
    is_macos: bool
    is_apple_silicon: bool
    has_usable_nvidia: bool = False
    has_rocm: bool = False
    rocm_gfx: str | None = None
    # (major, minor) from platform.mac_ver(); None off macOS or if unparseable.
    # Enforces a macOS artifact's min_os so a bundle that cannot load on this OS version is never picked.
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
def _llama_ggml_commit(tag: str) -> str | None:
    """The "-mix-" suffix of a fork tag "b<upstream_build>-mix-<suffix>".

    Despite the name this is NOT a ggml commit: it hashes the pinned PR set and
    stays constant while the base tag, and ggml with it, moves (one value covered
    b9909..b10001, whose ggml trees differ). Fallback for releases predating
    ggml_tree."""
    marker = "-mix-"
    idx = tag.rfind(marker)
    end = idx + len(marker)
    return tag[end:] if idx >= 0 and end < len(tag) else None


_PUBLISHED_GGML_TREE_CACHE: dict[tuple[str, str], str | None] = {}


def published_llama_ggml_tree(tag: Any, repo: Any = None) -> str | None:
    """ggml tree id recorded in the published llama release for ``tag``.

    An install made before ggml_tree was recorded has no tree locally, and
    nothing can backfill one without reinstalling llama. The release manifest
    carries it, so read that instead of falling back to the "-mix-" suffix,
    which does not track ggml. Failures return None: the caller then uses the
    old fallback rather than refusing an otherwise valid install, so this
    fetches once with no retries -- the retry policy exists for downloads we
    need, and an unreachable host would otherwise stall every uncached tag."""
    if not isinstance(tag, str) or not tag:
        return None
    if not (isinstance(repo, str) and repo):
        repo = llama.DEFAULT_PUBLISHED_REPO
    key = (repo, tag)
    if key in _PUBLISHED_GGML_TREE_CACHE:
        return _PUBLISHED_GGML_TREE_CACHE[key]
    tree = None
    try:
        payload = _download_host_json_once(
            release_asset_download_url(repo, tag, llama.DEFAULT_PUBLISHED_MANIFEST_ASSET)
        )
        if isinstance(payload, dict):
            candidate = payload.get("ggml_tree")
            if isinstance(candidate, str) and candidate:
                tree = candidate
    except Exception as exc:
        log(f"slim_selection: could not read ggml_tree for llama {tag}: {exc}")
    _PUBLISHED_GGML_TREE_CACHE[key] = tree
    return tree


def installed_llama_tree_repo() -> str | None:
    """Repo whose published ggml tree describes the installed llama binaries.

    None when there is nothing to read, or when the binaries came from a repo
    other than the one that published the release: recorded_ggml_tree() leaves
    the marker's tree unset in that case precisely because the fork tree does
    not describe an upstream-built libggml, so it must not be inferred either."""
    metadata = llama.load_prebuilt_metadata(llama.default_managed_llama_dir())
    if metadata is None:
        return None
    published = metadata.get("published_repo")
    if not isinstance(published, str) or not published:
        return None
    binary = metadata.get("binary_repo")
    if isinstance(binary, str) and binary and binary != published:
        return None
    return published


def llama_runtime_pairs(
    installed_tag: str,
    required_tag: Any,
    *,
    installed_ggml_tree: Any = None,
    required_ggml_tree: Any = None,
    installed_repo: Any = None,
) -> bool:
    """Whether an installed llama tag can back a slim bundle needing required_tag.
    An exact tag always pairs; otherwise the ggml tree ids decide, since they
    change exactly when ggml does. The "-mix-" suffix does not track ggml at all
    (see _llama_ggml_commit) and is only used when either tree is missing.
    requires_ggml_sonames stays the per-file ABI gate.

    installed_repo is the repo that published the installed runtime (from
    installed_llama_tree_repo()); without it the installed tag's tree is not
    inferred, since only the caller knows the binaries came from that release."""
    if not isinstance(required_tag, str):
        return False
    if installed_tag == required_tag:
        return True
    # An install predating ggml_tree has no tree locally, so read it from that
    # tag's published release rather than stranding the install on a suffix
    # comparison that ignores ggml.
    if not (isinstance(installed_ggml_tree, str) and installed_ggml_tree):
        if isinstance(installed_repo, str) and installed_repo:
            installed_ggml_tree = published_llama_ggml_tree(installed_tag, installed_repo)
    # Only probe the required release once the installed side actually resolved
    if installed_ggml_tree and not (isinstance(required_ggml_tree, str) and required_ggml_tree):
        required_ggml_tree = published_llama_ggml_tree(required_tag)
    if installed_ggml_tree and required_ggml_tree:
        return installed_ggml_tree == required_ggml_tree
    commit = _llama_ggml_commit(installed_tag)
    return commit is not None and commit == _llama_ggml_commit(required_tag)


def _ships_gpu_ggml_module(llama_bin_dir: Path, os_key: str) -> bool:
    """Whether a paired llama runtime carries a GPU ggml backend module.
    Only the cpu bundle links ggml against libomp, and bundle_profile cannot tell
    the two apart: it is absent on both the published rocm artifacts and every
    upstream-sourced install, so the files on disk decide."""
    for backend, per_os in SLIM_BACKEND_MODULE_GLOBS.items():
        glob = per_os.get(os_key)
        # is_file(): a versioned alias still counts, a directory or dangling link does not.
        if backend != "cpu" and glob and any(path.is_file() for path in llama_bin_dir.glob(glob)):
            return True
    return False


def _ships_windows_gpu_ggml_module(llama_bin_dir: Path) -> bool:
    """Windows spelling of _ships_gpu_ggml_module; kept for call-site clarity."""
    return _ships_gpu_ggml_module(llama_bin_dir, "windows")


def _is_linux_libomp_soname(name: str) -> bool:
    """LLVM's OpenMP runtime: bundled, so it can go missing. Not host-provided libgomp,
    and not libomptarget, which a prefix match would sweep up."""
    lowered = name.lower()
    return lowered == "libomp.so" or lowered.startswith("libomp.so.")


def _ggml_stack_imports_libomp(llama_bin_dir: Path, backend: str) -> bool | None:
    """Whether the ggml libraries whisper will load actually import LLVM's libomp.

    True on the first import found; False only once every relevant library was inspected
    and none imported it; None when any could not be inspected.

    The selected backend module is inspected alongside the core: it is dlopen'd after
    startup, so a dependency living only there is invisible to a --help smoke test.
    """
    patterns = ["libggml-base.so*", "libggml.so*"]
    module_glob = SLIM_BACKEND_MODULE_GLOBS.get(backend, {}).get("linux")
    if module_glob:
        patterns.append(module_glob)

    inspected_any = False
    inspected_all = True
    for pattern in patterns:
        for path in sorted(llama_bin_dir.glob(pattern)):
            if not path.is_file():
                continue
            needed = _elf_needed(path)
            if needed is None:
                inspected_all = False
                continue
            inspected_any = True
            if any(_is_linux_libomp_soname(dep) for dep in needed):
                return True
    return False if inspected_any and inspected_all else None


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
    if not llama_runtime_pairs(
        llama_tag,
        requires_tag,
        installed_ggml_tree = installed_llama_ggml_tree(),
        required_ggml_tree = artifact.get("requires_ggml_tree"),
        installed_repo = installed_llama_tree_repo(),
    ):
        log(
            f"slim_selection: {asset} skipped: installed llama tag {llama_tag!r} "
            f"does not pair with required {requires_tag!r}"
        )
        return None
    sonames = artifact.get("requires_ggml_sonames")
    if not isinstance(sonames, list) or not sonames:
        log(f"slim_selection: {asset} skipped: manifest lists no requires_ggml_sonames")
        return None
    required_sonames = [str(name) for name in sonames]
    if host.is_windows and _ships_windows_gpu_ggml_module(llama_bin_dir):
        # The shared Windows manifest lists libomp only because the cpu bundle's ggml links against it, so
        # requiring it for a GPU bundle only mis-rejects; link_ggml_runtime still wires it. Dropping the
        # aarch64 name too is safe only while llama publishes no Windows arm64 GPU bundle, whose
        # clang-built ggml really does need LLVM OpenMP: re-check this gate before adding one.
        required_sonames = [
            name
            for name in required_sonames
            if not (name.lower().startswith("libomp") and name.lower().endswith(".dll"))
        ]
    if (
        host.whisper_os == "linux"
        and host.whisper_arch == "arm64"
        and _ships_gpu_ggml_module(llama_bin_dir, "linux")
        and _ggml_stack_imports_libomp(llama_bin_dir, backend) is False
    ):
        # Same shape as the Windows case above: the arm64 manifest names libomp only because
        # the cpu bundle's ggml links it, so CUDA (GNU libgomp) was rejected for a file it
        # never needs. Evidence, not arch -- the arm64 Vulkan bundle really does import it.
        # `is False`, not `is not True`: dropping this on no evidence is unrecoverable, since
        # the sidecar sends the loader error to DEVNULL and serving never falls back.
        # arm64 only: linux-x64 ggml really imports its bundled libomp
        # (test_linux_slim_still_requires_manifest_libomp).
        required_sonames = [name for name in required_sonames if not _is_linux_libomp_soname(name)]
    missing = [name for name in required_sonames if not (llama_bin_dir / name).is_file()]
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
    installed_tree = installed_llama_ggml_tree()
    installed_repo = installed_llama_tree_repo()
    # Normalise the tree: llama_runtime_pairs treats a non-string as absent, but
    # an unhashable one (list/dict) would blow up the set first.
    required_pairs = {
        (
            artifact.get("requires_llama_tag"),
            artifact["requires_ggml_tree"]
            if isinstance(artifact.get("requires_ggml_tree"), str)
            else None,
        )
        for artifact in os_compatible
        if isinstance(artifact.get("requires_llama_tag"), str)
    }
    required_tags = {tag for tag, _tree in required_pairs}
    if required_tags and not any(
        llama_runtime_pairs(
            installed_tag,
            tag,
            installed_ggml_tree = installed_tree,
            required_ggml_tree = tree,
            installed_repo = installed_repo,
        )
        for tag, tree in required_pairs
    ):
        required_tag = sorted(required_tags)[0]
        return (
            f"slim bundle requires llama.cpp {required_tag}; installed llama.cpp is {installed_tag}"
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


def _download_host_json_once(url: str) -> Any:
    """One attempt, for a probe whose failure path is a cheap fallback.

    A refused or blackholed host is a retryable URLError, so the default policy
    would spend four attempts with backoff per tag before returning the answer
    the caller already has a fallback for.

    Only the ATTEMPT count is overridden. ``download_bytes`` falls back to
    ``auth_headers(url)`` when headers are absent, so passing a bare User-Agent
    silently dropped auth: a private published repo 404s and the caller drops to
    the "-mix-" suffix compare this exists to replace, and an anonymous
    huggingface.co fetch shares the per-IP limit that 429s CI fleets."""
    data = download_bytes(url, timeout = 30, attempts = 1, headers = auth_headers(url))
    return json.loads(data.decode("utf-8"))


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
            encoding = "utf-8",
            errors = "replace",
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
                [*command, str(path)],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
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
        required = name in SLIM_ROCM_REQUIRED_RUNTIME_DIRS
        files = (
            [path for path in source_root.rglob("*") if path.is_file()]
            if source_root.is_dir()
            else []
        )
        if not files:
            if not required:
                # hipBLASLt has no kernels for this target, and llama pairs without the catalog and runs, so
                # whisper must pair the same way.
                log(f"slim install: paired ROCm runtime ships no {name} kernel catalog; skipping")
                continue
            missing = "is missing its" if not source_root.is_dir() else "has an empty"
            raise PrebuiltFallback(f"paired ROCm runtime {missing} {name} kernel catalog")
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


# Every bin layout this installer produces, walked whatever the host says: the backfill runs from
# settle_kept_install, which prebuilt_core hands the install directory alone.
_RUNTIME_RECORD_BIN_DIRS = (("build", "bin"), ("build", "bin", "Release"))
_RUNTIME_RECORD_SERVER_NAMES = ("whisper-server", "whisper-server.exe")


def _stat_record(path: Path) -> "dict[str, Any] | None":
    """size + mtime_ns for one regular file, or None when it is neither."""
    try:
        if not path.is_file():
            return None
        info = path.stat()
    except OSError:
        return None
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _runtime_file_records(
    install_dir: Path,
    linked_libraries: "Iterable[str] | None" = None,
    linked_runtime_directories: "Iterable[str] | None" = None,
) -> "dict[str, dict[str, Any]]":
    """What the installed payload is made of, in a form a later run can re-check without
    running it.

    The same two tiers as the llama installer's runtime_file_records, because the two
    halves of a whisper install fail differently:

      * whisper-server gets size + sha256. Its bytes decide whether dictation works at
        all, and every reuse path keeps an install without ever starting it, so the
        digest is what stands in for "it launched once". One small binary is ~10 ms.
      * the ggml libraries a slim bundle hardlinks get size + mtime_ns, stat only.
        installed_tree_is_intact checks those for EXISTENCE by name, so a
        libggml-base.so truncated by a full disk or an interrupted extract passes it --
        and on a CUDA or ROCm pairing those libraries are most of the bytes. A size
        comparison catches that for the price of a stat; hashing hundreds of MB of
        kernels on every update would not be worth it.

      * the ROCm kernel catalogs a slim bundle links get size + mtime_ns, recursively.
        installed_tree_is_intact asks only that each of those directories hold ANY file
        (`rglob("*")`), so a rocblas/ that lost or truncated one TensileLibrary blob and
        kept the rest satisfied it, and the keep-existing path then reported an install
        whose catalog no longer loads. Stat only, for the same reason as the libraries:
        rocblas is hundreds of MB and this runs on every update.

    *linked_libraries* and *linked_runtime_directories* come from the slim selection being
    installed (or from the marker being backfilled); without them only the binary tier is
    recorded, which is what a fat bundle -- and a caller with no wiring in hand -- can
    honestly say.
    """
    records: dict[str, dict[str, Any]] = {}
    for parts in _RUNTIME_RECORD_BIN_DIRS:
        bin_dir = install_dir.joinpath(*parts)
        for name in linked_libraries or ():
            # Bare filenames only: a marker naming ../.. must not send the record outside the install.
            if not isinstance(name, str) or not name or Path(name).name != name:
                continue
            record = _stat_record(bin_dir / name)
            if record is not None:
                records[(bin_dir / name).relative_to(install_dir).as_posix()] = record
        for name in linked_runtime_directories or ():
            # Same bare-name rule: rglob on a marker-supplied path would walk out of the install.
            if not isinstance(name, str) or not name or Path(name).name != name:
                continue
            runtime_dir = bin_dir / name
            if not runtime_dir.is_dir():
                continue
            for path in sorted(runtime_dir.rglob("*")):
                # Files only; stat follows symlinks, which is what the loader does too.
                if not path.is_file():
                    continue
                record = _stat_record(path)
                if record is not None:
                    records[path.relative_to(install_dir).as_posix()] = record
        # Last, so a server the wiring loop happened to match is upgraded to the hashed tier.
        for name in _RUNTIME_RECORD_SERVER_NAMES:
            candidate = bin_dir / name
            record = _stat_record(candidate)
            if record is None:
                continue
            try:
                record["sha256"] = sha256_file(candidate)
            except (OSError, MemoryError) as exc:
                # No record at all: a size-only tier would keep a same-size corrupt server forever.
                log(f"could not hash {name} for the runtime record ({exc}); not recording")
                return {}
            records[candidate.relative_to(install_dir).as_posix()] = record
    return records


def runtime_file_records(install_dir: Path, selection: Any) -> "dict[str, dict[str, Any]]":
    """prebuilt_core.write_prebuilt_metadata's hook: the record for the tree just staged.

    Slim bundles pass their wired ggml filenames through, so the marker describes the
    hardlinks it is about to claim as well as the server it shipped.
    """
    is_slim = getattr(selection, "install_kind", None) == "slim"
    linked = selection.linked_libraries if is_slim else None
    runtime_dirs = getattr(selection, "linked_runtime_directories", None) if is_slim else None
    return _runtime_file_records(install_dir, linked, runtime_dirs)


def _runtime_files_match(install_dir: Path, marker: "dict[str, Any]") -> bool:
    """Whether every recorded payload file is still the file that was installed.

    Sizes for everything, digests for the server that carries one (see
    _runtime_file_records for why the split). Copied from the llama installer's
    _runtime_files_match, including both of its deliberate choices:

      * mtime_ns is recorded but deliberately NOT compared. A restore from backup, an
        rsync or a container layer rewrites it without changing a byte, and the answer
        to a mismatch here is a 200-400 MB re-download.
      * it FAILS CLOSED on a record that is present but not a non-empty mapping of
        mappings. This is the evidence that replaces actually starting whisper-server,
        so a record that cannot be read is not proof of anything.

    The one divergence from llama is where "absent" is decided, and backwards
    compatibility forces it: every whisper marker already on a user's disk predates this
    key, and rejecting those would re-download an install that is perfectly fine. So an
    ABSENT record is accepted here and backfilled once, under the install lock, by
    settle_kept_install -> _backfill_runtime_file_records. It is
    _existing_install_is_intact -- the no-network fast path, which keeps an install
    having looked at no bytes at all -- that demands the key and takes the full path once
    without it, exactly as it already does for paired_llama_ggml_tree.
    """
    if "runtime_files" not in marker:
        return True
    recorded = marker.get("runtime_files")
    if not isinstance(recorded, dict) or not recorded:
        log(f"existing install at {install_dir} has an unusable payload record; reinstalling")
        return False
    for relative, expected in recorded.items():
        if not isinstance(expected, dict):
            log(f"existing install at {install_dir} has an unusable record for {relative}")
            return False
        candidate = install_dir / relative
        try:
            info = candidate.stat()
            if info.st_size != expected.get("size"):
                log(
                    f"existing install at {install_dir} rejected: {relative} is "
                    f"{info.st_size} bytes"
                )
                return False
            digest = expected.get("sha256")
            if digest is not None and sha256_file(candidate) != digest:
                log(
                    f"existing install at {install_dir} rejected: {relative} does not match "
                    f"the recorded digest"
                )
                return False
        except OSError as exc:
            log(f"existing install at {install_dir} rejected: {relative} is unreadable ({exc})")
            return False
    # An unrecorded file is not evidence against the install; a RECORDED one that vanished is.
    return True


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
    if runtime is None or not llama_runtime_pairs(
        runtime[1],
        artifact.get("requires_llama_tag"),
        installed_ggml_tree = installed_llama_ggml_tree(),
        required_ggml_tree = artifact.get("requires_ggml_tree"),
        installed_repo = installed_llama_tree_repo(),
    ):
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
    return installed_tree_is_intact(install_dir, host)


def kept_install_needs_settling(install_dir: Path) -> bool:
    """Whether settle_kept_install has anything to write: a marker with no payload
    record, or a slim marker missing either half of its pairing record."""
    marker = load_prebuilt_metadata(install_dir)
    if not marker:
        return False
    # Empty counts as absent: _runtime_file_records answers {} for a server it could not hash.
    if not marker.get("runtime_files"):
        return True
    if marker.get("install_kind") != "slim":
        return False
    return not all(
        isinstance(marker.get(key), str) and marker.get(key)
        for key in ("paired_llama_ggml_tree", "paired_llama_runtime_id")
    )


def settle_kept_install(install_dir: Path) -> None:
    """core.install_selected_prebuilt's hook for a kept install, called under the lock.

    The backfill is a read-modify-write of the marker. Done from existing_install_matches
    it ran once BEFORE the lock, where another installer swapping in a new release
    between the read and the replace would have had its fresh marker overwritten with
    the old release's fields plus the backfilled tree.
    """
    _backfill_slim_pairing_record(install_dir)
    _backfill_runtime_file_records(install_dir)


def _backfill_runtime_file_records(install_dir: Path) -> None:
    """Record the payload's sizes and digests on a marker written before that key existed.

    An install made by an older Unsloth Studio carries no runtime_files, and
    existing_install_current_without_plan demands one: without this backfill every such
    install would take the full path on EVERY update -- fetching the release, its
    manifest and its checksum index each time -- instead of once. Nothing here re-downloads
    anything: the comparison in installed_tree_is_intact accepts an absent record precisely
    so that an upgrade never costs 200-400 MB for a tree that is fine.

    Added, never corrected: a marker that already carries a record was written by a run
    that hashed the bytes it installed, and overwriting it would re-bless bytes this run
    did not choose. What this writes is what is on disk NOW, which is the only thing an
    offline run can honestly say -- a legacy install already damaged before this ran is
    recorded as damaged. That is the inherent limit of a record introduced after the fact;
    from the moment it is written the bytes are pinned.

    Never raises -- the install is already valid, and a read-only marker must not fail
    setup over a metadata refresh.
    """
    marker = load_prebuilt_metadata(install_dir)
    if not marker or marker.get("runtime_files"):
        return
    is_slim = marker.get("install_kind") == "slim"
    linked = marker.get("linked_libraries") if is_slim else None
    runtime_dirs = marker.get("linked_runtime_directories") if is_slim else None
    records = _runtime_file_records(
        install_dir,
        linked if isinstance(linked, list) else None,
        runtime_dirs if isinstance(runtime_dirs, list) else None,
    )
    # An empty record is not evidence, and writing one would fail closed on every later update.
    if not records:
        return
    marker["runtime_files"] = records
    # llama's writer: same atomic temp-and-replace, mode and owner kept, never raises.
    if llama._write_marker(metadata_path(install_dir), marker):
        log(f"existing {COMPONENT} install reused; recorded {len(records)} payload files")


def _wiring_matches_live_llama(install_dir: Path, marker: dict) -> bool:
    """Whether this install's wired libraries ARE the live llama runtime's, right now.

    Recording a pairing the wiring has not been checked against is how a FALSE one gets
    written: the backfill reads whichever llama marker is live at that moment, and a llama
    that was replaced before the first migration -- or by another installer holding its own
    per-directory lock during this one -- leaves whisper hardlinked to the previous libraries
    while the marker claims the new identity. The next fast check then believes it.

    _link_or_copy hardlinks, so a shared (st_dev, st_ino) PROVES the two names are one file.
    It falls back to shutil.copy2 across filesystems, where the inodes legitimately differ, so
    an inode mismatch is not evidence of staleness on its own and the bytes are compared
    instead. Unverifiable answers False: recording nothing costs the fast path, recording a
    guess costs the install.
    """
    linked = marker.get("linked_libraries")
    linked_from = marker.get("linked_from")
    if not isinstance(linked, list) or not linked or not isinstance(linked_from, str):
        return False
    source_dir = Path(linked_from)
    if not source_dir.is_dir():
        return False
    bin_dir = installed_server_path(install_dir, detect_host()).parent
    checked = 0
    for name in linked:
        if not isinstance(name, str) or Path(name).name != name:
            return False
        ours, theirs = bin_dir / name, source_dir / name
        try:
            a, b = ours.stat(), theirs.stat()
        except OSError:
            return False
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            # Copied rather than hardlinked, or genuinely stale. Only the bytes can say.
            try:
                if a.st_size != b.st_size or core.sha256_file(ours) != core.sha256_file(theirs):
                    return False
            except OSError:
                return False
        checked += 1
    return checked > 0


def _backfill_slim_pairing_record(install_dir: Path) -> None:
    """Record the paired llama runtime on a slim marker written before those keys existed.

    existing_install_current_without_plan refuses a slim install whose marker cannot say
    which llama runtime it hardlinks, so without this an install made before this PR
    would fetch the release, its manifest and its checksum index on EVERY update rather
    than once. This is the only place that re-examines a slim install without
    reinstalling it, and it runs only after the fingerprint and the wiring have just been
    confirmed, so what it writes describes a pairing it verified.

    Backfilled rather than re-wired on purpose. An older marker records nothing about which
    gfx bundle it was paired against, so a swap that already happened cannot be detected
    from it at all, and reinstalling every such install to find out would re-download the
    bundle for every existing user to answer a question about a swap that almost never
    happened. Recording the current pairing makes the NEXT one detectable, which is the
    same trade paired_llama_ggml_tree already makes.

    Added, never corrected: a marker that already names a pairing was written by a run that
    installed against it. Never raises -- the install is already valid, and a read-only
    marker must not fail setup over a metadata refresh.
    """
    marker = load_prebuilt_metadata(install_dir)
    if not marker or marker.get("install_kind") != "slim":
        return
    # Only record a pairing this install can be SHOWN to have. Without this the backfill
    # writes whichever llama is live at the moment it runs, which is a guess whenever llama
    # changed before whisper's first migration or while another installer held llama's own
    # lock, and the runtime-id check then trusts the guess forever.
    if not _wiring_matches_live_llama(install_dir, marker):
        log(
            f"existing {COMPONENT} install reused; its wiring does not match the live "
            "llama.cpp runtime, so no pairing was recorded"
        )
        return
    added: list[str] = []
    for key, live in (
        ("paired_llama_ggml_tree", installed_paired_runtime_tree),
        ("paired_llama_runtime_id", installed_paired_runtime_id),
    ):
        recorded = marker.get(key)
        if isinstance(recorded, str) and recorded:
            continue
        value = live()
        if not value:
            continue
        marker[key] = value
        added.append(f"{key}={value}")
    if not added:
        return
    # llama's writer: same atomic temp-and-replace, mode and owner kept, never raises.
    if llama._write_marker(metadata_path(install_dir), marker):
        log(f"existing {COMPONENT} install reused; recorded its pairing ({', '.join(added)})")


def installed_tree_is_intact(install_dir: Path, host: HostInfo) -> bool:
    """The on-disk half of existing_install_matches, without the fingerprint.

    Split out so existing_install_current_without_plan -- which runs before anything is
    resolved and so has no selection to fingerprint against -- makes exactly the same
    demands of the tree. Two copies of these rules is how a fast path comes to accept an
    install the slow path repairs.
    """
    server = installed_server_path(install_dir, host)
    try:
        if not server.is_file() or server.stat().st_size == 0:
            return False
    except OSError:
        return False
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
        # Which llama INSTALL, not which source tree: a per-gfx ROCm reselection swaps the asset
        # within one release, and the hardlinks survive llama's directory swap on purpose, so the
        # previous GPU's kernels stay wired while every other check passes. Absent is a pre-key
        # marker, which settle_kept_install records rather than re-downloading.
        recorded_runtime_id = marker.get("paired_llama_runtime_id")
        if isinstance(recorded_runtime_id, str) and recorded_runtime_id:
            if recorded_runtime_id != installed_paired_runtime_id():
                log(
                    f"existing slim install at {install_dir} is wired to a superseded llama "
                    "runtime; reinstalling"
                )
                return False
        runtime_dirs = marker.get("linked_runtime_directories")
        # Subset plus required, not equality: a target without hipBLASLt kernels wires rocblas alone and is
        # complete (#8364), while an unknown name or a missing rocblas still means stale wiring.
        if (
            marker.get("backend") == "rocm"
            and not host.is_windows
            and (
                not isinstance(runtime_dirs, list)
                or not set(runtime_dirs) <= set(SLIM_ROCM_RUNTIME_DIRS)
                or not set(SLIM_ROCM_REQUIRED_RUNTIME_DIRS) <= set(runtime_dirs)
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
    # The only check here that reads the payload's BYTES: a whisper-server truncated to a non-zero
    # length is still a non-empty executable. Absent on a pre-record marker, which is kept and
    # backfilled under the lock, so upgrading from an older Studio never re-downloads.
    return _runtime_files_match(install_dir, marker)


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
    try:
        return core.fetch_release_for_install(
            _OPS, repo, published_release_tag = published_release_tag
        )
    except PrebuiltFallback:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        # install_prebuilt's keep path reads only PrebuiltFallback, so without this wrap an offline
        # update printed "prebuilt install failed" over an intact tree.
        raise PrebuiltFallback(
            f"could not fetch release {repo}@{published_release_tag or 'latest'}: {exc}"
        ) from exc


@dataclass(frozen = True)
class WhisperReleasePlan:
    bundle: ReleaseBundle
    selection: InstallSelection | None
    artifact: dict[str, Any]
    resolved_backend: str
    used_fallback: bool
    # The macOS release walk-back behind bundle (core.WalkBack); None when bundle is the newest.
    walk_back: core.WalkBack | None = None


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
    # An upstream version pin searches manifests directly.
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

    # An API limit or network failure is "could not answer", which the keep path handles.
    try:
        compatible_tags = _published_release_tags(published_repo)
    except (OSError, RuntimeError) as exc:
        raise PrebuiltFallback(f"could not list {published_repo} releases: {exc}") from exc
    for release_tag in compatible_tags:
        if first_bundle is not None and release_tag == first_bundle.release_tag:
            continue
        try:
            bundle = _fetch_release_candidate(published_repo, release_tag)
        except PrebuiltFallback:
            continue
        if not _bundle_matches_whisper_tag(bundle, whisper_tag):
            continue
        try:
            # Establish host compatibility before fetching or trusting this candidate's checksum index: once
            # selected, integrity failures must stop the install rather than silently downgrade again.
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
        walk_back = (
            core.walk_back_for(host, first_bundle.release_tag)
            if first_bundle is not None and not requested_specific_tag
            else None
        )
        if walk_back is not None:
            # So the marker-only check holds the install while that release is newest for this macOS.
            plan = replace(
                plan,
                walk_back = walk_back,
                selection = (
                    replace(plan.selection, walk_back = walk_back)
                    if plan.selection is not None
                    else None
                ),
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


def installed_paired_runtime_tree() -> str | None:
    """The ggml tree of the llama runtime a slim bundle would hardlink right now.

    Read from the live llama marker, so prebuilt_core can record it beside
    paired_llama_tag without knowing anything about llama.cpp.
    """
    tree = installed_llama_ggml_tree()
    return tree if isinstance(tree, str) and tree else None


def installed_paired_runtime_id(install_dir: "Path | None" = None) -> str | None:
    """Which llama INSTALL the hardlinks point into, not which source tree built it.

    ggml_tree cannot answer this: llama publishes a per-gfx ROCm bundle per release, so
    re-selecting for another gfx target swaps the asset while the tree id stays put. Its
    install_fingerprint covers asset, asset_sha256 and runtime_sha256, so it moves whenever
    the bytes behind the hardlinks are superseded.
    """
    root = install_dir if install_dir is not None else llama.default_managed_llama_dir()
    metadata = llama.load_prebuilt_metadata(root)
    if not metadata:
        return None
    recorded = metadata.get("install_fingerprint")
    return recorded if isinstance(recorded, str) and recorded else None


def _existing_install_is_intact(
    install_dir: Path, host: HostInfo, *, published_repo: str, requested_backend: str
) -> dict[str, Any] | None:
    """The marker of a whisper.cpp install worth keeping, or None if there is none.

    Everything existing_install_current_without_plan can establish WITHOUT asking which
    release is newest: that a previous run of this installer finished and wrote the
    marker, that it wrote it for this repo and this backend, that a slim install still
    hardlinks the llama ggml tree it was wired against, and that the tree on disk is the
    shape the marker describes.

    Split out because install_prebuilt's failed-lookup path has to make exactly these
    demands and no others -- it runs precisely when "is there a newer release" is the one
    question nothing can answer -- and a second copy of the remaining rules is how a keep
    path comes to hold an install the install path would have repaired.
    """
    marker = load_prebuilt_metadata(install_dir)
    if not marker:
        return None
    if marker.get("schema_version") != SCHEMA_VERSION or marker.get("component") != COMPONENT:
        return None
    if (marker.get("published_repo") or "") != published_repo:
        return None
    if marker.get("backend") != requested_backend:
        return None
    # A bundle for another architecture (a home directory carried between machines) is not intact.
    # A marker predating the recorded os/arch is read from its asset name, a convention a custom
    # repository need not follow.
    os_token, arch_token = host_platform_tokens(host)
    recorded_os, recorded_arch = marker.get("os"), marker.get("arch")
    if isinstance(recorded_os, str) and isinstance(recorded_arch, str):
        if (recorded_os, recorded_arch) != (os_token, arch_token):
            return None
    else:
        recorded_asset = marker.get("asset")
        if (
            not isinstance(recorded_asset, str)
            or f"-{os_token}-{arch_token}-" not in recorded_asset
        ):
            return None
    # An install restored onto an older Mac has the right tokens and fails at load time.
    min_os = marker.get("min_os")
    coverage = marker.get("coverage")
    if min_os is None and isinstance(coverage, dict):
        min_os = coverage.get("min_os")
    if host.is_macos and min_os is not None:
        if not _macos_min_os_ok(host, min_os):
            return None
    recorded_release = marker.get("release_tag")
    if not isinstance(recorded_release, str) or not recorded_release:
        return None
    # A marker without a fingerprint is not a record of a finished install.
    recorded_fingerprint = marker.get("install_fingerprint")
    if not isinstance(recorded_fingerprint, str) or not recorded_fingerprint:
        return None
    # ...and self-consistent: with no plan to compare against, this is what stands between a
    # release_tag edited over an old binary and "current". Predating it recomputes to None.
    if core.marker_install_fingerprint(marker) != recorded_fingerprint:
        return None
    # A slim install is only as intact as the llama ggml it hardlinks; predating the key is full path once.
    if marker.get("install_kind") == "slim":
        recorded_tree = marker.get("paired_llama_ggml_tree")
        if not isinstance(recorded_tree, str) or not recorded_tree:
            return None
        if recorded_tree != installed_llama_ggml_tree():
            return None
    # ...and the payload record itself: this path never asks the network and never starts
    # whisper-server, so the recorded digests are the only evidence the bytes are the installed
    # ones. Predating the record is full path once, as with paired_llama_ggml_tree above.
    if not marker.get("runtime_files") or not isinstance(marker.get("runtime_files"), dict):
        return None
    if not installed_tree_is_intact(install_dir, host):
        return None
    return marker


# What the fork appends to the upstream tag it packages: v1.9.2 -> v1.9.2-unsloth.17.
_PACKAGING_SUFFIX = "-unsloth."


def _api_newest_release_tag_for_upstream(
    repo: str, whisper_tag: str, recorded_release: str
) -> "str | None":
    """The newest published release packaging *whisper_tag* (v1.9.2-unsloth.17, .18, ...).

    The release the marker records is known to package it, so it is a candidate too;
    the newest of them is what _release_plan_for_host would take.
    """
    wanted = _normalized_upstream_tag(whisper_tag)
    try:
        releases = llama.github_releases(
            repo, max_pages = llama.DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES
        )
    except Exception as exc:  # noqa: BLE001 - unreachable is a reason to do the work
        log(f"could not list the {COMPONENT} releases packaging {whisper_tag} ({exc})")
        return None
    matching = []
    for release in releases:
        tag = release.get("tag_name") if isinstance(release, dict) else None
        if not isinstance(tag, str):
            continue
        # The upstream part can carry a hyphen (v1.9.2-rc1), so the tag is neither cut at its first
        # hyphen nor matched on any: both read v1.9.2-rc1-unsloth.2 as a packaging of v1.9.2.
        packaged = _normalized_upstream_tag(tag)
        if (
            tag == recorded_release
            or packaged == wanted
            or packaged.startswith(wanted + _PACKAGING_SUFFIX)
        ):
            matching.append(release)
    # Repo-aware selectability: whisper is never ggml-org/llama.cpp, so this is the
    # plain drafts-and-prereleases-dropped rule it has always had.
    return llama._newest_release_tag_from_releases(repo, matching)


def existing_install_current_without_plan(
    install_dir: Path,
    host: HostInfo,
    *,
    whisper_tag: str,
    published_repo: str,
    published_release_tag: str | None,
    requested_backend: str,
) -> bool:
    """Whether the whisper.cpp install on disk is already the one this run would make.

    Runs BEFORE _release_plan_for_host, which fetches the release, its manifest and its
    checksum index on every update to conclude that nothing moved. Every check is on-disk
    evidence or a tag comparison; the one network call is the same HEAD the llama
    pre-check makes, and only when the requested tag is "latest".

    The slim pairing is what makes this component's version of the check different: a
    slim whisper bundle hardlinks the llama runtime's ggml libraries, so a llama install
    that moved underneath it invalidates a whisper install whose own release did not.
    That half is _existing_install_is_intact, and it runs FIRST, for the reason llama's
    pre-check orders its own probes that way: a box whose install is damaged should not
    pay a network round trip to learn it must reinstall anyway.
    """
    if llama.prebuilt_full_check_requested():
        return False
    marker = _existing_install_is_intact(
        install_dir,
        host,
        published_repo = published_repo,
        requested_backend = requested_backend,
    )
    if marker is None:
        return False
    recorded_release = str(marker.get("release_tag"))
    pinned = (published_release_tag or "").strip()
    requested = (whisper_tag or "latest").strip().lower()
    if requested not in ("", "latest"):
        # An upstream pin, checked even with a release pin: the full path refuses a pinned release
        # targeting another upstream version (_bundle_matches_whisper_tag). This fork publishes
        # several packagings of one upstream tag and takes the newest, so without a release pin the
        # upstream_tag alone cannot answer.
        if _normalized_upstream_tag(
            str(marker.get("upstream_tag") or "")
        ) != _normalized_upstream_tag(whisper_tag):
            return False
    if pinned:
        if pinned != recorded_release:
            return False
    elif requested not in ("", "latest"):
        # The newest packaging of THAT version, which /releases/latest cannot name once a newer
        # upstream ships: ask the release list, by tag name, which only the fork's convention
        # supports; a custom repository is matched by manifest on the full path.
        if published_repo != DEFAULT_PUBLISHED_REPO:
            return False
        newest = _api_newest_release_tag_for_upstream(published_repo, whisper_tag, recorded_release)
        if not newest or newest != recorded_release:
            return False
    else:
        if not llama._download_host_resolve_enabled():
            return False
        try:
            latest = llama._download_host_latest_release_tag(published_repo)
        except Exception as exc:  # noqa: BLE001 - unreachable is a reason to do the work
            log(f"could not resolve the latest {COMPONENT} release without the API ({exc})")
            return False
        if not latest:
            return False
        if latest != recorded_release and not core.walk_back_stands(marker, host, latest):
            # A recorded macOS walk-back stands while the newest release is still the skipped one on
            # the same OS version; a newer release or an OS upgrade re-decides it.
            return False
    # "already matches" is the substring setup.sh:3635 and setup.ps1:6109 grep for.
    log(
        f"existing {COMPONENT} install already matches {recorded_release} "
        f"({marker.get('backend')}); nothing to do"
    )
    return True


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
    if not force and existing_install_current_without_plan(
        install_dir,
        host,
        whisper_tag = whisper_tag,
        published_repo = published_repo,
        published_release_tag = published_release_tag,
        requested_backend = requested_backend,
    ):
        return 0
    try:
        plan = _release_plan_for_host(
            host,
            published_repo = published_repo,
            published_release_tag = published_release_tag,
            whisper_tag = whisper_tag,
            requested_backend = requested_backend,
        )
    except (ReleaseCompatibilityError, core.ReleaseIntegrityError):
        # The lookup ANSWERED, so keeping would paper over a real answer. Two kinds:
        # ReleaseCompatibilityError, no published bundle pairs with this host's llama.cpp runtime
        # (real release skew, setup names both tags from exit 2); and ReleaseIntegrityError, the
        # release was fetched and found untrustworthy -- an asset outside the checksum index, or a
        # manifest digest disagreeing with it. Reporting "update unavailable, existing prebuilt
        # kept" over a tamper signal would turn it into a routine offline notice.
        raise
    except PrebuiltFallback as exc:
        # llama.cpp's rule: a lookup that could not answer says nothing about the tree on disk. A
        # strict offline update used to print "prebuilt install failed" over a healthy install.
        # Anything this RUN asked for that keeping would ignore fails instead. Not --has-rocm or
        # --rocm-gfx (both entrypoints forward DETECTED hardware on every AMD host); not
        # --published-repo, --backend or --cpu-fallback (a backend request), which
        # _existing_install_is_intact compares against the marker. UNSLOTH_WHISPER_FORCE_COMPILE
        # counts: setup.sh runs the source build only after a nonzero exit.
        explicit_release_request = (
            force
            or bool((published_release_tag or "").strip())
            or (whisper_tag or "latest").strip().lower() not in ("", "latest")
            or os.environ.get("UNSLOTH_WHISPER_FORCE_COMPILE", "").strip() == "1"
        )
        marker = (
            None
            if explicit_release_request
            else _existing_install_is_intact(
                install_dir,
                host,
                published_repo = published_repo,
                requested_backend = requested_backend,
            )
        )
        if marker is None:
            raise
        # "keeping the existing complete install" is the substring setup.sh and setup.ps1 grep for
        # llama's identical outcome. Not "already matches" or "installed": both name a release this
        # run never fetched.
        log(
            f"{COMPONENT} update unavailable, existing prebuilt kept; keeping the "
            f"existing complete install of {marker.get('release_tag')}"
        )
        # llama.cpp's wording, so update_flow reads both installers alike. log_lines: a multi-line
        # reason is otherwise indistinguishable from unprefixed diagnostics.
        log_lines(f"prebuilt update reason: {exc}".splitlines())
        return EXIT_SUCCESS
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


def unavailable_payload(published_repo: str, exc: BaseException) -> dict[str, Any]:
    """The resolver's negative answer, carrying WHY: the same split the install path
    makes (ReleaseCompatibilityError is exit 2, everything else exit 1). Flattened, a
    caller cannot tell a confirmed pairing gap from a probe that never answered."""
    return {
        "prebuilt_available": False,
        "repo": published_repo,
        "unavailable_reason": (
            "incompatible" if isinstance(exc, ReleaseCompatibilityError) else "unresolved"
        ),
    }


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
    except PrebuiltFallback as exc:
        return unavailable_payload(published_repo, exc)
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


# The declarative form of everything above, for descriptor-driven consumers; the shipped CLI runs
# through this module's wrappers so the monkeypatch seams stay intact.
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
        except PrebuiltFallback as exc:
            payload = unavailable_payload(args.published_repo, exc)
        except Exception as exc:  # noqa: BLE001 - probe must never crash the caller
            log(f"resolve failed: {exc}")
            payload = unavailable_payload(args.published_repo, exc)
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
        # One prefixed line each: a multi-line reason is otherwise indistinguishable
        # from unprefixed diagnostics for whoever reads this output back.
        log_lines(f"prebuilt install failed: {exc}".splitlines())
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        log(f"unexpected error: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
