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

The heavy machinery -- verified downloads with retries and token-safe redirects,
safe archive extraction, the install lock, host detection (NVIDIA caps + driver,
ROCm gfx, macOS version), and the coverage-aware CUDA selection primitives -- is
imported from ``install_llama_prebuilt.py`` (same directory); this module only
keeps the whisper manifest dialect, install-tree assembly, CPU fallback policy,
marker, and CLI.

Mirrors ``install_node_prebuilt.py`` / ``install_llama_prebuilt.py`` so the setup
scripts drive it the same way. Exit codes: 0 success (or already current), 1 error,
2 source fallback, 3 busy. A re-run that already matches logs "already matches"
and returns 0 without downloading (the scripts grep it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Put studio/ on sys.path so install_llama_prebuilt resolves whether this file is
# run as a script (from any cwd) or imported by the tests.
_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import install_llama_prebuilt as llama  # noqa: E402

# Shared machinery re-exported as module globals. The retained code resolves these
# by bare name at call time so tests can monkeypatch them on this module.
PrebuiltFallback = llama.PrebuiltFallback
BusyInstallConflict = llama.BusyInstallConflict
auth_headers = llama.auth_headers
download_bytes = llama.download_bytes
download_file = llama.download_file
fetch_json = llama.fetch_json
sha256_file = llama.sha256_file
_URL_OPENER = llama._URL_OPENER
install_lock = llama.install_lock
install_lock_path = llama.install_lock_path
parse_macos_version = llama.parse_macos_version
llama_detect_host = llama.detect_host
detect_torch_cuda_runtime_preference = llama.detect_torch_cuda_runtime_preference


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

DEFAULT_PUBLISHED_REPO = "unslothai/whisper.cpp"
# Release assets published by the fork's prebuilt CI.
MANIFEST_ASSET_NAME = "whisper-prebuilt-manifest.json"
SHA256_ASSET_NAME = "whisper-prebuilt-sha256.json"

METADATA_FILENAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# Backends the installer knows how to select. Asset filenames carry a finer
# accelerator token (e.g. cuda12); selection matches the manifest artifact's
# coarse `backend` field so a new CUDA toolkit needs no code change here.
SUPPORTED_BACKENDS = ("cpu", "cuda", "metal", "vulkan", "rocm")

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


# ── Backend selection ──
def auto_detect_backend(host: HostInfo) -> str:
    """Host-preferred backend: Apple Silicon -> metal, usable NVIDIA -> cuda,
    AMD gfx -> rocm, otherwise cpu."""
    if host.is_apple_silicon:
        return "metal"
    if host.has_usable_nvidia:
        return "cuda"
    if host.has_rocm:
        return "rocm"
    return "cpu"


def resolve_backend(host: HostInfo, requested: str | None, *, cpu_fallback: bool) -> str:
    """Resolve the effective backend from --backend / --cpu-fallback and detection."""
    if cpu_fallback:
        return "cpu"
    value = (requested or "auto").strip().lower()
    if value in {"", "auto"}:
        return auto_detect_backend(host)
    if value in SUPPORTED_BACKENDS:
        return value
    raise PrebuiltFallback(
        f"unsupported --backend '{requested}'; choose from auto,{','.join(SUPPORTED_BACKENDS)}"
    )


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


# ── Manifest (release-side artifact catalogue) ──
def validate_schema_version(payload: dict[str, Any], *, label: str) -> None:
    schema_version = payload.get("schema_version")
    if schema_version is None:
        return
    try:
        normalized = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise PrebuiltFallback(f"{label} schema_version was not an integer") from exc
    if normalized != SCHEMA_VERSION:
        raise PrebuiltFallback(f"{label} schema_version={normalized} is unsupported")


def parse_manifest(payload: Any, *, label: str = MANIFEST_ASSET_NAME) -> dict[str, Any]:
    """Validate a whisper.cpp prebuilt manifest and return it normalized.

    Rejects a manifest with an unknown schema_version or a component other than
    'whisper.cpp'. Returns a dict with keys: schema_version, component,
    studio_protocol, upstream_tag, source_commit, artifacts (list of dicts).
    """
    if not isinstance(payload, dict):
        raise PrebuiltFallback(f"{label} was not a JSON object")
    validate_schema_version(payload, label = label)
    component = payload.get("component")
    if component != COMPONENT:
        raise PrebuiltFallback(f"{label} describes component {component!r}, expected {COMPONENT!r}")

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise PrebuiltFallback(f"{label} omitted an 'artifacts' list")
    artifacts: list[dict[str, Any]] = []
    for index, raw in enumerate(artifacts_raw):
        if not isinstance(raw, dict):
            log(f"{label} artifact[{index}] ignored: not an object")
            continue
        asset = raw.get("asset")
        if not isinstance(asset, str) or not asset:
            log(f"{label} artifact[{index}] ignored: missing asset name")
            continue
        artifacts.append(raw)

    studio_protocol = payload.get("studio_protocol")
    return {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "studio_protocol": studio_protocol if isinstance(studio_protocol, str) else None,
        "upstream_tag": payload.get("upstream_tag")
        if isinstance(payload.get("upstream_tag"), str)
        else None,
        "source_commit": payload.get("source_commit")
        if isinstance(payload.get("source_commit"), str)
        else None,
        "artifacts": artifacts,
    }


def _macos_min_os_ok(host: HostInfo, min_os: Any) -> bool:
    """True if a macOS artifact requiring `min_os` can load here. The manifest
    labels it `macos-<version>` (e.g. `macos-14.0`), so strip that prefix before
    parsing or every entry parses as None and the guard is a no-op. Unknown host
    or min_os -> True (defer to runtime validation). Llama parity."""
    if not isinstance(min_os, str) or not min_os.strip():
        return True
    raw = min_os.strip()
    if raw.lower().startswith("macos-"):
        raw = raw[len("macos-") :]
    required = parse_macos_version(raw)
    if required is None or host.macos_version is None:
        return True
    return host.macos_version >= required


def _artifacts_for_host(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> list[dict[str, Any]]:
    """Manifest artifacts matching this host os/arch/backend. On macOS, drop any
    whose `min_os` exceeds the host version (llama parity); ignored off macOS."""
    return [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("os") == host.whisper_os
        and artifact.get("arch") == host.whisper_arch
        and artifact.get("backend") == backend
        and (not host.is_macos or _macos_min_os_ok(host, artifact.get("min_os")))
    ]


# ── On-disk CUDA runtime detection (llama's dir scan, exact SONAMEs) ──
def detected_linux_runtime_lines() -> list[str]:
    """`cuda<major>` lines whose libcudart + libcublas SONAMEs are on disk, newest
    first. Exact-name matching: the loader resolves the SONAME (libcudart.so.13),
    so a versioned-only file without that symlink is not loadable and a `{lib}*`
    glob would overreport."""
    detected: list[str] = []
    for major in range(llama._MAX_PROBE_CUDA_MAJOR, llama._MIN_CUDA_MAJOR - 1, -1):
        required = [f"libcudart.so.{major}", f"libcublas.so.{major}"]
        dirs = llama.linux_runtime_dirs_for_required_libraries(required)
        if all(
            any(llama.dir_provides_exact_library(directory, library) for directory in dirs)
            for library in required
        ):
            detected.append(f"cuda{major}")
    return detected


def detected_cuda_runtime_lines(*, is_windows: bool) -> list[str]:
    """Platform-appropriate on-disk CUDA runtime-line detection (newest first)."""
    if is_windows:
        return llama.detected_windows_runtime_lines()[0]
    return detected_linux_runtime_lines()


# ── Coverage-aware selection (llama primitives over the whisper manifest) ──
def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _published_view(artifact: dict[str, Any]) -> Any:
    """Adapt a whisper manifest artifact to llama's PublishedLlamaArtifact so the
    shared coverage primitives apply, normalizing dotted SMs to bare strings."""
    return llama.PublishedLlamaArtifact(
        asset_name = str(artifact.get("asset") or ""),
        install_kind = "whisper",
        runtime_line = artifact.get("runtime_line"),
        coverage_class = artifact.get("coverage_class"),
        supported_sms = [
            sm
            for sm in (
                llama.normalize_compute_cap(value)
                for value in (artifact.get("supported_sms") or ())
            )
            if sm is not None
        ],
        min_sm = _coerce_int(artifact.get("min_sm")),
        max_sm = _coerce_int(artifact.get("max_sm")),
        bundle_profile = None,
        rank = _coerce_int(artifact.get("rank")) or 0,
        gfx_target = artifact.get("gfx_target"),
        mapped_targets = [str(value) for value in (artifact.get("mapped_targets") or ())],
    )


def _select_cuda_artifact(
    candidates: list[dict[str, Any]], host: HostInfo, log_lines: list[str]
) -> dict[str, Any] | None:
    """Coverage-aware CUDA pick (llama's linux_cuda_choice_from_release policy):
    a runtime line is usable only when the driver supports it AND its libs are on
    disk (bundles omit libcudart/libcublas); a Blackwell host prefers the highest
    covering major, torch's line is a non-Blackwell tie-break; within a line every
    host SM must be covered, tightest SM range wins, portable is the per-line
    fallback (and the only acceptable class with unknown caps). None when nothing
    covers -- the caller then falls back to CPU."""
    views = [(_published_view(artifact), artifact) for artifact in candidates]
    host_sms = llama.normalize_compute_caps(host.compute_caps)
    detected = detected_cuda_runtime_lines(is_windows = host.is_windows)
    # Duck-typed on driver_cuda_version; the major floor is platform independent.
    driver_lines = llama.compatible_linux_runtime_lines(host)
    runtime_lines = [line for line in detected if line in driver_lines]
    log_lines.append(
        f"cuda_selection: detected_sms={','.join(host_sms) if host_sms else 'unknown'}"
    )
    log_lines.append("cuda_selection: driver_runtime_lines=" + (",".join(driver_lines) or "none"))
    log_lines.append("cuda_selection: detected_runtime_lines=" + (",".join(detected) or "none"))
    log_lines.append(
        "cuda_selection: compatible_runtime_lines=" + (",".join(runtime_lines) or "none")
    )
    if not runtime_lines:
        log_lines.append("cuda_selection: no usable CUDA runtime line (driver + on-disk runtime)")
        return None

    ordered = list(runtime_lines)
    blackwell_lines = (
        [
            line
            for line in llama._blackwell_capable_linux_runtime_lines(
                host_sms, [view for view, _ in views]
            )
            if line in ordered
        ]
        if llama._host_is_blackwell(host)
        else []
    )
    if blackwell_lines:
        ordered = blackwell_lines + [line for line in ordered if line not in blackwell_lines]
        log_lines.append(
            "cuda_selection: blackwell_runtime_override prefer=" + ",".join(blackwell_lines)
        )
    elif host.torch_runtime_line in ordered:
        ordered = [host.torch_runtime_line] + [
            line for line in ordered if line != host.torch_runtime_line
        ]
        log_lines.append(f"cuda_selection: torch_preferred_runtime_line={host.torch_runtime_line}")

    for runtime_line in ordered:
        # llama's accept rule: full SM coverage with real metadata; with unknown
        # caps only a portable bundle is acceptable.
        accepted = [
            (view, raw)
            for view, raw in views
            if view.runtime_line == runtime_line
            and llama._artifact_covers_sms(view, host_sms)
            and (host_sms or view.coverage_class == "portable")
        ]
        coverage = [(view, raw) for view, raw in accepted if view.coverage_class != "portable"]
        chosen = (
            min(
                coverage,
                key = lambda item: (llama._sm_range(item[0]), item[0].rank, item[0].max_sm or 0),
            )
            if coverage
            else next(iter(accepted), None)
        )
        if chosen is not None:
            view, raw = chosen
            log_lines.append(
                f"cuda_selection: selected {view.asset_name} runtime_line={runtime_line} "
                f"coverage_class={view.coverage_class}"
            )
            return raw
    log_lines.append("cuda_selection: no artifact covered the host GPUs")
    return None


def _select_rocm_artifact(
    candidates: list[dict[str, Any]], host_gfx: str | None, log_lines: list[str]
) -> dict[str, Any] | None:
    """Exact ROCm match: the host gfx must be listed in the bundle's
    mapped_targets or equal its umbrella gfx_target -- never a loose prefix, so an
    in-family but unbuilt arch (e.g. gfx1033) is not served the family bundle."""
    if not host_gfx:
        return None
    gfx = host_gfx.lower().strip()
    for artifact in candidates:
        mapped = {str(target).lower() for target in (artifact.get("mapped_targets") or ())}
        family = str(artifact.get("gfx_target") or "").lower()
        if gfx in mapped or (family and gfx == family):
            log_lines.append(f"rocm_selection: gpu={host_gfx} selected {artifact.get('asset')}")
            return artifact
    return None


def select_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """Best manifest artifact for this host os/arch and backend, or None.

    CPU/Metal/Vulkan take the first match. CUDA and ROCm are coverage-aware via
    llama's primitives: CUDA matches compute caps + driver against each bundle's
    supported_sms/min_sm/max_sm/runtime_line and picks the tightest
    Blackwell-aware profile, ROCm matches the gfx target exactly. So a B200
    (sm_100) gets a Blackwell bundle. None when nothing covers the host -- the
    caller then falls back to CPU."""
    candidates = _artifacts_for_host(manifest, host, backend)
    if not candidates:
        return None
    if backend not in {"cuda", "rocm"}:
        # cpu / metal / vulkan: no per-GPU coverage to match; first match wins.
        return candidates[0]
    log_lines: list[str] = []
    if backend == "cuda":
        chosen = _select_cuda_artifact(candidates, host, log_lines)
    else:
        chosen = _select_rocm_artifact(candidates, host.rocm_gfx, log_lines)
    for line in log_lines:
        log(line)
    return chosen


def select_artifact_with_cpu_fallback(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> tuple[dict[str, Any], str, bool]:
    """Select the backend artifact, else the CPU artifact of the same release.

    Returns (artifact, effective_backend, used_cpu_fallback). Raises
    PrebuiltFallback when neither the requested backend nor CPU has an asset.
    """
    artifact = select_artifact(manifest, host, backend)
    if artifact is not None:
        return artifact, backend, False
    if backend != "cpu":
        cpu_artifact = select_artifact(manifest, host, "cpu")
        if cpu_artifact is not None:
            log(
                f"no '{backend}' asset for {host.whisper_os}-{host.whisper_arch}; "
                f"falling back to the CPU asset of the same release"
            )
            return cpu_artifact, "cpu", True
    raise PrebuiltFallback(
        f"no whisper.cpp prebuilt asset for {host.whisper_os}-{host.whisper_arch} "
        f"(backend '{backend}')"
    )


def artifact_coverage(artifact: dict[str, Any]) -> dict[str, Any]:
    """The sm/gfx/min_os coverage recorded for an artifact (marker + fingerprint)."""
    coverage: dict[str, Any] = {}
    for key in ("sm_coverage", "gfx_coverage", "min_os", "sm", "gfx"):
        if key in artifact and artifact.get(key) is not None:
            coverage[key] = artifact.get(key)
    return coverage


# ── Release checksum index (trust anchor: the release's own sha256 asset) ──
def _valid_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
        return digest
    return None


def parse_release_checksums(repo: str, release_tag: str, payload: Any) -> dict[str, str]:
    """Asset name -> sha256 from a release's ``whisper-prebuilt-sha256.json``.

    Mirrors ``install_llama_prebuilt.py``: the checksum index published alongside
    the bundles is the authority for each asset's sha256. It is validated for
    schema/component and that its ``release_tag`` matches the release we resolved,
    so a redirected or mismatched index is rejected. A malformed index fails
    closed (source build)."""
    label = f"{SHA256_ASSET_NAME} in {repo}@{release_tag}"
    if not isinstance(payload, dict):
        raise PrebuiltFallback(f"{label} was not a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PrebuiltFallback(f"{label} has an unexpected schema_version")
    if payload.get("component") != COMPONENT:
        raise PrebuiltFallback(f"{label} did not describe {COMPONENT}")
    payload_tag = payload.get("release_tag")
    if not isinstance(payload_tag, str) or not payload_tag:
        raise PrebuiltFallback(f"{label} omitted release_tag")
    if payload_tag != release_tag:
        raise PrebuiltFallback(
            f"{label} release_tag={payload_tag} did not match the resolved release {release_tag}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PrebuiltFallback(f"{label} omitted an 'artifacts' map")
    checksums: dict[str, str] = {}
    for asset_name, entry in artifacts.items():
        if not isinstance(asset_name, str) or not asset_name or not isinstance(entry, dict):
            continue
        digest = _valid_sha256(entry.get("sha256"))
        if digest is not None:
            checksums[asset_name] = digest
    if not checksums:
        raise PrebuiltFallback(f"{label} carried no usable sha256 entries")
    return checksums


def fetch_release_checksums(bundle: ReleaseBundle) -> dict[str, str]:
    """Download + parse the release's ``whisper-prebuilt-sha256.json``. Fails
    closed (source build) if the release does not publish it."""
    url = bundle.asset_urls.get(SHA256_ASSET_NAME)
    if not url:
        raise PrebuiltFallback(
            f"release {bundle.repo}@{bundle.release_tag} has no {SHA256_ASSET_NAME}; "
            f"cannot verify a download"
        )
    try:
        raw = download_bytes(url, timeout = 30, headers = auth_headers(url))
        payload = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(
            f"could not fetch {SHA256_ASSET_NAME} from {bundle.repo}@{bundle.release_tag}: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltFallback(
            f"{SHA256_ASSET_NAME} in {bundle.repo}@{bundle.release_tag} was not valid JSON"
        ) from exc
    return parse_release_checksums(bundle.repo, bundle.release_tag, payload)


def expected_sha256_for(
    checksums: dict[str, str],
    asset_name: str,
    *,
    manifest_sha256: str | None = None,
) -> str:
    """The sha256 the archive must match: the release checksum-index entry for
    this asset. An asset absent from the index fails closed. If the manifest also
    embeds a sha256 for the asset it must agree with the index (a mismatch means a
    tampered manifest)."""
    digest = checksums.get(asset_name)
    if digest is None:
        raise PrebuiltFallback(
            f"{asset_name} is not covered by {SHA256_ASSET_NAME}; refusing an unverifiable download"
        )
    embedded = _valid_sha256(manifest_sha256)
    if embedded is not None and embedded != digest:
        raise PrebuiltFallback(
            f"manifest sha256 for {asset_name} disagrees with {SHA256_ASSET_NAME}; "
            f"refusing a possibly tampered release"
        )
    log(f"verifying {asset_name} against {SHA256_ASSET_NAME} sha256={digest}")
    return digest


# ── Verified download (retries once on checksum mismatch) ──
def download_file_verified(
    url: str, destination: Path, *, expected_sha256: str, label: str
) -> None:
    # Kept local (llama's variant calls its own download_file): resolving the
    # module-global download_file keeps the test seam monkeypatchable.
    for attempt in range(1, 3):
        download_file(url, destination)
        actual = sha256_file(destination)
        if actual == expected_sha256:
            log(f"verified {label} sha256={actual}")
            return
        log(f"{label} checksum mismatch {attempt}/2: expected={expected_sha256} actual={actual}")
        destination.unlink(missing_ok = True)
        if attempt == 2:
            raise PrebuiltFallback(f"{label} checksum mismatch after retry")


# ── GitHub release resolution ──
def release_asset_map(release: dict[str, Any]) -> dict[str, str]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        return {}
    return {
        asset["name"]: asset.get("browser_download_url", "")
        for asset in assets
        if isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and isinstance(asset.get("browser_download_url"), str)
    }


def github_release(repo: str, tag: str) -> dict[str, Any]:
    payload = fetch_json(
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe = '')}"
    )
    if not isinstance(payload, dict):
        raise PrebuiltFallback(f"unexpected release payload for {repo}@{tag}")
    return payload


@dataclass
class ReleaseBundle:
    repo: str
    release_tag: str
    manifest: dict[str, Any]
    asset_urls: dict[str, str]


def fetch_release_bundle(repo: str, release_tag: str) -> ReleaseBundle:
    """Fetch a fork release, download+validate its manifest asset, return the bundle.

    This is the single network seam for release resolution; tests inject a fake to
    exercise selection/install offline. A release that is missing or unreachable
    (404, rate limit, no network) is surfaced as a PrebuiltFallback so the caller
    falls back to a source build rather than erroring out.
    """
    try:
        release = github_release(repo, release_tag)
    except PrebuiltFallback:
        raise
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(f"could not fetch release {repo}@{release_tag}: {exc}") from exc
    resolved_tag = release.get("tag_name")
    resolved_tag = resolved_tag if isinstance(resolved_tag, str) and resolved_tag else release_tag
    asset_urls = release_asset_map(release)
    manifest_url = asset_urls.get(MANIFEST_ASSET_NAME)
    if not manifest_url:
        raise PrebuiltFallback(
            f"release {repo}@{resolved_tag} has no {MANIFEST_ASSET_NAME}; cannot select a prebuilt"
        )
    try:
        manifest_bytes = download_bytes(
            manifest_url, timeout = 30, headers = auth_headers(manifest_url)
        )
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(
            f"could not fetch {MANIFEST_ASSET_NAME} from {repo}@{resolved_tag}: {exc}"
        ) from exc
    try:
        manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltFallback(
            f"{MANIFEST_ASSET_NAME} in {repo}@{resolved_tag} was not valid JSON"
        ) from exc
    manifest = parse_manifest(
        manifest_payload, label = f"{MANIFEST_ASSET_NAME} in {repo}@{resolved_tag}"
    )
    return ReleaseBundle(
        repo = repo, release_tag = resolved_tag, manifest = manifest, asset_urls = asset_urls
    )


def release_asset_download_url(repo: str, release_tag: str, asset_name: str) -> str:
    """Deterministic download-host URL for a release asset. This is github.com
    (a redirect to the CDN), NOT api.github.com, so it is not subject to the
    unauthenticated 60-req/hour API rate limit."""
    return (
        f"https://github.com/{urllib.parse.quote(repo, safe = '/')}/releases/download/"
        f"{urllib.parse.quote(release_tag, safe = '')}/"
        f"{urllib.parse.quote(asset_name, safe = '')}"
    )


def asset_download_url(bundle: ReleaseBundle, asset_name: str) -> str:
    url = bundle.asset_urls.get(asset_name)
    if url:
        return url
    # A manifest can list an asset the release JSON did not surface; fall back to
    # the deterministic release-download URL.
    return release_asset_download_url(bundle.repo, bundle.release_tag, asset_name)


# ── Download-host fast path (resolve + fetch the JSON assets with no GitHub API) ──
# api.github.com is rate-limited to 60 req/hour unauthenticated; the download host
# (github.com/releases/...) is not. Mirroring install_llama_prebuilt.py, resolve the
# release and fetch its manifest + checksum index straight from the download host,
# falling back to the API only on a 404 / malformed asset / tag mismatch.
_DOWNLOAD_HOST_UA = {"User-Agent": "unsloth-studio-whisper-prebuilt"}


def _download_host_latest_release_tag(repo: str) -> str | None:
    """Latest release tag from github.com/<repo>/releases/latest via its redirect
    target (no api.github.com call). None on 404 so the caller falls back to the
    API. Note: /releases/latest resolves by created_at/make_latest, which can lag
    the published_at newest the freshness check uses -- acceptable for install."""
    url = f"https://github.com/{urllib.parse.quote(repo, safe = '/')}/releases/latest"
    request = urllib.request.Request(url, method = "HEAD", headers = dict(_DOWNLOAD_HOST_UA))
    try:
        with _URL_OPENER.open(request, timeout = 30) as response:
            final_url = response.geturl()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    marker = "/releases/tag/"
    index = final_url.find(marker)
    if index == -1:
        return None
    tag = urllib.parse.unquote(final_url[index + len(marker) :]).strip("/")
    return tag or None


def _download_host_json(url: str) -> Any:
    """Plain unauthenticated GET of a public release JSON asset from the CDN."""
    data = download_bytes(url, timeout = 30, headers = dict(_DOWNLOAD_HOST_UA))
    return json.loads(data.decode("utf-8"))


def _resolve_release_via_download_host(
    repo: str, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]] | None:
    """Resolve the release + manifest + checksum index entirely from the download
    host, with zero api.github.com calls. Returns None (caller falls back to the
    API) on a missing/renamed asset, a 404, or a tag mismatch. Fetches the checksum
    index first (an in-progress release can publish it before the manifest)."""
    release_tag = (published_release_tag or "").strip() or _download_host_latest_release_tag(repo)
    if not release_tag:
        return None
    sha_url = release_asset_download_url(repo, release_tag, SHA256_ASSET_NAME)
    try:
        sha_payload = _download_host_json(sha_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except (
        urllib.error.URLError,
        OSError,
        socket.timeout,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    try:
        checksums = parse_release_checksums(repo, release_tag, sha_payload)
    except PrebuiltFallback:
        return None  # schema/component/tag mismatch -> let the API path decide
    manifest_url = release_asset_download_url(repo, release_tag, MANIFEST_ASSET_NAME)
    try:
        manifest_payload = _download_host_json(manifest_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except (
        urllib.error.URLError,
        OSError,
        socket.timeout,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    try:
        manifest = parse_manifest(
            manifest_payload, label = f"{MANIFEST_ASSET_NAME} in {repo}@{release_tag}"
        )
    except PrebuiltFallback:
        return None
    # Tag-pinned CDN URLs for every asset the install may fetch; asset_download_url
    # also reconstructs any missing one, so this stays a pure download-host path.
    names = {
        str(a.get("asset"))
        for a in manifest.get("artifacts", [])
        if isinstance(a, dict) and a.get("asset")
    }
    names |= {MANIFEST_ASSET_NAME, SHA256_ASSET_NAME}
    asset_urls = {name: release_asset_download_url(repo, release_tag, name) for name in names}
    bundle = ReleaseBundle(
        repo = repo, release_tag = release_tag, manifest = manifest, asset_urls = asset_urls
    )
    return bundle, checksums


# ── Archive extraction (llama's guarded extractor + tar exec-bit restore) ──
def extract_archive(archive_path: Path, destination: Path) -> None:
    """llama's traversal/symlink-guarded extraction, then restore tar exec bits
    (llama writes plain files; whisper-server must stay executable)."""
    llama.extract_archive(archive_path, destination)
    if os.name == "nt" or not archive_path.name.endswith(".tar.gz"):
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not (member.isfile() and member.mode & 0o111):
                continue
            # Paths were already traversal-validated by the extractor above.
            target = destination / Path(member.name.replace("\\", "/"))
            if target.is_file():
                os.chmod(target, target.stat().st_mode | 0o111)


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


def _locate_server_in_tree(root: Path, host: HostInfo) -> Path:
    name = server_binary_name(host)
    matches = sorted(root.rglob(name))
    if not matches:
        raise PrebuiltFallback(f"archive did not contain a {name} binary")
    return matches[0]


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


def _swap_into_place(staged_root: Path, install_dir: Path) -> None:
    """Atomically replace install_dir with staged_root (same filesystem), with rollback."""
    install_dir.parent.mkdir(parents = True, exist_ok = True)
    backup: Path | None = None
    if install_dir.exists():
        backup = install_dir.parent / f".{install_dir.name}.old-{os.getpid()}"
        os.replace(install_dir, backup)
    try:
        os.replace(staged_root, install_dir)
    except OSError:
        if backup is not None and not install_dir.exists():
            os.replace(backup, install_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors = True)


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


# ── Metadata / marker ──
def metadata_path(install_dir: Path) -> Path:
    return install_dir / METADATA_FILENAME


def compute_install_fingerprint(
    *,
    published_repo: str,
    release_tag: str,
    upstream_tag: str | None,
    source_commit: str | None,
    asset: str,
    asset_sha256: str,
    backend: str,
    runtime_line: str | None,
    coverage: dict[str, Any],
) -> str:
    payload = {
        "published_repo": published_repo,
        "release_tag": release_tag,
        "upstream_tag": upstream_tag,
        "source_commit": source_commit,
        "asset": asset,
        "asset_sha256": asset_sha256,
        "backend": backend,
        "runtime_line": runtime_line,
        "coverage": coverage,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen = True)
class InstallSelection:
    """The identity of a chosen prebuilt: everything the marker/fingerprint record."""

    published_repo: str
    release_tag: str
    upstream_tag: str | None
    source_commit: str | None
    asset: str
    asset_sha256: str
    backend: str
    runtime_line: str | None
    coverage: dict[str, Any]
    studio_protocol: str | None

    def fingerprint(self) -> str:
        return compute_install_fingerprint(
            published_repo = self.published_repo,
            release_tag = self.release_tag,
            upstream_tag = self.upstream_tag,
            source_commit = self.source_commit,
            asset = self.asset,
            asset_sha256 = self.asset_sha256,
            backend = self.backend,
            runtime_line = self.runtime_line,
            coverage = self.coverage,
        )


def selection_from_artifact(
    *,
    published_repo: str,
    release_tag: str,
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    backend: str,
    asset_sha256: str,
) -> InstallSelection:
    return InstallSelection(
        published_repo = published_repo,
        release_tag = release_tag,
        upstream_tag = manifest.get("upstream_tag"),
        source_commit = manifest.get("source_commit"),
        asset = str(artifact.get("asset")),
        asset_sha256 = asset_sha256,
        backend = backend,
        runtime_line = artifact.get("runtime_line")
        if isinstance(artifact.get("runtime_line"), str)
        else None,
        coverage = artifact_coverage(artifact),
        studio_protocol = manifest.get("studio_protocol"),
    )


def write_prebuilt_metadata(install_dir: Path, selection: InstallSelection) -> None:
    coverage = selection.coverage
    payload = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "published_repo": selection.published_repo,
        "release_tag": selection.release_tag,
        "upstream_tag": selection.upstream_tag,
        "source_commit": selection.source_commit,
        "asset": selection.asset,
        "asset_sha256": selection.asset_sha256,
        "backend": selection.backend,
        "runtime_line": selection.runtime_line,
        "sm_coverage": coverage.get("sm_coverage") or coverage.get("sm"),
        "gfx_coverage": coverage.get("gfx_coverage") or coverage.get("gfx"),
        "min_os": coverage.get("min_os"),
        "studio_protocol": selection.studio_protocol,
        "install_fingerprint": selection.fingerprint(),
        "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path(install_dir).write_text(json.dumps(payload, indent = 2) + "\n")


def load_whisper_prebuilt_metadata(install_dir: Path) -> dict[str, Any] | None:
    path = metadata_path(install_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def existing_install_matches(
    install_dir: Path, host: HostInfo, selection: InstallSelection
) -> bool:
    """True iff the marker records this exact selection and the server binary is on disk."""
    metadata = load_whisper_prebuilt_metadata(install_dir)
    if metadata is None:
        return False
    if not whisper_server_path(install_dir, host).is_file():
        return False
    recorded = metadata.get("install_fingerprint")
    if not isinstance(recorded, str) or recorded != selection.fingerprint():
        return False
    log(
        f"existing whisper.cpp install already matches {selection.release_tag} "
        f"({selection.backend}); nothing to do"
    )
    return True


# ── Orchestration ──
def resolve_newest_release_tag(repo: str) -> str:
    """Newest published (non-draft/non-prerelease) release tag for `repo`, by
    ``published_at`` -- the same selection ``install_llama_prebuilt.py`` and the
    freshness check use, NOT GitHub's ``/releases/latest`` pointer (which sorts by
    commit date and can lag the newest build)."""
    payload = fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=30")
    if not isinstance(payload, list):
        raise PrebuiltFallback(f"unexpected releases payload for {repo}")
    published = [
        r
        for r in payload
        if isinstance(r, dict)
        and not r.get("draft")
        and not r.get("prerelease")
        and isinstance(r.get("tag_name"), str)
        and r.get("tag_name")
    ]
    if not published:
        raise PrebuiltFallback(f"{repo} has no published prebuilt release yet")
    newest = max(published, key = lambda r: r.get("published_at") or "")
    return newest["tag_name"]


def resolve_release_tag(published_repo: str, *, published_release_tag: str | None) -> str:
    """The release tag to install: an explicit override, else the newest published
    release resolved at runtime."""
    override = (published_release_tag or "").strip()
    if override:
        return override
    return resolve_newest_release_tag(published_repo)


def fetch_release_for_install(
    repo: str, *, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]]:
    """Resolve the release + manifest + checksum index, preferring the download
    host (no api.github.com rate limit) and falling back to the GitHub API. This is
    the single network seam the install/probe paths use."""
    fast = _resolve_release_via_download_host(repo, published_release_tag)
    if fast is not None:
        bundle, checksums = fast
        log(f"resolved {repo}@{bundle.release_tag} via the download host (no GitHub API)")
        return bundle, checksums
    release_tag = resolve_release_tag(repo, published_release_tag = published_release_tag)
    bundle = fetch_release_bundle(repo, release_tag)
    checksums = fetch_release_checksums(bundle)
    return bundle, checksums


def _install_from_bundle(
    install_dir: Path, host: HostInfo, bundle: ReleaseBundle, selection: InstallSelection
) -> None:
    staging_root = install_dir.parent / INSTALL_STAGING_ROOT_NAME
    staging_root.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = f"{install_dir.name}.staging-", dir = staging_root))
    try:
        archive_path = staging / selection.asset
        url = asset_download_url(bundle, selection.asset)
        log(f"downloading {url}")
        download_file_verified(
            url, archive_path, expected_sha256 = selection.asset_sha256, label = selection.asset
        )
        extract_dir = staging / "extracted"
        extract_archive(archive_path, extract_dir)

        server = _locate_server_in_tree(extract_dir, host)
        bundle_root = server.parent

        staged_root = staging / "staged"
        _assemble_install_tree(bundle_root, staged_root, host)
        _validate_staged_server(staged_root, host)
        write_prebuilt_metadata(staged_root, selection)
        _swap_into_place(staged_root, install_dir)
    finally:
        shutil.rmtree(staging, ignore_errors = True)
        try:
            staging_root.rmdir()
        except OSError:
            pass


def plan_selection(
    host: HostInfo,
    bundle: ReleaseBundle,
    *,
    published_repo: str,
    backend: str,
    checksums: dict[str, str],
) -> InstallSelection:
    """Choose an artifact (with CPU fallback) and resolve its trusted sha256 from
    the release checksum index."""
    artifact, effective_backend, _used_cpu = select_artifact_with_cpu_fallback(
        bundle.manifest, host, backend
    )
    asset = str(artifact.get("asset"))
    manifest_sha256 = artifact.get("sha256") if isinstance(artifact.get("sha256"), str) else None
    expected_sha = expected_sha256_for(
        checksums,
        asset,
        manifest_sha256 = manifest_sha256,
    )
    return selection_from_artifact(
        published_repo = published_repo,
        release_tag = bundle.release_tag,
        manifest = bundle.manifest,
        artifact = artifact,
        backend = effective_backend,
        asset_sha256 = expected_sha,
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
    effective_backend = resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    log(
        f"target whisper.cpp from {published_repo} "
        f"({host.whisper_os}-{host.whisper_arch}, backend {effective_backend})"
    )

    bundle, checksums = fetch_release_for_install(
        published_repo, published_release_tag = published_release_tag
    )
    release_tag = bundle.release_tag
    selection = plan_selection(
        host,
        bundle,
        published_repo = published_repo,
        backend = effective_backend,
        checksums = checksums,
    )

    if not force and existing_install_matches(install_dir, host, selection):
        return EXIT_SUCCESS

    with install_lock(install_lock_path(install_dir)):
        # Re-check under the lock: a concurrent run may have just finished.
        if not force and existing_install_matches(install_dir, host, selection):
            return EXIT_SUCCESS
        _install_from_bundle(install_dir, host, bundle, selection)

    server = whisper_server_path(install_dir, host)
    if not server.is_file():
        raise PrebuiltFallback(f"post-install verification failed: {server} is missing")
    log(f"installed whisper.cpp {release_tag} ({selection.backend}) at {install_dir}")
    return EXIT_SUCCESS


def resolve_prebuilt(
    host: HostInfo,
    *,
    published_repo: str,
    published_release_tag: str | None,
    backend: str | None,
    cpu_fallback: bool,
) -> dict[str, Any]:
    """Host-aware "is a prebuilt available" probe. No archive download."""
    effective_backend = resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    try:
        bundle, _checksums = fetch_release_for_install(
            published_repo, published_release_tag = published_release_tag
        )
        artifact, resolved_backend, used_cpu = select_artifact_with_cpu_fallback(
            bundle.manifest, host, effective_backend
        )
    except PrebuiltFallback:
        return {"prebuilt_available": False, "repo": published_repo}
    return {
        "prebuilt_available": True,
        "repo": published_repo,
        "release_tag": bundle.release_tag,
        "upstream_tag": bundle.manifest.get("upstream_tag"),
        "backend": resolved_backend,
        "requested_backend": effective_backend,
        "cpu_fallback": used_cpu,
        "asset": str(artifact.get("asset")),
        "os": host.whisper_os,
        "arch": host.whisper_arch,
        "runtime_line": artifact.get("runtime_line"),
    }


def emit_resolver_output(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, sort_keys = True))
        return
    if "asset" in payload and payload.get("prebuilt_available"):
        print(payload["asset"])
        return
    print(json.dumps(payload, sort_keys = True))


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
