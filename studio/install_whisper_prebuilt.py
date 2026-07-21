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
import platform
import random
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
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None


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

GITHUB_AUTH_HOSTS = {"api.github.com", "github.com"}
RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HTTP_FETCH_ATTEMPTS = 4
HTTP_FETCH_BASE_DELAY_SECONDS = 0.75
INSTALL_LOCK_TIMEOUT_SECONDS = 300
INSTALL_STAGING_ROOT_NAME = ".staging"

# Master switch for the staged runtime smoke test (start whisper-server + a tiny
# transcription) before the atomic activate. Disabled by default so a bundle whose
# GPU forward pass JIT-compiles kernels does not stall every install; the check and
# its CPU-asset retry are kept intact -- set this True to re-enable them.
_RUN_STAGED_PREBUILT_VALIDATION = False

# PowerShell renders stderr as NativeCommandError noise; main() flips logs to stdout.
_LOG_TO_STDOUT = False


class PrebuiltFallback(RuntimeError):
    """Recoverable failure -- caller should fall back to source build (exit code 2)."""


class BusyInstallConflict(RuntimeError):
    """Another process holds the install lock (exit code 3)."""


def log(message: str) -> None:
    print(f"[whisper-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr)


# ── Host detection ──
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


def _has_usable_nvidia() -> bool:
    """Best-effort: an nvidia-smi that lists at least one GPU. Never raises."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            timeout = 10,
            **_windows_hidden_kwargs(),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and any(line.strip() for line in result.stdout.splitlines())


def _detect_rocm_gfx() -> tuple[bool, str | None]:
    """Best-effort ROCm detection returning (has_rocm, gfx_target). Never raises."""
    # WSL2 ROCDXG: rocminfo enumerates the GPU over /dev/dxg only when
    # HSA_ENABLE_DXG_DETECTION=1 (a no-op on bare metal), and can live only under
    # /opt/rocm/bin (its profile.d PATH drop-in reaches login shells only). Probe
    # accordingly or a WSL ROCm host is misdetected as CPU-only. Mirrors
    # install_llama_prebuilt.py's WSL detection.
    dxg_env = {**os.environ}
    dxg_env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    for tool, args in (
        ("rocminfo", []),
        ("amd-smi", ["static", "--asic"]),
    ):
        binary = shutil.which(tool)
        if not binary and tool == "rocminfo" and os.access("/opt/rocm/bin/rocminfo", os.X_OK):
            binary = "/opt/rocm/bin/rocminfo"
        if not binary:
            continue
        try:
            result = subprocess.run(
                [binary, *args],
                capture_output = True,
                text = True,
                timeout = 10,
                env = dxg_env if tool == "rocminfo" else None,
                **_windows_hidden_kwargs(),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            token = line.strip().lower()
            if "gfx" not in token:
                continue
            for word in token.replace(":", " ").replace("\t", " ").split():
                # Real GPU ISA id: gfx<nonzero><alnum...> (gfx908, gfx90a, gfx1100).
                # Skip the CPU agent (gfx000) and generic ISA lines (gfx11-generic).
                if (
                    word.startswith("gfx")
                    and len(word) > 3
                    and word[3] != "0"
                    and word[3:].isalnum()
                ):
                    return True, word
    return False, None


def detect_host() -> HostInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"
    is_macos = system == "Darwin"

    if system == "Linux":
        whisper_os = "linux"
    elif is_macos:
        whisper_os = "macos"
    elif is_windows:
        whisper_os = "windows"
    else:
        raise PrebuiltFallback(f"unsupported operating system for whisper.cpp prebuilt: {system}")

    if machine in {"x86_64", "amd64", "x64"}:
        whisper_arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        whisper_arch = "arm64"
    else:
        raise PrebuiltFallback(f"unsupported CPU architecture for whisper.cpp prebuilt: {machine}")

    is_apple_silicon = is_macos and whisper_arch == "arm64"
    archive_ext = ".zip" if is_windows else ".tar.gz"

    has_usable_nvidia = False
    has_rocm = False
    rocm_gfx: str | None = None
    if system == "Linux" or is_windows:
        # macOS has no CUDA/ROCm; skip the probes there.
        has_usable_nvidia = _has_usable_nvidia()
        if not has_usable_nvidia:
            has_rocm, rocm_gfx = _detect_rocm_gfx()

    return HostInfo(
        system = system,
        machine = machine,
        whisper_os = whisper_os,
        whisper_arch = whisper_arch,
        archive_ext = archive_ext,
        is_windows = is_windows,
        is_macos = is_macos,
        is_apple_silicon = is_apple_silicon,
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
        rocm_gfx = rocm_gfx,
    )


def apply_host_overrides(
    host: HostInfo,
    *,
    has_rocm: bool = False,
    rocm_gfx: str | None = None,
    force_cpu: bool = False,
) -> HostInfo:
    """Apply CLI overrides (setup forwards hardware hints) onto a detected host."""
    from dataclasses import replace

    if force_cpu:
        return replace(
            host,
            has_usable_nvidia = False,
            has_rocm = False,
            rocm_gfx = None,
            is_apple_silicon = False,
        )
    updates: dict[str, Any] = {}
    if has_rocm:
        updates["has_rocm"] = True
        updates["has_usable_nvidia"] = False
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


def select_artifact(
    manifest: dict[str, Any], host: HostInfo, backend: str
) -> dict[str, Any] | None:
    """First manifest artifact matching this host os/arch and the given backend."""
    for artifact in manifest.get("artifacts", []):
        if (
            artifact.get("os") == host.whisper_os
            and artifact.get("arch") == host.whisper_arch
            and artifact.get("backend") == backend
        ):
            return artifact
    return None


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


# ── HTTP (retry/backoff, token-safe cross-host redirects) ──
def parsed_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:  # noqa: BLE001
        return None
    return hostname.lower() if hostname else None


def should_send_github_auth(url: str | None) -> bool:
    return parsed_hostname(url) in GITHUB_AUTH_HOSTS


def auth_headers(url: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": "unsloth-studio-whisper-prebuilt"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and should_send_github_auth(url):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_api_headers(url: str | None = None) -> dict[str, str]:
    return {"Accept": "application/vnd.github+json", **auth_headers(url)}


def is_github_api_url(url: str | None) -> bool:
    return parsed_hostname(url) == "api.github.com"


class _CrossHostAuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host.

    GitHub redirects release-asset downloads to CDN hosts whose signed URLs can
    reject a foreign Authorization header; urllib forwards headers to redirect
    targets by default (requests/huggingface_hub strip them).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None and parsed_hostname(newurl) != parsed_hostname(req.full_url):
            new_request.headers.pop("Authorization", None)
            new_request.unredirected_hdrs.pop("Authorization", None)
        return new_request


_URL_OPENER = urllib.request.build_opener(_CrossHostAuthStrippingRedirectHandler())


def is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        # GitHub returns 403 (not 429) when the API rate limit is hit; treat that
        # against api.github.com as retryable so a CI fleet gets a backoff cycle
        # before the source-build fallback fires. Other-host 403s stay fatal.
        if exc.code == 403:
            return is_github_api_url(getattr(exc, "url", None))
        return exc.code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    return False


def sleep_backoff(attempt: int) -> None:
    delay = HTTP_FETCH_BASE_DELAY_SECONDS * (2 ** max(attempt - 1, 0))
    delay += random.uniform(0.0, 0.2)
    time.sleep(delay)


def download_bytes(
    url: str,
    *,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers = headers or auth_headers(url))
            with _URL_OPENER.open(request, timeout = timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(f"fetch failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying")
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str) -> Any:
    headers = github_api_headers(url) if is_github_api_url(url) else auth_headers(url)
    data = download_bytes(url, timeout = 30, headers = headers)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise PrebuiltFallback(f"unexpected JSON type from {url}: {type(payload).__name__}")
    return payload


def atomic_replace_from_tempfile(tmp_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    os.replace(tmp_path, destination)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_FETCH_ATTEMPTS + 1):
        tmp_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers = auth_headers(url))
            with tempfile.NamedTemporaryFile(
                prefix = destination.name + ".tmp-",
                dir = destination.parent,
                delete = False,
            ) as handle:
                tmp_path = Path(handle.name)
                with _URL_OPENER.open(request, timeout = 120) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file from {url}")
            atomic_replace_from_tempfile(tmp_path, destination)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok = True)
                except Exception:  # noqa: BLE001
                    pass
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(f"download failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying")
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file_verified(
    url: str, destination: Path, *, expected_sha256: str, label: str
) -> None:
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


# ── Safe archive extraction (zip + tar.gz, traversal/symlink guarded) ──
def _safe_extract_path(base: Path, member_name: str) -> Path:
    member_path = Path(member_name.replace("\\", "/"))
    if member_path.is_absolute():
        raise PrebuiltFallback(f"archive member used an absolute path: {member_name}")
    target = (base / member_path).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise PrebuiltFallback(f"archive member escaped destination: {member_name}") from exc
    return target


def _extract_zip_safely(source: Path, base: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = _safe_extract_path(base, member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise PrebuiltFallback(f"zip archive contained a symlink entry: {member.filename}")
            if member.is_dir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            perm = (member.external_attr >> 16) & 0o777
            if perm & 0o111:
                os.chmod(target, target.stat().st_mode | 0o111)


def _extract_tar_safely(source: Path, base: Path) -> None:
    # whisper.cpp bundles co-locate the server with its shared libs; some GPU
    # backends ship those as versioned symlinks (libwhisper.so -> libwhisper.so.1),
    # so defer links and resolve them after the regular files.
    pending_links: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            target = _safe_extract_path(base, member.name)
            if member.isdir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            if member.islnk() or member.issym():
                pending_links.append((member, target))
                continue
            if not member.isfile():
                raise PrebuiltFallback(f"tar archive contained an unsupported entry: {member.name}")
            target.parent.mkdir(parents = True, exist_ok = True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise PrebuiltFallback(f"tar archive entry could not be read: {member.name}")
            with extracted, target.open("wb") as dst:
                shutil.copyfileobj(extracted, dst)
            if member.mode & 0o111:
                os.chmod(target, target.stat().st_mode | 0o111)

    for member, target in pending_links:
        link_name = member.linkname.replace("\\", "/")
        link_path = Path(link_name)
        if link_path.is_absolute() or not link_name:
            raise PrebuiltFallback(
                f"archive link used an unsafe target: {member.name} -> {link_name}"
            )
        # tar symlink names are link-parent relative; hard-link names are archive-root relative.
        resolved = (target.parent / link_path if member.issym() else base / link_path).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise PrebuiltFallback(
                f"archive link escaped destination: {member.name} -> {link_name}"
            ) from exc
        target.parent.mkdir(parents = True, exist_ok = True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if member.issym():
            target.symlink_to(link_name)
        else:  # hard link
            shutil.copy2(resolved, target)


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    if archive_path.name.endswith(".zip"):
        _extract_zip_safely(archive_path, destination)
    elif archive_path.name.endswith(".tar.gz"):
        _extract_tar_safely(archive_path, destination)
    else:
        raise PrebuiltFallback(f"unsupported archive format: {archive_path.name}")


# ── Install lock (concurrent setup runs share one UNSLOTH_HOME) ──
def _windows_hidden_kwargs() -> dict[str, object]:
    if sys.platform != "win32":
        return {}
    kwargs: dict[str, object] = {}
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if flag:
        kwargs["creationflags"] = flag
    return kwargs


def install_lock_path(install_dir: Path) -> Path:
    return install_dir.parent / f".{install_dir.name}.install.lock"


def _pid_is_alive(pid: int) -> bool:
    """Best-effort process liveness check that never signals the process on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output = True,
                text = True,
                timeout = 5,
                **_windows_hidden_kwargs(),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return True
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError:
        return False
    return True


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents = True, exist_ok = True)
    if FileLock is None:
        fd: int | None = None
        deadline = time.monotonic() + INSTALL_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.fsync(fd)
                break
            except FileExistsError:
                try:
                    raw = lock_path.read_text().strip()
                except FileNotFoundError:
                    continue
                stale = False
                if raw:
                    try:
                        stale = not _pid_is_alive(int(raw))
                    except ValueError:
                        stale = True
                if stale:
                    # Atomically rename before unlinking so only one racer removes
                    # the stale lock; a process recreating it loses the rename and waits.
                    try:
                        stale_path = lock_path.with_name(f"{lock_path.name}.stale.{os.getpid()}")
                        os.replace(str(lock_path), str(stale_path))
                        stale_path.unlink(missing_ok = True)
                    except (OSError, ValueError):
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise BusyInstallConflict(
                        f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
                    )
                time.sleep(0.5)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            lock_path.unlink(missing_ok = True)
        return

    try:
        with FileLock(str(lock_path), timeout = INSTALL_LOCK_TIMEOUT_SECONDS):
            yield
    except FileLockTimeout as exc:
        raise BusyInstallConflict(
            f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
        ) from exc


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
            **_windows_hidden_kwargs(),
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
    _LOG_TO_STDOUT = True

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.resolve_prebuilt is not None:
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
        emit_resolver_output(payload, output_format = args.output_format)
        return EXIT_SUCCESS

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
