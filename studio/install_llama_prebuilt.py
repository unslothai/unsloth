#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross platform llama.cpp prebuilt installer for Unsloth Studio"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import random
import shutil
import site
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None
from pathlib import Path
from typing import Any, Iterable, Iterator


EXIT_SUCCESS = 0
EXIT_FALLBACK = 2
EXIT_ERROR = 1

APPROVED_PREBUILT_LLAMA_TAG = "b8508"
DEFAULT_LLAMA_TAG = os.environ.get("UNSLOTH_LLAMA_TAG", APPROVED_PREBUILT_LLAMA_TAG)
DEFAULT_PUBLISHED_REPO = os.environ.get(
    "UNSLOTH_LLAMA_RELEASE_REPO", "unslothai/llama.cpp"
)
DEFAULT_PUBLISHED_TAG = os.environ.get("UNSLOTH_LLAMA_RELEASE_TAG")
DEFAULT_PUBLISHED_MANIFEST_ASSET = os.environ.get(
    "UNSLOTH_LLAMA_RELEASE_MANIFEST_ASSET", "llama-prebuilt-manifest.json"
)
DEFAULT_PUBLISHED_SHA256_ASSET = os.environ.get(
    "UNSLOTH_LLAMA_RELEASE_SHA256_ASSET", "llama-prebuilt-sha256.json"
)
UPSTREAM_REPO = "ggml-org/llama.cpp"
UPSTREAM_RELEASES_API = f"https://api.github.com/repos/{UPSTREAM_REPO}/releases/latest"
TEST_MODEL_URL = (
    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
)
TEST_MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
VALIDATION_MODEL_CACHE_DIRNAME = ".cache"
VALIDATION_MODEL_CACHE_FILENAME = "stories260K.gguf"
INSTALL_LOCK_TIMEOUT_SECONDS = 300
INSTALL_STAGING_ROOT_NAME = ".staging"
GITHUB_AUTH_HOSTS = {"api.github.com", "github.com"}
RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HTTP_FETCH_ATTEMPTS = 4
HTTP_FETCH_BASE_DELAY_SECONDS = 0.75
SERVER_PORT_BIND_ATTEMPTS = 3
SERVER_BIND_RETRY_WINDOW_SECONDS = 5.0
TTY_PROGRESS_START_DELAY_SECONDS = 0.5


@dataclass
class HostInfo:
    system: str
    machine: str
    is_windows: bool
    is_linux: bool
    is_macos: bool
    is_x86_64: bool
    is_arm64: bool
    nvidia_smi: str | None
    driver_cuda_version: tuple[int, int] | None
    compute_caps: list[str]
    visible_cuda_devices: str | None
    has_physical_nvidia: bool
    has_usable_nvidia: bool
    has_rocm: bool = False


@dataclass
class AssetChoice:
    repo: str
    tag: str
    name: str
    url: str
    source_label: str
    runtime_name: str | None = None
    runtime_url: str | None = None
    is_ready_bundle: bool = False
    install_kind: str = ""
    bundle_profile: str | None = None
    runtime_line: str | None = None
    coverage_class: str | None = None
    supported_sms: list[str] | None = None
    min_sm: int | None = None
    max_sm: int | None = None
    selection_log: list[str] | None = None
    expected_sha256: str | None = None


@dataclass(frozen = True)
class PublishedLlamaArtifact:
    asset_name: str
    install_kind: str
    runtime_line: str | None
    coverage_class: str | None
    supported_sms: list[str]
    min_sm: int | None
    max_sm: int | None
    bundle_profile: str | None
    rank: int


@dataclass
class PublishedReleaseBundle:
    repo: str
    release_tag: str
    upstream_tag: str
    assets: dict[str, str]
    manifest_asset_name: str
    artifacts: list[PublishedLlamaArtifact]
    selection_log: list[str]


@dataclass
class LinuxCudaSelection:
    attempts: list[AssetChoice]
    selection_log: list[str]

    @property
    def primary(self) -> AssetChoice:
        if not self.attempts:
            raise RuntimeError("linux CUDA selection unexpectedly had no attempts")
        return self.attempts[0]


@dataclass
class CudaRuntimePreference:
    runtime_line: str | None
    selection_log: list[str]


@dataclass(frozen = True)
class ApprovedArtifactHash:
    asset_name: str
    sha256: str
    repo: str | None
    kind: str | None


@dataclass
class ApprovedReleaseChecksums:
    repo: str
    release_tag: str
    upstream_tag: str
    source_commit: str | None
    artifacts: dict[str, ApprovedArtifactHash]


class PrebuiltFallback(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[llama-prebuilt] {message}")


def log_lines(lines: Iterable[str]) -> None:
    for line in lines:
        log(line)


def parsed_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:
        return None
    if not hostname:
        return None
    return hostname.lower()


def should_send_github_auth(url: str | None) -> bool:
    return parsed_hostname(url) in GITHUB_AUTH_HOSTS


def auth_headers(url: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": "unsloth-studio-llama-prebuilt",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and should_send_github_auth(url):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_api_headers(url: str | None = None) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        **auth_headers(url),
    }


def is_github_api_url(url: str | None) -> bool:
    return parsed_hostname(url) == "api.github.com"


def is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    return False


def sleep_backoff(
    attempt: int, *, base_delay: float = HTTP_FETCH_BASE_DELAY_SECONDS
) -> None:
    delay = base_delay * (2 ** max(attempt - 1, 0))
    delay += random.uniform(0.0, 0.2)
    time.sleep(delay)


def atomic_write_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    with tempfile.NamedTemporaryFile(
        prefix = destination.name + ".tmp-",
        dir = destination.parent,
        delete = False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, destination)


def atomic_replace_from_tempfile(tmp_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    os.replace(tmp_path, destination)


def source_archive_logical_name(upstream_tag: str) -> str:
    return f"llama.cpp-source-{upstream_tag}.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256_digest(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    lowered = value.lower()
    if lowered.startswith("sha256:"):
        lowered = lowered.split(":", 1)[1]
    if len(lowered) != 64 or any(ch not in "0123456789abcdef" for ch in lowered):
        return None
    return lowered


def format_byte_count(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes:.1f} B"


class DownloadProgress:
    def __init__(self, label: str, total_bytes: int | None) -> None:
        self.label = label
        self.total_bytes = total_bytes if total_bytes and total_bytes > 0 else None
        self.start_time = time.monotonic()
        self.last_emit = 0.0
        term_ok = os.environ.get("TERM", "").lower() != "dumb"
        self.stream = (
            sys.stderr
            if sys.stderr.isatty()
            else sys.stdout
            if sys.stdout.isatty()
            else sys.stderr
        )
        self.is_tty = term_ok and self.stream.isatty()
        self.completed = False
        self.last_milestone_percent = -1
        self.last_milestone_bytes = 0
        self.has_rendered_tty_progress = False

    def _render(self, downloaded_bytes: int, *, final: bool = False) -> str:
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        speed = downloaded_bytes / elapsed
        speed_text = f"{format_byte_count(speed)}/s"
        if self.total_bytes is not None:
            percent = min(100.0, (downloaded_bytes / self.total_bytes) * 100.0)
            return (
                f"{self.label}: {percent:5.1f}% "
                f"({format_byte_count(downloaded_bytes)}/{format_byte_count(self.total_bytes)}) "
                f"at {speed_text}"
            )
        if final:
            return f"{self.label}: {format_byte_count(downloaded_bytes)} downloaded at {speed_text}"
        return f"{self.label}: {format_byte_count(downloaded_bytes)} downloaded at {speed_text}"

    def update(self, downloaded_bytes: int) -> None:
        now = time.monotonic()
        if self.is_tty:
            elapsed = now - self.start_time
            if not self.has_rendered_tty_progress:
                if (
                    self.total_bytes is not None
                    and downloaded_bytes >= self.total_bytes
                ):
                    return
                if elapsed < TTY_PROGRESS_START_DELAY_SECONDS:
                    return
            min_interval = 0.2
            if (
                self.has_rendered_tty_progress
                and not self.completed
                and (now - self.last_emit) < min_interval
            ):
                return
            self.last_emit = now
            line = self._render(downloaded_bytes)
            self.stream.write("\r\033[K" + line)
            self.stream.flush()
            self.has_rendered_tty_progress = True
            return

        should_emit = False
        if self.total_bytes is not None:
            percent = int((downloaded_bytes * 100) / max(self.total_bytes, 1))
            milestone_percent = min((percent // 25) * 25, 100)
            if (
                milestone_percent > self.last_milestone_percent
                and milestone_percent < 100
            ):
                self.last_milestone_percent = milestone_percent
                should_emit = True
        else:
            byte_step = 25 * 1024 * 1024
            if (
                downloaded_bytes - self.last_milestone_bytes >= byte_step
                and (now - self.last_emit) >= 5.0
            ):
                self.last_milestone_bytes = downloaded_bytes
                should_emit = True

        if not should_emit:
            return

        self.last_emit = now
        self.stream.write(self._render(downloaded_bytes) + "\n")
        self.stream.flush()

    def finish(self, downloaded_bytes: int) -> None:
        self.completed = True
        line = self._render(downloaded_bytes, final = True)
        if self.is_tty:
            if not self.has_rendered_tty_progress:
                return
            self.stream.write("\r\033[K")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


def download_label_from_url(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    return name or url


def download_bytes(
    url: str,
    *,
    timeout: int = 120,
    attempts: int = HTTP_FETCH_ATTEMPTS,
    headers: dict[str, str] | None = None,
    progress_label: str | None = None,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers = headers or auth_headers(url))
            with urllib.request.urlopen(request, timeout = timeout) as response:
                total_bytes: int | None = None
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total_bytes = int(content_length)
                progress = (
                    DownloadProgress(progress_label, total_bytes)
                    if progress_label
                    else None
                )
                data = bytearray()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if progress is not None:
                        progress.update(len(data))
                if progress is not None:
                    progress.finish(len(data))
                return bytes(data)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not is_retryable_url_error(exc):
                raise
            log(f"fetch failed ({attempt}/{attempts}) for {url}: {exc}; retrying")
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str) -> Any:
    data = download_bytes(
        url,
        timeout = 30,
        headers = github_api_headers(url)
        if is_github_api_url(url)
        else auth_headers(url),
    )
    if not data:
        raise RuntimeError(f"downloaded empty JSON payload from {url}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"downloaded invalid JSON from {url}: {exc}") from exc
    if not isinstance(payload, dict) and not isinstance(payload, list):
        raise RuntimeError(
            f"downloaded unexpected JSON type from {url}: {type(payload).__name__}"
        )
    return payload


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
                with urllib.request.urlopen(request, timeout = 120) as response:
                    total_bytes: int | None = None
                    content_length = response.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        total_bytes = int(content_length)
                    progress = DownloadProgress(
                        f"Downloading {destination.name}", total_bytes
                    )
                    downloaded_bytes = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded_bytes += len(chunk)
                        progress.update(downloaded_bytes)
                    progress.finish(downloaded_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file from {url}")
            atomic_replace_from_tempfile(tmp_path, destination)
            return
        except Exception as exc:
            last_exc = exc
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok = True)
                except Exception:
                    pass
            if attempt >= HTTP_FETCH_ATTEMPTS or not is_retryable_url_error(exc):
                raise
            log(
                f"download failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying"
            )
            sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def download_file_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    normalized_expected = normalize_sha256_digest(expected_sha256)
    if not normalized_expected:
        raise PrebuiltFallback(f"{label} did not have a valid approved sha256")

    for attempt in range(1, 3):
        download_file(url, destination)
        actual_sha256 = sha256_file(destination)
        if actual_sha256 == normalized_expected:
            log(f"verified {label} sha256={actual_sha256}")
            return

        log(
            f"{label} checksum mismatch on attempt {attempt}/2: "
            f"expected={normalized_expected} actual={actual_sha256}"
        )
        destination.unlink(missing_ok = True)
        if attempt == 2:
            raise PrebuiltFallback(
                f"{label} checksum mismatch after retry: expected={normalized_expected} actual={actual_sha256}"
            )
        log(f"retrying {label} download after checksum mismatch")


def upstream_source_archive_urls(tag: str) -> list[str]:
    encoded_tag = urllib.parse.quote(tag, safe = "")
    return [
        f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/refs/tags/{encoded_tag}",
        f"https://github.com/{UPSTREAM_REPO}/archive/refs/tags/{encoded_tag}.tar.gz",
    ]


def github_release_assets(repo: str, tag: str) -> dict[str, str]:
    payload = fetch_json(
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe = '')}"
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected release payload for {repo}@{tag}")
    return release_asset_map(payload)


def github_release(repo: str, tag: str) -> dict[str, Any]:
    payload = fetch_json(
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe = '')}"
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected release payload for {repo}@{tag}")
    return payload


def github_releases(repo: str, *, per_page: int = 100) -> list[dict[str, Any]]:
    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = fetch_json(
            f"https://api.github.com/repos/{repo}/releases?per_page={per_page}&page={page}"
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected releases payload for {repo}")
        page_items = [item for item in payload if isinstance(item, dict)]
        releases.extend(page_items)
        if len(payload) < per_page:
            break
        page += 1
    return releases


def latest_upstream_release_tag() -> str:
    payload = fetch_json(UPSTREAM_RELEASES_API)
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise RuntimeError(
            f"latest release tag was missing from {UPSTREAM_RELEASES_API}"
        )
    return tag


def normalize_compute_cap(value: Any) -> str | None:
    raw = str(value).strip()
    if not raw:
        return None
    if "." in raw:
        parts = raw.split(".", 1)
        if len(parts) != 2:
            return None
        major, minor = parts
        if not major.isdigit() or not minor.isdigit():
            return None
        return f"{int(major)}{int(minor)}"
    if raw.isdigit():
        return str(int(raw))
    return None


def normalize_compute_caps(compute_caps: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in compute_caps:
        normalized_value = normalize_compute_cap(raw)
        if normalized_value is None:
            continue
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized.append(normalized_value)
    normalized.sort(key = int)
    return normalized


def parse_cuda_visible_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw or raw == "-1":
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def supports_explicit_visible_device_matching(
    visible_devices: list[str] | None,
) -> bool:
    if not visible_devices:
        return False
    for token in visible_devices:
        lowered = token.lower()
        if token.isdigit() or lowered.startswith("gpu-"):
            continue
        return False
    return True


def select_visible_gpu_rows(
    gpu_rows: Iterable[tuple[str, str, str]],
    visible_devices: list[str] | None,
) -> list[tuple[str, str, str]]:
    rows = list(gpu_rows)
    if visible_devices is None:
        return rows
    if not visible_devices:
        return []

    by_index = {index: (index, uuid, cap) for index, uuid, cap in rows}
    by_uuid = {uuid.lower(): (index, uuid, cap) for index, uuid, cap in rows}
    selected: list[tuple[str, str, str]] = []
    seen_indices: set[str] = set()
    for token in visible_devices:
        row = by_index.get(token)
        if row is None:
            normalized_token = token.lower()
            row = by_uuid.get(normalized_token)
            if row is None and normalized_token.startswith("gpu-"):
                row = by_uuid.get(normalized_token)
            if row is None and not normalized_token.startswith("gpu-"):
                row = by_uuid.get("gpu-" + normalized_token)
        if row is None:
            continue
        index = row[0]
        if index in seen_indices:
            continue
        seen_indices.add(index)
        selected.append(row)
    return selected


def dir_provides_exact_library(directory: str | Path, library: str) -> bool:
    if not library:
        return False
    candidate = Path(directory) / library
    return candidate.exists() and (candidate.is_file() or candidate.is_symlink())


def linux_runtime_dirs_for_required_libraries(
    required_libraries: Iterable[str],
) -> list[str]:
    required = [library for library in required_libraries if library]
    candidates: list[str | Path] = []

    env_dirs = os.environ.get("CUDA_RUNTIME_LIB_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        candidates.extend(part for part in ld_library_path.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))
    cuda_roots.extend(
        Path(path) for path in glob_paths("/usr/local/cuda", "/usr/local/cuda-*")
    )

    for root in cuda_roots:
        candidates.extend(
            [
                root / "lib",
                root / "lib64",
                root / "targets" / "x86_64-linux" / "lib",
            ]
        )

    candidates.extend(
        Path(path)
        for path in glob_paths(
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        )
    )
    candidates.extend(
        Path(path)
        for path in glob_paths("/usr/local/lib/ollama/cuda_v*", "/usr/lib/wsl/lib")
    )
    candidates.extend(Path(path) for path in python_runtime_dirs())
    candidates.extend(Path(path) for path in ldconfig_runtime_dirs(required))

    resolved = dedupe_existing_dirs(candidates)
    if not required:
        return resolved

    matched: list[tuple[int, str]] = []
    for directory in resolved:
        base = Path(directory)
        provided = sum(
            1 for library in required if dir_provides_exact_library(directory, library)
        )
        if provided:
            matched.append((provided, directory))

    matched.sort(key = lambda item: item[0], reverse = True)
    return [directory for _, directory in matched]


def detected_linux_runtime_lines() -> tuple[list[str], dict[str, list[str]]]:
    line_requirements = {
        "cuda13": ["libcudart.so.13", "libcublas.so.13"],
        "cuda12": ["libcudart.so.12", "libcublas.so.12"],
    }
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for line, required in line_requirements.items():
        dirs = linux_runtime_dirs_for_required_libraries(required)
        library_matches: dict[str, list[str]] = {}
        matching_dirs: list[str] = []
        for library in required:
            matched_dirs = [
                directory
                for directory in dirs
                if any(Path(directory).glob(f"{library}*"))
            ]
            if not matched_dirs:
                library_matches = {}
                matching_dirs = []
                break
            library_matches[library] = matched_dirs
            for directory in matched_dirs:
                if directory not in matching_dirs:
                    matching_dirs.append(directory)
        if library_matches:
            detected.append(line)
            runtime_dirs[line] = matching_dirs
    return detected, runtime_dirs


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


def parse_published_artifact(raw: Any) -> PublishedLlamaArtifact | None:
    if not isinstance(raw, dict):
        raise ValueError("artifact entry was not an object")
    asset_name = raw.get("asset_name")
    install_kind = raw.get("install_kind")
    if not isinstance(asset_name, str) or not asset_name:
        raise ValueError("artifact.asset_name was missing or not a string")
    if not isinstance(install_kind, str) or not install_kind:
        raise ValueError(
            f"artifact {asset_name} install_kind was missing or not a string"
        )

    supported_sms_raw = raw.get("supported_sms", [])
    if not isinstance(supported_sms_raw, (list, tuple)):
        raise ValueError(f"artifact {asset_name} supported_sms must be a list or tuple")
    if any(not isinstance(value, (int, str)) for value in supported_sms_raw):
        raise ValueError(
            f"artifact {asset_name} supported_sms entries must be ints or strings"
        )
    supported_sms = normalize_compute_caps(supported_sms_raw)

    min_sm_raw = raw.get("min_sm")
    max_sm_raw = raw.get("max_sm")
    try:
        min_sm = int(min_sm_raw) if min_sm_raw is not None else None
        max_sm = int(max_sm_raw) if max_sm_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"artifact {asset_name} min_sm/max_sm were not integers"
        ) from exc
    runtime_line = raw.get("runtime_line")
    coverage_class = raw.get("coverage_class")
    bundle_profile = raw.get("bundle_profile")
    rank_raw = raw.get("rank", 1000)
    if runtime_line is not None and not isinstance(runtime_line, str):
        raise ValueError(f"artifact {asset_name} runtime_line was not a string")
    if coverage_class is not None and not isinstance(coverage_class, str):
        raise ValueError(f"artifact {asset_name} coverage_class was not a string")
    if bundle_profile is not None and not isinstance(bundle_profile, str):
        raise ValueError(f"artifact {asset_name} bundle_profile was not a string")
    try:
        rank = int(rank_raw)
    except (TypeError, ValueError):
        raise ValueError(f"artifact {asset_name} rank was not an integer")
    return PublishedLlamaArtifact(
        asset_name = asset_name,
        install_kind = install_kind,
        runtime_line = runtime_line
        if isinstance(runtime_line, str) and runtime_line
        else None,
        coverage_class = coverage_class
        if isinstance(coverage_class, str) and coverage_class
        else None,
        supported_sms = supported_sms,
        min_sm = min_sm,
        max_sm = max_sm,
        bundle_profile = bundle_profile
        if isinstance(bundle_profile, str) and bundle_profile
        else None,
        rank = rank,
    )


def parse_published_release_bundle(
    repo: str, release: dict[str, Any]
) -> PublishedReleaseBundle | None:
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or not release_tag:
        return None

    assets = release_asset_map(release)
    manifest_url = assets.get(DEFAULT_PUBLISHED_MANIFEST_ASSET)
    if not manifest_url:
        return None

    # Mixed repos are filtered by an explicit release-side manifest rather than
    # by release tag or asset filename conventions.
    manifest_payload = fetch_json(manifest_url)
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} was not a JSON object"
        )
    component = manifest_payload.get("component")
    upstream_tag = manifest_payload.get("upstream_tag")
    if component != "llama.cpp":
        return None
    if not isinstance(upstream_tag, str) or not upstream_tag:
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} in {repo}@{release_tag} omitted upstream_tag"
        )

    artifacts_payload = manifest_payload.get("artifacts")
    if not isinstance(artifacts_payload, list):
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} in {repo}@{release_tag} omitted artifacts"
        )

    artifacts: list[PublishedLlamaArtifact] = []
    for index, raw_artifact in enumerate(artifacts_payload):
        try:
            artifact = parse_published_artifact(raw_artifact)
        except ValueError as exc:
            log(
                f"published artifact ignored for {repo}@{release_tag} artifact[{index}]: {exc}"
            )
            continue
        if artifact is not None:
            artifacts.append(artifact)
    selection_log = [
        f"published_release: repo={repo}",
        f"published_release: tag={release_tag}",
        f"published_release: manifest={DEFAULT_PUBLISHED_MANIFEST_ASSET}",
        f"published_release: upstream_tag={upstream_tag}",
    ]
    return PublishedReleaseBundle(
        repo = repo,
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        assets = assets,
        manifest_asset_name = DEFAULT_PUBLISHED_MANIFEST_ASSET,
        artifacts = artifacts,
        selection_log = selection_log,
    )


def parse_approved_release_checksums(
    repo: str,
    release_tag: str,
    payload: Any,
) -> ApprovedReleaseChecksums:
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} was not a JSON object"
        )
    if payload.get("component") != "llama.cpp":
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} did not describe llama.cpp"
        )
    payload_release_tag = payload.get("release_tag")
    if not isinstance(payload_release_tag, str) or not payload_release_tag:
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} omitted release_tag"
        )
    if payload_release_tag != release_tag:
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} release_tag={payload_release_tag} "
            f"did not match pinned release tag {release_tag}"
        )
    upstream_tag = payload.get("upstream_tag")
    if not isinstance(upstream_tag, str) or not upstream_tag:
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} omitted upstream_tag"
        )
    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, dict):
        raise RuntimeError(
            f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} omitted artifacts"
        )

    artifacts: dict[str, ApprovedArtifactHash] = {}
    for asset_name, raw_entry in artifacts_payload.items():
        if not isinstance(asset_name, str) or not asset_name:
            raise RuntimeError(
                "published checksum asset used a non-string artifact key"
            )
        if not isinstance(raw_entry, dict):
            raise RuntimeError(
                f"published checksum entry for {asset_name} was not an object"
            )
        digest = normalize_sha256_digest(raw_entry.get("sha256"))
        if not digest:
            raise RuntimeError(
                f"published checksum entry for {asset_name} omitted a valid sha256"
            )
        repo_value = raw_entry.get("repo")
        kind_value = raw_entry.get("kind")
        artifacts[asset_name] = ApprovedArtifactHash(
            asset_name = asset_name,
            sha256 = digest,
            repo = repo_value if isinstance(repo_value, str) and repo_value else None,
            kind = kind_value if isinstance(kind_value, str) and kind_value else None,
        )

    source_commit = payload.get("source_commit")
    return ApprovedReleaseChecksums(
        repo = repo,
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        source_commit = source_commit
        if isinstance(source_commit, str) and source_commit
        else None,
        artifacts = artifacts,
    )


def load_approved_release_checksums(
    repo: str, release_tag: str
) -> ApprovedReleaseChecksums:
    try:
        release = github_release(repo, release_tag)
    except Exception as exc:
        raise PrebuiltFallback(
            f"approved prebuilt release {repo}@{release_tag} was not available"
        ) from exc
    assets = release_asset_map(release)
    checksum_url = assets.get(DEFAULT_PUBLISHED_SHA256_ASSET)
    if not checksum_url:
        raise PrebuiltFallback(
            f"approved prebuilt release {repo}@{release_tag} did not expose {DEFAULT_PUBLISHED_SHA256_ASSET}"
        )
    try:
        payload = fetch_json(checksum_url)
        checksums = parse_approved_release_checksums(repo, release_tag, payload)
    except PrebuiltFallback:
        raise
    except Exception as exc:
        raise PrebuiltFallback(
            f"approved checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} in {repo}@{release_tag} was invalid"
        ) from exc
    return checksums


def iter_published_release_bundles(
    repo: str, published_release_tag: str = ""
) -> Iterable[PublishedReleaseBundle]:
    releases = (
        [github_release(repo, published_release_tag)]
        if published_release_tag
        else github_releases(repo)
    )
    for release in releases:
        if not published_release_tag and (
            release.get("draft") or release.get("prerelease")
        ):
            continue
        try:
            bundle = parse_published_release_bundle(repo, release)
        except Exception as exc:
            release_tag = release.get("tag_name", "unknown")
            log(f"published release metadata ignored for {repo}@{release_tag}: {exc}")
            continue
        if bundle is None:
            continue
        yield bundle


def linux_cuda_choice_from_release(
    host: HostInfo,
    release: PublishedReleaseBundle,
    preferred_runtime_line: str | None = None,
    selection_preamble: Iterable[str] = (),
) -> LinuxCudaSelection | None:
    host_sms = normalize_compute_caps(host.compute_caps)
    detected_runtime_lines, runtime_dirs = detected_linux_runtime_lines()
    driver_runtime_lines = compatible_linux_runtime_lines(host)
    runtime_lines = [
        runtime_line
        for runtime_line in detected_runtime_lines
        if runtime_line in driver_runtime_lines
    ]
    ordered_runtime_lines = list(runtime_lines)
    selection_log = (
        list(release.selection_log)
        + list(selection_preamble)
        + [
            f"linux_cuda_selection: release={release.release_tag}",
            f"linux_cuda_selection: detected_sms={','.join(host_sms) if host_sms else 'unknown'}",
            "linux_cuda_selection: detected_runtime_lines="
            + (",".join(detected_runtime_lines) if detected_runtime_lines else "none"),
            "linux_cuda_selection: driver_runtime_lines="
            + (",".join(driver_runtime_lines) if driver_runtime_lines else "none"),
            "linux_cuda_selection: compatible_runtime_lines="
            + (",".join(runtime_lines) if runtime_lines else "none"),
        ]
    )
    for runtime_line in ("cuda13", "cuda12"):
        selection_log.append(
            "linux_cuda_selection: runtime_dirs "
            f"{runtime_line}="
            + (
                ",".join(runtime_dirs.get(runtime_line, []))
                if runtime_dirs.get(runtime_line)
                else "none"
            )
        )
    published_artifacts = [
        artifact
        for artifact in release.artifacts
        if artifact.install_kind == "linux-cuda"
    ]
    published_asset_names = sorted(
        artifact.asset_name for artifact in published_artifacts
    )
    selection_log.append(
        "linux_cuda_selection: published_assets="
        + (",".join(published_asset_names) if published_asset_names else "none")
    )

    if not host_sms:
        selection_log.append(
            "linux_cuda_selection: compute capability detection unavailable; prefer portable by runtime line"
        )
    if not runtime_lines:
        selection_log.append(
            "linux_cuda_selection: no Linux CUDA runtime line satisfied both runtime libraries and driver compatibility"
        )
        return None

    if preferred_runtime_line:
        if preferred_runtime_line in ordered_runtime_lines:
            ordered_runtime_lines = [preferred_runtime_line] + [
                runtime_line
                for runtime_line in ordered_runtime_lines
                if runtime_line != preferred_runtime_line
            ]
            selection_log.append(
                "linux_cuda_selection: torch_preferred_runtime_line="
                f"{preferred_runtime_line} reordered_attempts={','.join(ordered_runtime_lines)}"
            )
        else:
            selection_log.append(
                "linux_cuda_selection: torch_preferred_runtime_line="
                f"{preferred_runtime_line} unavailable_on_host"
            )

    attempts: list[AssetChoice] = []
    seen_attempts: set[str] = set()

    def add_attempt(
        artifact: PublishedLlamaArtifact, asset_url: str, reason: str
    ) -> None:
        asset_name = artifact.asset_name
        if asset_name in seen_attempts:
            return
        seen_attempts.add(asset_name)
        attempts.append(
            AssetChoice(
                repo = release.repo,
                tag = release.release_tag,
                name = asset_name,
                url = asset_url,
                source_label = "published",
                is_ready_bundle = True,
                install_kind = "linux-cuda",
                bundle_profile = artifact.bundle_profile,
                runtime_line = artifact.runtime_line,
                coverage_class = artifact.coverage_class,
                supported_sms = artifact.supported_sms,
                min_sm = artifact.min_sm,
                max_sm = artifact.max_sm,
                selection_log = list(selection_log)
                + [
                    "linux_cuda_selection: selected "
                    f"{asset_name} runtime_line={artifact.runtime_line} coverage_class={artifact.coverage_class} reason={reason}"
                ],
            )
        )

    for runtime_line in ordered_runtime_lines:
        coverage_candidates: list[tuple[PublishedLlamaArtifact, str]] = []
        portable_candidate: tuple[PublishedLlamaArtifact, str] | None = None
        for artifact in published_artifacts:
            if artifact.runtime_line != runtime_line:
                continue
            asset_name = artifact.asset_name
            asset_url = release.assets.get(asset_name)
            if not asset_url:
                selection_log.append(
                    f"linux_cuda_selection: reject {asset_name} missing asset"
                )
                continue
            if not host_sms and artifact.coverage_class != "portable":
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=unknown_compute_caps_prefer_portable"
                )
                continue

            if not artifact.supported_sms:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=artifact_missing_supported_sms"
                )
                continue
            if artifact.min_sm is None or artifact.max_sm is None:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    "reason=artifact_missing_sm_bounds"
                )
                continue

            supported_sms = {str(value) for value in artifact.supported_sms}
            missing_sms = [sm for sm in host_sms if sm not in supported_sms]
            out_of_range_sms = [
                sm
                for sm in host_sms
                if not (artifact.min_sm <= int(sm) <= artifact.max_sm)
            ]
            reasons: list[str] = []
            if missing_sms:
                reasons.append(f"missing_sms={','.join(missing_sms)}")
            if out_of_range_sms:
                reasons.append(f"out_of_range_sms={','.join(out_of_range_sms)}")
            if reasons:
                selection_log.append(
                    "linux_cuda_selection: reject "
                    f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                    f"coverage={artifact.min_sm}-{artifact.max_sm} supported={','.join(artifact.supported_sms)} "
                    f"reasons={' '.join(reasons)}"
                )
                continue

            selection_log.append(
                "linux_cuda_selection: accept "
                f"{asset_name} runtime_line={runtime_line} coverage_class={artifact.coverage_class} "
                f"coverage={artifact.min_sm}-{artifact.max_sm} supported={','.join(artifact.supported_sms)}"
            )
            if artifact.coverage_class == "portable":
                portable_candidate = (artifact, asset_url)
            else:
                coverage_candidates.append((artifact, asset_url))

        if coverage_candidates:
            artifact, url = sorted(
                coverage_candidates,
                key = lambda item: (
                    (item[0].max_sm or 0) - (item[0].min_sm or 0),
                    item[0].rank,
                    item[0].max_sm or 0,
                ),
            )[0]
            add_attempt(artifact, url, "best coverage for runtime line")
        if portable_candidate:
            artifact, url = portable_candidate
            add_attempt(artifact, url, "portable fallback for runtime line")

    if not attempts:
        return None

    selection_log.append(
        "linux_cuda_selection: attempt_order="
        + ",".join(choice.name for choice in attempts)
    )
    for attempt in attempts:
        attempt.selection_log = list(selection_log) + [
            "linux_cuda_selection: attempt "
            f"{attempt.name} runtime_line={attempt.runtime_line} coverage_class={attempt.coverage_class}"
        ]
    return LinuxCudaSelection(attempts = attempts, selection_log = selection_log)


def latest_published_linux_cuda_tag(host: HostInfo, published_repo: str) -> str | None:
    for release in iter_published_release_bundles(published_repo):
        if linux_cuda_choice_from_release(host, release):
            return release.upstream_tag
    return None


def iter_upstream_releases() -> Iterable[dict[str, Any]]:
    for release in github_releases(UPSTREAM_REPO):
        if release.get("draft") or release.get("prerelease"):
            continue
        yield release


def pinned_published_release_bundle(
    repo: str, published_release_tag: str
) -> PublishedReleaseBundle:
    bundle = next(iter_published_release_bundles(repo, published_release_tag), None)
    if bundle is None:
        raise PrebuiltFallback(
            f"published release {repo}@{published_release_tag} did not expose a usable llama.cpp manifest"
        )
    return bundle


def resolve_requested_llama_tag(
    requested_tag: str | None,
    published_repo: str = "",
) -> str:
    """Resolve a llama.cpp tag for source-build fallback.

    Resolution order:
      1. Concrete tag (e.g. "b8508") -- returned as-is.
      2. "latest" with published_repo -- query the Unsloth release repo
         (e.g. unslothai/llama.cpp) for its latest release tag. This is the
         tested/approved version that matches the prebuilt binaries.
      3. "latest" without published_repo or if (2) fails -- query the upstream
         ggml-org/llama.cpp repo. This may return a newer, untested tag.

    The Unsloth repo is preferred because its releases are pinned to specific
    upstream tags that have been validated with Unsloth Studio. Using the
    upstream bleeding-edge tag risks API/ABI incompatibilities.
    """
    if requested_tag and requested_tag != "latest":
        return requested_tag
    # Prefer the Unsloth release repo tag (tested/approved) over bleeding-edge
    # upstream. For example, unslothai/llama.cpp may publish b8508 while
    # ggml-org/llama.cpp latest is b8514. The source-build fallback should
    # compile the same version the prebuilt path would have installed.
    if published_repo:
        try:
            payload = fetch_json(
                f"https://api.github.com/repos/{published_repo}/releases/latest"
            )
            tag = payload.get("tag_name")
            if isinstance(tag, str) and tag:
                return tag
        except Exception:
            pass
    # Fall back to upstream ggml-org latest release tag
    return latest_upstream_release_tag()


def resolve_requested_install_tag(
    requested_tag: str | None,
    published_release_tag: str = "",
) -> str:
    approved_tag = APPROVED_PREBUILT_LLAMA_TAG
    normalized_requested = requested_tag or "latest"
    if normalized_requested not in {"latest", approved_tag}:
        raise PrebuiltFallback(
            f"prebuilt installs are pinned to approved release {approved_tag}; requested {normalized_requested}"
        )
    if published_release_tag and published_release_tag != approved_tag:
        raise PrebuiltFallback(
            f"prebuilt installs require published release tag {approved_tag}; requested {published_release_tag}"
        )
    return approved_tag


def run_capture(
    command: list[str],
    *,
    timeout: int = 30,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output = True,
        text = True,
        timeout = timeout,
        env = env,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )
    return result


def detect_host() -> HostInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_macos = system == "Darwin"
    is_x86_64 = machine in {"x86_64", "amd64"}
    is_arm64 = machine in {"arm64", "aarch64"}

    nvidia_smi = shutil.which("nvidia-smi")
    driver_cuda_version = None
    compute_caps: list[str] = []
    visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_device_tokens = parse_cuda_visible_devices(visible_cuda_devices)
    has_physical_nvidia = False
    has_usable_nvidia = False
    if nvidia_smi:
        try:
            result = run_capture([nvidia_smi], timeout = 20)
            merged = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if "NVIDIA-SMI" in merged:
                has_physical_nvidia = True
                has_usable_nvidia = visible_device_tokens != []
            for line in merged.splitlines():
                if "CUDA Version:" in line:
                    raw = line.split("CUDA Version:", 1)[1].strip().split()[0]
                    major, minor = raw.split(".", 1)
                    driver_cuda_version = (int(major), int(minor))
                    break
        except Exception:
            pass

        try:
            caps = run_capture(
                [
                    nvidia_smi,
                    "--query-gpu=index,uuid,compute_cap",
                    "--format=csv,noheader",
                ],
                timeout = 20,
            )
            visible_gpu_rows: list[tuple[str, str, str]] = []
            for raw in caps.stdout.splitlines():
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) != 3:
                    continue
                index, uuid, cap = parts
                visible_gpu_row = select_visible_gpu_rows(
                    [(index, uuid, cap)],
                    visible_device_tokens,
                )
                if not visible_gpu_row:
                    continue
                visible_gpu_rows.extend(visible_gpu_row)
                normalized_cap = normalize_compute_cap(cap)
                if normalized_cap is None:
                    continue
                if normalized_cap not in compute_caps:
                    compute_caps.append(normalized_cap)

            if visible_gpu_rows:
                has_usable_nvidia = True
            elif visible_device_tokens == []:
                has_usable_nvidia = False
            elif supports_explicit_visible_device_matching(visible_device_tokens):
                has_usable_nvidia = False
            elif has_physical_nvidia:
                has_usable_nvidia = True
        except Exception:
            pass

    # Detect AMD ROCm (HIP) -- require actual GPU, not just tools installed
    has_rocm = False
    if not is_macos:
        for _cmd, _marker in (
            (["rocminfo"], "gfx"),
            (["amd-smi", "list"], None),
        ):
            _exe = shutil.which(_cmd[0])
            if not _exe:
                continue
            try:
                _result = run_capture([_exe, *_cmd[1:]], timeout = 10)
            except Exception:
                continue
            if _result.returncode == 0 and _result.stdout.strip():
                if _marker is None or _marker in _result.stdout.lower():
                    has_rocm = True
                    break

    return HostInfo(
        system = system,
        machine = machine,
        is_windows = is_windows,
        is_linux = is_linux,
        is_macos = is_macos,
        is_x86_64 = is_x86_64,
        is_arm64 = is_arm64,
        nvidia_smi = nvidia_smi,
        driver_cuda_version = driver_cuda_version,
        compute_caps = compute_caps,
        visible_cuda_devices = visible_cuda_devices,
        has_physical_nvidia = has_physical_nvidia,
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
    )


def pick_windows_cuda_runtime(host: HostInfo) -> str | None:
    if not host.driver_cuda_version:
        return None
    major, minor = host.driver_cuda_version
    if major > 13 or (major == 13 and minor >= 1):
        return "13.1"
    if major > 12 or (major == 12 and minor >= 4):
        return "12.4"
    return None


def compatible_linux_runtime_lines(host: HostInfo) -> list[str]:
    if not host.driver_cuda_version:
        return []
    major, _minor = host.driver_cuda_version
    if major >= 13:
        return ["cuda13", "cuda12"]
    if major >= 12:
        return ["cuda12"]
    return []


def windows_runtime_line_info() -> dict[str, tuple[str, ...]]:
    return {
        "cuda13": ("cudart64_13*.dll", "cublas64_13*.dll", "cublasLt64_13*.dll"),
        "cuda12": ("cudart64_12*.dll", "cublas64_12*.dll", "cublasLt64_12*.dll"),
    }


def detected_windows_runtime_lines() -> tuple[list[str], dict[str, list[str]]]:
    dirs = windows_runtime_dirs()
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for runtime_line, required_patterns in windows_runtime_line_info().items():
        matching_dirs = windows_runtime_dirs_for_patterns(required_patterns, dirs)
        if matching_dirs:
            detected.append(runtime_line)
            runtime_dirs[runtime_line] = matching_dirs
    return detected, runtime_dirs


def compatible_windows_runtime_lines(host: HostInfo) -> list[str]:
    driver_runtime = pick_windows_cuda_runtime(host)
    if driver_runtime == "13.1":
        return ["cuda13", "cuda12"]
    if driver_runtime == "12.4":
        return ["cuda12"]
    return []


def runtime_line_from_cuda_version(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    raw = str(cuda_version).strip()
    if not raw:
        return None
    major, _, _ = raw.partition(".")
    if major == "12":
        return "cuda12"
    if major == "13":
        return "cuda13"
    return None


def detect_torch_cuda_runtime_preference(host: HostInfo) -> CudaRuntimePreference:
    selection_log: list[str] = []
    if host.is_macos:
        selection_log.append("torch_cuda_preference: skipped on macOS")
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)
    if not (host.has_usable_nvidia and (host.is_linux or host.is_windows)):
        selection_log.append(
            "torch_cuda_preference: skipped because CUDA host prerequisites were not met"
        )
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    try:
        import torch
    except Exception as exc:
        selection_log.append(f"torch_cuda_preference: import failed: {exc}")
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        selection_log.append(
            "torch_cuda_preference: torch.version.cuda missing; skipping Torch shortcut"
        )
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        selection_log.append(
            f"torch_cuda_preference: torch.cuda.is_available() failed: {exc}"
        )
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    if not cuda_available:
        selection_log.append(
            "torch_cuda_preference: torch.cuda.is_available() returned False; falling back to normal selection"
        )
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    runtime_line = runtime_line_from_cuda_version(cuda_version)
    if runtime_line is None:
        selection_log.append(
            f"torch_cuda_preference: unsupported torch.version.cuda={cuda_version}; falling back to normal selection"
        )
        return CudaRuntimePreference(runtime_line = None, selection_log = selection_log)

    selection_log.append(
        "torch_cuda_preference: selected runtime_line="
        f"{runtime_line} from torch.version.cuda={cuda_version}"
    )
    return CudaRuntimePreference(runtime_line = runtime_line, selection_log = selection_log)


def windows_cuda_attempts(
    host: HostInfo,
    llama_tag: str,
    upstream_assets: dict[str, str],
    preferred_runtime_line: str | None,
    selection_preamble: Iterable[str] = (),
) -> list[AssetChoice]:
    selection_log = list(selection_preamble)
    runtime_by_line = {"cuda12": "12.4", "cuda13": "13.1"}
    driver_runtime = pick_windows_cuda_runtime(host)
    detected_runtime_lines, runtime_dirs = detected_windows_runtime_lines()
    compatible_runtime_lines = compatible_windows_runtime_lines(host)
    normal_runtime_lines: list[str]
    if detected_runtime_lines:
        normal_runtime_lines = [
            line for line in compatible_runtime_lines if line in detected_runtime_lines
        ]
    else:
        normal_runtime_lines = compatible_runtime_lines
    selection_log.append(
        "windows_cuda_selection: driver_runtime="
        + (driver_runtime if driver_runtime else "unknown")
    )
    selection_log.append(
        "windows_cuda_selection: detected_runtime_lines="
        + (",".join(detected_runtime_lines) if detected_runtime_lines else "none")
    )
    for runtime_line in ("cuda13", "cuda12"):
        selection_log.append(
            "windows_cuda_selection: runtime_dirs "
            f"{runtime_line}="
            + (
                ",".join(runtime_dirs.get(runtime_line, []))
                if runtime_dirs.get(runtime_line)
                else "none"
            )
        )
    if detected_runtime_lines:
        selection_log.append(
            "windows_cuda_selection: host_runtime_order="
            + (",".join(normal_runtime_lines) if normal_runtime_lines else "none")
        )
    else:
        selection_log.append(
            "windows_cuda_selection: no CUDA runtime DLL line detected; falling back to driver order"
        )
    if not normal_runtime_lines:
        if detected_runtime_lines:
            selection_log.append(
                "windows_cuda_selection: detected CUDA runtime DLLs were incompatible with the reported driver"
            )
        fallback_runtime_lines = (
            ["cuda13", "cuda12"]
            if driver_runtime == "13.1"
            else (["cuda12"] if driver_runtime == "12.4" else [])
        )
        normal_runtime_lines = fallback_runtime_lines

    runtime_order: list[str] = []
    if preferred_runtime_line and preferred_runtime_line in normal_runtime_lines:
        runtime_order.append(preferred_runtime_line)
        selection_log.append(
            "windows_cuda_selection: torch_preferred_runtime_line="
            f"{preferred_runtime_line} reordered_attempts"
        )
    elif preferred_runtime_line:
        selection_log.append(
            "windows_cuda_selection: torch_preferred_runtime_line="
            f"{preferred_runtime_line} unavailable_or_incompatible"
        )
    else:
        selection_log.append(
            "windows_cuda_selection: no Torch runtime preference available"
        )

    runtime_order.extend(
        runtime_line
        for runtime_line in normal_runtime_lines
        if runtime_line not in runtime_order
    )
    selection_log.append(
        "windows_cuda_selection: normal_runtime_order="
        + (",".join(normal_runtime_lines) if normal_runtime_lines else "none")
    )
    selection_log.append(
        "windows_cuda_selection: attempt_runtime_order="
        + (",".join(runtime_order) if runtime_order else "none")
    )

    attempts: list[AssetChoice] = []
    for runtime_line in runtime_order:
        runtime = runtime_by_line[runtime_line]
        upstream_name = f"llama-{llama_tag}-bin-win-cuda-{runtime}-x64.zip"
        asset_url = upstream_assets.get(upstream_name)
        if not asset_url:
            selection_log.append(
                f"windows_cuda_selection: skip missing asset {upstream_name}"
            )
            continue
        attempts.append(
            AssetChoice(
                repo = UPSTREAM_REPO,
                tag = llama_tag,
                name = upstream_name,
                url = asset_url,
                source_label = "upstream",
                install_kind = "windows-cuda",
                runtime_line = runtime_line,
                selection_log = list(selection_log)
                + [
                    f"windows_cuda_selection: selected {upstream_name} runtime={runtime}"
                ],
            )
        )
    return attempts


def resolve_windows_cuda_choices(
    host: HostInfo, llama_tag: str, upstream_assets: dict[str, str]
) -> list[AssetChoice]:
    torch_preference = detect_torch_cuda_runtime_preference(host)
    attempts = windows_cuda_attempts(
        host,
        llama_tag,
        upstream_assets,
        torch_preference.runtime_line,
        torch_preference.selection_log,
    )
    return attempts


def resolve_linux_cuda_choice(
    host: HostInfo, llama_tag: str, published_repo: str, published_release_tag: str
) -> LinuxCudaSelection:
    torch_preference = detect_torch_cuda_runtime_preference(host)
    skipped_tag_mismatches = 0
    for release in iter_published_release_bundles(
        published_repo, published_release_tag
    ):
        if release.upstream_tag != llama_tag:
            skipped_tag_mismatches += 1
            continue
        selection = linux_cuda_choice_from_release(
            host,
            release,
            preferred_runtime_line = torch_preference.runtime_line,
            selection_preamble = torch_preference.selection_log,
        )
        if selection is not None:
            return selection
    if skipped_tag_mismatches:
        log(
            "published Linux CUDA selection skipped "
            f"{skipped_tag_mismatches} release(s) with upstream_tag != {llama_tag}"
        )
    raise PrebuiltFallback("no compatible published Linux CUDA bundle was found")


def resolve_upstream_asset_choice(host: HostInfo, llama_tag: str) -> AssetChoice:
    upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)
    if host.is_linux and host.is_x86_64:
        # AMD ROCm: try upstream ROCm prebuilt first, then fall back to source build.
        # Source build (via setup.sh) compiles with -DGGML_HIP=ON and auto-detects
        # the exact GPU target via rocminfo, which is more reliable for consumer
        # GPUs (e.g. gfx1151) that may not be in the prebuilt.
        if host.has_rocm and not host.has_usable_nvidia:
            rocm_name = f"llama-{llama_tag}-bin-ubuntu-rocm-7.2-x64.tar.gz"
            if rocm_name in upstream_assets:
                log(f"AMD ROCm detected -- trying upstream prebuilt {rocm_name}")
                log(
                    "Note: prebuilt is compiled for ROCm 7.2; if your ROCm version differs, "
                    "this may fail preflight and fall back to a source build (safe)"
                )
                return AssetChoice(
                    repo = UPSTREAM_REPO,
                    tag = llama_tag,
                    name = rocm_name,
                    url = upstream_assets[rocm_name],
                    source_label = "upstream",
                    install_kind = "linux-rocm",
                )
            # No ROCm prebuilt available -- fall back to source build
            raise PrebuiltFallback(
                "AMD ROCm detected but no upstream ROCm prebuilt found; "
                "falling back to source build with HIP support"
            )

        upstream_name = f"llama-{llama_tag}-bin-ubuntu-x64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream Linux CPU asset was not found")
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = llama_tag,
            name = upstream_name,
            url = upstream_assets[upstream_name],
            source_label = "upstream",
            install_kind = "linux-cpu",
        )

    if host.is_windows and host.is_x86_64:
        if host.has_usable_nvidia:
            attempts = resolve_windows_cuda_choices(host, llama_tag, upstream_assets)
            if attempts:
                return attempts[0]
            raise PrebuiltFallback("no compatible Windows CUDA asset was found")

        # AMD ROCm on Windows: try HIP prebuilt
        if host.has_rocm:
            hip_name = f"llama-{llama_tag}-bin-win-hip-radeon-x64.zip"
            if hip_name in upstream_assets:
                log(
                    f"AMD ROCm detected on Windows -- trying upstream HIP prebuilt {hip_name}"
                )
                return AssetChoice(
                    repo = UPSTREAM_REPO,
                    tag = llama_tag,
                    name = hip_name,
                    url = upstream_assets[hip_name],
                    source_label = "upstream",
                    install_kind = "windows-hip",
                )
            log(
                "AMD ROCm detected on Windows but no HIP prebuilt found -- falling back to CPU"
            )

        upstream_name = f"llama-{llama_tag}-bin-win-cpu-x64.zip"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream Windows CPU asset was not found")
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = llama_tag,
            name = upstream_name,
            url = upstream_assets[upstream_name],
            source_label = "upstream",
            install_kind = "windows-cpu",
        )

    if host.is_macos and host.is_arm64:
        upstream_name = f"llama-{llama_tag}-bin-macos-arm64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream macOS arm64 asset was not found")
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = llama_tag,
            name = upstream_name,
            url = upstream_assets[upstream_name],
            source_label = "upstream",
            install_kind = "macos-arm64",
        )

    if host.is_macos and host.is_x86_64:
        upstream_name = f"llama-{llama_tag}-bin-macos-x64.tar.gz"
        if upstream_name not in upstream_assets:
            raise PrebuiltFallback("upstream macOS x64 asset was not found")
        return AssetChoice(
            repo = UPSTREAM_REPO,
            tag = llama_tag,
            name = upstream_name,
            url = upstream_assets[upstream_name],
            source_label = "upstream",
            install_kind = "macos-x64",
        )

    raise PrebuiltFallback(
        f"no prebuilt policy exists for {host.system} {host.machine}"
    )


def resolve_asset_choice(
    host: HostInfo, llama_tag: str, published_repo: str, published_release_tag: str
) -> AssetChoice:
    if host.is_linux and host.is_x86_64 and host.has_usable_nvidia:
        return resolve_linux_cuda_choice(
            host, llama_tag, published_repo, published_release_tag
        ).primary
    return resolve_upstream_asset_choice(host, llama_tag)


def extract_archive(archive_path: Path, destination: Path) -> None:
    def safe_extract_path(base: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        member_path = Path(normalized)
        if member_path.is_absolute():
            raise PrebuiltFallback(
                f"archive member used an absolute path: {member_name}"
            )

        target = (base / member_path).resolve()
        base_resolved = base.resolve()
        try:
            target.relative_to(base_resolved)
        except ValueError as exc:
            raise PrebuiltFallback(
                f"archive member escaped destination: {member_name}"
            ) from exc
        return target

    def safe_link_target(
        base: Path, member_name: str, link_name: str, target: Path
    ) -> tuple[str, Path]:
        normalized = link_name.replace("\\", "/")
        link_path = Path(normalized)
        if link_path.is_absolute():
            raise PrebuiltFallback(
                f"archive link used an absolute target: {member_name} -> {link_name}"
            )
        if not normalized:
            raise PrebuiltFallback(f"archive link used an empty target: {member_name}")

        resolved = (target.parent / link_path).resolve()
        base_resolved = base.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError as exc:
            raise PrebuiltFallback(
                f"archive link escaped destination: {member_name} -> {link_name}"
            ) from exc
        return normalized, resolved

    def extract_zip_safely(source: Path, base: Path) -> None:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = safe_extract_path(base, member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise PrebuiltFallback(
                        f"zip archive contained a symlink entry: {member.filename}"
                    )
                if member.is_dir():
                    target.mkdir(parents = True, exist_ok = True)
                    continue
                target.parent.mkdir(parents = True, exist_ok = True)
                with archive.open(member, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    def extract_tar_safely(source: Path, base: Path) -> None:
        pending_links: list[tuple[tarfile.TarInfo, Path]] = []
        with tarfile.open(source, "r:gz") as archive:
            for member in archive.getmembers():
                target = safe_extract_path(base, member.name)
                if member.isdir():
                    target.mkdir(parents = True, exist_ok = True)
                    continue
                if member.islnk() or member.issym():
                    pending_links.append((member, target))
                    continue
                if not member.isfile():
                    raise PrebuiltFallback(
                        f"tar archive contained an unsupported entry: {member.name}"
                    )
                target.parent.mkdir(parents = True, exist_ok = True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise PrebuiltFallback(
                        f"tar archive entry could not be read: {member.name}"
                    )
                with extracted, target.open("wb") as dst:
                    shutil.copyfileobj(extracted, dst)

        unresolved = list(pending_links)
        while unresolved:
            next_round: list[tuple[tarfile.TarInfo, Path]] = []
            progressed = False
            for member, target in unresolved:
                normalized_link, resolved_target = safe_link_target(
                    base, member.name, member.linkname, target
                )
                if not resolved_target.exists() and not resolved_target.is_symlink():
                    next_round.append((member, target))
                    continue
                if resolved_target.is_dir():
                    raise PrebuiltFallback(
                        f"archive link targeted a directory: {member.name} -> {member.linkname}"
                    )

                target.parent.mkdir(parents = True, exist_ok = True)
                if target.exists() or target.is_symlink():
                    target.unlink()

                if member.issym():
                    target.symlink_to(normalized_link)
                else:
                    shutil.copy2(resolved_target, target)
                progressed = True

            if not progressed:
                details = ", ".join(
                    f"{member.name} -> {member.linkname}" for member, _ in next_round
                )
                raise PrebuiltFallback(
                    f"tar archive contained unresolved link entries: {details}"
                )
            unresolved = next_round

    destination.mkdir(parents = True, exist_ok = True)
    if archive_path.name.endswith(".zip"):
        extract_zip_safely(archive_path, destination)
        return
    if archive_path.name.endswith(".tar.gz"):
        extract_tar_safely(archive_path, destination)
        return
    raise PrebuiltFallback(f"unsupported archive format: {archive_path.name}")


def copy_globs(
    source_dir: Path, destination: Path, patterns: list[str], *, required: bool = True
) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    matched_sources: dict[str, Path] = {}
    for path in sorted(
        (candidate for candidate in source_dir.rglob("*") if candidate.is_file()),
        key = lambda candidate: (
            len(candidate.relative_to(source_dir).parts),
            str(candidate),
        ),
    ):
        for pattern in patterns:
            if fnmatch.fnmatch(path.name, pattern):
                previous = matched_sources.get(path.name)
                if previous is not None and previous != path:
                    raise PrebuiltFallback(
                        f"ambiguous archive layout for {path.name}: "
                        f"{previous.relative_to(source_dir)} and {path.relative_to(source_dir)}"
                    )
                matched_sources[path.name] = path
                break

    if required and not matched_sources:
        raise PrebuiltFallback(f"required files missing from {source_dir}: {patterns}")

    for name, path in matched_sources.items():
        shutil.copy2(path, destination / name)


def ensure_converter_scripts(install_dir: Path, llama_tag: str) -> None:
    canonical = install_dir / "convert_hf_to_gguf.py"
    if not canonical.exists():
        # Hydrated source tree should have placed this file already.
        # Fall back to a network fetch so the install is not blocked.
        raw_base = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{llama_tag}"
        source_url = f"{raw_base}/convert_hf_to_gguf.py"
        data = download_bytes(
            source_url,
            progress_label = f"Downloading {download_label_from_url(source_url)}",
        )
        if not data:
            raise RuntimeError(f"downloaded empty converter script from {source_url}")
        if b"import " not in data and b"def " not in data and b"#!/" not in data:
            raise RuntimeError(
                f"downloaded converter script did not look like Python source: {source_url}"
            )
        atomic_write_bytes(canonical, data)
    legacy = install_dir / "convert-hf-to-gguf.py"
    if legacy.exists() or legacy.is_symlink():
        legacy.unlink()
    try:
        legacy.symlink_to("convert_hf_to_gguf.py")
    except OSError:
        shutil.copy2(canonical, legacy)


def extracted_archive_root(extract_dir: Path) -> Path:
    children = [path for path in extract_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def copy_directory_contents(source_dir: Path, destination: Path) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    for item in source_dir.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok = True)
        else:
            shutil.copy2(item, target)


def hydrate_source_tree(
    upstream_tag: str,
    install_dir: Path,
    work_dir: Path,
    *,
    expected_sha256: str,
) -> None:
    archive_path = work_dir / f"llama.cpp-source-{upstream_tag}.tar.gz"
    source_urls = upstream_source_archive_urls(upstream_tag)
    extract_dir = Path(tempfile.mkdtemp(prefix = "source-extract-", dir = work_dir))

    try:
        log(f"downloading llama.cpp source tree for upstream tag {upstream_tag}")
        last_exc: Exception | None = None
        downloaded = False
        for index, source_url in enumerate(source_urls):
            try:
                if index > 0:
                    log(
                        f"retrying source tree download from fallback URL: {source_url}"
                    )
                download_file_verified(
                    source_url,
                    archive_path,
                    expected_sha256 = expected_sha256,
                    label = f"llama.cpp source tree for {upstream_tag}",
                )
                downloaded = True
                break
            except Exception as exc:
                last_exc = exc
                if index == len(source_urls) - 1:
                    raise
                log(f"source tree download failed from {source_url}: {exc}")
        if not downloaded:
            assert last_exc is not None
            raise last_exc
        extract_archive(archive_path, extract_dir)
        source_root = extracted_archive_root(extract_dir)
        required_paths = [
            source_root / "CMakeLists.txt",
            source_root / "convert_hf_to_gguf.py",
            source_root / "gguf-py",
        ]
        missing = [
            str(path.relative_to(source_root))
            for path in required_paths
            if not path.exists()
        ]
        if missing:
            raise PrebuiltFallback(
                "upstream source archive was missing required repo files: "
                + ", ".join(missing)
            )
        copy_directory_contents(source_root, install_dir)
    except PrebuiltFallback:
        raise
    except Exception as exc:
        raise PrebuiltFallback(
            f"failed to hydrate upstream llama.cpp source tree for {upstream_tag}: {exc}"
        ) from exc
    finally:
        remove_tree(extract_dir)


def normalize_install_layout(install_dir: Path, host: HostInfo) -> tuple[Path, Path]:
    build_bin = install_dir / "build" / "bin"
    if host.is_windows:
        exec_dir = build_bin / "Release"
        exec_dir.mkdir(parents = True, exist_ok = True)
        return exec_dir / "llama-server.exe", exec_dir / "llama-quantize.exe"

    install_dir.mkdir(parents = True, exist_ok = True)
    build_bin.mkdir(parents = True, exist_ok = True)
    return install_dir / "llama-server", install_dir / "llama-quantize"


def discover_installed_executable(install_dir: Path, executable_name: str) -> Path:
    direct = install_dir / executable_name
    if direct.exists() and direct.is_file():
        return direct
    candidate = next(
        (path for path in install_dir.rglob(executable_name) if path.is_file()), None
    )
    if candidate is None:
        raise PrebuiltFallback(f"{executable_name} was not installed")
    return candidate


def write_exec_wrapper(entrypoint: Path, target: Path) -> None:
    relative_target = os.path.relpath(target, entrypoint.parent)
    script = "\n".join(
        [
            "#!/bin/sh",
            f'exec "$(dirname "$0")/{relative_target}" "$@"',
            "",
        ]
    )
    atomic_write_bytes(entrypoint, script.encode("utf-8"))
    os.chmod(entrypoint, 0o755)


def create_exec_entrypoint(entrypoint: Path, target: Path) -> None:
    if entrypoint == target:
        return
    if entrypoint.exists() or entrypoint.is_symlink():
        entrypoint.unlink()
    try:
        entrypoint.symlink_to(os.path.relpath(target, entrypoint.parent))
    except Exception:
        write_exec_wrapper(entrypoint, target)


def overlay_directory_for_choice(
    install_dir: Path, choice: AssetChoice, host: HostInfo
) -> Path:
    if host.is_windows or choice.install_kind.startswith("windows"):
        path = install_dir / "build" / "bin" / "Release"
    else:
        path = install_dir / "build" / "bin"
    path.mkdir(parents = True, exist_ok = True)
    return path


def runtime_patterns_for_choice(choice: AssetChoice) -> list[str]:
    if choice.install_kind in {"linux-cpu", "linux-cuda", "linux-rocm"}:
        return [
            "llama-server",
            "llama-quantize",
            "libllama.so*",
            "libggml.so*",
            "libggml-base.so*",
            "libmtmd.so*",
            "libggml-cpu-*.so*",
            "libggml-cuda.so*",
            "libggml-hip.so*",
            "libggml-rpc.so*",
        ]
    if choice.install_kind in {"macos-arm64", "macos-x64"}:
        return ["llama-server", "llama-quantize", "lib*.dylib"]
    if choice.install_kind in {"windows-cpu", "windows-cuda", "windows-hip"}:
        return ["*.exe", "*.dll"]
    raise PrebuiltFallback(
        f"unsupported install kind for runtime overlay: {choice.install_kind}"
    )


def metadata_patterns_for_choice(choice: AssetChoice) -> list[str]:
    patterns = ["BUILD_INFO.txt", "THIRD_PARTY_LICENSES.txt"]
    if choice.install_kind.startswith("windows"):
        patterns.append("LICENSE.txt")
    else:
        patterns.append("LICENSE")
    return patterns


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents = True, exist_ok = True)

    if FileLock is None:
        # Fallback: exclusive file creation as a simple lock.
        # Write our PID so stale locks from crashed processes can be detected.
        fd: int | None = None
        deadline = time.monotonic() + INSTALL_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.fsync(fd)
                break
            except FileExistsError:
                # Check if the holder process is still alive
                stale = False
                try:
                    raw = lock_path.read_text().strip()
                except FileNotFoundError:
                    # Lock vanished between our open attempt and read -- retry
                    continue
                if not raw:
                    # File exists but PID not yet written -- another process
                    # just created it. Wait briefly for the write to land.
                    time.sleep(0.1)
                    continue
                try:
                    holder_pid = int(raw)
                    os.kill(holder_pid, 0)  # signal 0 = existence check
                except ValueError:
                    # PID unreadable (corrupted file)
                    stale = True
                except ProcessLookupError:
                    # Process is dead
                    stale = True
                except PermissionError:
                    # Process is alive but owned by another user -- not stale
                    pass
                if stale:
                    lock_path.unlink(missing_ok = True)
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for concurrent install lock: {lock_path}"
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
        with FileLock(lock_path, timeout = INSTALL_LOCK_TIMEOUT_SECONDS):
            yield
    except FileLockTimeout as exc:
        raise RuntimeError(
            f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for concurrent install lock: {lock_path}"
        ) from exc


def install_lock_path(install_dir: Path) -> Path:
    return install_dir.parent / f".{install_dir.name}.install.lock"


def install_staging_root(install_dir: Path) -> Path:
    root = install_dir.parent / INSTALL_STAGING_ROOT_NAME
    root.mkdir(parents = True, exist_ok = True)
    return root


def prune_install_staging_root(install_dir: Path) -> None:
    root = install_dir.parent / INSTALL_STAGING_ROOT_NAME
    try:
        root.rmdir()
    except OSError:
        pass


def create_install_staging_dir(install_dir: Path) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix = f"{install_dir.name}.staging-", dir = install_staging_root(install_dir)
        )
    )
    log(f"created install staging dir {staging_dir}")
    return staging_dir


def unique_install_side_path(install_dir: Path, label: str) -> Path:
    root = install_staging_root(install_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    prefix = f"{install_dir.name}.{label}-{timestamp}-{os.getpid()}"
    candidate = root / prefix
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = root / f"{prefix}-{counter}"
    return candidate


def remove_tree(path: Path | None) -> None:
    if path and path.exists():
        shutil.rmtree(path, ignore_errors = True)


def remove_tree_logged(path: Path | None, label: str) -> None:
    if not path:
        return
    if not path.exists():
        log(f"{label} already absent at {path}")
        return
    log(f"removing {label} at {path}")
    try:
        shutil.rmtree(path)
    except Exception as exc:
        log(f"failed to remove {label} at {path}: {exc}")
        raise


def cleanup_install_side_paths(
    install_dir: Path,
    *,
    staging_dir: Path | None = None,
    rollback_dir: Path | None = None,
    failed_dir: Path | None = None,
    active_dir: Path | None = None,
) -> None:
    cleanup_failures: list[str] = []
    for label, path in (
        ("failed install path", failed_dir),
        ("rollback path", rollback_dir),
        ("active install path", active_dir),
        ("staging dir", staging_dir),
    ):
        if not path:
            continue
        try:
            remove_tree_logged(path, label)
        except Exception as exc:
            cleanup_failures.append(f"{label} ({path}): {exc}")
    prune_install_staging_root(install_dir)
    if cleanup_failures:
        raise RuntimeError("cleanup failed for " + "; ".join(cleanup_failures))


def confirm_install_tree(install_dir: Path, host: HostInfo) -> None:
    if host.is_windows:
        expected = [
            install_dir / "build" / "bin" / "Release" / "llama-server.exe",
            install_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
            install_dir / "convert_hf_to_gguf.py",
            install_dir / "gguf-py",
        ]
    else:
        expected = [
            install_dir / "llama-server",
            install_dir / "llama-quantize",
            install_dir / "build" / "bin" / "llama-server",
            install_dir / "build" / "bin" / "llama-quantize",
            install_dir / "convert_hf_to_gguf.py",
            install_dir / "gguf-py",
        ]

    expected.append(install_dir / "UNSLOTH_PREBUILT_INFO.json")
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            "activated install was missing expected files: " + ", ".join(missing)
        )


def activate_install_tree(staging_dir: Path, install_dir: Path, host: HostInfo) -> None:
    rollback_dir: Path | None = None
    failed_dir: Path | None = None
    try:
        if install_dir.exists():
            rollback_dir = unique_install_side_path(install_dir, "rollback")
            log(f"moving existing install to rollback path {rollback_dir}")
            os.replace(install_dir, rollback_dir)
            log(f"moved existing install to rollback path {rollback_dir.name}")

        log(f"activating staged install {staging_dir} -> {install_dir}")
        os.replace(staging_dir, install_dir)
        log(f"activated staged install at {install_dir}")
        log(f"confirming activated install tree at {install_dir}")
        confirm_install_tree(install_dir, host)
        log(f"activated install tree confirmed at {install_dir}")
    except Exception as exc:
        log(f"activation failed for staged install: {exc}")
        try:
            if install_dir.exists():
                failed_dir = unique_install_side_path(install_dir, "failed")
                log(f"moving failed active install to {failed_dir}")
                os.replace(install_dir, failed_dir)
            elif staging_dir.exists():
                failed_dir = staging_dir
                staging_dir = None
                log(f"retaining failed staging tree at {failed_dir}")

            if rollback_dir and rollback_dir.exists():
                log(f"restoring rollback path {rollback_dir} -> {install_dir}")
                os.replace(rollback_dir, install_dir)
                log(f"restored previous install from rollback path {rollback_dir.name}")
                raise PrebuiltFallback(
                    "staged prebuilt validation passed but activation failed; restored previous install "
                    f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                ) from exc
        except PrebuiltFallback:
            raise
        except Exception as rollback_exc:
            log(f"rollback after failed activation also failed: {rollback_exc}")

        log(
            "rollback restoration failed; cleaning staging, install, and rollback paths before source build fallback"
        )
        cleanup_error: Exception | None = None
        try:
            cleanup_install_side_paths(
                install_dir,
                staging_dir = staging_dir,
                rollback_dir = rollback_dir,
                failed_dir = failed_dir,
                active_dir = install_dir,
            )
        except Exception as cleanup_exc:
            cleanup_error = cleanup_exc
            log(f"cleanup after rollback failure also failed: {cleanup_exc}")
        details = textwrap.shorten(str(exc), width = 200, placeholder = "...")
        if cleanup_error is not None:
            raise PrebuiltFallback(
                "staged prebuilt validation passed but activation and rollback failed; "
                f"cleanup also reported errors ({details}; cleanup={cleanup_error})"
            ) from exc
        raise PrebuiltFallback(
            "staged prebuilt validation passed but activation and rollback failed; "
            f"cleaned install state for fresh source build ({details})"
        ) from exc
    else:
        if rollback_dir:
            remove_tree_logged(rollback_dir, "rollback path")
    finally:
        remove_tree(failed_dir)
        remove_tree(staging_dir)
        prune_install_staging_root(install_dir)


def install_from_archives(
    choice: AssetChoice, host: HostInfo, install_dir: Path, work_dir: Path
) -> tuple[Path, Path]:
    main_archive = work_dir / choice.name
    log(f"downloading {choice.name} from {choice.source_label} release")
    if not choice.expected_sha256:
        raise PrebuiltFallback(
            f"approved checksum was missing for selected asset {choice.name}"
        )
    download_file_verified(
        choice.url,
        main_archive,
        expected_sha256 = choice.expected_sha256,
        label = f"prebuilt archive {choice.name}",
    )

    install_dir.mkdir(parents = True, exist_ok = True)
    extract_dir = Path(tempfile.mkdtemp(prefix = "extract-", dir = work_dir))

    try:
        extract_archive(main_archive, extract_dir)
        source_dir = extract_dir
        overlay_dir = overlay_directory_for_choice(install_dir, choice, host)
        copy_globs(
            source_dir, overlay_dir, runtime_patterns_for_choice(choice), required = True
        )
        copy_globs(
            source_dir,
            install_dir,
            metadata_patterns_for_choice(choice),
            required = False,
        )
    finally:
        remove_tree(extract_dir)

    if host.is_windows:
        exec_dir = install_dir / "build" / "bin" / "Release"
        server_src = next(exec_dir.glob("llama-server.exe"), None)
        quantize_src = next(exec_dir.glob("llama-quantize.exe"), None)
        if server_src is None or quantize_src is None:
            raise PrebuiltFallback("windows executables were not installed correctly")
        return server_src, quantize_src

    build_bin = install_dir / "build" / "bin"
    source_server = build_bin / "llama-server"
    source_quantize = build_bin / "llama-quantize"
    if not source_server.exists() or not source_quantize.exists():
        raise PrebuiltFallback(
            "unix executables were not installed correctly into build/bin"
        )
    os.chmod(source_server, 0o755)
    os.chmod(source_quantize, 0o755)

    root_server = install_dir / "llama-server"
    root_quantize = install_dir / "llama-quantize"
    if source_server != root_server:
        create_exec_entrypoint(root_server, source_server)
    if source_quantize != root_quantize:
        create_exec_entrypoint(root_quantize, source_quantize)
    build_server = build_bin / "llama-server"
    build_quantize = build_bin / "llama-quantize"
    if source_server != build_server:
        create_exec_entrypoint(build_server, source_server)
    if source_quantize != build_quantize:
        create_exec_entrypoint(build_quantize, source_quantize)

    return source_server, source_quantize


def ensure_repo_shape(install_dir: Path) -> None:
    required = [
        install_dir / "CMakeLists.txt",
        install_dir / "convert_hf_to_gguf.py",
        install_dir / "gguf-py",
    ]
    missing = [
        str(path.relative_to(install_dir)) for path in required if not path.exists()
    ]
    if missing:
        raise PrebuiltFallback(
            "hydrated llama.cpp source tree was missing: " + ", ".join(missing)
        )


def validation_model_cache_path(install_dir: Path) -> Path:
    cache_dir = install_dir.parent / VALIDATION_MODEL_CACHE_DIRNAME
    cache_dir.mkdir(parents = True, exist_ok = True)
    return cache_dir / VALIDATION_MODEL_CACHE_FILENAME


def validated_validation_model_bytes(data: bytes) -> bytes:
    if not data:
        raise RuntimeError(f"downloaded empty validation model from {TEST_MODEL_URL}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != TEST_MODEL_SHA256:
        raise RuntimeError(
            "validation model checksum mismatch: "
            f"expected={TEST_MODEL_SHA256} actual={digest}"
        )
    return data


def download_validation_model(path: Path, cache_path: Path | None = None) -> None:
    try:
        data: bytes | None = None
        if cache_path and cache_path.exists():
            try:
                data = validated_validation_model_bytes(cache_path.read_bytes())
                log(f"using cached tiny GGUF validation model from {cache_path}")
            except Exception as exc:
                log(
                    f"cached tiny GGUF validation model was invalid; refreshing cache ({exc})"
                )
                data = None
        if data is None:
            log("downloading tiny GGUF validation model")
            data = validated_validation_model_bytes(
                download_bytes(
                    TEST_MODEL_URL,
                    progress_label = f"Downloading {download_label_from_url(TEST_MODEL_URL)}",
                )
            )
            if cache_path is not None:
                atomic_write_bytes(cache_path, data)
        atomic_write_bytes(path, data)
    except Exception as exc:
        raise PrebuiltFallback(f"validation model unavailable: {exc}") from exc


def free_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def read_log_excerpt(log_path: Path, *, max_lines: int = 60) -> str:
    try:
        content = log_path.read_text(encoding = "utf-8", errors = "replace")
    except FileNotFoundError:
        return ""
    return "\n".join(content.splitlines()[-max_lines:])


def is_retryable_server_bind_error(
    exc: Exception | None,
    output: str = "",
    *,
    exited_quickly: bool = False,
) -> bool:
    haystack = output.lower()
    bind_markers = (
        "address already in use",
        "only one usage of each socket address",
        "failed to bind",
        "bind failed",
        "failed to listen",
        "errno 98",
        "errno 10048",
    )
    if any(marker in haystack for marker in bind_markers):
        return True

    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if exited_quickly and isinstance(reason, ConnectionRefusedError):
            return True
        if isinstance(reason, OSError) and reason.errno in {
            98,
            99,
            111,
            10048,
            10049,
            10061,
        }:
            return exited_quickly
    if exited_quickly and isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, OSError) and exc.errno in {98, 99, 111, 10048, 10049, 10061}:
        return exited_quickly
    return False


def dedupe_existing_dirs(paths: Iterable[str | Path]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def linux_missing_libraries(
    binary_path: Path, *, env: dict[str, str] | None = None
) -> list[str]:
    try:
        result = run_capture(["ldd", str(binary_path)], timeout = 20, env = env)
    except Exception:
        return []

    missing: list[str] = []
    for line in (result.stdout + result.stderr).splitlines():
        line = line.strip()
        if "=> not found" not in line:
            continue
        library = line.split("=>", 1)[0].strip()
        if library and library not in missing:
            missing.append(library)
    return missing


def python_runtime_dirs() -> list[str]:
    candidates: list[Path] = []
    search_roots = [Path(entry) for entry in sys.path if entry]
    try:
        search_roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            search_roots.append(Path(user_site))
    except Exception:
        pass

    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob("nvidia/*/lib"))
        candidates.extend(root.glob("nvidia/*/bin"))
        candidates.extend(root.glob("torch/lib"))
    return dedupe_existing_dirs(candidates)


def ldconfig_runtime_dirs(required_libraries: Iterable[str]) -> list[str]:
    try:
        result = run_capture(["ldconfig", "-p"], timeout = 20)
    except Exception:
        return []

    required = set(required_libraries)
    candidates: list[str] = []
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        library, _, location = line.partition("=>")
        library = library.strip().split()[0]
        if required and library not in required:
            continue
        path = Path(location.strip()).parent
        candidates.append(str(path))
    return dedupe_existing_dirs(candidates)


def linux_runtime_dirs(binary_path: Path) -> list[str]:
    missing = linux_missing_libraries(binary_path)
    if not missing:
        return []
    return linux_runtime_dirs_for_required_libraries(missing)


def preflight_linux_installed_binaries(
    binaries: Iterable[Path],
    install_dir: Path,
    host: HostInfo,
) -> None:
    if not host.is_linux:
        return

    issues: list[str] = []
    for binary_path in binaries:
        env = binary_env(binary_path, install_dir, host)
        missing = linux_missing_libraries(binary_path, env = env)
        if not missing:
            continue
        runtime_dirs = [
            part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part
        ]
        issues.append(
            f"{binary_path.name}: missing={','.join(missing)} "
            f"ld_library_path={','.join(runtime_dirs) if runtime_dirs else 'none'}"
        )

    if issues:
        raise PrebuiltFallback(
            "linux extracted binary preflight failed:\n" + "\n".join(issues)
        )


def glob_paths(*patterns: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?[]"):
            matches.extend(str(path) for path in Path("/").glob(pattern.lstrip("/")))
        else:
            matches.append(pattern)
    return matches


def windows_runtime_dirs() -> list[str]:
    candidates: list[str | Path] = []

    env_dirs = os.environ.get("CUDA_RUNTIME_DLL_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)

    path_dirs = os.environ.get("PATH", "")
    if path_dirs:
        candidates.extend(part for part in path_dirs.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))

    for root in cuda_roots:
        candidates.extend([root / "bin", root / "lib" / "x64"])

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    toolkit_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_base.is_dir():
        candidates.extend(toolkit_base.glob("v*/bin"))
        candidates.extend(toolkit_base.glob("v*/lib/x64"))

    candidates.extend(Path(path) for path in python_runtime_dirs())
    return dedupe_existing_dirs(candidates)


def windows_runtime_dirs_for_patterns(
    required_patterns: Iterable[str],
    candidate_dirs: Iterable[str] | None = None,
) -> list[str]:
    directories = (
        list(candidate_dirs) if candidate_dirs is not None else windows_runtime_dirs()
    )
    matching_dirs: list[str] = []
    for pattern in required_patterns:
        matched_dirs = [
            directory for directory in directories if any(Path(directory).glob(pattern))
        ]
        if not matched_dirs:
            return []
        for directory in matched_dirs:
            if directory not in matching_dirs:
                matching_dirs.append(directory)
    return matching_dirs


def windows_runtime_dirs_for_runtime_line(runtime_line: str | None) -> list[str]:
    if not runtime_line:
        return []
    patterns = windows_runtime_line_info().get(runtime_line)
    if not patterns:
        return []
    return windows_runtime_dirs_for_patterns(patterns)


def binary_env(
    binary_path: Path,
    install_dir: Path,
    host: HostInfo,
    *,
    runtime_line: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if host.is_windows:
        path_dirs = [
            str(binary_path.parent),
            *windows_runtime_dirs_for_runtime_line(runtime_line),
        ]
        existing = [part for part in env.get("PATH", "").split(os.pathsep) if part]
        env["PATH"] = os.pathsep.join(dedupe_existing_dirs([*path_dirs, *existing]))
    elif host.is_linux:
        ld_dirs = [
            str(binary_path.parent),
            str(install_dir),
            *linux_runtime_dirs(binary_path),
        ]
        existing = [
            part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dedupe_existing_dirs([*ld_dirs, *existing])
        )
    elif host.is_macos:
        dyld_dirs = [str(binary_path.parent), str(install_dir)]
        existing = [
            part for part in env.get("DYLD_LIBRARY_PATH", "").split(os.pathsep) if part
        ]
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            dedupe_existing_dirs([*dyld_dirs, *existing])
        )
    return env


def validate_quantize(
    quantize_path: Path,
    probe_path: Path,
    quantized_path: Path,
    install_dir: Path,
    host: HostInfo,
    *,
    runtime_line: str | None = None,
) -> None:
    command = [str(quantize_path), str(probe_path), str(quantized_path), "Q6_K", "2"]
    result = subprocess.run(
        command,
        capture_output = True,
        text = True,
        timeout = 120,
        env = binary_env(quantize_path, install_dir, host, runtime_line = runtime_line),
    )
    if (
        result.returncode != 0
        or not quantized_path.exists()
        or quantized_path.stat().st_size == 0
    ):
        raise PrebuiltFallback(
            "llama-quantize validation failed:\n"
            + result.stdout
            + ("\n" + result.stderr if result.stderr else "")
        )


def validate_server(
    server_path: Path,
    probe_path: Path,
    host: HostInfo,
    install_dir: Path,
    *,
    runtime_line: str | None = None,
) -> None:
    last_failure: PrebuiltFallback | None = None
    for port_attempt in range(1, SERVER_PORT_BIND_ATTEMPTS + 1):
        port = free_local_port()
        command = [
            str(server_path),
            "-m",
            str(probe_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "32",
            "--parallel",
            "1",
            "--threads",
            "1",
            "--ubatch-size",
            "32",
            "--batch-size",
            "32",
        ]
        if host.has_usable_nvidia or (host.is_macos and host.is_arm64):
            command.extend(["--n-gpu-layers", "1"])

        log_fd, log_name = tempfile.mkstemp(prefix = "llama-server-", suffix = ".log")
        os.close(log_fd)
        log_path = Path(log_name)
        process: subprocess.Popen[str] | None = None
        try:
            with log_path.open("w", encoding = "utf-8", errors = "replace") as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout = log_handle,
                    stderr = subprocess.STDOUT,
                    text = True,
                    env = binary_env(
                        server_path, install_dir, host, runtime_line = runtime_line
                    ),
                )
                deadline = time.time() + 20
                startup_started = time.time()
                response_body = ""
                last_error: Exception | None = None
                while time.time() < deadline:
                    if process.poll() is not None:
                        process.wait(timeout = 5)
                        log_handle.flush()
                        output = read_log_excerpt(log_path)
                        exited_quickly = (
                            time.time() - startup_started
                        ) <= SERVER_BIND_RETRY_WINDOW_SECONDS
                        failure = PrebuiltFallback(
                            "llama-server exited during startup:\n" + output
                        )
                        if (
                            port_attempt < SERVER_PORT_BIND_ATTEMPTS
                            and is_retryable_server_bind_error(
                                last_error,
                                output,
                                exited_quickly = exited_quickly,
                            )
                        ):
                            log(
                                f"llama-server startup hit a port race on {port}; retrying with a fresh port "
                                f"({port_attempt}/{SERVER_PORT_BIND_ATTEMPTS})"
                            )
                            last_failure = failure
                            break
                        raise failure

                    payload = json.dumps({"prompt": "a", "n_predict": 1}).encode(
                        "utf-8"
                    )
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/completion",
                        data = payload,
                        headers = {"Content-Type": "application/json"},
                    )
                    try:
                        with urllib.request.urlopen(request, timeout = 5) as response:
                            status_code = response.status
                            response_body = response.read().decode("utf-8", "replace")
                            if status_code == 200:
                                return
                            last_error = RuntimeError(
                                f"unexpected HTTP status {status_code}"
                            )
                    except urllib.error.HTTPError as exc:
                        response_body = exc.read().decode("utf-8", "replace")
                        last_error = exc
                    except Exception as exc:
                        last_error = exc
                    time.sleep(0.5)
                else:
                    log_handle.flush()
                    output = read_log_excerpt(log_path)
                    raise PrebuiltFallback(
                        "llama-server completion validation timed out"
                        + (f" ({last_error})" if last_error else "")
                        + ":\n"
                        + output
                        + ("\n" + response_body if response_body else "")
                    )
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout = 5)
            try:
                log_path.unlink(missing_ok = True)
            except Exception:
                pass
    if last_failure is not None:
        raise last_failure
    raise PrebuiltFallback("llama-server validation failed unexpectedly")


def collect_system_report(
    host: HostInfo, choice: AssetChoice | None, install_dir: Path
) -> str:
    lines = [
        f"platform={host.system} machine={host.machine}",
        f"driver_cuda_version={host.driver_cuda_version}",
        f"compute_caps={','.join(host.compute_caps) if host.compute_caps else 'unknown'}",
        f"cuda_visible_devices={host.visible_cuda_devices if host.visible_cuda_devices is not None else 'unset'}",
        f"has_physical_nvidia={host.has_physical_nvidia}",
        f"has_usable_nvidia={host.has_usable_nvidia}",
        f"chosen_asset={(choice.name if choice else 'none')}",
        f"asset_source={(choice.source_label if choice else 'none')}",
    ]
    if host.is_linux and host.has_physical_nvidia:
        runtime_lines, runtime_dirs = detected_linux_runtime_lines()
        lines.append(
            "linux_runtime_lines="
            + (",".join(runtime_lines) if runtime_lines else "none")
        )
        for runtime_line in ("cuda13", "cuda12"):
            lines.append(
                f"linux_runtime_dirs_{runtime_line}="
                + (
                    ",".join(runtime_dirs.get(runtime_line, []))
                    if runtime_dirs.get(runtime_line)
                    else "none"
                )
            )
    if choice and choice.selection_log:
        lines.append("selection_log:")
        lines.extend(choice.selection_log)
    if host.nvidia_smi:
        try:
            smi = run_capture([host.nvidia_smi], timeout = 20)
            excerpt = "\n".join((smi.stdout + smi.stderr).splitlines()[:20])
            lines.append("nvidia-smi:")
            lines.append(excerpt)
        except Exception as exc:
            lines.append(f"nvidia-smi error: {exc}")

    if host.is_linux:
        server_binary = install_dir / "llama-server"
        if server_binary.exists():
            server_env = binary_env(server_binary, install_dir, host)
            lines.append(
                "linux_missing_libs="
                + (
                    ",".join(linux_missing_libraries(server_binary, env = server_env))
                    or "none"
                )
            )
            lines.append(
                "linux_runtime_dirs="
                + (
                    ",".join(
                        [
                            part
                            for part in server_env.get("LD_LIBRARY_PATH", "").split(
                                os.pathsep
                            )
                            if part
                        ]
                    )
                    or "none"
                )
            )
            try:
                ldd = run_capture(
                    ["ldd", str(server_binary)], timeout = 20, env = server_env
                )
                lines.append("ldd llama-server:")
                lines.append((ldd.stdout + ldd.stderr).strip())
            except Exception as exc:
                lines.append(f"ldd error: {exc}")
    elif host.is_windows:
        lines.append(
            "windows_runtime_dirs=" + (",".join(windows_runtime_dirs()) or "none")
        )
        runtime_lines, runtime_dirs = detected_windows_runtime_lines()
        lines.append(
            "windows_runtime_lines="
            + (",".join(runtime_lines) if runtime_lines else "none")
        )
        for runtime_line in ("cuda13", "cuda12"):
            lines.append(
                f"windows_runtime_dirs_{runtime_line}="
                + (
                    ",".join(runtime_dirs.get(runtime_line, []))
                    if runtime_dirs.get(runtime_line)
                    else "none"
                )
            )
    elif host.is_macos:
        server_binary = install_dir / "llama-server"
        if server_binary.exists():
            try:
                otool = run_capture(["otool", "-L", str(server_binary)], timeout = 20)
                lines.append("otool -L llama-server:")
                lines.append((otool.stdout + otool.stderr).strip())
            except Exception as exc:
                lines.append(f"otool error: {exc}")

    return "\n".join(lines)


def apply_approved_hashes(
    attempts: Iterable[AssetChoice],
    checksums: ApprovedReleaseChecksums,
) -> list[AssetChoice]:
    approved_attempts: list[AssetChoice] = []
    missing_assets: list[str] = []
    for attempt in attempts:
        approved = checksums.artifacts.get(attempt.name)
        if approved is None:
            missing_assets.append(attempt.name)
            continue
        attempt.expected_sha256 = approved.sha256
        approved_attempts.append(attempt)
    if not approved_attempts:
        missing_text = ", ".join(missing_assets) if missing_assets else "none"
        raise PrebuiltFallback(
            "approved checksum asset did not contain the selected prebuilt archive(s): "
            f"{missing_text}"
        )
    return approved_attempts


def require_approved_source_hash(
    checksums: ApprovedReleaseChecksums, llama_tag: str
) -> ApprovedArtifactHash:
    source_asset_name = source_archive_logical_name(llama_tag)
    approved_source = checksums.artifacts.get(source_asset_name)
    if approved_source is None:
        raise PrebuiltFallback(
            f"approved checksum asset did not contain source archive {source_asset_name}"
        )
    return approved_source


def resolve_install_attempts(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
) -> tuple[str, str, list[AssetChoice], ApprovedReleaseChecksums]:
    requested_tag = llama_tag
    resolved_tag = resolve_requested_install_tag(llama_tag, published_release_tag)
    checksums = load_approved_release_checksums(published_repo, resolved_tag)
    require_approved_source_hash(checksums, resolved_tag)

    if host.is_linux and host.is_x86_64 and host.has_usable_nvidia:
        linux_cuda_selection = resolve_linux_cuda_choice(
            host, resolved_tag, published_repo, published_release_tag
        )
        attempts = apply_approved_hashes(linux_cuda_selection.attempts, checksums)
        if not attempts:
            raise PrebuiltFallback("no compatible Linux CUDA asset was found")
        log_lines(linux_cuda_selection.selection_log)
        return requested_tag, resolved_tag, attempts, checksums

    if host.is_windows and host.is_x86_64 and host.has_usable_nvidia:
        upstream_assets = github_release_assets(UPSTREAM_REPO, resolved_tag)
        attempts = apply_approved_hashes(
            resolve_windows_cuda_choices(host, resolved_tag, upstream_assets), checksums
        )
        if not attempts:
            raise PrebuiltFallback("no compatible Windows CUDA asset was found")
        if attempts[0].selection_log:
            log_lines(attempts[0].selection_log)
        return requested_tag, resolved_tag, attempts, checksums

    choice = resolve_asset_choice(
        host, resolved_tag, published_repo, published_release_tag
    )
    approved_attempts = apply_approved_hashes([choice], checksums)
    if choice.selection_log:
        log_lines(choice.selection_log)
    return requested_tag, resolved_tag, approved_attempts, checksums


def write_prebuilt_metadata(
    install_dir: Path,
    *,
    requested_tag: str,
    llama_tag: str,
    choice: AssetChoice,
    prebuilt_fallback_used: bool,
) -> None:
    metadata = {
        "requested_tag": requested_tag,
        "tag": llama_tag,
        "asset": choice.name,
        "source": choice.source_label,
        "bundle_profile": choice.bundle_profile,
        "runtime_line": choice.runtime_line,
        "coverage_class": choice.coverage_class,
        "prebuilt_fallback_used": prebuilt_fallback_used,
        "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(metadata, indent = 2) + "\n"
    )


def validate_prebuilt_choice(
    choice: AssetChoice,
    host: HostInfo,
    install_dir: Path,
    work_dir: Path,
    probe_path: Path,
    *,
    requested_tag: str,
    llama_tag: str,
    approved_checksums: ApprovedReleaseChecksums,
    prebuilt_fallback_used: bool,
    quantized_path: Path,
) -> tuple[Path, Path]:
    source_archive = approved_checksums.artifacts.get(
        source_archive_logical_name(llama_tag)
    )
    if source_archive is None:
        raise PrebuiltFallback(
            f"approved checksum asset did not contain source archive {source_archive_logical_name(llama_tag)}"
        )
    log(f"hydrating upstream llama.cpp source for {llama_tag} into {install_dir}")
    hydrate_source_tree(
        llama_tag,
        install_dir,
        work_dir,
        expected_sha256 = source_archive.sha256,
    )
    log(f"overlaying prebuilt bundle {choice.name} into {install_dir}")
    server_path, quantize_path = install_from_archives(
        choice, host, install_dir, work_dir
    )
    preflight_linux_installed_binaries((server_path, quantize_path), install_dir, host)
    ensure_repo_shape(install_dir)
    write_prebuilt_metadata(
        install_dir,
        requested_tag = requested_tag,
        llama_tag = llama_tag,
        choice = choice,
        prebuilt_fallback_used = prebuilt_fallback_used,
    )
    validate_quantize(
        quantize_path,
        probe_path,
        quantized_path,
        install_dir,
        host,
        runtime_line = choice.runtime_line,
    )
    validate_server(
        server_path,
        probe_path,
        host,
        install_dir,
        runtime_line = choice.runtime_line,
    )
    log(f"staged prebuilt validation succeeded for {choice.name}")
    return server_path, quantize_path


def validate_prebuilt_attempts(
    attempts: Iterable[AssetChoice],
    host: HostInfo,
    install_dir: Path,
    work_dir: Path,
    probe_path: Path,
    *,
    requested_tag: str,
    llama_tag: str,
    approved_checksums: ApprovedReleaseChecksums,
) -> tuple[AssetChoice, Path, bool]:
    attempt_list = list(attempts)
    if not attempt_list:
        raise PrebuiltFallback("no prebuilt bundle attempts were available")

    tried_fallback = False
    for index, attempt in enumerate(attempt_list):
        if index > 0:
            tried_fallback = True
            log(
                "retrying CUDA prebuilt "
                f"{attempt.name} install_kind={attempt.install_kind} "
                f"runtime_line={attempt.runtime_line} coverage_class={attempt.coverage_class}"
            )

        staging_dir = create_install_staging_dir(install_dir)
        quantized_path = work_dir / f"stories260K-q4-{index}.gguf"
        if quantized_path.exists():
            quantized_path.unlink()
        try:
            validate_prebuilt_choice(
                attempt,
                host,
                staging_dir,
                work_dir,
                probe_path,
                requested_tag = requested_tag,
                llama_tag = llama_tag,
                approved_checksums = approved_checksums,
                prebuilt_fallback_used = tried_fallback,
                quantized_path = quantized_path,
            )
        except Exception as exc:
            remove_tree(staging_dir)
            prune_install_staging_root(install_dir)
            if isinstance(exc, PrebuiltFallback):
                attempt_error = exc
            else:
                attempt_error = PrebuiltFallback(
                    f"candidate attempt failed before activation for {attempt.name}: {exc}"
                )
            if index == len(attempt_list) - 1:
                raise attempt_error from exc
            log(
                "selected CUDA bundle failed before activation; trying next prebuilt fallback "
                f"({textwrap.shorten(str(attempt_error), width = 200, placeholder = '...')})"
            )
            continue

        return attempt, staging_dir, tried_fallback

    raise PrebuiltFallback("no prebuilt bundle passed validation")


def install_prebuilt(
    install_dir: Path, llama_tag: str, published_repo: str, published_release_tag: str
) -> None:
    host = detect_host()
    choice: AssetChoice | None = None
    try:
        with install_lock(install_lock_path(install_dir)):
            if install_dir.exists():
                log(
                    f"existing llama.cpp install detected at {install_dir}; validating staged prebuilt update before replacement"
                )
            else:
                log(
                    f"no existing llama.cpp install detected at {install_dir}; performing fresh prebuilt install"
                )
            requested_tag, llama_tag, attempts, approved_checksums = (
                resolve_install_attempts(
                    llama_tag,
                    host,
                    published_repo,
                    published_release_tag,
                )
            )
            choice = attempts[0]
            log(
                f"selected {choice.name} ({choice.source_label}) for {host.system} {host.machine}"
            )
            with tempfile.TemporaryDirectory(prefix = "unsloth-llama-prebuilt-") as tmp:
                work_dir = Path(tmp)
                probe_path = work_dir / "stories260K.gguf"
                download_validation_model(
                    probe_path, validation_model_cache_path(install_dir)
                )
                choice, selected_staging_dir, _ = validate_prebuilt_attempts(
                    attempts,
                    host,
                    install_dir,
                    work_dir,
                    probe_path,
                    requested_tag = requested_tag,
                    llama_tag = llama_tag,
                    approved_checksums = approved_checksums,
                )
                activate_install_tree(selected_staging_dir, install_dir, host)
                try:
                    ensure_converter_scripts(install_dir, llama_tag)
                except Exception as exc:
                    log(
                        "converter script fetch failed after activation; install remains valid "
                        f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                    )
    except PrebuiltFallback as exc:
        log("prebuilt install path failed; falling back to source build")
        log(f"prebuilt fallback reason: {exc}")
        report = collect_system_report(host, choice, install_dir)
        print(report)
        raise SystemExit(EXIT_FALLBACK) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Install and validate a prebuilt llama.cpp bundle for Unsloth Studio."
    )
    parser.add_argument("--install-dir", help = "Target ~/.unsloth/llama.cpp directory")
    parser.add_argument(
        "--llama-tag",
        default = DEFAULT_LLAMA_TAG,
        help = f"llama.cpp release tag. Prebuilt installs are pinned to the approved tag {APPROVED_PREBUILT_LLAMA_TAG}.",
    )
    parser.add_argument(
        "--published-repo",
        default = DEFAULT_PUBLISHED_REPO,
        help = "Published bundle repository",
    )
    parser.add_argument(
        "--published-release-tag",
        default = DEFAULT_PUBLISHED_TAG,
        help = "Published GitHub release tag to pin. By default, scan releases until a compatible llama.cpp bundle is found.",
    )
    resolve_group = parser.add_mutually_exclusive_group()
    resolve_group.add_argument(
        "--resolve-llama-tag",
        nargs = "?",
        const = "latest",
        help = "Resolve a llama.cpp tag such as 'latest' to the logical upstream release tag.",
    )
    resolve_group.add_argument(
        "--resolve-install-tag",
        nargs = "?",
        const = "latest",
        help = "Resolve a llama.cpp tag such as 'latest' to the concrete tag installable on the current host.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resolve_llama_tag is not None:
        # Pass published_repo so the resolver prefers the Unsloth release tag
        # (tested/approved) over the upstream ggml-org bleeding-edge tag.
        print(resolve_requested_llama_tag(args.resolve_llama_tag, args.published_repo))
        return EXIT_SUCCESS

    if args.resolve_install_tag is not None:
        print(
            resolve_requested_install_tag(
                args.resolve_install_tag, args.published_release_tag or ""
            )
        )
        return EXIT_SUCCESS

    if not args.install_dir:
        raise SystemExit(
            "install_llama_prebuilt.py: --install-dir is required unless --resolve-llama-tag or --resolve-install-tag is used"
        )
    install_prebuilt(
        install_dir = Path(args.install_dir).expanduser().resolve(),
        llama_tag = args.llama_tag,
        published_repo = args.published_repo,
        published_release_tag = args.published_release_tag or "",
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        message = textwrap.shorten(str(exc), width = 400, placeholder = "...")
        log(f"fatal helper error: {message}")
        raise SystemExit(EXIT_ERROR)
