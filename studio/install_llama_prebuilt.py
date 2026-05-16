#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross platform llama.cpp prebuilt installer for Unsloth Studio"""

from __future__ import annotations

import argparse
import errno
import fnmatch
import hashlib
import json
import os
import platform
import random
import re
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
from dataclasses import dataclass, field

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
EXIT_BUSY = 3


def windows_hidden_subprocess_kwargs() -> dict[str, object]:
    """Return Windows-only subprocess kwargs that suppress console windows."""
    if sys.platform != "win32":
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_factory is not None and startf_use_showwindow:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


# Prefer "latest" over "master" -- "master" bypasses the prebuilt resolver
# (no matching GitHub release), forces a source build, and causes HTTP 422
# errors. Only use "master" temporarily when the latest release is missing
# support for a new model architecture.
DEFAULT_LLAMA_TAG = os.environ.get("UNSLOTH_LLAMA_TAG", "latest")
# Default published repo for prebuilt release resolution. Linux uses
# Unsloth prebuilts; setup.sh/setup.ps1 pass --published-repo explicitly
# for macOS/Windows to override with ggml-org/llama.cpp when needed.
DEFAULT_PUBLISHED_REPO = "unslothai/llama.cpp"
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
JSON_FETCH_ATTEMPTS = 3
DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES = env_int(
    "UNSLOTH_LLAMA_GITHUB_RELEASE_SCAN_MAX_PAGES",
    5,
    minimum = 1,
)
SERVER_PORT_BIND_ATTEMPTS = 3
SERVER_BIND_RETRY_WINDOW_SECONDS = 5.0
TTY_PROGRESS_START_DELAY_SECONDS = 0.5
DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS = env_int(
    "UNSLOTH_LLAMA_MAX_PREBUILT_RELEASE_FALLBACKS",
    2,
    minimum = 1,
)
FORCE_COMPILE_DEFAULT_REF = os.environ.get("UNSLOTH_LLAMA_FORCE_COMPILE_REF", "master")

DIRECT_LINUX_BUNDLE_PROFILES: dict[str, dict[str, Any]] = {
    "cuda12-older": {
        "runtime_line": "cuda12",
        "coverage_class": "older",
        "supported_sms": ["70", "75", "80", "86", "89"],
        "min_sm": 70,
        "max_sm": 89,
        "rank": 10,
    },
    "cuda12-newer": {
        "runtime_line": "cuda12",
        "coverage_class": "newer",
        "supported_sms": ["86", "89", "90", "100", "120"],
        "min_sm": 86,
        "max_sm": 120,
        "rank": 20,
    },
    "cuda12-portable": {
        "runtime_line": "cuda12",
        "coverage_class": "portable",
        "supported_sms": ["70", "75", "80", "86", "89", "90", "100", "120"],
        "min_sm": 70,
        "max_sm": 120,
        "rank": 30,
    },
    "cuda13-older": {
        "runtime_line": "cuda13",
        "coverage_class": "older",
        "supported_sms": ["75", "80", "86", "89"],
        "min_sm": 75,
        "max_sm": 89,
        "rank": 40,
    },
    "cuda13-newer": {
        "runtime_line": "cuda13",
        "coverage_class": "newer",
        "supported_sms": ["86", "89", "90", "100", "120"],
        "min_sm": 86,
        "max_sm": 120,
        "rank": 50,
    },
    "cuda13-portable": {
        "runtime_line": "cuda13",
        "coverage_class": "portable",
        "supported_sms": ["75", "80", "86", "89", "90", "100", "120"],
        "min_sm": 75,
        "max_sm": 120,
        "rank": 60,
    },
}


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
    # Paired runtime archive (Windows CUDA cudart bundle). When set,
    # install_from_archives also downloads it and overlays its DLLs on
    # top of the main install. See unslothai/unsloth#5106.
    runtime_name: str | None = None
    runtime_url: str | None = None
    runtime_sha256: str | None = None
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
    manifest_sha256: str | None = None
    source_repo: str | None = None
    source_repo_url: str | None = None
    source_ref_kind: str | None = None
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    source_commit: str | None = None
    source_commit_short: str | None = None
    assets: dict[str, str] = field(default_factory = dict)
    manifest_asset_name: str = DEFAULT_PUBLISHED_MANIFEST_ASSET
    artifacts: list[PublishedLlamaArtifact] = field(default_factory = list)
    selection_log: list[str] = field(default_factory = list)


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
    source_repo: str | None = None
    source_repo_url: str | None = None
    source_ref_kind: str | None = None
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    source_commit: str | None = None
    source_commit_short: str | None = None
    artifacts: dict[str, ApprovedArtifactHash] = field(default_factory = dict)


@dataclass(frozen = True)
class ResolvedPublishedRelease:
    bundle: PublishedReleaseBundle
    checksums: ApprovedReleaseChecksums


@dataclass(frozen = True)
class SourceBuildPlan:
    source_url: str
    source_ref: str
    source_ref_kind: str
    compatibility_upstream_tag: str
    source_repo: str | None = None
    source_repo_url: str | None = None
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    source_commit: str | None = None


@dataclass(frozen = True)
class InstallReleasePlan:
    requested_tag: str
    llama_tag: str
    release_tag: str
    attempts: list[AssetChoice]
    approved_checksums: ApprovedReleaseChecksums


class PrebuiltFallback(RuntimeError):
    pass


class BusyInstallConflict(RuntimeError):
    pass


class ExistingInstallSatisfied(RuntimeError):
    def __init__(self, choice: AssetChoice, used_fallback: bool):
        super().__init__(f"existing install already matches candidate {choice.name}")
        self.choice = choice
        self.used_fallback = used_fallback


def _os_error_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    if isinstance(exc, OSError):
        for value in (
            getattr(exc, "strerror", None),
            getattr(exc, "filename", None),
            getattr(exc, "filename2", None),
        ):
            if isinstance(value, str) and value:
                messages.append(value)
    text = str(exc)
    if text:
        messages.append(text)
    return [message.lower() for message in messages if message]


def is_busy_lock_error(exc: BaseException) -> bool:
    if isinstance(exc, BusyInstallConflict):
        return True
    if isinstance(exc, OSError):
        if exc.errno in {
            errno.EACCES,
            errno.EBUSY,
            errno.EPERM,
            errno.ETXTBSY,
        }:
            return True
        if getattr(exc, "winerror", None) in {5, 32, 145}:
            return True
    for message in _os_error_messages(exc):
        if any(
            needle in message
            for needle in (
                "access is denied",
                "being used by another process",
                "device or resource busy",
                "permission denied",
                "text file busy",
                "file is in use",
                "process cannot access the file",
                "cannot create a file when that file already exists",
            )
        ):
            return True
    return False


def log(message: str) -> None:
    print(f"[llama-prebuilt] {message}", file = sys.stderr)


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
        # GitHub returns 403 (not the standard 429) when the API rate
        # limit is hit. Anonymous calls share a 60-req/hour bucket per
        # runner IP, which CI fleets can exhaust trivially. Treat 403
        # against api.github.com as retryable so we get one or two
        # backoff cycles before the source-build fallback fires; honour
        # Retry-After / X-RateLimit-Reset in sleep_backoff for accurate
        # waits. Real 403s on other hosts (private artefact downloads,
        # auth failures) stay non-retryable.
        if exc.code == 403:
            return is_github_api_url(getattr(exc, "url", None))
        return exc.code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    return False


_RATE_LIMIT_WAIT_CAP_SECONDS = 60.0


def _http_error_retry_delay(exc: Exception) -> float | None:
    """Extract a recommended wait from rate-limit headers on a 403/429.

    Returns None when no header is present or the indicated wait is
    longer than _RATE_LIMIT_WAIT_CAP_SECONDS (in which case the caller
    should not block on it -- the source-build fallback is faster).
    """
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after and retry_after.strip().isdigit():
        wait = float(retry_after.strip())
        return wait if wait <= _RATE_LIMIT_WAIT_CAP_SECONDS else None
    rate_reset = headers.get("X-RateLimit-Reset")
    if rate_reset and rate_reset.strip().isdigit():
        wait = float(rate_reset.strip()) - time.time()
        if 0.0 < wait <= _RATE_LIMIT_WAIT_CAP_SECONDS:
            return wait + 1.0  # +1s of slack so the bucket is fresh
    return None


def sleep_backoff(
    attempt: int,
    *,
    base_delay: float = HTTP_FETCH_BASE_DELAY_SECONDS,
    exc: Exception | None = None,
) -> None:
    delay = base_delay * (2 ** max(attempt - 1, 0))
    header_delay = _http_error_retry_delay(exc) if exc is not None else None
    if header_delay is not None:
        delay = max(delay, header_delay)
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


def exact_source_archive_logical_name(source_commit: str) -> str:
    return f"llama.cpp-source-commit-{source_commit}.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_sha256_digest(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    lowered = value.lower()
    if lowered.startswith("sha256:"):
        lowered = lowered.split(":", 1)[1]
    if len(lowered) != 64 or any(ch not in "0123456789abcdef" for ch in lowered):
        return None
    return lowered


def normalize_source_ref_kind(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"tag", "branch", "pull", "commit", "custom"}:
        return normalized
    return None


def normalize_source_commit(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) < 7 or len(normalized) > 40:
        return None
    if any(ch not in "0123456789abcdef" for ch in normalized):
        return None
    return normalized


def validate_schema_version(payload: dict[str, Any], *, label: str) -> None:
    schema_version = payload.get("schema_version")
    if schema_version is None:
        return
    try:
        normalized = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} schema_version was not an integer") from exc
    if normalized != 1:
        raise RuntimeError(f"{label} schema_version={normalized} is unsupported")


def repo_slug_from_source(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    normalized = normalized.removesuffix(".git")
    if normalized.startswith("https://github.com/"):
        slug = normalized[len("https://github.com/") :]
    elif normalized.startswith("http://github.com/"):
        slug = normalized[len("http://github.com/") :]
    elif normalized.startswith("git@github.com:"):
        slug = normalized[len("git@github.com:") :]
    else:
        slug = normalized
    slug = slug.strip("/")
    parts = slug.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def source_url_from_repo_slug(repo_slug: str | None) -> str | None:
    if not isinstance(repo_slug, str) or not repo_slug:
        return None
    return f"https://github.com/{repo_slug}"


def source_repo_clone_url(repo: str | None, repo_url: str | None) -> str | None:
    if isinstance(repo_url, str) and repo_url.strip():
        return repo_url.strip().removesuffix(".git")
    return source_url_from_repo_slug(repo_slug_from_source(repo))


def infer_source_ref_kind(ref: str | None) -> str:
    if not isinstance(ref, str):
        return "tag"
    normalized = ref.strip()
    lowered = normalized.lower()
    if not normalized:
        return "tag"
    if lowered.startswith("refs/pull/") or lowered.startswith("pull/"):
        return "pull"
    if (
        lowered.startswith("refs/heads/")
        or lowered in {"main", "master", "head"}
        or lowered.startswith("origin/")
    ):
        return "branch"
    normalized_commit = normalize_source_commit(normalized)
    if normalized_commit is not None:
        return "commit"
    return "tag"


def normalized_ref_aliases(ref: str | None) -> set[str]:
    if not isinstance(ref, str):
        return set()
    normalized = ref.strip()
    if not normalized:
        return set()
    aliases = {normalized}
    lowered = normalized.lower()
    commit = normalize_source_commit(normalized)
    if commit is not None:
        aliases.add(commit)
    if lowered.startswith("refs/heads/"):
        aliases.add(normalized.split("/", 2)[2])
    elif "/" not in normalized and infer_source_ref_kind(normalized) == "branch":
        aliases.add(f"refs/heads/{normalized}")
    if lowered.startswith("refs/pull/"):
        aliases.add(normalized.removeprefix("refs/"))
    elif lowered.startswith("pull/"):
        aliases.add(f"refs/{normalized}")
    return aliases


def refs_match(candidate_ref: str | None, requested_ref: str | None) -> bool:
    candidate_aliases = normalized_ref_aliases(candidate_ref)
    requested_aliases = normalized_ref_aliases(requested_ref)
    if not candidate_aliases or not requested_aliases:
        return False
    if candidate_aliases & requested_aliases:
        return True
    candidate_commit = normalize_source_commit(candidate_ref)
    requested_commit = normalize_source_commit(requested_ref)
    if candidate_commit and requested_commit:
        return candidate_commit.startswith(
            requested_commit
        ) or requested_commit.startswith(candidate_commit)
    return False


def checkout_friendly_ref(ref_kind: str | None, ref: str | None) -> str | None:
    """Normalize a source ref to a form that ``git clone --branch`` accepts.

    Fully qualified branch refs like ``refs/heads/main`` are stripped to
    ``main``; tag refs like ``refs/tags/b8508`` are stripped to ``b8508``.
    Pull refs like ``refs/pull/123/head`` are left as-is since they are
    always fetched explicitly rather than cloned with ``--branch``.
    """
    if not isinstance(ref, str) or not ref:
        return ref
    lowered = ref.lower()
    if ref_kind == "branch" and lowered.startswith("refs/heads/"):
        return ref.split("/", 2)[2]
    if ref_kind == "tag" and lowered.startswith("refs/tags/"):
        return ref.split("/", 2)[2]
    return ref


def windows_cuda_upstream_asset_names(llama_tag: str, runtime: str) -> list[str]:
    return [
        f"llama-{llama_tag}-bin-win-cuda-{runtime}-x64.zip",
        f"cudart-llama-bin-win-cuda-{runtime}-x64.zip",
    ]


def windows_cuda_asset_aliases(
    asset_name: str,
    *,
    compatibility_tag: str | None = None,
) -> list[str]:
    aliases: list[str] = []
    legacy_match = re.fullmatch(
        r"llama-(?P<tag>[^/]+)-bin-win-cuda-(?P<runtime>\d+\.\d+)-x64\.zip",
        asset_name,
    )
    if legacy_match:
        runtime = legacy_match.group("runtime")
        aliases.append(f"cudart-llama-bin-win-cuda-{runtime}-x64.zip")
        if compatibility_tag:
            aliases.append(f"llama-{compatibility_tag}-bin-win-cuda-{runtime}-x64.zip")
        return aliases

    current_match = re.fullmatch(
        r"cudart-llama-bin-win-cuda-(?P<runtime>\d+\.\d+)-x64\.zip",
        asset_name,
    )
    if current_match and compatibility_tag:
        runtime = current_match.group("runtime")
        aliases.append(f"llama-{compatibility_tag}-bin-win-cuda-{runtime}-x64.zip")
    return aliases


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
            sleep_backoff(attempt, exc = exc)
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str) -> Any:
    attempts = JSON_FETCH_ATTEMPTS if is_github_api_url(url) else 1
    last_decode_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = download_bytes(
                url,
                timeout = 30,
                headers = github_api_headers(url)
                if is_github_api_url(url)
                else auth_headers(url),
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and is_github_api_url(url):
                hint = ""
                if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
                    hint = (
                        "; set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits"
                    )
                raise RuntimeError(f"GitHub API returned 403 for {url}{hint}") from exc
            raise
        if not data:
            last_decode_exc = RuntimeError(f"downloaded empty JSON payload from {url}")
        else:
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_decode_exc = RuntimeError(
                    f"downloaded invalid JSON from {url}: {exc}"
                )
            else:
                if not isinstance(payload, dict) and not isinstance(payload, list):
                    raise RuntimeError(
                        f"downloaded unexpected JSON type from {url}: {type(payload).__name__}"
                    )
                return payload
        if attempt >= attempts:
            assert last_decode_exc is not None
            raise last_decode_exc
        log(f"json fetch failed ({attempt}/{attempts}) for {url}; retrying")
        sleep_backoff(attempt)
    assert last_decode_exc is not None
    raise last_decode_exc


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
            sleep_backoff(attempt, exc = exc)
    assert last_exc is not None
    raise last_exc


def download_file_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> None:
    normalized_expected = normalize_sha256_digest(expected_sha256)
    if not normalized_expected:
        download_file(url, destination)
        log(
            f"downloaded {label} without a published sha256; relying on install validation"
        )
        return

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


def commit_source_archive_urls(repo: str, source_commit: str) -> list[str]:
    encoded_commit = urllib.parse.quote(source_commit, safe = "")
    return [
        f"https://codeload.github.com/{repo}/tar.gz/{encoded_commit}",
        f"https://github.com/{repo}/archive/{encoded_commit}.tar.gz",
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


def github_releases(
    repo: str,
    *,
    per_page: int = 100,
    max_pages: int = 0,
) -> list[dict[str, Any]]:
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
        if max_pages > 0 and page > max_pages:
            break
    return releases


def latest_upstream_release_tag() -> str:
    payload = fetch_json(UPSTREAM_RELEASES_API)
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise RuntimeError(
            f"latest release tag was missing from {UPSTREAM_RELEASES_API}"
        )
    return tag


def is_release_tag_like(value: str | None) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"b\d+", value.strip()))


def release_time_sort_key(release: dict[str, Any]) -> tuple[str, int]:
    published_at = release.get("published_at")
    created_at = release.get("created_at")
    release_id = release.get("id")
    timestamp = (
        published_at
        if isinstance(published_at, str) and published_at
        else created_at
        if isinstance(created_at, str) and created_at
        else ""
    )
    try:
        normalized_id = int(release_id)
    except (TypeError, ValueError):
        normalized_id = 0
    return (timestamp, normalized_id)


def iter_release_payloads_by_time(
    repo: str,
    published_release_tag: str = "",
    requested_tag: str = "",
) -> Iterable[dict[str, Any]]:
    if published_release_tag:
        yield github_release(repo, published_release_tag)
        return

    if (
        requested_tag
        and requested_tag != "latest"
        and is_release_tag_like(requested_tag)
    ):
        try:
            yield github_release(repo, requested_tag)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                log(
                    f"release tag {requested_tag} not found in {repo}; scanning recent releases"
                )
            else:
                raise
        except Exception:
            raise

    releases = [
        release
        for release in github_releases(
            repo, max_pages = DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES
        )
        if isinstance(release, dict)
        and not release.get("draft")
        and not release.get("prerelease")
    ]
    releases.sort(key = release_time_sort_key, reverse = True)
    for release in releases:
        yield release


def direct_release_matches_request(
    *, release_tag: str, llama_tag: str, requested_tag: str
) -> bool:
    if requested_tag == "latest":
        return True
    for candidate in (release_tag, llama_tag):
        if refs_match(candidate, requested_tag):
            return True
    return False


def synthetic_checksums_for_release(
    repo: str, release_tag: str, upstream_tag: str
) -> ApprovedReleaseChecksums:
    return ApprovedReleaseChecksums(
        repo = repo,
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        artifacts = {},
    )


def parse_direct_linux_release_bundle(
    repo: str, release: dict[str, Any]
) -> PublishedReleaseBundle | None:
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or not release_tag:
        return None

    assets = release_asset_map(release)
    artifacts: list[PublishedLlamaArtifact] = []
    inferred_labels: list[str] = []

    linux_asset_re = re.compile(
        r"^app-(?P<label>.+)-(?P<target>linux-x64(?:-cpu)?|linux-x64-(?:cuda12|cuda13)-(?:older|newer|portable))\.tar\.gz$"
    )
    for asset_name in sorted(assets):
        match = linux_asset_re.fullmatch(asset_name)
        if not match:
            continue
        inferred_labels.append(match.group("label"))
        target = match.group("target")
        if target in {"linux-x64", "linux-x64-cpu"}:
            artifacts.append(
                PublishedLlamaArtifact(
                    asset_name = asset_name,
                    install_kind = "linux-cpu",
                    runtime_line = None,
                    coverage_class = None,
                    supported_sms = [],
                    min_sm = None,
                    max_sm = None,
                    bundle_profile = None,
                    rank = 1000,
                )
            )
            continue

        bundle_profile = target.removeprefix("linux-x64-")
        profile = DIRECT_LINUX_BUNDLE_PROFILES.get(bundle_profile)
        if profile is None:
            continue
        artifacts.append(
            PublishedLlamaArtifact(
                asset_name = asset_name,
                install_kind = "linux-cuda",
                runtime_line = str(profile["runtime_line"]),
                coverage_class = str(profile["coverage_class"]),
                supported_sms = [str(value) for value in profile["supported_sms"]],
                min_sm = int(profile["min_sm"]),
                max_sm = int(profile["max_sm"]),
                bundle_profile = bundle_profile,
                rank = int(profile["rank"]),
            )
        )

    if not artifacts:
        return None

    upstream_tag = (
        release_tag
        if is_release_tag_like(release_tag)
        else inferred_labels[0]
        if len(set(inferred_labels)) == 1 and inferred_labels
        else release_tag
    )
    selection_log = [
        f"published_release: repo={repo}",
        f"published_release: tag={release_tag}",
        f"published_release: upstream_tag={upstream_tag}",
        "published_release: direct_asset_scan=linux",
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


def direct_linux_release_plan(
    release: dict[str, Any],
    host: HostInfo,
    repo: str,
    requested_tag: str,
) -> InstallReleasePlan | None:
    bundle = parse_direct_linux_release_bundle(repo, release)
    if bundle is None:
        return None
    if not direct_release_matches_request(
        release_tag = bundle.release_tag,
        llama_tag = bundle.upstream_tag,
        requested_tag = requested_tag,
    ):
        return None

    attempts: list[AssetChoice] = []
    if host.has_usable_nvidia:
        selection = linux_cuda_choice_from_release(host, bundle)
        if selection is not None:
            attempts.extend(selection.attempts)
    cpu_choice = published_asset_choice_for_kind(bundle, "linux-cpu")
    if cpu_choice is not None:
        attempts.append(cpu_choice)
    if not attempts:
        raise PrebuiltFallback("no compatible Linux prebuilt asset was found")
    return InstallReleasePlan(
        requested_tag = requested_tag,
        llama_tag = bundle.upstream_tag,
        release_tag = bundle.release_tag,
        attempts = attempts,
        approved_checksums = synthetic_checksums_for_release(
            repo,
            bundle.release_tag,
            bundle.upstream_tag,
        ),
    )


def direct_upstream_release_plan(
    release: dict[str, Any],
    host: HostInfo,
    repo: str,
    requested_tag: str,
) -> InstallReleasePlan | None:
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or not release_tag:
        return None
    if not direct_release_matches_request(
        release_tag = release_tag,
        llama_tag = release_tag,
        requested_tag = requested_tag,
    ):
        return None

    assets = release_asset_map(release)
    attempts: list[AssetChoice] = []
    if host.is_windows and host.is_x86_64:
        if host.has_usable_nvidia:
            torch_preference = detect_torch_cuda_runtime_preference(host)
            attempts.extend(
                windows_cuda_attempts(
                    host,
                    release_tag,
                    assets,
                    torch_preference.runtime_line,
                    torch_preference.selection_log,
                )
            )
        cpu_asset = f"llama-{release_tag}-bin-win-cpu-x64.zip"
        cpu_url = assets.get(cpu_asset)
        if cpu_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = cpu_asset,
                    url = cpu_url,
                    source_label = "upstream",
                    install_kind = "windows-cpu",
                )
            )
    elif host.is_macos and host.is_arm64:
        asset_name = f"llama-{release_tag}-bin-macos-arm64.tar.gz"
        asset_url = assets.get(asset_name)
        if asset_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = asset_name,
                    url = asset_url,
                    source_label = "upstream",
                    install_kind = "macos-arm64",
                )
            )
    elif host.is_macos and host.is_x86_64:
        asset_name = f"llama-{release_tag}-bin-macos-x64.tar.gz"
        asset_url = assets.get(asset_name)
        if asset_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = asset_name,
                    url = asset_url,
                    source_label = "upstream",
                    install_kind = "macos-x64",
                )
            )
    elif host.is_linux and host.is_x86_64 and not host.has_usable_nvidia:
        asset_name = f"llama-{release_tag}-bin-ubuntu-x64.tar.gz"
        asset_url = assets.get(asset_name)
        if asset_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = asset_name,
                    url = asset_url,
                    source_label = "upstream",
                    install_kind = "linux-cpu",
                )
            )
    if not attempts:
        raise PrebuiltFallback("no compatible upstream prebuilt asset was found")
    return InstallReleasePlan(
        requested_tag = requested_tag,
        llama_tag = release_tag,
        release_tag = release_tag,
        attempts = attempts,
        approved_checksums = synthetic_checksums_for_release(
            repo,
            release_tag,
            release_tag,
        ),
    )


def resolve_simple_install_release_plans(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
    *,
    max_release_fallbacks: int = DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS,
) -> tuple[str, list[InstallReleasePlan]]:
    repo = published_repo or DEFAULT_PUBLISHED_REPO
    requested_tag = normalized_requested_llama_tag(llama_tag)
    allow_older_release_fallback = (
        requested_tag == "latest" and not published_release_tag
    )
    release_limit = max(1, max_release_fallbacks)
    plans: list[InstallReleasePlan] = []
    last_error: PrebuiltFallback | None = None

    try:
        releases = iter_release_payloads_by_time(
            repo, published_release_tag, requested_tag
        )
        for release in releases:
            try:
                if host.is_linux and repo == "unslothai/llama.cpp":
                    plan = direct_linux_release_plan(release, host, repo, requested_tag)
                else:
                    plan = direct_upstream_release_plan(
                        release, host, repo, requested_tag
                    )
                if plan is None:
                    continue
            except PrebuiltFallback as exc:
                last_error = exc
                if not allow_older_release_fallback:
                    raise
                release_tag = release.get("tag_name") or "unknown"
                log(
                    "published release skipped for install planning: "
                    f"{repo}@{release_tag} ({exc})"
                )
                continue

            plans.append(plan)
            if not allow_older_release_fallback or len(plans) >= release_limit:
                break
    except PrebuiltFallback:
        raise
    except Exception as exc:
        raise PrebuiltFallback(
            f"failed to inspect published releases in {repo}: {exc}"
        ) from exc

    if plans:
        return requested_tag, plans
    if last_error is not None:
        raise last_error
    raise PrebuiltFallback(
        f"no installable published llama.cpp releases were found in {repo}"
    )


def normalized_requested_llama_tag(requested_tag: str | None) -> str:
    if isinstance(requested_tag, str):
        normalized = requested_tag.strip()
        if normalized:
            return normalized
    return "latest"


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
    manifest_bytes = download_bytes(
        manifest_url,
        timeout = 30,
        headers = auth_headers(manifest_url),
    )
    manifest_sha256 = sha256_bytes(manifest_bytes)
    try:
        manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} was not valid JSON"
        ) from exc
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(
            f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} was not a JSON object"
        )
    validate_schema_version(
        manifest_payload,
        label = f"published manifest {DEFAULT_PUBLISHED_MANIFEST_ASSET} in {repo}@{release_tag}",
    )
    component = manifest_payload.get("component")
    upstream_tag = manifest_payload.get("upstream_tag")
    source_repo = manifest_payload.get("source_repo")
    source_repo_url = manifest_payload.get("source_repo_url")
    source_ref_kind = normalize_source_ref_kind(manifest_payload.get("source_ref_kind"))
    requested_source_ref = manifest_payload.get("requested_source_ref")
    resolved_source_ref = manifest_payload.get("resolved_source_ref")
    source_commit = normalize_source_commit(manifest_payload.get("source_commit"))
    source_commit_short = manifest_payload.get("source_commit_short")
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
    if isinstance(source_repo, str) and source_repo:
        selection_log.append(f"published_release: source_repo={source_repo}")
    if source_commit:
        selection_log.append(f"published_release: source_commit={source_commit}")
    return PublishedReleaseBundle(
        repo = repo,
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        manifest_sha256 = manifest_sha256,
        source_repo = source_repo
        if isinstance(source_repo, str) and source_repo
        else None,
        source_repo_url = source_repo_url
        if isinstance(source_repo_url, str) and source_repo_url
        else None,
        source_ref_kind = source_ref_kind,
        requested_source_ref = requested_source_ref
        if isinstance(requested_source_ref, str) and requested_source_ref
        else None,
        resolved_source_ref = resolved_source_ref
        if isinstance(resolved_source_ref, str) and resolved_source_ref
        else None,
        source_commit = source_commit,
        source_commit_short = source_commit_short
        if isinstance(source_commit_short, str) and source_commit_short
        else None,
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
    validate_schema_version(
        payload,
        label = f"published checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET}",
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

    source_commit = normalize_source_commit(payload.get("source_commit"))
    source_commit_short = payload.get("source_commit_short")
    source_repo = payload.get("source_repo")
    source_repo_url = payload.get("source_repo_url")
    source_ref_kind = normalize_source_ref_kind(payload.get("source_ref_kind"))
    requested_source_ref = payload.get("requested_source_ref")
    resolved_source_ref = payload.get("resolved_source_ref")
    return ApprovedReleaseChecksums(
        repo = repo,
        release_tag = release_tag,
        upstream_tag = upstream_tag,
        source_repo = source_repo
        if isinstance(source_repo, str) and source_repo
        else None,
        source_repo_url = source_repo_url
        if isinstance(source_repo_url, str) and source_repo_url
        else None,
        source_ref_kind = source_ref_kind,
        requested_source_ref = requested_source_ref
        if isinstance(requested_source_ref, str) and requested_source_ref
        else None,
        resolved_source_ref = resolved_source_ref
        if isinstance(resolved_source_ref, str) and resolved_source_ref
        else None,
        source_commit = source_commit,
        source_commit_short = source_commit_short
        if isinstance(source_commit_short, str) and source_commit_short
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
        else github_releases(repo, max_pages = DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES)
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
    for release in github_releases(
        UPSTREAM_REPO, max_pages = DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES
    ):
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


def validated_checksums_for_bundle(
    repo: str, bundle: PublishedReleaseBundle
) -> ApprovedReleaseChecksums:
    checksums = load_approved_release_checksums(repo, bundle.release_tag)
    manifest_hash = checksums.artifacts.get(bundle.manifest_asset_name)
    if manifest_hash is not None and bundle.manifest_sha256 is not None:
        if manifest_hash.sha256 != bundle.manifest_sha256:
            raise PrebuiltFallback(
                "published manifest checksum did not match the approved checksum asset"
            )
    # Accept bundles that carry only an exact-commit source archive
    # (e.g. llama.cpp-source-commit-<sha>.tar.gz) without requiring the
    # legacy llama.cpp-source-<upstream_tag>.tar.gz entry.
    if exact_source_archive_hash(checksums) is None:
        require_approved_source_hash(checksums, bundle.upstream_tag)
    return checksums


def published_release_matches_request(
    bundle: PublishedReleaseBundle, requested_ref: str
) -> bool:
    if requested_ref == "latest":
        return True
    for candidate in (
        bundle.upstream_tag,
        bundle.requested_source_ref,
        bundle.resolved_source_ref,
        bundle.source_commit,
    ):
        if refs_match(candidate, requested_ref):
            return True
    return False


def resolve_published_release(
    requested_tag: str | None,
    published_repo: str,
    published_release_tag: str = "",
) -> ResolvedPublishedRelease:
    repo = published_repo or DEFAULT_PUBLISHED_REPO
    normalized_requested = normalized_requested_llama_tag(requested_tag)

    if published_release_tag:
        bundle = pinned_published_release_bundle(repo, published_release_tag)
        if not published_release_matches_request(bundle, normalized_requested):
            raise PrebuiltFallback(
                "published release "
                f"{repo}@{published_release_tag} targeted upstream tag {bundle.upstream_tag}, "
                f"but requested {normalized_requested}"
            )
        return ResolvedPublishedRelease(
            bundle = bundle,
            checksums = validated_checksums_for_bundle(repo, bundle),
        )

    skipped_invalid = 0
    for bundle in iter_published_release_bundles(repo):
        if not published_release_matches_request(bundle, normalized_requested):
            continue
        try:
            checksums = validated_checksums_for_bundle(repo, bundle)
        except PrebuiltFallback as exc:
            skipped_invalid += 1
            log(
                "published release ignored for install resolution: "
                f"{repo}@{bundle.release_tag} ({exc})"
            )
            continue
        return ResolvedPublishedRelease(bundle = bundle, checksums = checksums)

    if normalized_requested == "latest":
        if skipped_invalid:
            raise PrebuiltFallback(
                f"no usable published llama.cpp releases were available in {repo}"
            )
        raise PrebuiltFallback(
            f"no published llama.cpp releases were available in {repo}"
        )

    raise PrebuiltFallback(
        f"no published prebuilt release in {repo} matched upstream tag {normalized_requested}"
    )


def iter_resolved_published_releases(
    requested_tag: str | None,
    published_repo: str,
    published_release_tag: str = "",
) -> Iterable[ResolvedPublishedRelease]:
    repo = published_repo or DEFAULT_PUBLISHED_REPO
    normalized_requested = normalized_requested_llama_tag(requested_tag)

    if published_release_tag:
        bundle = pinned_published_release_bundle(repo, published_release_tag)
        if not published_release_matches_request(bundle, normalized_requested):
            raise PrebuiltFallback(
                "published release "
                f"{repo}@{published_release_tag} targeted upstream tag {bundle.upstream_tag}, "
                f"but requested {normalized_requested}"
            )
        yield ResolvedPublishedRelease(
            bundle = bundle,
            checksums = validated_checksums_for_bundle(repo, bundle),
        )
        return

    matched_any = False
    skipped_invalid = 0
    yielded_valid = False
    for bundle in iter_published_release_bundles(repo):
        if not published_release_matches_request(bundle, normalized_requested):
            continue
        matched_any = True
        try:
            checksums = validated_checksums_for_bundle(repo, bundle)
        except PrebuiltFallback as exc:
            skipped_invalid += 1
            log(
                "published release ignored for install resolution: "
                f"{repo}@{bundle.release_tag} ({exc})"
            )
            continue
        yielded_valid = True
        yield ResolvedPublishedRelease(bundle = bundle, checksums = checksums)

    if yielded_valid:
        return

    if matched_any:
        if skipped_invalid:
            raise PrebuiltFallback(
                f"no usable published llama.cpp releases were available in {repo}"
            )
        return

    if normalized_requested == "latest":
        raise PrebuiltFallback(
            f"no published llama.cpp releases were available in {repo}"
        )

    raise PrebuiltFallback(
        f"no published prebuilt release in {repo} matched upstream tag {normalized_requested}"
    )


def resolve_requested_llama_tag(
    requested_tag: str | None,
    published_repo: str = "",
    published_release_tag: str = "",
) -> str:
    """Resolve a llama.cpp tag for source-build fallback.

    Resolution order:
      1. Concrete tag (e.g. "b8508") -- returned as-is.
      2. "latest" with published_repo -- resolve the latest usable Unsloth
         published release bundle and return its upstream_tag. This is the
         preferred version that matches the published prebuilt metadata.
      3. "latest" without published_repo or if (2) fails -- query the upstream
         ggml-org/llama.cpp repo. This may return a newer, untested tag.

    The Unsloth repo is preferred because its releases are pinned to specific
    upstream tags that have been validated with Unsloth Studio. Using the
    upstream bleeding-edge tag risks API/ABI incompatibilities.
    """
    normalized_requested = normalized_requested_llama_tag(requested_tag)
    if normalized_requested != "latest":
        return normalized_requested
    # Prefer the Unsloth release repo tag (tested/approved) over bleeding-edge
    # upstream. For example, unslothai/llama.cpp may publish b8508 while
    # ggml-org/llama.cpp latest is b8514. The source-build fallback should
    # compile the same version the prebuilt path would have installed.
    if published_repo:
        try:
            return resolve_published_release(
                "latest",
                published_repo,
                published_release_tag,
            ).bundle.upstream_tag
        except Exception:
            pass
    # Fall back to upstream ggml-org latest release tag
    return latest_upstream_release_tag()


def resolve_requested_install_tag(
    requested_tag: str | None,
    published_release_tag: str = "",
    published_repo: str = DEFAULT_PUBLISHED_REPO,
) -> str:
    return resolve_published_release(
        requested_tag,
        published_repo,
        published_release_tag,
    ).bundle.upstream_tag


def exact_source_archive_hash(
    checksums: ApprovedReleaseChecksums,
) -> ApprovedArtifactHash | None:
    if not checksums.source_commit:
        return None
    return checksums.artifacts.get(
        exact_source_archive_logical_name(checksums.source_commit)
    )


def source_clone_url_from_checksums(checksums: ApprovedReleaseChecksums) -> str | None:
    return source_repo_clone_url(checksums.source_repo, checksums.source_repo_url)


def source_build_plan_for_release(
    release: ResolvedPublishedRelease,
) -> SourceBuildPlan:
    checksums = release.checksums
    exact_source = exact_source_archive_hash(checksums)
    source_repo = checksums.source_repo or release.bundle.source_repo
    source_repo_url = checksums.source_repo_url or release.bundle.source_repo_url
    requested_source_ref = (
        checksums.requested_source_ref or release.bundle.requested_source_ref
    )
    resolved_source_ref = (
        checksums.resolved_source_ref or release.bundle.resolved_source_ref
    )
    source_commit = checksums.source_commit or release.bundle.source_commit
    source_ref_kind = checksums.source_ref_kind or release.bundle.source_ref_kind
    source_url = source_repo_clone_url(source_repo, source_repo_url)
    if exact_source is not None and source_url and source_commit:
        return SourceBuildPlan(
            source_url = source_url,
            source_ref = source_commit,
            source_ref_kind = "commit",
            compatibility_upstream_tag = release.bundle.upstream_tag,
            source_repo = source_repo,
            source_repo_url = source_repo_url,
            requested_source_ref = requested_source_ref,
            resolved_source_ref = resolved_source_ref,
            source_commit = source_commit,
        )
    source_ref = checkout_friendly_ref(
        source_ref_kind, resolved_source_ref or requested_source_ref
    )
    if (
        source_url
        and source_ref
        and source_ref_kind in {"tag", "branch", "pull", "commit"}
    ):
        return SourceBuildPlan(
            source_url = source_url,
            source_ref = source_ref,
            source_ref_kind = source_ref_kind,
            compatibility_upstream_tag = release.bundle.upstream_tag,
            source_repo = source_repo,
            source_repo_url = source_repo_url,
            requested_source_ref = requested_source_ref,
            resolved_source_ref = resolved_source_ref,
            source_commit = source_commit,
        )
    return SourceBuildPlan(
        source_url = source_url_from_repo_slug(UPSTREAM_REPO)
        or "https://github.com/ggml-org/llama.cpp",
        source_ref = release.bundle.upstream_tag,
        source_ref_kind = "tag",
        compatibility_upstream_tag = release.bundle.upstream_tag,
        source_repo = source_repo,
        source_repo_url = source_repo_url,
        requested_source_ref = requested_source_ref,
        resolved_source_ref = resolved_source_ref,
        source_commit = source_commit,
    )


def resolve_source_build_plan(
    requested_tag: str | None,
    published_repo: str,
    published_release_tag: str = "",
) -> SourceBuildPlan:
    normalized_requested = normalized_requested_llama_tag(requested_tag)
    if normalized_requested != "latest":
        try:
            release = resolve_published_release(
                normalized_requested,
                published_repo,
                published_release_tag,
            )
            return source_build_plan_for_release(release)
        except Exception:
            pass
        inferred_kind = infer_source_ref_kind(normalized_requested)
        return SourceBuildPlan(
            source_url = "https://github.com/ggml-org/llama.cpp",
            source_ref = checkout_friendly_ref(inferred_kind, normalized_requested)
            or normalized_requested,
            source_ref_kind = inferred_kind,
            compatibility_upstream_tag = normalized_requested,
        )

    if published_repo:
        try:
            release = resolve_published_release(
                "latest",
                published_repo,
                published_release_tag,
            )
            return source_build_plan_for_release(release)
        except Exception:
            pass
    latest_tag = latest_upstream_release_tag()
    return SourceBuildPlan(
        source_url = "https://github.com/ggml-org/llama.cpp",
        source_ref = latest_tag,
        source_ref_kind = "tag",
        compatibility_upstream_tag = latest_tag,
    )


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
        **windows_hidden_subprocess_kwargs(),
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
        # Require `nvidia-smi -L` to actually list a GPU before treating the
        # host as NVIDIA. The banner text "NVIDIA-SMI ..." is printed even
        # when the command fails to communicate with the driver (e.g. stale
        # container leftovers), which would otherwise misclassify an AMD
        # ROCm host as NVIDIA and short-circuit the ROCm path.
        try:
            listing = run_capture([nvidia_smi, "-L"], timeout = 20)
            gpu_lines = [
                line for line in listing.stdout.splitlines() if line.startswith("GPU ")
            ]
            if gpu_lines:
                has_physical_nvidia = True
                has_usable_nvidia = visible_device_tokens != []
        except Exception:
            pass

        try:
            result = run_capture([nvidia_smi], timeout = 20)
            merged = "\n".join(part for part in (result.stdout, result.stderr) if part)
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
                # Older nvidia-smi versions (pre -L support) hit the
                # except in the first try block but still succeed here,
                # leaving has_physical_nvidia unset. Mirror the -L path
                # so downstream diagnostics on line ~4390 still run.
                if not has_physical_nvidia:
                    has_physical_nvidia = True
            elif visible_device_tokens == []:
                has_usable_nvidia = False
            elif supports_explicit_visible_device_matching(visible_device_tokens):
                has_usable_nvidia = False
            elif has_physical_nvidia:
                has_usable_nvidia = True
        except Exception:
            pass

    # Detect AMD ROCm (HIP) -- require actual GPU, not just tools installed

    def _amd_smi_has_gpu(stdout: str) -> bool:
        """Check for 'GPU: <number>' data rows, not just a table header."""
        return bool(re.search(r"(?im)^gpu\s*[:\[]\s*\d", stdout))

    has_rocm = False
    if is_linux:
        for _cmd, _check in (
            # rocminfo: look for a real gfx GPU id (3-4 chars, nonzero first digit).
            # gfx000 is the CPU agent; ROCm 6.1+ also emits generic ISA lines like
            # "gfx11-generic" or "gfx9-4-generic" which only have 1-2 digits before
            # the dash and must not be treated as a real GPU.
            (
                ["rocminfo"],
                lambda out: bool(re.search(r"gfx[1-9][0-9a-z]{2,3}", out.lower())),
            ),
            (["amd-smi", "list"], _amd_smi_has_gpu),
        ):
            _exe = shutil.which(_cmd[0])
            if not _exe:
                continue
            try:
                _result = run_capture([_exe, *_cmd[1:]], timeout = 10)
            except Exception:
                continue
            if _result.returncode == 0 and _result.stdout.strip():
                if _check(_result.stdout):
                    has_rocm = True
                    break
    elif is_windows:
        # Windows: prefer active probes that validate GPU presence
        for _cmd, _check in (
            (["hipinfo"], lambda out: "gcnarchname" in out.lower()),
            (["amd-smi", "list"], _amd_smi_has_gpu),
        ):
            _exe = shutil.which(_cmd[0])
            if not _exe:
                continue
            try:
                _result = run_capture([_exe, *_cmd[1:]], timeout = 10)
            except Exception:
                continue
            if _result.returncode == 0 and _result.stdout.strip():
                if _check(_result.stdout):
                    has_rocm = True
                    break
        # Note: amdhip64.dll presence alone is NOT treated as GPU evidence
        # since the HIP SDK can be installed without an AMD GPU.

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
    if major > 13 or (major == 13):  # and minor >= 1):
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
        selected_name = None
        asset_url = None
        for candidate_name in windows_cuda_upstream_asset_names(llama_tag, runtime):
            asset_url = upstream_assets.get(candidate_name)
            if asset_url:
                selected_name = candidate_name
                break
        if not asset_url or not selected_name:
            selection_log.append(
                "windows_cuda_selection: skip missing assets "
                + ",".join(windows_cuda_upstream_asset_names(llama_tag, runtime))
            )
            continue
        # Pair the cudart bundle when upstream ships it. Without this
        # the binary needs a system CUDA toolkit on PATH at runtime
        # (#5106). Only pair when the selected main archive is the
        # binary archive, not the cudart archive itself.
        runtime_archive_name: str | None = None
        runtime_archive_url: str | None = None
        if selected_name.startswith(f"llama-"):
            cudart_name = f"cudart-llama-bin-win-cuda-{runtime}-x64.zip"
            cudart_url = upstream_assets.get(cudart_name)
            if cudart_url and cudart_url != asset_url:
                runtime_archive_name = cudart_name
                runtime_archive_url = cudart_url
        attempt_log = list(selection_log) + [
            f"windows_cuda_selection: selected {selected_name} runtime={runtime}"
        ]
        if runtime_archive_name:
            attempt_log.append(
                f"windows_cuda_selection: paired runtime archive {runtime_archive_name}"
            )
        else:
            attempt_log.append(
                "windows_cuda_selection: no paired runtime archive found; "
                "binary will rely on a system CUDA toolkit at runtime"
            )
        attempts.append(
            AssetChoice(
                repo = UPSTREAM_REPO,
                tag = llama_tag,
                name = selected_name,
                url = asset_url,
                source_label = "upstream",
                install_kind = "windows-cuda",
                runtime_line = runtime_line,
                runtime_name = runtime_archive_name,
                runtime_url = runtime_archive_url,
                selection_log = attempt_log,
            )
        )
    return attempts


def published_windows_cuda_attempts(
    host: HostInfo,
    release: PublishedReleaseBundle,
    preferred_runtime_line: str | None,
    selection_preamble: Iterable[str] = (),
) -> list[AssetChoice]:
    selection_log = list(release.selection_log) + list(selection_preamble)
    runtime_by_line = {"cuda12": "12.4", "cuda13": "13.1"}
    runtime_order = windows_cuda_attempts(
        host,
        release.upstream_tag,
        {
            f"llama-{release.upstream_tag}-bin-win-cuda-{runtime}-x64.zip": "published"
            for runtime in runtime_by_line.values()
        },
        preferred_runtime_line,
        selection_log,
    )
    published_artifacts = [
        artifact
        for artifact in release.artifacts
        if artifact.install_kind == "windows-cuda"
    ]
    artifacts_by_runtime: dict[str, list[PublishedLlamaArtifact]] = {}
    for artifact in published_artifacts:
        if not artifact.runtime_line:
            continue
        artifacts_by_runtime.setdefault(artifact.runtime_line, []).append(artifact)

    attempts: list[AssetChoice] = []
    for ordered_attempt in runtime_order:
        runtime_line = ordered_attempt.runtime_line
        if not runtime_line:
            continue
        candidates = sorted(
            artifacts_by_runtime.get(runtime_line, []),
            key = lambda artifact: (artifact.rank, artifact.asset_name),
        )
        for artifact in candidates:
            asset_url = release.assets.get(artifact.asset_name)
            if not asset_url:
                continue
            # See windows_cuda_attempts: pair the cudart bundle.
            runtime_archive_name: str | None = None
            runtime_archive_url: str | None = None
            if artifact.asset_name.startswith("llama-"):
                runtime = runtime_by_line[runtime_line]
                cudart_name = f"cudart-llama-bin-win-cuda-{runtime}-x64.zip"
                cudart_url = release.assets.get(cudart_name)
                if cudart_url and cudart_url != asset_url:
                    runtime_archive_name = cudart_name
                    runtime_archive_url = cudart_url
            attempt_log = list(ordered_attempt.selection_log or []) + [
                "windows_cuda_selection: selected published asset "
                f"{artifact.asset_name} for runtime_line={runtime_line}"
            ]
            if runtime_archive_name:
                attempt_log.append(
                    f"windows_cuda_selection: paired published runtime archive {runtime_archive_name}"
                )
            attempts.append(
                AssetChoice(
                    repo = release.repo,
                    tag = release.release_tag,
                    name = artifact.asset_name,
                    url = asset_url,
                    source_label = "published",
                    install_kind = "windows-cuda",
                    runtime_line = runtime_line,
                    runtime_name = runtime_archive_name,
                    runtime_url = runtime_archive_url,
                    selection_log = attempt_log,
                )
            )
            break
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
    host: HostInfo, release: PublishedReleaseBundle
) -> LinuxCudaSelection:
    torch_preference = detect_torch_cuda_runtime_preference(host)
    selection = linux_cuda_choice_from_release(
        host,
        release,
        preferred_runtime_line = torch_preference.runtime_line,
        selection_preamble = torch_preference.selection_log,
    )
    if selection is not None:
        return selection
    raise PrebuiltFallback("no compatible published Linux CUDA bundle was found")


def published_asset_choice_for_kind(
    release: PublishedReleaseBundle,
    install_kind: str,
) -> AssetChoice | None:
    candidates = sorted(
        (
            artifact
            for artifact in release.artifacts
            if artifact.install_kind == install_kind
        ),
        key = lambda artifact: (artifact.rank, artifact.asset_name),
    )
    for artifact in candidates:
        asset_url = release.assets.get(artifact.asset_name)
        if not asset_url:
            continue
        return AssetChoice(
            repo = release.repo,
            tag = release.release_tag,
            name = artifact.asset_name,
            url = asset_url,
            source_label = "published",
            install_kind = install_kind,
            runtime_line = artifact.runtime_line,
            selection_log = list(release.selection_log)
            + [
                f"published_selection: selected {artifact.asset_name} install_kind={install_kind}"
            ],
        )
    return None


def _detect_host_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm runtime, or None.

    Best-effort read from /opt/rocm/.info/version, amd-smi version, and
    hipconfig --version. Used to pick a compatible upstream llama.cpp
    ROCm prebuilt rather than always taking the numerically newest one
    (which can be newer than the host runtime).
    """
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            # Explicit length guard avoids relying on the broad except
            # below to swallow IndexError when the version file contains
            # a single component (e.g. "6\n" on a partial install).
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run(
                [hipconfig, "--version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                raw = (result.stdout or "").strip().split("\n")[0]
                parts = raw.split(".")
                if (
                    len(parts) >= 2
                    and parts[0].isdigit()
                    and parts[1].split("-")[0].isdigit()
                ):
                    return int(parts[0]), int(parts[1].split("-")[0])
        except Exception:
            pass

    # Distro package-manager fallbacks. Mirrors install.sh::get_torch_index_url
    # and _detect_rocm_version() in install_python_stack.py so package-managed
    # ROCm hosts without /opt/rocm/.info/version still report a usable version
    # and the <= host version filter in resolve_upstream_asset_choice picks
    # the correct upstream prebuilt instead of the newest-regardless fallback.
    for _cmd in (
        ["dpkg-query", "-W", "-f=${Version}\n", "rocm-core"],
        ["rpm", "-q", "--qf", "%{VERSION}\n", "rocm-core"],
    ):
        _exe = shutil.which(_cmd[0])
        if not _exe:
            continue
        try:
            _result = subprocess.run(
                [_exe, *_cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            continue
        if _result.returncode != 0 or not _result.stdout.strip():
            continue
        _raw = _result.stdout.strip()
        # dpkg can prepend an epoch ("1:6.3.0-1"); strip it before parsing.
        _raw = re.sub(r"^\d+:", "", _raw)
        _m = re.match(r"(\d+)[.-](\d+)", _raw)
        if _m:
            return int(_m.group(1)), int(_m.group(2))
    return None


def resolve_upstream_asset_choice(host: HostInfo, llama_tag: str) -> AssetChoice:
    upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)
    if host.is_linux and host.is_x86_64:
        # AMD ROCm: try upstream ROCm prebuilt first, then fall back to source build.
        # Source build (via setup.sh) compiles with -DGGML_HIP=ON and auto-detects
        # the exact GPU target via rocminfo, which is more reliable for consumer
        # GPUs (e.g. gfx1151) that may not be in the prebuilt.
        if host.has_rocm and not host.has_usable_nvidia:
            # Scan upstream assets for any rocm-<version> prebuilt. When the
            # host ROCm runtime version is known, pick the newest candidate
            # whose major.minor is <= host version -- otherwise a ROCm 6.4
            # host would download the rocm-7.2 tarball, fail preflight, and
            # fall back to a source build even though a compatible 6.4
            # prebuilt exists. If no compatible candidate matches (e.g. host
            # runtime is older than every published prebuilt), fall back to
            # the numerically newest so we at least try something.
            _rocm_pattern = re.compile(
                rf"llama-{re.escape(llama_tag)}-bin-ubuntu-rocm-([0-9]+(?:\.[0-9]+)*)-x64\.tar\.gz"
            )
            rocm_candidates: list[tuple[tuple[int, ...], str]] = []
            for _name in upstream_assets:
                _m = _rocm_pattern.match(_name)
                if _m is None:
                    continue
                _parts = tuple(int(p) for p in _m.group(1).split("."))
                rocm_candidates.append((_parts, _name))
            rocm_candidates.sort(reverse = True)
            _host_rocm_version = _detect_host_rocm_version()
            _compatible: list[tuple[tuple[int, ...], str]] = rocm_candidates
            if _host_rocm_version is not None:
                _compatible = [
                    item
                    for item in rocm_candidates
                    if item[0][:2] <= _host_rocm_version
                ]
            if rocm_candidates and not _compatible:
                # Fall back to the newest candidate so a source build is
                # not forced when the host runtime is older than every
                # published prebuilt: preflight will still catch a true
                # incompatibility and trigger a fallback.
                _compatible = rocm_candidates[:1]
            if _compatible:
                rocm_name = _compatible[0][1]
                if _host_rocm_version is not None:
                    log(
                        f"AMD ROCm {_host_rocm_version[0]}.{_host_rocm_version[1]} "
                        f"detected -- trying upstream prebuilt {rocm_name}"
                    )
                else:
                    log(f"AMD ROCm detected -- trying upstream prebuilt {rocm_name}")
                log(
                    "Note: if your ROCm runtime version differs significantly, "
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


def resolve_asset_choice(host: HostInfo, llama_tag: str) -> AssetChoice:
    if host.is_linux and host.is_x86_64 and host.has_usable_nvidia:
        raise PrebuiltFallback(
            "Linux CUDA installs require a compatible published bundle; upstream fallback is not available"
        )
    return resolve_upstream_asset_choice(host, llama_tag)


def resolve_release_asset_choice(
    host: HostInfo,
    llama_tag: str,
    release: PublishedReleaseBundle,
    checksums: ApprovedReleaseChecksums,
) -> list[AssetChoice]:
    if host.is_windows and host.is_x86_64 and host.has_usable_nvidia:
        torch_preference = detect_torch_cuda_runtime_preference(host)
        published_attempts = published_windows_cuda_attempts(
            host,
            release,
            torch_preference.runtime_line,
            torch_preference.selection_log,
        )
        if published_attempts:
            try:
                return apply_approved_hashes(published_attempts, checksums)
            except PrebuiltFallback as exc:
                log(
                    "published Windows CUDA assets ignored for install planning: "
                    f"{release.repo}@{release.release_tag} ({exc})"
                )
        upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)
        return apply_approved_hashes(
            resolve_windows_cuda_choices(host, llama_tag, upstream_assets),
            checksums,
        )

    published_choice: AssetChoice | None = None
    if host.is_windows and host.is_x86_64:
        # AMD Windows hosts should prefer a hash-approved published
        # Windows HIP bundle when one exists, but otherwise fall through
        # to resolve_asset_choice() so the upstream HIP prebuilt is
        # tried before the CPU fallback. Hard-pinning the published
        # windows-cpu bundle here would make the new HIP path
        # unreachable.
        if host.has_rocm:
            published_choice = published_asset_choice_for_kind(release, "windows-hip")
        else:
            published_choice = published_asset_choice_for_kind(release, "windows-cpu")
    elif host.is_macos and host.is_arm64:
        published_choice = published_asset_choice_for_kind(release, "macos-arm64")
    elif host.is_macos and host.is_x86_64:
        published_choice = published_asset_choice_for_kind(release, "macos-x64")

    if published_choice is not None:
        try:
            return apply_approved_hashes([published_choice], checksums)
        except PrebuiltFallback as exc:
            log(
                "published platform asset ignored for install planning: "
                f"{release.repo}@{release.release_tag} {published_choice.name} ({exc})"
            )

    return apply_approved_hashes([resolve_asset_choice(host, llama_tag)], checksums)


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

    def _try_repair_missing_slash(
        member_name: str, link_name: str, archive_names: set[str]
    ) -> str | None:
        """Some upstream llama.cpp Mac releases (e.g. b9165, b9169) ship
        symlinks whose linkname is missing the directory separator AND
        the leading character of the file basename between the
        top-level dir and the rest of the path:

            llama-b9165/libggml-rpc.0.dylib -> llama-b9165ibggml-rpc.0.11.1.dylib

        That cannot be resolved as written. Detect the pattern
        (linkname starts with the top-level dir name but no following
        slash) and search archive entries under that dir for a real
        file whose basename ends with the mangled suffix. Only accept
        when the suffix uniquely identifies a real archive entry.
        Returns the corrected linkname expressed relative to the
        member's parent directory -- callers join it with
        `target.parent`, so a full `top/file` path would double the
        prefix into `top/top/file`."""
        if "/" not in member_name or "/" in link_name:
            return None
        top, _, _ = member_name.partition("/")
        if not link_name.startswith(top) or len(link_name) <= len(top):
            return None
        bad_suffix = link_name[len(top) :]
        if not bad_suffix or bad_suffix.startswith("/"):
            return None
        prefix = f"{top}/"
        candidates = [
            name
            for name in archive_names
            if name.startswith(prefix)
            and "/" not in name[len(prefix) :]
            and name[len(prefix) :].endswith(bad_suffix)
        ]
        if len(candidates) != 1:
            return None
        # Strip the top-level dir so the caller's `target.parent / Path(...)`
        # composition resolves inside the staging dir, not into a duplicate
        # `top/top/...` path.
        return candidates[0][len(prefix) :]

    def safe_link_target(
        base: Path,
        member_name: str,
        link_name: str,
        target: Path,
        archive_names: set[str],
    ) -> tuple[str, Path]:
        normalized = link_name.replace("\\", "/")
        repaired = _try_repair_missing_slash(member_name, normalized, archive_names)
        if repaired is not None:
            normalized = repaired
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
        archive_names: set[str] = set()
        with tarfile.open(source, "r:gz") as archive:
            for member in archive.getmembers():
                archive_names.add(member.name)
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
                    base, member.name, member.linkname, target, archive_names
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
    source_ref: str,
    install_dir: Path,
    work_dir: Path,
    *,
    source_repo: str = UPSTREAM_REPO,
    expected_sha256: str | None,
    source_label: str | None = None,
    exact_source: bool = False,
) -> None:
    archive_path = work_dir / f"llama.cpp-source-{source_ref}.tar.gz"
    source_urls = (
        commit_source_archive_urls(source_repo, source_ref)
        if exact_source
        else upstream_source_archive_urls(source_ref)
    )
    label = source_label or f"llama.cpp source tree for {source_ref}"
    extract_dir = Path(tempfile.mkdtemp(prefix = "source-extract-", dir = work_dir))

    try:
        log(f"downloading {label}")
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
                    label = label,
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
        raise PrebuiltFallback(f"failed to hydrate {label}: {exc}") from exc
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


def paired_runtime_dll_patterns(choice: AssetChoice) -> list[str]:
    """Filename patterns the paired runtime archive is allowed to drop
    into the install. Used for the second copy_globs pass in
    install_from_archives, narrower than runtime_patterns_for_choice so
    the runtime archive cannot overwrite main-archive payload like
    llama-server.exe. Only Windows CUDA has paired runtimes today."""
    if choice.install_kind == "windows-cuda":
        return ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"]
    return []


def runtime_patterns_for_choice(choice: AssetChoice) -> list[str]:
    if choice.install_kind in {"linux-cpu", "linux-cuda", "linux-rocm"}:
        return [
            "llama-server",
            "llama-quantize",
            "libllama-common.so*",
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
                try:
                    os.write(fd, f"{os.getpid()}\n".encode())
                    os.fsync(fd)
                except Exception:
                    os.close(fd)
                    fd = None
                    lock_path.unlink(missing_ok = True)
                    raise
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
                    if time.monotonic() >= deadline:
                        raise BusyInstallConflict(
                            f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for concurrent install lock: {lock_path}"
                        )
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
                    raise BusyInstallConflict(
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
        raise BusyInstallConflict(
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
                if is_busy_lock_error(exc):
                    raise BusyInstallConflict(
                        "staged prebuilt validation passed but the existing install could not be replaced "
                        "because llama.cpp appears to still be in use; restored previous install "
                        f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                    ) from exc
                raise PrebuiltFallback(
                    "staged prebuilt validation passed but activation failed; restored previous install "
                    f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                ) from exc
        except (BusyInstallConflict, PrebuiltFallback):
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
            try:
                remove_tree_logged(rollback_dir, "rollback path")
            except Exception as cleanup_exc:
                log(
                    f"non-fatal: rollback cleanup failed after successful activation: {cleanup_exc}"
                )
    finally:
        remove_tree(failed_dir)
        remove_tree(staging_dir)
        prune_install_staging_root(install_dir)


def install_from_archives(
    choice: AssetChoice, host: HostInfo, install_dir: Path, work_dir: Path
) -> tuple[Path, Path]:
    main_archive = work_dir / choice.name
    log(f"downloading {choice.name} from {choice.source_label} release")
    download_file_verified(
        choice.url,
        main_archive,
        expected_sha256 = choice.expected_sha256,
        label = f"prebuilt archive {choice.name}",
    )

    install_dir.mkdir(parents = True, exist_ok = True)
    extract_dir = Path(tempfile.mkdtemp(prefix = "extract-", dir = work_dir))
    runtime_extract_dir: Path | None = None

    try:
        extract_archive(main_archive, extract_dir)
        # Download the paired runtime archive into its own temp dir to
        # avoid copy_globs's ambiguous-layout guard on shared names
        # like LICENSE.txt. Two passes of copy_globs land both archives
        # in the same overlay dir. Fixes #5106.
        if choice.runtime_url and choice.runtime_name:
            runtime_archive = work_dir / choice.runtime_name
            log(
                f"downloading paired runtime archive {choice.runtime_name} "
                f"from {choice.source_label} release"
            )
            download_file_verified(
                choice.runtime_url,
                runtime_archive,
                expected_sha256 = choice.runtime_sha256,
                label = f"prebuilt runtime archive {choice.runtime_name}",
            )
            runtime_extract_dir = Path(
                tempfile.mkdtemp(prefix = "extract-runtime-", dir = work_dir)
            )
            extract_archive(runtime_archive, runtime_extract_dir)
        source_dir = extract_dir
        overlay_dir = overlay_directory_for_choice(install_dir, choice, host)
        copy_globs(
            source_dir, overlay_dir, runtime_patterns_for_choice(choice), required = True
        )
        if runtime_extract_dir is not None:
            # The runtime archive only contributes the CUDA DLLs.
            # Restrict the overlay to the cudart bundle's known
            # filenames (cudart64_X.dll / cublas64_X.dll /
            # cublasLt64_X.dll) rather than the broad ``*.exe`` /
            # ``*.dll`` set from runtime_patterns_for_choice, so a
            # malformed runtime archive can never overwrite
            # llama-server.exe or other main-archive payload. The
            # upstream cudart-llama-bin-win-cuda-X.Y-x64.zip currently
            # ships exactly these three DLLs (verified against b9103
            # cuda-12.4 and cuda-13.1 bundles).
            copy_globs(
                runtime_extract_dir,
                overlay_dir,
                paired_runtime_dll_patterns(choice),
                required = False,
            )
        copy_globs(
            source_dir,
            install_dir,
            metadata_patterns_for_choice(choice),
            required = False,
        )
    finally:
        remove_tree(extract_dir)
        if runtime_extract_dir is not None:
            remove_tree(runtime_extract_dir)

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
        # ``nvidia/<pkg>/lib`` -- Linux convention; harmless on Windows
        # where the directory simply does not exist on real wheels.
        candidates.extend(root.glob("nvidia/*/lib"))
        # ``nvidia/<pkg>/bin`` -- legacy modular Windows wheels
        # (``nvidia-cuda-runtime-cu12``, ``nvidia-cublas-cu12``).
        candidates.extend(root.glob("nvidia/*/bin"))
        # ``nvidia/<pkg>/bin/x86_64`` and ``.../bin/x64`` -- current
        # CUDA 13 Windows wheel layout (the unsuffixed
        # ``nvidia-cuda-runtime`` 13.x and ``nvidia-cublas`` 13.x
        # packages ship under ``nvidia/cu13/bin/x86_64/cudart64_13.dll``).
        # Without these, Windows preflight CUDA detection misses cu13
        # installs and falls back to the upstream cudart bundle path
        # even when usable DLLs are already on disk (#5106). Kept in
        # sync with the backend resolver
        # ``llama_cpp.LlamaCppBackend._windows_pip_nvidia_dll_dirs``.
        candidates.extend(root.glob("nvidia/*/bin/x86_64"))
        candidates.extend(root.glob("nvidia/*/bin/x64"))
        # ``nvidia/<pkg>/Library/bin`` -- conda-style wheel repacks.
        candidates.extend(root.glob("nvidia/*/Library/bin"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x86_64"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x64"))
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
        **windows_hidden_subprocess_kwargs(),
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
    install_kind: str | None = None,
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
        # Only enable GPU offload for assets that actually ship GPU code.
        # Gating on `host.has_rocm` alone breaks the intentional CPU
        # fallback on AMD Windows hosts without a HIP prebuilt: the CPU
        # binary would be launched with `--n-gpu-layers 1` and fail
        # validation. Use the resolved install_kind as the source of
        # truth and fall back to host detection when the caller did not
        # pass one (keeps backwards compatibility with older call sites).
        _gpu_kinds = {
            "linux-cuda",
            "linux-rocm",
            "windows-cuda",
            "windows-hip",
            "macos-arm64",
        }
        if install_kind is not None:
            _enable_gpu_layers = install_kind in _gpu_kinds
        else:
            # Older call sites that don't pass install_kind: keep ROCm
            # hosts in the GPU-validation path so an AMD-only Linux host
            # is exercised against the actual hardware rather than the
            # CPU fallback. NVIDIA and macOS-arm64 are already covered.
            _enable_gpu_layers = (
                host.has_usable_nvidia
                or host.has_rocm
                or (host.is_macos and host.is_arm64)
            )
        if _enable_gpu_layers:
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
                    **windows_hidden_subprocess_kwargs(),
                )
                deadline = time.time() + 60
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
    def approved_hash_for_attempt(attempt: AssetChoice) -> ApprovedArtifactHash | None:
        candidate_names = [attempt.name]
        if (
            isinstance(attempt.tag, str)
            and attempt.tag
            and attempt.tag != checksums.upstream_tag
            and attempt.name.startswith("llama-")
        ):
            legacy_prefix = f"llama-{attempt.tag}-"
            compatibility_prefix = f"llama-{checksums.upstream_tag}-"
            compatibility_name = (
                attempt.name.replace(legacy_prefix, compatibility_prefix, 1)
                if attempt.name.startswith(legacy_prefix)
                else attempt.name
            )
            candidate_names.append(compatibility_name)
        candidate_names.extend(
            windows_cuda_asset_aliases(
                attempt.name,
                compatibility_tag = checksums.upstream_tag,
            )
        )
        seen_names: set[str] = set()
        for candidate_name in candidate_names:
            if candidate_name in seen_names:
                continue
            seen_names.add(candidate_name)
            approved = checksums.artifacts.get(candidate_name)
            if approved is not None:
                return approved
        return None

    approved_attempts: list[AssetChoice] = []
    missing_assets: list[str] = []
    for attempt in attempts:
        approved = approved_hash_for_attempt(attempt)
        if approved is None:
            missing_assets.append(attempt.name)
            continue
        attempt.expected_sha256 = approved.sha256
        # Resolve the paired runtime archive's hash too. Drop the pair
        # if the manifest does not list it -- never install an
        # unverified archive.
        if attempt.runtime_name and attempt.runtime_url:
            runtime_approved = checksums.artifacts.get(attempt.runtime_name)
            if runtime_approved is None:
                attempt.runtime_name = None
                attempt.runtime_url = None
                attempt.runtime_sha256 = None
            else:
                attempt.runtime_sha256 = runtime_approved.sha256
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


def preferred_source_archive(
    checksums: ApprovedReleaseChecksums, llama_tag: str
) -> tuple[str, str, ApprovedArtifactHash | None, bool]:
    exact_source = exact_source_archive_hash(checksums)
    exact_repo = repo_slug_from_source(checksums.source_repo) or repo_slug_from_source(
        checksums.source_repo_url
    )
    if exact_source is not None and exact_repo and checksums.source_commit:
        return (
            exact_repo,
            checksums.source_commit,
            exact_source,
            True,
        )
    legacy = checksums.artifacts.get(source_archive_logical_name(llama_tag))
    return (
        UPSTREAM_REPO,
        llama_tag,
        legacy,
        False,
    )


def selected_source_archive_metadata(
    checksums: ApprovedReleaseChecksums,
    llama_tag: str,
) -> tuple[str, str | None]:
    _source_repo, _source_ref, source_archive, _exact_source = preferred_source_archive(
        checksums, llama_tag
    )
    if source_archive is None:
        return source_archive_logical_name(llama_tag), None
    return source_archive.asset_name, source_archive.sha256


def resolve_install_attempts(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
) -> tuple[str, str, list[AssetChoice], ApprovedReleaseChecksums]:
    requested_tag, plans = resolve_install_release_plans(
        llama_tag,
        host,
        published_repo,
        published_release_tag,
    )
    if not plans:
        raise PrebuiltFallback("no prebuilt release plans were available")
    plan = plans[0]
    return requested_tag, plan.llama_tag, plan.attempts, plan.approved_checksums


def resolve_install_release_plans(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
    *,
    max_release_fallbacks: int = DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS,
) -> tuple[str, list[InstallReleasePlan]]:
    requested_tag = normalized_requested_llama_tag(llama_tag)
    allow_older_release_fallback = (
        requested_tag == "latest" and not published_release_tag
    )
    release_limit = max(1, max_release_fallbacks)
    plans: list[InstallReleasePlan] = []
    last_error: PrebuiltFallback | None = None

    for resolved_release in iter_resolved_published_releases(
        llama_tag,
        published_repo,
        published_release_tag,
    ):
        bundle = resolved_release.bundle
        checksums = resolved_release.checksums
        resolved_tag = bundle.upstream_tag
        try:
            if host.is_linux and host.is_x86_64 and host.has_usable_nvidia:
                linux_cuda_selection = resolve_linux_cuda_choice(host, bundle)
                attempts = apply_approved_hashes(
                    linux_cuda_selection.attempts, checksums
                )
                if not attempts:
                    raise PrebuiltFallback("no compatible Linux CUDA asset was found")
                log_lines(linux_cuda_selection.selection_log)
            else:
                attempts = resolve_release_asset_choice(
                    host,
                    resolved_tag,
                    bundle,
                    checksums,
                )
                if not attempts:
                    raise PrebuiltFallback("no compatible prebuilt asset was found")
                if attempts[0].selection_log:
                    log_lines(attempts[0].selection_log)
        except PrebuiltFallback as exc:
            last_error = exc
            if not allow_older_release_fallback:
                raise
            log(
                "published release skipped for install planning: "
                f"{bundle.repo}@{bundle.release_tag} upstream_tag={resolved_tag} ({exc})"
            )
            continue

        plans.append(
            InstallReleasePlan(
                requested_tag = requested_tag,
                llama_tag = resolved_tag,
                release_tag = bundle.release_tag,
                attempts = attempts,
                approved_checksums = checksums,
            )
        )

        if not allow_older_release_fallback or len(plans) >= release_limit:
            break

    if plans:
        return requested_tag, plans
    if last_error is not None:
        raise last_error
    raise PrebuiltFallback("no installable published llama.cpp releases were found")


def write_prebuilt_metadata(
    install_dir: Path,
    *,
    requested_tag: str,
    llama_tag: str,
    release_tag: str,
    choice: AssetChoice,
    approved_checksums: ApprovedReleaseChecksums,
    prebuilt_fallback_used: bool,
) -> None:
    source_asset_name, source_sha256 = selected_source_archive_metadata(
        approved_checksums,
        llama_tag,
    )
    # expected_install_fingerprint is the source of truth for what the
    # fingerprint must contain. Calling it here -- instead of inlining a
    # parallel payload -- prevents drift where new keys (e.g. the cudart
    # pair fields added for #5106) are added to one side but not the
    # other, which would cause every install to look stale.
    fingerprint = expected_install_fingerprint(
        llama_tag = llama_tag,
        release_tag = release_tag,
        choice = choice,
        approved_checksums = approved_checksums,
    )
    if fingerprint is None:
        raise PrebuiltFallback(f"cannot compute install fingerprint for {choice.name}")
    metadata = {
        "requested_tag": requested_tag,
        "tag": llama_tag,
        "release_tag": release_tag,
        "published_repo": approved_checksums.repo,
        "asset": choice.name,
        "asset_sha256": choice.expected_sha256,
        "source": choice.source_label,
        "source_asset": source_asset_name,
        "source_sha256": source_sha256,
        "source_commit": approved_checksums.source_commit,
        "source_commit_short": approved_checksums.source_commit_short,
        "source_repo": approved_checksums.source_repo,
        "source_repo_url": approved_checksums.source_repo_url,
        "source_ref_kind": approved_checksums.source_ref_kind,
        "requested_source_ref": approved_checksums.requested_source_ref,
        "resolved_source_ref": approved_checksums.resolved_source_ref,
        "bundle_profile": choice.bundle_profile,
        "runtime_line": choice.runtime_line,
        "coverage_class": choice.coverage_class,
        "install_fingerprint": fingerprint,
        "prebuilt_fallback_used": prebuilt_fallback_used,
        "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(metadata, indent = 2) + "\n"
    )


def expected_install_fingerprint(
    *,
    llama_tag: str,
    release_tag: str,
    choice: AssetChoice,
    approved_checksums: ApprovedReleaseChecksums,
) -> str | None:
    source_asset_name, source_sha256 = selected_source_archive_metadata(
        approved_checksums,
        llama_tag,
    )
    payload = {
        "published_repo": approved_checksums.repo,
        "release_tag": release_tag,
        "upstream_tag": llama_tag,
        "asset": choice.name,
        "asset_sha256": choice.expected_sha256,
        "source": choice.source_label,
        "source_asset": source_asset_name,
        "source_sha256": source_sha256,
        "runtime_line": choice.runtime_line,
        # Including the paired runtime archive (Windows cudart bundle)
        # in the fingerprint is what forces existing #5106 installs to
        # refresh: pre-PR installs hashed nothing in this slot, post-PR
        # paired installs hash the cudart sha. Without these two keys
        # an existing cudart-less install would keep matching the new
        # choice and never re-overlay the cudart DLLs.
        "runtime_asset": choice.runtime_name,
        "runtime_sha256": choice.runtime_sha256,
        "bundle_profile": choice.bundle_profile,
        "coverage_class": choice.coverage_class,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()


def load_prebuilt_metadata(install_dir: Path) -> dict[str, Any] | None:
    metadata_path = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding = "utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def runtime_payload_health_groups(choice: AssetChoice) -> list[list[str]]:
    if choice.install_kind == "linux-cpu":
        return [
            ["libllama-common.so*"],
            ["libllama.so*"],
            ["libggml.so*"],
            ["libggml-base.so*"],
            ["libggml-cpu-*.so*"],
            ["libmtmd.so*"],
        ]
    if choice.install_kind == "linux-cuda":
        return [
            ["libllama-common.so*"],
            ["libllama.so*"],
            ["libggml.so*"],
            ["libggml-base.so*"],
            ["libggml-cpu-*.so*"],
            ["libmtmd.so*"],
            ["libggml-cuda.so*"],
        ]
    if choice.install_kind in {"macos-arm64", "macos-x64"}:
        return [
            ["libllama*.dylib"],
            ["libggml*.dylib"],
            ["libmtmd*.dylib"],
        ]
    if choice.install_kind == "linux-rocm":
        return [
            ["libllama-common.so*"],
            ["libllama.so*"],
            ["libggml.so*"],
            ["libggml-base.so*"],
            ["libggml-cpu-*.so*"],
            ["libmtmd.so*"],
            ["libggml-hip.so*"],
        ]
    if choice.install_kind == "windows-cpu":
        return [["llama.dll"]]
    if choice.install_kind == "windows-cuda":
        groups = [["llama.dll"], ["ggml-cuda.dll"]]
        # When the cudart bundle was paired in (#5106) require all
        # three of its DLLs alongside the main archive's payload.
        # install_kind alone is not enough -- legacy installs without
        # the cudart pair must still pass the health check on the
        # no-pair fallback path, otherwise pair-less builds would loop
        # on reinstall forever. The upstream cudart bundle ships
        # cudart64_X.dll + cublas64_X.dll + cublasLt64_X.dll; missing
        # any one of them still breaks GPU initialisation.
        if choice.runtime_name:
            groups.append(["cudart64_*.dll"])
            groups.append(["cublas64_*.dll"])
            groups.append(["cublasLt64_*.dll"])
        return groups
    if choice.install_kind == "windows-hip":
        return [["llama.dll"], ["*hip*.dll"]]
    return []


def install_runtime_dir(install_dir: Path, host: HostInfo) -> Path:
    if host.is_windows:
        return install_dir / "build" / "bin" / "Release"
    return install_dir / "build" / "bin"


def runtime_payload_is_healthy(
    install_dir: Path, host: HostInfo, choice: AssetChoice
) -> bool:
    runtime_dir = install_runtime_dir(install_dir, host)
    if not runtime_dir.exists():
        return False
    for pattern_group in runtime_payload_health_groups(choice):
        matched = False
        for pattern in pattern_group:
            if any(runtime_dir.glob(pattern)):
                matched = True
                break
        if not matched:
            return False
    return True


def existing_install_matches_choice(
    install_dir: Path,
    host: HostInfo,
    *,
    llama_tag: str,
    release_tag: str,
    choice: AssetChoice,
    approved_checksums: ApprovedReleaseChecksums,
) -> bool:
    if not install_dir.exists():
        return False

    metadata = load_prebuilt_metadata(install_dir)
    if metadata is None:
        return False

    try:
        confirm_install_tree(install_dir, host)
    except Exception:
        return False

    if not runtime_payload_is_healthy(install_dir, host, choice):
        return False

    # Verify primary executables still exist (catches partial deletion)
    runtime_dir = install_runtime_dir(install_dir, host)
    ext = ".exe" if host.is_windows else ""
    for binary in ("llama-server", "llama-quantize"):
        if not (runtime_dir / f"{binary}{ext}").exists():
            return False
    expected_fingerprint = expected_install_fingerprint(
        llama_tag = llama_tag,
        release_tag = release_tag,
        choice = choice,
        approved_checksums = approved_checksums,
    )
    if not expected_fingerprint:
        return False

    recorded_fingerprint = metadata.get("install_fingerprint")
    if not isinstance(recorded_fingerprint, str) or not recorded_fingerprint:
        return False

    if recorded_fingerprint != expected_fingerprint:
        return False

    expected_pairs = {
        "release_tag": release_tag,
        "published_repo": approved_checksums.repo,
        "tag": llama_tag,
        "asset": choice.name,
        "asset_sha256": choice.expected_sha256,
        "source": choice.source_label,
        "runtime_line": choice.runtime_line,
        "bundle_profile": choice.bundle_profile,
        "coverage_class": choice.coverage_class,
    }
    for key, expected in expected_pairs.items():
        if metadata.get(key) != expected:
            return False
    return True


def existing_install_matches_plan(
    install_dir: Path,
    host: HostInfo,
    plan: InstallReleasePlan,
) -> bool:
    if not plan.attempts:
        return False
    return existing_install_matches_choice(
        install_dir,
        host,
        llama_tag = plan.llama_tag,
        release_tag = plan.release_tag,
        choice = plan.attempts[0],
        approved_checksums = plan.approved_checksums,
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
    release_tag: str,
    approved_checksums: ApprovedReleaseChecksums,
    prebuilt_fallback_used: bool,
    quantized_path: Path,
) -> tuple[Path, Path]:
    source_repo, source_ref, source_archive, exact_source = preferred_source_archive(
        approved_checksums, llama_tag
    )
    if exact_source:
        log(
            f"hydrating exact llama.cpp source for {source_repo}@{source_ref} into {install_dir}"
        )
    else:
        log(f"hydrating upstream llama.cpp source for {llama_tag} into {install_dir}")
    hydrate_source_tree(
        source_ref,
        install_dir,
        work_dir,
        source_repo = source_repo,
        expected_sha256 = source_archive.sha256 if source_archive is not None else None,
        source_label = (
            f"llama.cpp source tree for {source_repo}@{source_ref}"
            if exact_source
            else f"llama.cpp source tree for {llama_tag}"
        ),
        exact_source = exact_source,
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
        release_tag = release_tag,
        choice = choice,
        approved_checksums = approved_checksums,
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
        install_kind = choice.install_kind,
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
    release_tag: str,
    approved_checksums: ApprovedReleaseChecksums,
    initial_fallback_used: bool = False,
    existing_install_dir: Path | None = None,
) -> tuple[AssetChoice, Path, bool]:
    attempt_list = list(attempts)
    if not attempt_list:
        raise PrebuiltFallback("no prebuilt bundle attempts were available")

    tried_fallback = initial_fallback_used
    for index, attempt in enumerate(attempt_list):
        if index > 0:
            tried_fallback = True
            log(
                "retrying CUDA prebuilt "
                f"{attempt.name} install_kind={attempt.install_kind} "
                f"runtime_line={attempt.runtime_line} coverage_class={attempt.coverage_class}"
            )

        if existing_install_dir is not None and existing_install_matches_choice(
            existing_install_dir,
            host,
            llama_tag = llama_tag,
            release_tag = release_tag,
            choice = attempt,
            approved_checksums = approved_checksums,
        ):
            log(
                "existing llama.cpp install already matches fallback candidate "
                f"{attempt.name}; skipping reinstall"
            )
            raise ExistingInstallSatisfied(attempt, tried_fallback)

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
                release_tag = release_tag,
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
    install_dir: Path,
    llama_tag: str,
    published_repo: str,
    published_release_tag: str,
    *,
    simple_policy: bool = False,
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
            if simple_policy:
                requested_tag, release_plans = resolve_simple_install_release_plans(
                    llama_tag,
                    host,
                    published_repo,
                    published_release_tag,
                )
            else:
                requested_tag, release_plans = resolve_install_release_plans(
                    llama_tag,
                    host,
                    published_repo,
                    published_release_tag,
                )
            if release_plans and existing_install_matches_plan(
                install_dir, host, release_plans[0]
            ):
                current = release_plans[0]
                log(
                    "existing llama.cpp install already matches selected release "
                    f"{current.release_tag} upstream_tag={current.llama_tag}; skipping download and install"
                )
                return
            with tempfile.TemporaryDirectory(prefix = "unsloth-llama-prebuilt-") as tmp:
                work_dir = Path(tmp)
                probe_path = work_dir / "stories260K.gguf"
                download_validation_model(
                    probe_path, validation_model_cache_path(install_dir)
                )
                release_count = len(release_plans)
                for release_index, plan in enumerate(release_plans):
                    choice = plan.attempts[0]
                    if existing_install_matches_plan(install_dir, host, plan):
                        log(
                            "existing llama.cpp install already matches fallback release "
                            f"{plan.release_tag} upstream_tag={plan.llama_tag}; skipping reinstall"
                        )
                        return
                    log(
                        "selected "
                        f"{choice.name} ({choice.source_label}) from published release "
                        f"{plan.release_tag} for {host.system} {host.machine}"
                    )
                    try:
                        choice, selected_staging_dir, _ = validate_prebuilt_attempts(
                            plan.attempts,
                            host,
                            install_dir,
                            work_dir,
                            probe_path,
                            requested_tag = requested_tag,
                            llama_tag = plan.llama_tag,
                            release_tag = plan.release_tag,
                            approved_checksums = plan.approved_checksums,
                            initial_fallback_used = release_index > 0,
                            existing_install_dir = install_dir,
                        )
                    except ExistingInstallSatisfied:
                        return
                    except PrebuiltFallback as exc:
                        if release_index == release_count - 1:
                            raise
                        log(
                            "published release "
                            f"{plan.release_tag} upstream_tag={plan.llama_tag} failed; "
                            "trying the next older published prebuilt "
                            f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                        )
                        continue

                    activate_install_tree(selected_staging_dir, install_dir, host)
                    try:
                        ensure_converter_scripts(install_dir, plan.llama_tag)
                    except Exception as exc:
                        log(
                            "converter script fetch failed after activation; install remains valid "
                            f"({textwrap.shorten(str(exc), width = 200, placeholder = '...')})"
                        )
                    return
    except BusyInstallConflict as exc:
        log("prebuilt install path is blocked by an in-use llama.cpp install")
        log(f"prebuilt busy reason: {exc}")
        raise SystemExit(EXIT_BUSY) from exc
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
        help = (
            "llama.cpp release tag. Defaults to the latest usable published Unsloth "
            "release unless UNSLOTH_LLAMA_TAG overrides it."
        ),
    )
    parser.add_argument(
        "--published-repo",
        default = DEFAULT_PUBLISHED_REPO,
        help = "Published bundle repository",
    )
    parser.add_argument(
        "--published-release-tag",
        default = DEFAULT_PUBLISHED_TAG,
        help = (
            "Published GitHub release tag to pin. By default, scan releases "
            "until a usable published llama.cpp release bundle is found."
        ),
    )
    parser.add_argument(
        "--simple-policy",
        action = "store_true",
        help = "Use the simplified platform-specific prebuilt selection policy.",
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
        help = (
            "Resolve a llama.cpp tag such as 'latest' to the concrete upstream tag "
            "selected by the current published-release policy."
        ),
    )
    resolve_group.add_argument(
        "--resolve-source-build",
        nargs = "?",
        const = "latest",
        help = ("Resolve the source-build fallback plan."),
    )
    parser.add_argument(
        "--output-format",
        choices = ("plain", "json"),
        default = "plain",
        help = "Resolver output format. Defaults to plain.",
    )
    return parser.parse_args()


def emit_resolver_output(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, sort_keys = True))
        return
    if "llama_tag" in payload:
        print(payload["llama_tag"])
        return
    if {
        "source_url",
        "source_ref_kind",
        "source_ref",
    }.issubset(payload):
        print(
            "\t".join(
                (
                    str(payload["source_url"]),
                    str(payload["source_ref_kind"]),
                    str(payload["source_ref"]),
                )
            )
        )
        return
    print(json.dumps(payload, sort_keys = True))


def main() -> int:
    args = parse_args()
    if args.resolve_llama_tag is not None:
        resolved = resolve_requested_llama_tag(
            args.resolve_llama_tag,
            args.published_repo,
            args.published_release_tag or "",
        )
        emit_resolver_output(
            {
                "requested_tag": normalized_requested_llama_tag(args.resolve_llama_tag),
                "llama_tag": resolved,
            },
            output_format = args.output_format,
        )
        return EXIT_SUCCESS

    if args.resolve_install_tag is not None:
        resolved = resolve_requested_install_tag(
            args.resolve_install_tag,
            args.published_release_tag or "",
            args.published_repo,
        )
        emit_resolver_output(
            {
                "requested_tag": normalized_requested_llama_tag(
                    args.resolve_install_tag
                ),
                "llama_tag": resolved,
            },
            output_format = args.output_format,
        )
        return EXIT_SUCCESS

    if args.resolve_source_build is not None:
        plan = resolve_source_build_plan(
            args.resolve_source_build,
            args.published_repo,
            args.published_release_tag or "",
        )
        emit_resolver_output(
            {
                "requested_tag": normalized_requested_llama_tag(
                    args.resolve_source_build
                ),
                "source_url": plan.source_url,
                "source_ref_kind": plan.source_ref_kind,
                "source_ref": plan.source_ref,
                "compatibility_upstream_tag": plan.compatibility_upstream_tag,
            },
            output_format = args.output_format,
        )
        return EXIT_SUCCESS

    if not args.install_dir:
        raise SystemExit(
            "install_llama_prebuilt.py: --install-dir is required unless --resolve-llama-tag, --resolve-install-tag, or --resolve-source-build is used"
        )
    install_prebuilt(
        install_dir = Path(args.install_dir).expanduser().resolve(),
        llama_tag = args.llama_tag,
        published_repo = args.published_repo,
        published_release_tag = args.published_release_tag or "",
        simple_policy = args.simple_policy,
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BusyInstallConflict as exc:
        log(
            f"fatal helper busy conflict: {textwrap.shorten(str(exc), width = 400, placeholder = '...')}"
        )
        raise SystemExit(EXIT_BUSY)
    except PrebuiltFallback as exc:
        # Expected when the published repo (e.g. ggml-org/llama.cpp) has no
        # prebuilt manifest.  Exit quietly with EXIT_FALLBACK so the caller
        # falls back to source build without a noisy "fatal helper error".
        log(textwrap.shorten(str(exc), width = 400, placeholder = "..."))
        raise SystemExit(EXIT_FALLBACK)
    except Exception as exc:
        message = textwrap.shorten(str(exc), width = 400, placeholder = "...")
        log(f"fatal helper error: {message}")
        raise SystemExit(EXIT_ERROR)
