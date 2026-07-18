#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross platform llama.cpp prebuilt installer for Unsloth Studio"""

from __future__ import annotations

import argparse
import atexit
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
import stat
import struct
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
from dataclasses import dataclass, field, replace as dataclasses_replace

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Literal


EXIT_SUCCESS = 0
EXIT_FALLBACK = 2
EXIT_ERROR = 1
EXIT_BUSY = 3

# DiskPart-prompt suppression. RunAsInvoker does NOT stop amd-smi's runtime
# elevation (its manifest is asInvoker), so this is just harmless belt-and-
# suspenders for manifest-elevating tools. The real guard is _amd_smi_allowed():
# we don't spawn amd-smi on Windows w/o a HIP SDK (or opt-in).
if platform.system() == "Windows":
    os.environ.setdefault("__COMPAT_LAYER", "RunAsInvoker")


def _path_inside_venv(path: str) -> bool:
    """True if ``path`` is inside the active venv (sys.prefix).

    The venv hipInfo.exe (AMD wheel, put on PATH by the bnb fix) is NOT a HIP SDK
    (_amd_smi_allowed)."""
    try:
        # realpath (not abspath): resolve symlinks/8.3 names so an aliased venv matches.
        _root = os.path.normcase(os.path.realpath(sys.prefix))
        # Guard a root-dir prefix (C:\ or /): commonpath would match every path on
        # it. A venv is never at root, so treat that as outside.
        if os.path.dirname(_root) == _root:
            return False
        return os.path.normcase(os.path.commonpath([os.path.realpath(path), _root])) == _root
    except (ValueError, OSError):
        # Different drive / unresolvable -> treat as outside the venv.
        return False


def _external_hipinfo_on_path() -> bool:
    """True if a hipinfo OUTSIDE the venv is on PATH.

    shutil.which returns only the first hit, so the venv hipInfo could shadow a
    real HIP SDK's; scan every PATH entry and skip the venv copy."""
    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        _dir = _dir.strip('"')  # PATH entries can be quoted on Windows
        if not _dir:
            continue
        _candidate = os.path.join(_dir, "hipinfo.exe")
        if os.path.isfile(_candidate) and not _path_inside_venv(_candidate):
            return True
    return False


def _amd_smi_allowed() -> bool:
    """Whether it is safe to spawn amd-smi here.

    On Windows w/o a working HIP runtime, amd-smi elevates a child and pops a
    UAC/DiskPart prompt RunAsInvoker can't suppress. Only call it on Windows
    when a HIP SDK is detectable (hipinfo present) or UNSLOTH_ENABLE_AMD_SMI=1;
    Linux/macOS always allowed. When skipped, the gfx arch still arrives via the
    forwarded --rocm-gfx, so prebuilt selection is unaffected.
    """
    if platform.system() != "Windows":
        return True
    flag = os.environ.get("UNSLOTH_ENABLE_AMD_SMI", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    # A real HIP SDK lets amd-smi run un-elevated; hipinfo-on-PATH is the proxy.
    # Ignore the venv hipInfo.exe (AMD wheel via bnb fix): not a HIP SDK, doesn't
    # stop amd-smi's DiskPart UAC.
    if _external_hipinfo_on_path():
        return True
    for _var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        _root = os.environ.get(_var)
        if not _root:
            continue
        _candidate = os.path.join(_root, "bin", "hipinfo.exe")
        if os.path.isfile(_candidate) and not _path_inside_venv(_candidate):
            return True
    return False


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


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
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


TEST_MODEL_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
TEST_MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
VALIDATION_MODEL_CACHE_DIRNAME = ".cache"
VALIDATION_MODEL_CACHE_FILENAME = "stories260K.gguf"
# Master switch for the staged runtime smoke test (llama-quantize + llama-server)
# in validate_prebuilt_choice. Disabled for now: the llama-server GPU forward pass
# JIT-compiles CUDA kernels on first load and stalls every install and update by
# minutes on Blackwell (sm_100). The check and the source-build fallback it triggers
# are kept intact -- set this to True to re-enable them.
_RUN_STAGED_PREBUILT_VALIDATION = False
INSTALL_LOCK_TIMEOUT_SECONDS = 300
INSTALL_STAGING_ROOT_NAME = ".staging"
GITHUB_AUTH_HOSTS = {"api.github.com", "github.com"}
HF_AUTH_HOSTS = {"huggingface.co", "www.huggingface.co"}
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
# Deeper macOS-only walk-back: upstream can ship a run of prebuilts built for a
# newer macOS than the host, only caught at validate time, so an older host must
# skip the whole run. Free on new hosts (first plan validates, extras unused).
DEFAULT_MAX_MACOS_RELEASE_FALLBACKS = env_int(
    "UNSLOTH_LLAMA_MAX_MACOS_RELEASE_FALLBACKS",
    16,
    minimum = 1,
)
# Last upstream macOS release before ggml-org moved macOS builds to Tahoe.
_PINNED_MACOS_FALLBACK_TAG = "b9415"
_PINNED_MACOS_LATEST_FLOOR = (26, 0)
FORCE_COMPILE_DEFAULT_REF = os.environ.get("UNSLOTH_LLAMA_FORCE_COMPILE_REF", "master")

# Lowest CUDA major we ship prebuilts for, and the highest major we probe for
# installed runtime libraries. Detection and runtime-line derivation are
# generated per major so a new toolkit (cuda14, ...) needs no code change while
# llama.cpp keeps the cudart64_<major>.dll / libcudart.so.<major> naming.
_MIN_CUDA_MAJOR = 12
_MAX_PROBE_CUDA_MAJOR = 19

# Blackwell floor is sm_100: data-center parts (B100/B200 sm_100, B300/GB300
# sm_103) sit below consumer Blackwell (RTX 50 sm_120); the family needs toolkit
# >= 12.8, except sm_103/sm_121 which need 12.9. (120 here wrongly excluded the
# sm_100/103 data-center hosts.)
_BLACKWELL_MIN_SM = 100
_BLACKWELL_MIN_TOOLKIT = (12, 8)
# SMs that need a newer toolkit than the family floor (CUDA 12.9 added native
# sm_103/sm_121 targets; 12.8 covers sm_100/101/120).
_BLACKWELL_SM_MIN_TOOLKIT = {103: (12, 9), 121: (12, 9)}


def _cuda_runtime_lines_for_major(major: int) -> list[str]:
    """Runtime lines a driver of this CUDA major can use, newest major first
    down to the minimum we ship. A driver runs its own major and any older one
    (backward compatibility)."""
    return [f"cuda{m}" for m in range(major, _MIN_CUDA_MAJOR - 1, -1)]


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
    rocm_gfx_target: str | None = None
    # (major, minor) from platform.mac_ver(); None off macOS or if unparseable.
    # Skips a macos prebuilt whose minimum-OS exceeds this host.
    macos_version: tuple[int, int] | None = None


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
    # ROCm bundles only: the umbrella gfx target (e.g. "gfx110X") and the
    # concrete gfx archs it covers (e.g. ["gfx1100", "gfx1101", ...]).
    gfx_target: str | None = None
    mapped_targets: list[str] = field(default_factory = list)


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


class ValidationLaunchUnavailable(RuntimeError):
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


# Status logs default to stderr so resolver modes keep stdout machine-readable
# (setup.sh json.load()s the whole stdout). main() flips this for the install
# path, where PowerShell otherwise renders stderr as NativeCommandError noise.
_LOG_TO_STDOUT = False


def log(message: str) -> None:
    print(f"[llama-prebuilt] {message}", file = sys.stdout if _LOG_TO_STDOUT else sys.stderr)


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


def should_send_hf_auth(url: str | None) -> bool:
    return parsed_hostname(url) in HF_AUTH_HOSTS


def auth_headers(url: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": "unsloth-studio-llama-prebuilt",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and should_send_github_auth(url):
        headers["Authorization"] = f"Bearer {token}"
        return headers
    # Anonymous huggingface.co fetches share a per-IP rate limit that CI
    # fleets exhaust (HTTP 429), sinking the prebuilt path into a source
    # build. Authenticate when a token is available.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and should_send_hf_auth(url):
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


class _CrossHostAuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host.

    huggingface.co redirects file downloads to CDN hosts whose signed URLs
    can reject foreign Authorization headers; urllib forwards headers to
    redirect targets by default (requests/huggingface_hub strip them).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None and parsed_hostname(newurl) != parsed_hostname(req.full_url):
            new_request.headers.pop("Authorization", None)
            new_request.unredirected_hdrs.pop("Authorization", None)
        return new_request


_URL_OPENER = urllib.request.build_opener(_CrossHostAuthStrippingRedirectHandler())


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
        return candidate_commit.startswith(requested_commit) or requested_commit.startswith(
            candidate_commit
        )
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
    asset_name: str, *, compatibility_tag: str | None = None
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


def _published_windows_cuda_runtime(
    upstream_assets: dict[str, str], major: int, driver: tuple[int, int] | None
) -> str | None:
    """Highest cuda-<major>.<minor> published upstream that `driver` can run by
    default CUDA compatibility, i.e. (major, minor) <= driver. None if nothing
    qualifies. Gating on the driver (not just the major) keeps a 13.3 build off
    a driver that only advertises 13.1, where it would otherwise rely on the
    unguaranteed minor-version-compatibility path."""
    if driver is None:
        return None
    best: int | None = None
    for name in upstream_assets:
        m = re.search(r"-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$", name)
        if m and int(m.group(1)) == major:
            minor = int(m.group(2))
            if (major, minor) <= driver and (best is None or minor > best):
                best = minor
    return f"{major}.{best}" if best is not None else None


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


def _progress_percent_step() -> int:
    """Non-tty milestone granularity. The in-app updater sets
    UNSLOTH_PROGRESS_PERCENT_STEP=5 to stream finer progress lines."""
    try:
        step = int(os.environ.get("UNSLOTH_PROGRESS_PERCENT_STEP", "25"))
    except ValueError:
        return 25
    return min(max(step, 1), 50)


class DownloadProgress:
    def __init__(self, label: str, total_bytes: int | None) -> None:
        self.label = label
        self.total_bytes = total_bytes if total_bytes and total_bytes > 0 else None
        self.start_time = time.monotonic()
        self.last_emit = 0.0
        term_ok = os.environ.get("TERM", "").lower() != "dumb"
        self.stream = (
            sys.stderr if sys.stderr.isatty() else sys.stdout if sys.stdout.isatty() else sys.stderr
        )
        self.is_tty = term_ok and self.stream.isatty()
        self.completed = False
        self.milestone_step = _progress_percent_step()
        self.last_milestone_percent = -1
        self.last_milestone_bytes = 0
        self.has_rendered_tty_progress = False

    def _render(
        self,
        downloaded_bytes: int,
        *,
        final: bool = False,
    ) -> str:
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
                if self.total_bytes is not None and downloaded_bytes >= self.total_bytes:
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
            step = self.milestone_step
            milestone_percent = min((percent // step) * step, 100)
            if milestone_percent > self.last_milestone_percent and milestone_percent < 100:
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
            with _URL_OPENER.open(request, timeout = timeout) as response:
                total_bytes: int | None = None
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total_bytes = int(content_length)
                progress = DownloadProgress(progress_label, total_bytes) if progress_label else None
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
                headers = github_api_headers(url) if is_github_api_url(url) else auth_headers(url),
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and is_github_api_url(url):
                hint = ""
                if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
                    hint = "; set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits"
                raise RuntimeError(f"GitHub API returned 403 for {url}{hint}") from exc
            raise
        if not data:
            last_decode_exc = RuntimeError(f"downloaded empty JSON payload from {url}")
        else:
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_decode_exc = RuntimeError(f"downloaded invalid JSON from {url}: {exc}")
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
                with _URL_OPENER.open(request, timeout = 120) as response:
                    total_bytes: int | None = None
                    content_length = response.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        total_bytes = int(content_length)
                    progress = DownloadProgress(f"Downloading {destination.name}", total_bytes)
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
            log(f"download failed ({attempt}/{HTTP_FETCH_ATTEMPTS}) for {url}: {exc}; retrying")
            sleep_backoff(attempt, exc = exc)
    assert last_exc is not None
    raise last_exc


def download_file_verified(
    url: str, destination: Path, *, expected_sha256: str | None, label: str
) -> None:
    normalized_expected = normalize_sha256_digest(expected_sha256)
    if not normalized_expected:
        download_file(url, destination)
        log(f"downloaded {label} without a published sha256; relying on install validation")
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


def release_asset_download_url(
    repo: str | None, release_tag: str | None, asset_name: str | None
) -> str | None:
    """Direct download URL for a release asset, or None if any part is missing.
    A mix build's merged commit is never pushed, so its source tree is only
    reachable as this asset (codeload would 404 on the merge commit)."""
    if not repo or not release_tag or not asset_name:
        return None
    return (
        f"https://github.com/{repo}/releases/download/"
        f"{urllib.parse.quote(release_tag, safe = '')}/{urllib.parse.quote(asset_name, safe = '')}"
    )


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
        raise RuntimeError(f"latest release tag was missing from {UPSTREAM_RELEASES_API}")
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

    if requested_tag and requested_tag != "latest" and is_release_tag_like(requested_tag):
        try:
            yield github_release(repo, requested_tag)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                log(f"release tag {requested_tag} not found in {repo}; scanning recent releases")
            else:
                raise
        except Exception:
            raise

    releases = [
        release
        for release in github_releases(repo, max_pages = DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES)
        if isinstance(release, dict) and not release.get("draft") and not release.get("prerelease")
    ]
    releases.sort(key = release_time_sort_key, reverse = True)
    for release in releases:
        yield release


def direct_release_matches_request(*, release_tag: str, llama_tag: str, requested_tag: str) -> bool:
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
        r"^app-(?P<label>.+)-(?P<target>linux-x64(?:-cpu)?|linux-x64-cuda\d+-(?:older|newer|portable))\.tar\.gz$"
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
        profile = _resolve_linux_bundle_profile(bundle_profile)
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
    release: dict[str, Any], host: HostInfo, repo: str, requested_tag: str
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
        # Prefer the cudart major Studio loads at runtime (torch's bundled
        # libcudart), not the newest on disk. Otherwise a stray cuda13
        # runtime outranks the torch cuda12 the binary links against.
        torch_preference = detect_torch_cuda_runtime_preference(host)
        selection = linux_cuda_choice_from_release(
            host,
            bundle,
            preferred_runtime_line = torch_preference.runtime_line,
            selection_preamble = torch_preference.selection_log,
        )
        if selection is not None:
            attempts.extend(selection.attempts)
    elif not host.has_rocm:
        # A ROCm-only host gets no CPU asset: leaving attempts empty lets the
        # raise below trigger a HIP source build instead of shipping a CPU
        # binary on a GPU host (this ggml-org path has no per-gfx ROCm asset).
        cpu_choice = published_asset_choice_for_kind(bundle, "linux-cpu")
        if cpu_choice is not None:
            attempts.append(cpu_choice)
    # NVIDIA hosts whose CUDA selection produced nothing fall through to the
    # raise below (mirroring the ROCm policy above): the caller then walks
    # back to an older release that still ships a usable CUDA line instead of
    # silently installing a CPU binary on a GPU host. Today's walk-back only
    # works because partial releases ship no CPU bundle; this keeps it working
    # if a future partial release does.
    if not attempts:
        raise PrebuiltFallback("no compatible Linux prebuilt asset was found")
    approved_checksums = synthetic_checksums_for_release(
        repo,
        bundle.release_tag,
        bundle.upstream_tag,
    )
    resolved_upstream_tag = bundle.upstream_tag
    if DEFAULT_PUBLISHED_SHA256_ASSET in bundle.assets and not is_release_tag_like(
        bundle.upstream_tag
    ):
        approved_checksums = load_approved_release_checksums(repo, bundle.release_tag)
        # Require exact source provenance for branch/pull/commit releases.
        # Mirrors validated_checksums_for_bundle so incomplete metadata fails
        # closed instead of degrading to the legacy branch-as-tag source
        # hydration path this PR eliminates.
        if (
            not approved_checksums.source_commit
            or exact_source_archive_hash(approved_checksums) is None
            or source_clone_url_from_checksums(approved_checksums) is None
        ):
            raise PrebuiltFallback(
                f"approved checksum asset {DEFAULT_PUBLISHED_SHA256_ASSET} for "
                f"{repo}@{bundle.release_tag} did not contain exact source provenance"
            )
        attempts = apply_approved_hashes(attempts, approved_checksums)
    return InstallReleasePlan(
        requested_tag = requested_tag,
        llama_tag = resolved_upstream_tag,
        release_tag = bundle.release_tag,
        attempts = attempts,
        approved_checksums = approved_checksums,
    )


def direct_upstream_release_plan(
    release: dict[str, Any], host: HostInfo, repo: str, requested_tag: str
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
            attempts[:] = _drop_blackwell_incapable_windows_cuda(host, attempts)
        elif host.has_rocm:
            hip_asset = f"llama-{release_tag}-bin-win-hip-radeon-x64.zip"
            hip_url = assets.get(hip_asset)
            if hip_url:
                attempts.append(
                    AssetChoice(
                        repo = repo,
                        tag = release_tag,
                        name = hip_asset,
                        url = hip_url,
                        source_label = "upstream",
                        install_kind = "windows-hip",
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
    elif host.is_windows and host.is_arm64:
        # Upstream ggml-org/llama.cpp ships llama-bNNNN-bin-win-cpu-arm64.zip
        # (visible in the b9334 release manifest). Without this branch the
        # selector returned 0 attempts and the installer fell back to a
        # source build on every Windows ARM64 host.
        cpu_asset = f"llama-{release_tag}-bin-win-cpu-arm64.zip"
        cpu_url = assets.get(cpu_asset)
        if cpu_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = cpu_asset,
                    url = cpu_url,
                    source_label = "upstream",
                    install_kind = "windows-arm64",
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
    elif host.is_linux and host.is_x86_64 and not host.has_usable_nvidia and not host.has_rocm:
        # ROCm hosts are excluded: this ggml-org path ships no per-gfx ROCm
        # asset, so they fall through to the empty-attempts raise (HIP source
        # build) rather than silently getting a CPU binary on a GPU host.
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
    elif host.is_linux and host.is_arm64 and not host.has_usable_nvidia:
        # Upstream ggml-org/llama.cpp ships llama-bNNNN-bin-ubuntu-arm64.tar.gz
        # (visible in the b9334 release manifest). Without this branch the
        # selector returned 0 attempts and the installer fell back to a
        # source build on every Linux ARM64 host (DGX Spark, Ampere
        # Altra, GitHub-hosted ubuntu-24.04-arm runners, etc.).
        asset_name = f"llama-{release_tag}-bin-ubuntu-arm64.tar.gz"
        asset_url = assets.get(asset_name)
        if asset_url:
            attempts.append(
                AssetChoice(
                    repo = repo,
                    tag = release_tag,
                    name = asset_name,
                    url = asset_url,
                    source_label = "upstream",
                    install_kind = "linux-arm64",
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


def pinned_macos_release_tag(host: HostInfo, repo: str) -> str | None:
    if repo != UPSTREAM_REPO or not host.is_macos or host.macos_version is None:
        return None
    if host.macos_version >= _PINNED_MACOS_LATEST_FLOOR:
        return None
    return _PINNED_MACOS_FALLBACK_TAG


def resolve_simple_install_release_plans(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
    *,
    max_release_fallbacks: int = DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS,
) -> tuple[str, list[InstallReleasePlan]]:
    repo = published_repo or DEFAULT_PUBLISHED_REPO
    # The fork (unslothai) ships a manifest describing every bundle's GPU/arch
    # coverage, so all fork hosts select from it. Upstream (ggml-org) ships no
    # manifest and is selected by asset filename in the loop below.
    if repo == DEFAULT_PUBLISHED_REPO:
        return _fork_manifest_release_plans(
            llama_tag,
            host,
            published_repo,
            published_release_tag,
            max_release_fallbacks = max_release_fallbacks,
        )
    requested_tag = normalized_requested_llama_tag(llama_tag)
    allow_older_release_fallback = requested_tag == "latest" and not published_release_tag
    if allow_older_release_fallback:
        pinned_macos = pinned_macos_release_tag(host, repo)
        if pinned_macos is not None:
            requested_tag = pinned_macos
            allow_older_release_fallback = False
    release_limit = max(1, max_release_fallbacks)
    plans: list[InstallReleasePlan] = []
    last_error: PrebuiltFallback | None = None

    try:
        releases = iter_release_payloads_by_time(repo, published_release_tag, requested_tag)
        for release in releases:
            try:
                plan = direct_upstream_release_plan(release, host, repo, requested_tag)
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
        raise PrebuiltFallback(f"failed to inspect published releases in {repo}: {exc}") from exc

    if plans:
        return requested_tag, plans
    if last_error is not None:
        raise last_error
    raise PrebuiltFallback(f"no installable published llama.cpp releases were found in {repo}")


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


def supports_explicit_visible_device_matching(visible_devices: list[str] | None) -> bool:
    if not visible_devices:
        return False
    for token in visible_devices:
        lowered = token.lower()
        if token.isdigit() or lowered.startswith("gpu-"):
            continue
        return False
    return True


def select_visible_gpu_rows(
    gpu_rows: Iterable[tuple[str, str, str]], visible_devices: list[str] | None
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


def linux_runtime_dirs_for_required_libraries(required_libraries: Iterable[str]) -> list[str]:
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
    cuda_roots.extend(Path(path) for path in glob_paths("/usr/local/cuda", "/usr/local/cuda-*"))

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
        Path(path) for path in glob_paths("/usr/local/lib/ollama/cuda_v*", "/usr/lib/wsl/lib")
    )
    candidates.extend(Path(path) for path in python_runtime_dirs())
    candidates.extend(Path(path) for path in ldconfig_runtime_dirs(required))

    resolved = dedupe_existing_dirs(candidates)
    if not required:
        return resolved

    matched: list[tuple[int, str]] = []
    for directory in resolved:
        base = Path(directory)
        provided = sum(1 for library in required if dir_provides_exact_library(directory, library))
        if provided:
            matched.append((provided, directory))

    matched.sort(key = lambda item: item[0], reverse = True)
    return [directory for _, directory in matched]


def detected_linux_runtime_lines() -> tuple[list[str], dict[str, list[str]]]:
    line_requirements = {
        f"cuda{m}": [f"libcudart.so.{m}", f"libcublas.so.{m}"]
        for m in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1)
    }
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for line, required in line_requirements.items():
        dirs = linux_runtime_dirs_for_required_libraries(required)
        library_matches: dict[str, list[str]] = {}
        matching_dirs: list[str] = []
        for library in required:
            matched_dirs = [
                directory for directory in dirs if any(Path(directory).glob(f"{library}*"))
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
        raise ValueError(f"artifact {asset_name} install_kind was missing or not a string")

    supported_sms_raw = raw.get("supported_sms", [])
    if not isinstance(supported_sms_raw, (list, tuple)):
        raise ValueError(f"artifact {asset_name} supported_sms must be a list or tuple")
    if any(not isinstance(value, (int, str)) for value in supported_sms_raw):
        raise ValueError(f"artifact {asset_name} supported_sms entries must be ints or strings")
    supported_sms = normalize_compute_caps(supported_sms_raw)

    min_sm_raw = raw.get("min_sm")
    max_sm_raw = raw.get("max_sm")
    try:
        min_sm = int(min_sm_raw) if min_sm_raw is not None else None
        max_sm = int(max_sm_raw) if max_sm_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact {asset_name} min_sm/max_sm were not integers") from exc
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
    gfx_target_raw = raw.get("gfx_target")
    gfx_target = (
        gfx_target_raw.strip()
        if isinstance(gfx_target_raw, str) and gfx_target_raw.strip()
        else None
    )
    mapped_raw = raw.get("mapped_targets", [])
    mapped_targets = (
        [value.strip() for value in mapped_raw if isinstance(value, str) and value.strip()]
        if isinstance(mapped_raw, (list, tuple))
        else []
    )
    return PublishedLlamaArtifact(
        asset_name = asset_name,
        install_kind = install_kind,
        runtime_line = runtime_line if isinstance(runtime_line, str) and runtime_line else None,
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
        gfx_target = gfx_target,
        mapped_targets = mapped_targets,
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
            log(f"published artifact ignored for {repo}@{release_tag} artifact[{index}]: {exc}")
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
        source_repo = source_repo if isinstance(source_repo, str) and source_repo else None,
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
    repo: str, release_tag: str, payload: Any
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
            raise RuntimeError("published checksum asset used a non-string artifact key")
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"published checksum entry for {asset_name} was not an object")
        digest = normalize_sha256_digest(raw_entry.get("sha256"))
        if not digest:
            raise RuntimeError(f"published checksum entry for {asset_name} omitted a valid sha256")
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
        source_repo = source_repo if isinstance(source_repo, str) and source_repo else None,
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


def load_approved_release_checksums(repo: str, release_tag: str) -> ApprovedReleaseChecksums:
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
        if not published_release_tag and (release.get("draft") or release.get("prerelease")):
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


def _artifact_covers_sms(artifact: PublishedLlamaArtifact, host_sms: Iterable[str]) -> bool:
    """True when every host SM is listed in the artifact's supported_sms and
    falls within its [min_sm, max_sm] range."""
    if not artifact.supported_sms or artifact.min_sm is None or artifact.max_sm is None:
        return False
    supported = {str(value) for value in artifact.supported_sms}
    return all(sm in supported and artifact.min_sm <= int(sm) <= artifact.max_sm for sm in host_sms)


def _sm_range(artifact: PublishedLlamaArtifact) -> int:
    """SM-coverage span used as a sort key, where a tighter (smaller) range wins.
    A bundle with no SM metadata (legacy/upstream-named) gets a max range so it
    sorts last and can't outrank a real targeted bundle whose tight range would
    otherwise sort first."""
    if artifact.min_sm is not None and artifact.max_sm is not None:
        return artifact.max_sm - artifact.min_sm
    return 9999


def _blackwell_capable_linux_runtime_lines(
    host_sms: list[str], artifacts: list[PublishedLlamaArtifact]
) -> list[str]:
    """CUDA runtime lines (highest major first) shipping a bundle that covers every
    visible host SM. Lets a Blackwell host prefer a native sm_120 line over torch's
    reported line, mirroring the Windows Blackwell preference."""
    lines: set[str] = set()
    for artifact in artifacts:
        line = artifact.runtime_line
        # Only rank "cuda<major>" lines; ignore malformed/future-format values
        # (e.g. "cuda13.1") so they are skipped, as pre-existing code does, rather
        # than crashing the major sort.
        if not (line and line.startswith("cuda") and line[len("cuda") :].isdigit()):
            continue
        if not artifact.supported_sms or artifact.min_sm is None or artifact.max_sm is None:
            continue
        supported = {str(value) for value in artifact.supported_sms}
        if all(
            sm in supported and artifact.min_sm <= int(sm) <= artifact.max_sm for sm in host_sms
        ):
            lines.add(line)
    return sorted(lines, key = lambda line: int(line[len("cuda") :]), reverse = True)


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
    # arm64 CUDA hosts (DGX Spark / Grace Hopper) consume linux-arm64-cuda
    # bundles; x64 hosts consume linux-cuda. The SM / runtime-line matching
    # below is arch-agnostic and applies to both.
    cuda_install_kind = "linux-arm64-cuda" if host.is_arm64 else "linux-cuda"
    published_artifacts = [
        artifact for artifact in release.artifacts if artifact.install_kind == cuda_install_kind
    ]
    published_asset_names = sorted(artifact.asset_name for artifact in published_artifacts)
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

    blackwell_lines = (
        [
            line
            for line in _blackwell_capable_linux_runtime_lines(host_sms, published_artifacts)
            if line in ordered_runtime_lines
        ]
        if _host_is_blackwell(host)
        else []
    )
    if blackwell_lines:
        # Blackwell host: prefer the highest CUDA-major line shipping an sm_120 bundle
        # over torch's line (matches the Windows Blackwell preference).
        ordered_runtime_lines = blackwell_lines + [
            line for line in ordered_runtime_lines if line not in blackwell_lines
        ]
        selection_log.append(
            "linux_cuda_selection: blackwell_runtime_override prefer="
            + ",".join(blackwell_lines)
            + (f" over torch_preferred={preferred_runtime_line}" if preferred_runtime_line else "")
        )
    elif preferred_runtime_line:
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

    def add_attempt(artifact: PublishedLlamaArtifact, asset_url: str, reason: str) -> None:
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
                install_kind = cuda_install_kind,
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
                selection_log.append(f"linux_cuda_selection: reject {asset_name} missing asset")
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
                sm for sm in host_sms if not (artifact.min_sm <= int(sm) <= artifact.max_sm)
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
                    _sm_range(item[0]),
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
        "linux_cuda_selection: attempt_order=" + ",".join(choice.name for choice in attempts)
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
    for release in github_releases(UPSTREAM_REPO, max_pages = DEFAULT_GITHUB_RELEASE_SCAN_MAX_PAGES):
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
    elif source_clone_url_for_release(checksums, bundle) is None:
        # No source repo in either the checksum payload or the manifest bundle:
        # preferred_source_archive would silently fall back to ggml-org source at
        # the upstream tag, pairing the prebuilt with a possibly mismatched tree.
        # Fail closed so the resolver skips this release.
        raise PrebuiltFallback(
            f"approved checksum asset for {repo}@{bundle.release_tag} declared an "
            "exact source archive without a source repo to clone it from"
        )
    return checksums


def published_release_matches_request(bundle: PublishedReleaseBundle, requested_ref: str) -> bool:
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
        raise PrebuiltFallback(f"no published llama.cpp releases were available in {repo}")

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
        raise PrebuiltFallback(f"no published llama.cpp releases were available in {repo}")

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


def exact_source_archive_hash(checksums: ApprovedReleaseChecksums) -> ApprovedArtifactHash | None:
    if not checksums.source_commit:
        return None
    return checksums.artifacts.get(exact_source_archive_logical_name(checksums.source_commit))


def source_clone_url_for_release(
    checksums: ApprovedReleaseChecksums, bundle: PublishedReleaseBundle
) -> str | None:
    # Single source of truth for "where do we clone source from", shared by the
    # validation gate and source_build_plan_for_release: take the repo from the
    # checksum payload, falling back field by field to the manifest bundle.
    return source_repo_clone_url(
        checksums.source_repo or bundle.source_repo,
        checksums.source_repo_url or bundle.source_repo_url,
    )


def source_build_plan_for_release(release: ResolvedPublishedRelease) -> SourceBuildPlan:
    checksums = release.checksums
    exact_source = exact_source_archive_hash(checksums)
    source_repo = checksums.source_repo or release.bundle.source_repo
    source_repo_url = checksums.source_repo_url or release.bundle.source_repo_url
    requested_source_ref = checksums.requested_source_ref or release.bundle.requested_source_ref
    resolved_source_ref = checksums.resolved_source_ref or release.bundle.resolved_source_ref
    source_commit = checksums.source_commit or release.bundle.source_commit
    source_ref_kind = checksums.source_ref_kind or release.bundle.source_ref_kind
    source_url = source_clone_url_for_release(checksums, release.bundle)
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
    source_ref = checkout_friendly_ref(source_ref_kind, resolved_source_ref or requested_source_ref)
    if source_url and source_ref and source_ref_kind in {"tag", "branch", "pull", "commit"}:
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


def _subprocess_failure(command: list[str], exc: BaseException) -> PrebuiltFallback:
    name = command[0] if command else "subprocess"
    if isinstance(exc, FileNotFoundError):
        return PrebuiltFallback(f"{name} was not found")
    if isinstance(exc, PermissionError):
        return PrebuiltFallback(f"{name} was not executable")
    if isinstance(exc, subprocess.TimeoutExpired):
        return PrebuiltFallback(f"{name} timed out after {exc.timeout} seconds")
    return PrebuiltFallback(f"{name} launch failed: {exc}")


def _run_subprocess_capture(
    command: list[str],
    *,
    timeout: int = 30,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    # amd-smi on Windows auto-elevates and pops a UAC/DiskPart prompt mid-install;
    # RunAsInvoker forces it un-elevated. Callers already fall back to WMI/name
    # detection. Mirrors install.ps1's Invoke-AmdSmiNoElevate; Windows-only.
    if (
        command
        and platform.system() == "Windows"
        and os.path.basename(command[0]).lower().startswith("amd-smi")
    ):
        env = {**(os.environ if env is None else env), "__COMPAT_LAYER": "RunAsInvoker"}
    try:
        result = subprocess.run(
            command,
            capture_output = True,
            text = True,
            timeout = timeout,
            env = env,
            **windows_hidden_subprocess_kwargs(),
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as exc:
        raise _subprocess_failure(command, exc) from exc
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )
    return result


def run_capture(
    command: list[str],
    *,
    timeout: int = 30,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_subprocess_capture(command, timeout = timeout, check = check, env = env)


def _pick_rocm_gfx_target(out: str) -> str | None:
    """Choose the gfx target rocminfo / hipinfo report for the active GPU.

    A bare first-match picked the wrong device on mixed APU + dGPU hosts
    (e.g. Strix Halo gfx1151 + discrete RX 7900 gfx1100). Respect
    HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES so the
    asset matches what HIP actually runs on. Falls back to the first GPU when
    no env var is set.

    rocminfo / hipinfo print the same gfx token multiple times per GPU (Name,
    ISA, marketing-name). We first try to split the output on per-GPU section
    headers (rocminfo: "Agent N" blocks, hipinfo: "device#N" entries) and take
    exactly one gfx token per section. This gives the correct per-GPU list even
    on same-arch multi-GPU hosts (e.g. two RX 7900 XTX cards) where global
    dict.fromkeys dedup would collapse both cards to a single entry and make
    HIP_VISIBLE_DEVICES=1 point out of range.

    Falls back to insertion-order dedup when the output has no recognisable
    section markers (flat gfx-string inputs, unit-test stubs, etc.).

    Empty / "-1" env values mean no AMD GPU is visible to HIP: return None.
    """
    # Try to build a per-GPU token list by splitting on section boundaries.
    # rocminfo sections are introduced by "Agent N" lines (optionally between
    # rows of asterisks). hipinfo sections start with "device#N".
    _sections = re.split(
        r"(?mi)^\s*\*+\s*$\s*agent\s+\d+\s*$|\bdevice\s*#\s*\d+\b",
        out,
    )
    if len(_sections) > 1:
        # Section-based: one gfx token per GPU section preserves physical order.
        _tokens: list[str] = []
        for _sec in _sections[1:]:
            _m = re.search(r"gfx[1-9][0-9a-z]{2,3}", _sec.lower())
            if _m:
                _tokens.append(_m.group(0))
    else:
        # Fallback: insertion-order dedup (handles flat strings / unknown formats).
        _raw = re.findall(r"gfx[1-9][0-9a-z]{2,3}", out.lower())
        _tokens = list(dict.fromkeys(_raw))

    if not _tokens:
        return None

    _vis_raw = None
    # AMD's HIP runtime honours all three env vars with identical semantics.
    for _env in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        _val = os.environ.get(_env)
        if _val is not None:
            _vis_raw = _val
            break
    if _vis_raw is not None:
        _vis = _vis_raw.strip()
        # Empty or "-1" means "no AMD GPU visible" (matches the rest of Studio).
        if _vis == "" or _vis == "-1":
            return None
        _first = _vis.split(",")[0].strip()
        try:
            _idx = int(_first)
            if 0 <= _idx < len(_tokens):
                return _tokens[_idx]
        except ValueError:
            pass
    return _tokens[0]


def detect_host() -> HostInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_macos = system == "Darwin"
    is_x86_64 = machine in {"x86_64", "amd64"}
    is_arm64 = machine in {"arm64", "aarch64"}

    macos_version = parse_macos_version(platform.mac_ver()[0]) if is_macos else None

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
            gpu_lines = [line for line in listing.stdout.splitlines() if line.startswith("GPU ")]
            if gpu_lines:
                has_physical_nvidia = True
                has_usable_nvidia = visible_device_tokens != []
        except Exception:
            pass

        try:
            result = run_capture([nvidia_smi], timeout = 20)
            merged = "\n".join(part for part in (result.stdout, result.stderr) if part)
            # Newer NVIDIA drivers (e.g. 610.x on Windows) print
            # "CUDA UMD Version: X.Y" instead of the legacy
            # "CUDA Version: X.Y"; accept both spellings.
            cuda_match = re.search(
                r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)",
                merged,
            )
            if cuda_match is not None:
                driver_cuda_version = (
                    int(cuda_match.group(1)),
                    int(cuda_match.group(2)),
                )
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

    # Linux /proc/driver/nvidia/gpus fallback: the NVIDIA driver exposes one
    # subdir per GPU here regardless of nvidia-smi state, so a host whose
    # nvidia-smi is absent from PATH, wedged, or failing is still recognised as
    # NVIDIA. Mirrors the fallback added to install.sh / install_python_stack.py
    # in PR 6174 so the prebuilt installer does not misroute such hosts to ROCm
    # or CPU. driver_cuda_version / compute_caps stay unset here; downstream
    # CUDA asset selection treats unknown SMs as "prefer portable" and an
    # unknown driver runtime line as "no published CUDA match" (returns None,
    # no crash), so planning falls back to a source build with GGML_CUDA=ON.
    if is_linux and not has_physical_nvidia:
        try:
            proc_gpu_dir = "/proc/driver/nvidia/gpus"
            if os.path.isdir(proc_gpu_dir) and os.listdir(proc_gpu_dir):
                has_physical_nvidia = True
                has_usable_nvidia = visible_device_tokens != []
        except OSError:
            pass

    # Detect AMD ROCm (HIP) -- require actual GPU, not just tools installed
    # NVIDIA takes precedence: when an NVIDIA GPU is usable, skip ROCm probing
    # entirely so co-installed ROCm tools cannot misroute the host (PR 6174).

    def _amd_smi_has_gpu(stdout: str) -> bool:
        """Check for 'GPU: <number>' data rows, not just a table header."""
        return bool(re.search(r"(?im)^gpu\s*[:\[]\s*\d", stdout))

    has_rocm = False
    rocm_gfx_target: str | None = None
    if is_linux and not has_usable_nvidia:
        # WSL2 ROCDXG: the system rocminfo enumerates the GPU over /dev/dxg
        # only when HSA_ENABLE_DXG_DETECTION=1 (a no-op on bare metal), and
        # rocminfo can live only under /opt/rocm/bin (the profile.d PATH
        # drop-in reaches login shells only). Probe accordingly or a ROCDXG
        # WSL host is misdetected as CPU-only.
        _dxg_probe_env = {**os.environ}
        _dxg_probe_env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
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
            if not _exe and _cmd[0] == "rocminfo":
                _opt_rocminfo = "/opt/rocm/bin/rocminfo"
                if os.access(_opt_rocminfo, os.X_OK):
                    _exe = _opt_rocminfo
            if not _exe:
                continue
            try:
                _result = run_capture(
                    [_exe, *_cmd[1:]],
                    timeout = 10,
                    env = _dxg_probe_env if _cmd[0] == "rocminfo" else None,
                )
            except Exception:
                continue
            if _result.returncode == 0 and _result.stdout.strip():
                if _check(_result.stdout):
                    has_rocm = True
                    rocm_gfx_target = _pick_rocm_gfx_target(_result.stdout)
                    break
    elif is_windows and not has_usable_nvidia:
        # Windows: prefer active probes that validate GPU presence.
        # hipinfo / amd-smi are often NOT on PATH -- the HIP SDK installer
        # sets HIP_PATH / ROCM_PATH but does not always add the bin dir to
        # the system PATH.  Mirror setup.ps1's fallback: check the env-var
        # bin dirs before giving up so that `has_rocm` is not silently False
        # on machines where the PATH is not yet updated.
        def _resolve_exe(name: str) -> str | None:
            """Return full path to `name`, checking PATH then HIP_PATH/ROCM_PATH bin."""
            found = shutil.which(name)
            if found:
                return found
            for _env in ("HIP_PATH", "ROCM_PATH"):
                _root = os.environ.get(_env)
                if _root:
                    _candidate = os.path.join(_root, "bin", f"{name}.exe")
                    if os.path.isfile(_candidate):
                        return _candidate
            # AMD torch wheels ship hipInfo.exe into the venv Scripts dir
            # (next to python.exe) -- resolvable on driver-only hosts where no
            # SDK dir exists, so a standalone rerun can still detect the GPU.
            _venv_candidate = os.path.join(os.path.dirname(sys.executable), f"{name}.exe")
            if os.path.isfile(_venv_candidate):
                return _venv_candidate
            return None

        _win_probes = [(["hipinfo"], lambda out: "gcnarchname" in out.lower())]
        if _amd_smi_allowed():
            # Skipped on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt);
            # gfx arch still arrives via --rocm-gfx, so has_rocm is set by override.
            _win_probes.append((["amd-smi", "list"], _amd_smi_has_gpu))
        for _cmd, _check in _win_probes:
            _exe = _resolve_exe(_cmd[0])
            if not _exe:
                continue
            try:
                _result = run_capture([_exe, *_cmd[1:]], timeout = 10)
            except Exception:
                continue
            if _result.returncode == 0 and _result.stdout.strip():
                if _check(_result.stdout):
                    has_rocm = True
                    # hipinfo reports "gcnArchName: gfx1100" -- extract if present
                    rocm_gfx_target = _pick_rocm_gfx_target(_result.stdout)
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
        rocm_gfx_target = rocm_gfx_target,
        macos_version = macos_version,
    )


def _normalize_forwarded_gfx(value: str | None) -> str | None:
    """Extract a single gfx token from a forwarded --rocm-gfx / env value.
    setup.sh/setup.ps1 already picked the active GPU, so take the token as-is
    without re-applying visible-device selection. Ignore anything malformed."""
    if not value:
        return None
    m = re.search(r"gfx[1-9][0-9a-z]{2,3}", value.lower())
    return m.group(0) if m else None


def _apply_host_overrides(
    host: HostInfo,
    *,
    override_has_rocm: bool = False,
    override_rocm_gfx: str | None = None,
    force_cpu: bool = False,
) -> HostInfo:
    """Fold setup.sh/setup.ps1's forwarded detection into the host profile.
    A forwarded gfx (--rocm-gfx or UNSLOTH_ROCM_GFX_ARCH) is authoritative and
    implies ROCm: the installer's own hipinfo/amd-smi probe can miss the arch on
    amd-smi-only hosts or when setup inferred it from the GPU name, leaving
    rocm_gfx_target None and no per-gfx ROCm prebuilt selected. force_cpu is the
    opposite explicit signal (arm64 Linux GPU host whose source build failed):
    drop GPU attributes so the CPU prebuilt for this OS/arch is selected."""
    if force_cpu:
        return dataclasses_replace(
            host,
            has_usable_nvidia = False,
            has_physical_nvidia = False,
            has_rocm = False,
            rocm_gfx_target = None,
        )
    gfx = _normalize_forwarded_gfx(override_rocm_gfx)
    if gfx:
        return dataclasses_replace(host, has_rocm = True, rocm_gfx_target = gfx)
    if override_has_rocm and not host.has_rocm:
        return dataclasses_replace(host, has_rocm = True)
    return host


def published_repo_for_host(host: HostInfo, *, linux_amd_tooling_present: bool = False) -> str:
    """The release repo setup.sh / setup.ps1 pick for this host: macOS always the
    fork (ggml-org macOS bundles need too-new macOS); else CPU-only Linux/Windows
    -> ggml-org upstream (the fork ships no CPU bundle) and any usable GPU (NVIDIA
    or ROCm) -> the fork. linux_amd_tooling_present mirrors setup.sh routing Linux
    hosts that expose AMD tooling (rocminfo/amd-smi/hipconfig/hipinfo) to the fork
    even when the probe cannot confirm an active GPU. Mirrors the shell routing."""
    if host.is_macos:
        return DEFAULT_PUBLISHED_REPO
    has_gpu = (
        host.has_usable_nvidia or host.has_rocm or (host.is_linux and linux_amd_tooling_present)
    )
    return DEFAULT_PUBLISHED_REPO if has_gpu else UPSTREAM_REPO


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
    if major < _MIN_CUDA_MAJOR:
        return []
    return _cuda_runtime_lines_for_major(major)


def windows_runtime_line_info() -> dict[str, tuple[str, ...]]:
    # Generated per CUDA major (newest first) so a new toolkit is detected
    # without a code change while the cudart64_<major>.dll naming holds.
    return {
        f"cuda{m}": (
            f"cudart64_{m}*.dll",
            f"cublas64_{m}*.dll",
            f"cublasLt64_{m}*.dll",
        )
        for m in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1)
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
    if not host.driver_cuda_version:
        return []
    major, _minor = host.driver_cuda_version
    # cuda12 app bundles are toolkit-12.8 builds with bundled runtime libs; CUDA
    # minor-version compatibility runs them on any 12.x driver, same as Linux.
    if major < _MIN_CUDA_MAJOR:
        return []
    return _cuda_runtime_lines_for_major(major)


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
        selection_log.append(f"torch_cuda_preference: torch.cuda.is_available() failed: {exc}")
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
        normal_runtime_lines = compatible_runtime_lines

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
        selection_log.append("windows_cuda_selection: no Torch runtime preference available")

    runtime_order.extend(
        runtime_line for runtime_line in normal_runtime_lines if runtime_line not in runtime_order
    )
    # Keep every driver-compatible line reachable as a fallback, so a line gated
    # out by the driver version still drops to an older major (cuda13 -> cuda12).
    runtime_order.extend(
        runtime_line
        for runtime_line in compatible_runtime_lines
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
        major = int(runtime_line.removeprefix("cuda"))
        # Track whatever minor llama.cpp actually ships for this major
        # (cuda13 -> 13.1, 13.3, ...). Skip the line when the release has no
        # matching asset instead of guessing a now-missing name.
        runtime = _published_windows_cuda_runtime(upstream_assets, major, host.driver_cuda_version)
        if runtime is None:
            selection_log.append(
                f"windows_cuda_selection: no driver-supported asset for {runtime_line}"
            )
            continue
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
        if selected_name.startswith("llama-"):
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


def _windows_cuda_attempt_covers_blackwell(
    attempt: AssetChoice, min_toolkit: tuple[int, int] = _BLACKWELL_MIN_TOOLKIT
) -> bool:
    """True if a windows-cuda attempt is Blackwell-capable (app bundles via
    declared SMs; legacy upstream bundles via toolkit minor >= min_toolkit:
    12.8 for the family, 12.9 for sm_103/sm_121)."""
    if attempt.install_kind != "windows-cuda":
        return False
    # Legacy bundle: the toolkit minor binds (12.4 cannot offload Blackwell).
    m = re.search(r"-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$", attempt.name)
    if m is not None:
        return (int(m.group(1)), int(m.group(2))) >= min_toolkit
    # App-named bundles carry no minor and declare their SM coverage directly.
    return attempt.max_sm is not None and attempt.max_sm >= _BLACKWELL_MIN_SM


def _host_is_blackwell(host: HostInfo) -> bool:
    caps = normalize_compute_caps(host.compute_caps)
    return bool(caps) and int(caps[-1]) >= _BLACKWELL_MIN_SM


def _blackwell_min_toolkit_for_host(host: HostInfo) -> tuple[int, int]:
    """Minimum CUDA toolkit this Blackwell host needs: 12.8 for the family,
    12.9 if any of its SMs is sm_103/sm_121 (no native target before 12.9)."""
    req = _BLACKWELL_MIN_TOOLKIT
    for sm in normalize_compute_caps(host.compute_caps):
        req = max(req, _BLACKWELL_SM_MIN_TOOLKIT.get(int(sm), _BLACKWELL_MIN_TOOLKIT))
    return req


def _drop_blackwell_incapable_windows_cuda(
    host: HostInfo, attempts: list[AssetChoice]
) -> list[AssetChoice]:
    """On a Blackwell host, drop windows-cuda attempts that can't offload
    Blackwell (e.g. cuda-12.4): they load and validate but run a slow non-native
    path (RTX 5090: 7.1 vs 551.2 tok/s on cuda-13.3). Non-cuda attempts pass
    through so the host can still fall back to an honest CPU install."""
    if not _host_is_blackwell(host):
        return attempts
    min_toolkit = _blackwell_min_toolkit_for_host(host)
    return [
        attempt
        for attempt in attempts
        if attempt.install_kind != "windows-cuda"
        or _windows_cuda_attempt_covers_blackwell(attempt, min_toolkit)
    ]


def published_windows_cuda_attempts(
    host: HostInfo,
    release: PublishedReleaseBundle,
    preferred_runtime_line: str | None,
    selection_preamble: Iterable[str] = (),
) -> list[AssetChoice]:
    selection_log = list(release.selection_log) + list(selection_preamble)
    published_artifacts = [
        artifact for artifact in release.artifacts if artifact.install_kind == "windows-cuda"
    ]
    artifacts_by_runtime: dict[str, list[PublishedLlamaArtifact]] = {}
    for artifact in published_artifacts:
        if not artifact.runtime_line:
            continue
        artifacts_by_runtime.setdefault(artifact.runtime_line, []).append(artifact)

    # Order the runtime lines to try. Legacy upstream-named bundles encode a CUDA
    # minor in the filename and are driver-gated per minor. The fork's app-named
    # bundles carry no minor: their runtime line (cuda12/cuda13) is gated at the
    # CUDA *major* level (a 13.0 driver runs any cuda13 build) and by what torch
    # actually provides on the host, preferring the torch line. Routing them
    # through the synthetic-minor path wrongly dropped cuda13 on a 13.0 driver.
    legacy_minors: list[str] = []
    for artifact in published_artifacts:
        m = re.search(r"-bin-win-cuda-(\d+\.\d+)-x64\.zip$", artifact.asset_name)
        if m:
            legacy_minors.append(m.group(1))
    if legacy_minors:
        ordered_lines = [
            attempt.runtime_line
            for attempt in windows_cuda_attempts(
                host,
                release.upstream_tag,
                {
                    f"llama-{release.upstream_tag}-bin-win-cuda-{minor}-x64.zip": "published"
                    for minor in legacy_minors
                },
                preferred_runtime_line,
                selection_log,
            )
            if attempt.runtime_line
        ]
    else:
        detected, _ = detected_windows_runtime_lines()
        compatible = compatible_windows_runtime_lines(host)
        # Prefer lines whose runtime DLLs are on disk, but fall back to the
        # driver-derived order when none are detected (Windows torch bundles
        # cudart in torch/lib, which probing misses) or when detected DLLs are
        # incompatible with the driver. The app bundle ships its own runtime, so
        # the driver major is the real constraint. Mirrors the legacy
        # windows_cuda_attempts fallback; without it a torch-only host gets no
        # fork attempt and silently drops to the upstream build.
        ordered_lines = [line for line in compatible if line in detected] or list(compatible)
        if preferred_runtime_line and preferred_runtime_line in ordered_lines:
            ordered_lines = [preferred_runtime_line] + [
                line for line in ordered_lines if line != preferred_runtime_line
            ]
        selection_log.append(
            "windows_cuda_selection: app-bundle runtime lines (major-gated)="
            + (",".join(ordered_lines) if ordered_lines else "none")
        )

    host_sms = normalize_compute_caps(host.compute_caps)
    attempts: list[AssetChoice] = []
    for runtime_line in ordered_lines:
        if not runtime_line:
            continue
        # Pick the artifact whose SM coverage fits the host, preferring the
        # tightest targeted bundle and falling back to portable -- the same
        # policy as linux_cuda_choice_from_release. Without this, app-named
        # bundles (no minor in the filename) skip the SM filter and the
        # lowest-rank "older" bundle is chosen for every host, breaking newer
        # GPUs (e.g. Blackwell sm120 on a cuda12-older bundle capped at sm89).
        targeted: list[tuple[PublishedLlamaArtifact, str, re.Match[str] | None]] = []
        portable: tuple[PublishedLlamaArtifact, str, re.Match[str] | None] | None = None
        for artifact in artifacts_by_runtime.get(runtime_line, []):
            asset_url = release.assets.get(artifact.asset_name)
            if not asset_url:
                continue
            am = re.search(r"-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$", artifact.asset_name)
            # Legacy upstream-named bundles encode the minor; gate it against the
            # driver. app-named bundles carry no minor and are driver-gated at the
            # runtime-line level by windows_cuda_attempts above.
            if (
                am is not None
                and host.driver_cuda_version is not None
                and (int(am.group(1)), int(am.group(2))) > host.driver_cuda_version
            ):
                continue
            # Only SM-filter artifacts that declare full SM metadata (the app
            # bundles). Legacy/upstream-named artifacts without it keep the old
            # rank-based selection rather than being dropped.
            has_sm_info = (
                bool(artifact.supported_sms)
                and artifact.min_sm is not None
                and artifact.max_sm is not None
            )
            if host_sms and has_sm_info and not _artifact_covers_sms(artifact, host_sms):
                continue
            if not host_sms and has_sm_info and artifact.coverage_class != "portable":
                continue
            if artifact.coverage_class == "portable":
                portable = (artifact, asset_url, am)
            else:
                targeted.append((artifact, asset_url, am))
        chosen: tuple[PublishedLlamaArtifact, str, re.Match[str] | None] | None = None
        if targeted:
            chosen = sorted(
                targeted,
                key = lambda item: (
                    _sm_range(item[0]),
                    item[0].rank,
                    item[0].max_sm or 0,
                ),
            )[0]
        elif portable is not None:
            chosen = portable
        if chosen is None:
            continue
        artifact, asset_url, am = chosen
        # See windows_cuda_attempts: pair the cudart bundle for the real minor.
        runtime_archive_name: str | None = None
        runtime_archive_url: str | None = None
        if am is not None and artifact.asset_name.startswith("llama-"):
            runtime = f"{am.group(1)}.{am.group(2)}"
            cudart_name = f"cudart-llama-bin-win-cuda-{runtime}-x64.zip"
            cudart_url = release.assets.get(cudart_name)
            if cudart_url and cudart_url != asset_url:
                runtime_archive_name = cudart_name
                runtime_archive_url = cudart_url
        attempt_log = list(selection_log) + [
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
                bundle_profile = artifact.bundle_profile,
                coverage_class = artifact.coverage_class,
                supported_sms = artifact.supported_sms,
                min_sm = artifact.min_sm,
                max_sm = artifact.max_sm,
                selection_log = attempt_log,
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
    release: PublishedReleaseBundle, install_kind: str
) -> AssetChoice | None:
    candidates = sorted(
        (artifact for artifact in release.artifacts if artifact.install_kind == install_kind),
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
            + [f"published_selection: selected {artifact.asset_name} install_kind={install_kind}"],
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
    amd_smi = shutil.which("amd-smi") if _amd_smi_allowed() else None
    if amd_smi:
        try:
            # Off on Windows w/o a HIP SDK (avoids the UAC/DiskPart prompt);
            # hipconfig below and the version-file reads above cover that case.
            result = run_capture([amd_smi, "version"], timeout = 5)
            if result.returncode == 0:
                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = run_capture([hipconfig, "--version"], timeout = 5)
            if result.returncode == 0:
                raw = (result.stdout or "").strip().split("\n")[0]
                parts = raw.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].split("-")[0].isdigit():
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
            _result = run_capture([_exe, *_cmd[1:]], timeout = 5)
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


def published_rocm_choice_for_host(
    release: PublishedReleaseBundle, host: HostInfo, install_kind: str
) -> AssetChoice | None:
    """Select the published ROCm bundle whose gfx target covers the host GPU.

    The manifest's gfx_target uses umbrella family labels (gfx110X, gfx120X,
    ...). A host's detected gfx is matched against the bundle's concrete
    mapped_targets list, or against the family label itself. Returns None when no
    published bundle covers the GPU, so the caller falls back to a HIP source
    build."""
    if not host.rocm_gfx_target:
        return None
    gfx = host.rocm_gfx_target.lower().strip()
    for artifact in release.artifacts:
        if artifact.install_kind != install_kind:
            continue
        # Match the concrete built-arch list so an in-generation-but-unbuilt arch
        # (e.g. gfx1033 in the gfx103 family) is NOT served the family bundle and
        # falls back to a source build. Also accept the family label itself: the
        # llama.cpp update path re-derives --rocm-gfx from the family-named marker
        # asset, so an update forwards the family token (gfx110X), not a concrete
        # arch.
        mapped = {target.lower() for target in artifact.mapped_targets}
        if gfx not in mapped and gfx != (artifact.gfx_target or "").lower():
            continue
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
            selection_log = list(release.selection_log)
            + [
                f"rocm_selection: gpu={host.rocm_gfx_target} "
                f"selected published {artifact.asset_name}"
            ],
        )
    return None


def resolve_upstream_asset_choice(host: HostInfo, llama_tag: str) -> AssetChoice:
    upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)
    if host.is_linux and host.is_x86_64:
        # AMD ROCm: try upstream ROCm prebuilt first, then fall back to source build.
        # Source build (via setup.sh) compiles with -DGGML_HIP=ON and auto-detects
        # the exact GPU target via rocminfo, which is more reliable for consumer
        # GPUs (e.g. gfx1151) that may not be in the prebuilt.
        if host.has_rocm and not host.has_usable_nvidia:
            # Upstream combined ROCm tarball.
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
                    item for item in rocm_candidates if item[0][:2] <= _host_rocm_version
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
            attempts = _drop_blackwell_incapable_windows_cuda(
                host, resolve_windows_cuda_choices(host, llama_tag, upstream_assets)
            )
            if attempts:
                return attempts[0]
            # A Blackwell host left with only an sm_120-incapable cuda-12.4 build
            # (upstream gated off 13.3) has no usable GPU prebuilt here; fall
            # through to the CPU bundle rather than returning a build it cannot
            # offload. A non-Blackwell NVIDIA host with no CUDA asset at all is
            # still a hard fallback.
            if not _host_is_blackwell(host):
                raise PrebuiltFallback("no compatible Windows CUDA asset was found")

        # AMD ROCm on Windows: try upstream HIP prebuilt, then fall back to CPU
        if host.has_rocm:
            hip_name = f"llama-{llama_tag}-bin-win-hip-radeon-x64.zip"
            if hip_name in upstream_assets:
                log(f"AMD ROCm detected on Windows -- trying upstream HIP prebuilt {hip_name}")
                return AssetChoice(
                    repo = UPSTREAM_REPO,
                    tag = llama_tag,
                    name = hip_name,
                    url = upstream_assets[hip_name],
                    source_label = "upstream",
                    install_kind = "windows-hip",
                )
            log("AMD ROCm detected on Windows but no HIP prebuilt found -- falling back to CPU")

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

    raise PrebuiltFallback(f"no prebuilt policy exists for {host.system} {host.machine}")


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
            pin_attempts = _drop_blackwell_incapable_windows_cuda(host, published_attempts)
            try:
                return apply_approved_hashes(pin_attempts, checksums)
            except PrebuiltFallback as exc:
                log(
                    "published Windows CUDA assets ignored for install planning: "
                    f"{release.repo}@{release.release_tag} ({exc})"
                )
        upstream_assets = github_release_assets(UPSTREAM_REPO, llama_tag)
        upstream_attempts = _drop_blackwell_incapable_windows_cuda(
            host,
            resolve_windows_cuda_choices(host, llama_tag, upstream_assets),
        )
        return apply_approved_hashes(upstream_attempts, checksums)

    published_choice: AssetChoice | None = None
    if host.is_windows and host.is_x86_64:
        # AMD Windows hosts prefer the fork's per-gfx windows-rocm bundle when one
        # covers the GPU; otherwise fall through to resolve_asset_choice(). Note
        # that on the fork repo the upstream win-hip archive has no approved hash,
        # so apply_approved_hashes drops it and an uncovered gfx lands on a HIP
        # source build (auto-detecting its exact gfx) rather than an upstream
        # prebuilt. We still avoid hard-pinning windows-cpu here so a CPU bundle
        # never shadows that ROCm path.
        if host.has_rocm:
            published_choice = published_rocm_choice_for_host(release, host, "windows-rocm")
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

    return apply_approved_hashes(
        [resolve_asset_choice(host, llama_tag)],
        checksums,
    )


def extract_archive(archive_path: Path, destination: Path) -> None:
    def safe_extract_path(base: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        member_path = Path(normalized)
        if member_path.is_absolute():
            raise PrebuiltFallback(f"archive member used an absolute path: {member_name}")

        target = (base / member_path).resolve()
        base_resolved = base.resolve()
        try:
            target.relative_to(base_resolved)
        except ValueError as exc:
            raise PrebuiltFallback(f"archive member escaped destination: {member_name}") from exc
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
        base: Path, member_name: str, link_name: str, target: Path, archive_names: set[str]
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
                    raise PrebuiltFallback(f"tar archive entry could not be read: {member.name}")
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
                raise PrebuiltFallback(f"tar archive contained unresolved link entries: {details}")
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
    source_dir: Path,
    destination: Path,
    patterns: list[str],
    *,
    required: bool = True,
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


def ensure_diffusion_visual_server(
    install_dir: Path,
    host: HostInfo,
    release_tag: str | None,
    approved_checksums: ApprovedReleaseChecksums,
) -> None:
    """Best-effort placement of the DiffusionGemma visual-server binary next to
    llama-server in the install tree, so Studio can serve DiffusionGemma GGUFs
    without any DG_* env. This is an Unsloth artifact (not a ggml-org one), so it
    is optional: if it is already present we just make it executable, otherwise we
    try the published release and quietly skip on absence. A source build
    (setup.sh / setup.ps1) copies it from build/bin directly. Users can always
    build it from llama.cpp PR #24423 and point DG_VISUAL_BIN at it.
    """
    name = "llama-diffusion-gemma-visual-server" + (".exe" if host.is_windows else "")
    bin_dir = install_dir / "build" / ("bin/Release" if host.is_windows else "bin")
    target = bin_dir / name

    if target.exists():
        if not host.is_windows:
            try:
                target.chmod(0o755)
            except OSError:
                pass
        return

    if not release_tag:
        log(
            "diffusion visual server not bundled (no release tag); build it from llama.cpp "
            "PR #24423 and set DG_VISUAL_BIN if you want native DiffusionGemma serving"
        )
        return

    try:
        assets = github_release_assets(DEFAULT_PUBLISHED_REPO, release_tag)
        match = None
        unapproved_matches: list[str] = []
        for asset_name, url in assets.items():
            low = asset_name.lower()
            if "llama-diffusion-gemma-visual-server" not in low:
                continue
            if host.is_windows and not low.endswith(".exe"):
                continue
            if (not host.is_windows) and low.endswith(".exe"):
                continue
            # This binary is chmod'd executable and later launched by the
            # backend, so it must be covered by the approved checksum manifest
            # just like every other prebuilt artifact. An asset that matches the
            # name but is missing from the manifest is refused rather than run.
            approved = approved_checksums.artifacts.get(asset_name)
            if approved is None:
                unapproved_matches.append(asset_name)
                continue
            match = (asset_name, url, approved.sha256)
            break
        if match is None:
            if unapproved_matches:
                log(
                    "diffusion visual server asset(s) were present but omitted from the "
                    "approved checksum manifest; refusing unverified native executable: "
                    + ", ".join(unapproved_matches)
                )
            else:
                log(
                    "diffusion visual server not found in the published release; native "
                    "DiffusionGemma serving needs DG_VISUAL_BIN or a source build"
                )
            return
        bin_dir.mkdir(parents = True, exist_ok = True)
        download_file_verified(
            match[1],
            target,
            expected_sha256 = match[2],
            label = f"diffusion visual server {match[0]}",
        )
        if not host.is_windows:
            target.chmod(0o755)
        log(f"installed verified diffusion visual server: {match[0]}")
    except Exception as exc:
        log(
            "diffusion visual server fetch skipped "
            f"({textwrap.shorten(str(exc), width = 160, placeholder = '...')}); "
            "set DG_VISUAL_BIN or build from llama.cpp PR #24423 for native serving"
        )


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
    asset_url: str | None = None,
) -> None:
    archive_path = work_dir / f"llama.cpp-source-{source_ref}.tar.gz"
    repo_urls = (
        commit_source_archive_urls(source_repo, source_ref)
        if exact_source
        else upstream_source_archive_urls(source_ref)
    )
    # Prefer the published release asset (the only copy of a mix build's merged
    # tree); fall back to codeload/archive for vanilla builds whose commit is real.
    source_urls = ([asset_url] if asset_url else []) + repo_urls
    label = source_label or f"llama.cpp source tree for {source_ref}"
    extract_dir = Path(tempfile.mkdtemp(prefix = "source-extract-", dir = work_dir))

    try:
        log(f"downloading {label}")
        last_exc: Exception | None = None
        downloaded = False
        for index, source_url in enumerate(source_urls):
            try:
                if index > 0:
                    log(f"retrying source tree download from fallback URL: {source_url}")
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
            str(path.relative_to(source_root)) for path in required_paths if not path.exists()
        ]
        if missing:
            raise PrebuiltFallback(
                "upstream source archive was missing required repo files: " + ", ".join(missing)
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
    candidate = next((path for path in install_dir.rglob(executable_name) if path.is_file()), None)
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


def overlay_directory_for_choice(install_dir: Path, choice: AssetChoice, host: HostInfo) -> Path:
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
    # Broad shared-library glob + explicit binary names. Lets upstream
    # repackage the SO/DLL set (e.g. ggml-org/llama.cpp#23462 split the
    # per-binary entry code into paired ``lib<binary>-impl.so`` shared
    # libraries between b9279 and b9283) without us re-enumerating
    # every new file. Studio invokes llama-server, llama-quantize, and the
    # DiffusionGemma visual-server (when the bundle ships it, for native
    # DiffusionGemma serving); other CLIs upstream ships (llama-cli,
    # llama-bench, ...) are skipped.
    if choice.install_kind in {
        "linux-cpu",
        "linux-cuda",
        "linux-arm64-cuda",
        "linux-rocm",
        "linux-arm64",
    }:
        return ["llama-server", "llama-quantize", "llama-diffusion-gemma-visual-server", "lib*.so*"]
    if choice.install_kind in {"macos-arm64", "macos-x64"}:
        return [
            "llama-server",
            "llama-quantize",
            "llama-diffusion-gemma-visual-server",
            "lib*.dylib",
        ]
    if choice.install_kind in {
        "windows-cpu",
        "windows-cuda",
        "windows-hip",
        "windows-rocm",
        "windows-arm64",
    }:
        return [
            "llama-server.exe",
            "llama-quantize.exe",
            "llama-diffusion-gemma-visual-server.exe",
            "*.dll",
        ]
    raise PrebuiltFallback(f"unsupported install kind for runtime overlay: {choice.install_kind}")


def runtime_subdirs_for_choice(choice: AssetChoice) -> list[str]:
    """Subdirectory names within the archive root that must be copied into
    the overlay directory alongside the flat shared libraries.

    hipBLASLt and rocBLAS expect their Tensile kernel catalog trees
    (hipblaslt/library/<gfx>/ and rocblas/library/<gfx>/) to sit next to
    their shared libraries at runtime.  These trees are multi-level and
    cannot be handled by copy_globs (filename-only matching, flat copy)."""
    if choice.install_kind in {"linux-rocm", "windows-rocm", "windows-hip"}:
        return ["hipblaslt", "rocblas"]
    return []


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
        raise RuntimeError("activated install was missing expected files: " + ", ".join(missing))


def activate_staged_dir(staging_dir: Path, dst: Path) -> None:
    """Move a freshly extracted ``staging_dir`` onto ``dst``.

    ``os.replace`` is attempted first as the fast path. On Windows ARM64 the
    antivirus scanner can transiently hold a freshly extracted DLL open at the
    moment ``MoveFileEx`` runs, surfacing as ``[WinError 5] Access is denied``;
    a file-by-file copy bypasses the rename entirely.

    This fallback is intentionally limited to staging trees we just extracted.
    It must not be used to move an existing/active install aside: there an
    ``os.replace`` failure means the directory is genuinely in use, and a
    silent copy + ``rmtree`` could partially delete a live install.

    Only busy/lock errors (``is_busy_lock_error``) trigger the copy; anything
    else (disk full, cross-device, missing path) re-raises so it cannot leave
    a partially copied install behind. A copy is preferred over retrying the
    rename because antivirus scans of large DLLs can outlast any reasonable
    retry window.
    """
    try:
        os.replace(staging_dir, dst)
    except OSError as exc:
        if not is_busy_lock_error(exc):
            raise
        log(f"os.replace failed ({exc!r}); falling back to file-by-file copy of staging tree")
        shutil.copytree(staging_dir, dst, dirs_exist_ok = True)
        remove_tree(staging_dir)


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
        activate_staged_dir(staging_dir, install_dir)
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
            runtime_extract_dir = Path(tempfile.mkdtemp(prefix = "extract-runtime-", dir = work_dir))
            extract_archive(runtime_archive, runtime_extract_dir)
        source_dir = extract_dir
        overlay_dir = overlay_directory_for_choice(install_dir, choice, host)
        copy_globs(source_dir, overlay_dir, runtime_patterns_for_choice(choice), required = True)
        for _subdir in runtime_subdirs_for_choice(choice):
            _src_subdir = source_dir / _subdir
            if _src_subdir.is_dir():
                shutil.copytree(_src_subdir, overlay_dir / _subdir, dirs_exist_ok = True)
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
        raise PrebuiltFallback("unix executables were not installed correctly into build/bin")
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
    missing = [str(path.relative_to(install_dir)) for path in required if not path.exists()]
    if missing:
        raise PrebuiltFallback("hydrated llama.cpp source tree was missing: " + ", ".join(missing))


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
            f"validation model checksum mismatch: expected={TEST_MODEL_SHA256} actual={digest}"
        )
    return data


def _hf_resolve_url_parts(url: str) -> tuple[str, str, str] | None:
    """Parse a huggingface.co .../resolve/<rev>/<path> URL into
    (repo_id, revision, filename); None if it is not such a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    if (parsed.netloc or "").lower() not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = parsed.path.strip("/").split("/")
    # <owner>/<name>/resolve/<rev>/<path...>
    if len(parts) >= 5 and parts[2] == "resolve":
        return f"{parts[0]}/{parts[1]}", parts[3], "/".join(parts[4:])
    return None


def _fetch_validation_model_bytes() -> bytes:
    """Fetch the tiny GGUF validation model. Prefer huggingface_hub (completes
    TLS chains via AIA fetching that bare urllib can't on some Windows/proxy
    setups); fall back to the direct URL when hf_hub is unavailable or fails."""
    parts = _hf_resolve_url_parts(TEST_MODEL_URL)
    if parts is not None:
        repo_id, revision, filename = parts
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id = repo_id, filename = filename, revision = revision)
            return validated_validation_model_bytes(Path(local).read_bytes())
        except Exception as exc:
            log(
                f"huggingface_hub fetch of validation model failed ({exc}); "
                "falling back to direct URL"
            )
    return validated_validation_model_bytes(
        download_bytes(
            TEST_MODEL_URL,
            progress_label = f"Downloading {download_label_from_url(TEST_MODEL_URL)}",
        )
    )


def download_validation_model(path: Path, cache_path: Path | None = None) -> None:
    try:
        data: bytes | None = None
        if cache_path and cache_path.exists():
            try:
                data = validated_validation_model_bytes(cache_path.read_bytes())
                log(f"using cached tiny GGUF validation model from {cache_path}")
            except Exception as exc:
                log(f"cached tiny GGUF validation model was invalid; refreshing cache ({exc})")
                data = None
        if data is None:
            log("downloading tiny GGUF validation model")
            data = _fetch_validation_model_bytes()
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


_VALIDATION_LAUNCH_RUN = "run"
_VALIDATION_LAUNCH_SKIP = "skip"
_VALIDATION_LAUNCH_FALLBACK = "fallback"
_VALIDATION_PURPOSE_LDD = "ldd"
_VALIDATION_PURPOSE_QUANTIZE = "quantize"
_VALIDATION_PURPOSE_SERVER = "server"
_VALIDATION_NETWORK_POLICY_DIRECT = "direct"
_VALIDATION_NETWORK_POLICY_SANDBOX = "sandbox_loopback_only"
_LINUX_LDD_PROBE_OK = "ok"
_LINUX_LDD_PROBE_SKIPPED = "skipped"
_LINUX_LDD_PROBE_ERROR = "error"
_VALIDATION_SERVER_PROBE_MODE_HOST = "host_loopback_http"
_VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX = "sandbox_loopback_http"
_LINUX_SERVER_VALIDATION_HELPER_TIMEOUT_SECONDS = 60
_LINUX_SERVER_VALIDATION_HELPER_SHUTDOWN_SECONDS = 5
_LINUX_SERVER_VALIDATION_HELPER_CAPTURE_TIMEOUT_SECONDS = (
    _LINUX_SERVER_VALIDATION_HELPER_TIMEOUT_SECONDS
    + (2 * _LINUX_SERVER_VALIDATION_HELPER_SHUTDOWN_SECONDS)
)


@dataclass(frozen = True)
class _ValidationLaunchPlan:
    command: list[str]
    env: dict[str, str]
    action: str
    purpose: str
    sandbox_kind: str | None = None
    reason: str | None = None
    payload_command: list[str] | None = None
    payload_env: dict[str, str] | None = None
    network_policy: str | None = None
    server_probe_mode: str | None = None

    @property
    def is_runnable(self) -> bool:
        return self.action == _VALIDATION_LAUNCH_RUN

    @property
    def is_skipped(self) -> bool:
        return self.action == _VALIDATION_LAUNCH_SKIP

    @property
    def is_fallback(self) -> bool:
        return self.action == _VALIDATION_LAUNCH_FALLBACK


@dataclass(frozen = True)
class LinuxLibraryProbeResult:
    status: Literal["ok", "skipped", "error"]
    missing: list[str]
    output: str = ""
    reason: str | None = None


def _resolve_command_path(command: str) -> str | None:
    resolved = shutil.which(command)
    if resolved is None:
        return None
    return str(Path(resolved).resolve())


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        return candidate.resolve(strict = True)
    except Exception:
        return candidate


def _resolve_sandbox_command(command: list[str]) -> list[str]:
    if not command:
        return []
    resolved_command = list(command)
    command_path = Path(resolved_command[0])
    if command_path.is_absolute():
        resolved_command[0] = str(_resolve_existing_path(command_path))
        return resolved_command
    resolved_path = _resolve_command_path(resolved_command[0])
    if resolved_path is not None:
        resolved_command[0] = resolved_path
    return resolved_command


def _binary_is_setuid_root(path: str | Path) -> bool:
    try:
        stat_result = os.stat(path)
    except OSError:
        return False
    return stat_result.st_uid == 0 and bool(stat_result.st_mode & stat.S_ISUID)


_bwrap_sandbox_capability: dict[str, bool] = {}
_LINUX_DYNAMIC_LOADER_ENV_VARS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_ORIGIN_PATH",
)


def _bwrap_can_sandbox(bwrap_path: str) -> bool:
    # A present bwrap is not always usable: where unprivileged user namespaces are
    # restricted (Ubuntu >= 23.10 AppArmor default, nested/unprivileged containers,
    # hardened cloud VMs) a non-setuid bwrap fails to set up its uid map or loopback.
    # Probe once and cache so a broken sandbox degrades like an absent one instead of
    # failing every prebuilt validation and forcing a source build.
    cached = _bwrap_sandbox_capability.get(bwrap_path)
    if cached is not None:
        return cached
    ok = False
    try:
        result = run_capture(
            [
                bwrap_path,
                "--ro-bind",
                "/",
                "/",
                "--unshare-all",
                "--die-with-parent",
                _resolve_command_path("true") or "/bin/true",
            ],
            env = _linux_validation_launcher_env({}),
            timeout = 20,
        )
        ok = result.returncode == 0
    except Exception:
        ok = False
    _bwrap_sandbox_capability[bwrap_path] = ok
    return ok


def _append_existing_bwrap_bind(
    args: list[str], *, source: str | Path, readonly: bool, seen: set[str]
) -> None:
    path = Path(source)
    if not path.exists():
        return
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    args.extend(
        [
            "--ro-bind" if readonly else "--bind",
            key,
            key,
        ]
    )


_SANDBOX_LIBRARY_DIR_NAMES = frozenset({"lib", "lib64"})


def _is_broad_sandbox_library_path(path: str | Path, *, require_library_dir: bool = False) -> bool:
    candidate = Path(path)
    resolved = _resolve_existing_path(candidate)
    if not candidate.is_absolute() and not str(candidate).startswith(("/", "\\")):
        return True
    if len(resolved.parts) <= 2:
        return True
    if len(resolved.parts) <= 3 and resolved.parts[1].lower() in {"home", "users"}:
        return True
    if require_library_dir:
        if resolved.name.lower() not in _SANDBOX_LIBRARY_DIR_NAMES:
            return True
    try:
        if resolved == Path.home().resolve():
            return True
    except Exception:
        pass
    return False


def _sandbox_library_path_targets(
    env: dict[str, str],
    key: str,
    *,
    require_library_dir: bool = False,
) -> list[Path]:
    return [
        _resolve_existing_path(Path(part))
        for part in env.get(key, "").split(os.pathsep)
        if part
        and not _is_broad_sandbox_library_path(part, require_library_dir = require_library_dir)
    ]


def _linux_validation_launcher_env(payload_env: dict[str, str]) -> dict[str, str]:
    env = scrubbed_environ()
    for key in (*payload_env, *_LINUX_DYNAMIC_LOADER_ENV_VARS):
        env.pop(key, None)
    # Keep a stable base command search path.
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return env


def _linux_validation_setenv_args(payload_env: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in sorted(payload_env.items()):
        args.extend(["--setenv", key, value])
    return args


def _drop_server_gpu_layers(command: list[str]) -> list[str]:
    trimmed: list[str] = []
    index = 0
    while index < len(command):
        arg = command[index]
        if arg == "--n-gpu-layers" and index + 1 < len(command):
            index += 2
            continue
        trimmed.append(arg)
        index += 1
    return trimmed


def _extract_loopback_port(command: list[str]) -> int:
    for index, arg in enumerate(command):
        if arg != "--port":
            continue
        if index + 1 >= len(command):
            break
        try:
            return int(command[index + 1])
        except ValueError:
            break
    return 0


def _linux_validation_server_probe_command(
    server_command: list[str],
    payload_env: dict[str, str],
    *,
    timeout: int = _LINUX_SERVER_VALIDATION_HELPER_TIMEOUT_SECONDS,
) -> list[str]:
    for candidate in ("/usr/bin/python3", "/usr/bin/python"):
        if Path(candidate).exists():
            python = candidate
            break
    else:
        python = shutil.which("python3") or shutil.which("python")
    if python is None:
        raise RuntimeError("No python interpreter available for in-sandbox server probe")

    probe_script = textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        import tempfile
        import time
        import urllib.error
        import urllib.request

        command = {json.dumps(server_command)}
        payload_env = {json.dumps(payload_env)}
        body = b'{{"prompt":"a","n_predict":1}}'
        timeout = {int(timeout)}
        deadline = time.time() + timeout

        def read_tail(path, max_lines = 80):
            try:
                with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
                    return handle.read().splitlines()[-max_lines:]
            except Exception:
                return []

        log_fd, log_path = tempfile.mkstemp(prefix = "unsloth-server-validate-", suffix = ".log")
        os.close(log_fd)

        port = 0
        for index, arg in enumerate(command):
            if arg == "--port" and index + 1 < len(command):
                try:
                    port = int(command[index + 1])
                except ValueError:
                    port = 0
                break
        if port <= 0:
            print("failed: missing --port for sandboxed llama-server probe")
            raise SystemExit(1)

        try:
            with open(log_path, "w", encoding = "utf-8", errors = "replace") as log_handle:
                server_env = dict(os.environ)
                server_env.update(payload_env)
                process = subprocess.Popen(
                    command,
                    stdout = log_handle,
                    stderr = subprocess.STDOUT,
                    text = True,
                    env = server_env,
                )
                request = urllib.request.Request(
                    "http://127.0.0.1:%d/completion" % port,
                    data = body,
                    headers = {{"Content-Type": "application/json"}},
                    method = "POST",
                )
                while time.time() < deadline:
                    if process.poll() is not None:
                        print("server exited before completion probe, code=%s" % process.returncode)
                        for line in read_tail(log_path):
                            print(line)
                        raise SystemExit(process.returncode if process.returncode else 1)
                    try:
                        with urllib.request.urlopen(request, timeout = 5) as response:
                            _ = response.read(32)
                            if response.status == 200:
                                process.terminate()
                                try:
                                    process.wait(timeout = 5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                print("server validation probe succeeded for port=%s" % port)
                                for line in read_tail(log_path):
                                    print(line)
                                raise SystemExit(0)
                            print("server returned HTTP %s" % response.status)
                    except urllib.error.HTTPError as exc:
                        print("server returned HTTP %s" % exc.code)
                        _ = exc.read()
                    except Exception:
                        pass
                    time.sleep(0.25)

            print("server validation timed out waiting for loopback response on port=%s" % port)
            for line in read_tail(log_path):
                print(line)
            raise SystemExit(1)
        finally:
            if "process" in locals() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout = 5)
            os.unlink(log_path)
        """
    )
    return [python, "-c", probe_script]


def _linux_validation_bwrap_prefix(
    command: list[str],
    *,
    binary_path: Path,
    install_dir: Path,
    purpose: str,
    adapter_path: str,
    payload_env: dict[str, str],
    payload_command: list[str] | None = None,
    enable_gpu_layers: bool = False,
    gpu_backend: Literal["cuda", "rocm"] | None = None,
) -> list[str]:
    runtime_home = isolated_runtime_home()
    command_path = _resolve_existing_path(command[0])

    write_targets = {
        str(Path(runtime_home)),
    }
    readonly_targets: list[str | Path] = [
        install_dir,
        binary_path.parent,
        command_path.parent,
        "/bin",
        "/lib",
        "/lib64",
        "/usr/bin",
        "/usr/lib",
        "/usr/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
    ]
    if command_path.parent.name == "bin":
        readonly_targets.extend(
            [
                command_path.parent.parent / "lib",
                command_path.parent.parent / "lib64",
            ]
        )
    if Path("/nix/store") in command_path.parents:
        readonly_targets.append("/nix/store")
    readonly_targets.extend(_sandbox_library_path_targets(payload_env, "LD_LIBRARY_PATH"))
    server_command = payload_command or command

    if purpose == _VALIDATION_PURPOSE_QUANTIZE and len(command) >= 3:
        readonly_targets.append(Path(command[1]).parent)
        write_targets.add(str(Path(command[2]).parent))
    elif (
        purpose == _VALIDATION_PURPOSE_SERVER
        and len(server_command) >= 3
        and server_command[1] == "-m"
    ):
        readonly_targets.append(Path(server_command[2]).parent)

    enable_gpu_devices = (
        purpose == _VALIDATION_PURPOSE_SERVER
        and enable_gpu_layers
        and gpu_backend in {"cuda", "rocm"}
    )
    args = [adapter_path]
    if enable_gpu_devices:
        args.extend(
            [
                "--unshare-cgroup-try",
                "--unshare-ipc",
                "--unshare-net",
                "--unshare-pid",
                "--unshare-uts",
            ]
        )
    else:
        args.append("--unshare-all")
    args.extend(
        [
            "--die-with-parent",
            "--new-session",
            "--perms",
            "1777",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--setenv",
            "HOME",
            runtime_home,
        ]
    )
    is_server_helper = payload_command is not None and purpose == _VALIDATION_PURPOSE_SERVER
    if not is_server_helper:
        args.extend(_linux_validation_setenv_args(payload_env))

    seen: set[str] = set()
    if enable_gpu_devices:
        dev_nodes: list[str] = []
        readonly_gpu_targets = [
            "/sys/class/drm",
            "/sys/bus/pci",
            "/sys/dev/char",
            "/sys/devices",
            "/etc/vulkan",
            "/usr/share/vulkan",
        ]
        if gpu_backend == "cuda":
            dev_nodes.extend(
                [
                    "/dev/nvidiactl",
                    "/dev/nvidia-modeset",
                    "/dev/nvidia-uvm",
                    "/dev/nvidia-uvm-tools",
                ]
            )
            for path in Path("/dev").glob("nvidia*"):
                if re.match(r"^nvidia[0-9]+$", path.name):
                    dev_nodes.append(str(path))
            dev_nodes.extend(str(path) for path in Path("/dev/nvidia-caps").glob("nvidia-cap*"))
            readonly_gpu_targets.append("/proc/driver/nvidia")
            readonly_gpu_targets.append("/proc/driver/nvidia/capabilities")
        elif gpu_backend == "rocm":
            dev_nodes.extend(["/dev/kfd", "/dev/dxg"])
            dev_nodes.extend(str(node) for node in Path("/dev/dri").glob("card*"))
            dev_nodes.extend(str(node) for node in Path("/dev/dri").glob("renderD*"))

        for node in dev_nodes:
            node_path = Path(node)
            if str(node_path) in seen:
                continue
            if not node_path.exists():
                continue
            seen.add(str(node_path))
            args.extend(["--dev-bind-try", str(node_path), str(node_path)])

        readonly_targets.extend(readonly_gpu_targets)
    for target in readonly_targets:
        target_path = Path(target)
        if str(target_path) in write_targets:
            continue
        _append_existing_bwrap_bind(args, source = target_path, readonly = True, seen = seen)
    for target in sorted(write_targets):
        _append_existing_bwrap_bind(args, source = target, readonly = False, seen = seen)
    return args


def _sandbox_profile_path_literals(path: str | Path) -> list[str]:
    raw_path = Path(path)
    candidates: list[Path] = [raw_path]
    try:
        resolved = raw_path.resolve()
    except Exception:
        resolved = raw_path
    if resolved not in candidates:
        candidates.append(resolved)
    literals: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        literals.append(text.replace("\\", "\\\\").replace('"', '\\"'))
    return literals


def _macos_validation_sandbox_prefix(
    command: list[str],
    *,
    binary_path: Path,
    install_dir: Path,
    purpose: str,
    env: dict[str, str],
    adapter_path: str,
) -> list[str]:
    runtime_home = Path(isolated_runtime_home())
    read_targets: list[str | Path] = [
        "/bin",
        "/usr/bin",
        "/usr/lib",
        "/System",
        "/System/Library/Frameworks",
        "/System/Library",
        install_dir,
        binary_path.parent,
        runtime_home,
    ]
    read_targets.extend(
        _sandbox_library_path_targets(env, "DYLD_LIBRARY_PATH", require_library_dir = True)
    )
    write_targets: list[str | Path] = [runtime_home]
    if purpose == _VALIDATION_PURPOSE_QUANTIZE and len(command) >= 3:
        read_targets.append(Path(command[1]).parent)
        write_targets.append(Path(command[2]).parent)
    elif purpose == _VALIDATION_PURPOSE_SERVER and len(command) >= 3 and command[1] == "-m":
        read_targets.append(Path(command[2]).parent)

    exec_targets: list[str | Path] = [
        "/bin",
        "/usr/bin",
        "/usr/lib",
        "/System",
        "/System/Library",
        binary_path.parent,
    ]
    executable_map_targets: list[str | Path] = [
        "/usr/lib",
        "/System",
        "/System/Library",
        install_dir,
        binary_path.parent,
    ]
    executable_map_targets.extend(
        _sandbox_library_path_targets(env, "DYLD_LIBRARY_PATH", require_library_dir = True)
    )
    profile_parts = [
        "(version 1)",
        "(deny default)",
        '(import "bsd.sb")',
        "(allow process-exec",
    ]
    for target in exec_targets:
        for literal in _sandbox_profile_path_literals(target):
            profile_parts.append(f'(subpath "{literal}")')
    profile_parts.append(")")
    profile_parts.append("(allow file-map-executable")
    for target in executable_map_targets:
        for literal in _sandbox_profile_path_literals(target):
            profile_parts.append(f'(subpath "{literal}")')
    profile_parts.append(")")
    profile_parts.append("(allow file-read*")
    for target in read_targets:
        for literal in _sandbox_profile_path_literals(target):
            if literal == "/":
                profile_parts.append(f'(literal "{literal}")')
            else:
                profile_parts.append(f'(subpath "{literal}")')
    profile_parts.append(")")
    profile_parts.append("(allow file-write*")
    for target in write_targets:
        for literal in _sandbox_profile_path_literals(target):
            profile_parts.append(f'(subpath "{literal}")')
    profile_parts.append(")")
    if purpose == _VALIDATION_PURPOSE_SERVER:
        server_port = _extract_loopback_port(command)
        if server_port > 0:
            profile_parts.append(f'(allow network* (local ip "localhost:{server_port}"))')
            profile_parts.append(f'(allow network* (remote ip "localhost:{server_port}"))')
    profile = "".join(profile_parts)
    return [
        adapter_path,
        "-p",
        profile,
    ]


def _host_is_linux(host: HostInfo | None = None) -> bool:
    if host is not None:
        return host.is_linux
    return platform.system() == "Linux"


def _host_is_macos(host: HostInfo | None = None) -> bool:
    if host is not None:
        return host.is_macos
    return platform.system() == "Darwin"


def _host_is_windows(host: HostInfo | None = None) -> bool:
    if host is not None:
        return host.is_windows
    return platform.system() == "Windows"


def build_validation_sandbox_plan(
    command: list[str],
    *,
    binary_path: Path,
    install_dir: Path,
    purpose: str,
    env: dict[str, str],
    enable_gpu_layers: bool = False,
    gpu_backend: Literal["cuda", "rocm"] | None = None,
    host: HostInfo | None = None,
    runtime_line: str | None = None,
) -> _ValidationLaunchPlan:
    launcher_env = (
        _linux_validation_launcher_env(env) if _host_is_linux(host) else scrubbed_environ()
    )
    if _host_is_linux(host):
        bwrap_path = _resolve_command_path("bwrap")
        if bwrap_path is not None and _bwrap_can_sandbox(bwrap_path):
            payload_command = _resolve_sandbox_command(command)
            if (
                purpose == _VALIDATION_PURPOSE_SERVER
                and enable_gpu_layers
                and gpu_backend in {"cuda", "rocm"}
                and not _binary_is_setuid_root(bwrap_path)
            ):
                # Non-setuid bwrap needs a user namespace, which drops the host's
                # supplementary GPU device groups. Keep the loopback-only sandbox
                # and validate the bundle on the CPU path instead.
                payload_command = _drop_server_gpu_layers(payload_command)
                enable_gpu_layers = False
                gpu_backend = None
            network_policy = _VALIDATION_NETWORK_POLICY_SANDBOX
            server_probe_mode = (
                _VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX
                if purpose == _VALIDATION_PURPOSE_SERVER
                else None
            )
            launch_command = payload_command
            if purpose == _VALIDATION_PURPOSE_SERVER:
                try:
                    launch_command = _resolve_sandbox_command(
                        _linux_validation_server_probe_command(
                            payload_command,
                            env,
                            timeout = _LINUX_SERVER_VALIDATION_HELPER_TIMEOUT_SECONDS,
                        )
                    )
                except RuntimeError as exc:
                    return _ValidationLaunchPlan(
                        command = payload_command,
                        env = launcher_env,
                        action = _VALIDATION_LAUNCH_FALLBACK,
                        purpose = purpose,
                        reason = str(exc),
                        payload_command = payload_command,
                        payload_env = env,
                        network_policy = network_policy,
                        server_probe_mode = _VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX,
                        sandbox_kind = "linux_bwrap",
                    )
            adapter_command = _linux_validation_bwrap_prefix(
                launch_command,
                binary_path = binary_path,
                install_dir = install_dir,
                purpose = purpose,
                adapter_path = bwrap_path,
                payload_env = env,
                payload_command = payload_command,
                enable_gpu_layers = enable_gpu_layers,
                gpu_backend = gpu_backend,
            )
            return _ValidationLaunchPlan(
                command = [
                    *adapter_command,
                    *launch_command,
                ],
                env = launcher_env,
                action = _VALIDATION_LAUNCH_RUN,
                purpose = purpose,
                sandbox_kind = "linux_bwrap",
                payload_command = payload_command,
                payload_env = env,
                network_policy = network_policy,
                server_probe_mode = server_probe_mode,
            )
        if purpose == _VALIDATION_PURPOSE_LDD:
            return _ValidationLaunchPlan(
                command = command,
                env = launcher_env,
                action = _VALIDATION_LAUNCH_SKIP,
                purpose = purpose,
                sandbox_kind = "linux_bwrap",
                reason = "No Linux sandbox adapter was available; skip ldd probe",
            )
        return _ValidationLaunchPlan(
            command = command,
            env = launcher_env,
            action = _VALIDATION_LAUNCH_SKIP,
            purpose = purpose,
            sandbox_kind = "linux_bwrap",
            reason = "No Linux sandbox adapter was available; skip downloaded-binary validation",
            payload_command = command,
            payload_env = env,
            network_policy = None,
            server_probe_mode = None,
        )

    if _host_is_macos(host):
        sandbox_exec_path = _resolve_command_path("sandbox-exec")
        if sandbox_exec_path is not None:
            network_policy = (
                _VALIDATION_NETWORK_POLICY_SANDBOX
                if purpose == _VALIDATION_PURPOSE_SERVER
                else _VALIDATION_NETWORK_POLICY_DIRECT
            )
            launch_command = command
            return _ValidationLaunchPlan(
                command = [
                    *_macos_validation_sandbox_prefix(
                        command,
                        binary_path = binary_path,
                        install_dir = install_dir,
                        purpose = purpose,
                        env = env,
                        adapter_path = sandbox_exec_path,
                    ),
                    "/usr/bin/env",
                    "-i",
                    *[f"{name}={value}" for name, value in sorted(env.items())],
                    *launch_command,
                ],
                env = launcher_env,
                action = _VALIDATION_LAUNCH_RUN,
                purpose = purpose,
                sandbox_kind = "macos_sandbox_exec",
                payload_command = launch_command,
                payload_env = env,
                network_policy = network_policy,
                server_probe_mode = _VALIDATION_SERVER_PROBE_MODE_HOST,
            )
        return _ValidationLaunchPlan(
            command = command,
            env = launcher_env,
            action = _VALIDATION_LAUNCH_FALLBACK,
            purpose = purpose,
            reason = "No macOS sandbox-exec adapter was available for downloaded-binary validation",
            payload_command = command,
            payload_env = env,
            sandbox_kind = "macos_sandbox_exec",
            network_policy = _VALIDATION_NETWORK_POLICY_DIRECT
            if purpose == _VALIDATION_PURPOSE_SERVER
            else _VALIDATION_NETWORK_POLICY_DIRECT,
            server_probe_mode = _VALIDATION_SERVER_PROBE_MODE_HOST,
        )

    if _host_is_windows(host):
        return _ValidationLaunchPlan(
            command = command,
            env = env,
            action = _VALIDATION_LAUNCH_RUN,
            purpose = purpose,
            sandbox_kind = "windows_direct_validation",
            reason = "Running validation directly on Windows host",
            payload_command = command,
            payload_env = env,
            network_policy = _VALIDATION_NETWORK_POLICY_DIRECT,
            server_probe_mode = _VALIDATION_SERVER_PROBE_MODE_HOST,
        )

    return _ValidationLaunchPlan(
        command = command,
        env = env,
        action = _VALIDATION_LAUNCH_FALLBACK,
        purpose = purpose,
        reason = f"Unsupported platform for validation sandboxing: {platform.system()}",
    )


def _unavailable_validation_launch(plan: _ValidationLaunchPlan) -> ValidationLaunchUnavailable:
    reason = plan.reason or f"{plan.purpose} launch was skipped by sandbox policy"
    return ValidationLaunchUnavailable(reason)


def _run_validation_launch(
    plan: _ValidationLaunchPlan,
    *,
    timeout: int | None = None,
    stdout = None,
    popen: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
    if plan.is_skipped:
        raise _unavailable_validation_launch(plan)
    if plan.is_fallback:
        if plan.reason is None:
            raise PrebuiltFallback("validation launch skipped due to missing sandbox")
        raise PrebuiltFallback(plan.reason)
    if popen:
        try:
            return subprocess.Popen(
                plan.command,
                stdout = stdout,
                stderr = subprocess.STDOUT,
                text = True,
                env = plan.env,
                **windows_hidden_subprocess_kwargs(),
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise _subprocess_failure(plan.command, exc) from exc
    if timeout is None:
        raise ValueError("timeout is required for captured validation launches")
    return run_capture(plan.command, timeout = timeout, env = plan.env)


def _run_validation_capture(
    plan: _ValidationLaunchPlan, *, timeout: int
) -> subprocess.CompletedProcess[str]:
    result = _run_validation_launch(plan, timeout = timeout)
    assert isinstance(result, subprocess.CompletedProcess)
    return result


def _parse_ldd_missing_libraries(output: str) -> list[str]:
    missing: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if "=> not found" not in line:
            continue
        library = line.split("=>", 1)[0].strip()
        if library and library not in missing:
            missing.append(library)
    return missing


def _ldd_output_is_static_binary(output: str) -> bool:
    return "not a dynamic executable" in output.lower()


def _run_validation_ldd_probe(binary_path: Path, *, env: dict[str, str]) -> LinuxLibraryProbeResult:
    ldd_path = shutil.which("ldd")
    if ldd_path is None:
        return LinuxLibraryProbeResult(
            status = _LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = "ldd executable was not found",
        )
    plan = build_validation_sandbox_plan(
        [ldd_path, str(binary_path)],
        binary_path = binary_path,
        install_dir = binary_path.parent,
        purpose = _VALIDATION_PURPOSE_LDD,
        env = env,
    )
    if plan.is_skipped:
        return LinuxLibraryProbeResult(
            status = _LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = "ldd probe was skipped by validation policy",
        )
    if plan.is_fallback:
        return LinuxLibraryProbeResult(
            status = _LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = plan.reason or "ldd probe did not run",
        )
    try:
        result = _run_validation_capture(plan, timeout = 20)
    except Exception as exc:
        return LinuxLibraryProbeResult(
            status = _LINUX_LDD_PROBE_ERROR,
            missing = [],
            reason = f"ldd probe failed: {exc}",
        )
    if result.returncode != 0:
        reason = result.stdout if result.stdout else ""
        stderr = result.stderr if result.stderr else ""
        if stderr:
            reason = f"{reason} {stderr}" if reason else stderr
        if _ldd_output_is_static_binary(reason):
            return LinuxLibraryProbeResult(
                status = _LINUX_LDD_PROBE_OK,
                missing = [],
                reason = "static executable",
                output = result.stdout + result.stderr,
            )
        if not reason:
            reason = "ldd probe failed"
        return LinuxLibraryProbeResult(
            status = _LINUX_LDD_PROBE_ERROR,
            missing = [],
            reason = reason.strip(),
            output = result.stdout + result.stderr,
        )
    output = result.stdout + result.stderr
    return LinuxLibraryProbeResult(
        status = _LINUX_LDD_PROBE_OK,
        missing = _parse_ldd_missing_libraries(output),
        output = output,
    )


def _run_validation_popen(
    plan: _ValidationLaunchPlan,
    *,
    stdout,
    timeout: int | None = None,  # kept for parity with current validate_server call shape
) -> subprocess.Popen[str]:
    process = _run_validation_launch(plan, stdout = stdout, popen = True)
    return process  # type: ignore[return-value]


def linux_missing_libraries(binary_path: Path, *, env: dict[str, str] | None = None) -> list[str]:
    if env is None:
        env = scrubbed_environ()
    try:
        probe_output = _run_validation_ldd_probe(binary_path, env = env)
    except Exception:
        return []
    if probe_output.status != _LINUX_LDD_PROBE_OK:
        return []
    if not probe_output.output:
        return []
    return probe_output.missing


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
    # ldd may execute the binary, so probe it with a secret-free env.
    probe_result = _run_validation_ldd_probe(binary_path, env = scrubbed_environ())
    if probe_result.status != _LINUX_LDD_PROBE_OK:
        if probe_result.reason:
            log(f"Skipping Linux runtime dirs probe because {probe_result.reason}")
        return []
    missing = probe_result.missing
    if not missing:
        return []
    return linux_runtime_dirs_for_required_libraries(missing)


# macOS prebuilt compatibility. Upstream macos prebuilts built on a newer macOS
# (e.g. minos=26, referencing Metal-4 symbols) fail dyld load on macOS 14/15. We
# read the host macOS version and each binary's minimum-OS so selection can skip
# a too-new prebuilt and walk back to the newest release that runs on this host.
# Mach-O constants (Apple mach-o/fat.h, mach-o/loader.h, mach/machine.h).
_MACHO_FAT_MAGICS = {0xCAFEBABE, 0xCAFEBABF}  # universal binary (32/64-bit fat)
_LC_VERSION_MIN_MACOSX = 0x24  # legacy min-macOS load command
_LC_BUILD_VERSION = 0x32  # modern platform+minos+sdk load command
_MACHO_PLATFORM_MACOS = 1  # LC_BUILD_VERSION platform id for macOS (iOS=2, ...)
# CPU types (base | ABI64); used to pick the host slice in a fat binary.
_CPU_TYPE_X86_64 = 0x01000007
_CPU_TYPE_ARM64 = 0x0100000C


def parse_macos_version(value: str | None) -> tuple[int, int] | None:
    """Parse a macOS product version string into (major, minor).

    Handles "14.7.1", "15.5", "26.0" and bare "26". Returns None when the
    value is empty or cannot be parsed (callers then defer to runtime
    validation rather than rejecting every prebuilt)."""
    if not value:
        return None
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def host_supports_macos_minos(host: HostInfo, minos: tuple[int, int] | None) -> bool:
    """True if a prebuilt requiring `minos` can load on this host. Unknown host
    version or unknown minos -> True: let runtime validation decide instead of
    rejecting a binary we cannot reason about."""
    if minos is None or host.macos_version is None:
        return True
    return host.macos_version >= minos


def _macho_slice_minos(data: bytes, offset: int) -> tuple[int, int] | None:
    """Minimum macOS for a single thin Mach-O at `offset`, via LC_BUILD_VERSION
    (platform macOS) or the legacy LC_VERSION_MIN_MACOSX. None if absent."""
    if offset + 4 > len(data):
        return None
    magic = struct.unpack_from(">I", data, offset)[0]
    if magic in (0xFEEDFACE, 0xFEEDFACF):
        endian, is64 = ">", magic == 0xFEEDFACF
    elif magic in (0xCEFAEDFE, 0xCFFAEDFE):
        endian, is64 = "<", magic == 0xCFFAEDFE
    else:
        return None
    header_size = 32 if is64 else 28
    if offset + header_size > len(data):
        return None
    ncmds = struct.unpack_from(endian + "I", data, offset + 16)[0]
    cursor = offset + header_size
    for _ in range(ncmds):
        if cursor + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(endian + "II", data, cursor)
        if cmdsize < 8:
            break
        if cmd == _LC_BUILD_VERSION and cursor + 16 <= len(data):
            platform_id, minos = struct.unpack_from(endian + "II", data, cursor + 8)
            if platform_id == _MACHO_PLATFORM_MACOS:
                return (minos >> 16) & 0xFFFF, (minos >> 8) & 0xFF
        elif cmd == _LC_VERSION_MIN_MACOSX and cursor + 12 <= len(data):
            version = struct.unpack_from(endian + "I", data, cursor + 8)[0]
            return (version >> 16) & 0xFFFF, (version >> 8) & 0xFF
        cursor += cmdsize
    return None


def macho_minimum_macos(path: Path, host: HostInfo | None = None) -> tuple[int, int] | None:
    """Minimum macOS (major, minor) a Mach-O binary or dylib requires.

    Pure-Python so it works on consumer Macs without the Xcode command line
    tools (otool/vtool). For universal binaries it prefers the host-arch slice,
    else the highest minos found. Returns None for non-Mach-O files or when no
    version load command is present."""
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if len(data) < 8:
        return None
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic in _MACHO_FAT_MAGICS:
        is64 = magic == 0xCAFEBABF
        nfat = struct.unpack_from(">I", data, 4)[0]
        entry = 8
        slices: list[tuple[int, tuple[int, int]]] = []
        for _ in range(nfat):
            if is64:
                if entry + 32 > len(data):
                    break
                cputype = struct.unpack_from(">I", data, entry)[0]
                slice_offset = struct.unpack_from(">Q", data, entry + 8)[0]
                entry += 32
            else:
                if entry + 20 > len(data):
                    break
                cputype = struct.unpack_from(">I", data, entry)[0]
                slice_offset = struct.unpack_from(">I", data, entry + 8)[0]
                entry += 20
            minos = _macho_slice_minos(data, slice_offset)
            if minos is not None:
                slices.append((cputype, minos))
        if not slices:
            return None
        if host is not None:
            want = (
                _CPU_TYPE_ARM64 if host.is_arm64 else (_CPU_TYPE_X86_64 if host.is_x86_64 else None)
            )
            for cputype, minos in slices:
                if cputype == want:
                    return minos
        return max(minos for _cputype, minos in slices)
    return _macho_slice_minos(data, 0)


def looks_like_macos_incompatibility(text: str) -> bool:
    """True when dyld output means a prebuilt needs a newer macOS than the host
    (the runtime backstop for cases the static minos scan cannot read)."""
    if not text:
        return False
    if "built for macOS" in text and "newer than running OS" in text:
        return True
    return "Symbol not found" in text and "MTLResidency" in text


def macos_binary_minos_issues(
    binaries: Iterable[Path], install_dir: Path, host: HostInfo
) -> list[str]:
    """Issue strings for every installed Mach-O whose minimum macOS exceeds the
    host. Scans the given executables plus every bundled .dylib next to them --
    the dyld failure originates in libggml-metal.dylib, not the executable."""
    candidates: list[Path] = list(binaries)
    bin_dir = install_dir / "build" / "bin"
    if bin_dir.is_dir():
        candidates.extend(sorted(bin_dir.rglob("*.dylib")))

    issues: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        minos = macho_minimum_macos(path, host)
        if minos is not None and not host_supports_macos_minos(host, minos):
            issues.append(
                f"{path.name}: built for macOS {minos[0]}.{minos[1]} > "
                f"host macOS {host.macos_version[0]}.{host.macos_version[1]}"
            )
    return issues


def preflight_macos_installed_binaries(
    binaries: Iterable[Path], install_dir: Path, host: HostInfo
) -> None:
    """Reject a macos prebuilt whose minimum-OS is newer than the host. The
    upstream selector pins a loadable release up front, so here this is the
    post-download backstop; the published/fork path also uses it to advance the
    walk-back. No-op when the host macOS version is unknown (runtime validates)."""
    if not host.is_macos or host.macos_version is None:
        return
    issues = macos_binary_minos_issues(binaries, install_dir, host)
    if issues:
        raise PrebuiltFallback(
            "macos prebuilt requires a newer macOS than this host:\n" + "\n".join(issues)
        )


def preflight_linux_installed_binaries(
    binaries: Iterable[Path],
    install_dir: Path,
    host: HostInfo,
    *,
    allow_skipped_probe: bool = True,
) -> None:
    if not host.is_linux:
        return

    issues: list[str] = []
    for binary_path in binaries:
        env = binary_env(binary_path, install_dir, host)
        probe_result = _run_validation_ldd_probe(binary_path, env = env)
        if probe_result.status == _LINUX_LDD_PROBE_ERROR:
            raise PrebuiltFallback(
                f"linux extracted binary ldd probe errored for {binary_path.name}: "
                f"{probe_result.reason}"
            )
        if probe_result.status == _LINUX_LDD_PROBE_SKIPPED:
            if allow_skipped_probe:
                log(
                    f"linux extracted binary ldd probe skipped for {binary_path.name}"
                    + (f": {probe_result.reason}" if probe_result.reason else "")
                )
                continue
            issues.append(
                f"{binary_path.name}: linux ldd probe skipped"
                + (f" ({probe_result.reason})" if probe_result.reason else "")
            )
            continue
        missing = probe_result.missing
        if not missing:
            continue
        runtime_dirs = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
        issues.append(
            f"{binary_path.name}: missing={','.join(missing)} "
            f"ld_library_path={','.join(runtime_dirs) if runtime_dirs else 'none'}"
        )

    if issues:
        raise PrebuiltFallback("linux extracted binary preflight failed:\n" + "\n".join(issues))


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
    required_patterns: Iterable[str], candidate_dirs: Iterable[str] | None = None
) -> list[str]:
    directories = list(candidate_dirs) if candidate_dirs is not None else windows_runtime_dirs()
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


def _wsl_system_rocm_lib_dirs() -> list[str]:
    """System ROCm lib dir(s) for binary_env to load before a prebuilt's HIP.

    A prebuilt bundles a bare-metal HIP runtime that can't drive WSL's /dev/dxg
    and segfaults on the first GPU call -> validation fails, install falls back
    to a CPU build. Putting the system ROCm libs first loads the WSL-capable
    HIP (libamdhip64 + librocdxg) while the bundle still supplies libggml-hip /
    librocblas with the gfx1151 kernels. Strict no-op off WSL (needs /dev/dxg, a
    "microsoft" /proc/version, and a librocdxg-providing ROCm).
    """
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except OSError:
        return []
    out: list[str] = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            out.append(d)
    return out


# Secrets a downloaded llama.cpp binary never needs; keep them out of binary_env().
# The installer's own API calls read os.environ directly, so auth is unaffected.
_SECRET_ENV_EXACT_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_SECRET",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_RUNTIME_TOKEN",
        # Credential pointers (cluster / remote-host access).
        "KUBECONFIG",
        "SSH_AUTH_SOCK",
    }
)
# Case-insensitive substring markers for names we do not enumerate (no bare "KEY",
# which would hit benign runtime vars).
_SECRET_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "API_KEY",
)
# Proxy / index URLs embed creds in their value; the offline binaries never need them.
_SECRET_ENV_URL_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "FTP_PROXY",
        "RSYNC_PROXY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
    }
)
# Also drop values with URL userinfo creds (scheme://user:secret@host or token@host).
_URL_USERINFO_CREDENTIAL_RE = re.compile(r"://[^/@\s]+@")


def is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return (
        upper in _SECRET_ENV_EXACT_NAMES
        or upper in _SECRET_ENV_URL_NAMES
        or any(marker in upper for marker in _SECRET_ENV_MARKERS)
    )


def scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Drop secret-bearing variables before handing an env to a downloaded binary."""
    return {
        key: value
        for key, value in env.items()
        if not is_secret_env_name(key) and not _URL_USERINFO_CREDENTIAL_RE.search(value or "")
    }


# Home / cache pointers to on-disk token stores (~/.cache/huggingface/token,
# ~/.aws/credentials, ...). Stripping env tokens is not enough; point these at an
# empty home so the binary cannot read those files via $HOME.
_RUNTIME_HOME_POINTER_VARS = (
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_CACHE",
)
# Credential / config file pointers outside HOME; drop so lookups fall back to the
# empty home.
_CREDENTIAL_FILE_POINTER_VARS = (
    "NETRC",
    "PIP_CONFIG_FILE",
    "DOCKER_CONFIG",
    "GIT_CONFIG_GLOBAL",
)
# GitHub Actions command files: appending to these injects PATH/env into later steps.
_CI_COMMAND_FILE_VARS = (
    "GITHUB_ENV",
    "GITHUB_PATH",
    "GITHUB_OUTPUT",
    "GITHUB_STEP_SUMMARY",
    "BASH_ENV",
)
# Python-first sandbox helpers must not inherit host import roots.
_PYTHON_IMPORT_POINTER_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
)
_LD_ALLOWED_ENV_NAMES = frozenset({"LD_LIBRARY_PATH"})
_DYLD_ALLOWED_ENV_NAMES = frozenset({"DYLD_LIBRARY_PATH"})

_isolated_runtime_home_dir: str | None = None


def isolated_runtime_home() -> str:
    # Empty dir, created lazily and removed at exit. (A binary resolving the real
    # home via getpwuid is out of scope; that needs OS sandboxing.)
    global _isolated_runtime_home_dir
    if _isolated_runtime_home_dir is None:
        path = tempfile.mkdtemp(prefix = "unsloth-prebuilt-home-")
        atexit.register(shutil.rmtree, path, ignore_errors = True)
        _isolated_runtime_home_dir = path
    return _isolated_runtime_home_dir


def scrubbed_environ() -> dict[str, str]:
    # os.environ minus secrets, with home / credential pointers neutralised. Used for
    # the binary env and any probe (e.g. ldd) that runs the untrusted binary.
    env = scrub_env(os.environ.copy())
    runtime_home = isolated_runtime_home()
    for pointer in _RUNTIME_HOME_POINTER_VARS:
        env[pointer] = runtime_home
    # Windows rebuilds the profile from %HOMEDRIVE%%HOMEPATH% (no-op pair on POSIX).
    drive, tail = os.path.splitdrive(runtime_home)
    env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail or runtime_home
    for pointer in (
        *_CREDENTIAL_FILE_POINTER_VARS,
        *_CI_COMMAND_FILE_VARS,
        *_PYTHON_IMPORT_POINTER_VARS,
    ):
        env.pop(pointer, None)
    for key in tuple(env):
        if key.upper().startswith("LD_") and key.upper() not in _LD_ALLOWED_ENV_NAMES:
            env.pop(key, None)
            continue
        if key.upper().startswith("DYLD_") and key.upper() not in _DYLD_ALLOWED_ENV_NAMES:
            env.pop(key, None)
    return env


def binary_env(
    binary_path: Path,
    install_dir: Path,
    host: HostInfo,
    *,
    runtime_line: str | None = None,
) -> dict[str, str]:
    env = scrubbed_environ()
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
        # WSL: system HIP before the bundle's (which segfaults on /dev/dxg).
        _wsl_rocm = _wsl_system_rocm_lib_dirs()
        if _wsl_rocm:
            ld_dirs = [*_wsl_rocm, *ld_dirs]
            env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
        existing = [
            str(_resolve_existing_path(Path(part)))
            for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if part and not _is_broad_sandbox_library_path(part)
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dedupe_existing_dirs([*ld_dirs, *existing]))
    elif host.is_macos:
        dyld_dirs = [str(binary_path.parent), str(install_dir)]
        existing = [
            str(_resolve_existing_path(Path(part)))
            for part in env.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
            if part and not _is_broad_sandbox_library_path(part, require_library_dir = True)
        ]
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dedupe_existing_dirs([*dyld_dirs, *existing]))
    return env


def validate_quantize(
    quantize_path: Path,
    probe_path: Path,
    quantized_path: Path,
    install_dir: Path,
    host: HostInfo,
    *,
    runtime_line: str | None = None,
    require_launch: bool = False,
) -> None:
    env = binary_env(quantize_path, install_dir, host, runtime_line = runtime_line)
    plan = build_validation_sandbox_plan(
        [str(quantize_path), str(probe_path), str(quantized_path), "Q6_K", "2"],
        binary_path = quantize_path,
        install_dir = install_dir,
        host = host,
        runtime_line = runtime_line,
        purpose = _VALIDATION_PURPOSE_QUANTIZE,
        env = env,
    )
    try:
        result = _run_validation_capture(plan, timeout = 120)
    except ValidationLaunchUnavailable as exc:
        if require_launch:
            raise PrebuiltFallback(f"llama-quantize validation unavailable: {exc}") from exc
        log(f"llama-quantize validation skipped: {exc}")
        return
    if result.returncode != 0 or not quantized_path.exists() or quantized_path.stat().st_size == 0:
        combined = result.stdout + ("\n" + result.stderr if result.stderr else "")
        # Backstop for prebuilts the static minos scan could not read: a dyld
        # "built for macOS N" / missing Metal symbol failure means this binary
        # needs a newer macOS than the host, so fall back to an older release.
        prefix = (
            "macos prebuilt requires a newer macOS than this host: "
            if looks_like_macos_incompatibility(combined)
            else ""
        )
        raise PrebuiltFallback(prefix + "llama-quantize validation failed:\n" + combined)


def validate_server(
    server_path: Path,
    probe_path: Path,
    host: HostInfo,
    install_dir: Path,
    *,
    runtime_line: str | None = None,
    install_kind: str | None = None,
    require_launch: bool = False,
) -> None:
    last_failure: PrebuiltFallback | None = None
    gpu_backend: Literal["cuda", "rocm"] | None = None
    for port_attempt in range(1, SERVER_PORT_BIND_ATTEMPTS + 1):
        port = free_local_port()
        env = binary_env(server_path, install_dir, host, runtime_line = runtime_line)
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
            "linux-arm64-cuda",
            "linux-rocm",
            "windows-cuda",
            "windows-hip",
            "windows-rocm",
        }
        if install_kind is not None:
            _enable_gpu_layers = install_kind in _gpu_kinds
            if install_kind in {"linux-cuda", "linux-arm64-cuda"}:
                gpu_backend = "cuda"
            elif install_kind == "linux-rocm":
                gpu_backend = "rocm"
        else:
            # Older call sites that don't pass install_kind: keep ROCm
            # hosts in the GPU-validation path so an AMD-only Linux host
            # is exercised against the actual hardware rather than the
            # CPU fallback. NVIDIA stays covered here.
            _enable_gpu_layers = host.has_usable_nvidia or host.has_rocm
            if host.is_linux and host.has_usable_nvidia:
                gpu_backend = "cuda"
            elif host.is_linux and host.has_rocm:
                gpu_backend = "rocm"
        if _enable_gpu_layers:
            command.extend(["--n-gpu-layers", "1"])
        plan = build_validation_sandbox_plan(
            command,
            binary_path = server_path,
            install_dir = install_dir,
            host = host,
            runtime_line = runtime_line,
            purpose = _VALIDATION_PURPOSE_SERVER,
            env = env,
            enable_gpu_layers = _enable_gpu_layers,
            gpu_backend = gpu_backend,
        )
        if plan.server_probe_mode == _VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX:
            started_at = time.time()
            try:
                result = _run_validation_capture(
                    plan,
                    timeout = _LINUX_SERVER_VALIDATION_HELPER_CAPTURE_TIMEOUT_SECONDS,
                )
            except ValidationLaunchUnavailable as exc:
                if require_launch:
                    raise PrebuiltFallback(f"llama-server validation unavailable: {exc}") from exc
                log(f"llama-server validation skipped: {exc}")
                return
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode == 0:
                return
            exited_quickly = (time.time() - started_at) <= SERVER_BIND_RETRY_WINDOW_SECONDS
            failure = PrebuiltFallback("llama-server validation failed inside sandbox:\n" + output)
            if port_attempt < SERVER_PORT_BIND_ATTEMPTS and is_retryable_server_bind_error(
                RuntimeError(output),
                output,
                exited_quickly = exited_quickly,
            ):
                log(
                    f"llama-server startup hit a port race on {port}; retrying with a fresh port "
                    f"({port_attempt}/{SERVER_PORT_BIND_ATTEMPTS})"
                )
                last_failure = failure
                continue
            raise failure

        log_fd, log_name = tempfile.mkstemp(prefix = "llama-server-", suffix = ".log")
        os.close(log_fd)
        log_path = Path(log_name)
        process: subprocess.Popen[str] | None = None
        try:
            with log_path.open("w", encoding = "utf-8", errors = "replace") as log_handle:
                try:
                    process = _run_validation_popen(
                        plan,
                        stdout = log_handle,
                    )
                except ValidationLaunchUnavailable as exc:
                    if require_launch:
                        raise PrebuiltFallback(
                            f"llama-server validation unavailable: {exc}"
                        ) from exc
                    log(f"llama-server validation skipped: {exc}")
                    return
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
                        failure = PrebuiltFallback("llama-server exited during startup:\n" + output)
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

                    payload = json.dumps({"prompt": "a", "n_predict": 1}).encode("utf-8")
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
                            last_error = RuntimeError(f"unexpected HTTP status {status_code}")
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


def collect_system_report(host: HostInfo, choice: AssetChoice | None, install_dir: Path) -> str:
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
            "linux_runtime_lines=" + (",".join(runtime_lines) if runtime_lines else "none")
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
            ldd_probe = _run_validation_ldd_probe(server_binary, env = server_env)
            lines.append("linux_missing_libs_probe=" + ldd_probe.status)
            if ldd_probe.reason:
                lines.append("linux_missing_libs_probe_reason=" + ldd_probe.reason)
            if ldd_probe.status == _LINUX_LDD_PROBE_OK:
                lines.append(
                    "linux_missing_libs="
                    + (",".join(ldd_probe.missing) if ldd_probe.missing else "none")
                )
            lines.append(
                "linux_runtime_dirs="
                + (
                    ",".join(
                        [
                            part
                            for part in server_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                            if part
                        ]
                    )
                    or "none"
                )
            )
            try:
                if ldd_probe.status == _LINUX_LDD_PROBE_OK:
                    lines.append("ldd llama-server:")
                    lines.append((ldd_probe.output or "none").strip())
                else:
                    lines.append("ldd llama-server: skipped")
            except Exception as exc:
                lines.append(f"ldd error: {exc}")
    elif host.is_windows:
        lines.append("windows_runtime_dirs=" + (",".join(windows_runtime_dirs()) or "none"))
        runtime_lines, runtime_dirs = detected_windows_runtime_lines()
        lines.append(
            "windows_runtime_lines=" + (",".join(runtime_lines) if runtime_lines else "none")
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
    attempts: Iterable[AssetChoice], checksums: ApprovedReleaseChecksums
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


def exact_source_asset_url(
    approved_checksums: ApprovedReleaseChecksums,
    source_repo: str,
    source_archive: ApprovedArtifactHash | None,
    exact_source: bool,
    release_tag: str,
) -> str | None:
    """Release-asset URL for a mix build's merged source tree, or None.

    A mix build's merge commit is never pushed, so its codeload/archive URLs
    404; the only durable copy of the merged tree is the
    ``llama.cpp-source-commit-<sha>.tar.gz`` asset published alongside the
    prebuilt. Resolve its host and tag defensively so a manifest that omits the
    top-level ``repo``/``release_tag`` still reaches the asset: prefer the
    artifact's own repo, then the manifest repo, then the source repo; and prefer
    the manifest release tag, then the tag we actually installed the prebuilt
    from (its sibling on the same release). Without this the empty fields drop the
    only working URL and hydration falls through to the 404-ing commit archive.
    """
    if not exact_source or source_archive is None:
        return None
    return release_asset_download_url(
        source_archive.repo or approved_checksums.repo or source_repo,
        approved_checksums.release_tag or release_tag,
        source_archive.asset_name,
    )


def selected_source_archive_metadata(
    checksums: ApprovedReleaseChecksums, llama_tag: str
) -> tuple[str, str | None]:
    _source_repo, _source_ref, source_archive, _exact_source = preferred_source_archive(
        checksums, llama_tag
    )
    if source_archive is None:
        return source_archive_logical_name(llama_tag), None
    return source_archive.asset_name, source_archive.sha256


def resolve_install_attempts(
    llama_tag: str, host: HostInfo, published_repo: str, published_release_tag: str
) -> tuple[str, str, list[AssetChoice], ApprovedReleaseChecksums]:
    requested_tag, plans = _fork_manifest_release_plans(
        llama_tag,
        host,
        published_repo,
        published_release_tag,
    )
    if not plans:
        raise PrebuiltFallback("no prebuilt release plans were available")
    plan = plans[0]
    return requested_tag, plan.llama_tag, plan.attempts, plan.approved_checksums


def _linux_published_attempts(host: HostInfo, bundle: PublishedReleaseBundle) -> list[AssetChoice]:
    """Build the install attempts for a fork Linux host from a manifest-described
    bundle: CUDA, per-gfx ROCm, or (non-GPU) CPU. Same selection the upstream
    filename path used, just sourced from the manifest instead of reconstructed
    from asset names."""
    attempts: list[AssetChoice] = []
    if host.has_usable_nvidia:
        # Prefer the cudart major Studio loads at runtime (torch's bundled
        # libcudart), not the newest detected on disk. Without this a stray
        # cuda13 runtime outranks the torch cuda12 the binary links against.
        torch_preference = detect_torch_cuda_runtime_preference(host)
        selection = linux_cuda_choice_from_release(
            host,
            bundle,
            preferred_runtime_line = torch_preference.runtime_line,
            selection_preamble = torch_preference.selection_log,
        )
        if selection is not None:
            attempts.extend(selection.attempts)
    elif host.has_rocm:
        # Use the fork's own per-gfx ROCm bundle (hash-approved, ships the full
        # ROCm runtime). Do NOT append the CPU asset for ROCm-only hosts: if no
        # bundle covers the GPU we want validate_prebuilt_attempts to raise
        # PrebuiltFallback so the caller triggers the HIP source build, not
        # silently install a CPU-only binary.
        published_rocm = published_rocm_choice_for_host(bundle, host, "linux-rocm")
        if published_rocm is not None:
            attempts.append(published_rocm)
    else:
        # CPU-only host. A usable-NVIDIA host never reaches here -- if its CUDA
        # selection produced nothing we want an empty attempt list so the caller
        # source-builds with CUDA, not a CPU-only binary silently installed on a
        # GPU host (mirrors the ROCm branch, and Windows NVIDIA).
        cpu_choice = published_asset_choice_for_kind(bundle, "linux-cpu")
        if cpu_choice is not None:
            attempts.append(cpu_choice)
    return attempts


def _fork_manifest_release_plans(
    llama_tag: str,
    host: HostInfo,
    published_repo: str,
    published_release_tag: str,
    *,
    max_release_fallbacks: int = DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS,
) -> tuple[str, list[InstallReleasePlan]]:
    """Manifest-reading branch of resolve_simple_install_release_plans, used for
    the fork's bundles whose GPU/arch coverage lives in
    llama-prebuilt-manifest.json rather than in the filename: arm64 CUDA, Windows
    CUDA, per-gfx ROCm, and macOS. Linux x64 takes the faster filename path."""
    requested_tag = normalized_requested_llama_tag(llama_tag)
    allow_older_release_fallback = requested_tag == "latest" and not published_release_tag
    release_limit = max(1, max_release_fallbacks)
    # macOS may need to walk past a run of too-new prebuilts. Only when the host
    # version is known; otherwise keep the default (cannot tell up front).
    if host.is_macos and allow_older_release_fallback and host.macos_version is not None:
        release_limit = max(release_limit, DEFAULT_MAX_MACOS_RELEASE_FALLBACKS)
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
            if host.is_linux:
                linux_attempts = _linux_published_attempts(host, bundle)
                if not linux_attempts:
                    raise PrebuiltFallback("no compatible Linux prebuilt asset was found")
                attempts = apply_approved_hashes(linux_attempts, checksums)
                if not attempts:
                    raise PrebuiltFallback("no compatible Linux prebuilt asset was found")
                if attempts[0].selection_log:
                    log_lines(attempts[0].selection_log)
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
        # Binary-side repo/tag for non-fork sources (e.g. the ggml-org upstream
        # CPU/HIP prebuilts). published_repo/release_tag always refer to the
        # unsloth source tree; these capture where the actual binaries came from
        # so the install summary can show both.
        "binary_repo": choice.repo,
        "binary_release_tag": choice.tag,
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
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(metadata, indent = 2) + "\n")


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
    if choice.install_kind in {"linux-cpu", "linux-arm64"}:
        return [
            ["libllama-common.so*"],
            ["libllama.so*"],
            ["libggml.so*"],
            ["libggml-base.so*"],
            ["libggml-cpu*.so*"],
            ["libmtmd.so*"],
        ]
    if choice.install_kind in {"linux-cuda", "linux-arm64-cuda"}:
        return [
            ["libllama-common.so*"],
            ["libllama.so*"],
            ["libggml.so*"],
            ["libggml-base.so*"],
            ["libggml-cpu*.so*"],
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
            ["libggml-cpu*.so*"],
            ["libmtmd.so*"],
            ["libggml-hip.so*"],
        ]
    if choice.install_kind in {"windows-cpu", "windows-arm64"}:
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
    if choice.install_kind in {"windows-hip", "windows-rocm"}:
        return [["llama.dll"], ["*hip*.dll"]]
    return []


def install_runtime_dir(install_dir: Path, host: HostInfo) -> Path:
    if host.is_windows:
        return install_dir / "build" / "bin" / "Release"
    return install_dir / "build" / "bin"


def runtime_payload_is_healthy(install_dir: Path, host: HostInfo, choice: AssetChoice) -> bool:
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
    if host.is_linux:
        try:
            preflight_linux_installed_binaries(
                [runtime_dir / "llama-server", runtime_dir / "llama-quantize"],
                install_dir,
                host,
                allow_skipped_probe = True,
            )
        except Exception:
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
    install_dir: Path, host: HostInfo, plan: InstallReleasePlan
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
    # For an exact (mix) source the merge commit lives only in the release asset,
    # not in any repo, so fetch the asset directly; codeload stays the fallback.
    asset_url = exact_source_asset_url(
        approved_checksums, source_repo, source_archive, exact_source, release_tag
    )
    if exact_source:
        log(f"hydrating exact llama.cpp source for {source_repo}@{source_ref} into {install_dir}")
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
        asset_url = asset_url,
    )
    log(f"overlaying prebuilt bundle {choice.name} into {install_dir}")
    server_path, quantize_path = install_from_archives(choice, host, install_dir, work_dir)
    preflight_linux_installed_binaries((server_path, quantize_path), install_dir, host)
    preflight_macos_installed_binaries((server_path, quantize_path), install_dir, host)
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
    # Hashless external prebuilts are not in the approved-sha256
    # manifest and rely on the functional smoke test as their only integrity gate,
    # so they are always validated. For an approved bundle the sha256 manifest
    # already proves integrity, so its runtime smoke test -- a cold CUDA-JIT pass
    # costing minutes on Blackwell sm_100 -- is gated behind
    # _RUN_STAGED_PREBUILT_VALIDATION, disabled for now. The check and the
    # source-build fallback it triggers are kept intact; flip the flag to restore it.
    smoke_validation_required = choice.expected_sha256 is None
    if smoke_validation_required or _RUN_STAGED_PREBUILT_VALIDATION:
        validate_quantize(
            quantize_path,
            probe_path,
            quantized_path,
            install_dir,
            host,
            runtime_line = choice.runtime_line,
            require_launch = smoke_validation_required,
        )
        validate_server(
            server_path,
            probe_path,
            host,
            install_dir,
            runtime_line = choice.runtime_line,
            install_kind = choice.install_kind,
            require_launch = smoke_validation_required,
        )
        log(f"staged prebuilt validation completed for {choice.name}")
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

        if (
            existing_install_dir is not None
            and existing_install_matches_choice(
                existing_install_dir,
                host,
                llama_tag = llama_tag,
                release_tag = release_tag,
                choice = attempt,
                approved_checksums = approved_checksums,
            )
            # Skip a matching candidate unless it still needs the DiffusionGemma
            # backfill re-extract (gated per-attempt, not per-plan).
            and not diffusion_visual_server_backfill_needed(existing_install_dir, host, attempt)
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


def diffusion_visual_server_backfill_needed(
    install_dir: Path, host: HostInfo, choice: AssetChoice
) -> bool:
    """True when an existing install matches the tag but lacks the DiffusionGemma
    visual-server the chosen bundle ships. An install made before the visual-server
    entered the copy allowlist matches on tag yet is missing the binary, so the
    tag-match skip never backfills it (DiffusionGemma then fails with "runner not
    found"). Gated to the fork ("published") bundles that actually carry it, so
    upstream installs -- which never ship it -- can't thrash on repeated updates.
    Once a re-extract lands the binary this returns False, so it self-limits."""
    if choice.source_label != "published":
        return False
    name = "llama-diffusion-gemma-visual-server" + (".exe" if host.is_windows else "")
    if name not in runtime_patterns_for_choice(choice):
        return False
    for cand in (
        install_dir / name,
        install_dir / "build" / "bin" / name,
        install_dir / "build" / "bin" / "Release" / name,
    ):
        if cand.is_file():
            return False
    return True


def install_prebuilt(
    install_dir: Path,
    llama_tag: str,
    published_repo: str,
    published_release_tag: str,
    *,
    override_has_rocm: bool = False,
    override_rocm_gfx: str | None = None,
    force_cpu: bool = False,
) -> None:
    host = detect_host()
    host = _apply_host_overrides(
        host,
        override_has_rocm = override_has_rocm,
        override_rocm_gfx = override_rocm_gfx,
        force_cpu = force_cpu,
    )
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
            # Single resolver: linux-x64 takes the fast filename path internally,
            # every other fork host reads the manifest.
            requested_tag, release_plans = resolve_simple_install_release_plans(
                llama_tag,
                host,
                published_repo,
                published_release_tag,
            )
            if release_plans and existing_install_matches_plan(install_dir, host, release_plans[0]):
                current = release_plans[0]
                if diffusion_visual_server_backfill_needed(install_dir, host, current.attempts[0]):
                    log(
                        f"existing install matches {current.release_tag} but is missing the "
                        "DiffusionGemma visual-server; re-extracting the bundle to backfill it"
                    )
                else:
                    log(
                        "existing llama.cpp install already matches selected release "
                        f"{current.release_tag} upstream_tag={current.llama_tag}; skipping download and install"
                    )
                    return
            with tempfile.TemporaryDirectory(prefix = "unsloth-llama-prebuilt-") as tmp:
                work_dir = Path(tmp)
                probe_path = work_dir / "stories260K.gguf"
                download_validation_model(probe_path, validation_model_cache_path(install_dir))
                release_count = len(release_plans)
                for release_index, plan in enumerate(release_plans):
                    choice = plan.attempts[0]
                    backfill = diffusion_visual_server_backfill_needed(install_dir, host, choice)
                    if existing_install_matches_plan(install_dir, host, plan):
                        if backfill:
                            log(
                                f"existing install matches fallback {plan.release_tag} but is missing "
                                "the DiffusionGemma visual-server; re-extracting to backfill it"
                            )
                        else:
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
                            # Skip is gated per-attempt inside, so pass the dir always.
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
                    try:
                        ensure_diffusion_visual_server(
                            install_dir, host, plan.release_tag, plan.approved_checksums
                        )
                    except Exception as exc:
                        log(
                            "diffusion visual server step skipped; install remains valid "
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
        "--has-rocm",
        action = "store_true",
        default = False,
        help = (
            "Assert that an AMD ROCm GPU is present. When set, skips the internal "
            "hipinfo/amd-smi probe and forces has_rocm=True in the host profile. "
            "Used by setup.ps1/setup.sh to forward their own ROCm detection result "
            "so the HIP llama.cpp prebuilt is selected even when hipinfo is not on PATH."
        ),
    )
    parser.add_argument(
        "--rocm-gfx",
        default = os.environ.get("UNSLOTH_ROCM_GFX_ARCH"),
        help = (
            "Forward the AMD gfx target (e.g. gfx1151) that setup.ps1/setup.sh "
            "resolved, so the per-gfx ROCm prebuilt is selected even when the "
            "installer's own hipinfo/amd-smi probe cannot report it. Implies "
            "--has-rocm. Defaults to the UNSLOTH_ROCM_GFX_ARCH environment variable."
        ),
    )
    parser.add_argument(
        "--cpu-fallback",
        action = "store_true",
        default = False,
        help = (
            "Select the CPU prebuilt for this OS/arch even when a GPU is present. "
            "setup.sh uses this as a last resort for arm64 Linux GPU hosts whose "
            "source build failed (no arm64 CUDA prebuilt exists anywhere)."
        ),
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
    resolve_group.add_argument(
        "--resolve-prebuilt",
        nargs = "?",
        const = "latest",
        help = (
            "Report whether an official prebuilt exists for this host without "
            "downloading. Picks the host's published repo when --published-repo "
            "is left at the default. Use --output-format json."
        ),
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
                "requested_tag": normalized_requested_llama_tag(args.resolve_install_tag),
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
                "requested_tag": normalized_requested_llama_tag(args.resolve_source_build),
                "source_url": plan.source_url,
                "source_ref_kind": plan.source_ref_kind,
                "source_ref": plan.source_ref,
                "compatibility_upstream_tag": plan.compatibility_upstream_tag,
            },
            output_format = args.output_format,
        )
        return EXIT_SUCCESS

    if args.resolve_prebuilt is not None:
        # Host-aware "is a prebuilt available" probe, no download. A default repo
        # means "pick the repo for this host"; PrebuiltFallback == source build.
        host = _apply_host_overrides(
            detect_host(),
            override_has_rocm = args.has_rocm,
            override_rocm_gfx = args.rocm_gfx,
            force_cpu = args.cpu_fallback,
        )
        # setup.sh routes Linux hosts with AMD tooling to the fork even when no GPU
        # is probed; mirror that so a HIP source build is not offered a CPU prebuilt.
        amd_tooling = host.is_linux and any(
            shutil.which(t) for t in ("rocminfo", "amd-smi", "hipconfig", "hipinfo")
        )
        repo = (
            published_repo_for_host(host, linux_amd_tooling_present = amd_tooling)
            if args.published_repo == DEFAULT_PUBLISHED_REPO
            else args.published_repo
        )
        try:
            _requested, plans = resolve_simple_install_release_plans(
                args.resolve_prebuilt, host, repo, args.published_release_tag or ""
            )
            choice = plans[0].attempts[0] if plans and plans[0].attempts else None
            if choice is None:
                payload = {"prebuilt_available": False, "repo": repo}
            else:
                payload = {
                    "prebuilt_available": True,
                    "repo": repo,
                    "release_tag": plans[0].release_tag,
                    "llama_tag": plans[0].llama_tag,
                    "asset": choice.name,
                    "install_kind": choice.install_kind,
                }
        except PrebuiltFallback:
            payload = {"prebuilt_available": False, "repo": repo}
        emit_resolver_output(payload, output_format = args.output_format)
        return EXIT_SUCCESS

    if not args.install_dir:
        raise SystemExit(
            "install_llama_prebuilt.py: --install-dir is required unless --resolve-llama-tag, --resolve-install-tag, or --resolve-source-build is used"
        )
    # Install path only: route status logs to stdout (see _LOG_TO_STDOUT note).
    global _LOG_TO_STDOUT
    _LOG_TO_STDOUT = True
    install_prebuilt(
        install_dir = Path(args.install_dir).expanduser().resolve(),
        llama_tag = args.llama_tag,
        published_repo = args.published_repo,
        published_release_tag = args.published_release_tag or "",
        override_has_rocm = args.has_rocm,
        override_rocm_gfx = args.rocm_gfx,
        force_cpu = args.cpu_fallback,
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
