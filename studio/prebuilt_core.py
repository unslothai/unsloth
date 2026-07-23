#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component-agnostic machinery shared by the ggml-family prebuilt installers.

``install_llama_prebuilt.py`` and ``install_whisper_prebuilt.py`` both run on
this module, which owns two layers:

1. Primitives: HTTP fetches with retries and token-safe redirects, verified
   downloads, safe archive extraction, the install lock, sha256 helpers, and
   CUDA runtime-line / compute-capability selection. Moved verbatim out of
   ``install_llama_prebuilt.py``, which re-exports them under their old names.

2. The generic descriptor-driven install flow: release resolution (download
   host fast path, GitHub API fallback), checksum-index parsing, coverage-aware
   artifact selection, verified download + extraction + staged atomic
   activation, marker/fingerprint handling, and the resolve probe. Whisper runs
   entirely on this flow; a third component plugs in with a ``ComponentDescriptor``.

Seam rule: every function calling a collaborator tests monkeypatch takes an
``ops`` handle first. ``ops`` resolves names in the calling installer module's
globals first (so a ``monkeypatch.setattr`` still wins) and falls back to this
module's defaults, letting a descriptor-only component work with no installer
module behind it.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import json
import os
import random
import re
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
from typing import Any, Callable, Iterable, Iterator

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None


class PrebuiltFallback(RuntimeError):
    pass


class BusyInstallConflict(RuntimeError):
    pass


# ── Defaults a component module may override via its own globals ──
USER_AGENT = "unsloth-studio-prebuilt"
GITHUB_AUTH_HOSTS = {"api.github.com", "github.com"}
HF_AUTH_HOSTS = {"huggingface.co", "www.huggingface.co"}
RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HTTP_FETCH_ATTEMPTS = 4
HTTP_FETCH_BASE_DELAY_SECONDS = 0.75
JSON_FETCH_ATTEMPTS = 3
TTY_PROGRESS_START_DELAY_SECONDS = 0.5
INSTALL_LOCK_TIMEOUT_SECONDS = 300
INSTALL_STAGING_ROOT_NAME = ".staging"
SCHEMA_VERSION = 1
# Backend to retry when the preferred one has no covering asset; None disables it.
FALLBACK_BACKEND: str | None = "cpu"
_RATE_LIMIT_WAIT_CAP_SECONDS = 60.0

# Lowest CUDA major we ship prebuilts for, and highest we probe for installed
# runtime libraries. Detection and runtime-line derivation are generated per
# major, so a new toolkit (cuda14, ...) needs no code change while ggml keeps
# the cudart64_<major>.dll / libcudart.so.<major> naming.
_MIN_CUDA_MAJOR = 12
_MAX_PROBE_CUDA_MAJOR = 19

# Blackwell floor is sm_100 (B100/B200 sm_100, B300/GB300 sm_103 below consumer
# RTX 50 sm_120); the family needs toolkit >= 12.8, sm_103/sm_121 need 12.9.
_BLACKWELL_MIN_SM = 100
_BLACKWELL_MIN_TOOLKIT = (12, 8)
_BLACKWELL_SM_MIN_TOOLKIT = {103: (12, 9), 121: (12, 9)}


def log(message: str) -> None:
    print(f"[prebuilt-core] {message}", file = sys.stderr)


def _cuda_runtime_lines_for_major(major: int) -> list[str]:
    """Runtime lines a driver of this CUDA major can use (its own major and any
    older one, newest first down to the minimum we ship)."""
    return [f"cuda{m}" for m in range(major, _MIN_CUDA_MAJOR - 1, -1)]


# ── The ops seam ──
_MISSING = object()

# Core functions expecting an ``ops`` first argument. ModuleOps binds itself when
# a lookup falls back to the core default, so a descriptor-only component gets
# working defaults while an installer wrapper (or a test monkeypatch) always wins.
_OPS_FIRST_NAMES = {
    "auth_headers",
    "github_api_headers",
    "download_bytes",
    "fetch_json",
    "download_file",
    "download_file_verified",
    "download_file_verified_strict",
    "github_release",
    "github_release_assets",
    "download_host_latest_release_tag",
    "fetch_download_host_json",
    "linux_runtime_dirs_for_required_libraries",
    "detected_linux_runtime_lines",
    "detected_windows_runtime_lines",
    "parse_manifest",
    "parse_release_checksums",
    "fetch_release_checksums",
    "expected_sha256_for",
    "macos_min_os_ok",
    "artifacts_for_host",
    "select_artifact",
    "select_artifact_with_fallback",
    "auto_detect_backend",
    "resolve_backend",
    "fetch_release_bundle",
    "asset_download_url",
    "resolve_newest_release_tag",
    "resolve_release_tag",
    "resolve_release_via_download_host",
    "fetch_release_for_install",
    "write_prebuilt_metadata",
    "load_prebuilt_metadata",
    "existing_install_matches",
    "metadata_path",
    "selection_from_artifact",
    "plan_selection",
    "install_from_bundle",
    "install_prebuilt",
    "resolve_prebuilt",
    "installed_server_path",
    "assemble_install_tree",
    "validate_staged_server",
    "locate_server_in_tree",
    # Underscored aliases the installer modules expose for their tests.
    "_download_host_latest_release_tag",
    "_download_host_json",
    "_resolve_release_via_download_host",
    "_install_from_bundle",
}


class ModuleOps:
    """Late-binding name resolver over an installer module's globals.

    Lookup hits the wrapped globals first (so a ``monkeypatch.setattr`` is always
    observed), then this module's defaults (ops-first defaults come back bound).
    """

    def __init__(self, module_globals: dict[str, Any]) -> None:
        self._globals = module_globals

    def __getattr__(self, name: str) -> Any:
        value = self._globals.get(name, _MISSING)
        if value is _MISSING:
            value = globals().get(name, _MISSING)
            if value is _MISSING:
                raise AttributeError(f"prebuilt component namespace has no attribute {name!r}")
            if name in _OPS_FIRST_NAMES:
                return functools.partial(value, self)
        return value


@dataclass(frozen = True)
class ComponentDescriptor:
    """Everything the generic flow needs about one ggml-family component.

    Hooks mirror the module-level names an installer defines; a descriptor-only
    component supplies them here and ``component_ops`` builds a namespace with
    core defaults behind them.
    """

    component: str  # manifest "component" value, e.g. "whisper.cpp"
    log_prefix: str  # stderr prefix, e.g. "whisper-prebuilt"
    published_repo: str  # fork publishing the prebuilt releases
    manifest_asset_name: str
    sha256_asset_name: str
    metadata_filename: str  # marker written into the install dir
    user_agent: str
    supported_backends: tuple[str, ...] = ("cpu", "cuda", "metal", "vulkan", "rocm")
    schema_version: int = SCHEMA_VERSION
    # Backend to retry when the preferred one has no covering asset: "cpu"
    # (whisper) ships a CPU bundle in every release; None (llama) reports "no
    # prebuilt" so the caller falls back to a source build.
    fallback_backend: str | None = "cpu"
    staging_root_name: str = INSTALL_STAGING_ROOT_NAME
    run_staged_validation: bool = False
    # Hooks. Each mirrors the module-level function it replaces; None keeps the
    # core default (which raises if truly required).
    detect_host: Callable[[], Any] | None = None
    host_platform_tokens: Callable[[Any], tuple[str, str]] | None = None
    server_binary_name: Callable[[Any], str] | None = None
    runtime_bin_dir: Callable[[Path, Any], Path] | None = None
    auto_detect_backend: Callable[[Any], str] | None = None


def component_namespace(descriptor: ComponentDescriptor) -> dict[str, Any]:
    """Materialize a descriptor into the module-like namespace ``ModuleOps``
    resolves against; core defaults cover anything not listed here."""
    prefix = descriptor.log_prefix

    def component_log(message: str) -> None:
        print(f"[{prefix}] {message}", file = sys.stderr)

    namespace: dict[str, Any] = {
        "COMPONENT": descriptor.component,
        "SCHEMA_VERSION": descriptor.schema_version,
        "DEFAULT_PUBLISHED_REPO": descriptor.published_repo,
        "MANIFEST_ASSET_NAME": descriptor.manifest_asset_name,
        "SHA256_ASSET_NAME": descriptor.sha256_asset_name,
        "METADATA_FILENAME": descriptor.metadata_filename,
        "SUPPORTED_BACKENDS": descriptor.supported_backends,
        "USER_AGENT": descriptor.user_agent,
        "FALLBACK_BACKEND": descriptor.fallback_backend,
        "INSTALL_STAGING_ROOT_NAME": descriptor.staging_root_name,
        "_RUN_STAGED_PREBUILT_VALIDATION": descriptor.run_staged_validation,
        "log": component_log,
    }
    for hook in (
        "detect_host",
        "host_platform_tokens",
        "server_binary_name",
        "runtime_bin_dir",
        "auto_detect_backend",
    ):
        value = getattr(descriptor, hook)
        if value is not None:
            namespace[hook] = value
    return namespace


def component_ops(descriptor: ComponentDescriptor) -> ModuleOps:
    return ModuleOps(component_namespace(descriptor))


# ── Lock/busy classification ──
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


# ── HTTP: auth, redirects, retries ──
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


def auth_headers(ops: ModuleOps, url: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": ops.USER_AGENT,
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and should_send_github_auth(url):
        headers["Authorization"] = f"Bearer {token}"
        return headers
    # Anonymous huggingface.co fetches share a per-IP rate limit CI fleets
    # exhaust (HTTP 429); authenticate when a token is available.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and should_send_hf_auth(url):
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


class _CrossHostAuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host.

    huggingface.co redirects downloads to CDN hosts whose signed URLs can reject
    a foreign Authorization header; urllib forwards headers across redirects by
    default (requests/huggingface_hub strip them).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None and parsed_hostname(newurl) != parsed_hostname(req.full_url):
            new_request.headers.pop("Authorization", None)
            new_request.unredirected_hdrs.pop("Authorization", None)
        return new_request


_URL_OPENER = urllib.request.build_opener(_CrossHostAuthStrippingRedirectHandler())


def github_api_headers(ops: ModuleOps, url: str | None = None) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        **ops.auth_headers(url),
    }


def is_github_api_url(url: str | None) -> bool:
    return parsed_hostname(url) == "api.github.com"


def is_retryable_url_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        # GitHub returns 403 (not 429) on API rate-limit; anonymous calls share a
        # 60-req/hour bucket per runner IP that CI fleets exhaust. Treat 403
        # against api.github.com as retryable so we get a backoff cycle or two
        # (honouring Retry-After / X-RateLimit-Reset) before the source-build
        # fallback fires. 403s on other hosts (private downloads, auth) stay non-retryable.
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


def _http_error_retry_delay(exc: Exception) -> float | None:
    """Recommended wait from rate-limit headers on a 403/429.

    None when no header is present or the wait exceeds
    _RATE_LIMIT_WAIT_CAP_SECONDS (the source-build fallback is faster).
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


# ── Atomic writes and hashing ──
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


def validate_schema_version(
    payload: dict[str, Any],
    *,
    label: str,
    schema_version: int = SCHEMA_VERSION,
    error: type[Exception] = RuntimeError,
) -> None:
    value = payload.get("schema_version")
    if value is None:
        return
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise error(f"{label} schema_version was not an integer") from exc
    if normalized != schema_version:
        raise error(f"{label} schema_version={normalized} is unsupported")


# ── Download progress ──
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
    UNSLOTH_PROGRESS_PERCENT_STEP=5 for finer progress lines."""
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


# ── Downloads ──
def download_bytes(
    ops: ModuleOps,
    url: str,
    *,
    timeout: int = 120,
    attempts: int | None = None,
    headers: dict[str, str] | None = None,
    progress_label: str | None = None,
) -> bytes:
    if attempts is None:
        attempts = ops.HTTP_FETCH_ATTEMPTS
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers = headers or ops.auth_headers(url))
            with ops._URL_OPENER.open(request, timeout = timeout) as response:
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
            ops.log(f"fetch failed ({attempt}/{attempts}) for {url}: {exc}; retrying")
            sleep_backoff(attempt, exc = exc)
    assert last_exc is not None
    raise last_exc


def fetch_json(ops: ModuleOps, url: str) -> Any:
    attempts = ops.JSON_FETCH_ATTEMPTS if is_github_api_url(url) else 1
    last_decode_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = ops.download_bytes(
                url,
                timeout = 30,
                headers = ops.github_api_headers(url)
                if is_github_api_url(url)
                else ops.auth_headers(url),
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
        ops.log(f"json fetch failed ({attempt}/{attempts}) for {url}; retrying")
        sleep_backoff(attempt)
    assert last_decode_exc is not None
    raise last_decode_exc


def download_file(ops: ModuleOps, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents = True, exist_ok = True)
    attempts = ops.HTTP_FETCH_ATTEMPTS
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp_path: Path | None = None
        try:
            request = urllib.request.Request(url, headers = ops.auth_headers(url))
            with tempfile.NamedTemporaryFile(
                prefix = destination.name + ".tmp-",
                dir = destination.parent,
                delete = False,
            ) as handle:
                tmp_path = Path(handle.name)
                with ops._URL_OPENER.open(request, timeout = 120) as response:
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
            if attempt >= attempts or not is_retryable_url_error(exc):
                raise
            ops.log(f"download failed ({attempt}/{attempts}) for {url}: {exc}; retrying")
            sleep_backoff(attempt, exc = exc)
    assert last_exc is not None
    raise last_exc


def download_file_verified(
    ops: ModuleOps, url: str, destination: Path, *, expected_sha256: str | None, label: str
) -> None:
    normalized_expected = normalize_sha256_digest(expected_sha256)
    if not normalized_expected:
        ops.download_file(url, destination)
        ops.log(f"downloaded {label} without a published sha256; relying on install validation")
        return

    for attempt in range(1, 3):
        ops.download_file(url, destination)
        actual_sha256 = ops.sha256_file(destination)
        if actual_sha256 == normalized_expected:
            ops.log(f"verified {label} sha256={actual_sha256}")
            return

        ops.log(
            f"{label} checksum mismatch on attempt {attempt}/2: "
            f"expected={normalized_expected} actual={actual_sha256}"
        )
        destination.unlink(missing_ok = True)
        if attempt == 2:
            raise PrebuiltFallback(
                f"{label} checksum mismatch after retry: expected={normalized_expected} actual={actual_sha256}"
            )
        ops.log(f"retrying {label} download after checksum mismatch")


def download_file_verified_strict(
    ops: ModuleOps, url: str, destination: Path, *, expected_sha256: str, label: str
) -> None:
    """Verified download with a required digest (the generic flow fails closed on
    a missing checksum, so None never reaches here)."""
    for attempt in range(1, 3):
        ops.download_file(url, destination)
        actual = ops.sha256_file(destination)
        if actual == expected_sha256:
            ops.log(f"verified {label} sha256={actual}")
            return
        ops.log(
            f"{label} checksum mismatch {attempt}/2: expected={expected_sha256} actual={actual}"
        )
        destination.unlink(missing_ok = True)
        if attempt == 2:
            raise PrebuiltFallback(f"{label} checksum mismatch after retry")


# ── GitHub release primitives ──
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


def github_release(
    ops: ModuleOps,
    repo: str,
    tag: str,
    *,
    error: type[Exception] = RuntimeError,
) -> dict[str, Any]:
    payload = ops.fetch_json(
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe = '')}"
    )
    if not isinstance(payload, dict):
        raise error(f"unexpected release payload for {repo}@{tag}")
    return payload


def github_release_assets(ops: ModuleOps, repo: str, tag: str) -> dict[str, str]:
    payload = ops.fetch_json(
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe = '')}"
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected release payload for {repo}@{tag}")
    return release_asset_map(payload)


def release_asset_download_url(repo: str, release_tag: str, asset_name: str) -> str:
    """Tag-pinned asset URL on the release-assets CDN (github.com redirect, NOT
    api.github.com, so no 60-req/hour unauthenticated limit)."""
    return (
        f"https://github.com/{urllib.parse.quote(repo, safe = '/')}/releases/download/"
        f"{urllib.parse.quote(release_tag, safe = '')}/"
        f"{urllib.parse.quote(asset_name, safe = '')}"
    )


def download_host_latest_release_tag(ops: ModuleOps, repo: str) -> str | None:
    """Latest release tag from github.com/<repo>/releases/latest via its redirect
    target (no api.github.com call). None on 404 so the caller falls back to the
    API. /releases/latest resolves by created_at/make_latest, which can lag the
    published_at newest the freshness check uses -- acceptable for install."""
    url = f"https://github.com/{urllib.parse.quote(repo, safe = '/')}/releases/latest"
    request = urllib.request.Request(
        url,
        method = "HEAD",
        headers = {"User-Agent": ops.USER_AGENT},
    )
    try:
        with ops._URL_OPENER.open(request, timeout = 30) as response:
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


def fetch_download_host_json(ops: ModuleOps, url: str) -> Any:
    # Public CDN asset: plain unauthenticated GET, not the rate-limited API.
    data = ops.download_bytes(
        url,
        timeout = 30,
        headers = {"User-Agent": ops.USER_AGENT},
    )
    return json.loads(data.decode("utf-8"))


# ── Archive extraction (traversal/symlink guarded) ──
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
        """Repair a mangled symlink from some upstream llama.cpp Mac releases
        (e.g. b9165, b9169) whose linkname drops the separator AND the file
        basename's leading char between the top-level dir and the rest:

            llama-b9165/libggml-rpc.0.dylib -> llama-b9165ibggml-rpc.0.11.1.dylib

        Detect the pattern (linkname starts with the top-level dir but no
        following slash), then find the archive entry under that dir whose
        basename ends with the mangled suffix; only accept a unique match.
        Returns the corrected linkname relative to the member's parent dir --
        callers join it with `target.parent`, so a full `top/file` path would
        double the prefix into `top/top/file`."""
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
        # resolves inside the staging dir, not a duplicate `top/top/...` path.
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


def restore_tar_exec_bits(archive_path: Path, destination: Path) -> None:
    """Re-apply tar exec bits after the guarded extraction writes plain files;
    server binaries must stay executable on Unix."""
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


# ── Install lock ──
@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents = True, exist_ok = True)

    if FileLock is None:
        # Fallback lock: exclusive file creation, writing our PID so stale locks
        # from crashed processes can be detected.
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
                stale = False
                try:
                    raw = lock_path.read_text().strip()
                except FileNotFoundError:
                    # Lock vanished between our open and read -- retry
                    continue
                if not raw:
                    # Exists but PID not yet written; wait for the write to land.
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
                    stale = True  # PID unreadable (corrupted file)
                except ProcessLookupError:
                    stale = True  # holder is dead
                except PermissionError:
                    pass  # alive but owned by another user -- not stale
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


# ── macOS version parsing ──
def parse_macos_version(value: str | None) -> tuple[int, int] | None:
    """Parse a macOS product version into (major, minor).

    Handles "14.7.1", "15.5", "26.0", bare "26". None when empty/unparseable
    (callers then defer to runtime validation rather than reject every prebuilt)."""
    if not value:
        return None
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


# ── GPU and CUDA runtime-line selection primitives ──
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


def linux_runtime_dirs_for_required_libraries(
    ops: ModuleOps, required_libraries: Iterable[str]
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
    cuda_roots.extend(Path(path) for path in ops.glob_paths("/usr/local/cuda", "/usr/local/cuda-*"))

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
        for path in ops.glob_paths(
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
        Path(path) for path in ops.glob_paths("/usr/local/lib/ollama/cuda_v*", "/usr/lib/wsl/lib")
    )
    candidates.extend(Path(path) for path in ops.python_runtime_dirs())
    candidates.extend(Path(path) for path in ops.ldconfig_runtime_dirs(required))

    resolved = ops.dedupe_existing_dirs(candidates)
    if not required:
        return resolved

    matched: list[tuple[int, str]] = []
    for directory in resolved:
        provided = sum(1 for library in required if dir_provides_exact_library(directory, library))
        if provided:
            matched.append((provided, directory))

    matched.sort(key = lambda item: item[0], reverse = True)
    return [directory for _, directory in matched]


def detected_linux_runtime_lines(ops: ModuleOps) -> tuple[list[str], dict[str, list[str]]]:
    """`cuda<major>` lines with a matching libcudart/libcublas file on disk (glob
    match, so a versioned-only file counts), plus the dirs that matched."""
    line_requirements = {
        f"cuda{m}": [f"libcudart.so.{m}", f"libcublas.so.{m}"]
        for m in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1)
    }
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for line, required in line_requirements.items():
        dirs = ops.linux_runtime_dirs_for_required_libraries(required)
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


def windows_runtime_line_info() -> dict[str, tuple[str, ...]]:
    # Generated per CUDA major (newest first) so a new toolkit is detected without
    # a code change while the cudart64_<major>.dll naming holds.
    return {
        f"cuda{m}": (
            f"cudart64_{m}*.dll",
            f"cublas64_{m}*.dll",
            f"cublasLt64_{m}*.dll",
        )
        for m in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1)
    }


def detected_windows_runtime_lines(ops: ModuleOps) -> tuple[list[str], dict[str, list[str]]]:
    dirs = ops.windows_runtime_dirs()
    detected: list[str] = []
    runtime_dirs: dict[str, list[str]] = {}
    for runtime_line, required_patterns in windows_runtime_line_info().items():
        matching_dirs = ops.windows_runtime_dirs_for_patterns(required_patterns, dirs)
        if matching_dirs:
            detected.append(runtime_line)
            runtime_dirs[runtime_line] = matching_dirs
    return detected, runtime_dirs


def compatible_linux_runtime_lines(host: Any) -> list[str]:
    if not host.driver_cuda_version:
        return []
    major, _minor = host.driver_cuda_version
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


@dataclass
class CudaRuntimePreference:
    runtime_line: str | None
    selection_log: list[str]


def detect_torch_cuda_runtime_preference(host: Any) -> CudaRuntimePreference:
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


def artifact_covers_sms(artifact: Any, host_sms: Iterable[str]) -> bool:
    """True when every host SM is in the artifact's supported_sms and within its
    [min_sm, max_sm] range."""
    if not artifact.supported_sms or artifact.min_sm is None or artifact.max_sm is None:
        return False
    supported = {str(value) for value in artifact.supported_sms}
    return all(sm in supported and artifact.min_sm <= int(sm) <= artifact.max_sm for sm in host_sms)


def sm_range(artifact: Any) -> int:
    """SM-coverage span as a sort key (tighter range wins). A bundle with no SM
    metadata gets a max range so it sorts last, never outranking a targeted bundle."""
    if artifact.min_sm is not None and artifact.max_sm is not None:
        return artifact.max_sm - artifact.min_sm
    return 9999


def blackwell_capable_linux_runtime_lines(host_sms: list[str], artifacts: list[Any]) -> list[str]:
    """CUDA runtime lines (highest major first) shipping a bundle covering every
    visible host SM. Lets a Blackwell host prefer a native sm_120 line over torch's
    reported line, mirroring the Windows Blackwell preference."""
    lines: set[str] = set()
    for artifact in artifacts:
        line = artifact.runtime_line
        # Only rank "cuda<major>" lines; skip malformed/future-format values
        # (e.g. "cuda13.1") rather than crash the major sort.
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


def host_is_blackwell(host: Any) -> bool:
    caps = normalize_compute_caps(host.compute_caps)
    return bool(caps) and int(caps[-1]) >= _BLACKWELL_MIN_SM


def blackwell_min_toolkit_for_host(host: Any) -> tuple[int, int]:
    """Minimum CUDA toolkit this Blackwell host needs: 12.8 for the family, 12.9
    if any SM is sm_103/sm_121 (no native target before 12.9)."""
    req = _BLACKWELL_MIN_TOOLKIT
    for sm in normalize_compute_caps(host.compute_caps):
        req = max(req, _BLACKWELL_SM_MIN_TOOLKIT.get(int(sm), _BLACKWELL_MIN_TOOLKIT))
    return req


# ════════════════════════════════════════════════════════════════════════════
# Generic descriptor-driven install flow (whisper dialect: one release carries
# a manifest of os/arch/backend artifacts plus a same-origin checksum index).
# ════════════════════════════════════════════════════════════════════════════
def host_platform_tokens(host: Any) -> tuple[str, str]:
    """Default (os, arch) asset tokens; components with their own HostInfo field
    names override this hook."""
    return host.os_token, host.arch_token


# ── Manifest parsing ──
def parse_manifest(ops: ModuleOps, payload: Any, *, label: str) -> dict[str, Any]:
    """Validate a component prebuilt manifest and return it normalized.

    Rejects an unknown schema_version or wrong component. Returns keys:
    schema_version, component, studio_protocol, upstream_tag, source_commit,
    artifacts (list of dicts).
    """
    component = ops.COMPONENT
    if not isinstance(payload, dict):
        raise PrebuiltFallback(f"{label} was not a JSON object")
    validate_schema_version(
        payload, label = label, schema_version = ops.SCHEMA_VERSION, error = PrebuiltFallback
    )
    manifest_component = payload.get("component")
    if manifest_component != component:
        raise PrebuiltFallback(
            f"{label} describes component {manifest_component!r}, expected {component!r}"
        )

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise PrebuiltFallback(f"{label} omitted an 'artifacts' list")
    artifacts: list[dict[str, Any]] = []
    for index, raw in enumerate(artifacts_raw):
        if not isinstance(raw, dict):
            ops.log(f"{label} artifact[{index}] ignored: not an object")
            continue
        asset = raw.get("asset")
        if not isinstance(asset, str) or not asset:
            ops.log(f"{label} artifact[{index}] ignored: missing asset name")
            continue
        artifacts.append(raw)

    studio_protocol = payload.get("studio_protocol")
    return {
        "schema_version": ops.SCHEMA_VERSION,
        "component": component,
        "studio_protocol": studio_protocol if isinstance(studio_protocol, str) else None,
        "upstream_tag": payload.get("upstream_tag")
        if isinstance(payload.get("upstream_tag"), str)
        else None,
        "source_commit": payload.get("source_commit")
        if isinstance(payload.get("source_commit"), str)
        else None,
        "artifacts": artifacts,
    }


def macos_min_os_ok(ops: ModuleOps, host: Any, min_os: Any) -> bool:
    """True if a macOS artifact requiring `min_os` can load here. The manifest
    labels it `macos-<version>` (e.g. `macos-14.0`); strip that prefix before
    parsing or every entry parses as None and the guard no-ops. Unknown host or
    min_os -> True (defer to runtime validation)."""
    if not isinstance(min_os, str) or not min_os.strip():
        return True
    raw = min_os.strip()
    if raw.lower().startswith("macos-"):
        raw = raw[len("macos-") :]
    required = ops.parse_macos_version(raw)
    if required is None or host.macos_version is None:
        return True
    return host.macos_version >= required


def artifacts_for_host(
    ops: ModuleOps, manifest: dict[str, Any], host: Any, backend: str
) -> list[dict[str, Any]]:
    """Manifest artifacts matching this host os/arch/backend. On macOS, drop any
    whose `min_os` exceeds the host version."""
    os_token, arch_token = ops.host_platform_tokens(host)
    return [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("os") == os_token
        and artifact.get("arch") == arch_token
        and artifact.get("backend") == backend
        and (not host.is_macos or ops.macos_min_os_ok(host, artifact.get("min_os")))
    ]


def select_artifact(
    ops: ModuleOps, manifest: dict[str, Any], host: Any, backend: str
) -> dict[str, Any] | None:
    """First manifest artifact matching this host os/arch and backend, or None
    (caller then applies the component's fallback policy). No accelerator
    capability matching here: whisper bundles are slim per os/arch (the paired
    llama.cpp installer already picked SM/gfx-appropriate ggml backends), and
    llama keeps its own selection chain in install_llama_prebuilt.py."""
    candidates = ops.artifacts_for_host(manifest, host, backend)
    return candidates[0] if candidates else None


def select_artifact_with_fallback(
    ops: ModuleOps, manifest: dict[str, Any], host: Any, backend: str
) -> tuple[dict[str, Any], str, bool]:
    """Select the backend artifact, else the descriptor's fallback-backend
    artifact of the same release (whisper: CPU; llama: none, so a GPU miss
    surfaces as "no prebuilt" and the caller does a source build).

    Returns (artifact, effective_backend, used_fallback). Raises PrebuiltFallback
    when neither the requested backend nor the fallback has an asset."""
    os_token, arch_token = ops.host_platform_tokens(host)
    artifact = ops.select_artifact(manifest, host, backend)
    if artifact is not None:
        return artifact, backend, False
    fallback_backend = ops.FALLBACK_BACKEND
    if fallback_backend and backend != fallback_backend:
        fallback_artifact = ops.select_artifact(manifest, host, fallback_backend)
        if fallback_artifact is not None:
            ops.log(
                f"no '{backend}' asset for {os_token}-{arch_token}; "
                f"falling back to the {fallback_backend.upper()} asset of the same release"
            )
            return fallback_artifact, fallback_backend, True
    raise PrebuiltFallback(
        f"no {ops.COMPONENT} prebuilt asset for {os_token}-{arch_token} (backend '{backend}')"
    )


def artifact_coverage(artifact: dict[str, Any]) -> dict[str, Any]:
    """The sm/gfx/min_os coverage recorded for an artifact (marker/fingerprint)."""
    coverage: dict[str, Any] = {}
    for key in ("sm_coverage", "gfx_coverage", "min_os", "sm", "gfx"):
        if key in artifact and artifact.get(key) is not None:
            coverage[key] = artifact.get(key)
    return coverage


# ── Backend resolution ──
def auto_detect_backend(ops: ModuleOps, host: Any) -> str:
    """Host-preferred backend: Apple Silicon -> metal, usable NVIDIA -> cuda,
    AMD gfx -> rocm, otherwise cpu."""
    if host.is_apple_silicon:
        return "metal"
    if host.has_usable_nvidia:
        return "cuda"
    if host.has_rocm:
        return "rocm"
    return "cpu"


def resolve_backend(ops: ModuleOps, host: Any, requested: str | None, *, cpu_fallback: bool) -> str:
    """Resolve the effective backend from --backend / --cpu-fallback and detection."""
    if cpu_fallback:
        return "cpu"
    value = (requested or "auto").strip().lower()
    if value in {"", "auto"}:
        return ops.auto_detect_backend(host)
    if value in ops.SUPPORTED_BACKENDS:
        return value
    raise PrebuiltFallback(
        f"unsupported --backend '{requested}'; choose from auto,{','.join(ops.SUPPORTED_BACKENDS)}"
    )


# ── Release checksum index (trust anchor: the release's own sha256 asset) ──
def valid_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
        return digest
    return None


def parse_release_checksums(
    ops: ModuleOps, repo: str, release_tag: str, payload: Any
) -> dict[str, str]:
    """Asset name -> sha256 from a release's checksum-index asset.

    That index is the authority for each asset's sha256. It is validated for
    schema/component and that its ``release_tag`` matches the resolved release,
    so a redirected or mismatched index is rejected; malformed fails closed."""
    label = f"{ops.SHA256_ASSET_NAME} in {repo}@{release_tag}"
    if not isinstance(payload, dict):
        raise PrebuiltFallback(f"{label} was not a JSON object")
    if payload.get("schema_version") != ops.SCHEMA_VERSION:
        raise PrebuiltFallback(f"{label} has an unexpected schema_version")
    if payload.get("component") != ops.COMPONENT:
        raise PrebuiltFallback(f"{label} did not describe {ops.COMPONENT}")
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
        digest = valid_sha256(entry.get("sha256"))
        if digest is not None:
            checksums[asset_name] = digest
    if not checksums:
        raise PrebuiltFallback(f"{label} carried no usable sha256 entries")
    return checksums


def fetch_release_checksums(ops: ModuleOps, bundle: "ReleaseBundle") -> dict[str, str]:
    """Download + parse the release's checksum-index asset; fails closed if the
    release does not publish it."""
    sha_asset = ops.SHA256_ASSET_NAME
    url = bundle.asset_urls.get(sha_asset)
    if not url:
        raise PrebuiltFallback(
            f"release {bundle.repo}@{bundle.release_tag} has no {sha_asset}; "
            f"cannot verify a download"
        )
    try:
        raw = ops.download_bytes(url, timeout = 30, headers = ops.auth_headers(url))
        payload = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(
            f"could not fetch {sha_asset} from {bundle.repo}@{bundle.release_tag}: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltFallback(
            f"{sha_asset} in {bundle.repo}@{bundle.release_tag} was not valid JSON"
        ) from exc
    return ops.parse_release_checksums(bundle.repo, bundle.release_tag, payload)


def expected_sha256_for(
    ops: ModuleOps,
    checksums: dict[str, str],
    asset_name: str,
    *,
    manifest_sha256: str | None = None,
) -> str:
    """The sha256 the archive must match, from the release checksum-index entry.
    An asset absent from the index fails closed. Any sha256 the manifest embeds
    for the asset must agree with the index (a mismatch means a tampered manifest)."""
    digest = checksums.get(asset_name)
    if digest is None:
        raise PrebuiltFallback(
            f"{asset_name} is not covered by {ops.SHA256_ASSET_NAME}; "
            f"refusing an unverifiable download"
        )
    embedded = valid_sha256(manifest_sha256)
    if embedded is not None and embedded != digest:
        raise PrebuiltFallback(
            f"manifest sha256 for {asset_name} disagrees with {ops.SHA256_ASSET_NAME}; "
            f"refusing a possibly tampered release"
        )
    ops.log(f"verifying {asset_name} against {ops.SHA256_ASSET_NAME} sha256={digest}")
    return digest


# ── Release resolution ──
@dataclass
class ReleaseBundle:
    repo: str
    release_tag: str
    manifest: dict[str, Any]
    asset_urls: dict[str, str]


def fetch_release_bundle(ops: ModuleOps, repo: str, release_tag: str) -> ReleaseBundle:
    """Fetch a fork release, download+validate its manifest, return the bundle.

    The single network seam for API-path release resolution; tests inject a fake
    to exercise selection/install offline. A missing or unreachable release (404,
    rate limit, no network) surfaces as PrebuiltFallback so the caller does a
    source build rather than error out.
    """
    manifest_asset = ops.MANIFEST_ASSET_NAME
    try:
        release = ops.github_release(repo, release_tag)
    except PrebuiltFallback:
        raise
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(f"could not fetch release {repo}@{release_tag}: {exc}") from exc
    resolved_tag = release.get("tag_name")
    resolved_tag = resolved_tag if isinstance(resolved_tag, str) and resolved_tag else release_tag
    asset_urls = ops.release_asset_map(release)
    manifest_url = asset_urls.get(manifest_asset)
    if not manifest_url:
        raise PrebuiltFallback(
            f"release {repo}@{resolved_tag} has no {manifest_asset}; cannot select a prebuilt"
        )
    try:
        manifest_bytes = ops.download_bytes(
            manifest_url, timeout = 30, headers = ops.auth_headers(manifest_url)
        )
    except (urllib.error.URLError, OSError, socket.timeout) as exc:
        raise PrebuiltFallback(
            f"could not fetch {manifest_asset} from {repo}@{resolved_tag}: {exc}"
        ) from exc
    try:
        manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltFallback(
            f"{manifest_asset} in {repo}@{resolved_tag} was not valid JSON"
        ) from exc
    manifest = ops.parse_manifest(
        manifest_payload, label = f"{manifest_asset} in {repo}@{resolved_tag}"
    )
    return ReleaseBundle(
        repo = repo, release_tag = resolved_tag, manifest = manifest, asset_urls = asset_urls
    )


def asset_download_url(ops: ModuleOps, bundle: ReleaseBundle, asset_name: str) -> str:
    url = bundle.asset_urls.get(asset_name)
    if url:
        return url
    # A manifest can list an asset the release JSON omitted; fall back to the
    # deterministic release-download URL.
    return ops.release_asset_download_url(bundle.repo, bundle.release_tag, asset_name)


def resolve_newest_release_tag(ops: ModuleOps, repo: str) -> str:
    """Newest published (non-draft/non-prerelease) release tag for `repo` by
    ``published_at`` -- what the freshness checks use, NOT GitHub's
    ``/releases/latest`` pointer (sorts by commit date, can lag the newest build)."""
    payload = ops.fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=30")
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


def resolve_release_tag(
    ops: ModuleOps, published_repo: str, *, published_release_tag: str | None
) -> str:
    """The release tag to install: an explicit override, else the newest
    published release resolved at runtime."""
    override = (published_release_tag or "").strip()
    if override:
        return override
    return ops.resolve_newest_release_tag(published_repo)


def resolve_release_via_download_host(
    ops: ModuleOps, repo: str, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]] | None:
    """Resolve the release + manifest + checksum index entirely from the download
    host, with zero api.github.com calls. None (caller falls back to the API) on a
    missing/renamed asset, a 404, or a tag mismatch. Fetches the checksum index
    first (an in-progress release can publish it before the manifest)."""
    manifest_asset = ops.MANIFEST_ASSET_NAME
    sha_asset = ops.SHA256_ASSET_NAME
    release_tag = (published_release_tag or "").strip() or ops._download_host_latest_release_tag(
        repo
    )
    if not release_tag:
        return None
    sha_url = ops.release_asset_download_url(repo, release_tag, sha_asset)
    try:
        sha_payload = ops._download_host_json(sha_url)
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
        checksums = ops.parse_release_checksums(repo, release_tag, sha_payload)
    except PrebuiltFallback:
        return None  # schema/component/tag mismatch -> let the API path decide
    manifest_url = ops.release_asset_download_url(repo, release_tag, manifest_asset)
    try:
        manifest_payload = ops._download_host_json(manifest_url)
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
        manifest = ops.parse_manifest(
            manifest_payload, label = f"{manifest_asset} in {repo}@{release_tag}"
        )
    except PrebuiltFallback:
        return None
    # Tag-pinned CDN URLs for every asset the install may fetch (asset_download_url
    # reconstructs any missing one), keeping this a pure download-host path.
    names = {
        str(a.get("asset"))
        for a in manifest.get("artifacts", [])
        if isinstance(a, dict) and a.get("asset")
    }
    names |= {manifest_asset, sha_asset}
    asset_urls = {name: ops.release_asset_download_url(repo, release_tag, name) for name in names}
    bundle = ReleaseBundle(
        repo = repo, release_tag = release_tag, manifest = manifest, asset_urls = asset_urls
    )
    return bundle, checksums


def fetch_release_for_install(
    ops: ModuleOps, repo: str, *, published_release_tag: str | None
) -> tuple[ReleaseBundle, dict[str, str]]:
    """Resolve the release + manifest + checksum index, preferring the download
    host (no api.github.com rate limit) and falling back to the GitHub API. The
    single network seam the install/probe paths use."""
    fast = ops._resolve_release_via_download_host(repo, published_release_tag)
    if fast is not None:
        bundle, checksums = fast
        ops.log(f"resolved {repo}@{bundle.release_tag} via the download host (no GitHub API)")
        return bundle, checksums
    release_tag = ops.resolve_release_tag(repo, published_release_tag = published_release_tag)
    bundle = ops.fetch_release_bundle(repo, release_tag)
    checksums = ops.fetch_release_checksums(bundle)
    return bundle, checksums


# ── Marker and fingerprint ──
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
    # Slim pairing identity (whisper slim bundles ride the llama ggml runtime);
    # all None for fat installs, never part of the fingerprint.
    install_kind: str | None = None
    paired_llama_tag: str | None = None
    linked_from: str | None = None
    # Filenames the slim wiring hardlinked beside the server; the sidecar launch
    # guard verifies exactly these instead of hardcoded per-OS names.
    linked_libraries: tuple[str, ...] | None = None
    runtime_wiring_version: int | None = None
    linked_runtime_directories: tuple[str, ...] | None = None

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
    ops: ModuleOps,
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
        coverage = ops.artifact_coverage(artifact),
        studio_protocol = manifest.get("studio_protocol"),
    )


def metadata_path(ops: ModuleOps, install_dir: Path) -> Path:
    return install_dir / ops.METADATA_FILENAME


def write_prebuilt_metadata(ops: ModuleOps, install_dir: Path, selection: InstallSelection) -> None:
    coverage = selection.coverage
    payload = {
        "schema_version": ops.SCHEMA_VERSION,
        "component": ops.COMPONENT,
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
    if selection.install_kind == "slim":
        # Additive slim fields; fat markers keep the legacy payload exactly.
        payload["install_kind"] = "slim"
        payload["paired_llama_tag"] = selection.paired_llama_tag
        payload["linked_from"] = selection.linked_from
        if selection.linked_libraries is not None:
            payload["linked_libraries"] = list(selection.linked_libraries)
        if selection.runtime_wiring_version is not None:
            payload["runtime_wiring_version"] = selection.runtime_wiring_version
        if selection.linked_runtime_directories is not None:
            payload["linked_runtime_directories"] = list(selection.linked_runtime_directories)
    ops.metadata_path(install_dir).write_text(
        json.dumps(payload, indent = 2) + "\n", encoding = "utf-8"
    )


def load_prebuilt_metadata(ops: ModuleOps, install_dir: Path) -> dict[str, Any] | None:
    path = ops.metadata_path(install_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def existing_install_matches(
    ops: ModuleOps, install_dir: Path, host: Any, selection: InstallSelection
) -> bool:
    """True iff the marker records this exact selection and the server binary is on disk."""
    metadata = ops.load_prebuilt_metadata(install_dir)
    if metadata is None:
        return False
    if not ops.installed_server_path(install_dir, host).is_file():
        return False
    recorded = metadata.get("install_fingerprint")
    if not isinstance(recorded, str) or recorded != selection.fingerprint():
        return False
    ops.log(
        f"existing {ops.COMPONENT} install already matches {selection.release_tag} "
        f"({selection.backend}); nothing to do"
    )
    return True


# ── Install tree assembly and staged activation ──
def locate_server_in_tree(ops: ModuleOps, root: Path, host: Any) -> Path:
    name = ops.server_binary_name(host)
    matches = sorted(root.rglob(name))
    if not matches:
        raise PrebuiltFallback(f"archive did not contain a {name} binary")
    return matches[0]


def assemble_install_tree(ops: ModuleOps, bundle_root: Path, staged_root: Path, host: Any) -> Path:
    """Lay out staged_root as a full install: <runtime bin dir>/<server + libs>.

    Everything beside the server in the archive (shared libs, backend kernel
    subdirs, license/build-info) is co-located into the canonical bin dir so the
    server's RUNPATH=$ORIGIN resolves its libs.
    """
    bin_dir = ops.runtime_bin_dir(staged_root, host)
    bin_dir.mkdir(parents = True, exist_ok = True)
    for entry in sorted(bundle_root.iterdir()):
        dest = bin_dir / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, dest, symlinks = True)
        else:
            shutil.copy2(entry, dest, follow_symlinks = False)
    server = bin_dir / ops.server_binary_name(host)
    if not server.exists():
        raise PrebuiltFallback(f"staged install is missing the {ops.COMPONENT} server binary")
    if not host.is_windows:
        os.chmod(server, server.stat().st_mode | 0o111)
    return server


def swap_into_place(staged_root: Path, install_dir: Path) -> None:
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


def validate_staged_server(ops: ModuleOps, staged_root: Path, host: Any) -> None:
    """Optional pre-activate smoke test, gated off by default (the component's
    _RUN_STAGED_PREBUILT_VALIDATION switch)."""
    if not ops._RUN_STAGED_PREBUILT_VALIDATION:
        return
    server = ops.installed_server_path(staged_root, host)
    env = os.environ.copy()
    bin_dir = str(ops.runtime_bin_dir(staged_root, host))
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = bin_dir + (os.pathsep + env[var] if env.get(var) else "")
    try:
        result = subprocess.run(
            [str(server), "--help"],
            capture_output = True,
            text = True,
            timeout = 60,
            env = env,
            **ops.windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrebuiltFallback(f"staged {ops.COMPONENT} server failed to launch: {exc}") from exc
    if result.returncode != 0:
        raise PrebuiltFallback(
            f"staged {ops.COMPONENT} server --help exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )


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


def install_from_bundle(
    ops: ModuleOps, install_dir: Path, host: Any, bundle: ReleaseBundle, selection: InstallSelection
) -> None:
    staging_root = install_dir.parent / ops.INSTALL_STAGING_ROOT_NAME
    staging_root.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = f"{install_dir.name}.staging-", dir = staging_root))
    try:
        archive_path = staging / selection.asset
        url = ops.asset_download_url(bundle, selection.asset)
        ops.log(f"downloading {url}")
        ops.download_file_verified(
            url, archive_path, expected_sha256 = selection.asset_sha256, label = selection.asset
        )
        extract_dir = staging / "extracted"
        ops.extract_archive(archive_path, extract_dir)

        server = ops.locate_server_in_tree(extract_dir, host)
        bundle_root = server.parent

        staged_root = staging / "staged"
        ops.assemble_install_tree(bundle_root, staged_root, host)
        # Component hook (default no-op): slim whisper wires the llama ggml
        # runtime here so staged validation sees the final linked tree; a returned
        # selection (with the wired filenames) supersedes the input so the marker
        # records what was actually linked.
        updated = ops.prepare_runtime_payload(staged_root, host, selection)
        if updated is not None:
            selection = updated
        ops.validate_staged_server(staged_root, host)
        ops.write_prebuilt_metadata(staged_root, selection)
        ops._swap_into_place(staged_root, install_dir)
    finally:
        shutil.rmtree(staging, ignore_errors = True)
        try:
            staging_root.rmdir()
        except OSError:
            pass


def plan_selection(
    ops: ModuleOps,
    host: Any,
    bundle: ReleaseBundle,
    *,
    published_repo: str,
    backend: str,
    checksums: dict[str, str],
) -> InstallSelection:
    """Choose an artifact (with the component's fallback policy) and resolve its
    trusted sha256 from the release checksum index."""
    artifact, effective_backend, _used_fallback = ops.select_artifact_with_fallback(
        bundle.manifest, host, backend
    )
    asset = str(artifact.get("asset"))
    manifest_sha256 = artifact.get("sha256") if isinstance(artifact.get("sha256"), str) else None
    expected_sha = ops.expected_sha256_for(
        checksums,
        asset,
        manifest_sha256 = manifest_sha256,
    )
    return ops.selection_from_artifact(
        published_repo = published_repo,
        release_tag = bundle.release_tag,
        manifest = bundle.manifest,
        artifact = artifact,
        backend = effective_backend,
        asset_sha256 = expected_sha,
    )


def install_prebuilt(
    ops: ModuleOps,
    install_dir: Path,
    *,
    published_repo: str,
    published_release_tag: str | None = None,
    backend: str | None = "auto",
    cpu_fallback: bool = False,
    force: bool = False,
    host: Any = None,
) -> int:
    if host is None:
        host = ops.detect_host()
    effective_backend = ops.resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    os_token, arch_token = ops.host_platform_tokens(host)
    ops.log(
        f"target {ops.COMPONENT} from {published_repo} "
        f"({os_token}-{arch_token}, backend {effective_backend})"
    )

    bundle, checksums = ops.fetch_release_for_install(
        published_repo, published_release_tag = published_release_tag
    )
    selection = ops.plan_selection(
        host,
        bundle,
        published_repo = published_repo,
        backend = effective_backend,
        checksums = checksums,
    )

    return install_selected_prebuilt(
        ops,
        install_dir,
        host = host,
        bundle = bundle,
        selection = selection,
        force = force,
    )


def install_selected_prebuilt(
    ops: ModuleOps,
    install_dir: Path,
    *,
    host: Any,
    bundle: ReleaseBundle,
    selection: InstallSelection,
    force: bool,
) -> int:
    """Validate and activate an already selected release plan.

    Components that need to examine more than one published release can keep
    release selection component-specific while sharing the lock, idempotency,
    atomic activation, and post-install verification path.
    """

    if not force and ops.existing_install_matches(install_dir, host, selection):
        return 0

    with ops.install_lock(ops.install_lock_path(install_dir)):
        # Re-check under the lock: a concurrent run may have just finished.
        if not force and ops.existing_install_matches(install_dir, host, selection):
            return 0
        ops._install_from_bundle(install_dir, host, bundle, selection)

    server = ops.installed_server_path(install_dir, host)
    if not server.is_file():
        raise PrebuiltFallback(f"post-install verification failed: {server} is missing")
    ops.log(
        f"installed {ops.COMPONENT} {bundle.release_tag} " f"({selection.backend}) at {install_dir}"
    )
    return 0


def resolve_prebuilt(
    ops: ModuleOps,
    host: Any,
    *,
    published_repo: str,
    published_release_tag: str | None,
    backend: str | None,
    cpu_fallback: bool,
) -> dict[str, Any]:
    """Host-aware "is a prebuilt available" probe. No archive download."""
    effective_backend = ops.resolve_backend(host, backend, cpu_fallback = cpu_fallback)
    try:
        bundle, _checksums = ops.fetch_release_for_install(
            published_repo, published_release_tag = published_release_tag
        )
        artifact, resolved_backend, used_fallback = ops.select_artifact_with_fallback(
            bundle.manifest, host, effective_backend
        )
    except PrebuiltFallback:
        return {"prebuilt_available": False, "repo": published_repo}
    os_token, arch_token = ops.host_platform_tokens(host)
    payload = {
        "prebuilt_available": True,
        "repo": published_repo,
        "release_tag": bundle.release_tag,
        "upstream_tag": bundle.manifest.get("upstream_tag"),
        "backend": resolved_backend,
        "requested_backend": effective_backend,
        "cpu_fallback": used_fallback,
        "asset": str(artifact.get("asset")),
        "os": os_token,
        "arch": arch_token,
        "runtime_line": artifact.get("runtime_line"),
    }
    # Component hook (default none): whisper adds install_kind slim|fat. The JSON
    # emitter sorts keys, so an appended field can't perturb the legacy key order.
    payload.update(ops.resolver_payload_extra(artifact))
    return payload


def prepare_runtime_payload(staged_root: Path, host: Any, selection: InstallSelection) -> Any:
    """Post-assemble hook; the core stages nothing extra. Components override it
    to wire external runtime files into the staged bin dir (whisper slim hardlinks
    the llama install's ggml libraries) before validation, and may return an
    updated InstallSelection for the marker to record instead."""
    return None


def resolver_payload_extra(artifact: dict[str, Any]) -> dict[str, Any]:
    """Additive --resolve-prebuilt payload fields; the core adds none."""
    return {}


def installed_server_path(ops: ModuleOps, install_dir: Path, host: Any) -> Path:
    return ops.runtime_bin_dir(install_dir, host) / ops.server_binary_name(host)


# Underscored aliases so ops lookups that mirror the installer modules' private
# names still resolve for a descriptor-only component.
_download_host_latest_release_tag = download_host_latest_release_tag
_download_host_json = fetch_download_host_json
_resolve_release_via_download_host = resolve_release_via_download_host
_install_from_bundle = install_from_bundle
_swap_into_place = swap_into_place


def emit_resolver_output(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, sort_keys = True))
        return
    if "asset" in payload and payload.get("prebuilt_available"):
        print(payload["asset"])
        return
    print(json.dumps(payload, sort_keys = True))
