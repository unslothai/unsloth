# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import ModuleType

from filelock import FileLock, Timeout

from utils.native_path_leases import child_env_without_native_path_secret
from utils.paths.storage_roots import cache_root
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)


@dataclass(frozen = True)
class PinnedSource:
    name: str
    package: str
    repository: str
    revision: str
    required_files: tuple[str, ...]
    omitted_files: tuple[str, ...] = ()
    generated_files: tuple[tuple[str, str], ...] = ()
    source_tree_digest: str | None = None
    runtime_tree_digest: str | None = None
    archive_url: str | None = None


SPARK_TTS_SOURCE = PinnedSource(
    name = "Spark-TTS",
    package = "sparktts",
    repository = "https://github.com/SparkAudio/Spark-TTS",
    revision = "2f1ea9082400547242641f5271b6f941c9f439d1",
    required_files = (
        "sparktts/models/audio_tokenizer.py",
        "sparktts/utils/audio.py",
    ),
    generated_files = (("sparktts/__init__.py", ""),),
    source_tree_digest = "20ff9f4c9e380b89248b828e9f39ec14572c43ff4a8d87b76190dbb3214b1b27",
    runtime_tree_digest = "f14510e491a87ab287910e1d3f80e6b3d1bcea91b7f91f3baa45a3181a6993ba",
    archive_url = (
        "https://github.com/SparkAudio/Spark-TTS/archive/"
        "2f1ea9082400547242641f5271b6f941c9f439d1.tar.gz"
    ),
)

OUTETTS_SOURCE = PinnedSource(
    name = "OuteTTS",
    package = "outetts",
    repository = "https://github.com/edwko/OuteTTS",
    revision = "f5eac6e70d792844c6a6959d900a47af2c061a5b",
    required_files = (
        "outetts/models/config.py",
        "outetts/utils/preprocessing.py",
        "outetts/version/v3/audio_processor.py",
        "outetts/version/v3/prompt_processor.py",
    ),
    omitted_files = (
        "outetts/interface.py",
        "outetts/models/gguf_model.py",
    ),
    generated_files = (("outetts/__init__.py", ""),),
    source_tree_digest = "817299085cb018839d37bf43505c9a742188bdb0f6ead8e1ea19a8643f0bb49f",
    runtime_tree_digest = "b9f878aeb2de4d3ab0a5b1f75a5d04f2a137e4369143099bb24f6d6a41301fab",
    archive_url = (
        "https://github.com/edwko/OuteTTS/archive/"
        "f5eac6e70d792844c6a6959d900a47af2c061a5b.tar.gz"
    ),
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IMPORT_LOCK = threading.RLock()

_DAC_REPOSITORY = "ibm-research/DAC.speech.v1.0"
_DAC_REVISION = "1ea7f64cd0678415e2d8c32d67b190722cb9b149"
_DAC_FILENAME = "weights_24khz_1.5kbps_v1.0.pth"
_DAC_SIZE = 295731578
_DAC_SHA256 = "d77ca0b04df942ec64e6a7a162bcac093b1127700acdaec0079f40d32c4405fb"

_ARCHIVE_MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
_ARCHIVE_MAX_MEMBERS = 10_000
_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_ARCHIVE_MAX_TAR_BYTES = 160 * 1024 * 1024
_ARCHIVE_SOCKET_TIMEOUT_SECONDS = 15
_ARCHIVE_DOWNLOAD_DEADLINE_SECONDS = 300

# Git for Windows still enforces MAX_PATH (260) unless told otherwise, and the pinned cache
# nests a 40-char revision, a staging dir and .git/objects under the studio home; a
# venv-inferred home already reaches ~253 chars, so a slightly longer one fails the
# checkout with "Filename too long". Passed per-invocation so no user config is touched.
_GIT_LONG_PATHS = ["-c", "core.longpaths=true"]


def _git(arguments: list[str], *, source_name: str) -> subprocess.CompletedProcess:
    env = child_env_without_native_path_secret()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", *_GIT_LONG_PATHS, *arguments],
            check = True,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 300,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to install the pinned {source_name} source") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Timed out while installing the pinned {source_name} source") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        message = f"Could not install the pinned {source_name} source"
        raise RuntimeError(f"{message}: {detail}" if detail else message) from error


def _git_bytes(
    arguments: list[str], *, source_name: str, input_data: bytes
) -> subprocess.CompletedProcess:
    env = child_env_without_native_path_secret()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", *_GIT_LONG_PATHS, *arguments],
            check = True,
            capture_output = True,
            input = input_data,
            timeout = 300,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to install the pinned {source_name} source") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Timed out while installing the pinned {source_name} source") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors = "replace").strip()
        message = f"Could not install the pinned {source_name} source"
        raise RuntimeError(f"{message}: {detail}" if detail else message) from error


def _generated_cache_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return "/__pycache__/" in f"/{normalized}" and normalized.endswith((".pyc", ".pyo"))


def _package_path_parts(relative: str, spec: PinnedSource, *, kind: str) -> tuple[str, ...]:
    normalized = relative.replace("\\", "/")
    parts = tuple(normalized.split("/"))
    if (
        normalized != relative
        or not normalized
        or normalized.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or any(PureWindowsPath(part).drive for part in parts)
        or parts[0] != spec.package
    ):
        raise ValueError(f"Invalid {kind} path for {spec.name}: {relative}")
    return parts


def _configured_package_paths(
    relatives: tuple[str, ...], spec: PinnedSource, *, kind: str
) -> tuple[str, ...]:
    validated = []
    seen = set()
    for relative in relatives:
        _package_path_parts(relative, spec, kind = kind)
        if relative in seen:
            raise ValueError(f"Invalid {kind} path for {spec.name}: {relative}")
        seen.add(relative)
        validated.append(relative)
    return tuple(validated)


def _generated_file_contents(spec: PinnedSource) -> dict[str, bytes]:
    generated = {}
    for relative, content in spec.generated_files:
        _package_path_parts(relative, spec, kind = "generated")
        if relative in generated:
            raise ValueError(f"Invalid generated path for {spec.name}: {relative}")
        generated[relative] = content.encode("utf-8")
    return generated


def _tracked_package_blobs(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    output = _git(
        [
            "-C",
            str(checkout),
            "ls-tree",
            "-r",
            "-z",
            spec.revision,
            "--",
            spec.package,
        ],
        source_name = spec.name,
    ).stdout
    blobs = {}
    for record in (record for record in output.split("\0") if record):
        metadata, separator, relative = record.partition("\t")
        fields = metadata.split(" ")
        if separator != "\t" or len(fields) != 3:
            raise ValueError(f"Invalid tracked tree entry for {spec.name}")
        mode, object_type, object_id = fields
        _package_path_parts(relative, spec, kind = "tracked")
        if (
            mode not in ("100644", "100755")
            or object_type != "blob"
            or _REVISION_PATTERN.fullmatch(object_id) is None
            or relative in blobs
        ):
            raise ValueError(f"Invalid tracked tree entry for {spec.name}: {relative}")
        blobs[relative] = object_id
    return blobs


def _pinned_blob_digests(
    checkout: Path, object_ids: tuple[str, ...], spec: PinnedSource
) -> dict[str, str]:
    unique_object_ids = tuple(dict.fromkeys(object_ids))
    if not unique_object_ids:
        return {}
    result = _git_bytes(
        ["-C", str(checkout), "cat-file", "--batch"],
        source_name = spec.name,
        input_data = "".join(f"{object_id}\n" for object_id in unique_object_ids).encode("ascii"),
    ).stdout
    digests = {}
    offset = 0
    for expected_object_id in unique_object_ids:
        header_end = result.find(b"\n", offset)
        if header_end < 0:
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        fields = result[offset:header_end].split(b" ")
        if len(fields) != 3:
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        object_id, object_type, size_value = fields
        try:
            size = int(size_value)
        except ValueError as error:
            raise ValueError(f"Invalid pinned blob data for {spec.name}") from error
        content_start = header_end + 1
        content_end = content_start + size
        if (
            object_id.decode("ascii", errors = "replace") != expected_object_id
            or object_type != b"blob"
            or size < 0
            or content_end >= len(result)
            or result[content_end : content_end + 1] != b"\n"
        ):
            raise ValueError(f"Invalid pinned blob data for {spec.name}")
        digests[expected_object_id] = hashlib.sha256(result[content_start:content_end]).hexdigest()
        offset = content_end + 1
    if offset != len(result):
        raise ValueError(f"Invalid pinned blob data for {spec.name}")
    return digests


def _package_file(root: Path, relative: str, spec: PinnedSource) -> Path:
    parts = _package_path_parts(relative, spec, kind = "tracked")
    path = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed in {spec.name} source")
    if not path.is_file():
        raise ValueError(f"Missing tracked file in {spec.name} source: {relative}")
    return path


def _checkout_manifest(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    package_root = checkout / spec.package
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"Missing {spec.package} package")
    omitted = _configured_package_paths(spec.omitted_files, spec, kind = "omitted")
    excluded = set(omitted) | set(_generated_file_contents(spec))
    tracked_blobs = _tracked_package_blobs(checkout, spec)
    pinned_digests = _pinned_blob_digests(
        checkout,
        tuple(tracked_blobs.values()),
        spec,
    )
    manifest = {}
    for relative, object_id in tracked_blobs.items():
        path = _package_file(checkout, relative, spec)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pinned_digests[object_id]:
            raise ValueError(f"Tracked file does not match the pinned {spec.name} blob: {relative}")
        if relative not in excluded:
            manifest[relative] = digest
    return manifest


def _manifest_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(
        manifest,
        sort_keys = True,
        separators = (",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _filesystem_source_manifest(source: Path, spec: PinnedSource) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Missing {spec.name} source")
    package_root = source / spec.package
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"Missing {spec.package} package")
    excluded = set(_configured_package_paths(spec.omitted_files, spec, kind = "omitted")) | set(
        _generated_file_contents(spec)
    )
    manifest = {}
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in {spec.name} source")
        if path.is_dir() or _generated_cache_path(relative):
            continue
        if not path.is_file():
            raise ValueError(f"Special files are not allowed in {spec.name} source")
        if relative not in excluded:
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _sealed_source_manifest(source: Path, spec: PinnedSource) -> dict[str, str] | None:
    if spec.source_tree_digest is None:
        return None
    try:
        manifest = _filesystem_source_manifest(source, spec)
    except (OSError, ValueError):
        return None
    if _manifest_digest(manifest) != spec.source_tree_digest:
        return None
    return manifest


def _runtime_manifest(runtime: Path, spec: PinnedSource) -> dict[str, str]:
    package_root = runtime / spec.package
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"Missing {spec.package} package")
    manifest = {}
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(runtime).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in {spec.name} runtime source")
        if path.is_dir() or _generated_cache_path(relative):
            continue
        if not path.is_file():
            raise ValueError(f"Special files are not allowed in {spec.name} runtime source")
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _expected_runtime_manifest(checkout: Path, spec: PinnedSource) -> dict[str, str]:
    manifest = _checkout_manifest(checkout, spec)
    for relative, content in _generated_file_contents(spec).items():
        manifest[relative] = hashlib.sha256(content).hexdigest()
    return manifest


def _valid_checkout(path: Path, spec: PinnedSource) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        required_files = _configured_package_paths(
            spec.required_files,
            spec,
            kind = "required",
        )
        for relative in required_files:
            required = path.joinpath(*_package_path_parts(relative, spec, kind = "required"))
            if required.is_symlink() or not required.is_file():
                return False
        head = (
            _git(
                ["-C", str(path), "rev-parse", "HEAD"],
                source_name = spec.name,
            )
            .stdout.strip()
            .lower()
        )
        branch = _git(
            ["-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            source_name = spec.name,
        ).stdout.strip()
        origin = _git(
            ["-C", str(path), "remote", "get-url", "origin"],
            source_name = spec.name,
        ).stdout.strip()
        status = _git(
            ["-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            source_name = spec.name,
        ).stdout
        ignored = _git(
            ["-C", str(path), "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            source_name = spec.name,
        ).stdout
        _checkout_manifest(path, spec)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        head == spec.revision
        and branch == "HEAD"
        and origin.rstrip("/").removesuffix(".git")
        == spec.repository.rstrip("/").removesuffix(".git")
        and not status
        and not ignored
    )


def _clear_read_only(function, path, _error) -> None:
    # Git marks .git/objects read-only, and Windows refuses to delete a read-only file, so
    # replacing a checkout dies with WinError 5. Only retry when the path really is not
    # writable: an open handle (WinError 32) must still surface rather than spin.
    if os.access(path, os.W_OK):
        raise
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    function(path)


def _remove_owned_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok = True)
    elif path.is_dir():
        # onexc replaced onerror in 3.12; the handler signature is the same either way.
        handler = (
            {"onexc": _clear_read_only}
            if sys.version_info >= (3, 12)
            else {"onerror": _clear_read_only}
        )
        shutil.rmtree(path, **handler)


def _replace_owned_directory(staging: Path, destination: Path) -> None:
    displaced = None
    if destination.exists() or destination.is_symlink():
        displaced = destination.with_name(f".{destination.name}.invalid-{uuid.uuid4().hex}")
        os.replace(destination, displaced)
    try:
        os.replace(staging, destination)
    except Exception:
        if displaced is not None and not destination.exists():
            os.replace(displaced, destination)
            displaced = None
        raise
    finally:
        if displaced is not None:
            _remove_owned_path(displaced)


def _install_checkout(destination: Path, spec: PinnedSource) -> None:
    workspace = Path(tempfile.mkdtemp(prefix = ".source-", dir = destination.parent))
    checkout = workspace / "checkout"
    hooks = workspace / "hooks"
    hooks.mkdir()
    hook_config = f"core.hooksPath={hooks}"
    try:
        _git(["init", "--quiet", str(checkout)], source_name = spec.name)
        _git(
            ["-C", str(checkout), "config", "core.autocrlf", "false"],
            source_name = spec.name,
        )
        _git(
            ["-C", str(checkout), "remote", "add", "origin", spec.repository],
            source_name = spec.name,
        )
        _git(
            [
                "-c",
                hook_config,
                "-C",
                str(checkout),
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                "origin",
                spec.revision,
            ],
            source_name = spec.name,
        )
        fetched = (
            _git(
                ["-C", str(checkout), "rev-parse", "FETCH_HEAD^{commit}"],
                source_name = spec.name,
            )
            .stdout.strip()
            .lower()
        )
        if fetched != spec.revision:
            raise RuntimeError(f"{spec.name} returned a different revision than the pinned source")
        _git(
            [
                "-c",
                hook_config,
                "-C",
                str(checkout),
                "checkout",
                "--quiet",
                "--detach",
                spec.revision,
            ],
            source_name = spec.name,
        )
        if not _valid_checkout(checkout, spec):
            raise RuntimeError(f"The downloaded {spec.name} source failed integrity validation")
        _replace_owned_directory(checkout, destination)
    finally:
        _remove_owned_path(workspace)


def _archive_root_name(spec: PinnedSource) -> str:
    repository_name = spec.repository.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    if not repository_name:
        raise RuntimeError(f"Invalid pinned {spec.name} repository")
    return f"{repository_name}-{spec.revision}"


def _download_archive(url: str, destination: Path, spec: PinnedSource) -> None:
    request = urllib.request.Request(url, headers = {"User-Agent": "Unsloth-Studio"})
    deadline = time.monotonic() + _ARCHIVE_DOWNLOAD_DEADLINE_SECONDS
    try:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out downloading the pinned {spec.name} source archive")
        with urllib.request.urlopen(
            request,
            timeout = _ARCHIVE_SOCKET_TIMEOUT_SECONDS,
        ) as response:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out downloading the pinned {spec.name} source archive")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised_size = int(content_length)
                except ValueError as error:
                    raise RuntimeError(f"Invalid {spec.name} archive response size") from error
                if advertised_size < 0 or advertised_size > _ARCHIVE_MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(f"The pinned {spec.name} archive is too large")
            total = 0
            read_chunk = getattr(response, "read1", None)
            if not callable(read_chunk):
                read_chunk = response.read
            with destination.open("wb") as handle:
                while True:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Timed out downloading the pinned {spec.name} source archive"
                        )
                    chunk = read_chunk(1024 * 1024)
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Timed out downloading the pinned {spec.name} source archive"
                        )
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _ARCHIVE_MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(f"The pinned {spec.name} archive is too large")
                    handle.write(chunk)
    except RuntimeError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"Could not download the pinned {spec.name} source archive") from error


def _archive_member_parts(member: tarfile.TarInfo, spec: PinnedSource) -> tuple[str, ...]:
    name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
    parts = tuple(name.split("/"))
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in parts)
        or any(PureWindowsPath(part).drive for part in parts)
        or parts[0] != _archive_root_name(spec)
    ):
        raise RuntimeError(f"Invalid path in the pinned {spec.name} source archive")
    return parts


class _BoundedArchiveReader:
    def __init__(self, handle, limit: int):
        self._handle = handle
        self._limit = limit
        self._read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._limit - self._read
        requested = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self._handle.read(requested)
        self._read += len(data)
        if self._read > self._limit:
            raise RuntimeError("The pinned source archive expands too large")
        return data


def _install_archive_source(destination: Path, spec: PinnedSource) -> None:
    if spec.archive_url is None or spec.source_tree_digest is None:
        raise RuntimeError(f"The pinned {spec.name} source archive is not configured")
    workspace = Path(tempfile.mkdtemp(prefix = ".archive-", dir = destination.parent))
    archive = workspace / "source.tar.gz"
    staging = workspace / "source"
    staging.mkdir()
    try:
        _download_archive(spec.archive_url, archive, spec)
        member_count = 0
        uncompressed_bytes = 0
        extracted = set()
        try:
            with archive.open("rb") as compressed:
                with gzip.GzipFile(fileobj = compressed, mode = "rb") as decompressed:
                    reader = _BoundedArchiveReader(decompressed, _ARCHIVE_MAX_TAR_BYTES)
                    with tarfile.open(fileobj = reader, mode = "r|") as bundle:
                        for member in bundle:
                            member_count += 1
                            if member_count > _ARCHIVE_MAX_MEMBERS:
                                raise RuntimeError(
                                    f"The pinned {spec.name} archive has too many entries"
                                )
                            parts = _archive_member_parts(member, spec)
                            if member.isdir():
                                continue
                            if not member.isfile() or member.size < 0:
                                raise RuntimeError(
                                    f"The pinned {spec.name} archive contains a non-regular file"
                                )
                            uncompressed_bytes += member.size
                            if uncompressed_bytes > _ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                                raise RuntimeError(
                                    f"The pinned {spec.name} archive expands too large"
                                )
                            if len(parts) < 3 or parts[1] != spec.package:
                                continue
                            relative = "/".join(parts[1:])
                            _package_path_parts(relative, spec, kind = "archive")
                            if relative in extracted:
                                raise RuntimeError(
                                    f"The pinned {spec.name} archive contains duplicate files"
                                )
                            extracted.add(relative)
                            source_file = bundle.extractfile(member)
                            if source_file is None:
                                raise RuntimeError(
                                    f"The pinned {spec.name} archive contains an unreadable file"
                                )
                            destination_file = staging.joinpath(*parts[1:])
                            destination_file.parent.mkdir(parents = True, exist_ok = True)
                            remaining = member.size
                            with source_file, destination_file.open("wb") as handle:
                                while remaining:
                                    chunk = source_file.read(min(1024 * 1024, remaining))
                                    if not chunk:
                                        raise RuntimeError(
                                            f"The pinned {spec.name} archive ended unexpectedly"
                                        )
                                    handle.write(chunk)
                                    remaining -= len(chunk)
        except (tarfile.TarError, EOFError, OSError) as error:
            raise RuntimeError(f"The pinned {spec.name} source archive is invalid") from error
        if _sealed_source_manifest(staging, spec) is None:
            raise RuntimeError(f"The pinned {spec.name} source archive failed integrity validation")
        _replace_owned_directory(staging, destination)
    finally:
        _remove_owned_path(workspace)


def _valid_runtime(
    runtime: Path,
    spec: PinnedSource,
    checkout: Path | None = None,
) -> bool:
    if runtime.is_symlink() or not runtime.is_dir():
        return False
    try:
        required_files = _configured_package_paths(
            spec.required_files,
            spec,
            kind = "required",
        )
        omitted_files = _configured_package_paths(
            spec.omitted_files,
            spec,
            kind = "omitted",
        )
        top_level = {path.name for path in runtime.iterdir() if path.name != "__pycache__"}
        if top_level != {spec.package}:
            return False
        for relative in required_files:
            required = runtime.joinpath(*_package_path_parts(relative, spec, kind = "required"))
            if required.is_symlink() or not required.is_file():
                return False
        for relative in omitted_files:
            omitted = runtime.joinpath(*_package_path_parts(relative, spec, kind = "omitted"))
            if omitted.exists() or omitted.is_symlink():
                return False
        for relative, content in _generated_file_contents(spec).items():
            generated = runtime / relative
            if generated.is_symlink() or not generated.is_file():
                return False
            if generated.read_bytes() != content:
                return False
        manifest = _runtime_manifest(runtime, spec)
        if spec.runtime_tree_digest is not None:
            return _manifest_digest(manifest) == spec.runtime_tree_digest
        return checkout is not None and manifest == _expected_runtime_manifest(checkout, spec)
    except (OSError, RuntimeError, ValueError):
        return False


def _install_runtime(runtime: Path, checkout: Path, spec: PinnedSource) -> None:
    workspace = Path(tempfile.mkdtemp(prefix = ".runtime-", dir = runtime.parent))
    staging = workspace / "runtime"
    staging.mkdir()
    try:
        if spec.source_tree_digest is not None:
            source_manifest = _sealed_source_manifest(checkout, spec)
            if source_manifest is None:
                raise RuntimeError(f"The cached {spec.name} source failed integrity validation")
        else:
            source_manifest = _checkout_manifest(checkout, spec)
        for relative, expected_digest in source_manifest.items():
            source_file = checkout / relative
            destination_file = staging / relative
            destination_file.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(source_file, destination_file)
            if hashlib.sha256(destination_file.read_bytes()).hexdigest() != expected_digest:
                raise RuntimeError(f"{spec.name} source changed while preparing its runtime")
        for relative, content in _generated_file_contents(spec).items():
            destination_file = staging / relative
            destination_file.parent.mkdir(parents = True, exist_ok = True)
            destination_file.write_bytes(content)
        if not _valid_runtime(staging, spec, checkout):
            raise RuntimeError(f"The prepared {spec.name} runtime failed integrity validation")
        _replace_owned_directory(staging, runtime)
    finally:
        _remove_owned_path(workspace)


def ensure_pinned_source(
    spec: PinnedSource, *, legacy_sources: tuple[Path | str, ...] = ()
) -> Path:
    revision = spec.revision.lower()
    if _REVISION_PATTERN.fullmatch(revision) is None or revision != spec.revision:
        raise RuntimeError(f"{spec.name} source revision must be a lowercase full Git commit")
    for digest in (spec.source_tree_digest, spec.runtime_tree_digest):
        if digest is not None and _SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(f"{spec.name} source digest must be a lowercase SHA-256")
    if (spec.source_tree_digest is None) != (spec.runtime_tree_digest is None):
        raise RuntimeError(f"{spec.name} source and runtime digests must be configured together")

    parent = cache_root() / "third-party-sources" / spec.name
    version_root = parent / revision
    checkout = version_root / "source"
    runtime = version_root / "runtime-v1"
    if _valid_runtime(runtime, spec):
        return runtime.resolve()

    version_root.mkdir(parents = True, exist_ok = True)
    try:
        with FileLock(str(parent / ".install.lock"), timeout = 300):
            if _valid_runtime(runtime, spec):
                return runtime.resolve()

            source = None
            if spec.source_tree_digest is not None:
                for candidate in (checkout, *(Path(value) for value in legacy_sources)):
                    if _sealed_source_manifest(candidate, spec) is not None:
                        source = candidate
                        break
            elif _valid_checkout(checkout, spec):
                source = checkout

            if source is not None and _valid_runtime(runtime, spec, source):
                return runtime.resolve()

            if source is None:
                from utils.utils import hf_env_offline

                if hf_env_offline():
                    raise RuntimeError(
                        f"The pinned {spec.name} source is not cached and Studio is offline"
                    )
                if spec.archive_url is not None:
                    _install_archive_source(checkout, spec)
                else:
                    _install_checkout(checkout, spec)
                source = checkout
            _install_runtime(runtime, source, spec)
    except Timeout as error:
        raise RuntimeError(f"Timed out waiting for another {spec.name} installation") from error

    if not _valid_runtime(runtime, spec, checkout):
        raise RuntimeError(f"The installed {spec.name} source failed integrity validation")
    return runtime.resolve()


def ensure_spark_tts_source(model_repo_path: Path | str | None = None) -> Path:
    legacy_parent = Path(model_repo_path).parent if model_repo_path is not None else Path.cwd()
    legacy_sources = (legacy_parent / "Spark-TTS",)
    return ensure_pinned_source(SPARK_TTS_SOURCE, legacy_sources = legacy_sources)


def ensure_outetts_source() -> Path:
    backend_root = Path(__file__).resolve().parents[1]
    return ensure_pinned_source(
        OUTETTS_SOURCE,
        legacy_sources = (
            backend_root / "core" / "inference" / "OuteTTS",
            backend_root / "core" / "training" / "inference" / "OuteTTS",
        ),
    )


def _artifact_matches(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def _install_verified_artifact(source: Path, destination: Path) -> None:
    workspace = Path(tempfile.mkdtemp(prefix = ".artifact-", dir = destination.parent))
    staging = workspace / destination.name
    try:
        shutil.copyfile(source, staging)
        if not _artifact_matches(
            staging,
            expected_size = _DAC_SIZE,
            expected_sha256 = _DAC_SHA256,
        ):
            raise RuntimeError("The cached DAC speech weights changed during migration")
        os.replace(staging, destination)
    finally:
        _remove_owned_path(workspace)


def _default_legacy_dac_weights_path() -> Path | None:
    if sys.platform == "win32":
        appdata = (os.environ.get("APPDATA") or "").strip()
        if not appdata:
            return None
        return Path(appdata) / "outeai" / "dac" / _DAC_FILENAME
    return Path.home() / ".cache" / "outeai" / "dac" / _DAC_FILENAME


def ensure_dac_speech_weights(legacy_path: Path | str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    from utils.hf_cache_settings import active_hf_hub_cache
    from utils.utils import hf_env_offline

    hub_cache = Path(active_hf_hub_cache())
    destination = (
        hub_cache
        / "studio-pinned-artifacts"
        / _DAC_REPOSITORY.replace("/", "--")
        / _DAC_REVISION
        / _DAC_FILENAME
    )
    if _artifact_matches(
        destination,
        expected_size = _DAC_SIZE,
        expected_sha256 = _DAC_SHA256,
    ):
        return destination.resolve()

    def _verified_legacy() -> Path | None:
        candidate = (
            Path(legacy_path) if legacy_path is not None else _default_legacy_dac_weights_path()
        )
        if candidate is not None and _artifact_matches(
            candidate,
            expected_size = _DAC_SIZE,
            expected_sha256 = _DAC_SHA256,
        ):
            return candidate.resolve()
        return None

    try:
        destination.parent.mkdir(parents = True, exist_ok = True)
    except OSError:
        # A read-only or full hub cache must not hide weights we can already verify.
        fallback = _verified_legacy()
        if fallback is None:
            raise
        return fallback
    try:
        with FileLock(str(destination.parent / ".install.lock"), timeout = 300):
            if _artifact_matches(
                destination,
                expected_size = _DAC_SIZE,
                expected_sha256 = _DAC_SHA256,
            ):
                return destination.resolve()

            legacy = (
                Path(legacy_path) if legacy_path is not None else _default_legacy_dac_weights_path()
            )
            if legacy is not None and _artifact_matches(
                legacy,
                expected_size = _DAC_SIZE,
                expected_sha256 = _DAC_SHA256,
            ):
                # Same as the download branch below: the copy is an optimisation, so a full
                # disk must not reject weights that already passed the size and sha256 check.
                try:
                    _install_verified_artifact(legacy, destination)
                except OSError:
                    return legacy.resolve()
                return destination.resolve()

            offline = hf_env_offline()
            download_error = None
            downloaded = None
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id = _DAC_REPOSITORY,
                        filename = _DAC_FILENAME,
                        revision = _DAC_REVISION,
                        cache_dir = str(hub_cache),
                        local_files_only = offline,
                    )
                )
            except Exception as error:
                download_error = error

            if downloaded is not None and _artifact_matches(
                downloaded,
                expected_size = _DAC_SIZE,
                expected_sha256 = _DAC_SHA256,
            ):
                # Populate the pinned destination so later loads hit the fast path above
                # instead of re-downloading and re-hashing 295 MB under the install lock.
                # The copy is an optimisation, so a full disk must not fail a verified
                # download; fall back to the hub path the caller used before.
                try:
                    _install_verified_artifact(downloaded, destination)
                except OSError:
                    return downloaded.resolve()
                return destination.resolve()

            if download_error is not None:
                raise RuntimeError(
                    "The pinned DAC speech weights are unavailable in the active Hugging Face cache"
                ) from download_error
            raise RuntimeError("The downloaded DAC speech weights failed integrity validation")
    except OSError:
        # Same reasoning as the mkdir above: taking the lock needs a writable cache, and
        # verified weights we already hold are a better answer than failing the load.
        fallback = _verified_legacy()
        if fallback is None:
            raise
        return fallback
    except Timeout as error:
        raise RuntimeError("Timed out waiting for the DAC speech weights installation") from error


def _module_is_inside(module: ModuleType, package_root: Path) -> bool:
    origins = []
    origin = getattr(module, "__file__", None)
    if origin:
        origins.append(origin)
    origins.extend(getattr(module, "__path__", ()) or ())
    if not origins:
        return False
    for value in origins:
        try:
            if not Path(value).resolve().is_relative_to(package_root):
                return False
        except (OSError, ValueError):
            return False
    return True


def _purge_package_bytecode(package_root: Path) -> None:
    # This is the only thing stopping a stale or planted .pyc from shadowing a verified .py:
    # the manifest skips __pycache__ entirely, and the origin audit reads __file__, which
    # still names the .py. So only the concurrent-purge race is tolerated (another worker
    # deleting the same tree without the install lock); a PermissionError must stay fatal.
    for directory, child_directories, files in os.walk(package_root, topdown = True):
        directory_path = Path(directory)
        for name in tuple(child_directories):
            path = directory_path / name
            if path.is_symlink():
                child_directories.remove(name)
                if name == "__pycache__":
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
            elif name == "__pycache__":
                child_directories.remove(name)
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(path)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                (directory_path / name).unlink(missing_ok = True)


def _remove_package_modules(package: str) -> None:
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)


def import_pinned_module(module_name: str, *, package: str, source: Path | str) -> ModuleType:
    if module_name != package and not module_name.startswith(f"{package}."):
        raise ValueError(f"Only {package} modules can be imported from this pinned source")
    source_root = Path(source).resolve()
    unresolved_package_root = source_root / package
    if unresolved_package_root.is_symlink() or not unresolved_package_root.is_dir():
        raise RuntimeError(f"The pinned {package} package is missing")
    package_root = unresolved_package_root.resolve()
    package_init = package_root / "__init__.py"
    if package_init.is_symlink() or not package_init.is_file():
        raise RuntimeError(f"The pinned {package} package is not sealed")

    with _IMPORT_LOCK:
        for name, loaded_module in list(sys.modules.items()):
            if name != package and not name.startswith(f"{package}."):
                continue
            if not _module_is_inside(loaded_module, package_root):
                sys.modules.pop(name, None)

        source_value = str(source_root)
        while source_value in sys.path:
            sys.path.remove(source_value)
        sys.path.insert(0, source_value)
        try:
            # Inside the try: anything raising here would otherwise strand the cache dir at
            # sys.path[0] for the process lifetime, with nothing imported and no rollback.
            _purge_package_bytecode(package_root)
            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
            invalid_modules = sorted(
                name
                # Snapshot: another thread importing here would otherwise raise
                # "dictionary changed size during iteration" out of a good codec load.
                for name, loaded_module in list(sys.modules.items())
                if (name == package or name.startswith(f"{package}."))
                and not _module_is_inside(loaded_module, package_root)
            )
            if invalid_modules:
                names = ", ".join(invalid_modules)
                raise RuntimeError(
                    f"{package} loaded package modules from outside the pinned source: {names}"
                )
            return module
        except BaseException:
            while source_value in sys.path:
                sys.path.remove(source_value)
            _remove_package_modules(package)
            raise


def deactivate_pinned_package(package: str, source: Path | str | None) -> None:
    with _IMPORT_LOCK:
        if source is not None:
            source_value = str(Path(source).resolve())
            while source_value in sys.path:
                sys.path.remove(source_value)
        _remove_package_modules(package)


def import_sparktts_module(module_name: str, source: Path | str) -> ModuleType:
    return import_pinned_module(module_name, package = "sparktts", source = source)


def import_outetts_module(module_name: str, source: Path | str) -> ModuleType:
    return import_pinned_module(module_name, package = "outetts", source = source)
