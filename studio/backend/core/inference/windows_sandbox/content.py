# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Content-addressed private runtime generations, not startup trust approval.

The broker supplies an explicit file list and the runtime/dependency/profile/helper
digests. No interpreter, package hooks or copied files execute here. This storage
layer is not selected by the production launcher until its other gates pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import time

from .content_files import PathLease, native_files
from .content_access import (
    RuntimeReadLease,
    read_readers,
    recover_readers,
    validate_acl,
)
from .dependencies import FileIdentity, checked_path
from .profiles import WindowsRuntimeError

MAX_FILES = 4096
MAX_BYTES = 1024 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
_BUILD_NAME = re.compile(r"\.build-[0-9a-f]{32}")
_RESERVED = re.compile(r"(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", re.IGNORECASE)
_STORE_MARKER = b'{"kind":"unsloth-windows-runtime-store","version":2}\n'


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", message)


def _digest(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise _invalid("Invalid runtime content digest.")
    return value


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise _invalid("Invalid runtime content filename.")
    parts = value.split("/")
    if len(parts) > 16 or any(
        not part
        or part in (".", "..")
        or part.endswith((".", " "))
        or any(ord(c) < 32 or c in '\\:<>"|?*' for c in part)
        or _RESERVED.fullmatch(part.split(".", 1)[0])
        for part in parts
    ):
        raise _invalid("Runtime content paths cannot escape or use Windows aliases.")
    if PurePosixPath(value).is_absolute():
        raise _invalid("Runtime content paths must be relative.")
    return value


def _encode(value):
    return json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf-8")


def _validate_manifest(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "context", "files"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise _invalid("Unknown runtime manifest schema.")
    context = value["context"]
    if not isinstance(context, dict) or set(context) != {
        "runtime",
        "dependencies",
        "profile",
        "helper",
    }:
        raise _invalid("Incomplete runtime content context.")
    for digest in context.values():
        _digest(digest)
    files = value["files"]
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        raise _invalid("Runtime file count limit exceeded.")
    names, total = set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise _invalid("Malformed runtime file record.")
        name = _relative(item["path"]).casefold()
        if name in names:
            raise _invalid("Colliding runtime content paths.")
        names.add(name)
        _digest(item["sha256"])
        if type(item["size"]) is not int or not 0 <= item["size"] <= MAX_FILE_BYTES:
            raise _invalid("Runtime file byte limit exceeded.")
        total += item["size"]
    if total > MAX_BYTES:
        raise _invalid("Runtime generation byte limit exceeded.")
    for name in names:
        if any(
            str(parent) in names for parent in PurePosixPath(name).parents if str(parent) != "."
        ):
            raise _invalid("Runtime file/directory collision.")
    encoded = _encode(value)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise _invalid("Runtime manifest byte limit exceeded.")
    return encoded


@dataclass(frozen = True)
class SnapshotFile:
    source: FileIdentity
    relative_path: str


@dataclass(frozen = True)
class SnapshotSpec:
    files: tuple[SnapshotFile, ...]
    runtime_digest: str
    dependency_digest: str
    profile_digest: str
    helper_digest: str

    def manifest(self):
        if not isinstance(self.files, tuple) or not 0 < len(self.files) <= MAX_FILES:
            raise _invalid("Runtime file count limit exceeded.")
        value = {
            "version": 1,
            "context": {
                "runtime": self.runtime_digest,
                "dependencies": self.dependency_digest,
                "profile": self.profile_digest,
                "helper": self.helper_digest,
            },
            "files": sorted(
                (
                    {
                        "path": item.relative_path,
                        "sha256": item.source.sha256,
                        "size": item.source.size,
                    }
                    for item in self.files
                ),
                key = lambda item: item["path"],
            ),
        }
        return _validate_manifest(value)


@dataclass(frozen = True)
class ContentGeneration:
    digest: str
    directory: Path
    files: tuple[Path, ...]
    # A copied hash is integrity evidence, not approval for privileged startup.
    trust_classification: str = "payload_only"


class RuntimeContentStore:
    def __init__(
        self,
        root: str | Path,
        *,
        _existing_only = False,
    ):
        self.api = native_files()
        spelled = Path(root)
        if not spelled.is_absolute() or spelled.parent == spelled:
            raise _invalid("A dedicated absolute runtime-store directory is required.")
        parent = checked_path(spelled.parent)
        if not spelled.name or spelled.name in (".", ".."):
            raise _invalid("Invalid runtime-store directory.")
        self.root = parent / _relative(spelled.name)
        self.api.require_ntfs(parent)
        with PathLease() as pins:
            pins.directory(parent)
            if not os.path.lexists(self.root):
                if _existing_only:
                    raise _invalid("The owned runtime store is unavailable for cleanup.")
                self.api.mkdir(self.root)
                self.api.create(self.root / ".store", _STORE_MARKER)
                self.api.create(self.root / ".lock", b"")
                self.api.mkdir(self.root / ".readers")
            self._validate_root(pins)
        if not _existing_only:
            self.recover_readers()
            self.recover_builds()

    def _validate_root(self, pins):
        try:
            readers = read_readers(self, pins)
        except OSError as exc:
            raise _invalid("The existing directory has no valid runtime reader registry.") from exc
        validate_acl(self, pins.directory(self.root), self.root, readers)
        handle = pins.file(self.root / ".store")
        self.api.require_private(handle)
        if self.api.read(handle, 1024) != _STORE_MARKER:
            raise _invalid("The existing directory is not this runtime store.")
        return readers

    @contextmanager
    def _mutation(self):
        # Kernel ownership, not a PID/lockfile-exists check. A dead broker releases
        # the handle; concurrent updates fail explicitly without replaying payloads.
        with PathLease() as pins:
            self.api.require_private(pins.file(self.root / ".lock", exclusive = True))
            readers = self._validate_root(pins)
            yield readers

    @staticmethod
    def _paths(manifest):
        files = {"manifest.json", ".lease"}
        directories = {"files"}
        for item in manifest["files"]:
            name = "files/" + item["path"]
            files.add(name)
            directories.update(
                str(parent) for parent in PurePosixPath(name).parents if str(parent) != "."
            )
        return files, directories

    def _inventory(
        self,
        directory,
        pins,
        readers = None,
    ):
        files: set[str]
        directories: set[str]
        files, directories, pending = set(), set(), [directory]
        while pending:
            current = pending.pop()
            validate_acl(self, pins.directory(current), current, readers or {})
            for child in current.iterdir():
                if len(files) + len(directories) >= 65536:
                    raise _invalid("Runtime directory scan limit exceeded.")
                info = child.lstat()
                if getattr(info, "st_file_attributes", 0) & 0x400:
                    raise _invalid("A runtime generation contains a reparse point.")
                name = child.relative_to(directory).as_posix()
                if child.is_dir():
                    directories.add(name)
                    pending.append(child)
                else:
                    files.add(name)
        return files, directories

    def _verify(
        self,
        digest,
        pins,
        readers = None,
    ):
        directory = self.root / _digest(digest)
        validate_acl(self, pins.directory(directory), directory, readers or {})
        handle = pins.file(directory / "manifest.json")
        self.api.require_private(handle)
        data = self.api.read(handle, MAX_MANIFEST_BYTES)
        try:
            manifest = json.loads(data)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise _invalid("Invalid cached runtime manifest.") from exc
        if _validate_manifest(manifest) != data or hashlib.sha256(data).hexdigest() != digest:
            raise _invalid("Runtime manifest does not match its content address.")
        if self._inventory(directory, pins, readers) != self._paths(manifest):
            raise _invalid("Runtime generation contains missing or unlisted entries.")
        paths = []
        for item in manifest["files"]:
            path = directory / "files" / item["path"]
            handle = pins.file(path)
            validate_acl(self, handle, path, readers or {})
            content = self.api.read(handle, item["size"])
            if (
                len(content) != item["size"]
                or hashlib.sha256(content).hexdigest() != item["sha256"]
            ):
                raise _invalid("Cached runtime bytes changed.")
            paths.append(path)
        return ContentGeneration(digest, directory, tuple(paths))

    def _lease_marker(
        self,
        pins,
        directory,
        *,
        exclusive = False,
    ):
        handle = pins.file(directory / ".lease", exclusive = exclusive)
        self.api.require_private(handle)
        if self.api.info(handle).size:
            raise _invalid("Unexpected runtime lease-marker content.")

    @contextmanager
    def lease(self, digest):
        with PathLease() as pins:
            with self._mutation() as readers:
                directory = self.root / _digest(digest)
                self._lease_marker(pins, directory)
                generation = self._verify(digest, pins, readers)
            # All content and namespace handles remain non-inheritable and open
            # throughout use. Startup must separately bind this lease to a Job.
            yield generation

    def read_access(
        self,
        digest,
        sid,
        *,
        name = None,
    ):
        """Prepare read-only grants; the launcher must bind its process before resume."""
        return RuntimeReadLease(self, _digest(digest), sid, name = name)

    def recover_readers(self):
        return recover_readers(self)

    def publish(self, spec: SnapshotSpec):
        data = spec.manifest()
        digest = hashlib.sha256(data).hexdigest()
        destination = self.root / digest
        with self._mutation() as readers:
            if os.path.lexists(destination):
                with PathLease() as pins:
                    self._lease_marker(pins, destination)
                    self._verify(digest, pins, readers)
                return digest
            staging = self.root / (".build-" + secrets.token_hex(16))
            created_files, created_dirs = [], [staging]
            self.api.mkdir(staging)
            try:
                # Record exact intended names before copying. Recovery never
                # interprets source paths or recursively removes unknown files.
                created_files.append(staging / ".build.json")
                self.api.create(staging / ".build.json", data)
                self.api.mkdir(staging / "files")
                created_dirs.append(staging / "files")
                for item in sorted(spec.files, key = lambda item: item.relative_path):
                    source = item.source
                    with PathLease() as pins:
                        source_path = checked_path(source.path)
                        handle = pins.file(source_path)
                        before = source_path.stat()
                        security = self.api.security_text(handle)
                        content = self.api.read(handle, source.size)
                        if (
                            (before.st_dev, before.st_ino, before.st_size)
                            != (source.device, source.inode, source.size)
                            or len(content) != source.size
                            or hashlib.sha256(content).hexdigest() != source.sha256
                            or self.api.security_text(handle) != security
                        ):
                            raise WindowsRuntimeError(
                                "WINDOWS_SANDBOX_RUNTIME_CHANGED",
                                "Runtime source changed since discovery.",
                            )
                        target = staging / "files" / item.relative_path
                        for parent in reversed(target.parents):
                            if parent.is_relative_to(staging / "files") and not parent.exists():
                                self.api.mkdir(parent)
                                created_dirs.append(parent)
                        created_files.append(target)
                        self.api.create(target, content)
                for name, content in ((".lease", b""), ("manifest.json", data)):
                    target = staging / name
                    created_files.append(target)
                    self.api.create(target, content)
                (staging / ".build.json").unlink()
                # Destination absent under the exclusive store lock. No overwrite
                # or repair of an existing generation, even if it was tampered.
                self._publish_directory(staging, destination)
            except BaseException as original:
                try:
                    self._remove_owned(created_files, created_dirs, staging)
                except Exception as cleanup:
                    raise _invalid(
                        f"Runtime build failed; cleanup also failed: {cleanup}"
                    ) from original
                raise
        return digest

    @staticmethod
    def _publish_directory(staging, destination):
        # A transient open descendant can deny an NTFS directory rename even
        # after our copy handles close. Retry only this no-replace metadata
        # operation, under the caller's store lock, never copying/executing again.
        deadline = time.monotonic() + 0.25
        for delay in (0.025, 0.05, 0.1, None):
            try:
                os.rename(staging, destination)
                return
            except OSError as error:
                if (
                    getattr(error, "winerror", None) not in (5, 32)
                    or delay is None
                    or os.path.lexists(destination)
                    or time.monotonic() + delay >= deadline
                ):
                    raise
                time.sleep(delay)
                if time.monotonic() >= deadline:
                    raise

    def _remove_owned(self, files, directories, boundary):
        # Only exact validated cache paths; never recursively delete a source,
        # follow a reparse point, restore a DACL, or remove unexpected entries.
        if boundary.parent != self.root or not (
            _BUILD_NAME.fullmatch(boundary.name) or _DIGEST.fullmatch(boundary.name)
        ):
            raise _invalid("Invalid content cleanup boundary.")
        metadata = [boundary / ".lease", boundary / "manifest.json", boundary / ".build.json"]
        ordered = [path for path in files if path not in metadata]
        ordered.extend(path for path in metadata if path in files)
        for path in ordered:
            if not path.is_relative_to(boundary):
                raise _invalid("Content cleanup escaped its generation.")
            _relative(path.relative_to(boundary).as_posix())
            if os.path.lexists(path):
                with PathLease() as pins:
                    self.api.require_private(pins.file(path))
                path.unlink()
        for path in sorted(set(directories), key = lambda item: len(item.parts), reverse = True):
            if not path.is_relative_to(boundary):
                raise _invalid("Content cleanup escaped its generation.")
            if path != boundary:
                _relative(path.relative_to(boundary).as_posix())
            if path.exists():
                with PathLease() as pins:
                    self.api.require_private(pins.directory(path))
                path.rmdir()

    def recover_builds(self):
        """Reconcile interrupted builds only after obtaining kernel ownership.

        Publishers hold the store mutation handle for their entire build. Thus
        no active build can be reclaimed merely because its PID/mtime looks old.
        Invalid ownership metadata or unknown entries remain explicit failures.
        """
        recovered = 0
        with self._mutation():
            for index, directory in enumerate(self.root.iterdir()):
                if index >= 4096:
                    raise _invalid("Runtime-store recovery entry limit exceeded.")
                if not _BUILD_NAME.fullmatch(directory.name):
                    continue
                with PathLease() as pins:
                    self.api.require_private(pins.directory(directory))
                    actual_files, actual_dirs = self._inventory(directory, pins)
                    if actual_files:
                        marker_name = (
                            ".build.json" if ".build.json" in actual_files else "manifest.json"
                        )
                        marker = pins.file(directory / marker_name)
                        self.api.require_private(marker)
                        data = self.api.read(marker, MAX_MANIFEST_BYTES)
                        try:
                            manifest = json.loads(data)
                        except (ValueError, UnicodeError, RecursionError) as exc:
                            raise _invalid(
                                "Interrupted runtime build has invalid ownership metadata."
                            ) from exc
                        if _validate_manifest(manifest) != data:
                            raise _invalid("Interrupted runtime build metadata is not canonical.")
                        expected_files, expected_dirs = self._paths(manifest)
                        if (
                            not actual_files <= expected_files | {".build.json"}
                            or not actual_dirs <= expected_dirs
                        ):
                            raise _invalid("Interrupted runtime build contains unowned entries.")
                self._remove_owned(
                    [directory / path for path in sorted(actual_files)],
                    [directory, *(directory / path for path in actual_dirs)],
                    directory,
                )
                recovered += 1
        return recovered

    def collect(self, digest):
        """Remove one validated idle generation; a live lease always prevents GC."""
        digest = _digest(digest)
        directory = self.root / digest
        with self._mutation() as readers:
            if any(record["digest"] == digest for record in readers.values()):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_STORE_BUSY", "Runtime generation still has read-grant owners."
                )
            with PathLease() as pins:
                self._lease_marker(pins, directory, exclusive = True)
                generation = self._verify(digest, pins)
            # Other store users cannot acquire a lease until mutation unlocks.
            files = [directory / ".lease", directory / "manifest.json", *generation.files]
            directories = {directory, directory / "files"}
            for path in generation.files:
                directories.update(
                    parent for parent in path.parents if parent.is_relative_to(directory)
                )
            self._remove_owned(files, directories, directory)
