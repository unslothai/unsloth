# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded, payload-only package file inventory without imports or .pth processing.

Files retain their selected package-root order and relative layout. No package
DLL is added to a privileged startup load list. The content store must separately
copy and verify these identities; inventory does not grant startup trust.
"""

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat
import time

from .content import SnapshotFile, _relative
from .dependencies import FileIdentity, checked_path
from .profiles import WindowsRuntimeError

CHUNK_BYTES = 1024 * 1024


@dataclass(frozen = True)
class PackageSnapshotBounds:
    roots: int = 8
    entries: int = 100000
    files: int = 65536
    bytes: int = 16 * 1024 * 1024 * 1024
    file_bytes: int = 4 * 1024 * 1024 * 1024
    depth: int = 12
    seconds: int = 120


def _limit(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", message)


def _changed(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_RUNTIME_CHANGED", message)


def _fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _hash_regular_file(path: Path, *, limit: int, check_deadline) -> FileIdentity:
    path = checked_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Package files must be ordinary single-link files."
        )
    if before.st_size > limit:
        raise _limit("Package file or total byte limit exceeded.")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _fingerprint(opened) != _fingerprint(before):
            raise _changed("Package file changed before hashing.")
        while True:
            check_deadline()
            block = stream.read(min(CHUNK_BYTES, limit - count + 1))
            if not block:
                break
            count += len(block)
            if count > limit:
                raise _limit("Package file or total byte limit exceeded while hashing.")
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = checked_path(path).stat()
    if (
        count != before.st_size
        or _fingerprint(before) != _fingerprint(after)
        or _fingerprint(after) != _fingerprint(current)
        or before.st_ctime_ns != current.st_ctime_ns
        or opened.st_ctime_ns != after.st_ctime_ns
    ):
        raise _changed("Package file changed while hashing.")
    return FileIdentity(str(path), digest.hexdigest(), count, before.st_dev, before.st_ino)


def inventory_package_files(
    package_paths: tuple[str, ...], *, bounds: PackageSnapshotBounds = PackageSnapshotBounds()
) -> tuple[SnapshotFile, ...]:
    """Snapshot candidates for explicit selected roots; callers enforce a hard deadline.

    Cooperative checks bound work between filesystem reads. They do not interrupt
    a stuck filesystem operation. .pth files are copied as inert bytes and never
    read as instructions or used to discover additional package roots.
    """
    if type(bounds) is not PackageSnapshotBounds or any(
        type(n) is not int or n <= 0 for n in asdict(bounds).values()
    ):
        raise _limit("Invalid package inventory bounds.")
    if type(package_paths) is not tuple or len(package_paths) > bounds.roots:
        raise _limit("Package root limit exceeded.")
    deadline = time.monotonic() + bounds.seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise _limit("Package inventory deadline exceeded.")

    roots = []
    for path in package_paths:
        check_deadline()
        if type(path) is not str:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID",
                "Selected package paths must be explicit strings.",
            )
        root = checked_path(path)
        if not root.is_dir() or root.parent == root:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Invalid selected package root."
            )
        spelling = str(root).casefold()
        for other in roots:
            previous = str(other).casefold()
            if (
                spelling == previous
                or spelling.startswith(previous + os.sep)
                or previous.startswith(spelling + os.sep)
            ):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_RUNTIME_INVALID", "Selected package roots overlap or alias."
                )
        roots.append(root)
    result = []
    entry_count = total = 0
    names = set()

    def visit(directory, root, index, depth):
        nonlocal entry_count, total
        check_deadline()
        if depth > bounds.depth:
            raise _limit("Package directory depth exceeded.")
        directory = checked_path(directory)
        before = directory.stat()
        if not stat.S_ISDIR(before.st_mode):
            raise _changed("Package directory changed into a special file.")
        with os.scandir(directory) as entries:
            for entry in entries:
                check_deadline()
                entry_count += 1
                if entry_count > bounds.entries:
                    raise _limit("Package entry limit exceeded.")
                path = Path(entry.path)
                info = entry.stat(follow_symlinks = False)
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_RUNTIME_INVALID", "Package reparse points are unsupported."
                    )
                # Import caches change independently of package source/data.
                # Ordinary sourceless .pyc modules outside this directory remain.
                if entry.name.casefold() == "__pycache__" and stat.S_ISDIR(info.st_mode):
                    continue
                relative = _relative(f"packages/{index}/{path.relative_to(root).as_posix()}")
                if relative.casefold() in names:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_RUNTIME_INVALID",
                        "Package filenames collide under Windows case folding.",
                    )
                names.add(relative.casefold())
                if stat.S_ISDIR(info.st_mode):
                    visit(path, root, index, depth + 1)
                elif stat.S_ISREG(info.st_mode):
                    if len(result) >= bounds.files:
                        raise _limit("Package file count exceeded.")
                    identity = _hash_regular_file(
                        path,
                        limit = min(bounds.file_bytes, bounds.bytes - total),
                        check_deadline = check_deadline,
                    )
                    total += identity.size
                    result.append(SnapshotFile(identity, relative))
                else:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_RUNTIME_INVALID", "Package special files are unsupported."
                    )
        after = checked_path(directory).stat()
        if _fingerprint(before) != _fingerprint(after) or before.st_ctime_ns != after.st_ctime_ns:
            raise _changed("Package directory changed during inventory.")

    for index, root in enumerate(roots):
        visit(root, root, index, 0)
    return tuple(sorted(result, key = lambda item: item.relative_path))
