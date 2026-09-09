# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded, scope-aware AGENTS.md discovery below an identity-bound root."""

import codecs
import errno
import html
import os
import stat
from pathlib import Path
from typing import Optional

from .common import AgentWorkspaceError


DEFAULT_MAX_FILES = 16
DEFAULT_MAX_TARGET_DEPTH = 128
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024
DEFAULT_MAX_FILE_BYTES = 32 * 1024
_INSTRUCTION_FILENAMES = ("AGENTS.override.md", "AGENTS.md")


def secure_instruction_traversal_supported() -> bool:
    return (
        os.name != "nt"
        and os.open in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NONBLOCK")
        and hasattr(os, "O_NOFOLLOW")
    )


def _target_parts(target: Optional[str]) -> tuple[str, ...]:
    if not target:
        return ()
    candidate = Path(target)
    parts = tuple(part for part in candidate.parts if part not in {"", "."})
    if (
        candidate.is_absolute()
        or ".." in parts
        or any("\x00" in part or any(ord(character) < 32 for character in part) for part in parts)
    ):
        raise AgentWorkspaceError("Instruction target is invalid or escapes the project root.")
    if len(parts) > DEFAULT_MAX_TARGET_DEPTH:
        raise AgentWorkspaceError("Instruction target exceeds the supported directory depth.")
    return parts


def _normalize_identity(expected_identity: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    if expected_identity is None:
        return None
    try:
        return int(expected_identity[0]), int(expected_identity[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project root identity is invalid.") from exc


def _open_verified_root(
    root: Path, expected_identity: Optional[tuple[int, int]]
) -> tuple[Path, int]:
    descriptor = None
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
        descriptor = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        path_metadata = resolved.stat(follow_symlinks = False)
        opened_metadata = os.fstat(descriptor)
        path_identity = (int(path_metadata.st_dev), int(path_metadata.st_ino))
        opened_identity = (int(opened_metadata.st_dev), int(opened_metadata.st_ino))
        expected = _normalize_identity(expected_identity)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened_metadata.st_mode)
            or path_identity != opened_identity
            or (expected is not None and opened_identity != expected)
        ):
            raise AgentWorkspaceError("Project root identity changed.")
        return resolved, descriptor
    except AgentWorkspaceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise AgentWorkspaceError("Project root is unavailable.") from exc


def _open_directory_at(directory_fd: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd = directory_fd,
    )
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise NotADirectoryError(name)
    return descriptor


def _open_scope(root_fd: int, scope: tuple[str, ...]) -> int:
    current = os.dup(root_fd)
    try:
        for part in scope:
            following = _open_directory_at(current, part)
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


def _target_directory_parts(root_fd: int, parts: tuple[str, ...]) -> tuple[str, ...]:
    if not parts:
        return ()
    current = os.dup(root_fd)
    try:
        for index, part in enumerate(parts):
            descriptor = os.open(
                part,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd = current,
            )
            try:
                metadata = os.fstat(descriptor)
                if index == len(parts) - 1:
                    if stat.S_ISDIR(metadata.st_mode):
                        return parts
                    if stat.S_ISREG(metadata.st_mode):
                        return parts[:-1]
                    raise AgentWorkspaceError(
                        "Instruction target must be a regular file or directory."
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise AgentWorkspaceError("Instruction target is not a directory.")
                os.close(current)
                current = descriptor
                descriptor = -1
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError(
            "Instruction target is unavailable or crosses a symbolic link."
        ) from exc
    finally:
        os.close(current)


def _read_regular_at(directory_fd: int, name: str, limit: int) -> tuple[bytes, bool]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd = directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("Instruction path is not a regular file")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = os.read(descriptor, min(8192, limit + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("Instruction file changed while it was being read")
        return bytes(raw[:limit]), len(raw) > limit
    finally:
        os.close(descriptor)


def _read_instruction_at(directory_fd: int, limit: int):
    for name in _INSTRUCTION_FILENAMES:
        try:
            raw, truncated = _read_regular_at(directory_fd, name, limit)
        except FileNotFoundError:
            continue
        if raw.strip():
            return name, raw, truncated
    return None


def _decode(raw: bytes, truncated: bool = False) -> str:
    try:
        decoder = codecs.getincrementaldecoder("utf-8")()
        return decoder.decode(raw, final = not truncated)
    except UnicodeDecodeError as exc:
        raise AgentWorkspaceError("Instruction files must be valid UTF-8.") from exc


def _combined(layers: list[dict]) -> str:
    return "\n\n".join(
        f'<agents_instructions path="{html.escape(layer["path"], quote = True)}" '
        f'scope="{html.escape(layer["scope"], quote = True)}">\n'
        f'{html.escape(layer["content"], quote = False)}\n'
        "</agents_instructions>"
        for layer in layers
    )


def _resolve_posix(
    root: Path,
    target: Optional[str],
    *,
    max_files: int,
    max_total_bytes: int,
    max_file_bytes: int,
    expected_identity: Optional[tuple[int, int]],
) -> dict:
    _, root_fd = _open_verified_root(root, expected_identity)
    layers: list[dict] = []
    issues: list[dict] = []
    used = 0
    truncated = False
    try:
        target_directory = _target_directory_parts(root_fd, _target_parts(target))
        for depth in range(len(target_directory) + 1):
            if len(layers) >= max_files or used >= max_total_bytes:
                truncated = True
                break
            scope = target_directory[:depth]
            directory_fd = None
            try:
                directory_fd = _open_scope(root_fd, scope)
                remaining = min(max_file_bytes, max_total_bytes - used)
                selected = _read_instruction_at(directory_fd, remaining)
            except OSError as exc:
                issues.append(
                    {
                        "path": "/".join((*scope, "AGENTS.md")),
                        "reason": "symlink" if exc.errno == errno.ELOOP else "unreadable",
                    }
                )
                continue
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)
            if selected is None:
                continue
            name, raw, file_truncated = selected
            content = _decode(raw, file_truncated)
            used += len(raw)
            layers.append(
                {
                    "path": "/".join((*scope, name)),
                    "scope": "/".join(scope) or ".",
                    "content": content,
                    "truncated": file_truncated,
                    "bytesRead": len(raw),
                }
            )
            truncated = truncated or file_truncated
    finally:
        os.close(root_fd)
    return {
        "layers": layers,
        "combined": _combined(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": "Root guidance applies first; deeper guidance overrides it.",
        "bytesRead": used,
    }


def resolve_agents_instructions(
    root: Path | str,
    target: Optional[str] = None,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    expected_identity: Optional[tuple[int, int]] = None,
) -> dict:
    if min(max_files, max_total_bytes, max_file_bytes) <= 0:
        raise AgentWorkspaceError("Instruction limits must be positive.")
    path = Path(root)
    if not secure_instruction_traversal_supported():
        raise AgentWorkspaceError(
            "Secure project instructions are not available on this platform yet."
        )
    return _resolve_posix(
        path,
        target,
        max_files = max_files,
        max_total_bytes = max_total_bytes,
        max_file_bytes = max_file_bytes,
        expected_identity = expected_identity,
    )


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_MAX_TARGET_DEPTH",
    "resolve_agents_instructions",
    "secure_instruction_traversal_supported",
]
