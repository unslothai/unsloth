# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layered, scope-aware AGENTS.md loading for project runs."""

import errno
import html
import os
import stat
from pathlib import Path
from typing import Optional

from .common import AgentWorkspaceError


DEFAULT_MAX_FILES = 16
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024
DEFAULT_MAX_FILE_BYTES = 32 * 1024
DEFAULT_MAX_DIRECTORIES = 20_000
_INSTRUCTION_FILENAMES = ("AGENTS.override.md", "AGENTS.md")
_DISCOVERY_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)


def secure_instruction_traversal_supported() -> bool:
    posix_supported = (
        os.open in os.supports_dir_fd and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
    )
    if posix_supported:
        return True
    from .windows_traversal import windows_secure_traversal_supported

    return windows_secure_traversal_supported()


def _windows_instruction_target_directory(root, parts: tuple[str, ...]) -> tuple[str, ...]:
    if not parts:
        return ()
    try:
        for index in range(len(parts)):
            current = parts[: index + 1]
            kind = root.path_kind(current)
            last = index == len(parts) - 1
            if kind == "file":
                if last:
                    return parts[:-1]
                raise AgentWorkspaceError("Instruction target is not a directory.")
        return parts
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError(
            "Instruction target is unavailable or crosses a reparse point."
        ) from exc


def _windows_not_found(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in {2, 3} or exc.errno in {2, 3}


def _read_instruction_windows(root, scope: tuple[str, ...], limit: int):
    from .windows_traversal import WindowsTraversalRejected
    for name in _INSTRUCTION_FILENAMES:
        try:
            data = root.read_file((*scope, name), limit)
        except OSError as exc:
            if _windows_not_found(exc):
                continue
            if isinstance(exc, WindowsTraversalRejected) and exc.reparse:
                raise OSError(errno.ELOOP, "Instruction file is a reparse point") from exc
            raise
        return name, data.raw, data.truncated
    return None


def _read_regular_at(directory_fd: int, name: str, limit: int) -> tuple[bytes, bool]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd = directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("Instruction path is not a regular file")
        raw = os.read(descriptor, limit + 1)
        return raw[:limit], len(raw) > limit
    finally:
        os.close(descriptor)


def _read_instruction_at(directory_fd: int, limit: int) -> tuple[str, bytes, bool] | None:
    """Read the one applicable file for a scope, preferring the Codex override."""
    for name in _INSTRUCTION_FILENAMES:
        try:
            raw, truncated = _read_regular_at(directory_fd, name, limit)
        except FileNotFoundError:
            continue
        return name, raw, truncated
    return None


def _target_parts(target: Optional[str]) -> tuple[str, ...]:
    if not target:
        return ()
    candidate = Path(target)
    parts = tuple(part for part in candidate.parts if part not in {"", "."})
    if candidate.is_absolute() or ".." in parts:
        raise AgentWorkspaceError("Instruction target escapes the project root.")
    return parts


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
    current_fd = os.dup(root_fd)
    try:
        for part in scope:
            next_fd = _open_directory_at(current_fd, part)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _target_directory_parts(root_fd: int, parts: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve target type through stable directory descriptors."""
    if not parts:
        return ()
    current_fd = os.dup(root_fd)
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            descriptor = os.open(
                part,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd = current_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if last:
                    if stat.S_ISDIR(metadata.st_mode):
                        return parts
                    if stat.S_ISREG(metadata.st_mode):
                        return parts[:-1]
                    raise AgentWorkspaceError(
                        "Instruction target must be a regular file or directory."
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise AgentWorkspaceError("Instruction target is not a directory.")
                os.close(current_fd)
                current_fd = descriptor
                descriptor = -1
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError(
            "Instruction target is unavailable or crosses a symbolic link."
        ) from exc
    finally:
        os.close(current_fd)


def _normalize_expected_identity(
    expected_identity: Optional[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    if expected_identity is None:
        return None
    try:
        return int(expected_identity[0]), int(expected_identity[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project root identity is invalid.") from exc


def _assert_root_identity(
    root: Path, descriptor: int, expected_identity: Optional[tuple[int, int]]
) -> tuple[int, int]:
    try:
        current = root.stat(follow_symlinks = False)
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise AgentWorkspaceError(
            "Project root identity changed during instruction loading."
        ) from exc
    current_identity = (int(current.st_dev), int(current.st_ino))
    opened_identity = (int(opened.st_dev), int(opened.st_ino))
    expected = _normalize_expected_identity(expected_identity)
    if (
        not stat.S_ISDIR(current.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or current_identity != opened_identity
        or (expected is not None and opened_identity != expected)
    ):
        raise AgentWorkspaceError("Project root identity changed during instruction loading.")
    return opened_identity


def _open_verified_root(
    root: Path, expected_identity: Optional[tuple[int, int]] = None
) -> tuple[Path, int]:
    descriptor = None
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        _assert_root_identity(resolved, descriptor, expected_identity)
        return resolved, descriptor
    except AgentWorkspaceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise AgentWorkspaceError("Project root is unavailable.") from exc


def _combined_layers(layers: list[dict]) -> str:
    return "\n\n".join(
        f'<agents_instructions path="{html.escape(layer["path"], quote = True)}" '
        f'scope="{html.escape(layer["scope"], quote = True)}">\n'
        f'{html.escape(layer["content"], quote = False)}\n'
        "</agents_instructions>"
        for layer in layers
    )


def _decode_utf8_prefix(raw: bytes) -> str:
    for trim in range(0, min(3, len(raw)) + 1):
        candidate = raw if trim == 0 else raw[:-trim]
        try:
            return candidate.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start < max(0, len(candidate) - 4):
                raise
    return ""


def _append_windows_instruction_layer(
    verified_root, scope: tuple[str, ...], *, limit: int, layers: list[dict], issues: list[dict]
) -> tuple[int, bool]:
    relative = "/".join((*scope, "AGENTS.md"))
    try:
        selected = _read_instruction_windows(verified_root, scope, limit)
    except OSError as exc:
        issues.append(
            {
                "path": relative,
                "reason": "symlink" if exc.errno == errno.ELOOP else "unreadable",
            }
        )
        return 0, False
    if selected is None:
        return 0, False
    selected_name, raw, read_truncated = selected
    relative = "/".join((*scope, selected_name))
    try:
        content = _decode_utf8_prefix(raw)
    except UnicodeDecodeError:
        issues.append({"path": relative, "reason": "invalid-utf8"})
        return 0, read_truncated
    layers.append(
        {
            "path": relative,
            "scope": "/".join(scope) or ".",
            "content": content,
            "truncated": read_truncated,
            "bytesRead": len(raw),
        }
    )
    return len(raw), read_truncated


def _resolve_agents_instructions_windows(
    root: Path,
    target: Optional[str],
    *,
    max_files: int,
    max_total_bytes: int,
    max_file_bytes: int,
    expected_identity: Optional[tuple[int, int]],
) -> dict:
    from .windows_traversal import WindowsVerifiedRoot

    layers: list[dict] = []
    issues: list[dict] = []
    remaining = max_total_bytes
    truncated = False
    with WindowsVerifiedRoot.open(root, expected_identity) as verified_root:
        directory_parts = _windows_instruction_target_directory(
            verified_root, _target_parts(target)
        )
        for depth in range(len(directory_parts) + 1):
            if len(layers) >= max_files or remaining <= 0:
                truncated = True
                break
            scope = directory_parts[:depth]
            read, file_truncated = _append_windows_instruction_layer(
                verified_root,
                scope,
                limit = min(max_file_bytes, remaining),
                layers = layers,
                issues = issues,
            )
            remaining -= read
            truncated = truncated or file_truncated
        try:
            verified_root.recheck()
        except OSError as exc:
            raise AgentWorkspaceError(
                "Project root identity changed during instruction loading."
            ) from exc
    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": "later layers override earlier layers",
        "bytesRead": max_total_bytes - remaining,
    }


def _resolve_targeted_repository_instructions_windows(
    root: Path,
    targets: list[str],
    *,
    max_files: int,
    max_total_bytes: int,
    max_file_bytes: int,
    expected_identity: Optional[tuple[int, int]],
) -> dict:
    from .windows_traversal import WindowsVerifiedRoot

    layers: list[dict] = []
    issues: list[dict] = []
    normalized_targets: list[tuple[str, tuple[str, ...]]] = []
    remaining = max_total_bytes
    truncated = False
    with WindowsVerifiedRoot.open(root, expected_identity) as verified_root:
        seen_targets: set[str] = set()
        for raw_target in targets:
            parts = _target_parts(raw_target)
            normalized = "/".join(parts)
            if not normalized or normalized in seen_targets:
                continue
            directory_parts = _windows_instruction_target_directory(verified_root, parts)
            seen_targets.add(normalized)
            normalized_targets.append((normalized, directory_parts))

        scopes: set[tuple[str, ...]] = {()}
        for _target, directory_parts in normalized_targets:
            scopes.update(directory_parts[:depth] for depth in range(1, len(directory_parts) + 1))
        for scope in sorted(scopes, key = lambda value: (len(value), value)):
            if len(layers) >= max_files or remaining <= 0:
                truncated = True
                break
            read, file_truncated = _append_windows_instruction_layer(
                verified_root,
                scope,
                limit = min(max_file_bytes, remaining),
                layers = layers,
                issues = issues,
            )
            remaining -= read
            truncated = truncated or file_truncated
        try:
            verified_root.recheck()
        except OSError as exc:
            raise AgentWorkspaceError(
                "Project root identity changed during instruction loading."
            ) from exc
    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": (
            "root rules apply repository-wide; nested rules apply only beneath "
            "their selected target scope; deeper layers override ancestors; "
            "sibling scopes are isolated"
        ),
        "bytesRead": max_total_bytes - remaining,
        "targets": [target for target, _parts in normalized_targets],
    }


def _resolve_repository_instructions_windows(
    root: Path,
    *,
    max_files: int,
    max_total_bytes: int,
    max_file_bytes: int,
    max_directories: int,
    expected_identity: Optional[tuple[int, int]],
) -> dict:
    from .windows_traversal import WindowsVerifiedRoot

    stack: list[tuple[str, ...]] = [()]
    layers: list[dict] = []
    issues: list[dict] = []
    remaining = max_total_bytes
    directories_seen = 0
    truncated = False
    with WindowsVerifiedRoot.open(root, expected_identity) as verified_root:
        while stack:
            scope = stack.pop()
            directories_seen += 1
            if directories_seen > max_directories:
                truncated = True
                break
            if len(layers) < max_files and remaining > 0:
                read, file_truncated = _append_windows_instruction_layer(
                    verified_root,
                    scope,
                    limit = min(max_file_bytes, remaining),
                    layers = layers,
                    issues = issues,
                )
                remaining -= read
                truncated = truncated or file_truncated
            else:
                truncated = True
            try:
                entries = verified_root.list_directory(scope)
            except OSError:
                issues.append(
                    {
                        "path": "/".join(scope) or ".",
                        "reason": "unreadable-directory",
                    }
                )
                continue
            for entry in sorted(entries, key = lambda item: item.name, reverse = True):
                if (
                    entry.is_directory
                    and not entry.is_reparse
                    and entry.name not in _DISCOVERY_EXCLUDED_DIRECTORIES
                ):
                    stack.append((*scope, entry.name))
        try:
            verified_root.recheck()
        except OSError as exc:
            raise AgentWorkspaceError(
                "Project root identity changed during instruction loading."
            ) from exc
    layers.sort(key = lambda layer: (layer["scope"].count("/"), layer["scope"]))
    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": "rules apply only within their scope; deeper layers override ancestors",
        "bytesRead": max_total_bytes - remaining,
        "directoriesScanned": min(directories_seen, max_directories),
    }


def resolve_agents_instructions(
    root: Path,
    target: Optional[str] = None,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    expected_identity: Optional[tuple[int, int]] = None,
) -> dict:
    """Load AGENTS.md from root to the target, in increasing precedence order."""
    if max_files < 1 or max_total_bytes < 1 or max_file_bytes < 1:
        raise ValueError("Instruction bounds must be positive.")
    if os.name == "nt":
        return _resolve_agents_instructions_windows(
            root,
            target,
            max_files = max_files,
            max_total_bytes = max_total_bytes,
            max_file_bytes = max_file_bytes,
            expected_identity = expected_identity,
        )
    if not secure_instruction_traversal_supported():
        raise AgentWorkspaceError("Secure AGENTS.md traversal is unavailable on this platform.")
    root, root_fd = _open_verified_root(root, expected_identity)

    try:
        relative_parts = _target_directory_parts(root_fd, _target_parts(target))
    except Exception:
        os.close(root_fd)
        raise

    layers = []
    issues = []
    remaining = max_total_bytes
    truncated = False
    directory_fd = os.dup(root_fd)
    scope_parts: list[str] = []
    try:
        for depth in range(len(relative_parts) + 1):
            if len(layers) >= max_files or remaining <= 0:
                truncated = True
                break
            relative = "/".join((*scope_parts, "AGENTS.md"))
            try:
                selected = _read_instruction_at(directory_fd, min(max_file_bytes, remaining))
            except OSError as exc:
                issues.append(
                    {
                        "path": relative,
                        "reason": "symlink" if exc.errno == errno.ELOOP else "unreadable",
                    }
                )
            else:
                if selected is not None:
                    selected_name, raw, read_truncated = selected
                    relative = "/".join((*scope_parts, selected_name))
                    take = len(raw)
                    try:
                        content = _decode_utf8_prefix(raw)
                    except UnicodeDecodeError:
                        issues.append({"path": relative, "reason": "invalid-utf8"})
                    else:
                        layers.append(
                            {
                                "path": relative,
                                "scope": "/".join(scope_parts) or ".",
                                "content": content,
                                "truncated": read_truncated,
                                "bytesRead": take,
                            }
                        )
                        remaining -= take
                        truncated = truncated or read_truncated
            if depth == len(relative_parts):
                break
            try:
                next_fd = _open_directory_at(directory_fd, relative_parts[depth])
            except OSError as exc:
                raise AgentWorkspaceError("Instruction target changed during loading.") from exc
            os.close(directory_fd)
            directory_fd = next_fd
            scope_parts.append(relative_parts[depth])
    finally:
        try:
            os.close(directory_fd)
            _assert_root_identity(root, root_fd, expected_identity)
        finally:
            os.close(root_fd)

    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": "later layers override earlier layers",
        "bytesRead": max_total_bytes - remaining,
    }


def resolve_targeted_repository_instructions(
    root: Path,
    targets: list[str],
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    expected_identity: Optional[tuple[int, int]] = None,
) -> dict:
    """Load only root and target-applicable instruction layers.

    The root layer applies repository-wide. A nested layer is admitted only
    when at least one selected target is beneath its scope. Within a target
    chain, later and deeper layers override their ancestors. Sibling scopes do
    not affect one another.
    """
    if max_files < 1 or max_total_bytes < 1 or max_file_bytes < 1:
        raise ValueError("Instruction bounds must be positive.")
    if len(targets) > 128:
        raise ValueError("Too many repository instruction targets.")
    if os.name == "nt":
        return _resolve_targeted_repository_instructions_windows(
            root,
            targets,
            max_files = max_files,
            max_total_bytes = max_total_bytes,
            max_file_bytes = max_file_bytes,
            expected_identity = expected_identity,
        )
    if not secure_instruction_traversal_supported():
        raise AgentWorkspaceError("Secure AGENTS.md traversal is unavailable on this platform.")

    root, root_fd = _open_verified_root(root, expected_identity)
    issues: list[dict] = []
    layers: list[dict] = []
    remaining = max_total_bytes
    truncated = False
    try:
        normalized_targets: list[tuple[str, tuple[str, ...]]] = []
        seen_targets: set[str] = set()
        for raw_target in targets:
            parts = _target_parts(raw_target)
            normalized = "/".join(parts)
            if not normalized or normalized in seen_targets:
                continue
            directory_parts = _target_directory_parts(root_fd, parts)
            seen_targets.add(normalized)
            normalized_targets.append((normalized, directory_parts))

        scopes: set[tuple[str, ...]] = {()}
        for _target, directory_parts in normalized_targets:
            scopes.update(directory_parts[:depth] for depth in range(1, len(directory_parts) + 1))

        for scope in sorted(scopes, key = lambda value: (len(value), value)):
            if len(layers) >= max_files or remaining <= 0:
                truncated = True
                break
            try:
                directory_fd = _open_scope(root_fd, scope)
            except OSError:
                issues.append(
                    {
                        "path": "/".join(scope) or ".",
                        "reason": "changed-directory",
                    }
                )
                continue
            try:
                relative = "/".join((*scope, "AGENTS.md"))
                try:
                    selected = _read_instruction_at(directory_fd, min(max_file_bytes, remaining))
                except OSError as exc:
                    issues.append(
                        {
                            "path": relative,
                            "reason": ("symlink" if exc.errno == errno.ELOOP else "unreadable"),
                        }
                    )
                    continue
                if selected is None:
                    continue
                selected_name, raw, read_truncated = selected
                relative = "/".join((*scope, selected_name))
                try:
                    content = _decode_utf8_prefix(raw)
                except UnicodeDecodeError:
                    issues.append({"path": relative, "reason": "invalid-utf8"})
                    continue
                layers.append(
                    {
                        "path": relative,
                        "scope": "/".join(scope) or ".",
                        "content": content,
                        "truncated": read_truncated,
                        "bytesRead": len(raw),
                    }
                )
                remaining -= len(raw)
                truncated = truncated or read_truncated
            finally:
                os.close(directory_fd)
    finally:
        try:
            _assert_root_identity(root, root_fd, expected_identity)
        finally:
            os.close(root_fd)

    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": (
            "root rules apply repository-wide; nested rules apply only beneath "
            "their selected target scope; deeper layers override ancestors; "
            "sibling scopes are isolated"
        ),
        "bytesRead": max_total_bytes - remaining,
        "targets": [target for target, _parts in normalized_targets],
    }


def resolve_repository_instructions(
    root: Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    expected_identity: Optional[tuple[int, int]] = None,
) -> dict:
    """Discover bounded repository-wide instruction layers with explicit scopes.

    This form is suitable for a model request made before the target files are
    known. Each nested rule is labeled with the subtree where it applies.
    """
    if min(max_files, max_total_bytes, max_file_bytes, max_directories) < 1:
        raise ValueError("Instruction bounds must be positive.")
    if os.name == "nt":
        return _resolve_repository_instructions_windows(
            root,
            max_files = max_files,
            max_total_bytes = max_total_bytes,
            max_file_bytes = max_file_bytes,
            max_directories = max_directories,
            expected_identity = expected_identity,
        )
    if not secure_instruction_traversal_supported() or os.listdir not in os.supports_fd:
        raise AgentWorkspaceError("Secure AGENTS.md traversal is unavailable on this platform.")
    root, root_fd = _open_verified_root(root, expected_identity)
    stack: list[tuple[str, ...]] = [()]
    layers: list[dict] = []
    issues: list[dict] = []
    remaining = max_total_bytes
    directories_seen = 0
    truncated = False
    try:
        while stack:
            scope = stack.pop()
            try:
                directory_fd = _open_scope(root_fd, scope)
            except OSError:
                issues.append(
                    {
                        "path": "/".join(scope) or ".",
                        "reason": "changed-directory",
                    }
                )
                continue
            try:
                directories_seen += 1
                if directories_seen > max_directories:
                    truncated = True
                    break
                relative = "/".join((*scope, "AGENTS.md"))
                if len(layers) < max_files and remaining > 0:
                    try:
                        selected = _read_instruction_at(
                            directory_fd, min(max_file_bytes, remaining)
                        )
                    except OSError as exc:
                        issues.append(
                            {
                                "path": relative,
                                "reason": ("symlink" if exc.errno == errno.ELOOP else "unreadable"),
                            }
                        )
                    else:
                        if selected is not None:
                            selected_name, raw, read_truncated = selected
                            relative = "/".join((*scope, selected_name))
                            try:
                                content = _decode_utf8_prefix(raw)
                            except UnicodeDecodeError:
                                issues.append({"path": relative, "reason": "invalid-utf8"})
                            else:
                                layers.append(
                                    {
                                        "path": relative,
                                        "scope": "/".join(scope) or ".",
                                        "content": content,
                                        "truncated": read_truncated,
                                        "bytesRead": len(raw),
                                    }
                                )
                                remaining -= len(raw)
                                truncated = truncated or read_truncated
                elif remaining <= 0 or len(layers) >= max_files:
                    truncated = True

                try:
                    names = sorted(os.listdir(directory_fd), reverse = True)
                except OSError:
                    issues.append(
                        {
                            "path": "/".join(scope) or ".",
                            "reason": "unreadable-directory",
                        }
                    )
                    continue
                for name in names:
                    if name in _DISCOVERY_EXCLUDED_DIRECTORIES:
                        continue
                    try:
                        child_fd = _open_directory_at(directory_fd, name)
                    except OSError:
                        continue
                    os.close(child_fd)
                    stack.append((*scope, name))
            finally:
                os.close(directory_fd)
    finally:
        try:
            _assert_root_identity(root, root_fd, expected_identity)
        finally:
            os.close(root_fd)

    layers.sort(key = lambda layer: (layer["scope"].count("/"), layer["scope"]))
    return {
        "layers": layers,
        "combined": _combined_layers(layers),
        "truncated": truncated,
        "issues": issues,
        "precedence": "rules apply only within their scope; deeper layers override ancestors",
        "bytesRead": max_total_bytes - remaining,
        "directoriesScanned": min(directories_seen, max_directories),
    }
