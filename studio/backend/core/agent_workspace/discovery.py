# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ignore-aware, bounded repository discovery."""

import errno
import fnmatch
import os
import re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from utils.paths.sensitive import contains_sensitive_path_component

from .common import AgentWorkspaceError, run_bounded


DEFAULT_EXCLUDED_DIRS = frozenset(
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

_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "_netrc",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "tokens.json",
        "terraform.tfvars",
    }
)
_SENSITIVE_FILE_SUFFIXES = frozenset(
    {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx", ".tfvars"}
)
_SAFE_ENV_SUFFIXES = frozenset({".example", ".sample", ".template"})
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")
_QUERY_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "can",
        "change",
        "code",
        "describe",
        "explain",
        "file",
        "from",
        "help",
        "how",
        "into",
        "me",
        "please",
        "project",
        "repository",
        "that",
        "tell",
        "this",
        "what",
        "with",
        "you",
    }
)


class _NestedRepositoryBoundary(OSError):
    """A directory entry is an independently owned Git repository boundary."""


def secure_repository_traversal_supported() -> bool:
    posix_supported = (
        os.open in os.supports_dir_fd
        and os.listdir in os.supports_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )
    if posix_supported:
        return True
    from .windows_traversal import windows_secure_traversal_supported

    return windows_secure_traversal_supported()


def _is_sensitive_repository_path(relative: Path) -> bool:
    """Reject credential-shaped files before any preview bytes are read."""
    name = relative.name.lower()
    if contains_sensitive_path_component(str(relative)):
        return True
    if name in _SENSITIVE_FILE_NAMES or relative.suffix.lower() in _SENSITIVE_FILE_SUFFIXES:
        return True
    if relative.suffix.lower() in {".json", ".yaml", ".yml"} and relative.stem.lower() in {
        "credential",
        "credentials",
        "secret",
        "secrets",
        "token",
        "tokens",
    }:
        return True
    if name.startswith(".env.") and not any(name.endswith(suffix) for suffix in _SAFE_ENV_SUFFIXES):
        return True
    return False


@dataclass(frozen = True)
class _IgnoreRule:
    base: str
    pattern: str
    negated: bool
    directory_only: bool
    anchored: bool

    def matches(self, relative: str, is_directory: bool) -> bool:
        relative = relative.strip("/")
        base = self.base.strip("/")
        if base:
            if relative != base and not relative.startswith(base + "/"):
                return False
            local = relative[len(base) :].lstrip("/")
        else:
            local = relative
        pattern = self.pattern.strip("/")
        if self.directory_only:
            parts = local.split("/")
            directory_parts = parts if is_directory else parts[:-1]
            if self.anchored or "/" in pattern:
                return any(
                    fnmatch.fnmatchcase("/".join(directory_parts[:index]), pattern)
                    for index in range(1, len(directory_parts) + 1)
                )
            return any(fnmatch.fnmatchcase(part, pattern) for part in directory_parts)
        if self.anchored or "/" in pattern:
            return fnmatch.fnmatchcase(local, pattern)
        return any(fnmatch.fnmatchcase(part, pattern) for part in local.split("/"))


def _parse_gitignore(content: str, base: str) -> list[_IgnoreRule]:
    rules = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        elif line.startswith(r"\#"):
            line = line[1:]
        line = line.replace(r"\ ", " ")
        if not line:
            continue
        rules.append(
            _IgnoreRule(
                base = base,
                pattern = line,
                negated = negated,
                directory_only = line.endswith("/"),
                anchored = line.startswith("/"),
            )
        )
    return rules


def _ignored(relative: str, is_directory: bool, rules: Iterable[_IgnoreRule]) -> bool:
    ignored = False
    for rule in rules:
        if rule.matches(relative, is_directory):
            ignored = not rule.negated
    return ignored


def _git_candidates(
    root: Path,
    output_limit: int,
    cancelled = None,
) -> tuple[Optional[list[str]], bool]:
    if cancelled is not None and cancelled():
        return [], False
    try:
        code, output, truncated = run_bounded(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd = root,
            timeout_seconds = 20,
            output_limit = output_limit,
        )
    except AgentWorkspaceError:
        return None, False
    if cancelled is not None and cancelled():
        return [], False
    if code != 0:
        return None, False
    values = output.split("\0")
    if truncated and values:
        values = values[:-1]
    return [value for value in values if value], truncated


def _open_directory_at(directory_fd: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd = directory_fd,
    )
    if not stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise NotADirectoryError(name)
    return descriptor


def _open_scope(root_fd: int, scope: tuple[str, ...]) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in scope:
            next_descriptor = _open_directory_at(descriptor, part)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _directory_has_git_marker(directory_fd: int) -> bool:
    try:
        os.stat(".git", dir_fd = directory_fd, follow_symlinks = False)
    except FileNotFoundError:
        return False
    except OSError:
        # An unreadable or concurrently changing marker is still a boundary.
        return True
    return True


def _read_gitignore_at(directory_fd: int, base: str) -> tuple[_IgnoreRule, ...]:
    descriptor = None
    try:
        descriptor = os.open(
            ".gitignore",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd = directory_fd,
        )
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_size > 256 * 1024:
            return ()
        raw = bytearray()
        while len(raw) <= 256 * 1024:
            chunk = os.read(descriptor, min(8192, 256 * 1024 + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > 256 * 1024:
            return ()
        return tuple(_parse_gitignore(bytes(raw).decode("utf-8"), base))
    except (FileNotFoundError, OSError, UnicodeError):
        return ()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _walk_candidates(
    root_fd: int,
    *,
    max_directories: int,
    cancelled,
    truncation_reasons: list[str],
    skipped: dict[str, int],
) -> Iterable[str]:
    """Walk a non-Git project without resolving mutable directory path strings."""
    stack: list[tuple[tuple[str, ...], tuple[_IgnoreRule, ...]]] = [((), ())]
    directories_seen = 0
    while stack:
        if cancelled is not None and cancelled():
            truncation_reasons.append("cancelled")
            return
        if directories_seen >= max_directories:
            truncation_reasons.append("directory-limit")
            return
        scope, inherited = stack.pop()
        try:
            directory_fd = _open_scope(root_fd, scope)
        except OSError:
            continue
        try:
            if scope and _directory_has_git_marker(directory_fd):
                skipped["nestedRepository"] += 1
                continue
            directories_seen += 1
            base = "/".join(scope)
            rules = (*inherited, *_read_gitignore_at(directory_fd, base))
            try:
                names = sorted(os.listdir(directory_fd), reverse = True)
            except OSError:
                continue
            child_directories: list[str] = []
            files: list[str] = []
            for name in names:
                try:
                    metadata = os.stat(name, dir_fd = directory_fd, follow_symlinks = False)
                except OSError:
                    continue
                if stat_module.S_ISDIR(metadata.st_mode):
                    if name not in DEFAULT_EXCLUDED_DIRS:
                        child_directories.append(name)
                elif stat_module.S_ISREG(metadata.st_mode):
                    files.append(name)
            for name in child_directories:
                # Do not prune ignored directories. A nested or parent negation can
                # reinclude a descendant, so children still need to be considered.
                try:
                    child_fd = _open_directory_at(directory_fd, name)
                except OSError:
                    continue
                os.close(child_fd)
                stack.append(((*scope, name), rules))
            for name in files:
                relative = "/".join((*scope, name))
                if not _ignored(relative, False, rules):
                    yield relative
        finally:
            os.close(directory_fd)


def _looks_binary(sample: bytes) -> bool:
    if b"\0" in sample:
        return True
    if not sample:
        return False
    controls = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return controls / len(sample) > 0.10


def _open_beneath(root_fd: int, relative: Path) -> tuple[int, os.stat_result]:
    """Open one regular file below an already-open root without following links.

    Each path component is resolved with ``openat`` semantics. A concurrent
    rename or symlink swap therefore cannot redirect the read outside the root.
    """
    current_fd = os.dup(root_fd)
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                directory_flags | nofollow,
                dir_fd = current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | nofollow,
            dir_fd = current_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat_module.S_ISREG(metadata.st_mode):
            if stat_module.S_ISDIR(metadata.st_mode) and _directory_has_git_marker(file_fd):
                os.close(file_fd)
                raise _NestedRepositoryBoundary(str(relative))
            os.close(file_fd)
            raise OSError("Repository entry is not a regular file.")
        return file_fd, metadata
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
    root: Path, root_fd: int, expected_identity: Optional[tuple[int, int]]
) -> tuple[int, int]:
    try:
        current = root.stat(follow_symlinks = False)
        opened = os.fstat(root_fd)
    except OSError as exc:
        raise AgentWorkspaceError(
            "Project root identity changed during repository discovery."
        ) from exc
    current_identity = (int(current.st_dev), int(current.st_ino))
    opened_identity = (int(opened.st_dev), int(opened.st_ino))
    expected = _normalize_expected_identity(expected_identity)
    if (
        not stat_module.S_ISDIR(current.st_mode)
        or not stat_module.S_ISDIR(opened.st_mode)
        or current_identity != opened_identity
        or (expected is not None and opened_identity != expected)
    ):
        raise AgentWorkspaceError("Project root identity changed during repository discovery.")
    return opened_identity


def _open_verified_root(
    root: Path, expected_identity: Optional[tuple[int, int]]
) -> tuple[Path, int]:
    descriptor = None
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
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


def _read_gitignore_windows(verified_root, scope: tuple[str, ...], base: str):
    try:
        data = verified_root.read_file((*scope, ".gitignore"), 256 * 1024 + 1)
    except OSError:
        return ()
    if data.truncated or len(data.raw) > 256 * 1024:
        return ()
    try:
        return tuple(_parse_gitignore(data.raw.decode("utf-8"), base))
    except UnicodeError:
        return ()


def _walk_candidates_windows(
    verified_root,
    *,
    max_directories: int,
    cancelled,
    truncation_reasons: list[str],
    skipped: dict[str, int],
) -> Iterable[str]:
    """Walk a Windows tree while every admitted entry has a verified handle."""
    excluded = {name.casefold() for name in DEFAULT_EXCLUDED_DIRS}
    stack: list[tuple[tuple[str, ...], tuple[_IgnoreRule, ...]]] = [((), ())]
    directories_seen = 0
    while stack:
        if cancelled is not None and cancelled():
            truncation_reasons.append("cancelled")
            return
        if directories_seen >= max_directories:
            truncation_reasons.append("directory-limit")
            return
        scope, inherited = stack.pop()
        try:
            entries = verified_root.list_directory(scope)
        except OSError:
            continue
        if scope and any(entry.name.casefold() == ".git" for entry in entries):
            skipped["nestedRepository"] += 1
            continue
        directories_seen += 1
        base = "/".join(scope)
        rules = (*inherited, *_read_gitignore_windows(verified_root, scope, base))
        child_directories: list[str] = []
        files: list[str] = []
        for entry in entries:
            if entry.is_reparse:
                skipped["symlink"] += 1
            elif entry.is_directory and entry.name.casefold() not in excluded:
                child_directories.append(entry.name)
            elif entry.is_file:
                files.append(entry.name)
        for name in sorted(child_directories, reverse = True):
            stack.append(((*scope, name), rules))
        for name in sorted(files, reverse = True):
            relative = "/".join((*scope, name))
            if not _ignored(relative, False, rules):
                yield relative


def _build_repository_map_windows(
    root: Path,
    *,
    max_paths: int,
    max_total_bytes: int,
    max_file_bytes: int,
    preview_bytes: int,
    cancelled,
    expected_identity: Optional[tuple[int, int]],
) -> dict:
    from .windows_traversal import WindowsTraversalRejected, WindowsVerifiedRoot

    entries: list[dict] = []
    skipped = {
        "binary": 0,
        "large": 0,
        "symlink": 0,
        "sensitive": 0,
        "unreadable": 0,
        "excluded": 0,
        "nestedRepository": 0,
    }
    bytes_included = 0
    scanned = 0
    truncated_reasons: list[str] = []
    if cancelled is not None and cancelled():
        truncated_reasons.append("cancelled")

    with WindowsVerifiedRoot.open(root, expected_identity) as verified_root:
        candidates, candidate_list_truncated = _git_candidates(
            Path(verified_root.path),
            output_limit = max(1_000_000, max_paths * 512),
            cancelled = cancelled,
        )
        source = "git" if candidates is not None else "filesystem"
        iterable = (
            candidates
            if candidates is not None
            else _walk_candidates_windows(
                verified_root,
                max_directories = max_paths,
                cancelled = cancelled,
                truncation_reasons = truncated_reasons,
                skipped = skipped,
            )
        )
        excluded = {name.casefold() for name in DEFAULT_EXCLUDED_DIRS}
        for raw_relative in iterable:
            if cancelled is not None and cancelled():
                truncated_reasons.append("cancelled")
                break
            if scanned >= max_paths:
                truncated_reasons.append("path-limit")
                break
            scanned += 1
            normalized = str(raw_relative).replace("\\", "/")
            parts = tuple(part for part in normalized.split("/") if part)
            relative_path = Path(*parts)
            if (
                not parts
                or normalized.startswith("/")
                or ".." in parts
                or any(part.casefold() in excluded for part in parts)
            ):
                skipped["excluded"] += 1
                continue
            if _is_sensitive_repository_path(relative_path):
                skipped["sensitive"] += 1
                continue
            try:
                data = verified_root.read_file(parts, min(8192, max_file_bytes))
                if data.size > max_file_bytes:
                    skipped["large"] += 1
                    continue
                if _looks_binary(data.raw):
                    skipped["binary"] += 1
                    continue
                if bytes_included + data.size > max_total_bytes:
                    truncated_reasons.append("byte-limit")
                    break
                entries.append(
                    {
                        "path": "/".join(parts),
                        "size": data.size,
                        "modifiedNs": data.modified_ns,
                    }
                )
                bytes_included += data.size
            except WindowsTraversalRejected as exc:
                skipped["symlink" if exc.reparse else "unreadable"] += 1
            except OSError:
                skipped["unreadable"] += 1
        try:
            verified_root.recheck()
        except OSError as exc:
            raise AgentWorkspaceError(
                "Project root identity changed during repository discovery."
            ) from exc

    if candidate_list_truncated:
        truncated_reasons.append("candidate-output-limit")
    truncated_reasons = list(dict.fromkeys(truncated_reasons))
    return {
        "source": source,
        "entries": entries,
        "fileCount": len(entries),
        "pathsScanned": scanned,
        "bytesIncluded": bytes_included,
        "skipped": skipped,
        "truncated": bool(truncated_reasons),
        "truncationReasons": truncated_reasons,
        "limits": {
            "maxPaths": max_paths,
            "maxTotalBytes": max_total_bytes,
            "maxFileBytes": max_file_bytes,
            "previewBytes": preview_bytes,
        },
    }


def build_repository_map(
    root: Path,
    *,
    max_paths: int = 20_000,
    max_total_bytes: int = 2 * 1024 * 1024,
    max_file_bytes: int = 256 * 1024,
    preview_bytes: int = 0,
    cancelled = None,
    expected_identity: Optional[tuple[int, int]] = None,
) -> dict:
    """Return a deterministic map, excluding ignored, unsafe, binary and large files."""
    if min(max_paths, max_total_bytes, max_file_bytes) < 1 or preview_bytes < 0:
        raise ValueError("Repository-map bounds must be positive.")
    if not secure_repository_traversal_supported():
        raise AgentWorkspaceError("Secure repository traversal is unavailable on this platform.")
    if preview_bytes:
        raise AgentWorkspaceError(
            "Repository maps are metadata-only. Read file content through an approved workspace tool."
        )
    if os.name == "nt":
        return _build_repository_map_windows(
            root,
            max_paths = max_paths,
            max_total_bytes = max_total_bytes,
            max_file_bytes = max_file_bytes,
            preview_bytes = preview_bytes,
            cancelled = cancelled,
            expected_identity = expected_identity,
        )
    root, root_fd = _open_verified_root(root, expected_identity)
    entries = []
    skipped = {
        "binary": 0,
        "large": 0,
        "symlink": 0,
        "sensitive": 0,
        "unreadable": 0,
        "excluded": 0,
        "nestedRepository": 0,
    }
    bytes_included = 0
    scanned = 0
    truncated_reasons = []
    if cancelled is not None and cancelled():
        truncated_reasons.append("cancelled")
    candidates, candidate_list_truncated = _git_candidates(
        root,
        output_limit = max(1_000_000, max_paths * 512),
        cancelled = cancelled,
    )
    if cancelled is not None and cancelled():
        truncated_reasons.append("cancelled")
    source = "git" if candidates is not None else "filesystem"
    iterable = (
        candidates
        if candidates is not None
        else _walk_candidates(
            root_fd,
            max_directories = max_paths,
            cancelled = cancelled,
            truncation_reasons = truncated_reasons,
            skipped = skipped,
        )
    )

    try:
        for raw_relative in iterable:
            if cancelled is not None and cancelled():
                truncated_reasons.append("cancelled")
                break
            if scanned >= max_paths:
                truncated_reasons.append("path-limit")
                break
            scanned += 1
            relative_path = Path(raw_relative)
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or ".." in relative_path.parts
            ):
                skipped["excluded"] += 1
                continue
            if any(part in DEFAULT_EXCLUDED_DIRS for part in relative_path.parts):
                skipped["excluded"] += 1
                continue
            if _is_sensitive_repository_path(relative_path):
                skipped["sensitive"] += 1
                continue
            file_fd = None
            try:
                file_fd, metadata = _open_beneath(root_fd, relative_path)
                if metadata.st_size > max_file_bytes:
                    skipped["large"] += 1
                    continue
                sample = os.read(file_fd, min(8192, metadata.st_size))
                if _looks_binary(sample):
                    skipped["binary"] += 1
                    continue
                if bytes_included + metadata.st_size > max_total_bytes:
                    truncated_reasons.append("byte-limit")
                    break
                entries.append(
                    {
                        "path": relative_path.as_posix(),
                        "size": metadata.st_size,
                        "modifiedNs": metadata.st_mtime_ns,
                    }
                )
                bytes_included += metadata.st_size
            except _NestedRepositoryBoundary:
                skipped["nestedRepository"] += 1
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.ELOOP:
                    skipped["symlink"] += 1
                else:
                    skipped["unreadable"] += 1
            finally:
                if file_fd is not None:
                    os.close(file_fd)
    finally:
        try:
            _assert_root_identity(root, root_fd, expected_identity)
        finally:
            os.close(root_fd)

    if candidate_list_truncated:
        truncated_reasons.append("candidate-output-limit")
    truncated_reasons = list(dict.fromkeys(truncated_reasons))
    return {
        "source": source,
        "entries": entries,
        "fileCount": len(entries),
        "pathsScanned": scanned,
        "bytesIncluded": bytes_included,
        "skipped": skipped,
        "truncated": bool(truncated_reasons),
        "truncationReasons": truncated_reasons,
        "limits": {
            "maxPaths": max_paths,
            "maxTotalBytes": max_total_bytes,
            "maxFileBytes": max_file_bytes,
            "previewBytes": preview_bytes,
        },
    }


def repository_query_terms(query: str) -> frozenset[str]:
    """Return bounded-selection terms without reading repository state."""
    raw_tokens = [token.casefold().strip("./-") for token in _QUERY_TOKEN.findall(query)]
    return frozenset(
        token for token in raw_tokens if len(token) >= 3 and token not in _QUERY_STOP_WORDS
    )


def select_relevant_repository_paths(
    repository_map: dict,
    query: str,
    *,
    max_results: int = 12,
) -> list[dict]:
    """Select bounded path metadata relevant to caller text.

    This deliberately selects only entries already admitted by
    :func:`build_repository_map`. It never reads file content and never turns a
    large repository map into prompt context.
    """
    if max_results < 1 or max_results > 128:
        raise ValueError("Repository selection bounds must be between 1 and 128.")
    tokens = repository_query_terms(query)
    if not tokens:
        return []

    scored: list[tuple[int, str, dict]] = []
    folded_query = query.casefold()
    for entry in repository_map.get("entries", []):
        path = str(entry.get("path") or "")
        if not path:
            continue
        folded_path = path.casefold()
        basename = Path(path).name.casefold()
        stem = Path(path).stem.casefold()
        path_tokens = {
            token.casefold() for token in re.split(r"[^A-Za-z0-9_]+", folded_path) if token
        }
        score = 0
        if folded_path in folded_query:
            score += 100
        if basename in folded_query:
            score += 60
        if stem and stem in tokens:
            score += 30
        for token in tokens:
            if token == basename:
                score += 24
            elif token in path_tokens:
                score += 12
            elif token in folded_path:
                score += 4
        if score:
            scored.append((score, folded_path, entry))

    scored.sort(key = lambda item: (-item[0], item[1]))
    return [
        {
            "path": str(entry["path"]),
            "size": int(entry.get("size") or 0),
            "score": score,
        }
        for score, _folded_path, entry in scored[:max_results]
    ]
