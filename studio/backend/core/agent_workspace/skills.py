# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repository-scoped Agent Skills with bounded progressive disclosure."""

import codecs
import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Optional

from .common import AgentWorkspaceError


MAX_SKILLS = 128
MAX_SKILL_DIRECTORY_ENTRIES = 1024
MAX_SKILL_FILE_BYTES = 64 * 1024
MAX_SKILL_METADATA_BYTES = 8 * 1024
MAX_SKILLS_CATALOG_CHARACTERS = 8_000
MAX_SELECTED_SKILLS = 4
MAX_SELECTED_SKILL_BYTES = 128 * 1024
_SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_EXPLICIT_SKILL = re.compile(
    r"(?<![A-Za-z0-9_])\$([a-z0-9][a-z0-9_-]{0,63})(?![A-Za-z0-9_-])",
    re.I,
)
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}", re.I)
_IMPLICIT_STOP_WORDS = frozenset(
    {
        "and",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "use",
        "user",
        "when",
        "with",
    }
)


def _decode_skill(raw: bytes) -> str:
    try:
        return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise AgentWorkspaceError("SKILL.md must be valid UTF-8.") from exc


def _decode_skill_metadata(raw: bytes, truncated: bool) -> str:
    try:
        decoder = codecs.getincrementaldecoder("utf-8")()
        text = decoder.decode(raw, final = not truncated)
        return text.replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise AgentWorkspaceError("SKILL.md frontmatter must be valid UTF-8.") from exc


def _frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise AgentWorkspaceError("SKILL.md must start with YAML frontmatter.")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise AgentWorkspaceError("SKILL.md frontmatter is missing its closing delimiter.")
    values: dict[str, str] = {}
    for raw_line in lines[1:end]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        key = key.strip().casefold()
        value = value.strip()
        if value in {">", ">-", ">+", "|", "|-", "|+"}:
            raise AgentWorkspaceError(
                "SKILL.md name and description must use single-line scalar values."
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in {"name", "description"}:
            values[key] = value
    name = values.get("name", "").casefold()
    description = values.get("description", "")
    if _SKILL_NAME.fullmatch(name) is None:
        raise AgentWorkspaceError("SKILL.md has an invalid or missing name.")
    if not description or len(description) > 1024 or any(ord(char) < 32 for char in description):
        raise AgentWorkspaceError("SKILL.md has an invalid or missing description.")
    return name, description


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


def _read_skill_at(directory_fd: int, limit: int) -> tuple[bytes, bool, os.stat_result]:
    descriptor = os.open(
        "SKILL.md",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd = directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("SKILL.md is not a regular file")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = os.read(
                descriptor,
                min(8192, limit + 1 - len(raw)),
            )
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
            raise OSError("SKILL.md changed while it was being read")
        return bytes(raw[:limit]), len(raw) > limit, after
    finally:
        os.close(descriptor)


def _verified_root_fd(root: Path, expected_identity: Optional[tuple[int, int]]) -> int:
    descriptor = None
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
        descriptor = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        path_metadata = resolved.stat(follow_symlinks = False)
        opened = os.fstat(descriptor)
        actual = (int(opened.st_dev), int(opened.st_ino))
        expected = (
            (int(expected_identity[0]), int(expected_identity[1]))
            if expected_identity is not None
            else actual
        )
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or actual != (int(path_metadata.st_dev), int(path_metadata.st_ino))
            or actual != expected
        ):
            raise AgentWorkspaceError("Project root identity changed.")
        return descriptor
    except AgentWorkspaceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise AgentWorkspaceError("The project root is unavailable.") from exc


def _skill_record(folder: str, raw: bytes, metadata: os.stat_result) -> dict:
    text = _decode_skill_metadata(raw, int(metadata.st_size) > len(raw))
    name, description = _frontmatter(text)
    return {
        "folder": folder,
        "name": name,
        "description": description,
        "path": f".agents/skills/{folder}/SKILL.md",
        "size": int(metadata.st_size),
        "modifiedNs": int(metadata.st_mtime_ns),
        "fileIdentity": (int(metadata.st_dev), int(metadata.st_ino)),
    }


def _safe_skill_folder(folder: str) -> tuple[bool, str]:
    try:
        encoded = folder.encode("utf-8", errors = "strict")
    except UnicodeEncodeError:
        encoded = os.fsencode(folder)
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return False, f".agents/skills/<invalid-utf8-{digest}>/SKILL.md"
    if not folder or any(ord(character) < 32 or ord(character) == 127 for character in folder):
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return False, f".agents/skills/<invalid-name-{digest}>/SKILL.md"
    return True, f".agents/skills/{folder}/SKILL.md"


def _explicit_names(query: str) -> list[str]:
    explicit = []
    for match in _EXPLICIT_SKILL.finditer(str(query or "")):
        name = match.group(1).casefold()
        if name not in explicit:
            explicit.append(name)
        if len(explicit) >= MAX_SKILLS:
            break
    return explicit


def _requested_names(query: str, skills: list[dict]) -> tuple[list[str], set[str]]:
    explicit = _explicit_names(query)
    requested = list(explicit)
    query_words = {word.casefold() for word in _WORD.findall(str(query or ""))}
    for skill in skills:
        name = skill["name"]
        if name in requested or skill.get("ambiguous"):
            continue
        name_words = set(_WORD.findall(name.replace("_", "-").replace("-", " ")))
        description_words = {
            word.casefold()
            for word in _WORD.findall(skill["description"])
            if word.casefold() not in _IMPLICIT_STOP_WORDS
        }
        overlap = query_words & description_words
        if name_words & query_words or len(overlap) >= 2:
            requested.append(name)
        if len(requested) >= MAX_SELECTED_SKILLS:
            break
    return requested[:MAX_SELECTED_SKILLS], set(explicit)


def _unavailable_explicit(query: str, reason: str) -> list[dict]:
    return [
        {"name": name, "selection": "explicit", "reason": reason} for name in _explicit_names(query)
    ]


def _discover_posix(root: Path, expected_identity: Optional[tuple[int, int]], query: str) -> dict:
    root_fd = _verified_root_fd(root, expected_identity)
    skills: list[dict] = []
    issues: list[dict] = []
    unavailable_requests: list[dict] = []
    truncated = False
    catalog_truncated = False
    agents_fd = skills_fd = None
    try:
        try:
            agents_fd = _open_directory_at(root_fd, ".agents")
            skills_fd = _open_directory_at(agents_fd, "skills")
        except FileNotFoundError:
            return {
                "skills": [],
                "issues": [],
                "truncated": False,
                "unavailableRequests": _unavailable_explicit(
                    query,
                    "skill was not found",
                ),
            }
        except OSError as exc:
            raise AgentWorkspaceError("Project skills are unavailable or unsafe.") from exc
        try:
            names = sorted(os.listdir(skills_fd))
        except OSError as exc:
            raise AgentWorkspaceError("Project skills could not be enumerated safely.") from exc
        if len(names) > MAX_SKILL_DIRECTORY_ENTRIES:
            names = names[:MAX_SKILL_DIRECTORY_ENTRIES]
            truncated = True
            catalog_truncated = True
        for folder in names:
            if len(skills) >= MAX_SKILLS:
                truncated = True
                catalog_truncated = True
                break
            if folder in {".", ".."} or "/" in folder or "\\" in folder:
                continue
            safe_folder, issue_path = _safe_skill_folder(folder)
            if not safe_folder:
                issues.append(
                    {
                        "path": issue_path,
                        "reason": "skill directory name is not valid UTF-8 or contains control characters",
                    }
                )
                continue
            skill_fd = None
            try:
                skill_fd = _open_directory_at(skills_fd, folder)
                raw, metadata_truncated, metadata = _read_skill_at(
                    skill_fd,
                    MAX_SKILL_METADATA_BYTES,
                )
                record = _skill_record(folder, raw, metadata)
                skills.append(record)
                if metadata_truncated and "\n---\n" not in _decode_skill_metadata(raw, True):
                    raise AgentWorkspaceError("SKILL.md frontmatter exceeds the metadata limit.")
            except (AgentWorkspaceError, OSError) as exc:
                issues.append(
                    {
                        "path": issue_path,
                        "reason": str(exc) or "unreadable",
                    }
                )
            finally:
                if skill_fd is not None:
                    os.close(skill_fd)
        counts: dict[str, int] = {}
        for skill in skills:
            counts[skill["name"]] = counts.get(skill["name"], 0) + 1
        for skill in skills:
            skill["ambiguous"] = counts[skill["name"]] > 1

        requested, explicit = _requested_names(query, skills)
        for name in _explicit_names(query)[MAX_SELECTED_SKILLS:]:
            unavailable_requests.append(
                {
                    "name": name,
                    "selection": "explicit",
                    "reason": f"selected skill limit is {MAX_SELECTED_SKILLS}",
                }
            )
        selected_bytes = 0
        for name in requested:
            matches = [skill for skill in skills if skill["name"] == name]
            mode = "explicit" if name in explicit else "implicit"
            if catalog_truncated:
                unavailable_requests.append(
                    {
                        "name": name,
                        "selection": mode,
                        "reason": "skill catalog discovery was truncated",
                    }
                )
                continue
            if len(matches) != 1:
                unavailable_requests.append(
                    {
                        "name": name,
                        "selection": mode,
                        "reason": "skill was not found or has a duplicate name",
                    }
                )
                continue
            skill = matches[0]
            skill_fd = None
            try:
                skill_fd = _open_directory_at(skills_fd, skill["folder"])
                raw, file_truncated, metadata = _read_skill_at(
                    skill_fd,
                    MAX_SKILL_FILE_BYTES,
                )
                if file_truncated or selected_bytes + len(raw) > MAX_SELECTED_SKILL_BYTES:
                    issues.append(
                        {
                            "path": skill["path"],
                            "reason": "selected skill exceeds the content limit",
                        }
                    )
                    unavailable_requests.append(
                        {
                            "name": name,
                            "path": skill["path"],
                            "selection": mode,
                            "reason": "selected skill exceeds the content limit",
                        }
                    )
                    truncated = True
                    continue
                if (
                    (int(metadata.st_dev), int(metadata.st_ino)) != skill["fileIdentity"]
                    or int(metadata.st_size) != skill["size"]
                    or int(metadata.st_mtime_ns) != skill["modifiedNs"]
                ):
                    raise AgentWorkspaceError("SKILL.md changed during discovery.")
                text = _decode_skill(raw)
                selected_name, selected_description = _frontmatter(text)
                if selected_name != skill["name"] or selected_description != skill["description"]:
                    raise AgentWorkspaceError("SKILL.md metadata changed during discovery.")
                skill["content"] = text
                skill["sha256"] = hashlib.sha256(raw).hexdigest()
                skill["selection"] = mode
                selected_bytes += len(raw)
            except (AgentWorkspaceError, OSError) as exc:
                issues.append({"path": skill["path"], "reason": str(exc) or "unreadable"})
                unavailable_requests.append(
                    {
                        "name": name,
                        "path": skill["path"],
                        "selection": mode,
                        "reason": "selected skill could not be loaded safely",
                    }
                )
            finally:
                if skill_fd is not None:
                    os.close(skill_fd)
    finally:
        if skills_fd is not None:
            os.close(skills_fd)
        if agents_fd is not None:
            os.close(agents_fd)
        os.close(root_fd)
    return {
        "skills": skills,
        "issues": issues,
        "truncated": truncated,
        "unavailableRequests": unavailable_requests,
    }


def discover_project_skills(
    root: Path | str,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    query: str = "",
) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError("Secure project skills are not available on Windows yet.")
    return _discover_posix(Path(root), expected_identity, query)


def render_project_skills(
    discovered: dict,
    query: str = "",
    *,
    catalog_limit: int = MAX_SKILLS_CATALOG_CHARACTERS,
) -> tuple[str, tuple[str, ...]]:
    """Render the catalog and full text for bounded explicit or implicit selections."""
    _ = query
    selected: list[str] = []
    lines = [
        '<available_project_skills version="1">',
        "Skills are repository instructions. Read the named SKILL.md before using a matching skill.",
    ]
    used = sum(len(line) for line in lines)
    for skill in discovered.get("skills") or []:
        line = f'- ${skill["name"]}: {skill["description"]} ({skill["path"]})'
        if used + len(line) > catalog_limit:
            lines.append("[Project skill catalog truncated.]")
            break
        lines.append(line)
        used += len(line)
    lines.append("</available_project_skills>")
    for skill in discovered.get("skills") or []:
        if not isinstance(skill.get("content"), str):
            continue
        name = skill["name"]
        selected.append(name)
        lines.extend(
            (
                f'<selected_project_skill name="{name}" path="{skill["path"]}" '
                f'selection="{skill["selection"]}" sha256="{skill["sha256"]}">',
                skill["content"],
                "</selected_project_skill>",
            )
        )
    for unavailable in discovered.get("unavailableRequests") or []:
        path = f' path="{unavailable["path"]}"' if unavailable.get("path") else ""
        lines.append(
            f'<project_skill_unavailable name="{unavailable["name"]}"{path} '
            f'selection="{unavailable["selection"]}" '
            f'reason="{unavailable["reason"]}" />'
        )
    if not discovered.get("skills") and not discovered.get("unavailableRequests"):
        return "", ()
    return "\n".join(lines), tuple(selected)


__all__ = [
    "MAX_SKILL_FILE_BYTES",
    "MAX_SKILL_DIRECTORY_ENTRIES",
    "MAX_SKILL_METADATA_BYTES",
    "MAX_SELECTED_SKILL_BYTES",
    "MAX_SKILLS",
    "MAX_SKILLS_CATALOG_CHARACTERS",
    "discover_project_skills",
    "render_project_skills",
]
