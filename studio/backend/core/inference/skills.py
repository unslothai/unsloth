# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
import threading
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import yaml

from utils.paths import studio_root


MAX_SKILL_MD_BYTES = 512 * 1024
MAX_SKILL_FILE_BYTES = 2 * 1024 * 1024
MAX_SKILL_PAGE_CHARS = 8_000
MAX_SKILL_CATALOG_BYTES = 1_536
MAX_SKILL_RESOURCE_PATH_BYTES = 400
MAX_SKILL_PATH_COMPONENTS = 256
MAX_SKILLS_PER_ROOT = 1_000

_LOCK = threading.RLock()
_OVERRIDES_NAME = "skill-overrides.json"
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"com{index}" for index in "¹²³"}
    | {f"lpt{index}" for index in "¹²³"}
)
_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')


class SkillError(ValueError):
    pass


class SkillNotFoundError(SkillError):
    pass


def _normalize_skill_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise SkillError("Skill name must be a non-empty string.")
    normalized = unicodedata.normalize("NFKC", name.strip())
    if (
        len(normalized) > 64
        or normalized != normalized.lower()
        or normalized.startswith("-")
        or normalized.endswith("-")
        or "--" in normalized
        or not all(
            "a" <= character <= "z" or "0" <= character <= "9" or character == "-"
            for character in normalized
        )
    ):
        raise SkillError("Skill name must be 1-64 lowercase letters, numbers, or single hyphens.")
    if normalized.casefold() in _WINDOWS_RESERVED_STEMS:
        raise SkillError("Skill name cannot use a Windows reserved device name.")
    return normalized


def _is_linked_path(path: Path) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(reparse_point and attributes & reparse_point)


def _read_limited(
    path: Path,
    limit: int,
    *,
    contained_in: Optional[Path] = None,
) -> bytes:
    descriptor: Optional[int] = None
    try:
        if contained_in is not None and _is_linked_path(path):
            raise SkillError("Skill resources cannot use symbolic links or reparse points.")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise SkillError(f"{path.name} must be a regular file.")
        if contained_in is not None:
            if _is_linked_path(contained_in):
                raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            root = contained_in.resolve(strict = True)
            relative = path.relative_to(contained_in)
            current = contained_in
            for part in relative.parts:
                current = current / part
                if _is_linked_path(current):
                    raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            path.resolve(strict = True).relative_to(root)
            current_status = os.stat(path, follow_symlinks = False)
            if not os.path.samestat(status, current_status):
                raise SkillError("Skill resource changed while it was being opened.")
        if status.st_size > limit:
            raise SkillError(f"{path.name} exceeds the {limit // 1024} KB limit.")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(limit + 1)
    except SkillError:
        raise
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.ELOOP:
            raise SkillError(
                "Skill resources cannot use symbolic links or reparse points."
            ) from exc
        raise SkillError(f"Could not read {path.name}.") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if len(raw) > limit:
        raise SkillError(f"{path.name} exceeds the {limit // 1024} KB limit.")
    return raw


def _parse_skill_markdown(raw: bytes, parent_name: Optional[str] = None) -> dict:
    if len(raw) > MAX_SKILL_MD_BYTES:
        raise SkillError("SKILL.md exceeds the 512 KB limit.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillError("SKILL.md must be UTF-8 text.") from exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with YAML frontmatter.")
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None
    )
    if closing is None:
        raise SkillError("SKILL.md YAML frontmatter is not closed.")
    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    except (
        yaml.YAMLError,
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        ValueError,
    ) as exc:
        raise SkillError("SKILL.md contains invalid YAML frontmatter.") from exc
    if not isinstance(frontmatter, dict):
        raise SkillError("SKILL.md frontmatter must be a mapping.")

    name = _normalize_skill_name(frontmatter.get("name"))
    if parent_name is not None and name != parent_name:
        raise SkillError(f"Skill name '{name}' must match its parent directory '{parent_name}'.")
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillError("Skill description must be 1-1024 characters.")

    compatibility = frontmatter.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str) or not compatibility or len(compatibility) > 500
    ):
        raise SkillError("Skill compatibility must be 1-500 characters when provided.")
    metadata = frontmatter.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise SkillError("Skill metadata keys and values must be strings.")
    allowed_tools = frontmatter.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        raise SkillError("Skill allowed-tools must be a space-separated string.")
    license_value = frontmatter.get("license")
    if license_value is not None and not isinstance(license_value, str):
        raise SkillError("Skill license must be a string.")

    parsed = {
        "name": name,
        "description": description.strip(),
        **({"license": license_value} if license_value is not None else {}),
        **({"compatibility": compatibility} if compatibility is not None else {}),
        **({"metadata": metadata} if metadata is not None else {}),
        **({"allowed_tools": allowed_tools} if allowed_tools is not None else {}),
    }
    try:
        json.dumps(parsed, ensure_ascii = False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillError("Skill fields must contain valid Unicode.") from exc
    return parsed


def _validate_skill_dir(skill_dir: Path) -> dict:
    if _is_linked_path(skill_dir) or not skill_dir.is_dir():
        raise SkillError("Skill directory is missing or unsafe.")
    manifest = skill_dir / "SKILL.md"
    if _is_linked_path(manifest) or not manifest.is_file():
        raise SkillError("Skill directory must contain a regular SKILL.md file.")
    return _parse_skill_markdown(
        _read_limited(manifest, MAX_SKILL_MD_BYTES, contained_in = skill_dir),
        skill_dir.name,
    )


def _skill_roots(home: Optional[Path] = None) -> tuple[tuple[str, Path], ...]:
    base = home if home is not None else Path.home()
    return (("agents", base / ".agents" / "skills"), ("claude", base / ".claude" / "skills"))


def _override_path() -> Path:
    return studio_root() / _OVERRIDES_NAME


def _load_overrides() -> dict[str, bool]:
    try:
        payload = json.loads(_override_path().read_text(encoding = "utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError("Could not read skill enable overrides.") from exc
    if not isinstance(payload, dict):
        raise SkillError("Skill enable overrides contain invalid data.")
    overrides: dict[str, bool] = {}
    for name, enabled in payload.items():
        if not isinstance(enabled, bool):
            raise SkillError("Skill enable overrides contain invalid data.")
        try:
            normalized = _normalize_skill_name(name)
        except SkillError as exc:
            raise SkillError("Skill enable overrides contain invalid data.") from exc
        overrides[normalized] = enabled
    return overrides


def _save_overrides(overrides: dict[str, bool]) -> None:
    path = _override_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    fd, temporary_name = tempfile.mkstemp(
        prefix = ".skill-overrides-", suffix = ".json", dir = path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            json.dump(overrides, handle, sort_keys = True, separators = (",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _candidate_dirs(root: Path) -> list[Path]:
    try:
        resolved_root = root.expanduser().resolve(strict = True)
        if not resolved_root.is_dir():
            return []
        candidates = sorted(resolved_root.iterdir(), key = lambda path: path.name)
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        raise SkillError("Could not scan an Agent Skills directory.") from exc
    if len(candidates) > MAX_SKILLS_PER_ROOT:
        raise SkillError(f"Agent Skills directory exceeds the {MAX_SKILLS_PER_ROOT}-entry limit.")
    visible = []
    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        try:
            candidate.name.encode("utf-8")
        except UnicodeEncodeError:
            continue
        visible.append(candidate)
    return visible


def list_skills(*, home: Optional[Path] = None) -> list[dict]:
    with _LOCK:
        overrides = _load_overrides()
        records: list[dict] = []
        selected: dict[str, dict] = {}
        for source, root in _skill_roots(home):
            for candidate in _candidate_dirs(root):
                base = {
                    "name": candidate.name,
                    "description": "",
                    "source": source,
                    "enabled": False,
                    "valid": False,
                    "shadowed": False,
                }
                try:
                    metadata = _validate_skill_dir(candidate)
                except SkillError as exc:
                    records.append({**base, "error": str(exc)})
                    continue
                name = metadata["name"]
                if name in selected:
                    records.append(
                        {
                            **base,
                            **metadata,
                            "valid": True,
                            "shadowed": True,
                            "shadowed_by": selected[name]["source"],
                        }
                    )
                    continue
                record = {
                    **base,
                    **metadata,
                    "enabled": overrides.get(name, True),
                    "valid": True,
                }
                selected[name] = record
                records.append(record)
        return records


def enabled_skills(*, home: Optional[Path] = None) -> list[dict]:
    return [
        skill
        for skill in list_skills(home = home)
        if skill["valid"] and not skill["shadowed"] and skill["enabled"]
    ]


def _selected_skill(name: str, *, home: Optional[Path] = None) -> tuple[dict, Path]:
    normalized = _normalize_skill_name(name)
    for record in list_skills(home = home):
        if record["valid"] and not record["shadowed"] and record["name"] == normalized:
            roots = dict(_skill_roots(home))
            return record, roots[record["source"]] / normalized
    raise SkillNotFoundError(f"Skill '{normalized}' was not found.")


def set_skill_enabled(
    name: str,
    enabled: bool,
    *,
    home: Optional[Path] = None,
) -> dict:
    if not isinstance(enabled, bool):
        raise SkillError("Skill enabled state must be a boolean.")
    with _LOCK:
        record, _ = _selected_skill(name, home = home)
        overrides = _load_overrides()
        if enabled:
            overrides.pop(record["name"], None)
        else:
            overrides[record["name"]] = False
        _save_overrides(overrides)
        return {**record, "enabled": enabled}


def format_skill_catalog(skills: Optional[list[dict]] = None) -> str:
    candidates = enabled_skills() if skills is None else skills
    lines: list[str] = []
    size = 0
    for skill in candidates:
        line = f"- {skill['name']}: {' '.join(skill['description'].split())}"
        encoded = line.encode("utf-8")
        separator = 1 if lines else 0
        if size + separator + len(encoded) > MAX_SKILL_CATALOG_BYTES:
            continue
        lines.append(line)
        size += separator + len(encoded)
    return "\n".join(lines)


def _normalize_resource_path(resource: str) -> PurePosixPath:
    if not isinstance(resource, str):
        raise SkillError("Skill resource path must be a string.")
    normalized = resource.replace("\\", "/").strip() or "SKILL.md"
    path = PurePosixPath(normalized)
    if (
        normalized.startswith("/")
        or PureWindowsPath(resource).is_absolute()
        or not path.parts
        or len(path.parts) > MAX_SKILL_PATH_COMPONENTS
        or any(part in ("", ".", "..") for part in path.parts)
        or any(PureWindowsPath(part).drive for part in path.parts)
        or any(
            not part.rstrip(" .")
            or part != part.rstrip(" .")
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS
            or any(character in _WINDOWS_INVALID_CHARS or ord(character) < 32 for character in part)
            for part in path.parts
        )
    ):
        raise SkillError("Skill resource path must stay inside the skill directory.")
    try:
        if len(path.as_posix().encode("utf-8")) > MAX_SKILL_RESOURCE_PATH_BYTES or any(
            len(part.encode("utf-8")) > 255 for part in path.parts
        ):
            raise SkillError("Skill resource path is too long.")
    except UnicodeEncodeError as exc:
        raise SkillError("Skill resource path must contain valid Unicode.") from exc
    return path


def read_skill_resource(
    name: str,
    resource: str = "SKILL.md",
    offset: int = 0,
    *,
    page_chars: int = MAX_SKILL_PAGE_CHARS,
    home: Optional[Path] = None,
) -> str:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise SkillError("Skill resource offset must be a non-negative integer.")
    if isinstance(page_chars, bool) or not isinstance(page_chars, int) or page_chars <= 0:
        raise SkillError("Skill resource page size must be a positive integer.")
    with _LOCK:
        record, skill_dir = _selected_skill(name, home = home)
        if not record["enabled"]:
            raise SkillError(f"Skill '{record['name']}' is disabled.")
        path = _normalize_resource_path(resource)
        try:
            if _is_linked_path(skill_dir):
                raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            root = skill_dir.resolve(strict = True)
            candidate = skill_dir.joinpath(*path.parts)
            current = skill_dir
            for part in path.parts:
                current = current / part
                if _is_linked_path(current):
                    raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            candidate.resolve(strict = True).relative_to(root)
        except SkillError:
            raise
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise SkillError(f"Skill resource '{path.as_posix()}' was not found.") from exc
        except (OSError, ValueError) as exc:
            raise SkillError("Skill resource path must stay inside the skill directory.") from exc
        raw = _read_limited(candidate, MAX_SKILL_FILE_BYTES, contained_in = skill_dir)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError("Skill resources must be UTF-8 text.") from exc
        if "\x00" in content:
            raise SkillError("Skill resources must be UTF-8 text, not binary data.")
        if offset > len(content):
            raise SkillError("Skill resource offset is past the end of the file.")
        end = min(offset + min(page_chars, MAX_SKILL_PAGE_CHARS), len(content))
        normalized = path.as_posix()
        result = (
            f"Skill: {record['name']}\nResource: {normalized}\n"
            f"Characters: {offset}-{end} of {len(content)}\n\n{content[offset:end]}"
        )
        if end < len(content):
            result += (
                "\n\nResource continues. Call read_skill again with "
                f'name="{record["name"]}", resource="{normalized}", offset={end}.'
            )
        return result
