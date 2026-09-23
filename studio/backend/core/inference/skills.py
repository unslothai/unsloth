# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import tempfile
import threading
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import yaml

from utils.account_context import is_owner_context
from utils.paths import studio_root, workspace_root


MAX_SKILL_MD_BYTES = 512 * 1024
# Per skill, expanded: keeps a full catalog of 2,000 skills to tens of MB on the listing route.
MAX_SKILL_METADATA_BYTES = 16 * 1024
MAX_SKILL_FILE_BYTES = 2 * 1024 * 1024
MAX_SKILL_PAGE_CHARS = 8_000
MIN_SKILL_PAGE_CHARS = 64
MAX_SKILL_CATALOG_BYTES = 1_536
LARGE_SKILL_CATALOG_BYTES = 4_096
MAX_SKILL_RESOURCE_PATH_BYTES = 400
MAX_SKILL_PATH_COMPONENTS = 256
MAX_SKILLS_PER_ROOT = 1_000
MAX_SKILL_INSTRUCTIONS_BYTES = 256 * 1024

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_OVERRIDES_NAME = "skill-overrides.json"
# dir_fd opens pin reads to the discovered directory; Windows re-walks and checks identity.
_DIR_FD_OPENS = (
    hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
)
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


def _require_unlinked_agent_path(base: Path, *paths: Path) -> None:
    for path in paths:
        if _is_linked_path(path) or (path.exists() and not path.is_dir()):
            raise SkillError("Agent Skills directory is missing or unsafe.")
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise SkillError("Agent Skills directory is missing or unsafe.") from exc


def _write_new_skill_manifest(
    base: Path,
    name: str,
    manifest: bytes,
    *,
    root: Optional[Path] = None,
) -> None:
    base = base.resolve(strict=True)
    if root is None:
        # The owner's home: ~/.agents/skills, both levels checked.
        agents = base / ".agents"
        root = agents / "skills"
        ancestors: tuple[Path, ...] = (agents, root)
    else:
        # A managed account's private workspace: <workspace>/skills.
        root = base / root
        ancestors = (root,)
    _require_unlinked_agent_path(base, *ancestors)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_unlinked_agent_path(base, *ancestors)

    skill_dir = root / name
    skill_dir.mkdir(mode=0o700)
    skill_file = skill_dir / "SKILL.md"
    directories = (*ancestors, skill_dir)
    expected = [os.stat(path, follow_symlinks=False) for path in directories]

    def revalidate(descriptor: int) -> None:
        _require_unlinked_agent_path(base, *directories)
        current = [os.stat(path, follow_symlinks=False) for path in directories]
        file_status = os.stat(skill_file, follow_symlinks=False)
        if (
            not all(map(os.path.samestat, expected, current))
            or _is_linked_path(skill_file)
            or not os.path.samestat(file_status, os.fstat(descriptor))
        ):
            raise SkillError("Agent Skill path changed while the manifest was being written.")

    descriptor: Optional[int] = None

    created_status = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(skill_file, flags, 0o600)

        created_status = os.fstat(descriptor)
        revalidate(descriptor)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(manifest)
            handle.flush()
            os.fsync(handle.fileno())
            revalidate(handle.fileno())
    except Exception:
        # Closed before the cleanup: Windows refuses to unlink a file with an open handle.
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        try:
            current = [os.stat(path, follow_symlinks=False) for path in directories]
            if all(map(os.path.samestat, expected, current)):
                # Remove only the manifest this call created; another writer's file stays.
                if created_status is not None and os.path.samestat(
                    os.stat(skill_file, follow_symlinks=False), created_status
                ):
                    skill_file.unlink()
                skill_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_within(root: Path, identity, parts: tuple[str, ...]) -> int:
    """Open ``root/parts`` one component at a time relative to the previous directory
    descriptor, so a root or ancestor swapped for a link mid-walk cannot redirect the read."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(root, flags | os.O_DIRECTORY)
    try:
        if identity is not None and not os.path.samestat(os.fstat(descriptor), identity):
            raise SkillError("Skill directory changed after it was selected.")
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            part_flags = flags | (getattr(os, "O_BINARY", 0) if last else os.O_DIRECTORY)
            opened = os.open(part, part_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = opened
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_limited(
    path: Path,
    limit: int,
    *,
    contained_in: Optional[Path] = None,
    identity=None,
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
        if contained_in is not None and _DIR_FD_OPENS:
            descriptor = _open_within(contained_in, identity, path.relative_to(contained_in).parts)
        else:
            descriptor = os.open(path, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise SkillError(f"{path.name} must be a regular file.")
        if contained_in is not None and not _DIR_FD_OPENS:
            if _is_linked_path(contained_in):
                raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            root_status = os.stat(contained_in, follow_symlinks=False)
            if identity is not None and not os.path.samestat(root_status, identity):
                raise SkillError("Skill directory changed after it was selected.")
            root = contained_in.resolve(strict=True)
            relative = path.relative_to(contained_in)
            current = contained_in
            for part in relative.parts:
                current = current / part
                if _is_linked_path(current):
                    raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            path.resolve(strict=True).relative_to(root)
            current_status = os.stat(path, follow_symlinks=False)
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
    # Exact match: an indented `---` inside a block scalar is YAML content, not the closer.
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.rstrip() == "---"), None
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
    # Aliases let a small file expand on serialisation; bound the expanded size.
    if (
        metadata is not None
        and sum(len(k) + len(v) for k, v in metadata.items()) > MAX_SKILL_METADATA_BYTES
    ):
        raise SkillError("Skill metadata exceeds the 16 KB limit.")
    allowed_tools = frontmatter.get("allowed-tools")
    if allowed_tools is not None and (
        not isinstance(allowed_tools, str) or len(allowed_tools) > 1024
    ):
        raise SkillError(
            "Skill allowed-tools must be a space-separated string of at most 1024 characters."
        )
    license_value = frontmatter.get("license")
    if license_value is not None and (
        not isinstance(license_value, str) or len(license_value) > 1024
    ):
        raise SkillError("Skill license must be a string of at most 1024 characters.")

    parsed = {
        "name": name,
        "description": description.strip(),
        **({"license": license_value} if license_value is not None else {}),
        **({"compatibility": compatibility} if compatibility is not None else {}),
        **({"metadata": metadata} if metadata is not None else {}),
        **({"allowed_tools": allowed_tools} if allowed_tools is not None else {}),
    }
    try:
        json.dumps(parsed, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillError("Skill fields must contain valid Unicode.") from exc
    return parsed


def _validate_skill_dir(skill_dir: Path) -> tuple[dict, Path]:
    """Metadata plus the real directory. A linked entry (npx skills, Zed, dotfiles) is
    followed once here; every later read is pinned to the resolved directory, not the link."""
    target = skill_dir
    if _is_linked_path(skill_dir):
        try:
            target = skill_dir.resolve(strict=True)
        except OSError as exc:
            raise SkillError("Skill directory is missing or unsafe.") from exc
    if _is_linked_path(target) or not target.is_dir():
        raise SkillError("Skill directory is missing or unsafe.")
    manifest = target / "SKILL.md"
    if _is_linked_path(manifest) or not manifest.is_file():
        raise SkillError("Skill directory must contain a regular SKILL.md file.")
    metadata = _parse_skill_markdown(
        _read_limited(manifest, MAX_SKILL_MD_BYTES, contained_in=target),
        skill_dir.name,
    )
    return metadata, target


_BUNDLED_ROOT = ("bundled", Path(__file__).with_name("bundled_skills"))
_MANAGED_SKILLS_DIR = "skills"


def _owner_home() -> Path:
    # Seam for tests: the installation owner's home, where ~/.agents and ~/.claude live.
    return Path.home()


def _skill_roots(home: Optional[Path] = None) -> tuple[tuple[str, Path], ...]:
    if home is not None:
        base = home
        return (("agents", base / ".agents" / "skills"), ("claude", base / ".claude" / "skills"))
    if not is_owner_context():
        # Managed accounts read their own workspace plus bundled skills, never the host home.
        return (("agents", workspace_root() / _MANAGED_SKILLS_DIR), _BUNDLED_ROOT)
    base = _owner_home()
    return (
        ("agents", base / ".agents" / "skills"),
        ("claude", base / ".claude" / "skills"),
        _BUNDLED_ROOT,
    )


def _default_enabled(source: str) -> bool:
    # Bundled skills ship disabled so a fresh install does not add skill tools to every chat.
    return source != "bundled"


def _override_path() -> Path:
    # The owner keeps the install-root file; a managed account has its own inside its workspace.
    root = studio_root() if is_owner_context() else workspace_root()
    return root / _OVERRIDES_NAME


def _load_overrides() -> dict[str, bool]:
    # A damaged toggle file counts as empty; the next toggle rewrites it.
    path = _override_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable skill overrides at %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Ignoring skill overrides at %s: not a mapping", path)
        return {}
    overrides: dict[str, bool] = {}
    for name, enabled in payload.items():
        try:
            normalized = _normalize_skill_name(name)
        except SkillError:
            logger.warning("Ignoring skill override for invalid name %r in %s", name, path)
            continue
        if not isinstance(enabled, bool):
            logger.warning("Ignoring non-boolean skill override for %s in %s", normalized, path)
            continue
        overrides[normalized] = enabled
    return overrides


def _save_overrides(overrides: dict[str, bool]) -> None:
    path = _override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".skill-overrides-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(overrides, handle, sort_keys=True, separators=(",", ":"))
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
        resolved_root = root.expanduser().resolve(strict=True)
        if not resolved_root.is_dir():
            return []
        candidates = sorted(resolved_root.iterdir(), key=lambda path: path.name)
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        raise SkillError("Could not scan an Agent Skills directory.") from exc
    visible = []
    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        try:
            candidate.name.encode("utf-8")
        except UnicodeEncodeError:
            continue
        # Stray files are skipped; links still go through validation so they are reported.
        try:
            if not candidate.is_dir() and not _is_linked_path(candidate):
                continue
        except OSError:
            continue
        visible.append(candidate)
    if len(visible) > MAX_SKILLS_PER_ROOT:
        raise SkillError(f"Agent Skills directory exceeds the {MAX_SKILLS_PER_ROOT}-entry limit.")
    return visible


def _discover(home: Optional[Path]) -> list[tuple[dict, Optional[Path], Optional[os.stat_result]]]:
    """Every record with, for a valid skill, the directory it was read from and that
    directory's identity at discovery time, so a later read can refuse a swapped tree."""
    overrides = _load_overrides()
    found: list[tuple[dict, Optional[Path], Optional[os.stat_result]]] = []
    selected: dict[str, dict] = {}
    for source, root in _skill_roots(home):
        try:
            candidates = _candidate_dirs(root)
        except SkillError as exc:
            # One unreadable or oversized root must not hide the other roots' skills.
            found.append(
                (
                    {
                        "name": source,
                        "description": "",
                        "source": source,
                        "enabled": False,
                        "valid": False,
                        "shadowed": False,
                        "error": str(exc),
                    },
                    None,
                    None,
                )
            )
            continue
        for candidate in candidates:
            base = {
                "name": candidate.name,
                "description": "",
                "source": source,
                "enabled": False,
                "valid": False,
                "shadowed": False,
            }
            try:
                metadata, skill_dir = _validate_skill_dir(candidate)
                identity = os.stat(skill_dir, follow_symlinks=False)
            except OSError:
                found.append(
                    ({**base, "error": "Skill directory is missing or unsafe."}, None, None)
                )
                continue
            except SkillError as exc:
                found.append(({**base, "error": str(exc)}, None, None))
                continue
            name = metadata["name"]
            if name in selected:
                found.append(
                    (
                        {
                            **base,
                            **metadata,
                            "valid": True,
                            "shadowed": True,
                            "shadowed_by": selected[name]["source"],
                        },
                        None,
                        None,
                    )
                )
                continue
            record = {
                **base,
                **metadata,
                "enabled": overrides.get(name, _default_enabled(source)),
                "valid": True,
            }
            selected[name] = record
            found.append((record, skill_dir, identity))
    return found


def list_skills(*, home: Optional[Path] = None) -> list[dict]:
    with _LOCK:
        return [record for record, _, _ in _discover(home)]


def enabled_skills(*, home: Optional[Path] = None) -> list[dict]:
    return [
        skill
        for skill in list_skills(home=home)
        if skill["valid"] and not skill["shadowed"] and skill["enabled"]
    ]


def _selected_skill(
    name: str, *, home: Optional[Path] = None
) -> tuple[dict, Path, Optional[os.stat_result]]:
    normalized = _normalize_skill_name(name)
    for record, skill_dir, identity in _discover(home):
        if record["valid"] and not record["shadowed"] and record["name"] == normalized:
            return record, skill_dir, identity
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
        record, _, _ = _selected_skill(name, home=home)
        overrides = _load_overrides()
        if enabled == _default_enabled(record["source"]):
            overrides.pop(record["name"], None)
        else:
            overrides[record["name"]] = enabled
        _save_overrides(overrides)
        return {**record, "enabled": enabled}


def create_skill(
    name: str,
    description: str,
    instructions: str,
    *,
    home: Optional[Path] = None,
) -> dict:
    normalized = _normalize_skill_name(name)
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillError("Skill description must be 1-1024 characters.")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SkillError("Skill instructions must be non-empty UTF-8 text.")
    try:
        instruction_bytes = instructions.strip().encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillError("Skill instructions must be valid UTF-8 text.") from exc
    if len(instruction_bytes) > MAX_SKILL_INSTRUCTIONS_BYTES:
        raise SkillError("Skill instructions exceed the 256 KB limit.")

    frontmatter = yaml.safe_dump(
        {"name": normalized, "description": description.strip()},
        allow_unicode=True,
        sort_keys=False,
    )
    manifest = f"---\n{frontmatter}---\n\n{instructions.strip()}\n".encode("utf-8")
    metadata = _parse_skill_markdown(manifest, normalized)

    if home is not None:
        base, root = home, None
    elif is_owner_context():
        base, root = _owner_home(), None
    else:
        base, root = workspace_root(), Path(_MANAGED_SKILLS_DIR)
        # A fresh account's workspace may not exist yet; its own private root is safe to make.
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _LOCK:
        overrides = _load_overrides()
        had_override = normalized in overrides
        if had_override:
            enabled_overrides = dict(overrides)
            enabled_overrides.pop(normalized)
            try:
                _save_overrides(enabled_overrides)
            except OSError as exc:
                raise SkillError("Could not update skill enable overrides.") from exc

        try:
            try:
                _write_new_skill_manifest(base, normalized, manifest, root=root)
            except FileExistsError as exc:
                raise SkillError(f"Skill '{normalized}' already exists.") from exc
            except OSError as exc:
                raise SkillError(f"Could not create skill '{normalized}'.") from exc
        except Exception:
            if had_override:
                try:
                    _save_overrides(overrides)
                except OSError:
                    pass
            raise

    # The path as the user would name it, not the resolved host path.
    display = "~/.agents/skills" if root is None else f"{_MANAGED_SKILLS_DIR}"
    return {
        **metadata,
        "source": "agents",
        "enabled": True,
        "valid": True,
        "shadowed": False,
        "path": f"{display}/{normalized}/SKILL.md",
    }


def format_skill_catalog(
    skills: Optional[list[dict]] = None, *, budget: int = MAX_SKILL_CATALOG_BYTES
) -> str:
    candidates = enabled_skills() if skills is None else skills
    lines: list[str] = []
    size = 0
    dropped = 0
    for skill in candidates:
        line = f"- {skill['name']}: {' '.join(skill['description'].split())}"
        encoded = line.encode("utf-8")
        separator = 1 if lines else 0
        if size + separator + len(encoded) > budget:
            dropped += 1
            continue
        lines.append(line)
        size += separator + len(encoded)
    if dropped:
        # read_skill accepts any enabled name, so the model must know the list is cut.
        lines.append(f"- {dropped} more enabled skills not listed; mention one as @skill-name.")
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
            or any(character in _WINDOWS_INVALID_CHARS or ord(character) < 32 for character in part)
            for part in path.parts
        )
    ):
        raise SkillError("Skill resource path must stay inside the skill directory.")
    # Rejected on every OS: a portable skill cannot carry a name Windows refuses.
    if any(part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS for part in path.parts):
        raise SkillError("Skill resource path cannot use a Windows reserved device name.")
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
        record, skill_dir, identity = _selected_skill(name, home=home)
        if not record["enabled"]:
            raise SkillError(f"Skill '{record['name']}' is disabled.")
        path = _normalize_resource_path(resource)
        try:
            if _is_linked_path(skill_dir):
                raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            if identity is not None and not os.path.samestat(
                os.stat(skill_dir, follow_symlinks=False), identity
            ):
                raise SkillError("Skill directory changed after it was selected.")
            root = skill_dir.resolve(strict=True)
            candidate = skill_dir.joinpath(*path.parts)
            current = skill_dir
            for part in path.parts:
                current = current / part
                if _is_linked_path(current):
                    raise SkillError("Skill resources cannot use symbolic links or reparse points.")
            candidate.resolve(strict=True).relative_to(root)
        except SkillError:
            raise
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise SkillError(f"Skill resource '{path.as_posix()}' was not found.") from exc
        except (OSError, ValueError) as exc:
            raise SkillError("Skill resource path must stay inside the skill directory.") from exc
        raw = _read_limited(
            candidate, MAX_SKILL_FILE_BYTES, contained_in=skill_dir, identity=identity
        )
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
