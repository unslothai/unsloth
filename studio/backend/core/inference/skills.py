# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portable Agent Skills bundle storage and progressive resource loading."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import threading
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import yaml

from utils.paths import ensure_dir, studio_root


MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 20_000
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_FILES = 1_000
MAX_SKILL_FILE_BYTES = 2 * 1024 * 1024
MAX_SKILL_MD_BYTES = 512 * 1024
MAX_SKILL_PAGE_CHARS = 8_000
MAX_SKILL_CATALOG_BYTES = 1_536

_LOCK = threading.RLock()
_REGISTRY_NAME = ".registry.json"
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
        or not all(character.isalnum() or character == "-" for character in normalized)
    ):
        raise SkillError("Skill name must be 1-64 lowercase letters, numbers, or single hyphens.")
    return normalized


def _skills_root() -> Path:
    return ensure_dir(studio_root() / "skills")


def _builtin_skills_root() -> Path:
    return Path(__file__).with_name("builtin_skills")


def _builtin_skills() -> dict[str, tuple[Path, dict]]:
    root = _builtin_skills_root()
    if not root.is_dir():
        return {}
    bundled: dict[str, tuple[Path, dict]] = {}
    for candidate in root.iterdir():
        if candidate.name.startswith("."):
            continue
        metadata = _validate_installed_skill(candidate)
        bundled[metadata["name"]] = (candidate, metadata)
    return bundled


def _skill_directory(name: str) -> Path:
    installed = _skills_root() / name
    if installed.exists():
        return installed
    bundled = _builtin_skills().get(name)
    return bundled[0] if bundled is not None else installed


def _registry_path() -> Path:
    return _skills_root() / _REGISTRY_NAME


def _load_registry() -> dict[str, bool]:
    try:
        payload = json.loads(_registry_path().read_text(encoding = "utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError("Could not read the skill registry.") from exc
    if not isinstance(payload, dict):
        raise SkillError("Skill registry contains invalid data.")
    registry: dict[str, bool] = {}
    for key, value in payload.items():
        if not isinstance(value, bool):
            raise SkillError("Skill registry contains invalid data.")
        try:
            normalized = _normalize_skill_name(key)
        except SkillError as exc:
            raise SkillError("Skill registry contains invalid data.") from exc
        if normalized in registry:
            raise SkillError("Skill registry contains duplicate names.")
        registry[normalized] = value
    return registry


def _save_registry(registry: dict[str, bool]) -> None:
    root = _skills_root()
    fd, temporary_name = tempfile.mkstemp(prefix = ".registry-", suffix = ".json", dir = root)
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            json.dump(registry, handle, sort_keys = True, separators = (",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, _registry_path())
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _read_limited(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SkillError(f"Could not read {path.name}.") from exc
    if size > limit:
        raise SkillError(f"{path.name} exceeds the {limit // 1024} KB limit.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SkillError(f"Could not read {path.name}.") from exc


def _parse_skill_markdown(raw: bytes, parent_name: str) -> dict:
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
    except yaml.YAMLError as exc:
        raise SkillError("SKILL.md contains invalid YAML frontmatter.") from exc
    if not isinstance(frontmatter, dict):
        raise SkillError("SKILL.md frontmatter must be a mapping.")

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    name = _normalize_skill_name(name)
    normalized_parent = _normalize_skill_name(parent_name)
    if name != normalized_parent:
        raise SkillError(f"Skill name '{name}' must match its parent directory '{parent_name}'.")
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

    return {
        "name": name,
        "description": description.strip(),
        **({"license": license_value} if license_value else {}),
        **({"compatibility": compatibility} if compatibility else {}),
        **({"metadata": metadata} if metadata is not None else {}),
        **({"allowed_tools": allowed_tools} if allowed_tools is not None else {}),
    }


def _validate_installed_skill(skill_dir: Path) -> dict:
    if not skill_dir.is_dir() or skill_dir.is_symlink():
        raise SkillError("Skill directory is missing or unsafe.")
    manifest = skill_dir / "SKILL.md"
    if manifest.is_symlink() or not manifest.is_file():
        raise SkillError("Skill bundle must contain SKILL.md at its root.")
    return _parse_skill_markdown(_read_limited(manifest, MAX_SKILL_MD_BYTES), skill_dir.name)


def format_skill_catalog(skills: list[dict]) -> str:
    catalog = "\n".join(
        f"- {skill['name']}: {' '.join(skill['description'].split())}"
        for skill in skills
        if skill["enabled"]
    )
    if len(catalog.encode("utf-8")) > MAX_SKILL_CATALOG_BYTES:
        raise SkillError("Enabled skill catalog exceeds the 1536-byte limit.")
    return catalog


def _normalize_archive_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or PureWindowsPath(name).is_absolute()
        or any(PureWindowsPath(part).drive for part in path.parts)
        or any(not part.rstrip(" .") for part in path.parts)
        or any(part != part.rstrip(" .") for part in path.parts)
        or any(
            part.rstrip(" .").split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS
            for part in path.parts
        )
        or any(
            any(character in _WINDOWS_INVALID_CHARS or ord(character) < 32 for character in part)
            for part in path.parts
        )
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise SkillError(f"Archive contains unsafe path '{name}'.")
    return path


def _portable_archive_key(path: PurePosixPath) -> str:
    return "/".join(
        unicodedata.normalize("NFKC", part).rstrip(" .").casefold() for part in path.parts
    )


def _archive_source(
    archive: zipfile.ZipFile,
) -> tuple[dict, list[tuple[zipfile.ZipInfo, PurePosixPath]]]:
    files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    manifests: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    entries = archive.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise SkillError(f"Skill archive exceeds the {MAX_ARCHIVE_ENTRIES}-entry limit.")
    for entry in entries:
        path = _normalize_archive_name(entry.filename)
        portable_key = _portable_archive_key(path)
        if portable_key in seen:
            raise SkillError(f"Archive contains duplicate path '{entry.filename}'.")
        seen.add(portable_key)
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SkillError("Skill archives cannot contain symbolic links.")
        if entry.flag_bits & 0x1:
            raise SkillError("Encrypted skill archives are not supported.")
        if entry.is_dir():
            continue
        files.append((entry, path))
        if path.name == "SKILL.md":
            manifests.append((entry, path))
    if len(manifests) != 1:
        raise SkillError("Skill archive must contain exactly one SKILL.md.")

    manifest_entry, manifest_path = manifests[0]
    source_root = manifest_path.parent
    if manifest_entry.file_size > MAX_SKILL_MD_BYTES:
        raise SkillError("SKILL.md exceeds the 512 KB limit.")
    try:
        manifest_raw = archive.read(manifest_entry)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SkillError("Could not read SKILL.md from the archive.") from exc
    provisional_name = source_root.name if source_root.parts else ""
    if provisional_name:
        metadata = _parse_skill_markdown(manifest_raw, provisional_name)
    else:
        try:
            text = manifest_raw.decode("utf-8")
            lines = text.splitlines()
            closing = next(
                index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"
            )
            raw_metadata = yaml.safe_load("\n".join(lines[1:closing]))
            root_name = raw_metadata.get("name") if isinstance(raw_metadata, dict) else ""
        except (UnicodeDecodeError, StopIteration, yaml.YAMLError):
            root_name = ""
        metadata = _parse_skill_markdown(manifest_raw, root_name)
    if source_root.parts and any(path == source_root for _, path in files):
        raise SkillError("Archive contains conflicting file paths.")
    selected_files = [
        (entry, path.relative_to(source_root) if source_root.parts else path)
        for entry, path in files
        if not source_root.parts or source_root in path.parents
    ]
    if len(selected_files) > MAX_ARCHIVE_FILES:
        raise SkillError(f"Skill bundle exceeds the {MAX_ARCHIVE_FILES}-file limit.")
    selected_keys = {_portable_archive_key(path) for _, path in selected_files}
    for key in selected_keys:
        parts = key.split("/")
        if any("/".join(parts[:index]) in selected_keys for index in range(1, len(parts))):
            raise SkillError("Archive contains conflicting file paths.")
    if sum(entry.file_size for entry, _ in selected_files) > MAX_EXTRACTED_BYTES:
        raise SkillError("Skill bundle exceeds the 100 MB extracted-size limit.")
    return metadata, selected_files


def import_skill_archive(archive_path: Path, *, replace: bool = False) -> dict:
    try:
        archive_size = archive_path.stat().st_size
    except OSError as exc:
        raise SkillError("Could not read the skill archive.") from exc
    if archive_size <= 0:
        raise SkillError("Skill archive is empty.")
    if archive_size > MAX_ARCHIVE_BYTES:
        raise SkillError("Skill archive exceeds the 100 MB upload limit.")

    with _LOCK:
        root = _skills_root()
        temporary = Path(tempfile.mkdtemp(prefix = ".import-", dir = root))
        backup: Optional[Path] = None
        target: Optional[Path] = None
        try:
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    metadata, selected_files = _archive_source(archive)
                    skill_dir = temporary / metadata["name"]
                    skill_dir.mkdir()
                    for entry, relative_path in selected_files:
                        destination = skill_dir.joinpath(*relative_path.parts)
                        destination.parent.mkdir(parents = True, exist_ok = True)
                        with archive.open(entry) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
                        mode = entry.external_attr >> 16
                        destination.chmod(0o755 if mode & 0o111 else 0o644)
            except zipfile.BadZipFile as exc:
                raise SkillError("Skill bundle must be a valid ZIP archive.") from exc
            except NotImplementedError as exc:
                raise SkillError("Skill archive uses unsupported compression.") from exc

            metadata = _validate_installed_skill(skill_dir)
            target = root / metadata["name"]
            if target.exists() and not replace:
                raise SkillError(f"Skill '{metadata['name']}' is already installed.")
            registry = _load_registry()
            enabled = registry.get(metadata["name"], True)
            format_skill_catalog(
                [skill for skill in list_skills() if skill["name"] != metadata["name"]]
                + [{**metadata, "enabled": enabled}]
            )
            if target.exists():
                backup = root / f".backup-{metadata['name']}-{os.getpid()}-{threading.get_ident()}"
                os.replace(target, backup)
            os.replace(skill_dir, target)
            try:
                registry.setdefault(metadata["name"], True)
                _save_registry(registry)
            except Exception:
                os.replace(target, skill_dir)
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                    backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors = True)
            return {
                **metadata,
                "enabled": registry[metadata["name"]],
                "bundled": False,
            }
        except Exception:
            if (
                backup is not None
                and backup.exists()
                and target is not None
                and not target.exists()
            ):
                os.replace(backup, target)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors = True)


def list_skills() -> list[dict]:
    with _LOCK:
        root = _skills_root()
        registry = _load_registry()
        skills: list[dict] = []
        for candidate in root.iterdir():
            if candidate.name.startswith("."):
                continue
            try:
                metadata = _validate_installed_skill(candidate)
            except SkillError:
                continue
            skills.append(
                {
                    **metadata,
                    "enabled": registry.get(metadata["name"], True),
                    "bundled": False,
                }
            )
        installed_names = {skill["name"] for skill in skills}
        for name, (_, metadata) in _builtin_skills().items():
            if name not in installed_names:
                skills.append({**metadata, "enabled": registry.get(name, True), "bundled": True})
        return sorted(skills, key = lambda skill: skill["name"])


def set_skill_enabled(name: str, enabled: bool) -> dict:
    name = _normalize_skill_name(name)
    with _LOCK:
        bundled = not (_skills_root() / name).exists()
        metadata = _validate_installed_skill(_skill_directory(name))
        registry = _load_registry()
        if enabled:
            format_skill_catalog(
                [skill for skill in list_skills() if skill["name"] != name]
                + [{**metadata, "enabled": True}]
            )
        registry[name] = enabled
        _save_registry(registry)
        return {**metadata, "enabled": enabled, "bundled": bundled}


def delete_skill(name: str) -> None:
    name = _normalize_skill_name(name)
    with _LOCK:
        root = _skills_root()
        target = root / name
        if not target.exists() and name in _builtin_skills():
            raise SkillError(f"Skill '{name}' is bundled with Studio and cannot be deleted.")
        _validate_installed_skill(target)
        quarantine = Path(tempfile.mkdtemp(prefix = f".delete-{name}-", dir = root))
        backup = quarantine / name
        cleanup_quarantine = False
        try:
            os.replace(target, backup)
            registry = _load_registry()
            registry.pop(name, None)
            _save_registry(registry)
            cleanup_quarantine = True
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            cleanup_quarantine = not backup.exists()
            raise
        finally:
            if cleanup_quarantine:
                shutil.rmtree(quarantine, ignore_errors = True)


def read_skill_resource(
    name: str,
    resource: str = "SKILL.md",
    offset: int = 0,
    *,
    page_chars: int = MAX_SKILL_PAGE_CHARS,
) -> str:
    name = _normalize_skill_name(name)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise SkillError("Skill resource offset must be a non-negative integer.")
    if isinstance(page_chars, bool) or not isinstance(page_chars, int) or page_chars <= 0:
        raise SkillError("Skill resource page size must be a positive integer.")
    with _LOCK:
        skills = {skill["name"]: skill for skill in list_skills()}
        skill = skills.get(name)
        if skill is None:
            raise SkillError(f"Skill '{name}' is not installed.")
        if not skill["enabled"]:
            raise SkillError(f"Skill '{name}' is disabled.")

        normalized = resource.replace("\\", "/").strip() or "SKILL.md"
        path = PurePosixPath(normalized)
        if (
            "\x00" in normalized
            or normalized.startswith("/")
            or PureWindowsPath(resource).is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise SkillError("Skill resource path must stay inside the skill directory.")
        root = _skill_directory(name).resolve()
        candidate = root.joinpath(*path.parts)
        try:
            resolved = candidate.resolve(strict = True)
        except (FileNotFoundError, OSError) as exc:
            raise SkillError(f"Skill resource '{normalized}' was not found.") from exc
        if root != resolved and root not in resolved.parents:
            raise SkillError("Skill resource path must stay inside the skill directory.")
        if candidate.is_symlink() or not resolved.is_file():
            raise SkillError("Skill resource must be a regular file.")
        raw = _read_limited(resolved, MAX_SKILL_FILE_BYTES)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError("Skill resources must be UTF-8 text.") from exc
        if offset > len(content):
            raise SkillError("Skill resource offset is past the end of the file.")
        end = min(offset + min(page_chars, MAX_SKILL_PAGE_CHARS), len(content))
        result = (
            f"Skill: {name}\nResource: {normalized}\n"
            f"Characters: {offset}-{end} of {len(content)}\n\n{content[offset:end]}"
        )
        if end < len(content):
            result += (
                "\n\nResource continues. Call read_skill again with "
                f'name="{name}", resource="{normalized}", offset={end}.'
            )
        return result
