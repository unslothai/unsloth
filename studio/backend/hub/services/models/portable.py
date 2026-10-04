# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export a cached model to a plain folder (symlinks dereferenced) and import it back (#8798)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from hub.utils import gguf_plan

MANIFEST_NAME = "unsloth-export.json"
_MANIFEST_FORMAT = "unsloth-export"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class PortableModelError(ValueError):
    pass


def _hub_cache() -> Path:
    from utils.hf_cache_settings import get_hf_cache_paths
    return Path(get_hf_cache_paths().hub_cache).expanduser().resolve(strict = False)


def _safe_relative(name: str) -> str:
    rel = name.replace("\\", "/")
    parts = rel.split("/")
    if not rel or rel.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise PortableModelError(f"refusing file name outside the model folder: {name!r}")
    return rel


def _cached_repo(repo_id: str, hub_cache: Path):
    from huggingface_hub import scan_cache_dir

    try:
        scan = scan_cache_dir(hub_cache)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(f"no models cache at {hub_cache}") from exc
    for repo in scan.repos:
        if str(getattr(repo, "repo_type", "")) != "model":
            continue
        if str(getattr(repo, "repo_id", "")).lower() != repo_id.lower():
            continue
        return repo
    raise FileNotFoundError(f"{repo_id} is not in the models cache")


def _newest_revision(repo):
    revisions = list(getattr(repo, "revisions", ()))
    if not revisions:
        raise FileNotFoundError(f"{repo.repo_id} has no snapshot in the cache")
    return max(revisions, key = lambda rev: getattr(rev, "last_modified", 0) or 0)


def _snapshot_files(revision, variant: Optional[str]) -> list[tuple[str, Path]]:
    snapshot = Path(revision.snapshot_path)
    files: list[tuple[str, Path]] = []
    for entry in getattr(revision, "files", ()):
        file_path = Path(entry.file_path)
        try:
            rel = file_path.relative_to(snapshot).as_posix()
        except ValueError:
            continue
        # Same selection as a variant download: its own shards plus mmproj / MTP companions.
        if (
            variant
            and gguf_plan.is_gguf_filename(rel)
            and not gguf_plan.is_main_gguf_variant_path(rel, variant)
            and not gguf_plan.is_companion_gguf_path(rel)
        ):
            continue
        source = Path(getattr(entry, "blob_path", None) or file_path)
        if source.is_file():
            files.append((rel, source))
    return files


def _target_in(folder: Path, rel: str) -> Path:
    # Resolve the parent, not the target: an existing snapshot entry is a symlink into blobs/.
    target = folder / _safe_relative(rel)
    root = folder.resolve(strict = False)
    parent = target.parent.resolve(strict = False)
    if parent != root and root not in parent.parents:
        raise PortableModelError(f"refusing to write outside {folder}: {rel!r}")
    return target


def _copy_tree(files: list[tuple[str, Path]], folder: Path) -> int:
    total = 0
    for rel, source in files:
        target = _target_in(folder, rel)
        target.parent.mkdir(parents = True, exist_ok = True)
        shutil.copy2(source, target, follow_symlinks = True)
        total += target.stat().st_size
    return total


def _place_in_cache(files: list[tuple[str, Path]], repo_dir: Path, snapshot: Path) -> int:
    """huggingface_hub layout: ``blobs/<sha256>`` plus a snapshot symlink, or a plain copy where
    symlinks are refused (Windows without the privilege)."""
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    snapshot.mkdir(parents = True, exist_ok = True)
    total = 0
    for rel, source in files:
        target = _target_in(snapshot, rel)
        digest = hashlib.sha256()
        with open(source, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        blob = blobs / digest.hexdigest()
        if not blob.is_file():
            shutil.copy2(source, blob)
        target.parent.mkdir(parents = True, exist_ok = True)
        if target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(Path(os.path.relpath(blob, target.parent)))
        except OSError:
            shutil.copy2(blob, target)
        total += blob.stat().st_size
    return total


def export_cached_model(repo_id: str, variant: Optional[str], destination: str) -> dict[str, Any]:
    if not _REPO_ID.match(repo_id or ""):
        raise PortableModelError(f"not a model id: {repo_id!r}")
    hub_cache = _hub_cache()
    dest = Path(destination).expanduser().resolve(strict = False)
    if dest == hub_cache or hub_cache in dest.parents:
        raise PortableModelError(
            "the destination is inside the models cache; pick a folder outside it"
        )
    repo = _cached_repo(repo_id, hub_cache)
    revision = _newest_revision(repo)
    files = _snapshot_files(revision, variant)
    if not files:
        raise FileNotFoundError(
            f"{repo_id} has no files to export" + (f" for variant {variant}" if variant else "")
        )
    folder_name = repo_id.replace("/", "--") + (f"--{variant}" if variant else "")
    folder = dest / _safe_relative(folder_name)
    folder.mkdir(parents = True, exist_ok = True)
    size = _copy_tree(files, folder)
    manifest = {
        "format": _MANIFEST_FORMAT,
        "version": 1,
        "repo_id": repo_id,
        "variant": variant,
        "revision": str(getattr(revision, "commit_hash", "") or ""),
        "files": [rel for rel, _ in files],
    }
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent = 2), encoding = "utf-8")
    return {
        "status": "exported",
        "repo_id": repo_id,
        "variant": variant,
        "path": str(folder),
        "files": len(files),
        "size_bytes": size,
    }


def _read_manifest(source: Path) -> dict[str, Any]:
    manifest_path = source / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PortableModelError(f"not an Unsloth export: {MANIFEST_NAME} is missing from {source}")
    try:
        data = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        raise PortableModelError(f"unreadable {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != _MANIFEST_FORMAT:
        raise PortableModelError(f"{MANIFEST_NAME} is not an Unsloth export manifest")
    repo_id = data.get("repo_id")
    if not isinstance(repo_id, str) or not _REPO_ID.match(repo_id):
        raise PortableModelError(f"{MANIFEST_NAME} names no model id")
    files = data.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
        raise PortableModelError(f"{MANIFEST_NAME} lists no files")
    return data


def import_model_folder(source: str) -> dict[str, Any]:
    src = Path(source).expanduser().resolve(strict = False)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a folder")
    data = _read_manifest(src)
    repo_id: str = data["repo_id"]
    files = [_safe_relative(name) for name in data["files"]]
    missing = [name for name in files if not (src / name).is_file()]
    if missing:
        raise PortableModelError(f"the export is incomplete; missing {missing[0]!r}")
    hub_cache = _hub_cache()
    revision = str(data.get("revision") or "") or hashlib.sha256(repo_id.encode()).hexdigest()[:40]
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", revision):
        raise PortableModelError("the manifest revision is not a valid snapshot name")
    repo_dir = hub_cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    # Files already in the snapshot (another variant of the same revision) are kept as cached.
    present = [name for name in files if (snapshot / name).is_file()]
    todo = [(name, src / name) for name in files if name not in present]
    size = sum((snapshot / name).stat().st_size for name in present)
    if todo:
        size += _place_in_cache(todo, repo_dir, snapshot)
    status = "imported" if todo else "already_present"
    refs = repo_dir / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    ref = refs / "main"
    if not ref.exists():
        ref.write_text(revision, encoding = "utf-8")
    return {
        "status": status,
        "repo_id": repo_id,
        "variant": data.get("variant"),
        "path": str(snapshot),
        "files": len(files),
        "size_bytes": size,
    }
