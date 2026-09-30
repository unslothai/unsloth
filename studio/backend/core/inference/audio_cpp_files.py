# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a curated audio.cpp model's files live, and how audio.cpp is shown them.

The weights are ordinary Hub files in the shared HF cache, so the Model Hub,
Download Manager and deletion all see them. audio.cpp, however, picks a loader by
file extension after resolving symlinks, and an HF cache snapshot entry is a
symlink to an extension-less blob, so it cannot be handed the snapshot path.
``materialize`` hardlinks the files under their real names into a small link
farm beside the hub cache (same volume, so no copy and no extra disk). A cache
without symlinks already stores real files and is used as is, except on Windows,
where every model goes through the farm's short per-model path: the Windows
server cannot open a path of 260 characters or more.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AUDIO_CPP_REVISION,
    AudioCppModel,
    lookup,
)
from loggers import get_logger

logger = get_logger(__name__)

_LINK_FARM_DIRNAME = "unsloth-audiocpp-links"
# The Windows audiocpp_server fails to open a model path this long ("model path does not exist"),
# and the long-path prefix does not help.
_WINDOWS_MAX_MODEL_PATH = 259


def _hub_cache() -> Path:
    from utils.hf_cache_settings import active_hf_hub_cache
    return Path(active_hf_hub_cache())


def _is_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _snapshot_dirs(hub_cache: Path) -> list[Path]:
    """Snapshots of the umbrella repo: the pinned revision first, then ``refs/main``, then newest.

    Studio's own downloads pin ``AUDIO_CPP_REVISION``, which does not move ``refs/main``,
    so a model's files can sit in any snapshot; callers take the first one that holds all
    of them. The Download Manager fetches ``main``, which is why other snapshots still count.
    """
    repo_dir = hub_cache / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    try:
        candidates = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()]
    except OSError:
        return []
    try:
        main_sha = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
    except OSError:
        main_sha = ""

    def order(p: Path) -> tuple[int, float]:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (0 if p.name == AUDIO_CPP_REVISION else (1 if p.name == main_sha else 2), -mtime)

    return sorted(candidates, key = order)


def _find(model: AudioCppModel, hub_cache: Path) -> Optional[tuple[Path, list[Path]]]:
    for snapshot in _snapshot_dirs(hub_cache):
        files = _cached_files_in(snapshot, model)
        if files:
            return snapshot, files
    return None


def _cached_files_in(snapshot: Path, model: AudioCppModel) -> Optional[list[Path]]:
    """Every file the model needs inside ``snapshot``, or None when any is missing."""
    found: list[Path] = []
    for pattern in model.files:
        if _is_glob(pattern):
            base = snapshot / Path(pattern).parent
            matches = [
                p
                for p in (sorted(base.iterdir()) if base.is_dir() else [])
                if fnmatch.fnmatch(p.name, Path(pattern).name) and _file_complete(p)
            ]
            # A glob needs as many files as the package publishes: an interrupted download holding one
            # voice embedding must not count as a complete model.
            if len(matches) < max(1, model.min_glob_matches):
                return None
            found.extend(matches)
        else:
            path = snapshot / pattern
            if not _file_complete(path):
                return None
            found.append(path)
    return found


def _file_complete(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def cached_files(model: AudioCppModel, *, hub_cache: Optional[Path] = None) -> Optional[list[Path]]:
    """The model's files in the HF cache (snapshot paths), or None when not fully downloaded."""
    root = hub_cache if hub_cache is not None else _hub_cache()
    found = _find(model, root)
    return found[1] if found else None


def is_downloaded(model: AudioCppModel) -> bool:
    try:
        return cached_files(model) is not None
    except Exception:  # noqa: BLE001 - a probe never fails its caller
        return False


def materialize(model: AudioCppModel, *, hub_cache: Optional[Path] = None) -> str:
    """Path of the model's GGUF under its real name, for the server config.

    Raises ``FileNotFoundError`` when the model is not downloaded.
    """
    root = hub_cache if hub_cache is not None else _hub_cache()
    found = _find(model, root)
    if found is None:
        raise FileNotFoundError(f"{model.display_name} is not downloaded.")
    snapshot, files = found
    primary = snapshot / model.gguf_file
    windows = sys.platform == "win32"
    if not windows and not any(p.is_symlink() for p in files):
        return str(primary)
    prune_link_farm(root)
    # One short directory per model: the package's files keep their layout below the model's
    # folder (PocketTTS reads embeddings/ beside its GGUF).
    farm_root = _link_farm_root(root)
    if _is_link(farm_root):
        _unlink_link(farm_root)
    farm = farm_root / model.key
    served = farm / _package_relative(model, model.gguf_file)
    if windows and len(str(served)) > _WINDOWS_MAX_MODEL_PATH:
        from core.inference.audio_cpp_server import AudioCppUnavailableError
        raise AudioCppUnavailableError(
            f"The path audio.cpp would load {model.display_name} from is {len(str(served))} characters, "
            f"over the {_WINDOWS_MAX_MODEL_PATH} Windows allows. Move the Hugging Face cache to a shorter "
            "path in Settings and download the model again."
        )
    for src in files:
        rel = _package_relative(model, str(src.relative_to(snapshot)).replace("\\", "/"))
        dst = farm / rel
        blob = Path(os.path.realpath(src))
        # A link planted anywhere between the farm and the file would take the write elsewhere.
        for parent in [dst.parent, *dst.parent.parents]:
            if parent == farm_root.parent:
                break
            if _is_link(parent):
                _unlink_link(parent)
        if _already_materialized(dst, blob):
            continue
        dst.parent.mkdir(parents = True, exist_ok = True)
        # Unique per call: two first loads of one model (dictation materializes outside its load lock)
        # must not unlink each other's staging file.
        tmp = dst.with_name(f"{dst.name}.unsloth-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            try:
                os.link(blob, tmp)
            except OSError as exc:
                # Another volume or a filesystem without hardlinks: a copy is the only way to give the file its name.
                logger.info("audio.cpp: hardlink failed for %s (%s); copying", rel, exc)
                shutil.copyfile(blob, tmp)
                shutil.copystat(blob, tmp)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok = True)
    return str(served)


def _package_relative(model: AudioCppModel, repo_path: str) -> Path:
    """A package file's path below the model's own folder: the repo's first segment dropped."""
    parts = repo_path.replace("\\", "/").split("/")
    return Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])


def _link_farm_root(hub_cache: Path) -> Path:
    return hub_cache.parent / _LINK_FARM_DIRNAME


def _already_materialized(dst: Path, blob: Path) -> bool:
    """The farm already holds ``blob`` at ``dst``: the same inode, or (a copy) the same size and mtime."""
    try:
        if not dst.is_file():
            return False
        if os.path.samefile(dst, blob):
            return True
        a, b = dst.stat(), blob.stat()
        return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)
    except OSError:
        return False


def prune_link_farm(hub_cache: Optional[Path] = None) -> int:
    """Drop farm entries whose model is no longer downloaded, so a deleted model frees its disk.

    A hardlink keeps the blob's data alive after the cache deletes it, and a copy is a second
    full copy, so either way the farm entry must go once the cache no longer holds the model.
    Directories that name no curated model (an earlier layout, a removed catalog entry) go too.
    Returns the number of files removed. Never raises.
    """
    removed = 0
    try:
        root = hub_cache if hub_cache is not None else _hub_cache()
        farm = _link_farm_root(root)
        if _is_link(farm) or not farm.is_dir():
            return 0
        for model_dir in list(farm.iterdir()):
            if _is_link(model_dir):
                # Never materialize's work: drop the link itself, never what it points at.
                _unlink_link(model_dir)
                continue
            if not model_dir.is_dir():
                continue
            model = lookup(model_dir.name)
            keep = model is not None and _find(model, root) is not None
            files, dirs = _walk_no_links(model_dir)
            for path in files:
                try:
                    # Staging files of a live materialize are young; only a crashed one is old.
                    stale_tmp = (
                        ".unsloth-tmp" in path.name and time.time() - path.stat().st_mtime > 3600
                    )
                    if not keep or stale_tmp:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
            if keep:
                # A kept model's empty folders may be ones a concurrent materialize has just made.
                continue
            for directory in sorted([*dirs, model_dir], key = lambda p: -len(p.parts)):
                try:
                    directory.rmdir()
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001 - housekeeping never fails a load or delete
        logger.debug("audio.cpp: link farm prune failed: %s", exc)
    return removed


def _is_link(path: Path) -> bool:
    """A symlink or (Windows) junction: something the farm must never traverse or write through."""
    try:
        if path.is_symlink():
            return True
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(attrs & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)) if attrs else False
    except OSError:
        return False


def _unlink_link(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        try:
            os.rmdir(
                path
            )  # a Windows junction is removed as a directory, without touching its target
        except OSError:
            pass


def _walk_no_links(top: Path) -> tuple[list[Path], list[Path]]:
    """Files and directories under ``top``, never descending into (or returning) a link."""
    files: list[Path] = []
    dirs: list[Path] = []
    pending = [top]
    while pending:
        try:
            entries = list(pending.pop().iterdir())
        except OSError:
            continue
        for entry in entries:
            if _is_link(entry):
                _unlink_link(entry)
            elif entry.is_dir():
                dirs.append(entry)
                pending.append(entry)
            elif entry.is_file():
                files.append(entry)
    return files, dirs


def expand_repo_files(
    model: AudioCppModel, hf_token: Optional[str] = None
) -> list[tuple[str, int]]:
    """``(repo path, size)`` for every file the model needs, globs expanded from the Hub listing."""
    from huggingface_hub import HfApi

    api = HfApi(token = hf_token or None)
    out: list[tuple[str, int]] = []
    listed: dict[str, list] = {}
    for pattern in model.files:
        folder = str(Path(pattern).parent).replace("\\", "/")
        if folder not in listed:
            listed[folder] = [
                entry
                for entry in api.list_repo_tree(
                    AUDIO_CPP_REPO,
                    path_in_repo = folder,
                    recursive = False,
                    revision = AUDIO_CPP_REVISION,
                )
                if getattr(entry, "size", None) is not None
            ]
        for entry in listed[folder]:
            if fnmatch.fnmatch(entry.path, pattern):
                out.append((entry.path, int(entry.size or 0)))
    if not out:
        raise ValueError(f"{AUDIO_CPP_REPO} does not publish the files {model.display_name} needs.")
    return out
