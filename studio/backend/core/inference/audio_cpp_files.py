# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where an audio.cpp model's files live, and how audio.cpp is shown them.

The weights are ordinary Hub files in the shared HF cache, downloaded like any GGUF
variant, so the Model Hub, Download Manager and deletion all see them. audio.cpp,
however, picks a loader by file extension after resolving symlinks, and an HF cache
snapshot entry is a symlink to an extension-less blob, so it cannot be handed the
snapshot path. ``materialize`` hardlinks the variant's files under their real names
into a small link farm beside the hub cache (same volume, so no copy and no extra
disk). A cache without symlinks already stores real files and is used as is, except
on Windows, where every model goes through the farm's short per-model path: the
Windows server cannot open a path of 260 characters or more.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from core.inference.audio_cpp_models import AudioCppModel, snapshot_dirs
from loggers import get_logger

logger = get_logger(__name__)

_LINK_FARM_DIRNAME = "unsloth-audiocpp-links"
# Records which repo files a farm entry mirrors, so pruning needs no catalog.
_SOURCE_MARKER = ".unsloth-source.json"
# The Windows audiocpp_server fails to open a model path this long ("model path does not exist"),
# and the long-path prefix does not help.
_WINDOWS_MAX_MODEL_PATH = 259


def _hub_cache() -> Path:
    from utils.hf_cache_settings import active_hf_hub_cache
    return Path(active_hf_hub_cache())


def _file_complete(path: Path, size: int = 0) -> bool:
    try:
        if not path.is_file():
            return False
        actual = path.stat().st_size
        return actual > 0 and (size <= 0 or actual == size)
    except OSError:
        return False


def _find(model: AudioCppModel, hub_cache: Path) -> Optional[tuple[Path, list[Path]]]:
    """The first snapshot holding every file of the model's variant, with those files."""
    if model.local_path:
        path = Path(model.local_path)
        return (path.parent, [path]) if _file_complete(path) else None
    for snapshot in snapshot_dirs(model.repo_id, hub_cache):
        found = []
        for f in model.variant.files:
            path = snapshot / f.path
            if not _file_complete(path, f.size):
                break
            found.append(path)
        else:
            return snapshot, found
    return None


def cached_files(model: AudioCppModel, *, hub_cache: Optional[Path] = None) -> Optional[list[Path]]:
    """The variant's files in the HF cache (snapshot paths), or None when not fully downloaded."""
    root = hub_cache if hub_cache is not None else _hub_cache()
    try:
        found = _find(model, root)
    except OSError:
        return None
    return found[1] if found else None


def is_downloaded(model: AudioCppModel) -> bool:
    try:
        return cached_files(model) is not None
    except Exception:  # noqa: BLE001 - a probe never fails its caller
        return False


def missing_files(
    model: AudioCppModel, *, hub_cache: Optional[Path] = None
) -> list[tuple[str, int]]:
    """``(repo path, size)`` of the variant's files the best snapshot still lacks."""
    if model.local_path:
        return []
    root = hub_cache if hub_cache is not None else _hub_cache()
    snapshots = snapshot_dirs(model.repo_id, root)
    first: Optional[list[tuple[str, int]]] = None
    for snapshot in snapshots:
        missing = [
            (f.path, f.size)
            for f in model.variant.files
            if not _file_complete(snapshot / f.path, f.size)
        ]
        if not missing:
            return []
        if first is None:
            first = missing
    # Measured against refs/main (listed first), where the unpinned downloads land: the fewest
    # missing in an older snapshot would leave both incomplete.
    return first if first is not None else [(f.path, f.size) for f in model.variant.files]


def _relative(model: AudioCppModel, repo_path: str) -> Path:
    """A variant file's path below the model's own folder: the umbrella folder prefix dropped."""
    path = repo_path.replace("\\", "/")
    prefix = f"{model.folder}/" if model.folder else ""
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    parts = path.split("/")
    # Repo file names are untrusted: "embeddings/..\\..\\x" would land outside the farm.
    if any(p in ("", ".", "..") or ":" in p for p in parts):
        from core.inference.audio_cpp_server import AudioCppUnavailableError
        raise AudioCppUnavailableError(f"Refusing the model file path {repo_path!r}.")
    return Path(*parts)


def _served_path(model: AudioCppModel, farm: Path) -> Path:
    return farm if model.is_package else farm / _relative(model, model.variant.primary)


def materialize(model: AudioCppModel, *, hub_cache: Optional[Path] = None) -> str:
    """Path of the model for the server config: its GGUF under its real name, or a package's directory.

    Raises ``FileNotFoundError`` when the variant is not downloaded.
    """
    root = hub_cache if hub_cache is not None else _hub_cache()
    found = _find(model, root)
    if found is None:
        raise FileNotFoundError(f"{model.display_name} ({model.variant.key}) is not downloaded.")
    snapshot, files = found
    windows = sys.platform == "win32"
    if model.local_path:
        problem = served_path_problem(model, hub_cache = root)
        if problem:
            from core.inference.audio_cpp_server import AudioCppUnavailableError
            raise AudioCppUnavailableError(problem)
        return str(files[0])
    if not windows and not any(p.is_symlink() for p in files):
        base = snapshot / model.folder if model.folder else snapshot
        return str(base if model.is_package else snapshot / model.variant.primary)
    prune_link_farm(root)
    # One short directory per model and variant: the files keep their layout below the model's
    # folder (PocketTTS reads embeddings/ beside its GGUF, a package reads config/ and tokenizer/).
    farm_root = _link_farm_root(root)
    if _is_link(farm_root):
        _unlink_link(farm_root)
    farm = farm_root / model.key
    problem = served_path_problem(model, hub_cache = root)
    if problem:
        from core.inference.audio_cpp_server import AudioCppUnavailableError
        raise AudioCppUnavailableError(problem)
    for src, entry in zip(files, model.variant.files):
        rel = _relative(model, entry.path)
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
        try:
            dst.parent.mkdir(parents = True, exist_ok = True)
        except OSError as exc:
            from core.inference.audio_cpp_server import AudioCppUnavailableError
            raise AudioCppUnavailableError(
                f"The audio runtime needs a writable folder beside the Hugging Face cache ({farm_root}): "
                f"{exc}. Move the Hugging Face cache in Settings."
            ) from exc
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
    try:
        (farm / _SOURCE_MARKER).write_text(
            json.dumps(
                {
                    "repo_id": model.repo_id,
                    "files": [[f.path, f.size] for f in model.variant.files],
                }
            ),
            encoding = "utf-8",
        )
    except OSError as exc:
        logger.debug("audio.cpp: could not record the farm entry's source: %s", exc)
    return str(_served_path(model, farm))


def served_path_problem(model: AudioCppModel, *, hub_cache: Optional[Path] = None) -> Optional[str]:
    """Why the Windows server could not open the path ``materialize`` would hand it, or None. Checked
    before any download or eviction, so a refused model costs nothing."""
    if sys.platform != "win32":
        return None
    if model.local_path:
        longest = Path(model.local_path)
    else:
        root = hub_cache if hub_cache is not None else _hub_cache()
        farm = _link_farm_root(root) / model.key
        longest = max(
            (farm / _relative(model, f.path) for f in model.variant.files),
            key = lambda p: len(str(p)),
            default = _served_path(model, farm),
        )
    if len(str(longest)) <= _WINDOWS_MAX_MODEL_PATH:
        return None
    return (
        f"The path the audio runtime would load {model.display_name} from is {len(str(longest))} "
        f"characters, over the {_WINDOWS_MAX_MODEL_PATH} Windows allows. Move the Hugging Face cache "
        "to a shorter path in Settings and download the model again."
    )


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


def _source_still_cached(model_dir: Path, hub_cache: Path) -> bool:
    """Whether the repo files a farm entry mirrors are all still in the cache."""
    try:
        source = json.loads((model_dir / _SOURCE_MARKER).read_text(encoding = "utf-8"))
        repo_id = str(source["repo_id"])
        files = [(str(path), int(size)) for path, size in source["files"]]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not files:
        return False
    for snapshot in snapshot_dirs(repo_id, hub_cache):
        if all(_file_complete(snapshot / path, size) for path, size in files):
            return True
    return False


def prune_link_farm(hub_cache: Optional[Path] = None) -> int:
    """Drop farm entries whose files are no longer downloaded, so a deleted model frees its disk.

    A hardlink keeps the blob's data alive after the cache deletes it, and a copy is a second
    full copy, so either way the farm entry must go once the cache no longer holds the model.
    Entries without a source record (an earlier layout) go too.
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
            keep = _source_still_cached(model_dir, root)
            if not keep and not (model_dir / _SOURCE_MARKER).exists():
                # A materialize in flight writes its marker last; only an old entry lacks one for long.
                try:
                    if time.time() - model_dir.stat().st_mtime < 3600:
                        keep = True
                except OSError:
                    pass
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
