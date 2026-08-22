# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

NON_CHAT_TASKS = frozenset(
    {"text-to-image", "text-to-video", "text-to-speech", "image-diffusion-unsupported"}
)
LOCAL_SOURCES = frozenset({"models_dir", "lmstudio", "custom"})


@dataclass
class ModelEntry:
    group: str
    name: str
    detail: str
    model: str


def _run_title(run: dict) -> str:
    for key in ("display_name", "project_name", "model_name"):
        value = (run.get(key) or "").strip()
        if value:
            return value.split("/")[-1]
    return ""


def _folder_title(folder: str) -> str:
    name = folder.removeprefix("unsloth_")
    stem, _, suffix = name.rpartition("_")
    return stem if stem and suffix.isdigit() else name


def _short_date(value) -> str:
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value)
        else:
            dt = datetime.fromisoformat(str(value)).astimezone()
        return dt.strftime("%b %d").replace(" 0", " ")
    except Exception:
        return ""


def _dataset_title(dataset: Optional[str]) -> str:
    if not dataset:
        return ""
    base = os.path.basename(dataset.rstrip("/"))
    head, sep, tail = base.partition("_")
    if sep and len(head) == 32 and all(c in "0123456789abcdef" for c in head):
        base = tail
    return base.removesuffix(".jsonl").removesuffix(".json").removesuffix(".csv")


def _run_detail(model_type: str, path: str, run: Optional[dict]) -> str:
    parts = [] if model_type == "lora" else [model_type]
    if run:
        parts.append(_dataset_title(run.get("dataset_name")))
        if run.get("final_step"):
            parts.append(f"{run['final_step']} steps")
        parts.append(_short_date(run.get("started_at")))
    else:
        try:
            parts.append(_short_date(Path(path).stat().st_mtime))
        except OSError:
            pass
    return " · ".join(p for p in parts if p)


def _runs_by_output_dir() -> dict:
    try:
        from storage.studio_db import list_runs
        runs = list_runs(limit = 1000)["runs"]
    except Exception:
        return {}
    by_dir = {}
    for run in sorted(runs, key = lambda r: r.get("started_at") or ""):
        if run.get("output_dir"):
            by_dir[os.path.normpath(run["output_dir"])] = run
    return by_dir


def _path_can_chat(path: str, base_model: Optional[str] = None) -> Optional[bool]:
    """``False`` only for a locally identifiable non-chat checkpoint, else ``None``."""
    try:
        from hub.services.models.common import _local_path_can_chat
    except ImportError:
        return None
    return _local_path_can_chat(path, base_model)


def _gguf_export_task(path: str, name: str) -> Optional[str]:
    try:
        from hub.services.models.catalog_classification import _gguf_path_task
    except ImportError:
        return None
    return _gguf_path_task(path, (name,))


def trained_entries() -> List[ModelEntry]:
    from utils.models import scan_trained_models

    runs = _runs_by_output_dir()
    entries = []
    for folder, path, model_type in scan_trained_models():
        run = runs.get(os.path.normpath(path))
        base = run.get("model_name") if run and model_type == "lora" else None
        if _path_can_chat(path, base) is False:
            continue
        name = (_run_title(run) if run else "") or _folder_title(folder)
        entries.append(ModelEntry("Fine-tunes", name, _run_detail(model_type, path, run), path))
    return entries


def exported_entries() -> List[ModelEntry]:
    from utils.models import scan_exported_models

    entries = []
    for name, path, export_type, base in scan_exported_models():
        if _path_can_chat(path, base if export_type == "lora" else None) is False:
            continue
        if export_type == "gguf" and _gguf_export_task(path, name) in NON_CHAT_TASKS:
            continue
        entries.append(ModelEntry("Exports", name, export_type, path))
    return entries


def _quant_labels(repo_id: str, repo_path: str) -> str:
    from utils.models.model_config import _is_mmproj

    stem = repo_id.split("/")[-1].removesuffix("-GGUF").lower()
    quants = []
    for file in sorted(Path(repo_path).glob("snapshots/*/*.gguf")):
        if _is_mmproj(file.name) or file.name.lower().startswith("mtp-"):
            continue
        label = file.stem
        if label.lower().startswith(stem + "-"):
            label = label[len(stem) + 1 :]
        if label not in quants:
            quants.append(label)
    return ", ".join(quants)


def _cached_catalog_rows() -> tuple[list[dict], list[dict]]:
    from hub.services.models.cache_inventory import (
        _scan_cached_gguf,
        _scan_cached_models,
        all_hf_cache_scans,
    )
    from utils.hf_cache_settings import get_hf_cache_paths

    scans = all_hf_cache_scans()
    active_hub_cache = get_hf_cache_paths().hub_cache

    def newest_first(rows):
        return sorted(
            rows,
            key = lambda row: (-(row.get("last_modified") or 0.0), row["repo_id"].lower()),
        )

    gguf_rows = _scan_cached_gguf(cache_scans = scans, active_hub_cache = active_hub_cache)
    model_rows = _scan_cached_models(cache_scans = scans, active_hub_cache = active_hub_cache)
    return newest_first(gguf_rows), newest_first(model_rows)


def cached_entries() -> List[ModelEntry]:
    gguf_rows, model_rows = _cached_catalog_rows()
    entries = []
    for row in gguf_rows:
        if row.get("partial") or row.get("task") in NON_CHAT_TASKS:
            continue
        entries.append(
            ModelEntry(
                "Downloaded",
                row["repo_id"],
                _quant_labels(row["repo_id"], row["cache_path"]),
                row.get("load_id") or row["repo_id"],
            )
        )
    for row in model_rows:
        if row.get("partial") or row.get("companion") or row.get("task") in NON_CHAT_TASKS:
            continue
        # A cached embedding/CLIP repo has task None like any chat repo; can_chat is the gate.
        if row.get("capabilities", {}).get("can_chat") is False:
            continue
        # An untrusted or unrecognised diffusion repo carries no task either, and its pipeline
        # root has no config for can_chat to read, so neither of the gates above catches it.
        if row.get("diffusers"):
            continue
        entries.append(
            ModelEntry("Downloaded", row["repo_id"], "", row.get("load_id") or row["repo_id"])
        )
    return entries


def _local_dir_holds_a_payload(path: Path) -> bool:
    """Whether a scanned local directory holds something a load can actually read.

    ``_scan_models_dir`` lists a child on a config alone, so a weights-less config dir
    arrives with ``partial`` False and the picker would offer a folder that fails on
    selection. Anything that is not a readable directory fails open: a single-file GGUF
    is a file, not a dir.
    """
    try:
        if not path.is_dir():
            return True
    except OSError:
        return True
    from hub.services.models.common import (
        _is_diffusers_pipeline_dir,
        _is_main_gguf_filename,
        _is_model_directory,
    )
    from utils.paths.path_utils import is_appledouble_metadata

    return (
        any(
            _is_main_gguf_filename(file.name) and not is_appledouble_metadata(file)
            for file in path.glob("*.gguf")
        )
        or _is_model_directory(path)
        or _is_diffusers_pipeline_dir(path)
    )


def _local_is_a_diffusers_pipeline(model) -> bool:
    """Whether a local row is a diffusers pipeline, which never answers a chat turn.

    An unrecognised pipeline gets no task and no root config for can_chat to read, so both
    gates pass it while the payload check accepts its pipeline index. The cached path says
    this with the row's ``diffusers`` flag; this is the local twin.
    """
    from hub.services.models.catalog_classification import _local_is_diffusers
    return bool(_local_is_diffusers(model))


def _local_model_task(model) -> Optional[str]:
    from hub.services.models.catalog_classification import _local_model_task as classify
    return classify(model)


def _local_model_can_chat(model) -> Optional[bool]:
    from hub.services.models.catalog_classification import _local_model_can_chat as classify
    return classify(model)


def _local_catalog_rows():
    from hub.services.models.local_inventory import list_local_models_response
    response = asyncio.run(list_local_models_response(str(Path("./models").resolve())))
    return response.models


def local_folder_entries() -> List[ModelEntry]:
    entries = []
    for model in _local_catalog_rows():
        if model.source not in LOCAL_SOURCES or model.partial:
            continue
        is_gguf = model.model_format == "gguf" or model.path.lower().endswith(".gguf")
        # No format gate: _dir_model_format reports only "gguf" or None, so a safetensors
        # checkpoint arrives as None and a "safetensors" literal dropped every non-GGUF model.
        if _local_model_task(model) in NON_CHAT_TASKS:
            continue
        # No format gate, so embedding and CLIP exports get through; only this stops them.
        if _local_model_can_chat(model) is False:
            continue
        if not _local_dir_holds_a_payload(Path(model.path)):
            continue
        if _local_is_a_diffusers_pipeline(model):
            continue
        entries.append(
            ModelEntry(
                "Local folders",
                model.display_name,
                "gguf" if is_gguf else "",
                model.load_id or model.id,
            )
        )
    return entries


def _safe(fn) -> List[ModelEntry]:
    try:
        return fn()
    except Exception:
        return []


def _shorten_names(entries: List[ModelEntry]) -> None:
    repos = [e for e in entries if "/" in e.name]
    counts = {}
    for entry in repos:
        short = entry.name.split("/")[-1].lower()
        counts[short] = counts.get(short, 0) + 1
    for entry in repos:
        short = entry.name.split("/")[-1]
        if counts[short.lower()] == 1:
            entry.name = short


def list_chat_models() -> List[ModelEntry]:
    entries = []
    for fn in (trained_entries, exported_entries, cached_entries, local_folder_entries):
        entries.extend(_safe(fn))
    seen = set()
    unique = []
    for entry in entries:
        key = entry.model.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    _shorten_names(unique)
    return unique
