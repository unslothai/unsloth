# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

NON_CHAT_TASKS = frozenset(
    {
        "automatic-speech-recognition",
        "text-to-image",
        "text-to-video",
        "text-to-speech",
        "image-diffusion-unsupported",
    }
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


def _gguf_export_task(
    path: str,
    name: str,
    base_model: Optional[str] = None,
) -> Optional[str]:
    try:
        from hub.services.models.catalog_classification import _gguf_path_task
    except ImportError:
        return None
    return _gguf_path_task(path, (name, base_model))


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
        if export_type == "gguf" and _gguf_export_task(path, name, base) in NON_CHAT_TASKS:
            continue
        if export_type == "gguf":
            # No complete quant means every candidate is zero-byte or short a shard, which
            # survives resolve_model_config() and fails only at load time. Drop it only when
            # positively unloadable: a falsy result also means "could not tell", and hiding a
            # real export on a transient read is worse than listing it.
            load_id = _preferred_complete_gguf(path)
            if not load_id and not _local_dir_holds_a_payload(Path(path)):
                continue
        else:
            load_id = None
            # Positively loadable, not merely "not torn". scan_exported_models types a
            # checkpoint as "lora" on adapter_config.json alone, so an interrupted export
            # leaves a config and no weights: nothing torn because there is no payload, and
            # the picker offered a directory load_peft_weights raises ValueError on.
            if not _local_dir_holds_a_payload(Path(path)):
                continue
        entries.append(ModelEntry("Fine-tunes", name, export_type, load_id or path))
    return entries


def _pinned_snapshot(repo_path: Path, load_id: Optional[str]) -> Optional[Path]:
    """The snapshot a pinned row will open, when ``load_id`` names one inside this repo.

    A pin beats ``refs/main``, because the inventory pins exactly when the ref would resolve
    somewhere worse, so the ref names a revision this row never loads.
    """
    if not load_id:
        return None
    try:
        candidate = Path(load_id)
        snapshots = (repo_path / "snapshots").resolve(strict = False)
        for path in (candidate, *candidate.parents):
            if path.parent.resolve(strict = False) == snapshots and path.is_dir():
                return path
    except (OSError, ValueError):
        return None
    return None


def _reachable_snapshots(repo_path: Path, load_id: Optional[str] = None) -> List[Path]:
    """The snapshots a load through this row could actually open.

    The pinned one when the row carries a pin, else ``refs/main`` alone when it names one,
    since that is what a bare repo id resolves to and another revision's quants are files the
    selection cannot reach. A commit- or tag-pinned fetch leaves no ``refs/main``, and those
    rows are pinned rather than dropped, so they describe every snapshot instead.
    """
    snapshots = repo_path / "snapshots"
    try:
        available = sorted(path for path in snapshots.iterdir() if path.is_dir())
    except OSError:
        return []
    pinned = _pinned_snapshot(repo_path, load_id)
    if pinned is not None:
        # Returned as validated, not re-checked against `available`. inventory_scan resolves
        # the snapshot it pins while cache_path keeps the configured spelling, so under a
        # symlinked cache root the two name one directory and fail lexical equality, and the
        # membership test would drop a good pin back onto the ref it was pinned away from.
        return [pinned]
    try:
        # ValueError too: an undecodable ref raises UnicodeDecodeError, which is not an
        # OSError, and uncaught it leaves _safe hiding every Downloaded row over one repo.
        ref = (repo_path / "refs" / "main").read_text(encoding = "utf-8").strip()
    except (OSError, ValueError):
        ref = ""
    if ref:
        by_ref = snapshots / ref
        if by_ref in available:
            return [by_ref]
    return available


def _complete_quants(snapshot: Path) -> Optional[set]:
    """The quant keys under *snapshot* a load can resolve, or None if unknown.

    ``_preferred_complete_gguf`` narrows to this same set, so a label outside it names a
    quant selecting the row can never reach.
    """
    try:
        from hub.utils.inventory_scan import complete_snapshot_variants
        complete = complete_snapshot_variants(str(snapshot))
    except Exception:
        return None
    # Empty here means the check could not tell, not that nothing loads: a genuinely
    # incomplete repo arrives partial and never reaches the picker. Describe what is on disk.
    return set(complete) or None


def _quant_labels(
    repo_id: str,
    repo_path: str,
    load_id: Optional[str] = None,
) -> str:
    """The quants this cached GGUF repo offers, spelled the way Unsloth spells them.

    Through ``list_local_gguf_variants``, which is what ``_preferred_complete_gguf`` picks the
    load target with, so the label and the target cannot disagree. Globbing
    ``snapshots/*/*.gguf`` missed a per-quant subdirectory, gave a split quant one label per
    shard, and matched ``.GGUF`` only on Windows, where fnmatch normcases; the lister recurses,
    groups shard families, and tests the suffix case-insensitively.

    Scoped to the snapshot this row loads from and narrowed to the quants that snapshot can
    actually serve, so the detail never advertises a variant selecting the row cannot reach.

    Fails open to no detail, like every backend import here: a raise would leave ``_safe`` to
    swallow the whole cached source and every Downloaded row with it.
    """
    try:
        from hub.utils.gguf import list_local_gguf_variants
    except ImportError:
        return ""

    stem = repo_id.split("/")[-1].removesuffix("-GGUF").lower()
    quants = []
    for snapshot in _reachable_snapshots(Path(repo_path), load_id):
        try:
            variants, _ = list_local_gguf_variants(str(snapshot))
        except Exception:
            continue
        complete = _complete_quants(snapshot)
        for variant in variants:
            if complete is not None and variant.quant not in complete:
                continue
            label = variant.display_label or variant.quant
            if label.lower().startswith(stem + "-"):
                label = label[len(stem) + 1 :]
            if label and label not in quants:
                quants.append(label)
    return ", ".join(quants)


def _local_payload_is_torn(path: Path) -> bool:
    """Whether *path*'s own contents prove it cannot serve a load.

    Named apart from ``routes.models._snapshot_payload_is_torn``, which shares the rule but
    not the error policy: this one fails open, because ``_safe`` turns a raise here into an
    empty source rather than one dropped row.

    The shared rule the cache rows already use: a zero-byte weight, or a shard set naming a
    total it does not have, is not a payload. Fails open so an unreadable dir is not hidden.
    """
    try:
        from hub.utils.inventory_scan import _snapshot_cannot_serve_its_payload
    except ImportError:
        return False
    try:
        return bool(_snapshot_cannot_serve_its_payload(path))
    except (OSError, RuntimeError, ValueError):
        return False


def _gguf_file_is_loadable(path: Path) -> bool:
    """Whether a single-file GGUF can be read: non-empty, and whole if it names a shard total.

    ``_snapshot_cannot_serve_its_payload`` judges a directory; a scan row can name the ``.gguf``
    itself, and its folder may hold unrelated quants, so the file is judged on its own family.
    """
    try:
        if path.stat().st_size <= 0:
            return False
    except OSError:
        return True
    try:
        from hub.utils.inventory_scan import _GGUF_SPLIT_RE
    except ImportError:
        return True
    split = _GGUF_SPLIT_RE.search(path.name)
    if split is None:
        return True
    total = int(split.group(2))
    prefix = path.name[: split.start()]
    if int(split.group(1)) <= 0 or total <= 0 or int(split.group(1)) > total:
        return False
    present = set()
    try:
        for sibling in path.parent.iterdir():
            match = _GGUF_SPLIT_RE.search(sibling.name)
            if (
                match is None
                or sibling.name[: match.start()] != prefix
                or int(match.group(2)) != total
                or not sibling.is_file()
                or sibling.stat().st_size <= 0
            ):
                continue
            present.add(int(match.group(1)))
    except OSError:
        return True
    return present >= set(range(1, total + 1))


def _preferred_complete_gguf(path: str) -> Optional[str]:
    model_path = Path(path)
    try:
        root = model_path.parent if model_path.is_file() else model_path
        if not root.is_dir():
            return None
        from hub.utils.gguf import list_local_gguf_variants, pick_best_gguf
        from hub.utils.inventory_scan import complete_snapshot_variants

        variants, _ = list_local_gguf_variants(str(root))
        complete = complete_snapshot_variants(str(root))
        ready = [variant for variant in variants if variant.quant in complete]
        root_variants = [variant for variant in ready if "/" not in variant.quant]
        best = pick_best_gguf([variant.filename for variant in root_variants or ready])
        candidate = root / best if best else None
        return str(candidate) if candidate is not None and candidate.is_file() else None
    except Exception:
        return None


def _cached_gguf_load_id(row: dict) -> str:
    """A concrete local GGUF for a cached row, falling back to the identifier it came with.

    An inactive-cache row already carries a snapshot path, so this resolves the quant inside it.
    An ACTIVE-cache row carries the bare repo id, which is right for the inventory and wrong
    for a picker: from_identifier() sends a non-path through detect_gguf_model_remote(), whose
    GatedRepoError branch is permanent and returns None without the offline local-cache
    fallback, so a fully downloaded gated GGUF reads as non-GGUF and hits the Transformers
    loader. Resolving the on-disk file keeps the pick on the weights the row was listed for.
    """
    load_id = row.get("load_id") or row["repo_id"]
    resolved = _preferred_complete_gguf(load_id)
    if resolved:
        return resolved
    # The bare repo id is not a path, so resolve through the snapshot the row was built from.
    cache_path = row.get("cache_path")
    if cache_path:
        try:
            from hub.utils.inventory_scan import default_ref_snapshot
            snapshot = default_ref_snapshot(Path(cache_path))
        except Exception:
            snapshot = None
        if snapshot is None:
            snapshots = sorted(Path(cache_path).glob("snapshots/*"))
            snapshot = snapshots[-1] if snapshots else None
        if snapshot is not None:
            resolved = _preferred_complete_gguf(str(snapshot))
            if resolved:
                return resolved
    return load_id


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
        if (
            row.get("partial")
            or row.get("task") in NON_CHAT_TASKS
            or row.get("capabilities", {}).get("can_chat") is False
        ):
            continue
        entries.append(
            ModelEntry(
                "Downloaded",
                row["repo_id"],
                _quant_labels(row["repo_id"], row["cache_path"], row.get("load_id")),
                _cached_gguf_load_id(row),
            )
        )
    for row in model_rows:
        if row.get("partial") or row.get("companion") or row.get("task") in NON_CHAT_TASKS:
            continue
        # A cached embedding/CLIP repo has task None like any chat repo; can_chat is the gate.
        if row.get("capabilities", {}).get("can_chat") is False:
            continue
        # A diffusion repo also carries no task, and its pipeline root has no config for
        # can_chat to read, so neither gate above catches it.
        if row.get("diffusers"):
            continue
        entries.append(ModelEntry("Downloaded", row["repo_id"], "", _cached_model_load_id(row)))
    return entries


def _cached_model_load_id(row: dict) -> str:
    """Load target for a cached non-GGUF row; a local snapshot for adapters.

    An ACTIVE-cache adapter carries the bare repo id, and a LoRA resolved by id takes the
    remote branch of get_base_model_from_lora_identifier, which downloads adapter_config.json
    with no local_files_only. For a cached gated adapter with no token that fails and the base
    is never read, so a row advertised as local yields no model configuration. A snapshot path
    makes is_local_path true and takes the local reader instead.

    Only adapters, deliberately: pinning an ordinary cached repo would freeze a live repo the
    inventory chose not to pin.
    """
    load_id = row.get("load_id") or row["repo_id"]
    if row.get("model_format") != "adapter" or os.path.isabs(str(load_id)):
        return load_id
    cache_path = row.get("cache_path")
    if not cache_path:
        return load_id
    try:
        from hub.utils.inventory_scan import default_ref_snapshot
        snapshot = default_ref_snapshot(Path(cache_path))
    except Exception:
        snapshot = None
    if snapshot is None:
        snapshots = sorted(Path(cache_path).glob("snapshots/*"))
        snapshot = snapshots[-1] if snapshots else None
    return str(snapshot) if snapshot is not None else load_id


def _local_dir_holds_a_payload(path: Path) -> bool:
    """Whether a scanned local directory holds something a load can actually read.

    ``_scan_models_dir`` lists a child on a config alone, so a weights-less config dir
    arrives with ``partial`` False and the picker would offer a folder that fails on
    selection. Anything that is not a readable directory fails open: a single-file GGUF
    is a file, not a dir.
    """
    try:
        is_dir = path.is_dir()
    except OSError:
        return True
    if not is_dir:
        # A scan row can name the .gguf file itself; anything else still fails open.
        return _gguf_file_is_loadable(path) if path.suffix.lower() == ".gguf" else True
    from hub.services.models.common import (
        _is_diffusers_pipeline_dir,
        _is_main_gguf_filename,
        _is_model_directory,
    )
    from utils.paths.path_utils import is_appledouble_metadata

    # A pipeline keeps its weights in component subdirs, so the torn test below reads an
    # empty root and would call every pipeline unserviceable.
    if _is_diffusers_pipeline_dir(path):
        return True
    if _local_payload_is_torn(path):
        return False
    # iterdir, not glob("*.gguf"): the glob is case-sensitive on Linux and macOS while
    # _is_main_gguf_filename lowercases first, so a folder holding Model.GGUF was classified
    # as a GGUF model and then dropped by this gate, which decides whether it is listed.
    try:
        children = list(path.iterdir())
    except OSError:
        return True
    return any(
        _is_main_gguf_filename(file.name) and not is_appledouble_metadata(file) for file in children
    ) or _is_model_directory(path)


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


def _registered_custom_hf_cache(model) -> bool:
    if model.source != "hf_cache":
        return False
    try:
        from hub.storage.scan_folders import list_scan_folders
        from hub.utils.hf_cache_state import hf_cache_roots
        from hub.utils.paths import path_is_same_or_child

        model_path = Path(model.path)
        repo_dir = next(
            (
                path
                for path in (model_path, *model_path.parents)
                if path.name.startswith("models--")
            ),
            None,
        )
        if repo_dir is None:
            return False
        cache_root = repo_dir.parent

        def same_path(left: Path, right: Path) -> bool:
            return path_is_same_or_child(left, right) and path_is_same_or_child(right, left)

        if any(same_path(cache_root, root) for root in hf_cache_roots()):
            return False
        return any(
            same_path(cache_root, Path(folder["path"]).expanduser())
            for folder in list_scan_folders()
            if folder.get("path")
        )
    except Exception:
        return False


def local_folder_entries() -> List[ModelEntry]:
    entries = []
    for model in _local_catalog_rows():
        if (
            model.source not in LOCAL_SOURCES and not _registered_custom_hf_cache(model)
        ) or model.partial:
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
        target = model.load_id or model.id
        # A GGUF DIRECTORY goes through detect_gguf_model, which sorts by size and takes the
        # largest complete file, commonly the F16, while cached and exported rows resolve a
        # Q4-class quant. Same folder, dramatically bigger load by source alone, so an OOM
        # rather than a preference. Resolve it the same way here.
        if is_gguf:
            target = _preferred_complete_gguf(target) or target
        entries.append(
            ModelEntry(
                "Downloaded",
                model.display_name,
                "gguf" if is_gguf else "",
                target,
            )
        )
    return entries


def _safe(fn) -> List[ModelEntry]:
    """Run one source, and never let it take the other three down with it.

    Fails open, since a picker missing one group beats a traceback where a model list should
    be, but not SILENTLY: a source is every downloaded model or every fine-tune, so swallowing
    the reason turns "my HF cache is unreadable" into an empty list. UNSLOTH_DEBUG re-raises.
    """
    try:
        return fn()
    except Exception as error:
        # Same truthy set the rest of the tree uses (utils.utils, utils.transformers_version).
        if os.environ.get("UNSLOTH_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
            raise
        # Not at module scope: this module must pull in nothing beyond the inventory layer
        # (test_catalog_inventory_works_without_fastapi_or_routes).
        import typer

        typer.echo(
            f"Could not read one model source ({fn.__name__}): {type(error).__name__}: {error}. "
            f"Set UNSLOTH_DEBUG=1 for the traceback.",
            err = True,
        )
        return []


def _shorten_names(entries: List[ModelEntry]) -> None:
    # Counted over EVERY visible label, not just the prefixed ones: a fine-tune already named
    # Qwen3-0.6B is what an unsloth/Qwen3-0.6B row would collide with once shortened.
    counts = {}
    for entry in entries:
        short = entry.name.split("/")[-1].lower()
        counts[short] = counts.get(short, 0) + 1
    for entry in entries:
        if "/" not in entry.name:
            continue
        short = entry.name.split("/")[-1]
        if counts[short.lower()] == 1:
            entry.name = short


def _dedup_key(model: str):
    """Identity of a load target, for collapsing the same model reached by two sources.

    For anything that exists on disk the filesystem is ASKED rather than guessed: two paths
    are the same model when they are the same inode. That is the only rule that holds across
    the three cases that actually differ, and no platform test gets all three right --
    ``/models/Foo`` and ``/models/foo`` are two models on ext4, one model on NTFS, and one
    model on a stock case-insensitive APFS volume, where ``os.path.normcase`` is identity and
    would wrongly have shown the user the same model twice. It also collapses a path reached
    once directly and once through a symlink, which the old key could not see at all.

    Falling back on a string is only for what cannot be stat'd: a bare repo id, or a path that
    has gone away since the scan. A repo id keeps the case-insensitive treatment, since the Hub
    resolves ``Unsloth/Qwen3`` and ``unsloth/qwen3`` to one repo and the cache folds them into
    one directory; a vanished path uses ``normcase``, which at worst leaves two rows for a model
    that is not loadable from either of them.
    """
    try:
        stat = os.stat(model)
        return (stat.st_dev, stat.st_ino)
    except (OSError, ValueError):
        return os.path.normcase(model) if os.path.isabs(model) else model.lower()


def list_chat_models() -> List[ModelEntry]:
    entries = []
    for fn in (trained_entries, exported_entries, cached_entries, local_folder_entries):
        entries.extend(_safe(fn))
    seen = set()
    unique = []
    for entry in entries:
        key = _dedup_key(entry.model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    _shorten_names(unique)
    return unique
