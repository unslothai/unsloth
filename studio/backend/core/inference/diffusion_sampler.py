# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sampler (scheduler) choice for the diffusers image engine, plus the v-prediction markers a
single-file SDXL checkpoint carries. Names follow AUTOMATIC1111; the mapping is the diffusers
scheduler overview table. No torch/diffusers imports at module level."""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import Any, Optional

DEFAULT_SAMPLER = "default"
AUTOMATIC_SCHEDULE = "automatic"

# id -> (label, diffusers class, constructor overrides)
SAMPLERS: dict[str, tuple[str, Optional[str], dict[str, Any]]] = {
    DEFAULT_SAMPLER: ("Model default", None, {}),
    "euler": ("Euler", "EulerDiscreteScheduler", {}),
    "euler_a": ("Euler a", "EulerAncestralDiscreteScheduler", {}),
    "dpmpp_2m": (
        "DPM++ 2M",
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "solver_order": 2},
    ),
    "dpmpp_2m_sde": (
        "DPM++ 2M SDE",
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
    ),
}

SCHEDULE_TYPES: dict[str, tuple[str, Optional[str]]] = {
    AUTOMATIC_SCHEDULE: ("Automatic", None),
    "uniform": ("Uniform", None),
    "karras": ("Karras", "use_karras_sigmas"),
    "exponential": ("Exponential", "use_exponential_sigmas"),
}
_SIGMA_FLAGS = ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "use_lu_lambdas")

# AUTOMATIC1111's per-sampler default schedule (modules/sd_samplers_kdiffusion.py); absent = keep the config's.
_AUTOMATIC_FOR = {"dpmpp_2m": "karras", "dpmpp_2m_sde": "exponential"}


def _init_params(cls: type) -> frozenset[str]:
    import inspect
    try:
        return frozenset(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _class_for(sampler: str, resident: Any) -> Optional[type]:
    cls_name = SAMPLERS[sampler][1]
    if cls_name is None:
        return type(resident)
    import diffusers

    return getattr(diffusers, cls_name, None)


def _schedule_types_for(cls: type) -> list[str]:
    params = _init_params(cls)
    return [sid for sid, (_label, flag) in SCHEDULE_TYPES.items() if flag is None or flag in params]


@lru_cache(maxsize = 32)
def _options_for(scheduler_cls: type) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    compatible = {
        c if isinstance(c, str) else getattr(c, "__name__", "")
        for c in (getattr(scheduler_cls, "_compatibles", None) or ())
    }
    if not compatible:
        return ()
    import diffusers

    options = []
    for sid, (label, cls_name, _overrides) in SAMPLERS.items():
        if cls_name is not None and cls_name not in compatible:
            continue
        cls = scheduler_cls if cls_name is None else getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        options.append((sid, label, tuple(_schedule_types_for(cls))))
    return tuple(options) if len(options) > 1 else ()


def sampler_options(scheduler: Any) -> list[dict[str, Any]]:
    """Samplers the resident scheduler can be swapped for, each with the schedule types it takes.

    Empty when the scheduler lists no compatible classes (every flow-matching model), so the
    control is absent there rather than offering choices that cannot run. Read on every status poll,
    hence cached per class."""
    config = getattr(scheduler, "config", None) or {}
    # A DPM / UniPC scheduler driving a flow model lists the epsilon samplers as compatible too.
    if (
        scheduler is None
        or config.get("use_flow_sigmas")
        or config.get("prediction_type") == "flow_prediction"
    ):
        return []
    return [
        {"id": sid, "label": label, "schedule_types": list(types)}
        for sid, label, types in _options_for(type(scheduler))
    ]


def resolve_schedule(
    sampler: Optional[str],
    schedule_type: Optional[str],
    config: Any = None,
) -> Optional[str]:
    """The schedule a request runs: ``automatic`` resolves to the sampler's A1111 default, None = config's."""
    schedule_type = schedule_type or AUTOMATIC_SCHEDULE
    if schedule_type != AUTOMATIC_SCHEDULE:
        return schedule_type
    resolved = _AUTOMATIC_FOR.get(sampler or DEFAULT_SAMPLER)
    # Exponential spacing down from a zero-SNR sigma_max (~4096) washes out detail (NoobAI-XL V-Pred 1.0).
    if resolved == "exponential" and (config or {}).get("rescale_betas_zero_snr"):
        return "karras"
    return resolved


def build_scheduler(
    resident: Any, sampler: Optional[str], schedule_type: Optional[str]
) -> tuple[Optional[Any], Optional[str]]:
    """``(scheduler, resolved schedule)`` for a request; scheduler None keeps ``resident`` as is.

    Built from the resident config, so prediction type and zero-SNR settings carry over. Raises
    ValueError for a choice this model does not offer."""
    sampler = sampler or DEFAULT_SAMPLER
    schedule_type = schedule_type or AUTOMATIC_SCHEDULE
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{sampler}'.")
    if schedule_type not in SCHEDULE_TYPES:
        raise ValueError(f"Unknown schedule type '{schedule_type}'.")
    if sampler == DEFAULT_SAMPLER and schedule_type == AUTOMATIC_SCHEDULE:
        return None, None
    options = {o["id"]: o for o in sampler_options(resident)}
    if sampler not in options:
        raise ValueError(
            f"The {SAMPLERS[sampler][0]} sampler is not available for the loaded model."
        )
    if schedule_type not in options[sampler]["schedule_types"]:
        raise ValueError(
            f"The {SCHEDULE_TYPES[schedule_type][0]} schedule type does not apply to the "
            f"{SAMPLERS[sampler][0]} sampler."
        )
    cls = _class_for(sampler, resident)
    params = _init_params(cls)
    overrides = {k: v for k, v in SAMPLERS[sampler][2].items() if k in params}
    resolved = resolve_schedule(sampler, schedule_type, resident.config)
    if resolved is not None:
        flag = SCHEDULE_TYPES[resolved][1]
        for name in _SIGMA_FLAGS:
            if name in params:
                overrides[name] = name == flag
    built = cls.from_config(resident.config, **overrides)
    set_begin_index = getattr(built, "set_begin_index", None)
    if callable(set_begin_index):
        # Zero-SNR Karras grids repeat 999; without a begin index (SDXL ControlNet sets none) DPM++ overruns.
        set_timesteps = built.set_timesteps

        @wraps(set_timesteps)
        def _set_timesteps(*args: Any, **kwargs: Any) -> Any:
            out = set_timesteps(*args, **kwargs)
            set_begin_index(0)
            return out

        built.set_timesteps = _set_timesteps
    return built, resolved


def sampler_label(sampler: Optional[str]) -> Optional[str]:
    entry = SAMPLERS.get(sampler or "")
    return entry[0] if entry and sampler != DEFAULT_SAMPLER else None


def schedule_label(schedule_type: Optional[str]) -> Optional[str]:
    entry = SCHEDULE_TYPES.get(schedule_type or "")
    return entry[0] if entry and schedule_type != AUTOMATIC_SCHEDULE else None


def _safetensors_top_level_keys(path: str) -> frozenset[str]:
    import json
    import struct

    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        if size <= 0 or size > 100_000_000:
            return frozenset()
        header = json.loads(fh.read(size))
    return frozenset(k for k in header if "." not in k)


def checkpoint_prediction_overrides(path: Optional[str]) -> dict[str, Any]:
    """Scheduler overrides a single-file checkpoint asks for through its ``v_pred`` / ``ztsnr`` marker
    tensors (ComfyUI's detection, comfy/supported_models.py SDXL.model_type). Empty when unmarked."""
    if not path or not str(path).lower().endswith(".safetensors"):
        return {}
    try:
        keys = _safetensors_top_level_keys(str(path))
    except Exception:  # noqa: BLE001 - an unreadable header keeps the shipped scheduler
        return {}
    return _marker_overrides(keys)


def _marker_overrides(keys: Any) -> dict[str, Any]:
    if "v_pred" not in keys:
        return {}
    overrides: dict[str, Any] = {"prediction_type": "v_prediction"}
    if "ztsnr" in keys:
        # Zero terminal SNR samples from the last timestep (arXiv 2305.08891, section 3.3).
        overrides.update(rescale_betas_zero_snr = True, timestep_spacing = "trailing")
    return overrides


# Hub marker reads cached per (repo, snapshot commit) so later offline loads sample the same way.
_MARKS_FILE = "diffusion_prediction_markers.json"
_MAX_RECORDED_MARKS = 512


def _marks_record_path() -> Any:
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / _MARKS_FILE
    except Exception:  # noqa: BLE001 - no Studio cache root: nothing is recorded
        return None


def _snapshot_commit(root: Optional[str], repo_id: str, revision: Optional[str]) -> Optional[str]:
    """Snapshot commit from a ``snapshots/<commit>`` path, else the cached ``model_index.json``."""
    from pathlib import Path

    try:
        if root:
            folder = Path(str(root))
            if folder.parent.name == "snapshots" and folder.name:
                return folder.name
        from huggingface_hub import try_to_load_from_cache

        from utils.hf_cache_settings import active_hf_hub_cache

        hit = try_to_load_from_cache(
            repo_id,
            "model_index.json",
            cache_dir = active_hf_hub_cache(),
            revision = revision,
        )
    except Exception:  # noqa: BLE001 - no cache view: nothing to key on
        return None
    return Path(hit).parent.name if isinstance(hit, str) else None


def _marks_key(repo_id: str, commit: str) -> str:
    return f"{repo_id.strip().lower()}@{commit}"


def _recorded_marks(repo_id: str, commit: Optional[str]) -> list[dict[str, Any]]:
    import json

    path = _marks_record_path()
    if not commit or path is None:
        return []
    try:
        hit = json.loads(path.read_text(encoding = "utf-8")).get(_marks_key(repo_id, commit))
    except Exception:  # noqa: BLE001 - absent / unreadable record: nothing recorded
        return []
    if not isinstance(hit, list) or not all(_well_formed_marks(m) for m in hit):
        return []
    return [_marker_overrides(set(m.get("markers") or ())) for m in hit]


def _well_formed_marks(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    markers = entry.get("markers")
    return markers is None or (
        isinstance(markers, list) and all(isinstance(m, str) for m in markers)
    )


def _record_marks(repo_id: str, commit: Optional[str], markers: list[list[str]]) -> None:
    import contextlib
    import json
    import os
    import threading

    path = _marks_record_path()
    if not commit or path is None:
        return
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        try:
            from filelock import FileLock
            lock: Any = FileLock(f"{path}.lock", timeout = 30)
        except ImportError:
            lock = contextlib.nullcontext()
        # One read / evict / write under a cross-process lock, so concurrent probes never drop entries.
        with lock:
            try:
                record = json.loads(path.read_text(encoding = "utf-8"))
                if not isinstance(record, dict):
                    record = {}
            except Exception:  # noqa: BLE001 - absent / corrupt: start over
                record = {}
            key = _marks_key(repo_id, commit)
            record.pop(key, None)
            while len(record) >= _MAX_RECORDED_MARKS:
                record.pop(next(iter(record)))
            record[key] = [{"markers": sorted(m)} for m in markers]
            tmp.write_text(json.dumps(record), encoding = "utf-8")
            os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - read-only cache / lock timeout: the next offline load lacks the markers
        try:
            tmp.unlink()
        except OSError:
            pass


def _root_single_marks(
    root: Optional[str],
    repo_id: Optional[str] = None,
    hf_token: Any = None,
    allow_network: bool = False,
) -> list[dict[str, Any]]:
    """Marker overrides of each root ``.safetensors`` of a pipeline: off disk when the folder holds them,
    else off the Hub headers of ``repo_id``. A staged Hub pipeline snapshot keeps only
    ``model_index.json`` at its root (the packaged single file is never downloaded), so the folder
    alone would miss the markers. Header range reads only; empty on anything unreadable."""
    from pathlib import Path

    try:
        folder = Path(str(root)) if root else None
        singles = (
            sorted(
                p for p in folder.iterdir() if p.suffix.lower() == ".safetensors" and p.is_file()
            )
            if folder is not None and folder.is_dir()
            else []
        )
    except OSError:
        singles = []
    if singles:
        return [checkpoint_prediction_overrides(str(p)) for p in singles]
    rid = (repo_id or "").strip()
    if rid.count("/") != 1:
        return []
    try:
        if Path(rid).expanduser().exists():
            return []
        from .diffusion_families import pinned_revision_kwargs

        rev = pinned_revision_kwargs(rid)
        commit = _snapshot_commit(root, rid, rev.get("revision"))
    except Exception:  # noqa: BLE001 - odd id: the shipped scheduler config stands
        return []
    if not allow_network:
        # A cache-only load of a snapshot an online load already read keeps that read's verdict.
        return _recorded_marks(rid, commit)
    if commit:
        # Read the snapshot the record is keyed on, never a HEAD that moved past the staged copy.
        rev = {"revision": commit}
    try:
        from huggingface_hub import HfApi

        from hub.utils.hf_tokens import normalize_token

        # One normalisation for every Hub call here: False (anonymous) never becomes None (ambient).
        hf_token = normalize_token(hf_token)
        api = HfApi()
        names = sorted(
            f
            for f in api.list_repo_files(rid, token = hf_token, **rev)
            if "/" not in f and f.lower().endswith(".safetensors")
        )
        markers = [
            sorted(
                k
                for k in api.parse_safetensors_file_metadata(
                    rid, name, token = hf_token, **rev
                ).tensors
                if k in ("v_pred", "ztsnr")
            )
            for name in names
        ]
    except Exception:  # noqa: BLE001 - offline / gated / odd: an earlier read of this snapshot, else the config
        return _recorded_marks(rid, commit)
    _record_marks(rid, commit, markers)
    return [_marker_overrides(set(m)) for m in markers]


def pipeline_prediction_overrides(
    config: Any,
    root: Optional[str],
    repo_id: Optional[str] = None,
    hf_token: Any = None,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Scheduler overrides a diffusers SDXL pipeline directory asks for. Empty when it asks for none.

    Two sources. The repo's packaged root single file: NoobAI-XL V-Pred 1.0 ships an epsilon
    ``scheduler_config.json`` beside a root ``.safetensors`` carrying ``v_pred`` / ``ztsnr``, which is
    the file ComfyUI loads and samples as v-prediction with zero terminal SNR. Only when every root
    single file agrees, so a repo shipping an eps and a v-pred variant side by side keeps its config.
    And a config that declares v-prediction with ``rescale_betas_zero_snr`` but leading spacing,
    which never samples the terminal timestep that zero-SNR training exists for (the marker path
    sets ``trailing`` for the same reason)."""
    overrides: dict[str, Any] = {}
    marks = _root_single_marks(root, repo_id, hf_token, allow_network)
    if marks and marks[0] and all(m == marks[0] for m in marks):
        overrides.update(marks[0])
    config = config or {}
    prediction = overrides.get("prediction_type", config.get("prediction_type"))
    zsnr = overrides.get("rescale_betas_zero_snr", config.get("rescale_betas_zero_snr"))
    if prediction == "v_prediction" and zsnr and config.get("timestep_spacing") != "trailing":
        overrides["timestep_spacing"] = "trailing"
    return overrides


def _apply_scheduler_overrides(
    pipe: Any, overrides: dict[str, Any], logger: Any, source: str
) -> bool:
    scheduler = getattr(pipe, "scheduler", None)
    config = getattr(scheduler, "config", None)
    if not overrides or config is None:
        return False
    params = _init_params(type(scheduler))
    overrides = {k: v for k, v in overrides.items() if k in params}
    if all(config.get(k) == v for k, v in overrides.items()):
        return False
    pipe.scheduler = type(scheduler).from_config(config, **overrides)
    if logger is not None:
        logger.info("scheduler set from %s: %s", source, overrides)
    return True


def apply_checkpoint_prediction_type(
    pipe: Any,
    path: Optional[str],
    logger: Any = None,
) -> bool:
    """Rebuild ``pipe.scheduler`` for a v-prediction checkpoint. True when it changed; idempotent."""
    return _apply_scheduler_overrides(
        pipe, checkpoint_prediction_overrides(path), logger, "checkpoint markers"
    )


def apply_pipeline_prediction_type(
    pipe: Any,
    root: Optional[str],
    logger: Any = None,
    *,
    repo_id: Optional[str] = None,
    hf_token: Any = None,
    allow_network: bool = False,
) -> bool:
    """``apply_checkpoint_prediction_type`` for a diffusers pipeline directory
    (``pipeline_prediction_overrides``). True when it changed; idempotent."""
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    return _apply_scheduler_overrides(
        pipe,
        pipeline_prediction_overrides(config, root, repo_id, hf_token, allow_network),
        logger,
        "pipeline v-prediction",
    )
