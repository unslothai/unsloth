# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each job runs in a fresh subprocess (mp.get_context("spawn")): a clean
interpreter with no stale module state, which solves transformers
version-switching. Pattern follows core/data_recipe/jobs/worker.py.
"""

from __future__ import annotations

from loggers import get_logger
import importlib
import importlib.metadata
import math
import os
import shutil
import sys
import time
import traceback
import gc
import re
import types
import subprocess as _sp
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.training.training import TrainingProgress

# ── WSL AMD Strix Halo (gfx1151): enable ROCDXG before any torch import ──────
# Mirrors main.py. In WSL the AMD GPU is reached via the ROCDXG bridge (librocdxg.so
# over /dev/dxg), which HSA loads only when HSA_ENABLE_DXG_DETECTION=1 is set before
# torch touches the GPU; a worker spawned outside a login shell misses the installer's
# persisted env. Gated on both /dev/dxg and librocdxg.so, so other platforms no-op.
if sys.platform.startswith("linux") and "HSA_ENABLE_DXG_DETECTION" not in os.environ:
    try:
        if os.path.exists("/dev/dxg") and any(
            os.path.exists(_p + "/librocdxg.so") for _p in ("/opt/rocm/lib", "/opt/rocm/lib64")
        ):
            os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"
    except Exception:
        pass

logger = get_logger(__name__)
from utils.child_stdio import utf8_child_env

# Fresh spawned interpreter: re-apply the OS-trust-store injection.
from utils.native_tls import activate_native_tls

activate_native_tls()

from utils.hardware import apply_gpu_ids
from utils.hf_dataset_options import hf_dataset_split_instruction_names

# Light module on purpose: the MLX branch below runs on torch-less hosts, so it
# cannot reach these through core.training.trainer.
from core.training.dataset_bounds import (
    bound_dataset_rows,
    max_train_rows_for_config,
    record_row_bound,
    row_bound_for_resume,
    world_size_from_env,
)
from utils.training_runs import build_default_output_dir_name
from utils.wheel_utils import (
    direct_wheel_url,
    flash_attn_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)


def _output_dir_from_resume_checkpoint(resume_from_checkpoint: str | None) -> str | None:
    if not resume_from_checkpoint:
        return None
    path = Path(resume_from_checkpoint)
    return str(path.parent if path.name.startswith("checkpoint-") else path)


def _data_parallel_world_size() -> int:
    """Replicas that each draw a full batch of rows per optimizer step.

    Two things multiply row consumption and only one of them is in the env: a
    distributed launch (torchrun, accelerate, mpirun, mlx.launch), and plain
    DataParallel, which transformers reaches for whenever a non-distributed run
    sees more than one CUDA device -- it sets n_gpu to the visible device count and
    scales the train batch by it, so an extra visible GPU eats rows exactly as an
    extra rank does. XPU and MPS stay at one device there, so only CUDA counts.

    The larger of the two, never the sum: a distributed run forces n_gpu to 1, and a
    model-parallel one (a sharding device_map, which is what Unsloth's own multi-GPU
    load uses) forces it to 1 as well. Rounding up when the model turns out to be
    sharded rather than replicated only tokenizes a larger subset of a corpus this
    bound is orders of magnitude below anyway; rounding down means the run silently
    re-reads rows it has already trained on.

    torch is read out of sys.modules rather than imported: this also runs on the MLX
    path, on hosts where no torch exists, and a process that never imported it has
    no CUDA devices to count either.
    """
    sizes = [world_size_from_env()]
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            distributed = getattr(torch_module, "distributed", None)
            if (
                distributed is not None
                and distributed.is_available()
                and distributed.is_initialized()
            ):
                sizes.append(int(distributed.get_world_size()))
        except Exception:
            # A stubbed or half-initialised torch.distributed must not fail a run.
            pass
        try:
            sizes.append(int(torch_module.cuda.device_count()))
        except Exception:
            # A CPU-only build answers 0, which the filter below drops; a broken or
            # stubbed torch.cuda raises instead, and that must not fail a run either.
            pass
    return max([size for size in sizes if size > 0], default = 1)


def _model_local_files_only(config: dict) -> bool:
    return bool(config.get("model_snapshot_path"))


def _dataset_local_files_only(config: dict) -> bool:
    return bool(config.get("dataset_snapshot_path"))


def _untrainable_model_format_error(config: dict) -> str | None:
    model_format = str(config.get("model_format") or "").strip().lower()
    if model_format == "gguf":
        return "GGUF models are inference-only and cannot be trained."
    if model_format == "adapter":
        return "Adapter models are inference-only and cannot be trained as base models."
    return None


def _resolve_cached_model_load_name(config: dict) -> str:
    return config.get("model_snapshot_path") or config["model_name"]


def _effective_training_load_in_4bit(
    config: dict, model_load_target: str, hf_token: str | None
) -> bool:
    from .provenance import effective_training_load_in_4bit
    return effective_training_load_in_4bit(config, model_load_target, hf_token)


def _drop_model_pin(config: dict) -> str:
    config["model_snapshot_path"] = None
    return config["model_name"]


def _drop_model_pin_for_fallback(config: dict, hf_token: str | None) -> str:
    from utils.transformers_version import get_transformers_activation_tier

    active_target = _resolve_cached_model_load_name(config)
    fallback_target = config["model_name"]
    if not config.get("model_revision"):
        active_tier = get_transformers_activation_tier(active_target, hf_token)
        fallback_tier = get_transformers_activation_tier(fallback_target, hf_token)
        if active_tier != fallback_tier:
            raise RuntimeError(
                "The cached model is incomplete and its Hugging Face fallback requires "
                f"a different Transformers runtime ({active_tier} to {fallback_tier}). "
                "Remove the incomplete cached model and retry."
            )
    return _drop_model_pin(config)


def _is_model_cache_artifact_error(error: BaseException | None) -> bool:
    """Classify model-only failures that mean a local snapshot is incomplete.

    Transformers does not consistently report a missing tokenizer or processor as
    a file error.  Some families raise a bare ``TypeError`` after resolving a
    missing vocabulary path to ``None``.  Keep those otherwise-generic messages
    scoped to the model-cache retry path so they cannot make unrelated dataset or
    training failures retryable.
    """
    from hub.utils.dataset_cache import is_cache_artifact_error

    if is_cache_artifact_error(error):
        return True
    markers = (
        "can't load processor for",
        "can't load image processor for",
        "can't load feature extractor for",
        "stat: path should be string, bytes, os.pathlike or integer, not nonetype",
        "expected str, bytes or os.pathlike object, not nonetype",
        # SentencePiece/BPE families resolve a missing vocab path to None and dereference it, so
        # the failure arrives as a bare AttributeError with no cache-specific text. Without these,
        # 26 tokenizer families (XLMRoberta, MBart, NLLB, Bloom, ...) get zero Hub retry and a
        # pinned tokenizer-less snapshot is terminal. A false positive costs one Hub attempt.
        "'nonetype' object has no attribute 'endswith'",
        "'nonetype' object has no attribute 'readlines'",
        "argument should be a str or an os.pathlike object",
        "can't find a vocabulary file at path 'none'",
    )
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(marker in str(current).lower() for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def _model_offline_mode_enabled() -> bool:
    try:
        from utils.utils import hf_env_offline
        return hf_env_offline()
    except Exception:
        pass
    return any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _cache_artifact_fallback_allowed(
    config: dict, error: BaseException | None, resource: str
) -> bool:
    require_exact = bool(
        config.get("require_exact_resume_resources")
        or config.get(f"require_exact_{resource}_resource")
    )
    if resource == "dataset":
        from hub.utils.dataset_cache import dataset_cache_fallback_allowed
        return dataset_cache_fallback_allowed(
            error,
            require_exact = require_exact,
            revision = config.get("dataset_revision"),
        )
    if require_exact or _model_offline_mode_enabled():
        return False
    return _is_model_cache_artifact_error(error)


def _model_cache_fallback_error(config: dict, error: BaseException | None) -> RuntimeError | None:
    """Return an actionable error when an incomplete cache cannot be repaired."""
    if not _is_model_cache_artifact_error(error):
        return None
    if config.get("require_exact_resume_resources") or config.get("require_exact_model_resource"):
        return RuntimeError(
            "The exact cached model snapshot is incomplete, so this run cannot "
            "preserve its recorded model resources. Restore the missing model, "
            "tokenizer, or processor files and retry."
        )
    if _model_offline_mode_enabled():
        revision = config.get("model_revision")
        revision_text = f" at revision {revision}" if revision else ""
        return RuntimeError(
            "Offline mode is enabled, but the cached model snapshot is incomplete. "
            "Reconnect to download the missing model, tokenizer, or processor files"
            f"{revision_text}, or select a complete local model."
        )
    return None


def _mlx_revision_fallback_error(config: dict) -> RuntimeError | None:
    """Refuse an exact retry when MLX would remap the repo and drop its commit.

    ``FastMLXModel`` maps Unsloth bitsandbytes repositories to their full-precision
    base because MLX cannot read bnb-packed weights.  A commit from the selected
    repository has no guaranteed meaning in that different repository, so silently
    applying it (or dropping it) would violate the cache pin.
    """
    model_name = str(config.get("model_name") or "")
    revision = config.get("model_revision")
    if (
        revision
        and model_name.startswith("unsloth/")
        and model_name.endswith(("-unsloth-bnb-4bit", "-bnb-4bit"))
    ):
        return RuntimeError(
            "The cached model snapshot is incomplete, but MLX cannot safely retry "
            f"'{model_name}' at revision {revision}: MLX maps its bitsandbytes "
            "weights to a different base repository. Select that full-precision "
            "base model directly, or restore the missing cached files."
        )
    return None


def _require_strict_cached_dataset(config: dict, dataset: Any, split: str) -> Any:
    if (
        config.get("require_exact_resume_resources") or config.get("require_exact_dataset_resource")
    ) and dataset is None:
        raise FileNotFoundError(f"The exact cached dataset split '{split}' is no longer available.")
    return dataset


def _offline_mode_enabled() -> bool:
    return any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


def _verify_config_pins(config: dict, event_queue: Any) -> bool:
    require_model = bool(
        config.get("require_exact_resume_resources") or config.get("require_exact_model_resource")
    )
    require_dataset = bool(
        config.get("require_exact_resume_resources") or config.get("require_exact_dataset_resource")
    )
    if require_model or require_dataset:
        from core.training.provenance import (
            ExactResumeResourcesUnavailable,
            validate_exact_dataset_pin,
            validate_exact_model_pin,
        )

        try:
            model_snapshot = validate_exact_model_pin(config) if require_model else None
            dataset_snapshot = validate_exact_dataset_pin(config) if require_dataset else None
        except ExactResumeResourcesUnavailable as error:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(error),
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return False
        if model_snapshot is not None:
            config["model_snapshot_path"] = model_snapshot
            config["model_revision"] = Path(model_snapshot).name
        if dataset_snapshot is not None:
            config["dataset_snapshot_path"] = dataset_snapshot

    for message in config.get("cache_pin_warnings") or []:
        _send_status(event_queue, message)
    model_path = config.get("model_snapshot_path")
    require_validated_snapshot = bool(config.get("require_validated_model_snapshot"))
    if require_validated_snapshot and not model_path:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    "The cached model snapshot selected during preflight is no longer available."
                ),
                "stack": "",
                "ts": time.time(),
            }
        )
        return False
    if model_path and not require_model:
        from hub.utils.hf_cache_state import (
            latest_snapshot_from_cache_path,
            with_load_subdirs,
        )
        from utils.utils import canonical_model_repo_id

        pinned_repo_id = config.get("actual_model_repo_id") or canonical_model_repo_id(
            config["model_name"]
        )
        config["model_snapshot_path"] = latest_snapshot_from_cache_path(
            model_path,
            "model",
            pinned_repo_id,
            with_load_subdirs(config["model_name"], ("config.json", "adapter_config.json")),
        )
        if config["model_snapshot_path"] is None:
            if require_validated_snapshot:
                event_queue.put(
                    {
                        "type": "error",
                        "error": (
                            "The cached model snapshot selected during preflight is no "
                            "longer available."
                        ),
                        "stack": "",
                        "ts": time.time(),
                    }
                )
                return False
            if not config.get("model_revision"):
                config["actual_model_repo_id"] = None
        else:
            config["model_revision"] = Path(config["model_snapshot_path"]).name
    dataset_path = config.get("dataset_snapshot_path")
    if dataset_path and not require_dataset:
        from hub.utils.dataset_cache import (
            dataset_cache_path_from_cache_path,
            dataset_snapshot_from_cache_path,
        )

        resolved = dataset_cache_path_from_cache_path(
            dataset_path,
            config.get("hf_dataset") or "",
        )
        snapshot = (
            dataset_snapshot_from_cache_path(
                str(resolved),
                config.get("hf_dataset") or "",
            )
            if resolved is not None
            else None
        )
        if snapshot is not None:
            config["dataset_revision"] = snapshot.name
        config["dataset_snapshot_path"] = str(resolved) if resolved else None
    if (
        config.get("dataset_revision")
        and not config.get("dataset_snapshot_path")
        and _offline_mode_enabled()
    ):
        event_queue.put(
            {
                "type": "error",
                "error": (
                    "The selected dataset snapshot is incomplete and its exact "
                    "revision cannot be downloaded while offline."
                ),
                "stack": "",
                "ts": time.time(),
            }
        )
        return False
    return True


def _validate_training_worker_config(config: dict, event_queue: Any) -> bool:
    if not _verify_config_pins(config, event_queue):
        return False
    format_error = _untrainable_model_format_error(config)
    if format_error:
        event_queue.put(
            {
                "type": "error",
                "error": format_error,
                "stack": "",
                "ts": time.time(),
            }
        )
        return False
    return True


def _cached_dataset_row_limit(config: dict) -> int | None:
    slice_end = config.get("dataset_slice_end")
    slice_start = config.get("dataset_slice_start")
    if isinstance(slice_end, bool) or not isinstance(slice_end, int) or slice_end < 0:
        return None
    if slice_start is None:
        slice_start = 0
    if isinstance(slice_start, bool) or not isinstance(slice_start, int):
        return None
    return slice_end + 1 if slice_end >= max(slice_start, 0) else None


def _load_cached_dataset_for_config(
    config: dict,
    split: str | None,
    token: str | None = None,
    *,
    row_limit: int | None = None,
):
    hf_dataset = config.get("hf_dataset")
    local_path = config.get("dataset_snapshot_path")
    if not hf_dataset or not local_path:
        return None
    from hub.utils.dataset_cache import load_cached_hf_dataset

    kwargs = {
        "subset": config.get("subset"),
        "split": split or "train",
        "token": token,
    }
    if row_limit is not None:
        kwargs["row_limit"] = row_limit
    return load_cached_hf_dataset(hf_dataset, local_path, **kwargs)


def _load_hf_train_and_eval_datasets(
    config: dict,
    token: str | None,
    load_dataset: Callable,
    status_callback: Callable[[str], None],
    warning_callback: Callable[[str], None] | None = None,
):
    from core.training.eval_dataset import (
        EVAL_SPLIT_CANDIDATES,
        MIN_EVAL_ROWS,
        evaluation_enabled,
    )

    hf_dataset = config["hf_dataset"]
    subset = config.get("subset")
    train_split = config.get("train_split", "train") or "train"
    eval_split = config.get("eval_split")
    revision = config.get("dataset_revision")
    eval_enabled = evaluation_enabled(config.get("eval_steps"))
    require_exact = bool(
        config.get("require_exact_resume_resources") or config.get("require_exact_dataset_resource")
    )
    dataset = None
    loaded_from_cache = False
    config["_dataset_loaded_from_exact_snapshot"] = False

    def warn(message: str) -> None:
        if warning_callback is not None:
            warning_callback(message)
        else:
            logger.warning(message)

    def load_remote(split: str):
        kwargs = {"split": split, "token": token}
        if subset:
            kwargs["name"] = subset
        if revision:
            kwargs["revision"] = revision
        return load_dataset(hf_dataset, **kwargs)

    if _dataset_local_files_only(config):
        status_callback(f"Loading cached dataset: {hf_dataset}")
        try:
            row_limit = _cached_dataset_row_limit(config)
            if row_limit is None:
                dataset = _load_cached_dataset_for_config(config, train_split, token)
            else:
                dataset = _load_cached_dataset_for_config(
                    config,
                    train_split,
                    token,
                    row_limit = row_limit,
                )
            dataset = _require_strict_cached_dataset(config, dataset, train_split)
            loaded_from_cache = dataset is not None
        except Exception as error:
            if not _cache_artifact_fallback_allowed(config, error, "dataset"):
                raise
            status_callback("Cached dataset unavailable; downloading from the Hub...")

    if dataset is None:
        dataset = load_remote(train_split)

    eval_dataset = None
    explicit_separate_eval = bool(eval_split and eval_split != train_split)
    if eval_enabled and explicit_separate_eval:
        if loaded_from_cache:
            try:
                eval_dataset = _load_cached_dataset_for_config(config, eval_split, token)
                eval_dataset = _require_strict_cached_dataset(
                    config,
                    eval_dataset,
                    eval_split,
                )
            except Exception as error:
                if not _cache_artifact_fallback_allowed(config, error, "dataset"):
                    raise
                status_callback(
                    "Cached eval split unavailable; reloading train and eval from the Hub..."
                )
                remote_train = load_remote(train_split)
                remote_eval = load_remote(eval_split)
                dataset = remote_train
                eval_dataset = remote_eval
                loaded_from_cache = False
        else:
            eval_dataset = load_remote(eval_split)
    elif eval_enabled and not eval_split:
        auto_errors: list[tuple[str, Exception]] = []
        try:
            split_info = getattr(getattr(dataset, "info", None), "splits", None)
            if loaded_from_cache:
                available_splits = list(split_info or ())
            else:
                from datasets import get_dataset_split_names

                split_kwargs = {"path": hf_dataset}
                if subset:
                    split_kwargs["config_name"] = subset
                if revision:
                    split_kwargs["revision"] = revision
                if token:
                    split_kwargs["token"] = token
                available_splits = get_dataset_split_names(**split_kwargs)

            excluded_splits = set(hf_dataset_split_instruction_names(train_split))
            for candidate in EVAL_SPLIT_CANDIDATES:
                if candidate not in available_splits or candidate in excluded_splits:
                    continue
                try:
                    if loaded_from_cache:
                        candidate_dataset = _load_cached_dataset_for_config(
                            config,
                            candidate,
                            token,
                        )
                        candidate_dataset = _require_strict_cached_dataset(
                            config,
                            candidate_dataset,
                            candidate,
                        )
                    else:
                        candidate_dataset = load_remote(candidate)
                except Exception as error:
                    if require_exact:
                        raise
                    auto_errors.append((candidate, error))
                    continue
                if len(candidate_dataset) >= MIN_EVAL_ROWS:
                    eval_dataset = candidate_dataset
                    break
        except Exception as error:
            if require_exact:
                raise
            warn(
                "Automatic eval split detection failed; a held-out split will be created "
                f"from the training data when enough rows are available: {error}"
            )
        else:
            if eval_dataset is None and auto_errors:
                candidate, error = auto_errors[0]
                warn(
                    f"Automatic eval split '{candidate}' could not be loaded; a held-out "
                    "split will be created from the training data when enough rows are "
                    f"available: {error}"
                )

    from core.training.provenance import (
        attest_loaded_dataset,
        exact_dataset_snapshot_path,
    )

    snapshot, _ = attest_loaded_dataset(hf_dataset, dataset, eval_dataset)
    if snapshot is None and loaded_from_cache:
        snapshot = exact_dataset_snapshot_path(
            config.get("dataset_snapshot_path"),
            hf_dataset,
        )
    if snapshot is not None:
        config["dataset_snapshot_path"] = snapshot
        config["_dataset_loaded_from_exact_snapshot"] = True
    return dataset, eval_dataset


def _load_embedding_hf_dataset(
    config: dict, load_dataset: Callable, status_callback: Callable[[str], None]
):
    hf_dataset = str(config.get("hf_dataset") or "").strip()
    if not hf_dataset:
        return None

    subset = config.get("subset") or None
    train_split = config.get("train_split", "train") or "train"
    revision = config.get("dataset_revision")
    token = config.get("hf_token", "")
    token = token if token and token.strip() else None
    dataset = None
    config["_dataset_loaded_from_exact_snapshot"] = False

    if _dataset_local_files_only(config):
        status_callback(f"Loading cached dataset: {hf_dataset}")
        try:
            row_limit = _cached_dataset_row_limit(config)
            if row_limit is None:
                dataset = _load_cached_dataset_for_config(
                    config,
                    train_split,
                    token,
                )
            else:
                dataset = _load_cached_dataset_for_config(
                    config,
                    train_split,
                    token,
                    row_limit = row_limit,
                )
            dataset = _require_strict_cached_dataset(
                config,
                dataset,
                train_split,
            )
            if dataset is not None:
                from core.training.provenance import exact_dataset_snapshot_path
                snapshot = exact_dataset_snapshot_path(
                    config.get("dataset_snapshot_path"),
                    hf_dataset,
                )
                if snapshot is not None:
                    config["dataset_snapshot_path"] = snapshot
                    config["_dataset_loaded_from_exact_snapshot"] = True
        except Exception as error:
            if not _cache_artifact_fallback_allowed(config, error, "dataset"):
                raise
            status_callback("Cached dataset unavailable; downloading from the Hub...")
            dataset = None
            config["_dataset_loaded_from_exact_snapshot"] = False

    if dataset is None:
        load_kwargs = {
            "split": train_split,
            "token": token,
        }
        if revision:
            load_kwargs["revision"] = revision
        dataset = load_dataset(hf_dataset, subset, **load_kwargs)
    from core.training.provenance import attest_loaded_dataset

    snapshot, _ = attest_loaded_dataset(hf_dataset, dataset)
    if snapshot is not None:
        config["dataset_snapshot_path"] = snapshot
        config["_dataset_loaded_from_exact_snapshot"] = True
    return dataset


def _pre_detect_training_model(
    trainer,
    config: dict,
    model_name: str,
    hf_token: str | None,
    model_load_name: str,
    local_files_only: bool,
    model_revision: str | None = None,
) -> None:
    trainer.pre_detect_and_load_tokenizer(
        model_name = model_name,
        max_seq_length = config["max_seq_length"],
        hf_token = hf_token,
        is_dataset_image = config.get("is_dataset_image", False),
        is_dataset_audio = config.get("is_dataset_audio", False),
        trust_remote_code = config.get("trust_remote_code", False),
        model_load_name = model_load_name,
        local_files_only = local_files_only,
        model_revision = model_revision,
    )
    _check_finetune_targets_after_detect(trainer, config)


_NOTHING_TO_TRAIN = (
    "Nothing to train: select at least one layer family (finetune_language_layers or "
    "finetune_vision_layers) and at least one module type (finetune_attention_modules or "
    "finetune_mlp_modules)."
)


def _finetune_selectors(config: dict) -> tuple[bool, bool, bool, bool]:
    """(vision, language, attention, mlp), read exactly the way the consumers read them.

    A guard that models the run differently from the code it guards rejects runs that would
    have trained, so every default here is the CUDA consumer's own default for an omitted key.
    Only the MLX consumer defaults vision False, and _check_mlx_finetune_targets discards the
    vision element, so True is safe there too.
    """
    return (
        bool(config.get("finetune_vision_layers", True)),
        bool(config.get("finetune_language_layers", True)),
        bool(config.get("finetune_attention_modules", True)),
        bool(config.get("finetune_mlp_modules", True)),
    )


def _requests_all_linear(config: dict) -> bool:
    """Whether target_modules is PEFT's bare "all-linear" keyword rather than a leaf list.

    get_peft_model forces every selector True for the keyword, so all-linear with the
    selectors off trains every linear layer today and rejecting it would break the very
    requests the selectors are not consulted for. A list naming all-linear alongside other
    leaves is not the keyword: the caller strips it and the rest take the scoped path.
    """
    target_modules = config.get("target_modules")
    if isinstance(target_modules, str):
        return target_modules == "all-linear"
    if isinstance(target_modules, (list, tuple)):
        return list(target_modules) == ["all-linear"]
    return False


def _check_finetune_targets_after_detect(trainer, config: dict) -> None:
    """Reject a LoRA run that selects no adapter layers, once detection has settled which
    branch it takes. The request model cannot decide this: the codec/ASR branches ignore the
    selectors that is_audio_vlm reads, is_vlm needs a vision-capable model and not just an
    image-tagged dataset, and only the probe in pre_detect separates those. pre_detect is
    config/tokenizer only, so this still fires before any weights load, instead of surfacing
    as get_peft_regex's "No layers to finetune" with the model already in memory."""
    if config.get("training_type", "LoRA/QLoRA") != "LoRA/QLoRA":
        return  # Full Finetuning / CPT build adapters from target_modules alone
    if not (getattr(trainer, "is_vlm", False) or getattr(trainer, "is_audio_vlm", False)):
        return  # the text branch ignores these four
    if _requests_all_linear(config):
        return  # get_peft_model turns all five selectors on for the keyword; see below
    vision, language, attention, mlp = _finetune_selectors(config)
    # Mirror get_peft_regex's two guards: one layer family AND one module type.
    if not (vision or language) or not (attention or mlp):
        raise ValueError(_NOTHING_TO_TRAIN)


# Targets the MLX loader trains regardless of the layer-family flags: on the CPT path
# embed_tokens becomes a full trainable module and lm_head its own adapter.
_CPT_TARGET_NAMES = frozenset({"embed_tokens", "lm_head"})


def _names_a_cpt_target(target_modules) -> bool:
    """Whether an explicit target list names something that trains on its own."""
    if isinstance(target_modules, str):
        return target_modules in _CPT_TARGET_NAMES
    try:
        return any(name in _CPT_TARGET_NAMES for name in target_modules)
    except TypeError:  # not iterable -> not a list of names, so nothing is guaranteed
        return False


def _check_mlx_finetune_targets(config: dict) -> None:
    """MLX equivalent, called from the LoRA branch of the MLX worker.

    Two things differ from the CUDA path. FastMLXModel.get_peft_model is handed these
    selectors for text models too, so there is no is_vlm gate. And the caller back-fills
    finetune_language_layers whenever a module type is on, so only an empty module selection
    can survive here.

    Surviving the module-type filter is NOT enough to train. get_peft_model drops only the
    names it recognises as attention or MLP leaves, so a fused qkv, a c_fc or an expanded
    all-linear survives with both module types off -- but the text branch then gates the LoRA
    application on finetune_language_layers, and with all four selectors off the caller's
    back-fill never fires. Those runs apply no adapters at all: the model warns and trains
    nothing, and a VLM raises only once the weights are loaded.

    The exception is a target the loader handles independently of the layer families: naming
    embed_tokens or lm_head puts it on the CPT path, which trains whatever the flags say.

    An explicit list that merely filters down to nothing still gets the loader's own message,
    which names the two flags."""
    targets = config.get("target_modules")
    if targets:
        if _names_a_cpt_target(targets):
            return
        _, language, attention, mlp = _finetune_selectors(config)
        # Vision read the way the MLX call site reads it, NOT the way _finetune_selectors
        # does: that helper carries the CUDA consumer's defaults, where an omitted vision
        # selector means True, while MLX defaults it False and forces it False for a text
        # model. Taking True from an omitted key would wave through every legacy config
        # that never sent the selectors at all.
        vision = bool(config.get("finetune_vision_layers", False))
        # Any one of them leaves something that can train, or leaves the loader to say so
        # with a better message. Vision counts because this runs BEFORE detection, so a VLM
        # whose vision tower is the only selection must not be refused here;
        # _check_mlx_effective_targets catches the text case once is_vlm is known.
        if attention or mlp or language or vision:
            return
        raise ValueError(_NOTHING_TO_TRAIN)
    _, _, attention, mlp = _finetune_selectors(config)
    if not (attention or mlp):
        raise ValueError(_NOTHING_TO_TRAIN)


def _check_mlx_effective_targets(
    config: dict, *, finetune_language: bool, finetune_vision: bool
) -> None:
    """The same refusal, re-asked with the values get_peft_model will actually receive.

    ``_check_mlx_finetune_targets`` runs before the model is loaded, so it cannot tell a VLM
    from a text model and has to let a vision-only selection through. The call site can: it
    has forced vision to False for a text model and applied the language back-fill, so if
    both layer families are still off here, no adapter is coming and the run would train
    nothing but its own warning.

    Later than the preflight deliberately: this is the first point the answer is knowable,
    and it is still before the trainer is built and before a single step runs."""
    if finetune_language or finetune_vision:
        return
    if _names_a_cpt_target(config.get("target_modules") or ()):
        return
    raise ValueError(_NOTHING_TO_TRAIN)


def _reload_dataset_with_remote_model_tokenizer(
    trainer,
    config: dict,
    model_name: str,
    hf_token: str | None,
    reload_dataset: Callable[[], tuple],
    model_revision: str | None = None,
):
    _pre_detect_training_model(
        trainer,
        config,
        model_name,
        hf_token,
        model_name,
        False,
        model_revision,
    )
    return reload_dataset()


def _model_load_security_error(config: dict, load_target: str, hf_token: str | None) -> dict | None:
    from utils.models.model_config import get_base_model_from_lora_identifier
    from utils.security import (
        evaluate_file_security,
        evaluate_remote_code_consent_for_targets,
        load_scan_target,
        security_load_subdirs,
    )

    requested_targets = [load_target]
    try:
        base_model = get_base_model_from_lora_identifier(load_target, hf_token)
        if base_model:
            requested_targets.append(base_model)
    except Exception as error:
        logger.debug("Could not resolve LoRA base for security scan: %s", error)

    from utils.utils import hf_env_offline

    primary_name = config["model_name"]
    local_only_load = hf_env_offline()
    consent_load_subdirs: dict[str, tuple] = {}
    targets: list[str] = []
    for requested_target in dict.fromkeys(requested_targets):
        load_subdirs = security_load_subdirs(requested_target, hf_token)
        if requested_target == load_target and requested_target != primary_name:
            load_subdirs = tuple(
                dict.fromkeys((*load_subdirs, *security_load_subdirs(primary_name, hf_token)))
            )
        target, load_subdirs = load_scan_target(requested_target, load_subdirs)
        if target not in consent_load_subdirs:
            targets.append(target)
            consent_load_subdirs[target] = ()
        load_subdirs = tuple(dict.fromkeys((*consent_load_subdirs[target], *load_subdirs)))
        consent_load_subdirs[target] = load_subdirs
        decision = evaluate_file_security(
            target,
            hf_token = hf_token,
            load_subdirs = load_subdirs,
            local_only_load = local_only_load,
        )
        if decision.blocked:
            return {
                "error": decision.reason,
                "error_kind": "malware_blocked",
                "security": decision.response_payload(),
            }

    if not config.get("trust_remote_code", False):
        return None

    decision = evaluate_remote_code_consent_for_targets(
        targets,
        hf_token = hf_token,
        trust_remote_code = True,
        approved_fingerprint = config.get("approved_remote_code_fingerprint"),
        subject = config.get("subject"),
        load_subdirs_by_target = consent_load_subdirs,
    )
    if not decision.blocked:
        return None
    return {
        "error": (
            f"Model '{decision.model_name}' ships custom code flagged as "
            f"{decision.max_severity} by the security scan. Review it and "
            f"re-run with approval to proceed.\n\n{decision.findings_summary}"
        ),
        "error_kind": "remote_code_blocked",
        "remote_code": decision.response_payload(),
    }


_CAUSAL_CONV1D_RELEASE_TAG = "v1.6.1.post4"
_CAUSAL_CONV1D_PACKAGE_VERSION = "1.6.1"
_CAUSAL_CONV1D_MODEL_SUBSTRINGS = (
    "qwen3.5",
    "qwen3_5",
    "qwen3.6",
    "qwen3_6",
    "qwen3-next",
    "qwen3_next",
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
    "lfm2",
    "mamba",
    "jamba",
    "zamba",
    "bamba",
)
_MAMBA_SSM_RELEASE_TAG = "v2.3.1"
_MAMBA_SSM_PACKAGE_VERSION = "2.3.1"
_FLASH_ATTN_RUNTIME_MIN_SEQ_LEN = 32768
_FLASH_ATTN_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL"
# apache-tvm-ffi 0.1.10/0.1.11 crash Triton with "CUDA: misaligned address" on sm_100.
_TILELANG_PACKAGE_VERSION = "0.1.8"
_APACHE_TVM_FFI_PACKAGE_VERSION = "0.1.9"
_TILELANG_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_TILELANG_INSTALL"
# Pin both so plain pip can't silently upgrade torch under the worker (fla-core needs torch>=2.7).
_FLA_PACKAGE_VERSION = "0.5.0"
_FLA_CORE_PACKAGE_VERSION = "0.5.0"
_FLA_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FLA_INSTALL"
# `--no-deps` saves torch but loses fla-core's transitive deps; `packaging` is also undeclared upstream.
_FLA_RUNTIME_DEPS = ("einops", "packaging", "triton")
_FLA_MIN_TORCH = (2, 7)
_FLA_MIN_PYTHON = (3, 10)
# tilelang 0.1.8 ships wheels only for these Linux arches and macOS arm64; never fall back to its 93MB sdist.
_TILELANG_SUPPORTED_LINUX_MACHINES = frozenset(("x86_64", "amd64", "aarch64", "arm64"))
_TILELANG_INSTALL_TIMEOUT_S = 600
_TVM_FFI_BROKEN_VERSIONS = ("0.1.10", "0.1.11")
_FAST_PATH_HOOKS_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS"

# Module scope so the torch.library.Library registration isn't GC'd mid-run.
_WINDOWS_ROCM_GROUPED_MM_LIB = None


def _install_grouped_mm_cpu_fallback(torch_mod, logger, label):
    """Register a Python mm/bmm fallback for torch._grouped_mm and return the Library.

    RDNA4 (gfx1200/gfx1201) ships a null HIP _grouped_mm kernel on ROCm <= 7.12
    (fixed in 7.13; ROCm/TheRock #5284). JitDecomp dispatches _grouped_mm to the
    null kernel and crashes; overriding the CUDA dispatch key bypasses it. Shared
    by the Windows and Linux ROCm guards. Keep the returned Library referenced so
    the registration outlives the caller.
    """
    import warnings as _warnings

    _gm_lib = torch_mod.library.Library("aten", "IMPL")

    def _grouped_mm_safe_impl(
        self,
        mat2,
        offs = None,
        bias = None,
        out_dtype = None,
    ):
        """Python mm/bmm fallback for _grouped_mm on gfx120X (null HIP kernel, ROCm <= 7.12)."""
        _t = torch_mod
        if offs is None:
            # No offsets: 2-D -> mm, 3-D batched -> bmm (unconditional mm broke 3-D MoE).
            if self.dim() == 3 and mat2.dim() == 3:
                result = _t.bmm(self.contiguous(), mat2.contiguous())
            elif self.dim() == 3 and mat2.dim() == 2:
                result = _t.matmul(self.contiguous(), mat2.contiguous())
            elif self.dim() == 2 and mat2.dim() == 3:
                result = _t.matmul(self.contiguous(), mat2.contiguous())
            else:
                result = _t.mm(self.contiguous(), mat2.contiguous())
        else:
            # Grouped: offs[i] is the exclusive end-row of group i.
            offs_list = offs.tolist()
            pieces = []
            prev = 0
            for idx, end in enumerate(offs_list):
                end = int(end)
                a_part = self[prev:end].contiguous()
                b_part = mat2[idx].contiguous() if mat2.dim() == 3 else mat2.contiguous()
                pieces.append(_t.mm(a_part, b_part))
                prev = end
            # Include trailing rows not covered by offs.
            if prev < self.shape[0]:
                a_tail = self[prev:].contiguous()
                b_tail = mat2[-1].contiguous() if mat2.dim() == 3 else mat2.contiguous()
                pieces.append(_t.mm(a_tail, b_tail))
            result = (
                _t.cat(pieces, dim = 0)
                if pieces
                else _t.zeros(0, mat2.shape[-1], device = self.device, dtype = self.dtype)
            )
        if bias is not None:
            result = result + bias
        if out_dtype is not None:
            result = result.to(out_dtype)
        elif result.dtype != self.dtype:
            result = result.to(self.dtype)
        return result

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        _gm_lib.impl("_grouped_mm", _grouped_mm_safe_impl, "CUDA")
    logger.info(
        "%s: patched _grouped_mm CUDA dispatch (null HIP kernel on gfx120X, "
        "ROCm <= 7.12 -- bypassed with Python mm fallback)",
        label,
    )
    return _gm_lib


# Subprocesses don't inherit os.add_dll_directory registrations. Replicate main.py's
# Windows ROCm DLL setup so the first `import torch` finds amdhip64.dll; handles kept.
_ROCM_DLL_HANDLES: list = []
if sys.platform == "win32":

    def _add_rocm_dll_dirs_worker() -> None:
        _candidates: list[str] = []
        for _var in ("HIP_PATH", "ROCM_PATH"):
            _val = os.environ.get(_var)
            if _val:
                _candidates.append(os.path.join(_val, "bin"))
        _default_root = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm"
        )

        def _ver_key(name: str) -> tuple:
            # Numeric tuple key so "10.0" sorts after "7.0".
            parts = []
            for chunk in name.split("."):
                try:
                    parts.append((0, int(chunk)))
                except ValueError:
                    parts.append((1, chunk))
            return tuple(parts)

        try:
            if os.path.isdir(_default_root):
                for _ver in sorted(os.listdir(_default_root), key = _ver_key, reverse = True):
                    _bin = os.path.join(_default_root, _ver, "bin")
                    if os.path.isdir(_bin):
                        _candidates.append(_bin)
        except OSError:
            pass
        for _d in _candidates:
            if os.path.isdir(_d):
                try:
                    _ROCM_DLL_HANDLES.append(os.add_dll_directory(_d))
                except (OSError, AttributeError):
                    pass

    _add_rocm_dll_dirs_worker()
    del _add_rocm_dll_dirs_worker


def _model_wants_causal_conv1d(model_name: str) -> bool:
    name = model_name.lower()
    return any(key in name for key in _CAUSAL_CONV1D_MODEL_SUBSTRINGS)


def _hipcc_gcc_install_dir() -> str | None:
    """Highest-numbered ``/usr/lib/gcc/x86_64-linux-gnu/<N>`` that has BOTH the
    gcc runtime dir AND ``/usr/include/c++/<N>`` headers, or None.

    Ubuntu 24.04 ships gcc-14 runtime but not ``/usr/include/c++/14``; ROCm
    clang-20 picks the highest runtime dir, finds no ``<cstdlib>``, and the HIP
    build fails. The returned path is passed to clang via
    ``--gcc-install-dir``. Mirrors bbf004c in studio/setup.sh (PR #5301).
    """
    if not sys.platform.startswith("linux"):
        return None
    import platform as _platform

    if _platform.machine().lower() != "x86_64":
        return None
    for _ver in (14, 13, 12, 11):
        _runtime = f"/usr/lib/gcc/x86_64-linux-gnu/{_ver}/include"
        _headers = f"/usr/include/c++/{_ver}"
        if os.path.isdir(_runtime) and os.path.isdir(_headers):
            return f"/usr/lib/gcc/x86_64-linux-gnu/{_ver}"
    return None


def _is_importable(import_name: str) -> bool:
    # Invalidate finder caches so a package installed earlier in this process is seen.
    importlib.invalidate_caches()
    try:
        __import__(import_name)
        return True
    except Exception as exc:
        # A wrong-arch/ABI wheel raises OSError/RuntimeError ("undefined symbol"), not
        # ImportError, so catch everything and let the caller fall back.
        logger.debug("%s is not importable (%s: %s)", import_name, type(exc).__name__, exc)
        return False


# Kept in step with install_python_stack._FLASH_ATTN_IMPORT_PROBE_TIMEOUT.
_IMPORT_PROBE_TIMEOUT = 300


def _is_importable_isolated(import_name: str) -> bool:
    """Probe the import in a child process.

    A wrong-arch wheel can abort or segfault in its initialiser instead of raising, which
    would kill this worker rather than fall back. A child turns that into a return code
    (negative = fatal signal).
    """
    try:
        result = _sp.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; importlib.import_module(sys.argv[1])",
                import_name,
            ],
            stdout = _sp.DEVNULL,
            stderr = _sp.DEVNULL,
            timeout = _IMPORT_PROBE_TIMEOUT,
            # No env override: the probe must see what the real in-process import will.
        )
    except (OSError, _sp.TimeoutExpired) as exc:
        logger.debug("%s import probe did not complete (%s)", import_name, exc)
        return False
    if result.returncode != 0:
        logger.debug("%s import probe exited %s", import_name, result.returncode)
    return result.returncode == 0


def _uninstall_package(pypi_name: str, display_name: str) -> bool:
    """Remove a distribution. True iff it is gone afterwards."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "uninstall", "--python", sys.executable, pypi_name]
    else:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", pypi_name]
    result = _sp.run(
        cmd,
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = utf8_child_env(),
    )
    if result.returncode != 0:
        logger.warning("Could not remove the rejected %s install:\n%s", display_name, result.stdout)
        return False
    return True


def _distribution_present(pypi_name: str) -> bool:
    """Whether the distribution's METADATA is installed, without importing it.

    Metadata is what matters: unsloth/models/_utils.py gates on ``_package_available`` and
    only then imports the native module, so metadata left behind is what turns a rejected
    wheel into an in-process crash. Reading it never loads the extension, so this is safe
    even for a wheel that would abort.
    """
    importlib.invalidate_caches()
    try:
        importlib.metadata.distribution(pypi_name)
        return True
    except Exception:
        return False


def _reject_install(event_queue: Any, pypi_name: str, display_name: str, reason: str) -> None:
    """Discard an install that will not import, and say which state we ended in.

    Idempotent, so it is safe on EVERY exit rather than only the ones somebody remembered:
    it no-ops when the distribution is already gone. Leaving it in place is not the same as
    never having installed it, since the metadata gate above imports it anyway.
    """
    if not _distribution_present(pypi_name):
        return
    logger.warning("%s %s", display_name, reason)
    if _uninstall_package(pypi_name, display_name):
        _send_status(event_queue, f"{display_name} is not usable on this GPU; removed it")
    else:
        _send_status(
            event_queue,
            f"{display_name} is not usable on this GPU and could not be removed; "
            f"uninstall {pypi_name} manually before training",
        )


def _install_package_wheel_first(
    *, event_queue: Any, import_name: str, display_name: str, pypi_name: str, **kwargs: Any
) -> bool:
    """Install a fast-path package, wheel first, and never leave an unusable one behind.

    The two "touch nothing" guards run here, outside the cleanup: an already-working
    package returns before any subprocess, and offline changes nothing. Everything after
    them is an install attempt, so ANY unsuccessful exit -- timeout, failed install, bad
    import -- discards what is left rather than leaving metadata the in-process import
    would pick up. Enforced here rather than at each return because four separate exits
    have now been found that forgot to clean up.
    """
    if _is_importable(import_name):
        logger.info("%s already installed", display_name)
        return True

    if _model_offline_mode_enabled():
        logger.info("Skipping %s installation while offline", display_name)
        return False

    installed = False
    try:
        installed = _attempt_package_install(
            event_queue = event_queue,
            import_name = import_name,
            display_name = display_name,
            pypi_name = pypi_name,
            **kwargs,
        )
        return installed
    finally:
        if not installed:
            _reject_install(event_queue, pypi_name, display_name, "will not import")


def _attempt_package_install(
    *,
    event_queue: Any,
    import_name: str,
    display_name: str,
    pypi_name: str,
    pypi_version: str | None = None,
    filename_prefix: str | None = None,
    release_tag: str | None = None,
    release_base_url: str | None = None,
    wheel_url_builder: Callable[[dict[str, str] | None], str | None] | None = None,
    pypi_spec: str | None = None,
    pypi_status_message: str | None = None,
) -> bool:
    """The install itself. Call it through _install_package_wheel_first, never directly."""
    # Set when a wheel installed but would not import; see the uninstall before the fallback.
    wheel_rejected = False

    env = probe_torch_wheel_env(timeout = 30)
    if wheel_url_builder is not None:
        wheel_url = wheel_url_builder(env)
    else:
        wheel_url = direct_wheel_url(
            filename_prefix = filename_prefix,
            package_version = pypi_version,
            release_tag = release_tag,
            release_base_url = release_base_url,
            env = env,
        )

    if wheel_url is None:
        logger.info("No compatible %s wheel candidate", display_name)
    elif url_exists(wheel_url):
        _send_status(event_queue, f"Installing {display_name} for faster training...")
        for installer, result in install_wheel(
            wheel_url,
            python_executable = sys.executable,
            use_uv = bool(shutil.which("uv")),
            run = _sp.run,
        ):
            if result.returncode == 0:
                # A wheel can install yet fail to import (CUDA/ABI or arch mismatch), so
                # verify rather than trust the exit code, and do it out of process: a bad
                # one can take the worker down with it.
                if _is_importable_isolated(import_name):
                    logger.info("Installed prebuilt %s wheel successfully", display_name)
                    return True
                logger.warning(
                    "%s wheel installed but is not importable; falling back to PyPI",
                    display_name,
                )
                wheel_rejected = True
                break
            logger.warning(
                "%s failed to install %s wheel:\n%s",
                installer,
                display_name,
                result.stdout,
            )
    else:
        logger.info("No published %s wheel found: %s", display_name, wheel_url)

    is_hip = env and env.get("hip_version")
    if is_hip and not shutil.which("hipcc"):
        logger.error(
            "%s requires hipcc for source compilation on ROCm. "
            "Install the ROCm HIP SDK: https://rocm.docs.amd.com",
            display_name,
        )
        _send_status(
            event_queue,
            f"{display_name}: hipcc not found (ROCm HIP SDK required)",
        )
        return False

    if pypi_spec is None:
        pypi_spec = f"{pypi_name}=={pypi_version}"

    if pypi_status_message is None:
        if is_hip:
            pypi_status_message = (
                f"Compiling {display_name} from source for ROCm (this may take several minutes)..."
            )
        else:
            pypi_status_message = f"Installing {display_name} from PyPI for faster training..."

    if wheel_rejected:
        # Remove it rather than install over it: pip/uv would report the broken
        # distribution as already satisfying the spec and do nothing. --force-reinstall is
        # not the answer either, since both scope it to the whole resolved transaction,
        # which for flash-attn means torch and the running CUDA stack.
        #
        # A failure here is caught by the _reject_install in the finally below, which is
        # reached from every exit rather than only the ones that remember to clean up.
        _uninstall_package(pypi_name, display_name)

    _send_status(event_queue, pypi_status_message)

    plain_pypi_install = pypi_version is None
    if plain_pypi_install:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                pypi_spec,
            ]
        else:
            pypi_cmd = [sys.executable, "-m", "pip", "install", pypi_spec]
    else:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-build-isolation",
                "--no-deps",
            ]
            # Avoid stale cache artifacts from partial HIP source builds
            if is_hip:
                pypi_cmd.append("--no-cache")
            pypi_cmd.append(pypi_spec)
        else:
            pypi_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--no-cache-dir",
                pypi_spec,
            ]

    # ROCm source compilation can take 10-30 min; use a generous timeout. Non-HIP installs
    # keep the pre-existing "no timeout" behaviour so unrelated slow builds (causal-conv1d
    # on aarch64, unsupported torch/CUDA combos) aren't aborted at 5 minutes.
    _run_kwargs: dict[str, Any] = {
        "stdout": _sp.PIPE,
        "stderr": _sp.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        # Make the Python child emit the UTF-8 we decode above.
        "env": utf8_child_env(),
    }
    if is_hip:
        _run_kwargs["timeout"] = 1800
        # On Ubuntu 24.04 + ROCm clang-20 the HIP source build dies on a missing <cstdlib>
        # (gcc-14 runtime dir lacks C++ headers). Inject --gcc-install-dir for a gcc whose
        # headers exist, respecting any pre-existing one. Mirrors bbf004c in setup.sh (PR #5301).
        _existing_flags = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "")
        if "--gcc-install-dir" not in _existing_flags:
            _gcc_dir = _hipcc_gcc_install_dir()
            if _gcc_dir is not None:
                _appended = (f"{_existing_flags} --gcc-install-dir={_gcc_dir}").strip()
                _env = _run_kwargs.get("env", os.environ).copy()
                _env["HIPCC_COMPILE_FLAGS_APPEND"] = _appended
                _run_kwargs["env"] = _env
                logger.info(
                    "HIP source build for %s: appended "
                    "--gcc-install-dir=%s to HIPCC_COMPILE_FLAGS_APPEND",
                    display_name,
                    _gcc_dir,
                )

    try:
        result = _sp.run(pypi_cmd, **_run_kwargs)
    except _sp.TimeoutExpired:
        logger.error(
            "%s installation timed out after %ds",
            display_name,
            _run_kwargs.get("timeout"),
        )
        _send_status(
            event_queue,
            f"{display_name} installation timed out after {_run_kwargs.get('timeout')}s",
        )
        return False

    if result.returncode != 0:
        if is_hip:
            error_lines = (result.stdout or "").strip().splitlines()
            snippet = "\n".join(error_lines[-5:]) if error_lines else "(no output)"
            logger.error(
                "Failed to compile %s for ROCm:\n%s",
                display_name,
                result.stdout,
            )
            _send_status(
                event_queue,
                f"Failed to compile {display_name} for ROCm. "
                "Check that hipcc and ROCm development headers are installed.\n"
                f"{snippet}",
            )
        else:
            if sys.platform == "win32":
                # No prebuilt wheel and no source toolchain on Windows -- expected for packages like
                # causal-conv1d. Log at info so users aren't alarmed by what looks like an error.
                logger.info(
                    "%s is not available on Windows (no prebuilt wheel); skipping",
                    display_name,
                )
                logger.debug("Install output:\n%s", result.stdout)
            else:
                logger.error(
                    "Failed to install %s from PyPI:\n%s",
                    display_name,
                    result.stdout,
                )
        return False

    # rc=0 is not proof again here: pip/uv exit 0 on "Requirement already satisfied" without
    # installing anything. Returning False is enough; the caller's finally discards it.
    if not _is_importable_isolated(import_name):
        logger.warning("%s installed from PyPI but will not import", display_name)
        return False

    if is_hip:
        logger.info("Compiled and installed %s from source for ROCm", display_name)
    else:
        logger.info("Installed %s from PyPI", display_name)
    return True


def _ensure_causal_conv1d_fast_path(
    event_queue: Any,
    model_name: str,
    *,
    required: bool | None = None,
) -> None:
    if required is None:
        required = _model_wants_causal_conv1d(model_name)
    if not required:
        return
    if sys.platform == "win32":
        logger.info("causal-conv1d: no prebuilt wheel for Windows; skipping")
        return

    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


def _installed_torch_version_tuple() -> tuple[int, int] | None:
    """Return ``(major, minor)`` of the installed torch, else None."""
    try:
        from importlib.metadata import version as _pkg_version

        raw = _pkg_version("torch").split("+", 1)[0]
        parts = raw.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def _flash_linear_attention_importable() -> bool:
    """Catch any exception (not just ImportError) so a broken native lib doesn't abort the worker."""
    try:
        import fla.modules  # noqa: F401
        import fla.ops.gated_delta_rule  # noqa: F401
        return True
    except Exception as exc:
        logger.warning(
            "flash-linear-attention is not importable; continuing with install/fallback: %s",
            exc,
        )
        return False


def _flash_linear_attention_current(already_importable: bool | None = None) -> bool:
    """True iff FLA imports AND is at the pinned version (older FLA lacks gated_delta_rule kernels)."""
    if already_importable is None:
        already_importable = _flash_linear_attention_importable()
    if not already_importable:
        return False
    try:
        from importlib.metadata import version as _pkg_version
        from packaging.version import Version

        fla_v = Version(_pkg_version("flash-linear-attention"))
        core_v = Version(_pkg_version("fla-core"))
        return fla_v >= Version(_FLA_PACKAGE_VERSION) and core_v >= Version(
            _FLA_CORE_PACKAGE_VERSION
        )
    except Exception as exc:
        logger.warning(
            "flash-linear-attention importable but version check failed; treating as stale: %s",
            exc,
        )
        return False


def _ensure_flash_linear_attention_unconditional(event_queue: Any) -> bool:
    """Install pinned FLA + fla-core with --no-deps. Returns True iff importable post-call."""
    if os.getenv(_FLA_SKIP_ENV) == "1":
        return False
    if sys.platform == "win32":
        logger.info("Skipping flash-linear-attention install: no prebuilt wheel for Windows")
        return False
    if sys.version_info < _FLA_MIN_PYTHON:
        logger.info(
            "Skipping flash-linear-attention install: requires Python >= %d.%d, have %s",
            _FLA_MIN_PYTHON[0],
            _FLA_MIN_PYTHON[1],
            sys.version.split()[0],
        )
        return False
    torch_ver = _installed_torch_version_tuple()
    if torch_ver is not None and torch_ver < _FLA_MIN_TORCH:
        _send_status(
            event_queue,
            (
                f"Skipping flash-linear-attention install: fla-core requires "
                f"torch>={_FLA_MIN_TORCH[0]}.{_FLA_MIN_TORCH[1]}, have "
                f"{torch_ver[0]}.{torch_ver[1]}"
            ),
        )
        return False

    # Probe once so the --force-reinstall decision and short-circuit share a call count.
    already_importable = _flash_linear_attention_importable()
    if already_importable and _flash_linear_attention_current(already_importable = True):
        logger.info("flash-linear-attention already importable at the pinned version")
        return True

    if _model_offline_mode_enabled():
        logger.info("Skipping flash-linear-attention installation while offline")
        return False

    _send_status(
        event_queue,
        f"Installing flash-linear-attention=={_FLA_PACKAGE_VERSION} for faster training...",
    )

    # `--no-deps` blocks the silent torch upgrade; bring non-torch runtime deps in by hand.
    specs = [
        *_FLA_RUNTIME_DEPS,
        f"fla-core=={_FLA_CORE_PACKAGE_VERSION}",
        f"flash-linear-attention=={_FLA_PACKAGE_VERSION}",
    ]
    extra_args = ["--no-deps"]
    if already_importable:
        # Older FLA already imported; pip skips reinstall without this flag.
        extra_args.append("--force-reinstall")

    if shutil.which("uv"):
        pypi_cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            *extra_args,
            *specs,
        ]
    else:
        pypi_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *extra_args,
            *specs,
        ]

    try:
        result = _sp.run(
            pypi_cmd,
            stdout = _sp.PIPE,
            stderr = _sp.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = utf8_child_env(),
            timeout = _TILELANG_INSTALL_TIMEOUT_S,
        )
    except _sp.TimeoutExpired:
        logger.warning("flash-linear-attention install timed out; continuing")
        _send_status(event_queue, "flash-linear-attention install timed out; continuing")
        return False

    if result.returncode != 0:
        if sys.platform == "win32":
            logger.info(
                "flash-linear-attention not available on Windows (no prebuilt wheel); "
                "continuing on torch fallback"
            )
            logger.debug("Install output:\n%s", result.stdout)
        else:
            logger.warning(
                "flash-linear-attention install failed (continuing on torch fallback):\n%s",
                result.stdout,
            )
        _send_status(
            event_queue,
            "flash-linear-attention install failed; continuing without it",
        )
        return False

    # pip can exit 0 with a missing transitive runtime dep; verify the import.
    if not _flash_linear_attention_importable():
        _send_status(
            event_queue,
            "flash-linear-attention installed but is not importable; continuing without it",
        )
        return False

    logger.info("Installed flash-linear-attention for the FLA fast path")
    return True


def _ensure_flash_linear_attention(event_queue: Any, model_name: str) -> None:
    """Legacy model-name-gated FLA install, used when UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1."""
    if not _model_wants_tilelang(model_name):
        return
    _ensure_flash_linear_attention_unconditional(event_queue)


_SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
)


def _ensure_mamba_ssm(event_queue: Any, model_name: str) -> None:
    if not any(sub in model_name.lower() for sub in _SSM_MODEL_SUBSTRINGS):
        return

    logger.info("SSM model detected; setting up mamba-ssm after causal-conv1d")
    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = _MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = _MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


# Auto-derived from installed transformers: model_types whose modeling_*.py imports
# `from fla.*`. Empty when transformers can't be inspected -> skip tilelang pre-install.
_TRANSFORMERS_FLA_MODEL_TYPES_CACHE: frozenset[str] | None = None
_MODEL_NAME_SEP_CHARS = ("-", ".", "/", " ")


def _discover_fla_model_types() -> frozenset[str]:
    """Installed-transformers model_types whose modeling file imports `from fla.*`."""
    global _TRANSFORMERS_FLA_MODEL_TYPES_CACHE
    if _TRANSFORMERS_FLA_MODEL_TYPES_CACHE is not None:
        return _TRANSFORMERS_FLA_MODEL_TYPES_CACHE
    found: set[str] = set()
    try:
        import transformers
        models_root = Path(transformers.__file__).parent / "models"
        for modeling in models_root.glob("*/modeling_*.py"):
            try:
                src = modeling.read_text(encoding = "utf-8", errors = "ignore")
            except OSError:
                continue
            if "from fla." in src:
                found.add(modeling.parent.name)
    except Exception as exc:
        logger.debug("FLA model-type discovery skipped: %s", exc)
    _TRANSFORMERS_FLA_MODEL_TYPES_CACHE = frozenset(found)
    return _TRANSFORMERS_FLA_MODEL_TYPES_CACHE


def _model_wants_tilelang(model_name: str) -> bool:
    """True iff model_name normalizes to contain a discovered FLA model_type."""
    types = _discover_fla_model_types()
    if not types:
        return False
    name = model_name.lower()
    for sep in _MODEL_NAME_SEP_CHARS:
        name = name.replace(sep, "_")
    return any(t in name for t in types)


def _installed_tvm_ffi_version() -> str | None:
    """Installed apache-tvm-ffi version, or None if missing/unimportable."""
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("apache-tvm-ffi")
    except Exception:
        return None


def _tilelang_importable() -> bool:
    """Catch any exception (not just ImportError) so a broken native lib doesn't abort the worker."""
    try:
        import tilelang  # noqa: F401
        import tvm_ffi  # noqa: F401
        return True
    except Exception as exc:
        logger.warning(
            "tilelang/tvm_ffi is not importable; continuing with install/fallback: %s",
            exc,
        )
        return False


def _torch_has_hip() -> bool:
    """True iff torch is a ROCm build.

    `torch.version.hip` covers official PyTorch ROCm wheels; AMD SDK / Radeon
    wheels can leave it unset but still encode "rocm" in `torch.__version__`.
    """
    try:
        import torch as _torch
        return bool(
            getattr(_torch.version, "hip", None)
            or "rocm" in getattr(_torch, "__version__", "").lower()
        )
    except Exception:
        return False


def _rocm_classify_unified_memory(props: Any) -> tuple[str, bool]:
    """Classify a ROCm device as unified-memory (APU) or discrete.

    Returns ``(gcn_arch, is_unified)``:
    - ``gcn_arch``: canonical arch string (e.g. ``"gfx1151"``) when a known
      attribute is present, else ``""``.
    - ``is_unified``: ``True`` for AMD APUs with a shared GPU/system-RAM pool
      (gfx1150 Strix Point, gfx1151 Strix Halo, gfx1152 Krackan Point) — these
      need a lower ``set_per_process_memory_fraction`` cap to leave OS headroom.

    Classification priority:
    1. ``props.is_integrated`` truthy (hipDeviceProp_t.integrated -- the
       driver's own unified-memory answer; covers APUs beyond the hardcoded
       arch set, e.g. gfx1103 Phoenix iGPUs). Only ever upgrades to unified.
    2. ``gcnArchName`` / variant spellings (stable, naming-independent).
    3. Device-name substring match (last resort when all arch attrs absent;
       AMD SDK / Radeon wheels may not populate them):
         - gfx1150 Strix Point: ``Radeon 890M``, ``Radeon 880M``
         - gfx1151 Strix Halo / Gorgon Halo:  ``Radeon 8065S`` (Ryzen AI
                                Max+ 495), ``Radeon 8060S`` (Ryzen AI MAX+
                                395), ``Radeon 8050S`` (cut-down SKU)
         - gfx1152 Krackan Point: ``Radeon 860M``, ``Radeon 840M``
    """
    gcn_arch = ""
    for _attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        _v = (getattr(props, _attr, "") or "").split(":")[0].strip()
        if _v:
            gcn_arch = _v
            break

    # Driver's own answer first: hipDeviceProp_t.integrated (props.is_integrated, the same
    # gate PR #5988's UMA safetensors fast-load uses). Strictly additive -- only a truthy
    # value upgrades to unified, so a wheel that omits the field can't downgrade the known
    # APU set. Covers unified APUs outside the hardcoded arches (gfx1103 Phoenix, future).
    if getattr(props, "is_integrated", 0):
        return gcn_arch, True

    if gcn_arch:
        # gfx1152 is Krackan Point: same shared GPU/system-RAM pool as gfx1150/gfx1151.
        # Case-folded: the attribute is lowercase in practice but is not guaranteed.
        return gcn_arch, gcn_arch.lower() in {"gfx1150", "gfx1151", "gfx1152"}

    # Arch attrs absent -- fall back to device-name matching. Only reached under _hw.IS_ROCM,
    # so the NVIDIA GeForce 840M cannot collide with the Krackan markers.
    dev_lower = (getattr(props, "name", "") or "").lower()
    is_unified = (
        "890m" in dev_lower
        or "880m" in dev_lower
        or "8065s" in dev_lower
        or "8060s" in dev_lower
        or "8050s" in dev_lower
        or "860m" in dev_lower
        or "840m" in dev_lower
    )
    return gcn_arch, is_unified


# 16 GiB, not a percentage: on a 128 GiB Strix Halo a flat 20% withholds 25.6 GiB, while
# 0.90 there reserves 12.8 GiB and was measured as OS-starving. The constant sits between.
_UNIFIED_OS_RESERVE_BYTES = 16 * 1024**3
_UNIFIED_MAX_RESERVE_FRACTION = 0.20
_DISCRETE_MEM_FRACTION = 0.90
_MEM_FRACTION_ENV = "UNSLOTH_ROCM_MEM_FRACTION"


def _parse_mem_fraction_env(env_value: str | None) -> float | None:
    """``UNSLOTH_ROCM_MEM_FRACTION`` as a float, None when unset or unusable.

    Shared with the OOM guard's log line so it can say whether the override was
    actually honoured, rather than just whether the variable was set.
    """
    try:
        override = float(env_value)  # None -> TypeError, "" / "  " -> ValueError
    except (TypeError, ValueError):
        return None
    # Two-sided on purpose: NaN loses every comparison, so this rejects it. A one-sided
    # `override <= 0.0 or override > 1.0` would pass NaN to set_per_process_memory_fraction.
    return override if 0.0 < override <= 1.0 else None


def _allocator_divides_by_props_total(torch_version: str | None) -> bool:
    """Whether ``set_per_process_memory_fraction`` scales ``props.total_memory``.

    c10's ``CUDACachingAllocator::setMemoryFraction`` caps at
    ``fraction * device_prop.totalGlobalMem`` from torch 2.10, and at
    ``fraction * hipMemGetInfo total`` through 2.9. Unparsable versions answer True,
    so a surprise string keeps today's denominator rather than switching it.
    """
    release = str(torch_version or "").split("+", 1)[0].split(".")
    try:
        major, minor = int(release[0]), int(release[1])
    except (IndexError, ValueError):
        return True
    return (major, minor) >= (2, 10)


def _rocm_memory_fraction(
    total_bytes: int,
    is_unified: bool,
    platform: str,
    env_value: str | None = None,
    denominator_bytes: int | None = None,
) -> float:
    """Pick the ``set_per_process_memory_fraction`` cap for a ROCm device.

    ``total_bytes`` is the pool the reserve comes out of, always
    ``get_device_properties().total_memory``: on a unified APU that is what the OS
    shares, while ``hipMemGetInfo``'s total is a runtime budget spanning GTT.

    ``denominator_bytes`` is what the allocator multiplies the fraction by, when that
    is a different number (see ``_allocator_divides_by_props_total``). An absolute
    byte reserve only lands where intended if the two agree, so passing it re-solves
    the cap for the same allowed bytes, floored at the historical cap so a larger
    driver total can never leave this tighter than the 0.80 it replaced.

    - ``env_value`` (``UNSLOTH_ROCM_MEM_FRACTION``) wins when it parses to a
      float in ``(0.0, 1.0]``; anything else is ignored, never fatal.
    - Unified + win32: ``1.0``. The WDDM budget already excludes the OS share,
      so any sub-1.0 cap double-taxes it (see the guard's own comment).
    - Unified elsewhere: reserve ``min(_UNIFIED_MAX_RESERVE_FRACTION of total,
      _UNIFIED_OS_RESERVE_BYTES)``, then clamp the cap to ``_DISCRETE_MEM_FRACTION``
      so a huge pool never ends up looser than a discrete card. The percentage
      ceiling keeps small pools at exactly the historical cap.
    - Discrete: ``_DISCRETE_MEM_FRACTION``.
    """
    override = _parse_mem_fraction_env(env_value)
    if override is not None:
        return override

    if not is_unified:
        return _DISCRETE_MEM_FRACTION
    if platform == "win32":
        return 1.0
    if total_bytes <= 0:
        # The caller defaults a missing or None total to 0; with no pool size there is
        # nothing to solve against, so keep the historical cap.
        return 1.0 - _UNIFIED_MAX_RESERVE_FRACTION

    # Solved in fraction space, not bytes: (total - 0.20 * total) / total rounds
    # to 0.7999999999999999 on some pool sizes (12/24/28/48 GiB), which would
    # break the "never tighter than the historical 0.80" guarantee by a ULP.
    reserve_fraction = min(_UNIFIED_MAX_RESERVE_FRACTION, _UNIFIED_OS_RESERVE_BYTES / total_bytes)
    fraction = 1.0 - reserve_fraction

    if (
        reserve_fraction < _UNIFIED_MAX_RESERVE_FRACTION
        and denominator_bytes
        and denominator_bytes > 0
        and denominator_bytes != total_bytes
    ):
        # Re-solve for the same allowed bytes against the total the allocator scales.
        # Floored, so a larger driver total cannot leave a host tighter than the 0.80
        # this replaced. Byte arm only: the percentage arm is scale-free, and those
        # small pools are the OOM-prone ones that must stay exactly as they were.
        fraction = max(
            fraction * total_bytes / denominator_bytes,
            1.0 - _UNIFIED_MAX_RESERVE_FRACTION,
        )

    # Past ~160 GiB the byte reserve is under 10% of the pool, which would hand a unified
    # host a looser cap than a discrete card and invert the ordering the guard is built on.
    return min(fraction, _DISCRETE_MEM_FRACTION)


def _tilelang_platform_supported() -> bool:
    """True iff a tilelang 0.1.8 wheel will load: Linux x86_64/aarch64, non-HIP torch.

    HIP excluded: tilelang 0.1.8 has no HIP GEMM and crashes mid-backward.
    """
    import platform as _platform

    if not sys.platform.startswith("linux"):
        return False
    if _platform.machine().lower() not in _TILELANG_SUPPORTED_LINUX_MACHINES:
        return False
    if _torch_has_hip():
        return False
    return True


def _pip_install_cmd(*args: str) -> list[str]:
    """`uv pip install` if uv is on PATH, else `python -m pip install`."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", *args]


def _run_pip(cmd: list[str], event_queue: Any, label: str) -> bool:
    """Run a pip install and surface success/failure via status events."""
    try:
        result = _sp.run(
            cmd,
            stdout = _sp.PIPE,
            stderr = _sp.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = utf8_child_env(),
            timeout = _TILELANG_INSTALL_TIMEOUT_S,
        )
    except _sp.TimeoutExpired:
        logger.warning("%s install timed out; continuing", label)
        _send_status(event_queue, f"{label} install timed out; continuing")
        return False
    if result.returncode != 0:
        logger.warning("%s install failed (continuing without it):\n%s", label, result.stdout)
        _send_status(event_queue, f"{label} install failed; continuing")
        return False
    return True


def _ensure_tilelang_backend_unconditional(event_queue: Any) -> bool:
    """Install pinned tilelang + apache-tvm-ffi; two-step repair if a broken tvm-ffi is present.

    Returns True iff both import post-call. Step 1 downgrades a broken tvm-ffi
    with --force-reinstall --no-deps so torch / CUDA stay untouched; step 2 is a
    regular install for missing transitive deps. Bypass via
    UNSLOTH_STUDIO_SKIP_TILELANG_INSTALL=1.
    """
    if os.getenv(_TILELANG_SKIP_ENV) == "1":
        return False
    if sys.version_info < _FLA_MIN_PYTHON:
        logger.info(
            "Skipping tilelang install: requires Python >= %d.%d, have %s",
            _FLA_MIN_PYTHON[0],
            _FLA_MIN_PYTHON[1],
            sys.version.split()[0],
        )
        return False
    if not _tilelang_platform_supported():
        import platform as _platform
        logger.info(
            "Skipping tilelang install: no prebuilt wheel for %s/%s",
            sys.platform,
            _platform.machine(),
        )
        return False

    existing_tvm_ffi = _installed_tvm_ffi_version()
    needs_repair = existing_tvm_ffi in _TVM_FFI_BROKEN_VERSIONS

    if not needs_repair and _tilelang_importable():
        logger.info("tilelang + apache-tvm-ffi already installed")
        return True

    if _model_offline_mode_enabled():
        if needs_repair and os.environ.get("FLA_TILELANG") is None:
            os.environ["FLA_TILELANG"] = "0"
            logger.warning(
                "Disabling TileLang while offline because apache-tvm-ffi %s is unsafe",
                existing_tvm_ffi,
            )
        logger.info("Skipping TileLang installation while offline")
        return False

    # Step 1: --no-deps keeps --force-reinstall off torch/CUDA via the dep graph.
    if needs_repair:
        logger.info(
            "Forcing apache-tvm-ffi downgrade: %s is on the broken list",
            existing_tvm_ffi,
        )
        _send_status(
            event_queue,
            (
                f"Downgrading apache-tvm-ffi {existing_tvm_ffi} -> "
                f"{_APACHE_TVM_FFI_PACKAGE_VERSION} (broken-versions list)"
            ),
        )
        repair_cmd = _pip_install_cmd(
            "--only-binary=:all:",
            "--force-reinstall",
            "--no-deps",
            f"apache-tvm-ffi=={_APACHE_TVM_FFI_PACKAGE_VERSION}",
        )
        if not _run_pip(repair_cmd, event_queue, "TileLang backend repair"):
            return False

    # Step 2: regular install pulls transitive deps (z3-solver, ml-dtypes) without touching torch.
    _send_status(
        event_queue,
        f"Installing TileLang=={_TILELANG_PACKAGE_VERSION} for faster training...",
    )
    install_cmd = _pip_install_cmd(
        "--only-binary=:all:",
        f"apache-tvm-ffi=={_APACHE_TVM_FFI_PACKAGE_VERSION}",
        f"tilelang=={_TILELANG_PACKAGE_VERSION}",
    )
    if not _run_pip(install_cmd, event_queue, "TileLang backend"):
        return False

    # pip can exit 0 while a native lib (libz3.so) is missing; verify the import.
    if not _tilelang_importable():
        _send_status(
            event_queue,
            "TileLang backend installed but is not importable; continuing on the FLA Triton path",
        )
        return False

    logger.info("Installed TileLang backend for FLA fast path")
    return True


def _ensure_tilelang_backend(event_queue: Any, model_name: str) -> None:
    """Legacy substring-gated tilelang installer (opt-out path)."""
    if not _model_wants_tilelang(model_name):
        return
    _ensure_tilelang_backend_unconditional(event_queue)


# ── Fast-path hooks ──
# Wrap transformers' is_{flash_linear_attention,causal_conv1d}_available so the first call
# (at modeling import) drives the install; models that never query the gate pay nothing.
# UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring path.


def _rebind_in_already_imported_modules(*, attr_name: str, old_obj: Any, new_obj: Any) -> int:
    """Rebind `attr_name -> new_obj` in every module that imported `old_obj`.

    `from X import Y` creates a local binding that reassigning X.Y won't reach.
    Uses `__dict__.get` to skip lazy `__getattr__` aliases.
    """
    count = 0
    missing = object()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        module_dict = getattr(mod, "__dict__", None)
        if not isinstance(module_dict, dict):
            continue
        existing = module_dict.get(attr_name, missing)
        if existing is old_obj:
            try:
                setattr(mod, attr_name, new_obj)
                count += 1
            except Exception as exc:
                logger.debug("Could not rebind %s in %s: %s", attr_name, mod_name, exc)
    return count


def _install_fast_path_hooks(
    event_queue: Any,
    model_name: str,
    *,
    install_causal_conv1d: bool | None = None,
) -> None:
    """Hook transformers' is_*_available gates so the first call drives the install.

    Idempotent. UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring gate.
    """
    if os.getenv(_FAST_PATH_HOOKS_SKIP_ENV) == "1":
        logger.info("Fast-path hooks disabled via env; using substring fallback")
        return

    # On HIP torch even installed tilelang crashes FLA's dispatch; override with FLA_TILELANG=1.
    if _torch_has_hip() and os.environ.get("FLA_TILELANG") is None:
        os.environ["FLA_TILELANG"] = "0"
        logger.info(
            "HIP/ROCm torch detected; setting FLA_TILELANG=0 (no HIP GEMM in tilelang 0.1.8)"
        )

    try:
        from transformers.utils import import_utils as _iu
    except Exception as exc:
        logger.warning(
            "transformers.utils.import_utils not importable; skipping fast-path hooks: %s",
            exc,
        )
        return

    def _make_wrapper(
        original: Callable[[], bool],
        install_fn: Callable[[Any], bool],
        gate_name: str,
        post_available_fn: Callable[[Any], None] | None = None,
    ) -> Callable[[], bool]:
        state = {"installed": False}

        def wrapper() -> bool:
            if state["installed"]:
                return original()
            try:
                original.cache_clear()  # defensive; worker subprocess is fresh
            except AttributeError:
                pass
            ok = original()
            ran_install = False
            if not ok:
                ran_install = True
                logger.info("Hook fired for %s; triggering install", gate_name)
                try:
                    ok = bool(install_fn(event_queue))
                except Exception as exc:
                    logger.warning("%s install raised: %s; falling back to torch", gate_name, exc)
                    ok = False
                logger.info("%s hook done; available=%s", gate_name, ok)
            # Handles "gate already True but ancillary kernel broken" (tilelang missing while FLA imports).
            if ok and not ran_install and post_available_fn is not None:
                try:
                    post_available_fn(event_queue)
                except Exception as exc:
                    logger.warning("%s post-available step raised: %s; continuing", gate_name, exc)
            state["installed"] = True
            return ok

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        wrapper.cache_clear = getattr(original, "cache_clear", lambda: None)  # type: ignore[attr-defined]
        return wrapper

    def _fla_install(eq: Any) -> bool:
        # FLA alone ~2.35x; +tilelang adds ~26%. tilelang is GDN-only (Qwen3.5 family).
        if not _ensure_flash_linear_attention_unconditional(eq):
            logger.info("FLA install did not produce an importable runtime; skipping TileLang")
            return False
        if _model_wants_tilelang(model_name):
            _ensure_tilelang_backend_unconditional(eq)
        else:
            logger.info(
                "Model %r outside TileLang allowlist; FLA Triton path is sufficient",
                model_name,
            )
        return True

    def _fla_post_available(eq: Any) -> None:
        # FLA imports; repair tilelang if missing or on the broken tvm-ffi list.
        if not _model_wants_tilelang(model_name):
            return
        if _installed_tvm_ffi_version() not in _TVM_FFI_BROKEN_VERSIONS and _tilelang_importable():
            return
        _ensure_tilelang_backend_unconditional(eq)

    def _causal_conv1d_install(eq: Any) -> bool:
        if sys.platform == "win32":
            logger.info("causal-conv1d: no prebuilt wheel for Windows; skipping")
            return False
        ok = _install_package_wheel_first(
            event_queue = eq,
            import_name = "causal_conv1d",
            display_name = "causal-conv1d",
            pypi_name = "causal-conv1d",
            pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
            filename_prefix = "causal_conv1d",
            release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
            release_base_url = ("https://github.com/Dao-AILab/causal-conv1d/releases/download"),
        )
        return bool(ok)

    hooks = [
        ("is_flash_linear_attention_available", _fla_install, _fla_post_available),
    ]
    if install_causal_conv1d is None:
        install_causal_conv1d = _model_wants_causal_conv1d(model_name)
    if install_causal_conv1d:
        hooks.append(("is_causal_conv1d_available", _causal_conv1d_install, None))

    for gate_name, install_fn, post_fn in hooks:
        original = getattr(_iu, gate_name, None)
        if original is None:
            logger.info(
                "%s missing on transformers.utils.import_utils; skipping hook",
                gate_name,
            )
            continue
        wrapped = _make_wrapper(original, install_fn, gate_name, post_fn)
        setattr(_iu, gate_name, wrapped)
        rebound = _rebind_in_already_imported_modules(
            attr_name = gate_name, old_obj = original, new_obj = wrapped
        )
        logger.info("Installed fast-path hook on %s (rebound %d modules)", gate_name, rebound)


def _should_try_runtime_flash_attn_install(max_seq_length: int) -> bool:
    if os.getenv(_FLASH_ATTN_SKIP_ENV) == "1":
        return False
    if max_seq_length < _FLASH_ATTN_RUNTIME_MIN_SEQ_LEN:
        return False
    return sys.platform.startswith("linux")


def _ensure_flash_attn_for_long_context(event_queue: Any, max_seq_length: int) -> None:
    if not _should_try_runtime_flash_attn_install(max_seq_length):
        return

    installed = _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "flash_attn",
        display_name = "flash-attn",
        pypi_name = "flash-attn",
        wheel_url_builder = flash_attn_wheel_url,
        pypi_spec = "flash-attn",
        pypi_status_message = "Installing flash-attn from PyPI for long-context training...",
    )
    if not installed:
        _send_status(event_queue, "Continuing without flash-attn")


def _activate_transformers_version(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name, hf_token)


def _activate_transformers_version_or_warn(model_name: str, hf_token: str | None = None) -> None:
    """Activate the required transformers version for the MLX fast-path.

    Unlike the non-MLX path (which treats activation failure as fatal and
    reports it via the event queue), the MLX path is intentionally non-fatal:
    it falls through with whatever transformers version is installed. The
    failure used to be swallowed by a bare ``except: pass``, leaving no trace
    and only a confusing downstream crash. Log a warning instead so the cause
    is visible, while keeping the fall-through behaviour.
    """
    try:
        _activate_transformers_version(model_name, hf_token)
    except Exception as exc:
        logger.warning(
            "Failed to activate transformers version for '%s' (MLX); "
            "training may fail if this model requires a specific version. Error: %s",
            model_name,
            exc,
        )


def _mlx_vlm_max_resized_size(width: int, height: int, target: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or target <= 0:
        return width, height
    largest_side = max(width, height)
    if largest_side <= target:
        return width, height
    # Integer formula matches unsloth_zoo's collator (Python round() differs by
    # 1px on half-pixel cases). max(1, _) avoids a zero-side degenerate output.
    new_w = max(1, (width * target + largest_side // 2) // largest_side)
    new_h = max(1, (height * target + largest_side // 2) // largest_side)
    return new_w, new_h


_MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE = {}


def _mlx_vlm_resized_image_layout(processor = None) -> str | None:
    """Return the numpy image layout expected after Unsloth-side VLM resizing."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None
    cls = image_processor.__class__
    key = (getattr(cls, "__module__", ""), getattr(cls, "__qualname__", cls.__name__))
    if key in _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE:
        return _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE[key]
    copied_image_processor = _copy_mlx_vlm_image_processor(image_processor)
    layout = (
        _probe_mlx_vlm_numpy_image_layout(copied_image_processor)
        if copied_image_processor is not None
        else None
    )
    _MLX_VLM_RESIZED_IMAGE_LAYOUT_CACHE[key] = layout
    return layout


def _copy_mlx_vlm_image_processor(image_processor):
    import copy
    try:
        return copy.deepcopy(image_processor)
    except Exception:
        try:
            return copy.copy(image_processor)
        except Exception:
            return None


def _probe_mlx_vlm_numpy_image_layout(image_processor) -> str | None:
    try:
        import numpy as np
    except ImportError:
        return None

    def _accepts(candidate) -> bool:
        try:
            image_processor(images = [candidate])
            return True
        except TypeError:
            try:
                image_processor([candidate])
                return True
            except Exception:
                return False
        except Exception:
            return False

    # Asymmetric image so CHW-vs-HWC mistakes are visible to processors that skip 3D conversion.
    hwc = np.zeros((64, 96, 3), dtype = np.uint8)
    chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))
    if _accepts(hwc):
        return None
    if _accepts(chw):
        return "chw"
    return None


def _resize_mlx_vlm_image(
    image,
    resize,
    image_layout = None,
):
    if resize is None:
        return image
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return image
    if not isinstance(image, Image.Image):
        return image
    image = image.convert("RGB")
    new_size = _mlx_vlm_max_resized_size(*image.size, int(resize))
    if new_size != image.size:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(new_size, resampling)
    # On resize, hand mlx-vlm a writable RGB ndarray so its PIL-path square-resize is skipped
    # and HF processors don't warn on non-writable views. resize=None keeps the original PIL.
    array = np.array(image, copy = True)
    if image_layout == "chw":
        return np.ascontiguousarray(array.transpose(2, 0, 1))
    return array


def _resize_mlx_vlm_images(
    value,
    resize,
    image_layout = None,
):
    if isinstance(value, list):
        return [_resize_mlx_vlm_image(image, resize, image_layout = image_layout) for image in value]
    return _resize_mlx_vlm_image(value, resize, image_layout = image_layout)


def _adapt_for_mlx_vlm(
    items,
    resize = None,
    image_layout = None,
):
    """Adapt GPU-path VLM dataset output for mlx-vlm.

    The GPU path embeds PIL images in message content as
    {"type": "image", "image": PIL_Image}, but mlx-vlm's prepare_inputs needs
    images at top-level to produce pixel_values (any model type). Extract them
    and leave bare {"type": "image"} placeholders.
    """
    adapted = []
    for item in items:
        images = []
        messages = []
        for msg in item.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part.get("image")
                        if img is not None:
                            images.append(
                                _resize_mlx_vlm_image(
                                    img,
                                    resize,
                                    image_layout = image_layout,
                                )
                            )
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(part)
                messages.append({"role": msg["role"], "content": new_content})
            else:
                messages.append(msg)
        out = {"messages": messages}
        if images:
            out["image"] = images[0] if len(images) == 1 else images
        elif "image" in item:
            out["image"] = _resize_mlx_vlm_images(
                item["image"],
                resize,
                image_layout = image_layout,
            )
        elif "images" in item:
            out["images"] = _resize_mlx_vlm_images(
                item["images"],
                resize,
                image_layout = image_layout,
            )
        adapted.append(out)
    return adapted


_MLX_STUDIO_LR_SCHEDULERS = {"linear", "cosine", "constant"}


# Fallback alias map mirroring unsloth_zoo._normalize_mlx_optimizer_name, used only when
# mlx isn't importable. The zoo function stays the source of truth.
_MLX_STUDIO_ADAMW_ALIASES = frozenset(
    (
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_bnb_8bit",
        "paged_adamw_32bit",
        "adamw_torch",
        "adamw_torch_fused",
        "paged_adamw",
        "adamw_32bit",
        "adamw_hf",
        "adamw_anyprecision",
        "adamw_apex_fused",
    )
)
_MLX_STUDIO_NATIVE_OPTIMIZERS = ("adafactor", "adamw", "adam", "sgd", "muon", "lion")


def _normalize_mlx_studio_optimizer(value):
    try:
        from unsloth_zoo.mlx.trainer import _normalize_mlx_optimizer_name
        return _normalize_mlx_optimizer_name(value or "adamw_8bit")
    except (ImportError, ValueError):
        # Missing mlx, or an older zoo normalizer: map common adamw_* names locally.
        opt = str(getattr(value, "value", value) or "adamw_8bit").strip().lower()
        opt = opt.rsplit(".", 1)[-1].replace("-", "_")
        if opt in _MLX_STUDIO_ADAMW_ALIASES:
            opt = "adamw"
        if opt not in _MLX_STUDIO_NATIVE_OPTIMIZERS:
            supported = ", ".join(_MLX_STUDIO_NATIVE_OPTIMIZERS)
            raise ValueError(
                f"Unsupported optimizer for MLX training: {value!r}. "
                f"Supported optimizers: {supported}."
            )
        return opt


def _normalize_mlx_studio_scheduler(value):
    raw = str(value or "linear").strip().lower()
    if raw not in _MLX_STUDIO_LR_SCHEDULERS:
        supported = ", ".join(sorted(_MLX_STUDIO_LR_SCHEDULERS))
        raise ValueError(
            f"Unsupported LR scheduler for MLX training: {value!r}. Supported values: {supported}."
        )
    return raw


def _resolve_mlx_local_dataset_files(file_paths: list) -> list[str]:
    """Resolve CLI paths and Unsloth local dataset uploads without importing the GPU trainer."""
    from utils.paths import dataset_files_in_dir, resolve_dataset_path

    all_files: list[str] = []
    for dataset_file in file_paths or []:
        dataset_path = Path(os.path.expanduser(str(dataset_file)))
        if dataset_path.is_absolute():
            file_path = str(dataset_path)
        elif dataset_path.exists():
            file_path = str(dataset_path.resolve())
        else:
            file_path = str(resolve_dataset_path(str(dataset_file)))
        file_path_obj = Path(file_path)

        if file_path_obj.is_dir():
            all_files.extend(str(p) for p in dataset_files_in_dir(file_path_obj))
            continue

        all_files.append(str(file_path_obj))

    return all_files


def _mlx_local_dataset_loader_for_files(files: list[str]) -> str:
    first_ext = Path(files[0]).suffix.lower()
    if first_ext in (".json", ".jsonl"):
        return "json"
    if first_ext == ".csv":
        return "csv"
    if first_ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported dataset format: {files[0]}")


_MLX_WORKER_COMPLETE = "_mlx_worker_complete"


def _start_worker_stop_poller(
    stop_queue,
    on_stop: Callable[[bool], None],
    *,
    completion_type: str | None = None,
    timeout: float = 1.0,
):
    import queue as _queue
    import threading

    cancel_requested = False

    def poll_stop():
        nonlocal cancel_requested
        while True:
            try:
                msg = stop_queue.get(timeout = timeout)
                if not isinstance(msg, dict):
                    continue
                message_type = msg.get("type")
                if completion_type is not None and message_type == completion_type:
                    return
                if message_type != "stop":
                    continue
                if not bool(msg.get("save", True)):
                    cancel_requested = True
                on_stop(not cancel_requested)
                if cancel_requested:
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return

    stop_thread = threading.Thread(target = poll_stop, daemon = True)
    stop_thread.start()
    return stop_thread


def _start_mlx_stop_poller(stop_queue):
    stop_save = [True]
    stop_requested = [False]
    trainer_ref = [None]

    def is_stop_requested():
        return stop_requested[0]

    def apply_stop(save: bool) -> None:
        stop_save[0] = save
        stop_requested[0] = True
        trainer = trainer_ref[0]
        if trainer is not None:
            trainer.stop_requested = True

    stop_thread = _start_worker_stop_poller(
        stop_queue,
        apply_stop,
        completion_type = _MLX_WORKER_COMPLETE,
        timeout = 0.25,
    )
    return stop_save, stop_requested, trainer_ref, is_stop_requested, stop_thread


def _resolve_mlx_output_dir(config, model_name):
    from utils.paths import resolve_output_dir, default_run_dir_name

    output_dir = config.get("output_dir", "")
    if not output_dir:
        output_dir = f"{default_run_dir_name(model_name)}_{int(time.time())}"
        return str(resolve_output_dir(output_dir))
    if config.get("allow_external_output_dir"):
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return str(output_path.resolve())
    return str(resolve_output_dir(output_dir))


def _resolve_mlx_max_grad_norm(value):
    """Global-norm clip threshold for MLX runs; None keeps the trainer's default.

    The worker used to hardcode 0.0 and drop the requested value, so an API caller
    asking for a threshold got none. Unset stays 0.0 so MLX keeps its cheap
    per-parameter clipping: the gradient-norm chart is fed by report_grad_norm
    instead, which measures the same norm without changing what gets clipped.
    """
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Unsloth MLX: max_grad_norm={value!r} must be a non-negative float or None."
        )
    # inf clears a >= 0 check but never binds, so the run would train unclipped.
    if value < 0 or not math.isfinite(value):
        raise ValueError(
            f"Unsloth MLX: max_grad_norm={value} must be a finite value >= 0 "
            "(use 0 to disable global norm clipping)."
        )
    return value


def _run_mlx_training(event_queue, stop_queue, config):
    """Self-contained MLX training path for Apple Silicon.

    Uses unsloth_zoo's MLXTrainer directly (no torch/SFTTrainer). Mirrors the
    event_queue protocol so the parent process pump works unchanged.
    """
    import time
    import math
    from pathlib import Path

    def _send(event_type, **kwargs):
        if event_type == "status" and "message" not in kwargs:
            sm = kwargs.get("status_message")
            if sm is not None:
                kwargs["message"] = sm
        event_queue.put({"type": event_type, "ts": time.time(), **kwargs})

    _stop_save, _stop_requested, _trainer_ref, _is_stop_requested, _stop_thread = (
        _start_mlx_stop_poller(stop_queue)
    )

    _send("status", status_message = "Loading MLX libraries...")

    import mlx.core as mx

    try:
        from unsloth_zoo.mlx.loader import FastMLXModel
        from unsloth_zoo.mlx.trainer import (
            MLXTrainer,
            MLXTrainingConfig,
            train_on_responses_only,
        )
    except ImportError as e:
        raise ImportError(
            "Unsloth: MLX training requires unsloth-zoo with the MLX modules "
            "(unsloth_zoo.mlx.loader / unsloth_zoo.mlx.trainer). Reinstall via "
            "install.sh on Apple Silicon."
        ) from e
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset

    if mx.metal.is_available():
        info = mx.device_info()
        rec_bytes = info.get("max_recommended_working_set_size", 0) or 0
        if rec_bytes > 0:
            memory_cap = int(rec_bytes * 0.85)
            wired_cap = min(int(rec_bytes), memory_cap)
            mx.set_memory_limit(memory_cap)
            mx.set_wired_limit(wired_cap)

    model_name = config["model_name"]
    hf_token = config.get("hf_token") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    model_load_name = _resolve_cached_model_load_name(config)
    model_local_only = _model_local_files_only(config)
    model_revision = None if model_local_only else config.get("model_revision")

    if config.get("use_loftq"):
        message = "LoftQ is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)
    if config.get("use_dora"):
        message = "DoRA is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)
    if config.get("is_embedding"):
        message = "Embedding model training is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)
    if config.get("training_type") == "Continued Pretraining":
        message = "Continued Pretraining is not supported for MLX training yet."
        _send("error", error = message)
        raise NotImplementedError(message)

    optim_name = _normalize_mlx_studio_optimizer(config.get("optim", "adamw_8bit"))
    lr_scheduler_type = _normalize_mlx_studio_scheduler(config.get("lr_scheduler_type", "linear"))

    # ── 1. Load model ──
    # Force text-only for non-image datasets even on vision-capable models (e.g. Qwen3.5-VL on alpaca).
    _send("status", status_message = f"Loading {model_name}...")
    # Pull through resume_from_checkpoint so MLXTrainer.train() can restore optimizer + step
    # state. Previously dropped on the MLX path, so the Resume button silently restarted from
    # step 0 (the CUDA path has been forwarding it all along).
    resume_from_checkpoint = config.get("resume_from_checkpoint") or None
    is_dataset_image = bool(config.get("is_dataset_image", False))
    training_type = config.get("training_type", "LoRA/QLoRA")
    use_lora = training_type == "LoRA/QLoRA"
    # Before the download/load below: unlike the CUDA path, this needs none of the model.
    if use_lora:
        _check_mlx_finetune_targets(config)
    # Normalize seed; explicit None must not reach the seed chain.
    _raw_seed = config.get("random_seed", 3407)
    random_seed = 3407 if _raw_seed is None else int(_raw_seed)
    # `config.get(k, d)` only fills d when key is missing; handle explicit None too.
    _model_seed = config.get("model_random_state")
    model_random_state = random_seed if _model_seed is None else int(_model_seed)
    _lora_seed = config.get("lora_random_state")
    lora_random_state = random_seed if _lora_seed is None else int(_lora_seed)

    security_error = _model_load_security_error(config, model_load_name, hf_token)
    if security_error:
        _send("error", **security_error)
        return

    try:
        model, tokenizer = FastMLXModel.from_pretrained(
            model_load_name,
            load_in_4bit = config.get("load_in_4bit", True),
            full_finetuning = not use_lora,
            text_only = None if is_dataset_image else True,
            token = hf_token,
            trust_remote_code = bool(config.get("trust_remote_code", False)),
            random_state = model_random_state,
            revision = model_revision,
        )
    except Exception as error:
        if not model_local_only:
            raise
        fallback_error = _model_cache_fallback_error(config, error)
        if fallback_error is not None:
            raise fallback_error from error
        if not _cache_artifact_fallback_allowed(config, error, "model"):
            raise
        revision_error = _mlx_revision_fallback_error(config)
        if revision_error is not None:
            raise revision_error from error
        _send(
            "status",
            status_message = (
                f"Cached files for {model_name} are incomplete; retrying from Hugging Face..."
            ),
        )
        model_load_name = _drop_model_pin_for_fallback(config, hf_token)
        # Scan the Hub target we fall back to, not the cached pin already scanned above.
        security_error = _model_load_security_error(config, model_load_name, hf_token)
        if security_error:
            _send("error", **security_error)
            return
        model_local_only = False
        model_revision = config.get("model_revision")
        model, tokenizer = FastMLXModel.from_pretrained(
            model_load_name,
            load_in_4bit = config.get("load_in_4bit", True),
            full_finetuning = not use_lora,
            text_only = None if is_dataset_image else True,
            token = hf_token,
            trust_remote_code = bool(config.get("trust_remote_code", False)),
            random_state = model_random_state,
            revision = model_revision,
        )

    from utils.models.model_identity import restore_hf_cache_repo_identity

    restored_repo_id = restore_hf_cache_repo_identity(
        model,
        model_load_name,
        expected_repo_id = config.get("actual_model_repo_id") or model_name,
    )
    if restored_repo_id:
        logger.info(
            "Restored Hub model identity for saved MLX adapter metadata: %s",
            restored_repo_id,
        )

    loaded_model_for_provenance = model
    is_vlm = bool(is_dataset_image and getattr(model, "_is_vlm_model", False))
    model._is_vlm_model = is_vlm
    vision_image_size = config.get("vision_image_size")
    # DeepSeek OCR uses a coupled preset tuple; skip resize like the Torch path.
    _model_name_lower = str(config.get("model_name", "")).lower()
    _is_deepseek_ocr = "deepseek" in _model_name_lower and "ocr" in _model_name_lower
    if is_vlm and vision_image_size is not None and _is_deepseek_ocr:
        _send(
            "status",
            status_message = (
                "MLX vision image resize ignored for DeepSeek OCR (uses fixed Gundam preset)."
            ),
        )
        vision_image_size = None
    elif is_vlm and vision_image_size is not None:
        vision_image_size = int(vision_image_size)
        _send(
            "status",
            status_message = f"MLX vision image resize: {vision_image_size} (max dimension)",
        )
    # ── 2. Apply LoRA / full FT ──
    # gradient_checkpointing stays a string; get_peft_model and MLXTrainer both accept strings.
    gc_setting = config.get("gradient_checkpointing", "mlx")
    if isinstance(gc_setting, str):
        use_grad_checkpoint = (
            gc_setting if gc_setting.lower() not in ("false", "none", "") else False
        )
    else:
        use_grad_checkpoint = gc_setting

    if use_lora:
        _send("status", status_message = "Configuring LoRA adapters...")
        peft_kwargs = dict(
            r = config.get("lora_r", 16),
            lora_alpha = config.get("lora_alpha", 16),
            lora_dropout = config.get("lora_dropout", 0.0),
            use_rslora = config.get("use_rslora", False),
            init_lora_weights = config.get("init_lora_weights", True),
            random_state = lora_random_state,
            target_modules = config.get("target_modules")
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing = use_grad_checkpoint,
        )
        finetune_language = config.get("finetune_language_layers", True)
        finetune_attention = config.get("finetune_attention_modules", True)
        finetune_mlp = config.get("finetune_mlp_modules", True)
        finetune_vision = config.get("finetune_vision_layers", False) if is_vlm else False

        if (finetune_attention or finetune_mlp) and not finetune_language and not finetune_vision:
            finetune_language = True

        # is_vlm and the back-fill's outcome are known now; the preflight could only guess.
        _check_mlx_effective_targets(
            config,
            finetune_language = finetune_language,
            finetune_vision = finetune_vision,
        )

        peft_kwargs["finetune_language_layers"] = finetune_language
        peft_kwargs["finetune_attention_modules"] = finetune_attention
        peft_kwargs["finetune_mlp_modules"] = finetune_mlp
        if is_vlm:
            peft_kwargs["finetune_vision_layers"] = finetune_vision
        model = FastMLXModel.get_peft_model(model, **peft_kwargs)

    # ── 3. Load dataset ──
    _send("status", status_message = "Loading dataset...")
    hf_dataset = config.get("hf_dataset", "")
    slice_start = config.get("dataset_slice_start")
    slice_end = config.get("dataset_slice_end")
    config["_dataset_loaded_from_exact_snapshot"] = False

    # A max_steps run cannot reach the whole dataset, and everything below here
    # (formatting, templating, tokenization) maps over every row. Recomputed from
    # the config, never carried over from the parent, so a bound can never be stale.
    # The vision branch is gated on `not raw_text_mode`, so a raw or CPT run takes
    # the text path, which honours the requested packing.
    mlx_raw_text_mode = (
        training_type == "Continued Pretraining" or config.get("format_type") == "raw"
    )
    # An mlx.launch run shards the batch across its processes the same way DDP does,
    # and it advertises the count in the env this reads.
    mlx_max_train_rows = max_train_rows_for_config(
        config,
        branch_never_packs = is_vlm and not mlx_raw_text_mode,
        world_size = _data_parallel_world_size(),
    )
    # MLXTrainer resumes by jumping a batch cursor into a schedule rebuilt from
    # whatever dataset it is handed, so bounding a checkpoint written without one
    # continues on unrelated rows. Same marker, same rule as the CUDA path.
    mlx_max_train_rows, mlx_max_train_rows_seed = row_bound_for_resume(
        resume_from_checkpoint, mlx_max_train_rows, random_seed
    )

    # A bracketed split names rows the same way the numeric fields do.
    mlx_split_names_rows = "[" in (config.get("train_split") or "")

    def _slice(ds):
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(ds) - 1
            if end < start:
                return ds.select([])
            # The user named these rows; the bound below defers to that.
            return ds.select(range(start, min(end + 1, len(ds))))
        if mlx_split_names_rows:
            return ds
        return bound_dataset_rows(
            ds,
            mlx_max_train_rows,
            mlx_max_train_rows_seed,
            on_bound = lambda kept, total: _send(
                "status",
                status_message = f"Using {kept} of {total} rows (max_steps run)",
            ),
        )

    def _load_local(file_paths):
        from datasets import load_from_disk

        if len(file_paths) == 1:
            p = Path(file_paths[0])
            if p.is_dir() and ((p / "dataset_info.json").exists() or (p / "state.json").exists()):
                return load_from_disk(str(p))
        all_files = _resolve_mlx_local_dataset_files(file_paths)
        if not all_files:
            raise ValueError("No local dataset files found")
        loader = _mlx_local_dataset_loader_for_files(all_files)
        return load_dataset(loader, data_files = all_files, split = "train")

    eval_dataset = None
    if hf_dataset:
        dataset, eval_dataset = _load_hf_train_and_eval_datasets(
            config,
            hf_token,
            load_dataset,
            lambda message: _send("status", status_message = message),
            lambda message: _send("warning", message = message),
        )
        dataset = _slice(dataset)
    elif config.get("local_datasets"):
        dataset = _load_local(config["local_datasets"])
        dataset = _slice(dataset)
    elif config.get("s3_config"):
        from core.training.s3_dataset import (
            S3DownloadCancelled,
            prepare_s3_dataset_download,
        )

        _send("status", status_message = "Downloading dataset from S3...")
        try:
            s3_download = prepare_s3_dataset_download(
                config["s3_config"],
                cancel_callback = _is_stop_requested,
            )
            try:
                dataset = _load_local(s3_download.files)
            finally:
                s3_download.cleanup()
        except S3DownloadCancelled:
            _send("complete", output_dir = None, status_message = "Training cancelled")
            return
        dataset = _slice(dataset)
    else:
        raise ValueError("No dataset specified")

    _emit_resource_provenance(
        event_queue,
        config,
        loaded_model_for_provenance,
        model_load_target = model_load_name,
        model_load_in_4bit = bool(config.get("load_in_4bit")),
        dataset_loaded_from_exact_snapshot = bool(config.get("_dataset_loaded_from_exact_snapshot")),
    )

    # Eval dataset (separate split or local file)
    from core.training.eval_dataset import evaluation_enabled

    eval_enabled = evaluation_enabled(config.get("eval_steps"))
    if eval_enabled and not hf_dataset and config.get("local_eval_datasets"):
        eval_dataset = _load_local(config["local_eval_datasets"])

    # ── 3b. Format dataset (VLM or text) ──
    # Reuse the GPU format pipeline for VLM (OCR/caption/llava/sharegpt+images) and text.
    format_type = config.get("format_type", "")
    custom_format_mapping = config.get("custom_format_mapping")
    dataset_final_format = ""
    try:
        from utils.datasets import format_and_template_dataset
        def _fmt_progress(status_message = "", **_kw):
            _send("status", status_message = status_message)

        if is_vlm:
            _send("status", status_message = "Formatting VLM dataset...")
            vlm_info = format_and_template_dataset(
                dataset,
                model_name = model_name,
                tokenizer = tokenizer,
                is_vlm = True,
                dataset_name = hf_dataset or "local",
                custom_format_mapping = custom_format_mapping,
                progress_callback = _fmt_progress,
            )
            if vlm_info.get("success"):
                vision_image_layout = (
                    _mlx_vlm_resized_image_layout(tokenizer)
                    if vision_image_size is not None
                    else None
                )
                dataset = _adapt_for_mlx_vlm(
                    vlm_info["dataset"],
                    resize = vision_image_size,
                    image_layout = vision_image_layout,
                )
            else:
                errors = vlm_info.get("errors", [])
                raise ValueError(f"VLM dataset format conversion failed: {'; '.join(errors)}")
            if eval_dataset is not None:
                ev_info = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = True,
                    dataset_name = hf_dataset or "local",
                    custom_format_mapping = custom_format_mapping,
                )
                if ev_info.get("success"):
                    vision_image_layout = (
                        _mlx_vlm_resized_image_layout(tokenizer)
                        if vision_image_size is not None
                        else None
                    )
                    eval_dataset = _adapt_for_mlx_vlm(
                        ev_info["dataset"],
                        resize = vision_image_size,
                        image_layout = vision_image_layout,
                    )

        elif format_type:
            _send("status", status_message = f"Formatting dataset ({format_type})...")
            info = format_and_template_dataset(
                dataset,
                model_name = model_name,
                tokenizer = tokenizer,
                is_vlm = False,
                format_type = format_type,
                dataset_name = hf_dataset or "local",
                custom_format_mapping = custom_format_mapping,
                progress_callback = _fmt_progress,
            )
            if info.get("success", True):
                dataset = info.get("dataset", dataset)
            dataset_final_format = str(info.get("final_format", "") or "").lower()
            if eval_dataset is not None:
                ev = format_and_template_dataset(
                    eval_dataset,
                    model_name = model_name,
                    tokenizer = tokenizer,
                    is_vlm = False,
                    format_type = format_type,
                    dataset_name = hf_dataset or "local",
                    custom_format_mapping = custom_format_mapping,
                )
                if ev.get("success", True):
                    eval_dataset = ev.get("dataset", eval_dataset)
    except ImportError:
        _send("status", status_message = "Format helper unavailable, using raw dataset")

    if eval_enabled and eval_dataset is None:
        from core.training.eval_dataset import (
            MIN_TOTAL_ROWS_FOR_EVAL,
            split_dataset_for_evaluation,
        )
        split_result = split_dataset_for_evaluation(dataset)
        if split_result is None:
            _send(
                "warning",
                message = (
                    f"Evaluation is enabled, but the training dataset has only {len(dataset)} "
                    f"rows; at least {MIN_TOTAL_ROWS_FOR_EVAL} are required to create a "
                    "held-out eval split. Training will continue without evaluation."
                ),
            )
        else:
            dataset, eval_dataset = split_result

    # ── 4. Resolve training steps ──
    max_steps = config.get("max_steps", 0) or 0
    num_epochs = config.get("num_epochs", 3)
    max_seq_length = config.get("max_seq_length", 2048)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("gradient_accumulation_steps", 4)

    if max_steps <= 0:
        max_steps = max(
            1,
            math.ceil(len(dataset) / batch_size / grad_accum) * num_epochs,
        )

    lr_value = float(config.get("learning_rate", "2e-4"))

    # Warmup: prefer warmup_steps; fall back to warmup_ratio
    warmup_steps = config.get("warmup_steps")
    warmup_ratio = config.get("warmup_ratio")
    if warmup_steps is None and warmup_ratio is not None:
        warmup_steps = int(round(warmup_ratio * max_steps))
    if warmup_steps is None:
        warmup_steps = 5

    # ── 5. Build output dir ──
    # Resolve to ~/.unsloth/studio/outputs/ so the export page finds it
    from utils.paths import ensure_dir

    # Resume must land in the original run dir even when config lacks output_dir.
    resume_dir = config.get("output_dir", "") or _output_dir_from_resume_checkpoint(
        resume_from_checkpoint
    )
    output_dir = _resolve_mlx_output_dir(
        {**config, "output_dir": resume_dir} if resume_dir else config, model_name
    )
    ensure_dir(Path(output_dir))
    _emit_output_dir(event_queue, output_dir)
    # Pin the subset before any checkpoint lands here; a resume reads it back.
    if not record_row_bound(output_dir, mlx_max_train_rows, mlx_max_train_rows_seed) and (
        mlx_max_train_rows
    ):
        _send(
            "warning",
            message = (
                f"Could not record the max_steps row bound in {output_dir}: "
                "resuming this run later will read it as unbounded"
            ),
        )

    # ── 6. Create trainer ──
    raw_eval_steps = config.get("eval_steps", 0)
    if evaluation_enabled(raw_eval_steps):
        eval_steps_value = float(raw_eval_steps)
    else:
        eval_steps_value = 0.0
    if 0 < eval_steps_value < 1:
        eval_steps_val = max(1, int(eval_steps_value * max_steps))
    else:
        eval_steps_val = int(eval_steps_value)

    # Re-validate for direct worker callers; training.py normalizes the main path.
    max_grad_norm = _resolve_mlx_max_grad_norm(config.get("max_grad_norm"))
    max_grad_value = config.get("max_grad_value")
    if max_grad_value is not None:
        max_grad_value = float(max_grad_value)
        if max_grad_value < 0 or not math.isfinite(max_grad_value):
            raise ValueError(
                f"Unsloth MLX: max_grad_value={max_grad_value} must be finite and >= 0 "
                "(0 or None disables elementwise clipping)."
            )
    max_grad_leaf_norm = config.get("max_grad_leaf_norm")
    if max_grad_leaf_norm is not None:
        max_grad_leaf_norm = float(max_grad_leaf_norm)
        if max_grad_leaf_norm < 0 or not math.isfinite(max_grad_leaf_norm):
            raise ValueError(
                f"Unsloth MLX: max_grad_leaf_norm={max_grad_leaf_norm} must be finite and >= 0 "
                "(0 or None disables proportional leaf-norm clipping)."
            )
    weight_decay = config.get("weight_decay", 0.001)
    weight_decay = 0.001 if weight_decay is None else float(weight_decay)

    mlx_config_kwargs = dict(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        max_steps = max_steps,
        learning_rate = lr_value,
        warmup_steps = warmup_steps,
        lr_scheduler_type = lr_scheduler_type,
        optim = optim_name,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        max_grad_value = max_grad_value,
        logging_steps = 1,
        max_seq_length = max_seq_length,
        seed = random_seed,
        use_cce = True,
        compile = True,
        gradient_checkpointing = use_grad_checkpoint,
        streaming = is_vlm,
        packing = bool(config.get("packing", False)),
        output_dir = output_dir,
        save_steps = int(config.get("save_steps", 0) or 0),
        eval_steps = eval_steps_val,
    )

    # Also gates the masking skip below, so defined outside the feature-detect block.
    raw_text_mode = training_type == "Continued Pretraining" or format_type == "raw"

    # Feature-detect optional fields so this PR works without the paired zoo bump.
    _supported_fields = getattr(MLXTrainingConfig, "__dataclass_fields__", {})
    if "cast_norm_output_to_input_dtype" in _supported_fields:
        # Explicit None falls back to True (default).
        _raw_cast = config.get("cast_norm_output_to_input_dtype", True)
        mlx_config_kwargs["cast_norm_output_to_input_dtype"] = (
            True if _raw_cast is None else bool(_raw_cast)
        )
    if "dataset_order" in _supported_fields:
        mlx_config_kwargs["dataset_order"] = "torch_randperm"
    if "max_grad_leaf_norm" in _supported_fields:
        mlx_config_kwargs["max_grad_leaf_norm"] = max_grad_leaf_norm
    if "report_grad_norm" in _supported_fields:
        # Refills the gradient-norm chart. MLX returns a norm for free only under
        # global-norm clipping; asking for it beats switching clip modes, which
        # would alter the loss trajectory and cost VLM runs mx.compile.
        mlx_config_kwargs["report_grad_norm"] = True
    if "append_eos" in _supported_fields:
        # Unsloth SFT formatting owns rendered examples; raw/CPT text still needs MLX to append EOS.
        mlx_config_kwargs["append_eos"] = bool(raw_text_mode)

    trainer = MLXTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        args = MLXTrainingConfig(**mlx_config_kwargs),
    )
    _trainer_ref[0] = trainer
    if _stop_requested[0]:
        trainer.stop_requested = True

    # Tell the parent eval is configured so the frontend shows the eval chart
    if eval_dataset is not None and eval_steps_val > 0:
        _send("eval_configured")

    # ── 7. Apply train_on_responses_only if requested ──
    # Auto-detect markers from the chat template first, manual table as fallback. Mirror the
    # CUDA skips: raw/CPT text has no chat turns.
    # Check the resolved format too, since format_type="auto" can land on alpaca or raw.
    if (
        config.get("train_on_completions", False)
        and not raw_text_mode
        and dataset_final_format != "raw_text"
    ):
        _send("status", status_message = "Configuring response-only training...")
        # No catch: the helper handles detection failures and double misses, so an exception here
        # is a real masking failure that must fail the run, not silently train full sequences.
        from utils.datasets.completion_masking import apply_completion_masking

        trainer, masking_applied = apply_completion_masking(
            trainer,
            model_name,
            train_on_responses_only,
            notify = lambda level, message: _send("status", status_message = message),
            dataset_template = "alpaca" if dataset_final_format == "alpaca" else None,
        )
        if not masking_applied:
            # A miss changes the training objective for the whole run, so it belongs in the
            # sticky warning list the eval-split fallback already uses, not a status line
            # that scrolls past. Recovered detection failures stay status: masking applied.
            _send(
                "warning",
                message = (
                    f"'Train on completions' could not be applied for {model_name}: no "
                    f"instruction/response markers were found. Training will run on full "
                    f"sequences (prompts included)."
                ),
            )

    # ── 8. Setup wandb / tensorboard ──
    wandb_run = None
    tb_writer = None
    if config.get("enable_wandb", False):
        try:
            import wandb as _wandb

            wandb_token = config.get("wandb_token")
            if wandb_token:
                os.environ["WANDB_API_KEY"] = wandb_token
            # Keep the authenticated subject out of W&B run config (mirrors _sanitize_db_config).
            _wandb_sensitive = {"hf_token", "wandb_token", "s3_config", "subject"}
            wandb_run = _wandb.init(
                project = config.get("wandb_project") or "unsloth-mlx",
                config = {k: v for k, v in config.items() if k not in _wandb_sensitive},
                reinit = True,
            )
        except Exception as e:
            _send("status", status_message = f"wandb init failed: {e}")
    if config.get("enable_tensorboard", False):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                SummaryWriter = None
        if SummaryWriter is not None:
            try:
                tb_dir = config.get("tensorboard_dir") or f"{output_dir}/runs"
                tb_writer = SummaryWriter(log_dir = tb_dir)
            except Exception as e:
                _send("status", status_message = f"tensorboard init failed: {e}")
        else:
            _send(
                "status",
                status_message = "tensorboard unavailable (install tensorboardX)",
            )

    # ── 9. Real-time progress callback ──
    _send("status", status_message = f"Training {model_name}...")

    def _on_step(
        step,
        total,
        loss,
        lr,
        tok_s,
        peak_gb,
        elapsed,
        num_tokens,
        grad_norm = None,
    ):
        eta = (elapsed / step * (total - step)) if step > 0 else 0
        _send(
            "progress",
            step = step,
            epoch = round(step / total * num_epochs, 2) if total > 0 else 0,
            loss = loss,
            learning_rate = lr,
            total_steps = total,
            elapsed_seconds = elapsed,
            eta_seconds = max(0, eta),
            grad_norm = grad_norm,
            num_tokens = num_tokens,
            eval_loss = None,
            status_message = None,
            peak_memory_gb = peak_gb,
        )
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/peak_gb": peak_gb,
                        "train/num_tokens": num_tokens,
                        **({"train/grad_norm": grad_norm} if grad_norm is not None else {}),
                    },
                    step = step,
                )
            except Exception:
                pass
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("train/loss", loss, step)
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/tokens_per_sec", tok_s, step)
                tb_writer.add_scalar("train/peak_gb", peak_gb, step)
                if grad_norm is not None:
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            except Exception:
                pass

    trainer.add_step_callback(_on_step)

    def _on_eval(step, eval_loss, perplexity):
        _send("progress", step = step, eval_loss = eval_loss)
        if wandb_run is not None:
            try:
                wandb_run.log({"eval/loss": eval_loss, "eval/perplexity": perplexity}, step = step)
            except Exception:
                pass
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("eval/loss", eval_loss, step)
                tb_writer.add_scalar("eval/perplexity", perplexity, step)
            except Exception:
                pass

    trainer.add_eval_callback(_on_eval)

    _opt_ref = [None]
    _orig_build_optimizer = getattr(trainer, "_build_optimizer", None)

    if callable(_orig_build_optimizer):

        def _capture_optimizer(total_steps):
            _opt_ref[0] = _orig_build_optimizer(total_steps)
            return _opt_ref[0]

        trainer._build_optimizer = _capture_optimizer

    # ── 11. Run training ──
    gc.collect()
    mx.synchronize()
    _save_model = trainer.save_model

    def _skip_internal_final_save(*args, **kwargs):
        raise ValueError("worker owns final save")

    trainer.save_model = _skip_internal_final_save
    try:
        trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    finally:
        trainer.save_model = _save_model

    # ── 12. Save and finalize ──
    def _finish_tracking() -> None:
        # Runs on every save/finalize exit so TB/W&B never leak on early return.
        if tb_writer is not None:
            try:
                tb_writer.close()
            except Exception:
                pass
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    def _stop_checkpoint_ok() -> bool:
        if _write_mlx_stop_checkpoint(trainer, _opt_ref[0], output_dir):
            return True
        _send(
            "error",
            error = (
                "Failed to save a resumable checkpoint after stop. "
                "Model files were saved, but this run cannot be resumed."
            ),
            # A user stop finalizes as 'stopped'; keep this failure's error status so history explains it.
            keep_error_status = True,
            # Older checkpoints are stale; resuming would roll back past this stop.
            resume_blocked = True,
        )
        return False

    try:
        if trainer.stop_requested:
            if not _stop_save[0]:
                # Cancel (save=False): skip saving.
                _send("complete", output_dir = None, status_message = "Training cancelled")
            else:
                _send("status", status_message = "Saving stopped model...")
                mx.synchronize()
                trainer.save_model(output_dir)
                # Stop-and-save promises a resumable checkpoint, not just model files.
                if not _stop_checkpoint_ok():
                    return
                _send("complete", output_dir = output_dir, status_message = "Training stopped")
        else:
            _send("status", status_message = "Saving model...")
            mx.synchronize()
            trainer.save_model(output_dir)
            # A save-stop can race the natural final save; it made the same promise.
            if trainer.stop_requested and _stop_save[0] and not _stop_checkpoint_ok():
                return
            _send("complete", output_dir = output_dir, status_message = "Training completed")
    finally:
        _finish_tracking()


def _is_current_process_apple_silicon() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def run_mlx_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
    transformers_activated: bool = False,
    config_prevalidated: bool = False,
) -> None:
    """MLX worker entrypoint shared by Unsloth subprocesses and the CLI adapter."""
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.hf_xet_fallback import child_should_disable_xet

    if child_should_disable_xet(config):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    if not config_prevalidated and not _validate_training_worker_config(config, event_queue):
        return
    model_load_target = _resolve_cached_model_load_name(config)

    if not transformers_activated:
        # Must precede detect_hardware(): its MLX stack check imports mlx_lm, hence transformers.
        _activate_transformers_version_or_warn(
            model_load_target,
            config.get("hf_token") or None,
        )

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if _hw.DEVICE != _hw.DeviceType.MLX:
        event_queue.put(
            {
                "type": "error",
                "error": "MLX training requires Apple Silicon with the MLX backend available.",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    if config.get("is_dataset_audio"):
        event_queue.put(
            {
                "type": "error",
                "error": "Audio dataset training is not yet supported on Apple Silicon.",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    try:
        try:
            _run_mlx_training(event_queue, stop_queue, config)
        finally:
            try:
                stop_queue.put({"type": _MLX_WORKER_COMPLETE})
            except (EOFError, OSError, ValueError):
                pass
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )


def _training_job_is_local(config) -> bool:
    """True when neither the model nor the dataset needs the Hub, so the probe is wasted.

    Fail closed: anything unresolvable counts as remote, since skipping a needed probe
    costs the retry backoff the probe exists to avoid.
    """
    try:
        from utils.paths import is_local_path
    except Exception:
        return False
    if config.get("hf_dataset"):
        return False
    model = config.get("model_name")
    try:
        if not (model and is_local_path(model)):
            return False
        # A local checkpoint can name a remote base, which activation resolves and training and
        # security code later fetches. Readable from disk, so no network needed to decide.
        base, needs_hub = _recorded_local_base(model)
        if needs_hub:
            return False
        return not base or is_local_path(base)
    except Exception:
        return False


def _recorded_local_base(model_name) -> "tuple[str | None, bool]":
    """``(base, needs_hub)`` for the base this checkpoint records on disk.

    Delegates to the resolver's own disk reads so the gate cannot drift from what
    activation later resolves. Fail closed on an unavailable reader.
    """
    try:
        from utils.transformers_version import recorded_local_base
        return recorded_local_base(model_name)
    except Exception:
        return None, True


def run_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for progress/status/error events to the parent.
        stop_queue: mp.Queue for stop commands from the parent.
        config: Training config dict with all parameters.
    """
    # Off on Linux (forked map() workers deadlock); on spawn platforms map() is in-process.
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "true" if sys.platform in ("win32", "darwin") else "false"
    )
    os.environ["PYTHONWARNINGS"] = "ignore"  # before imports

    # HTTP-fallback respawn: disable Xet before any huggingface_hub import (read at import time).
    from utils.hf_xet_fallback import child_should_disable_xet

    if child_should_disable_xet(config):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(
            "Xet transport disabled for this training worker (HF_HUB_DISABLE_XET=1).",
            file = sys.stderr,
            flush = True,
        )

    # Offline auto-detect: skip ~25s of HF retries per call when the hub is unreachable.
    # Skipped for a filesystem-only job: a local checkpoint with a local dataset never reaches
    # the Hub, and probing unconditionally would add seconds to every such startup.
    if "HF_HUB_OFFLINE" not in os.environ and not _training_job_is_local(config):
        _offline = False
        _network_offline = False
        try:
            from utils.utils import hf_dns_dead, hf_env_offline, hf_probe_disabled

            # Hub ignores TRANSFORMERS_OFFLINE, so translate it before probing.
            _offline = hf_env_offline()
            # hf_dns_dead follows HF_ENDPOINT and stands down behind a proxy, so a mirror still counts.
            if not _offline:
                _offline = _network_offline = hf_dns_dead()
            if not _offline and not hf_probe_disabled():
                # DNS answers even without egress (WAN down, captive portal). These flags last the whole
                # job, so only a connection failure counts: a momentary 502/503 must not block downloads.
                from utils.transformers_version import hf_endpoint_unreachable
                _offline = _network_offline = hf_endpoint_unreachable(
                    gateway_errors_offline = False,
                    proxy_timeouts_offline = False,
                )
        except Exception:
            _offline = _network_offline = False
        if _offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            # Only when the network itself is the reason. TRANSFORMERS_OFFLINE alone asks for cached
            # model files, not a cache-only dataset: an uncached hf_dataset would fail the whole job.
            if _network_offline:
                os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            # logger isn't configured yet; print to stderr instead.
            print(
                "Hugging Face endpoint unreachable; HF_HUB_OFFLINE=1 set for this worker.",
                file = sys.stderr,
                flush = True,
            )

    import warnings
    from loggers.config import LogConfig, keep_progress_bars_countable

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    # This worker READS the bars: the monitor thread further down polls tqdm._instances
    # to turn the Hub download and "Loading checkpoint shards" bars into the UI's status
    # line, and a disabled bar is never registered there. So it redirects their output
    # instead of disabling them, and does so before the inherited
    # HF_HUB_DISABLE_PROGRESS_BARS reaches huggingface_hub's import-time constant.
    keep_progress_bars_countable()

    LogConfig.setup_logging(
        service_name = "unsloth-studio-training-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    apply_gpu_ids(config.get("resolved_gpu_ids"), backend = config.get("device_backend"))

    if not _validate_training_worker_config(config, event_queue):
        return

    model_name = config["model_name"]
    model_load_target = _resolve_cached_model_load_name(config)

    # ── 0. MLX FAST-PATH (must run before any torch/transformers imports) ──
    # Apple Silicon uses MLXTrainer directly -- skip torch imports / installs.
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from .training import is_apple_silicon_training_platform, should_use_mlx_training_backend

    mlx_backend_requested = is_apple_silicon_training_platform()

    mlx_transformers_activated = False
    if mlx_backend_requested and _is_current_process_apple_silicon():
        # Must precede detect_hardware(): its MLX stack check imports mlx_lm, hence transformers.
        _activate_transformers_version_or_warn(
            model_load_target,
            config.get("hf_token") or None,
        )
        mlx_transformers_activated = True

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if mlx_backend_requested or should_use_mlx_training_backend(device = _hw.DEVICE):
        run_mlx_training_process(
            event_queue = event_queue,
            stop_queue = stop_queue,
            config = config,
            transformers_activated = mlx_transformers_activated,
            config_prevalidated = True,
        )
        return

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(
            model_load_target,
            config.get("hf_token") or None,
        )
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1a. Auto-enable trust_remote_code for NemotronH/Nano models ──
    # NemotronH needs trust_remote_code=True to work around config-parsing bugs; other 5.x
    # models are native (it bypasses the compiler, disabling fused CE). Not Llama-Nemotron.
    from utils.security.trusted_org import is_trusted_org_repo

    _NEMOTRON_TRUST_SUBSTRINGS = ("nemotron_h", "nemotron-h", "nemotron-3-nano")
    _lowered = model_name.lower()
    if (
        any(sub in _lowered for sub in _NEMOTRON_TRUST_SUBSTRINGS)
        and (_lowered.startswith("unsloth/") or _lowered.startswith("nvidia/"))
        # Confirm a genuine first-party Hub repo (not a spoofed "unsloth/" name); authenticated.
        and is_trusted_org_repo(model_name, hf_token = config.get("hf_token") or None)
        and not config.get("trust_remote_code", False)
    ):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for Nemotron model: %s",
            model_name,
        )

    security_error = _model_load_security_error(
        config,
        _resolve_cached_model_load_name(config),
        config.get("hf_token") or None,
    )
    if security_error:
        event_queue.put({"type": "error", **security_error, "ts": time.time()})
        return

    # ── 1b. Install fast-path kernel libraries for the chosen model.
    # 1) causal-conv1d runs eagerly for matching architectures: some SSM modeling files
    #    lazy_load it without calling is_causal_conv1d_available.
    # 2) FLA + tilelang: gated by the runtime hook on is_flash_linear_attention_available.
    # 3) mamba-ssm + flash-attn keep their substring / size gates.
    # 4) UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 falls back to the substring path.
    try:
        from utils.ssm_runtime import resolved_model_wants_causal_conv1d

        wants_causal_conv1d = resolved_model_wants_causal_conv1d(
            model_name,
            model_load_target,
            config.get("hf_token") or None,
        )
        _ensure_causal_conv1d_fast_path(
            event_queue,
            model_name,
            required = wants_causal_conv1d,
        )
        if os.getenv(_FAST_PATH_HOOKS_SKIP_ENV) == "1":
            _ensure_flash_linear_attention(event_queue, model_name)
            _ensure_tilelang_backend(event_queue, model_name)
        else:
            _install_fast_path_hooks(
                event_queue,
                model_name,
                install_causal_conv1d = wants_causal_conv1d,
            )
        _ensure_mamba_ssm(event_queue, model_name)
        _ensure_flash_attn_for_long_context(
            event_queue,
            int(config.get("max_seq_length", 2048)),
        )
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    f"Please choose another model to train, since "
                    f"a fast-path kernel library "
                    f"(causal-conv1d / flash-linear-attention / "
                    f"mamba-ssm / tilelang) failed to install "
                    f"with error: {exc}"
                ),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # No start-method override: Dataset.map() imports Pool from `multiprocess`, so forcing
    # stdlib multiprocessing onto "fork" never reached it; the guard now asks multiprocess.

    # ── 1c. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401
            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 1d. Stub torchao on Windows ROCm ──
    # See core/_torchao_stub.py (no RCCL on Windows ROCm); run before transformers/unsloth_zoo.
    from core._torchao_stub import install_torchao_windows_rocm_stub

    install_torchao_windows_rocm_stub()

    # ── 1e. Ensure torch.distributed helper attrs are present ──
    # Single-GPU never inits the process group, but transformers/trl import these anyway.
    _td_stubs = {
        "is_initialized": lambda: False,
        "is_available": lambda: False,
        "is_torchelastic_launched": lambda: False,
        "get_rank": lambda: 0,
        "get_world_size": lambda: 1,
        "barrier": lambda: None,
    }

    try:
        import torch.distributed as _td
        for _name, _stub in _td_stubs.items():
            if not hasattr(_td, _name):
                setattr(_td, _name, _stub)
    except Exception:
        _td_mock = types.ModuleType("torch.distributed")
        for _name, _stub in _td_stubs.items():
            setattr(_td_mock, _name, _stub)
        sys.modules["torch.distributed"] = _td_mock
        try:
            import torch as _torch
            _torch.distributed = _td_mock
        except Exception:
            pass

    # ── 1f. Windows ROCm runtime patches ──
    # torch._grouped_mm has a null HIP kernel on gfx1200 (ROCm <= 7.12 Windows), causing
    # 0xC0000005 during training. JitDecomp (not torch.compile) dispatches _grouped_mm to the
    # null crash and TORCHDYNAMO_DISABLE doesn't cover it, so also override the CUDA dispatch
    # key with a Python fallback. Fixed in torch==2.11.0+rocm7.13.0, so gate on HIP < 7.13.
    # Schema: _grouped_mm(self, mat2, offs=None, bias=None, out_dtype=None); offs = group splits.
    global _WINDOWS_ROCM_GROUPED_MM_LIB
    if sys.platform == "win32":
        _torch_for_rocm = sys.modules.get("torch")
        # Broad check (torch.version.hip OR "rocm" in __version__): AMD SDK / Radeon wheels don't
        # always set torch.version.hip, and the BNB pin, dynamo-disable and fallback would skip.
        _build_version_for_rocm = (
            getattr(_torch_for_rocm, "__version__", "").lower()
            if _torch_for_rocm is not None
            else ""
        )
        _is_win_rocm_torch = bool(
            _torch_for_rocm is not None
            and (
                getattr(getattr(_torch_for_rocm, "version", None), "hip", None)
                or "rocm" in _build_version_for_rocm
            )
        )
        if _is_win_rocm_torch:
            # Belt-and-suspenders; the JitDecomp patch is the real fix, but this covers other paths.
            if "TORCHDYNAMO_DISABLE" not in os.environ:
                os.environ["TORCHDYNAMO_DISABLE"] = "1"
                logger.info("Windows ROCm: torch.compile (dynamo) disabled")

            # bitsandbytes' import-time get_rocm_gpu_arch() probe runs `hipinfo.exe` from PATH; the AMD
            # torch wheel ships it in the venv Scripts dir, which is on PATH only for activated venvs.
            # Prepend it so the probe succeeds instead of logging a scary (harmless) error on every
            # import. Normally inherited from main.py, but workers can also be spawned standalone.
            _scripts_dir = os.path.dirname(sys.executable)
            if os.path.isfile(os.path.join(_scripts_dir, "hipInfo.exe")):
                import shutil as _shutil
                if not _shutil.which("hipinfo.exe"):
                    os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")

            # BNB picks a rocm DLL from torch.version.hip, but AMD's Windows BNB wheel may ship a DLL
            # whose suffix doesn't match, so detect the actual DLL name and override. Installer-seeded
            # values are redetectable defaults; caller overrides stay authoritative.
            if (
                "BNB_ROCM_VERSION" not in os.environ
                or os.environ.get("UNSLOTH_BNB_ROCM_VERSION_SOURCE") == "sitecustomize"
            ):
                _bnb_rocm_ver = None
                _found_rocm_bnb = False
                try:
                    import glob as _glob
                    import importlib.util as _ilu
                    import re as _re

                    _bnb_spec = _ilu.find_spec("bitsandbytes")
                    if _bnb_spec and _bnb_spec.submodule_search_locations:
                        _all_vers: list[str] = []
                        for _pkg_dir in _bnb_spec.submodule_search_locations:
                            for _dll in _glob.glob(
                                os.path.join(_pkg_dir, "libbitsandbytes_rocm*.dll")
                            ):
                                _found_rocm_bnb = True
                                _m = _re.search(
                                    r"libbitsandbytes_rocm(\d+)\.dll",
                                    os.path.basename(_dll),
                                )
                                if _m:
                                    _all_vers.append(_m.group(1))
                        # Highest numeric suffix wins (glob order isn't sorted).
                        if _all_vers:
                            _bnb_rocm_ver = max(_all_vers, key = lambda v: int(v))
                except Exception:
                    pass
                # Only when a ROCm bnb DLL actually exists (mirrors main.py): without one the seeded value
                # and its marker stay untouched, so later import fixes can still redetect or opt out.
                # A DLL with an unparsable name falls back to the seeded value or "72".
                if _found_rocm_bnb:
                    _bnb_rocm_ver = _bnb_rocm_ver or os.environ.get("BNB_ROCM_VERSION") or "72"
                    os.environ["BNB_ROCM_VERSION"] = _bnb_rocm_ver
                    os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] = "detected"
                    logger.info(
                        "Windows ROCm: set BNB_ROCM_VERSION=%s "
                        "(detected from installed BNB wheel; "
                        "overrides torch.version.hip auto-detection)",
                        _bnb_rocm_ver,
                    )

            # Setting BNB_ROCM_VERSION makes bitsandbytes log a benign override notice on import;
            # drop only that record so real errors and mismatch warnings still show.
            if os.environ.get("BNB_ROCM_VERSION"):
                import logging as _logging
                _logging.getLogger("bitsandbytes.cextension").addFilter(
                    lambda _r: "environment variable detected" not in _r.getMessage()
                )

            # Parse HIP version for the kernel-fix gate below, falling back to the rocm version in
            # torch.__version__ when version.hip is unset (AMD SDK / Radeon wheels).
            def _hip_ver_at_least(major: int, minor: int) -> bool:
                _hip_str = getattr(getattr(_torch_for_rocm, "version", None), "hip", None)
                if not _hip_str:
                    # Try the standard "+rocmX.Y.Z" embedded version first.
                    _ver_match = re.search(r"rocm(\d+)\.(\d+)", _build_version_for_rocm)
                    if _ver_match:
                        return (
                            int(_ver_match.group(1)),
                            int(_ver_match.group(2)),
                        ) >= (major, minor)
                    # "+rocmsdk<date>" wheels postdate the gfx120X null-kernel fix (ROCm 7.13); treat as >= 7.13.
                    if "rocmsdk" in _build_version_for_rocm:
                        logger.debug(
                            "Windows ROCm: AMD SDK wheel detected (%r); "
                            "assuming HIP >= %d.%d (rocmsdk wheels post-date "
                            "the gfx120X null-kernel fix)",
                            _build_version_for_rocm,
                            major,
                            minor,
                        )
                        return True
                    return False
                try:
                    _parts = [int(x) for x in str(_hip_str).split(".")[:2]]
                    if len(_parts) < 2:
                        logger.warning(
                            "Windows ROCm: torch.version.hip %r has fewer than "
                            "two components; cannot compare against %d.%d",
                            _hip_str,
                            major,
                            minor,
                        )
                        return False
                    return (_parts[0], _parts[1]) >= (major, minor)
                except ValueError:
                    logger.warning(
                        "Windows ROCm: could not parse torch.version.hip %r as "
                        "a version number; assuming HIP < %d.%d",
                        _hip_str,
                        major,
                        minor,
                    )
                    return False

            # Only on affected versions (ROCm <= 7.12); 7.13+ uses the real GPU kernel.
            if not _hip_ver_at_least(7, 13):
                try:
                    _WINDOWS_ROCM_GROUPED_MM_LIB = _install_grouped_mm_cpu_fallback(
                        _torch_for_rocm, logger, "Windows ROCm"
                    )
                except Exception as _patch_exc:
                    logger.warning(
                        "Windows ROCm: could not patch _grouped_mm — "
                        "training may crash with 0xC0000005: %s",
                        _patch_exc,
                    )
            else:
                logger.info(
                    "Windows ROCm: HIP >= 7.13 — _grouped_mm kernel is functional, "
                    "skipping Python fallback (AMD fixed gfx1200 null kernel in ROCm 7.13)"
                )

    # ── 1f-linux. Linux ROCm RDNA4 _grouped_mm null kernel ──
    # The win32 guard above misses Linux: RDNA4 (gfx1200/gfx1201) hits the same null HIP
    # _grouped_mm kernel at ROCm <= 7.12 (fixed 7.13, ROCm/TheRock #5284). Gate on arch + HIP.
    if sys.platform.startswith("linux") and _hw.IS_ROCM:
        try:
            _torch_lin = sys.modules.get("torch")
            if _torch_lin is not None and _torch_lin.cuda.is_available():
                # Prefer torch.version.hip, else rocmX.Y from torch.__version__ (AMD SDK / Radeon wheels
                # leave it unset). Unknown version on gfx120X -> assume affected unless a post-fix rocmsdk.
                _hip_str = str(getattr(getattr(_torch_lin, "version", None), "hip", "") or "")
                _ver = getattr(_torch_lin, "__version__", "").lower()
                _m = re.match(r"(\d+)\.(\d+)", _hip_str) or re.search(r"rocm(\d+)\.(\d+)", _ver)
                if _m:
                    _hip_lt_713 = (int(_m.group(1)), int(_m.group(2))) < (7, 13)
                else:
                    _hip_lt_713 = "rocmsdk" not in _ver
                # Scan every visible GPU (device_map="balanced" can place layers on a later RDNA4 card).
                # Match gfx120X by arch, or by RX 9000 / R9700 name when the wheel omits gcnArchName.
                _rdna4 = False
                for _i in range(_torch_lin.cuda.device_count()):
                    _props = _torch_lin.cuda.get_device_properties(_i)
                    _lin_arch, _ = _rocm_classify_unified_memory(_props)
                    _lin_name = (getattr(_props, "name", "") or "").lower()
                    if _lin_arch.lower() in ("gfx1200", "gfx1201") or (
                        not _lin_arch and re.search(r"rx\s*90[0-9]0|r9700", _lin_name)
                    ):
                        _rdna4 = True
                        break
                if _rdna4 and _hip_lt_713:
                    _WINDOWS_ROCM_GROUPED_MM_LIB = _install_grouped_mm_cpu_fallback(
                        _torch_lin, logger, "Linux ROCm gfx120X"
                    )
        except Exception as _gm_lin_exc:
            logger.warning("Linux ROCm gfx120X: could not patch _grouped_mm: %s", _gm_lin_exc)

    # ── 1g. ROCm OOM guard ──
    # On ROCm, exhausting VRAM can hang the HIP driver instead of raising.
    # set_per_process_memory_fraction caps the allocator so PyTorch raises OutOfMemoryError
    # first. Unified hosts share GPU+system RAM and need OS headroom, so the cap depends on
    # the classification and the pool size (see _rocm_memory_fraction and
    # _rocm_classify_unified_memory). Skipped if no torch.
    if _hw.IS_ROCM:
        try:
            import torch as _torch_mem
            if _torch_mem.cuda.is_available():
                # Classify unified vs discrete (see _rocm_classify_unified_memory's docstring).
                _props = _torch_mem.cuda.get_device_properties(0)
                _dev_name = _props.name
                _gcn_arch, _is_unified = _rocm_classify_unified_memory(_props)
                if _is_unified and not _gcn_arch:
                    logger.debug(
                        "ROCm OOM guard: gcnArchName absent -- inferred "
                        "unified memory from device name %r; applying unified cap",
                        _dev_name,
                    )
                # Unified hosts on native Windows: mem_get_info's total is the WDDM budget the driver
                # grants HIP (BIOS carve + ~half of remaining RAM). The OS share is already outside it, so
                # any sub-1.0 starve-protection double-taxes (48.49 GiB budget -> 38.79 allowed) and
                # blocks loads that fit in free memory. Current AMD Windows wheels only enforce sub-1.0
                # fractions (gfx1151: 0.5 caps, 1.0 overcommits via WDDM), so 1.0 behaves like torch's
                # uncapped default. On Linux the total spans nearly all RAM, so keep a bounded headroom
                # (see _rocm_memory_fraction).
                # props.total_memory is the pool the reserve comes out of, and from torch
                # 2.10 also what the allocator scales. Through 2.9 it scales hipMemGetInfo's
                # total, a different number on a unified APU, so hand that to the helper on
                # those wheels and the reserve is the same bytes either way.
                _total_bytes = int(getattr(_props, "total_memory", 0) or 0)
                _driver_total = 0
                if not _allocator_divides_by_props_total(getattr(_torch_mem, "__version__", "")):
                    try:
                        _driver_total = int(_torch_mem.cuda.mem_get_info(0)[1])
                    except Exception:
                        _driver_total = 0
                _env_raw = os.environ.get(_MEM_FRACTION_ENV)
                _env_fraction = _parse_mem_fraction_env(_env_raw)
                if _env_raw and _env_fraction is None:
                    logger.warning(
                        "ROCm OOM guard: ignoring %s=%r (needs a float in (0.0, 1.0]); "
                        "using the computed cap instead",
                        _MEM_FRACTION_ENV,
                        _env_raw,
                    )
                _mem_fraction = _rocm_memory_fraction(
                    _total_bytes, _is_unified, sys.platform, _env_raw, _driver_total or None
                )
                # A wheel that reports no total still gets a cap; say so rather than
                # printing "0.0 of 0.0 GiB allowed" on the one host whose props are suspect.
                _allowed = (
                    f"{_total_bytes * _mem_fraction / 1024**3:.1f} of "
                    f"{_total_bytes / 1024**3:.1f} GiB allowed"
                    if _total_bytes > 0
                    else "device total unreported by this wheel"
                )
                _torch_mem.cuda.set_per_process_memory_fraction(_mem_fraction)
                logger.info(
                    "ROCm OOM guard: set_per_process_memory_fraction(%.4f) — "
                    "%s memory host (%s, %s), %s, %s",
                    _mem_fraction,
                    "unified" if _is_unified else "discrete",
                    _dev_name,
                    _gcn_arch or "unknown arch",
                    _allowed,
                    f"from {_MEM_FRACTION_ENV}"
                    if _env_fraction is not None
                    else f"computed; override with {_MEM_FRACTION_ENV}",
                )
                # When the totals differ the cap was solved against the driver's, so the
                # budget printed above is not the one enforced. Give both, and the headroom
                # that results, which the floor can leave under the intended reserve.
                if (
                    _is_unified
                    and sys.platform != "win32"
                    and _env_fraction is None
                    and _total_bytes > 0
                    and _driver_total > 0
                    and abs(_driver_total - _total_bytes) > _total_bytes // 100
                ):
                    logger.info(
                        "ROCm OOM guard: props.total_memory is %.1f GiB but this torch caps "
                        "against the driver's %.1f GiB, so the fraction is solved for that "
                        "total and %.1f GiB stays free against the intended %.1f GiB. Adjust "
                        "with %s.",
                        _total_bytes / 1024**3,
                        _driver_total / 1024**3,
                        (_total_bytes - _mem_fraction * _driver_total) / 1024**3,
                        _UNIFIED_OS_RESERVE_BYTES / 1024**3,
                        _MEM_FRACTION_ENV,
                    )
                # Unified Windows APUs: the WDDM budget is user-raisable, but nothing on the box says so
                # -- users see "48 GB VRAM" on a 96 GB machine. Say where the limit comes from.
                if _is_unified and sys.platform == "win32":
                    try:
                        import psutil as _psutil

                        _phys = _psutil.virtual_memory().total
                        _granted = _torch_mem.cuda.mem_get_info(0)[1]
                        if _granted < 0.75 * _phys:
                            logger.info(
                                "Windows grants the GPU %.1f GiB of %.1f GiB "
                                "system RAM (driver/WDDM budget). To raise it: "
                                "increase the BIOS UMA frame buffer size, or "
                                "AMD Software > Performance > Tuning > "
                                "Variable Graphics Memory.",
                                _granted / 1024**3,
                                _phys / 1024**3,
                            )
                    except Exception:
                        pass
        except Exception as _oom_guard_err:
            logger.debug("Could not set GPU memory fraction: %s", _oom_guard_err)

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
            default_run_dir_name,
        )

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 2b. EMBEDDING MODEL FAST-PATH ──
    # Embedding models use a different pipeline (FastSentenceTransformer +
    # SentenceTransformerTrainer + MultipleNegativesRankingLoss), so branch early.
    if config.get("is_embedding", False):
        try:
            _run_embedding_training(event_queue, stop_queue, config)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        return

    # ── 3. Create a fresh trainer instance ──
    trainer = UnslothTrainer()

    trainer.add_progress_callback(_create_trainer_progress_callback(event_queue))

    def _apply_stop(save: bool) -> None:
        trainer.should_stop = True
        trainer.save_on_stop = save
        logger.info("Stop signal received (save=%s)", save)

    _start_worker_stop_poller(stop_queue, _apply_stop)

    # ── 4. Execute the training pipeline ──
    # Order: detect -> dataset -> model -> prepare -> train, so both never hold VRAM at once.
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        model_load_name = _resolve_cached_model_load_name(config)
        model_local_only = _model_local_files_only(config)
        model_revision = None if model_local_only else config.get("model_revision")
        dataset_local_only = _dataset_local_files_only(config)
        eval_steps = config.get("eval_steps", 0.00)

        hf_dataset = config.get("hf_dataset", "")
        training_type = config.get("training_type", "LoRA/QLoRA")
        is_cpt_for_dataset = training_type == "Continued Pretraining"

        # Filled in below, after the model probe; the closure runs after both.
        max_train_rows = None
        max_train_rows_seed = config.get("random_seed", 3407)

        def _load_training_dataset():
            result = trainer.load_and_format_dataset(
                dataset_source = hf_dataset if hf_dataset and hf_dataset.strip() else None,
                format_type = config.get("format_type", ""),
                local_datasets = config.get("local_datasets") or None,
                local_eval_datasets = config.get("local_eval_datasets") or None,
                custom_format_mapping = config.get("custom_format_mapping"),
                subset = config.get("subset"),
                train_split = config.get("train_split", "train"),
                eval_split = config.get("eval_split"),
                dataset_streaming = config.get("dataset_streaming", False),
                eval_steps = eval_steps,
                dataset_slice_start = config.get("dataset_slice_start"),
                dataset_slice_end = config.get("dataset_slice_end"),
                is_cpt = is_cpt_for_dataset,
                s3_config = config.get("s3_config"),
                dataset_local_files_only = dataset_local_only,
                dataset_local_path = config.get("dataset_snapshot_path"),
                dataset_revision = config.get("dataset_revision"),
                require_exact_resume_resources = bool(
                    config.get("require_exact_resume_resources")
                    or config.get("require_exact_dataset_resource")
                ),
                max_train_rows = max_train_rows,
                max_train_rows_seed = max_train_rows_seed,
            )
            if isinstance(result, tuple):
                loaded_dataset, loaded_eval_dataset = result
            else:
                loaded_dataset = result
                loaded_eval_dataset = None
            if eval_steps is not None and float(eval_steps) <= 0:
                loaded_eval_dataset = None
            snapshot = getattr(trainer, "dataset_snapshot_path", None)
            if snapshot:
                config["dataset_snapshot_path"] = snapshot
            return loaded_dataset, loaded_eval_dataset

        # ── 4a. Lightweight detection + tokenizer (no VRAM) ──
        _send_status(event_queue, "Detecting model type...")
        try:
            _pre_detect_training_model(
                trainer,
                config,
                model_name,
                hf_token,
                model_load_name,
                model_local_only,
                model_revision,
            )
        except Exception as error:
            if not model_local_only:
                raise
            fallback_error = _model_cache_fallback_error(config, error)
            if fallback_error is not None:
                raise fallback_error from error
            if not _cache_artifact_fallback_allowed(config, error, "model"):
                raise
            _send_status(
                event_queue,
                f"Cached files for {model_name} are incomplete; retrying from Hugging Face...",
            )
            model_load_name = _drop_model_pin_for_fallback(config, hf_token)
            # Scan the Hub target we fall back to, not the cached pin already scanned above.
            security_error = _model_load_security_error(config, model_load_name, hf_token)
            if security_error:
                event_queue.put({"type": "error", **security_error, "ts": time.time()})
                return
            model_local_only = False
            model_revision = config.get("model_revision")
            _pre_detect_training_model(
                trainer,
                config,
                model_name,
                hf_token,
                model_load_name,
                model_local_only,
                model_revision,
            )
        if trainer.should_stop:
            event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            return

        # 4a has probed the model, so the packing opt-out can read the real branch
        # instead of guessing from the client's dataset flags. Streaming and explicit
        # train-split ranges opt out inside load_and_format_dataset.
        # Audio codecs are chosen before the raw-text bypass and use plain Trainers
        # with no packing argument, so they hold either way; the vision and audio-VLM
        # branches are gated on `not raw_text_mode`, so a raw or CPT run takes the
        # text path, which honours packing.
        raw_text_mode = is_cpt_for_dataset or config.get("format_type") == "raw"
        branch_never_packs = bool(getattr(trainer, "_audio_type", None)) or (
            bool(getattr(trainer, "is_vlm", False) or getattr(trainer, "is_audio_vlm", False))
            and not raw_text_mode
        )
        # Every replica draws its own batch per step, so the subset has to cover all of
        # them; sized here rather than in the config because it is a property of this
        # machine's launch. Model probing is done, so torch and the GPU mask are settled.
        max_train_rows = max_train_rows_for_config(
            config,
            branch_never_packs = branch_never_packs,
            world_size = _data_parallel_world_size(),
        )
        # A resume trains on the rows its first start chose, read back from the marker
        # beside the checkpoints. No marker means the checkpoint predates the bound and
        # trained on the whole dataset; since the trainer fast-forwards by batch count
        # over the current dataloader, bounding it now would continue on unrelated rows.
        resumed_rows, max_train_rows_seed = row_bound_for_resume(
            config.get("resume_from_checkpoint"), max_train_rows, max_train_rows_seed
        )
        if resumed_rows != max_train_rows:
            logger.info(
                "Resuming with the row bound recorded at the original start "
                f"({resumed_rows} rows) instead of {max_train_rows}\n"
            )
            if resumed_rows and max_train_rows and resumed_rows < max_train_rows:
                # Sized for fewer replicas than this machine has: the recorded subset
                # is what the run trained on and re-deriving it would continue on
                # unrelated rows, so it stays, but say that the extra ranks may reach
                # the end of it and start over.
                logger.info(
                    "That subset was sized for a smaller data-parallel world than this "
                    "one, so the added replicas may re-read rows; start a new run "
                    "instead of resuming to size it for this machine\n"
                )
        max_train_rows = resumed_rows

        # ── 4b. Load and format dataset (LLM helper may use VRAM briefly) ──
        _send_status(event_queue, "Loading and formatting dataset...")
        dataset, eval_dataset = _load_training_dataset()

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error or "Failed to load dataset",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── Start tqdm monitor early to capture download + tokenization bars ──
        import threading as _th

        _tqdm_stop = _th.Event()

        def _monitor_tqdm():
            from tqdm.auto import tqdm as _tqdm_cls
            while not _tqdm_stop.is_set():
                for bar in list(getattr(_tqdm_cls, "_instances", set())):
                    try:
                        n, total = bar.n or 0, bar.total or 0
                        desc = getattr(bar, "desc", "") or ""
                        if total > 0 and n > 0 and desc:
                            pct = min(int(n * 100 / total), 100)
                            _send_status(event_queue, f"{desc.strip()} {pct}% ({n:,}/{total:,})")
                    except (AttributeError, ReferenceError):
                        pass
                _tqdm_stop.wait(3)

        _tqdm_thread = _th.Thread(target = _monitor_tqdm, daemon = True)
        _tqdm_thread.start()

        training_type = config.get("training_type", "LoRA/QLoRA")
        is_cpt = training_type == "Continued Pretraining"
        use_lora = training_type in ("LoRA/QLoRA", "Continued Pretraining")
        cpt_trains_embeddings = False

        # ── 4c. Load training model (uses VRAM — dataset already formatted) ──
        # Watchdog lets the parent recover a stalled Xet download via respawn.
        _send_status(event_queue, "Loading model...")
        from utils.hf_xet_fallback import start_watchdog

        event_queue.put({"type": "model_load_started", "ts": time.time()})
        _load_watchdog_stop = start_watchdog(
            repo_ids = [model_name],
            on_stall = lambda msg: event_queue.put(
                {"type": "stall", "message": msg, "ts": time.time()}
            ),
            xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1",
        )
        # Latest-sidecar models load 16-bit: bnb 4-bit feeds quantized experts into unvalidated paths.
        try:
            _train_load_in_4bit = _effective_training_load_in_4bit(
                config,
                model_load_name,
                hf_token,
            )
            if config["load_in_4bit"] and not _train_load_in_4bit:
                logger.info(
                    "Latest-transformers sidecar active for %s - forcing a 16-bit "
                    "training load (4-bit is disabled for brand-new architectures)",
                    model_load_name,
                )
            success = trainer.load_model(
                model_name = model_name,
                max_seq_length = config["max_seq_length"],
                load_in_4bit = _train_load_in_4bit,
                full_finetuning = not use_lora,
                hf_token = hf_token,
                is_dataset_image = config.get("is_dataset_image", False),
                is_dataset_audio = config.get("is_dataset_audio", False),
                trust_remote_code = config.get("trust_remote_code", False),
                gpu_ids = config.get("resolved_gpu_ids"),
                model_load_name = model_load_name,
                local_files_only = model_local_only,
                actual_model_repo_id = config.get("actual_model_repo_id"),
                model_revision = model_revision,
            )
            fallback_error = (
                _model_cache_fallback_error(config, trainer.model_load_error)
                if not success and model_local_only and not trainer.should_stop
                else None
            )
            if fallback_error is not None:
                trainer.training_progress.error = str(fallback_error)
            if (
                not success
                and model_local_only
                and not trainer.should_stop
                and fallback_error is None
                and _cache_artifact_fallback_allowed(config, trainer.model_load_error, "model")
            ):
                _send_status(
                    event_queue,
                    f"Cached files for {model_name} are incomplete; retrying from Hugging Face...",
                )
                model_load_name = _drop_model_pin_for_fallback(config, hf_token)
                # Scan the Hub target we fall back to, not the cached pin already scanned above.
                security_error = _model_load_security_error(config, model_load_name, hf_token)
                if security_error:
                    event_queue.put({"type": "error", **security_error, "ts": time.time()})
                    return
                model_local_only = False
                model_revision = config.get("model_revision")
                trainer.model = None
                trainer.tokenizer = None
                dataset = None
                eval_dataset = None
                gc.collect()
                from utils.hardware import clear_gpu_cache

                clear_gpu_cache()
                _send_status(
                    event_queue,
                    "Reloading and formatting the dataset with the Hub tokenizer...",
                )
                dataset, eval_dataset = _reload_dataset_with_remote_model_tokenizer(
                    trainer,
                    config,
                    model_name,
                    hf_token,
                    _load_training_dataset,
                    model_revision,
                )
                if dataset is None or trainer.should_stop:
                    success = False
                else:
                    success = trainer.load_model(
                        model_name = model_name,
                        max_seq_length = config["max_seq_length"],
                        load_in_4bit = _train_load_in_4bit,
                        full_finetuning = not use_lora,
                        hf_token = hf_token,
                        is_dataset_image = config.get("is_dataset_image", False),
                        is_dataset_audio = config.get("is_dataset_audio", False),
                        trust_remote_code = config.get("trust_remote_code", False),
                        gpu_ids = config.get("resolved_gpu_ids"),
                        model_load_name = model_load_name,
                        local_files_only = model_local_only,
                        actual_model_repo_id = config.get("actual_model_repo_id"),
                        model_revision = model_revision,
                    )
        finally:
            _load_watchdog_stop.set()
            event_queue.put({"type": "model_load_completed", "ts": time.time()})
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                error_msg = trainer.training_progress.error or "Failed to load model"
                event_queue.put(
                    {
                        "type": "error",
                        "error": error_msg,
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        _emit_resource_provenance(
            event_queue,
            config,
            trainer.model,
            model_load_target = model_load_name,
            model_load_in_4bit = _train_load_in_4bit,
            dataset_loaded_from_exact_snapshot = bool(
                getattr(trainer, "dataset_loaded_from_exact_snapshot", False)
            ),
        )

        if eval_dataset is not None:
            event_queue.put(
                {
                    "type": "eval_configured",
                    "ts": time.time(),
                }
            )

        # ── 4d. Prepare model (LoRA, full finetuning, or CPT) ──
        if is_cpt:
            _send_status(event_queue, "Configuring LoRA for continued pretraining...")
            # Both go to modules_to_save: trained full-precision at
            # embedding_learning_rate, since LoRA on either never trains.
            # By leaf: PEFT resolves model.embed_tokens to the same module.
            _embedding_modules = ("embed_tokens", "lm_head")
            _user_modules = config.get("target_modules") or []
            _leaf = lambda m: str(m).rsplit(".", 1)[-1]  # noqa: E731
            _wants = [m for m in _user_modules if _leaf(m) in _embedding_modules]
            # Either module in modules_to_save fills the embedding_learning_rate group.
            cpt_trains_embeddings = bool(_wants)
            cpt_target_modules = [m for m in _user_modules if _leaf(m) not in _embedding_modules]
            if not cpt_target_modules:
                cpt_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            success = trainer.prepare_model_for_training(
                use_lora = True,
                target_modules = cpt_target_modules,
                modules_to_save = _wants or None,
                lora_r = config.get("lora_r", 128),
                lora_alpha = config.get("lora_alpha", 32),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
                use_dora = config.get("use_dora", False),
            )
        elif use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora = True,
                finetune_vision_layers = config.get("finetune_vision_layers", True),
                finetune_language_layers = config.get("finetune_language_layers", True),
                finetune_attention_modules = config.get("finetune_attention_modules", True),
                finetune_mlp_modules = config.get("finetune_mlp_modules", True),
                target_modules = config.get("target_modules"),
                lora_r = config.get("lora_r", 16),
                lora_alpha = config.get("lora_alpha", 16),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
                use_dora = config.get("use_dora", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora = False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error or "Failed to prepare model",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        lr_default = "5e-5" if is_cpt else "2e-4"
        try:
            lr_value = float(config.get("learning_rate", lr_default))
        except ValueError:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Invalid learning rate: {config.get('learning_rate')}",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Pydantic already validated embedding_learning_rate (Optional[float], gt=0, lt=1.0).
        embedding_lr_value = config.get("embedding_learning_rate")
        if is_cpt:
            if cpt_trains_embeddings:
                if embedding_lr_value is None:
                    # Default embedding_learning_rate = lr/10 (Unsloth CPT notebook).
                    embedding_lr_value = lr_value / 10.0
                    logger.info(
                        f"CPT: using default embedding_learning_rate={embedding_lr_value:.1e} "
                        f"(lr/10). Set explicitly to override.\n"
                    )
            elif embedding_lr_value is not None:
                logger.warning(
                    "CPT: embedding_learning_rate was provided but neither embed_tokens "
                    "nor lm_head is being trained; ignoring the override.\n"
                )
                embedding_lr_value = None

        resume_from_checkpoint = config.get("resume_from_checkpoint")
        output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
            resume_from_checkpoint
        )
        if not output_dir:
            output_dir = build_default_output_dir_name(
                model_name,
                config.get("project_name"),
            )
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))
        _emit_output_dir(event_queue, output_dir)
        # Pin the subset before any checkpoint lands here, so a resume reads it back
        # rather than deriving it from a config the user may have edited in between.
        if not record_row_bound(output_dir, max_train_rows, max_train_rows_seed) and max_train_rows:
            # Not fatal, and nothing to fall back to: the dataset is already bounded.
            # Say it, so a later resume reading this run as unbounded is explainable.
            logger.warning(
                f"Could not record the max_steps row bound in {output_dir}: "
                "resuming this run later will read it as unbounded\n"
            )

        tensorboard_dir = config.get("tensorboard_dir")
        if config.get("enable_tensorboard", False):
            tensorboard_dir = str(resolve_tensorboard_dir(tensorboard_dir))
            ensure_dir(Path(tensorboard_dir))

        # Start training directly — no inner thread, we ARE the subprocess.
        dataset_display = config.get("hf_dataset", "") or config.get("uploaded_file", "") or ""
        _send_status(
            event_queue,
            f'Training "{model_name}"'
            + (f"\nDataset = {dataset_display}" if dataset_display else ""),
        )
        max_steps = config.get("max_steps", 0)
        save_steps = config.get("save_steps", 0)

        trainer._train_worker(
            dataset,
            output_dir = output_dir,
            num_epochs = config.get("num_epochs", 3),
            learning_rate = lr_value,
            embedding_learning_rate = embedding_lr_value,
            batch_size = config.get("batch_size", 2),
            gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4),
            warmup_steps = config.get("warmup_steps"),
            warmup_ratio = config.get("warmup_ratio"),
            max_steps = max_steps if max_steps and max_steps > 0 else 0,
            save_steps = save_steps if save_steps and save_steps > 0 else 0,
            weight_decay = config.get("weight_decay", 0.001),
            random_seed = config.get("random_seed", 3407),
            packing = config.get("packing", False),
            train_on_completions = False if is_cpt else config.get("train_on_completions", False),
            enable_wandb = config.get("enable_wandb", False),
            wandb_project = config.get("wandb_project", "unsloth-training"),
            wandb_token = config.get("wandb_token"),
            enable_tensorboard = config.get("enable_tensorboard", False),
            tensorboard_dir = tensorboard_dir,
            eval_dataset = eval_dataset,
            eval_steps = eval_steps,
            max_seq_length = config.get("max_seq_length", 2048),
            vision_image_size = config.get("vision_image_size"),
            optim = config.get("optim", "adamw_8bit"),
            lr_scheduler_type = config.get("lr_scheduler_type", "linear"),
            is_cpt = is_cpt,
            resume_from_checkpoint = resume_from_checkpoint,
        )

        _tqdm_stop.set()

        progress = trainer.get_training_progress()
        if progress.error:
            event_queue.put(
                {
                    "type": "error",
                    "error": progress.error,
                    "stack": "",
                    "ts": time.time(),
                }
            )
        else:
            saved_output_dir = (
                None if trainer.should_stop and not trainer.save_on_stop else output_dir
            )
            event_queue.put(
                {
                    "type": "complete",
                    "output_dir": saved_output_dir,
                    "status_message": progress.status_message or "Training completed",
                    "ts": time.time(),
                }
            )

    except Exception as exc:
        _exc_str = str(exc).lower()
        _is_oom = (
            "out of memory" in _exc_str
            or "hip out of memory" in _exc_str
            or "cuda out of memory" in _exc_str
            or type(exc).__name__ == "OutOfMemoryError"
        )
        if _is_oom:
            _oom_msg = (
                "GPU ran out of VRAM during training.\n"
                "To fix: reduce max_seq_length (e.g. 2048–4096), enable "
                "gradient_checkpointing=True, lower per_device_train_batch_size, "
                "or use a smaller model / higher quantization."
            )
            logger.error("Training stopped: GPU OOM — %s", exc)
            event_queue.put(
                {
                    "type": "error",
                    "error": _oom_msg,
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )


def _send_status(event_queue: Any, message: str) -> None:
    """Send a status update to the parent process."""
    event_queue.put(
        {
            "type": "status",
            "message": message,
            "ts": time.time(),
        }
    )


def _emit_output_dir(event_queue: Any, output_dir: str) -> None:
    try:
        event_queue.put({"type": "output_dir", "output_dir": output_dir, "ts": time.time()})
    except Exception:
        pass


def _emit_resource_provenance(
    event_queue: Any,
    config: dict,
    model: Any,
    *,
    model_load_target: str,
    model_load_in_4bit: bool,
    dataset_loaded_from_exact_snapshot: bool,
) -> None:
    from core.training.provenance import (
        build_worker_provenance_event,
        incomplete_worker_provenance_event,
    )

    try:
        event = build_worker_provenance_event(
            config,
            model,
            model_load_target = model_load_target,
            model_load_in_4bit = model_load_in_4bit,
            dataset_loaded_from_exact_snapshot = dataset_loaded_from_exact_snapshot,
        )
    except Exception:
        logger.warning("Could not attest training resource provenance", exc_info = True)
        event = incomplete_worker_provenance_event("provenance_attestation_failed")
    event["ts"] = time.time()
    event_queue.put(event)


def _mlx_has_checkpoint_at_step(output_dir, step: int) -> bool:
    if step <= 0:
        return False
    from core.training.resume import is_resume_checkpoint_valid
    return is_resume_checkpoint_valid(
        Path(output_dir) / f"checkpoint-{step}", expected_step = step, backend = "mlx"
    )


def _write_mlx_stop_checkpoint(trainer, optimizer, output_dir) -> bool:
    """Write a full resume checkpoint for a stopped MLX run.

    Returns True when a checkpoint for the current training step exists.
    """
    step = int(getattr(trainer, "_global_step", 0) or 0)
    # A periodic save or a resumed run may already cover the current step.
    if _mlx_has_checkpoint_at_step(output_dir, step):
        return True
    if step <= 0 or optimizer is None:
        return False
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    if ckpt_dir.is_symlink():
        # Refuse a symlinked dir: it could redirect writes outside output_dir.
        logger.error("Refusing to write MLX stop checkpoint through symlink: %s", ckpt_dir)
        return False
    try:
        ckpt_dir.mkdir(parents = True, exist_ok = True)
        from unsloth_zoo.mlx.utils import (
            save_optimizer_state,
            save_trainable_adapters,
            save_trainer_state,
        )

        save_trainable_adapters(trainer.model, str(ckpt_dir))
        save_optimizer_state(optimizer, str(ckpt_dir))
        save_trainer_state(
            {
                "global_step": step,
                "train_loss_history": list(getattr(trainer, "_train_loss_history", [])),
            },
            str(ckpt_dir),
        )
        logger.info("Saved stop checkpoint to %s", ckpt_dir)
    except Exception:
        logger.exception("Failed to write stop checkpoint under %s", output_dir)
    return _mlx_has_checkpoint_at_step(output_dir, step)


def _create_trainer_progress_callback(event_queue: Any) -> Callable[[TrainingProgress], None]:
    """UnslothTrainer callback that reports training progress to the parent.

    Status events go out only while the status is non-empty, so the empty status the
    trainer reports on every log leaves the parent's last real status standing.

    The trainer shares one TrainingProgress for metrics and status, so a status-only
    update (an evaluation line, a warning) carries the last step's numbers unchanged.
    The parent appends every progress event to the loss / grad-norm / eval-loss
    histories without deduplicating the step, so those replays would plot the same
    point again. Only a changed measurement is published; the status still is.
    """

    sent_warnings: set[str] = set()
    last_metrics: list = [None]

    def _on_progress(progress: TrainingProgress) -> None:
        has_train_loss = progress.step > 0 and progress.loss is not None
        has_eval_loss = progress.eval_loss is not None
        # The end-of-run summary carries no loss (it is the mean, not a step), but it
        # does carry the elapsed time including the final evaluation, checkpoint save
        # and best-model reload, and a run stopped early never reaches total_steps, so
        # the flag the trainer sets on that record is what marks it terminal.
        is_terminal = bool(getattr(progress, "is_run_summary", False)) or (
            progress.total_steps > 0 and progress.step >= progress.total_steps
        )
        # Wall-clock fields are excluded: they move on every call, so keeping them
        # would make each status replay look like a new measurement.
        metrics = (
            progress.step,
            progress.loss,
            progress.learning_rate,
            progress.grad_norm,
            progress.num_tokens,
            progress.epoch,
            progress.eval_loss,
        )
        is_repeat = metrics == last_metrics[0]
        if (
            (progress.step == 0 and progress.total_steps > 0)
            or has_train_loss
            or has_eval_loss
            or is_terminal
        ) and not is_repeat:
            last_metrics[0] = metrics
            event_queue.put(
                {
                    "type": "progress",
                    "step": progress.step,
                    "epoch": progress.epoch,
                    "loss": progress.loss,
                    "learning_rate": progress.learning_rate,
                    "total_steps": progress.total_steps,
                    "elapsed_seconds": progress.elapsed_seconds,
                    "eta_seconds": progress.eta_seconds,
                    "grad_norm": progress.grad_norm,
                    "num_tokens": progress.num_tokens,
                    "eval_loss": progress.eval_loss,
                    "status_message": progress.status_message,
                    "ts": time.time(),
                }
            )
        if progress.status_message:
            _send_status(event_queue, progress.status_message)
        for message in progress.warnings:
            if message not in sent_warnings:
                sent_warnings.add(message)
                event_queue.put({"type": "warning", "message": message, "ts": time.time()})

    return _on_progress


def _create_embedding_progress_callback(
    event_queue: Any,
    *,
    total_steps: int,
    training_start_time: float,
    should_stop: Callable[[], bool],
):
    """TrainerCallback that reports embedding training progress to the parent.

    ``should_stop`` is polled in on_train_begin and on_step_end, so a stop signal
    arriving mid-run is seen.
    """
    from transformers import TrainerCallback

    class _EmbeddingProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Progress events carry an empty status, else the parent keeps showing "Starting...".
            if should_stop():
                return
            _send_status(event_queue, "Training in progress...")

        def on_log(
            self,
            args,
            state,
            control,
            logs = None,
            **kwargs,
        ):
            if not logs:
                return
            # Trainer's end-of-run summary carries train_runtime, samples and steps per
            # second, total_flos and memory, which nothing else here publishes. It used
            # to reach the log only through PrinterCallback's raw stdout dict.
            from core.training.trainer import _RESERVED_LOG_KEYS, _TRAINER_SUMMARY_KEYS

            if any(k in logs for k in _TRAINER_SUMMARY_KEYS):
                logger.info(
                    "trainer summary",
                    **{
                        k: v
                        for k, v in logs.items()
                        if isinstance(k, str) and k not in _RESERVED_LOG_KEYS
                    },
                )
            # See the note in trainer.py: "train_loss" in HF's terminal summary record is
            # the run mean, not a step loss, so it must not become the final step.
            loss_value = logs.get("loss")
            if loss_value is None and logs.get("train_loss") is not None:
                print(
                    f"Training finished: mean train_loss={logs.get('train_loss')} "
                    f"over {state.global_step} steps",
                    flush = True,
                )
            current_step = state.global_step

            elapsed = time.time() - training_start_time
            eta = None
            if current_step > 0 and total_steps > 0:
                remaining = total_steps - current_step
                if remaining > 0:
                    eta = (elapsed / current_step) * remaining

            event_queue.put(
                {
                    "type": "progress",
                    "step": current_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "loss": loss_value,
                    "learning_rate": logs.get("learning_rate", None),
                    "total_steps": total_steps,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "grad_norm": logs.get("grad_norm"),
                    "num_tokens": getattr(state, "num_input_tokens_seen", None),
                    "eval_loss": logs.get("eval_loss"),
                    "status_message": "",
                    "ts": time.time(),
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if should_stop():
                logger.info("Embedding training: stop at step %d", state.global_step)
                control.should_training_stop = True
                return control

    return _EmbeddingProgressCallback()


def _run_embedding_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Self-contained embedding model training pipeline.

    Uses FastSentenceTransformer + SentenceTransformerTrainer +
    MultipleNegativesRankingLoss — separate from UnslothTrainer's LLM/VLM/audio
    paths. Mirrors the reference embedding notebooks:
      All_MiniLM_L6_v2.py, BGE_M3.py, EmbeddingGemma_300M.py,
      ModernBert.py, Qwen3_Embedding_0_6B.py
    """
    import math

    model_name = config["model_name"]
    model_load_name = _resolve_cached_model_load_name(config)
    model_local_only = _model_local_files_only(config)
    model_revision = None if model_local_only else config.get("model_revision")
    training_start_time = time.time()

    # ── 1. Import embedding-specific libraries ──
    _send_status(event_queue, "Importing embedding libraries...")
    try:
        # Recover from a namespace-package shadow (embedding imports unsloth directly).
        from core.import_guards import ensure_real_packages

        ensure_real_packages("unsloth_zoo", "unsloth")
        from unsloth import FastSentenceTransformer, is_bfloat16_supported
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from datasets import Dataset
        from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset
        from utils.paths import datasets_root, resolve_output_dir, default_run_dir_name
    except ImportError as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import embedding libraries: {e}. "
                "Ensure 'sentence_transformers' and 'unsloth' are installed.",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # datasets is only in the process now, and setup_logging ran long before it, so
    # this is the first point where its Generating/Map bars can be quieted. Without
    # it a local JSON/CSV/Parquet or Hub load_dataset writes them into this worker's
    # structured log.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a run
        pass

    # ── Stop signal handling ──
    _should_stop = False
    _save_on_stop = True

    def _apply_stop(save: bool) -> None:
        nonlocal _should_stop, _save_on_stop
        _save_on_stop = save
        _should_stop = True
        logger.info(
            "Embedding training: stop signal received (save=%s)",
            _save_on_stop,
        )

    _start_worker_stop_poller(stop_queue, _apply_stop)

    # ── 2. Load model ──
    _send_status(event_queue, "Loading embedding model...")
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        max_seq_length = config.get("max_seq_length", 512)
        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        security_error = _model_load_security_error(config, model_load_name, hf_token)
        if security_error:
            event_queue.put({"type": "error", **security_error, "ts": time.time()})
            return

        try:
            model = FastSentenceTransformer.from_pretrained(
                model_name = model_load_name,
                max_seq_length = max_seq_length,
                full_finetuning = not use_lora,
                token = hf_token,
                revision = model_revision,
                use_exact_model_name = model_revision is not None,
            )
        except Exception as error:
            if not model_local_only:
                raise
            fallback_error = _model_cache_fallback_error(config, error)
            if fallback_error is not None:
                raise fallback_error from error
            if not _cache_artifact_fallback_allowed(config, error, "model"):
                raise
            _send_status(
                event_queue,
                f"Cached files for {model_name} are incomplete; retrying from Hugging Face...",
            )
            model_load_name = _drop_model_pin_for_fallback(config, hf_token)
            # Scan the Hub target we fall back to, not the cached pin already scanned above.
            security_error = _model_load_security_error(config, model_load_name, hf_token)
            if security_error:
                event_queue.put({"type": "error", **security_error, "ts": time.time()})
                return
            model_local_only = False
            model_revision = config.get("model_revision")
            model = FastSentenceTransformer.from_pretrained(
                model_name = model_load_name,
                max_seq_length = max_seq_length,
                full_finetuning = not use_lora,
                token = hf_token,
                revision = model_revision,
                use_exact_model_name = model_revision is not None,
            )
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load embedding model '{model_name}': {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    loaded_model_for_provenance = model
    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 3. Apply LoRA ──
    if use_lora:
        _send_status(event_queue, "Configuring LoRA adapters (FEATURE_EXTRACTION)...")
        try:
            gradient_checkpointing = config.get("gradient_checkpointing", False)
            # Normalize "none"/empty → False.
            if gradient_checkpointing in ("none", "", None):
                gradient_checkpointing = False

            model = FastSentenceTransformer.get_peft_model(
                model,
                r = config.get("lora_r", 32),
                target_modules = config.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = config.get("lora_alpha", 64),
                lora_dropout = config.get("lora_dropout", 0.0),
                bias = "none",
                use_gradient_checkpointing = gradient_checkpointing,
                random_state = config.get("random_seed", 3407),
                use_rslora = config.get("use_rslora", False),
                use_dora = config.get("use_dora", False),
                loftq_config = {"loftq_bits": 4, "loftq_iter": 1}
                if config.get("use_loftq")
                else None,
                task_type = "FEATURE_EXTRACTION",
            )
        except Exception as e:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Failed to configure LoRA for embedding model: {e}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
            return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 4. Load dataset ──
    _send_status(event_queue, "Loading dataset...")
    try:
        config["_dataset_loaded_from_exact_snapshot"] = False
        hf_dataset = str(config.get("hf_dataset") or "").strip()
        local_datasets = config.get("local_datasets") or []

        def _load_local_embedding_dataset(dataset_paths: list[str]):
            from utils.paths import dataset_files_in_dir

            all_files: list[str] = []
            for dataset_file in dataset_paths:
                file_path = (
                    dataset_file
                    if os.path.isabs(dataset_file)
                    else os.path.join(
                        str(datasets_root()),
                        dataset_file,
                    )
                )
                if os.path.isdir(file_path):
                    file_path_obj = Path(file_path)
                    all_files.extend(str(p) for p in dataset_files_in_dir(file_path_obj))
                else:
                    all_files.append(file_path)

            if not all_files:
                raise ValueError("No local dataset files found")

            first_ext = Path(all_files[0]).suffix.lower()
            if first_ext in (".json", ".jsonl"):
                loader = "json"
            elif first_ext == ".csv":
                loader = "csv"
            elif first_ext == ".parquet":
                loader = "parquet"
            else:
                raise ValueError(f"Unsupported local dataset format: {all_files[0]}")
            return load_dataset(loader, data_files = all_files, split = "train")

        if hf_dataset:
            dataset = _load_embedding_hf_dataset(
                config,
                load_dataset,
                lambda message: _send_status(event_queue, message),
            )
        elif local_datasets:
            dataset = _load_local_embedding_dataset(local_datasets)
        elif config.get("s3_config"):
            from core.training.s3_dataset import (
                S3DownloadCancelled,
                prepare_s3_dataset_download,
            )

            _send_status(event_queue, "Downloading dataset from S3...")
            s3_download = None
            try:
                s3_download = prepare_s3_dataset_download(
                    config["s3_config"],
                    cancel_callback = lambda: _should_stop,
                )
                dataset = _load_local_embedding_dataset(s3_download.files)
            except S3DownloadCancelled:
                event_queue.put(
                    {
                        "type": "complete",
                        "output_dir": None,
                        "status_message": "Training cancelled",
                        "ts": time.time(),
                    }
                )
                return
            finally:
                if s3_download is not None:
                    s3_download.cleanup()
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": "No dataset specified for embedding training.",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        slice_start = config.get("dataset_slice_start")
        slice_end = config.get("dataset_slice_end")
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(dataset)
            dataset = dataset.select(range(start, min(end + 1, len(dataset))))

        logger.info(f"Embedding dataset loaded: {len(dataset)} samples")
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load dataset: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    _emit_resource_provenance(
        event_queue,
        config,
        loaded_model_for_provenance,
        model_load_target = model_load_name,
        model_load_in_4bit = False,
        dataset_loaded_from_exact_snapshot = bool(config.get("_dataset_loaded_from_exact_snapshot")),
    )

    # ── 5. Create loss function ──
    loss = MultipleNegativesRankingLoss(model)

    # ── 6. Build training arguments ──
    _send_status(event_queue, "Configuring training...")
    try:
        lr_value = float(config.get("learning_rate", "2e-4"))
    except ValueError:
        event_queue.put(
            {
                "type": "error",
                "error": f"Invalid learning rate: {config.get('learning_rate')}",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
        resume_from_checkpoint
    )
    if not output_dir:
        output_dir = build_default_output_dir_name(
            model_name,
            config.get("project_name"),
        )
    output_dir = str(resolve_output_dir(output_dir))
    _emit_output_dir(event_queue, output_dir)

    num_epochs = config.get("num_epochs", 2)
    batch_size = config.get("batch_size", 256)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_steps_val = config.get("max_steps", 0)
    save_steps_val = config.get("save_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    warmup_steps_val = config.get("warmup_steps")
    log_frequency = config.get("log_frequency", 50)

    from core.training.trainer import _drop_hf_stdout_callbacks, _hf_stdout_progress_disabled

    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": lr_value,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 1,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": config.get("weight_decay", 0.001),
        "seed": config.get("random_seed", 3407),
        # Same reason as the UnslothTrainer path: this worker has no terminal, its
        # stdout is teed into the server log, and _create_embedding_progress_callback
        # already publishes every number the bar carries.
        "disable_tqdm": _hf_stdout_progress_disabled(),
    }

    if max_steps_val and max_steps_val > 0:
        training_args_kwargs["max_steps"] = max_steps_val
    else:
        training_args_kwargs["num_train_epochs"] = num_epochs if num_epochs > 0 else 2

    # warmup: prefer warmup_ratio (standard for embedding scripts), else steps
    if warmup_ratio is not None and warmup_ratio > 0:
        training_args_kwargs["warmup_ratio"] = warmup_ratio
    elif warmup_steps_val is not None and warmup_steps_val > 0:
        training_args_kwargs["warmup_steps"] = warmup_steps_val

    if save_steps_val and save_steps_val > 0:
        training_args_kwargs["save_steps"] = save_steps_val
        training_args_kwargs["save_strategy"] = "steps"

    args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # ── 7. Calculate total steps for progress tracking ──
    if max_steps_val and max_steps_val > 0:
        total_steps = max_steps_val
    else:
        effective_epochs = num_epochs if num_epochs > 0 else 2
        len_dataloader = math.ceil(len(dataset) / batch_size)
        steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * effective_epochs

    # ── 8. Create progress callback ──
    progress_callback = _create_embedding_progress_callback(
        event_queue,
        total_steps = total_steps,
        training_start_time = training_start_time,
        should_stop = lambda: _should_stop,
    )

    # ── 9. Create trainer and train ──
    _send_status(event_queue, "Starting embedding training...")
    try:
        trainer = SentenceTransformerTrainer(
            model = model,
            train_dataset = dataset,
            loss = loss,
            args = args,
            callbacks = [progress_callback],
        )
        # disable_tqdm only swaps ProgressCallback for PrinterCallback, which prints a
        # raw dict per step instead; both write to the same stdout.
        _drop_hf_stdout_callbacks(trainer)

        trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Embedding training failed: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 10. Save model ──
    if _should_stop and not _save_on_stop:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training cancelled",
                "ts": time.time(),
            }
        )
        return

    _send_status(event_queue, "Saving model...")
    try:
        if _should_stop and _save_on_stop:
            trainer._save_checkpoint(trainer.model, trial = None)
        model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        logger.info("Embedding model saved to %s", output_dir)
    except Exception as e:
        logger.error("Failed to save embedding model: %s", e)
        event_queue.put(
            {
                "type": "error",
                "error": f"Training completed but failed to save: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 11. Done ──
    event_queue.put(
        {
            "type": "complete",
            "output_dir": output_dir,
            "status_message": "Embedding training completed",
            "ts": time.time(),
        }
    )
