# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in REAP expert pruning before an export (Core's ``FastLanguageModel.reap_prune``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

# Core's unsloth/models/reap.py keys, plus the raw config.json names transformers maps onto them
# through attribute_map (HunYuan moe_topk, Ernie moe_k), which Core reads as num_experts_per_tok
# but a raw config dict does not; checked on text_config first, then the top-level config.
_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k", "moe_top_k", "top_k_experts", "moe_topk", "moe_k")
_N_GROUP_KEYS = ("n_group", "num_expert_group")
_TOPK_GROUP_KEYS = ("topk_group", "top_k_group")
REPORT_NAME = "reap_report.json"
# Written into each staging copy (the export worker's pid), so a parent can remove what a killed
# worker left behind without trusting a path it did not create.
_OWNER_FILE = ".unsloth_reap_owner"


def _first_int(
    config: dict,
    keys,
    per_layer = False,
) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if (
            per_layer
            and isinstance(value, list)
            and value
            and all(isinstance(v, int) for v in value)
        ):
            return max(value)
    return None


def moe_info_from_config(config: Optional[dict]) -> Optional[dict]:
    """``{num_experts, top_k}`` (plus ``n_group`` / ``topk_group`` for group-limited routers) for a
    routed-MoE config dict, else None."""
    if not isinstance(config, dict):
        return None
    holders = [h for h in (config.get("text_config"), config) if isinstance(h, dict)]

    # Like Core's _first_attr: each field from the first level that has it, text_config first.
    def first(keys, per_layer = False):
        return next((v for h in holders if (v := _first_int(h, keys, per_layer)) is not None), None)

    # Core refuses routers with identity (zero) experts and bidirectional / diffusion models.
    if any(h.get("zero_expert_num") for h in holders) or any(
        h.get("use_bidirectional_attention") == "all"
        or str(h.get("model_type", "")).startswith("diffusion")
        for h in holders
    ):
        return None
    experts = first(_NUM_EXPERTS_KEYS)
    top_k = first(_TOP_K_KEYS, per_layer = True)
    if not (experts and top_k and experts > top_k):
        return None
    info = {"num_experts": experts, "top_k": top_k}
    # Core prunes per group only when both are set, so the UI must count the same way.
    n_group, topk_group = first(_N_GROUP_KEYS), first(_TOPK_GROUP_KEYS)
    if n_group and topk_group:
        info.update(n_group = n_group, topk_group = topk_group)
    return info


def moe_export_info(
    model: str,
    hf_token = None,
    local_files_only = True,
    _depth = 0,
    adapter_base: Optional[str] = None,
) -> Optional[dict]:
    """MoE info for a local checkpoint / model folder or a Hub id (a LoRA folder resolves to its
    base, or to ``adapter_base`` when the caller already authorized it). Never raises: an
    unreadable config just hides the option."""
    try:
        from utils.models.diffusion_lm import _local_adapter_base
        from utils.models.model_config import _read_raw_config_json
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model):
            local = Path(normalize_path(model)).expanduser()
            if not (local / "config.json").is_file():
                base = adapter_base or _local_adapter_base(str(local))
                if not base or _depth >= 2 or base == model:
                    return None
                # A Hub base of a local adapter may not be cached yet: read it online.
                return moe_export_info(base, hf_token, is_local_path(base), _depth + 1)
        return moe_info_from_config(_read_raw_config_json(model, hf_token, local_files_only, None))
    except Exception as exc:
        logger.debug("MoE detection skipped for '%s': %s", model, exc)
        return None


def _options_key(options: dict) -> Tuple[float, int, Optional[str]]:
    return (
        float(options.get("sparsity", 0.25)),
        int(options.get("num_calibration_samples", 512)),
        (options.get("dataset") or "").strip() or None,
    )


def _loaded_in_4bit(model: Any) -> bool:
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    if quant is not None:
        get = quant.get if isinstance(quant, dict) else lambda k, d = None: getattr(quant, k, d)
        if get("load_in_4bit", False) or get("load_in_8bit", False):
            return True
    return False


def _calibration_dataset(dataset: Optional[str], hf_token):
    """A custom dataset opened with the request's token (export workers keep tokens out of their
    environment, so Core's own load would be anonymous); the default stays Core's."""
    if not dataset:
        return None
    from datasets import load_dataset

    kwargs = {} if hf_token is None else {"token": hf_token}
    # Streaming reads only the rows Core uses.
    return load_dataset(dataset, split = "train", streaming = True, **kwargs)


def apply_reap(
    backend: Any,
    options: Optional[dict],
    staging_parent,
    hf_token = None,
) -> Tuple[bool, str]:
    """Prune the loaded model once per load; a multi-format export reuses the pruned model.
    ``hf_token`` opens a private or gated calibration dataset."""
    if not options:
        return True, ""
    key = _options_key(options)
    applied = getattr(backend, "_reap_key", None)
    if applied == key:
        return True, ""
    if applied is not None:
        return False, (
            "This model was already pruned with different REAP settings. Start the export again "
            "to load a fresh copy."
        )
    model, tokenizer = backend.current_model, backend.current_tokenizer
    if model is None:
        return False, "No model loaded. Please select a checkpoint first."
    if _loaded_in_4bit(model):
        return False, (
            "REAP expert pruning needs the model loaded in 16-bit, but it was loaded in 4-bit. "
            "Turn off 4-bit loading and export again."
        )
    if getattr(model, "__dict__", {}).get("_unsloth_mtp_head") is not None:
        # Core refuses this: the attached MTP head would keep the old expert count.
        return False, (
            "REAP cannot prune a checkpoint with an attached MTP head. Export it without expert "
            "pruning."
        )
    from core.export.export import _staging_dir
    from unsloth import FastLanguageModel

    try:
        if getattr(backend, "is_peft", False):
            # REAP scores the fine-tuned experts, so the adapters are merged first (16-bit, in memory).
            logger.info("REAP: merging the LoRA adapters into the 16-bit base before pruning...")
            model = model.merge_and_unload()
            backend.current_model = model
            backend.is_peft = False
        sweep_reap_staging()
        staging = _staging_dir(staging_parent)
        _register_staging(staging.name)
        # Attached now, so an unload after a failure below also removes it.
        backend._reap_staging = staging
        sparsity, samples, dataset = key
        # save_directory repoints the model at the pruned copy, so every later save, merge or GGUF
        # conversion reads the pruned weights and never the original checkpoint. GGUF files are named
        # after that folder, so it carries the model's own name.
        source = str(getattr(model.config, "_name_or_path", "") or "")
        name = Path(source.rstrip("/\\") or "model").name
        FastLanguageModel.reap_prune(
            model,
            tokenizer,
            sparsity = sparsity,
            dataset = _calibration_dataset(dataset, hf_token),
            num_calibration_samples = samples,
            save_directory = str(Path(staging.name) / f"{name or 'model'}-REAP"),
        )
    except Exception as exc:
        # A failure part way leaves the loaded model merged and partly pruned; never export it.
        logger.warning("REAP failed, unloading the partly pruned model: %s", exc)
        try:
            backend.cleanup_memory()
        except Exception as cleanup_exc:
            logger.warning("Could not unload after a failed REAP run: %s", cleanup_exc)
        raise RuntimeError(
            f"REAP expert pruning failed: {exc}. The model was unloaded since it may be partly "
            "pruned; load it again to export."
        ) from exc
    # The model now points at the staging copy; Hub cards should name the model it came from.
    model._unsloth_reap_source = source or None
    backend._reap_key = key
    report = getattr(model, "_unsloth_reap_report", None) or {}
    kept = len(next(iter(report.values()), []))
    return True, f"REAP kept {kept} experts per MoE layer in {len(report)} layers."


def sync_reap_report(backend: Any, output_path: Optional[str]) -> None:
    """After a successful export: the report when the exported model is pruned, else remove one a
    previous REAP export left in a reused folder, so the folder never describes other weights."""
    if getattr(backend, "_reap_key", None) is not None:
        write_reap_report(backend, output_path)
        return
    if output_path and Path(output_path).is_dir():
        stale = Path(output_path) / REPORT_NAME
        try:
            stale.unlink(missing_ok = True)
        except OSError as exc:
            logger.warning(f"Could not remove the stale {REPORT_NAME}: {exc}")


def write_reap_report(backend: Any, output_path: Optional[str]) -> None:
    """``reap_report.json`` (kept expert ids per layer and the settings) next to the export."""
    model = getattr(backend, "current_model", None)
    report = getattr(model, "_unsloth_reap_report", None)
    key = getattr(backend, "_reap_key", None)
    if not report or key is None or not output_path or not Path(output_path).is_dir():
        return
    try:
        payload = {
            "method": "REAP (arXiv:2510.13999)",
            "sparsity": key[0],
            "num_calibration_samples": key[1],
            "dataset": key[2] or "mlabonne/open-perfectblend",
            "kept_experts": report,
        }
        (Path(output_path) / REPORT_NAME).write_text(
            json.dumps(payload, indent = 1), encoding = "utf-8"
        )
    except Exception as exc:
        logger.warning(f"Could not write {REPORT_NAME}: {exc}")


def discard_reap_staging(backend: Any) -> None:
    staging = getattr(backend, "_reap_staging", None)
    backend._reap_staging = None
    backend._reap_key = None
    if staging is not None:
        try:
            staging.cleanup()
        except Exception as exc:
            logger.warning(f"Could not remove the REAP staging copy: {exc}")
        sweep_reap_staging()


def reap_staging_parent(save_directory: Optional[str]) -> Path:
    """Where the pruned copy may stage: the chosen save folder's filesystem (an absolute path on
    another drive is allowed), else the exports root. ``_staging_dir`` then takes whichever of
    this and the temp dir has more room."""
    from utils.paths.storage_roots import resolve_export_dir, resolve_export_write_dir

    try:
        target = resolve_export_write_dir(save_directory)
    except Exception:
        return resolve_export_dir()
    # The save folder itself may not exist yet; its nearest existing ancestor is on the same drive.
    for candidate in (target, *target.parents):
        if candidate.is_dir():
            return candidate
    return resolve_export_dir()


def _registry() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / "reap_staging.json"


def _read_registry() -> list:
    try:
        entries = json.loads(_registry().read_text(encoding = "utf-8"))
        return [e for e in entries if isinstance(e, dict) and "path" in e and "pid" in e]
    except Exception:
        return []


def _write_registry(entries: list) -> None:
    try:
        path = _registry()
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries), encoding = "utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not update the REAP staging registry: %s", exc)


def _register_staging(path: str) -> None:
    import os

    pid = os.getpid()
    try:
        (Path(path) / _OWNER_FILE).write_text(str(pid), encoding = "utf-8")
    except OSError as exc:
        logger.debug("Could not mark the REAP staging copy: %s", exc)
    _write_registry(_read_registry() + [{"path": str(path), "pid": pid}])


def sweep_reap_staging(dead_pid: Optional[int] = None) -> None:
    """Remove staging copies whose worker is gone: a cancel kills the worker, so neither
    ``cleanup_memory`` nor the TemporaryDirectory finalizer runs. ``dead_pid`` is a worker the
    caller just killed; otherwise an owner counts as gone when its process no longer exists.
    Only folders carrying their owner marker are touched. Never raises."""
    import shutil

    try:
        import psutil
        alive = psutil.pid_exists
    except Exception:
        alive = None
    entries = _read_registry()
    if not entries:
        # Nothing was ever staged: no file is created or rewritten for users who never use REAP.
        return
    keep = []
    for entry in entries:
        path, pid = Path(entry["path"]), entry["pid"]
        if not path.exists():
            continue
        gone = pid == dead_pid or (dead_pid is None and alive is not None and not alive(pid))
        try:
            owned = (path / _OWNER_FILE).read_text(encoding = "utf-8").strip() == str(pid)
        except OSError:
            owned = False
        if gone and owned:
            shutil.rmtree(path, ignore_errors = True)
            if not path.exists():
                logger.info("Removed a REAP staging copy left by a stopped export: %s", path)
                continue
        keep.append(entry)
    if keep != entries:
        _write_registry(keep)
