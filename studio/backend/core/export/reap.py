# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in REAP expert pruning before an export (Core's ``FastLanguageModel.reap_prune``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

# Same keys Core's reap.py reads; checked on text_config first, then the top-level config.
_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k", "moe_top_k", "top_k_experts", "moe_topk", "moe_k")
REPORT_NAME = "reap_report.json"


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
    """``{num_experts, top_k}`` for a routed-MoE config dict, else None."""
    if not isinstance(config, dict):
        return None
    for holder in (config.get("text_config"), config):
        if not isinstance(holder, dict):
            continue
        experts = _first_int(holder, _NUM_EXPERTS_KEYS)
        top_k = _first_int(holder, _TOP_K_KEYS, per_layer = True)
        if experts and top_k and experts > top_k:
            return {"num_experts": experts, "top_k": top_k}
    return None


def moe_export_info(
    model: str,
    hf_token = None,
    local_files_only = True,
    _depth = 0,
) -> Optional[dict]:
    """MoE info for a local checkpoint / model folder or a Hub id (a LoRA folder resolves to its
    base). Never raises: an unreadable config just hides the option."""
    try:
        from utils.models.diffusion_lm import _local_adapter_base
        from utils.models.model_config import _read_raw_config_json
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model):
            local = Path(normalize_path(model)).expanduser()
            if not (local / "config.json").is_file():
                base = _local_adapter_base(str(local))
                if not base or _depth >= 2 or base == model:
                    return None
                return moe_export_info(base, hf_token, local_files_only, _depth + 1)
        return moe_info_from_config(_read_raw_config_json(model, hf_token, local_files_only, None))
    except Exception as exc:
        logger.debug("MoE detection skipped for '%s': %s", model, exc)
        return None


def _options_key(options: dict) -> Tuple[float, int, Optional[str]]:
    return (
        round(float(options.get("sparsity", 0.25)), 4),
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


def apply_reap(backend: Any, options: Optional[dict], staging_parent) -> Tuple[bool, str]:
    """Prune the loaded model once per load; a multi-format export reuses the pruned model."""
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
    from core.export.export import _staging_dir
    from unsloth import FastLanguageModel

    if getattr(backend, "is_peft", False):
        # REAP scores the fine-tuned experts, so the adapters are merged first (16-bit, in memory).
        logger.info("REAP: merging the LoRA adapters into the 16-bit base before pruning...")
        model = model.merge_and_unload()
        backend.current_model = model
        backend.is_peft = False
    staging = _staging_dir(staging_parent)
    sparsity, samples, dataset = key
    # save_directory repoints the model at the pruned copy, so every later save, merge or GGUF
    # conversion reads the pruned weights and never the original checkpoint. GGUF files are named
    # after that folder, so it carries the model's own name.
    name = Path(str(getattr(model.config, "_name_or_path", "") or "model").rstrip("/\\")).name
    FastLanguageModel.reap_prune(
        model,
        tokenizer,
        sparsity = sparsity,
        dataset = dataset,
        num_calibration_samples = samples,
        save_directory = str(Path(staging.name) / f"{name or 'model'}-REAP"),
    )
    backend._reap_key = key
    backend._reap_staging = staging
    report = getattr(model, "_unsloth_reap_report", None) or {}
    kept = len(next(iter(report.values()), []))
    return True, f"REAP kept {kept} experts per MoE layer in {len(report)} layers."


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
