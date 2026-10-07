# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF export of decision models (Clef, Laya) for the Export page.

The server side (``decision_preview``) only reads files: the parent process must not import
unsloth. The worker side (``run_decision_gguf_export``) calls unsloth.models.decision_gguf,
whose ``gguf_eligibility`` is the authoritative gate.
"""

import json
from pathlib import Path
from typing import Optional

# Mirrors unsloth.models.decision_gguf.DECISION_GGUF_QUANTIZATIONS (first = default); a test pins it.
DECISION_GGUF_QUANTIZATIONS = ("q8_0", "f16", "bf16", "q6_k", "q5_k_m", "q4_k_m")
_CLEF_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
_CLEF_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
_LAYA_FILES = ("rl_agent_config.json", "model.safetensors")
_LAYA_DIRS = ("encoder", "tokenizer")


class DecisionExportError(ValueError):
    """A decision checkpoint that cannot be exported to GGUF; the message is shown to the user."""


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _local_dir(checkpoint_path) -> Optional[Path]:
    if not checkpoint_path:
        return None
    try:
        folder = Path(str(checkpoint_path)).expanduser()
        return folder if folder.is_dir() else None
    except (OSError, ValueError):
        return None


def decision_kind(checkpoint_path) -> Optional[tuple]:
    """(layout, adapter_only) for a local decision checkpoint folder, else None. File checks only."""
    folder = _local_dir(checkpoint_path)
    if folder is None:
        return None
    # Same file rules as unsloth.models.decision.is_decision_checkpoint; merged weights win.
    if all((folder / name).is_file() for name in _CLEF_HEAD_FILES):
        if (folder / "config.json").is_file():
            return "clef", False
        if (folder / "adapter_config.json").is_file():
            return "clef", True
    if all((folder / name).is_file() for name in _LAYA_FILES) and all(
        (folder / name).is_dir() for name in _LAYA_DIRS
    ):
        return "laya", False
    return None


def _cached_base_config(base: str) -> Optional[dict]:
    local = _local_dir(base)
    if local is not None:
        return _read_json(local / "config.json")
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(base, "config.json")
    except Exception:
        return None
    return _read_json(Path(cached)) if isinstance(cached, str) else None


def _preview_eligibility(folder: Path, layout: str) -> tuple:
    """(eligible, reason); eligible None when only the worker can tell (base config not on disk)."""
    if layout == "laya":
        encoder = _read_json(folder / "encoder" / "config.json") or {}
        model_type = encoder.get("model_type")
        if model_type == "modernbert":
            return True, None
        return False, (
            "GGUF export of Laya decision models needs a ModernBERT encoder, and this one is "
            f"{model_type or 'unknown'}. It still serves through PyTorch in Studio's Decision API."
        )
    config = _read_json(folder / "config.json")
    if config is None:
        base = (_read_json(folder / "unsloth_decision_config.json") or {}).get("base_model") or (
            _read_json(folder / "adapter_config.json") or {}
        ).get("base_model_name_or_path")
        config = _cached_base_config(str(base)) if base else None
    if config is None:
        return None, None
    architectures = list(config.get("architectures") or [])
    if any(name in _CLEF_ARCHITECTURES for name in architectures):
        return True, None
    return False, (
        "GGUF export of this decision model is not supported: llama.cpp's decision graph is "
        f"built for Qwen3.5 backbones only, and this one is {', '.join(architectures) or 'unknown'}. "
        "It still serves through PyTorch in Studio's Decision API."
    )


def read_existing_export(folder) -> Optional[dict]:
    try:
        from core.systemone.gguf_export_contract import read_export
    except Exception:
        return None
    try:
        return read_export(folder)
    except Exception:
        return None


def current_export(folder, layout: str) -> Optional[dict]:
    """export.json when it was made from the folder as it is now, listing only quantizations
    whose files exist; None for a missing, stale or empty export."""
    try:
        from core.systemone.gguf_export_contract import EXPORT_DIR, fingerprint
    except Exception:
        return None
    data = read_existing_export(folder)
    if data is None or data["layout"] != layout:
        return None
    try:
        if data["source_fingerprint"] != fingerprint(folder, layout):
            return None
    except Exception:
        return None
    directory = Path(folder) / EXPORT_DIR
    present = {}
    for quant in data["quantizations"]:
        entry = data["files"][quant]
        names = [entry["model"]] + ([entry["mmproj"]] if entry.get("mmproj") else [])
        if quant not in present and all((directory / name).is_file() for name in names):
            present[quant] = entry
    if not present:
        return None
    return {**data, "quantizations": list(present), "files": present}


def decision_preview(checkpoint_path) -> Optional[dict]:
    """The Export page's decision block for a local folder, or None for a non-decision model."""
    kind = decision_kind(checkpoint_path)
    if kind is None:
        return None
    layout, adapter_only = kind
    folder = Path(str(checkpoint_path)).expanduser()
    eligible, reason = _preview_eligibility(folder, layout)
    return {
        "is_decision": True,
        "layout": layout,
        "adapter_only": adapter_only,
        "eligible": eligible,
        "reason": reason,
        "quantizations": list(DECISION_GGUF_QUANTIZATIONS),
        "default_quantization": DECISION_GGUF_QUANTIZATIONS[0],
        "output_dir": str(folder / "gguf"),
        "existing_export": current_export(folder, layout),
    }


def normalize_decision_quants(quantization_method) -> list:
    methods = (
        [quantization_method]
        if isinstance(quantization_method, str)
        else list(quantization_method or [])
    )
    chosen = []
    for method in methods:
        method = str(method).strip().lower()
        if not method:
            continue
        if method not in DECISION_GGUF_QUANTIZATIONS:
            raise DecisionExportError(
                f"Decision models export to GGUF as one of "
                f"{', '.join(q.upper() for q in DECISION_GGUF_QUANTIZATIONS)}, not {method.upper()}."
            )
        if method not in chosen:
            chosen.append(method)
    return chosen or [DECISION_GGUF_QUANTIZATIONS[0]]


def check_decision_eligibility(checkpoint_path) -> dict:
    """gguf_eligibility from the library; raises DecisionExportError with its reason when ineligible."""
    from unsloth.models.decision_gguf import gguf_eligibility

    eligibility = gguf_eligibility(str(Path(str(checkpoint_path)).expanduser()))
    if not eligibility.get("eligible"):
        raise DecisionExportError(
            eligibility.get("reason") or "This decision model cannot be exported to GGUF."
        )
    return eligibility


def run_decision_gguf_export(
    checkpoint_path,
    quantization_method,
    local_files_only: bool = False,
    print_output: bool = False,
    token = None,
) -> dict:
    """Writes <checkpoint_path>/gguf/ and returns its export.json content. ``token`` reaches the
    base model download of an adapter folder (False = anonymous, None = ambient)."""
    folder = Path(str(checkpoint_path)).expanduser()
    quants = normalize_decision_quants(quantization_method)
    kind = decision_kind(folder)
    if kind is None:
        raise DecisionExportError(f"{folder} is not a decision model checkpoint.")
    check_decision_eligibility(folder)
    _, adapter_only = kind
    if adapter_only:
        # Adapters only: FastDecisionModel keeps the head and calibration, save_pretrained_gguf merges.
        from unsloth import FastDecisionModel
        model, processor = FastDecisionModel.from_pretrained(
            str(folder),
            load_in_4bit = False,
            use_gradient_checkpointing = False,
            local_files_only = local_files_only,
            token = token,
        )
        try:
            return model.save_pretrained_gguf(
                str(folder),
                processor,
                quantization_method = quants,
                source_folder = str(folder),
                print_output = print_output,
                token = token,
            )
        finally:
            del model
    from unsloth.models.decision_gguf import export_decision_gguf

    return export_decision_gguf(
        str(folder),
        quants,
        source_folder = str(folder),
        print_output = print_output,
    )
