# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# A plain language model plus a fresh joint schema head = a Clef-style decision model that trains,
# evaluates and saves like a Cloudflare Clef checkpoint.

__all__ = [
    "load_lm_as_decision_model",
    "default_head_config",
    "freeze_backbone",
    "unfreeze_backbone",
]

import json
import warnings
from pathlib import Path
from typing import Optional

import torch

DECISION_HEADS = ("clef",)
_SOURCE_FILES = ["*.json", "*.jinja", "LICENSE*", "*.txt", "*.model"]


def _text_config(config):
    return getattr(config, "text_config", None) or config


def default_head_config(hidden_size: int, width: Optional[int] = None) -> dict:
    # Clef's head: width 1024, 2 routing + 4 decoder layers, 16 heads, ff 4096; small backbones get width 512.
    if width is None:
        width = 1024 if hidden_size >= 3072 else 512
    width = int(width)
    heads = max(1, width // 64)
    return dict(
        hidden_size = int(hidden_size),
        width = width,
        routing_layers = 2,
        layers = 4,
        heads = heads,
        feedforward = 4 * width,
    )


def _source_folder(model_name, token, revision, local_files_only) -> Optional[Path]:
    root = Path(str(model_name)).expanduser()
    if root.is_dir():
        return root
    try:
        from huggingface_hub import snapshot_download
        return Path(
            snapshot_download(
                str(model_name),
                token = token,
                revision = revision,
                local_files_only = local_files_only,
                allow_patterns = _SOURCE_FILES,
            )
        )
    except Exception:
        return None


def _head_from(head_init, hidden_size, token, revision):
    from safetensors.torch import load_file

    from .clef import HEAD_FILES, JointSchemaHead

    folder = Path(str(head_init)).expanduser()
    if not folder.is_dir():
        from huggingface_hub import snapshot_download
        folder = Path(
            snapshot_download(str(head_init), token = token, allow_patterns = list(HEAD_FILES))
        )
    config = json.loads((folder / HEAD_FILES[1]).read_text(encoding = "utf-8"))
    if int(config["hidden_size"]) != int(hidden_size):
        raise ValueError(
            f"Unsloth: the head in {head_init} reads hidden size {config['hidden_size']}, "
            f"but this backbone has hidden size {hidden_size}."
        )
    warnings.warn(
        f"Unsloth: starting from the decision head in {head_init}. A released Clef head was trained "
        "on its own fine-tuned backbone, so it is only a warm start for another backbone.",
        stacklevel = 3,
    )
    head = JointSchemaHead(**config)
    head.load_state_dict(load_file(str(folder / HEAD_FILES[0])), strict = True)
    return head


def _load_backbone(model_name, max_len, dtype, load_in_4bit, full_finetuning, token, gc, kwargs):
    from .decision import _clef_bnb_config, _device, _pin_device_map

    if _device().type != "cpu":
        from .loader import FastModel

        # Same rules as decision._load_clef: dynamic 4-bit config, float16 puts Qwen3.5 on the float32 path.
        if load_in_4bit and not full_finetuning and kwargs.get("quantization_config") is None:
            kwargs["quantization_config"] = _clef_bnb_config(dtype)
        _pin_device_map(kwargs)
        backbone, processor = FastModel.from_pretrained(
            str(model_name),
            max_seq_length = max_len,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            full_finetuning = full_finetuning,
            token = token,
            use_gradient_checkpointing = gc,
            **kwargs,
        )
        return backbone, processor, True
    if load_in_4bit:
        raise NotImplementedError("Unsloth: load_in_4bit needs a GPU.")
    from transformers import AutoConfig, AutoTokenizer

    hub = {
        "token": token,
        "revision": kwargs.get("revision"),
        "local_files_only": kwargs.get("local_files_only", False),
    }
    config = AutoConfig.from_pretrained(str(model_name), **hub)
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText as AutoClass, AutoProcessor
        processor = AutoProcessor.from_pretrained(str(model_name), **hub)
    else:
        from transformers import AutoModelForCausalLM as AutoClass
        processor = AutoTokenizer.from_pretrained(str(model_name), **hub)
    backbone = AutoClass.from_pretrained(str(model_name), dtype = dtype or torch.float32, **hub)
    if not full_finetuning:
        backbone.requires_grad_(False)
    return backbone, processor, False


def load_lm_as_decision_model(
    model_name: str,
    decision_head: str = "clef",
    head_width: Optional[int] = None,
    head_init: Optional[str] = None,
    head_config: Optional[dict] = None,
    max_seq_length: Optional[int] = None,
    dtype = None,
    load_in_4bit: bool = False,
    full_finetuning: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    use_gradient_checkpointing = "unsloth",
    random_state: int = 3407,
    **kwargs,
):
    from .clef import JointSchemaHead
    from .decision import (
        CLEF_MAX_LEN,
        ClefDecisionModel,
        _attach_clef_saving,
        _mark_full_finetuning,
    )

    if decision_head not in DECISION_HEADS:
        raise ValueError(
            f"Unsloth: decision_head must be one of {DECISION_HEADS}, not {decision_head!r}."
        )
    max_len = int(max_seq_length or CLEF_MAX_LEN)
    if revision is not None:
        kwargs["revision"] = revision
    if local_files_only:
        kwargs["local_files_only"] = True
    backbone, processor, fast = _load_backbone(
        model_name,
        max_len,
        dtype,
        load_in_4bit,
        full_finetuning,
        token,
        use_gradient_checkpointing,
        kwargs,
    )
    backbone.config.use_cache = False
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not callable(getattr(backbone, "get_output_embeddings", None)) or (
        backbone.get_output_embeddings() is None
    ):
        raise ValueError(
            f"Unsloth: {model_name} has no output embeddings, which the decision head reads."
        )
    hidden_size = int(_text_config(backbone.config).hidden_size)
    if head_init is not None:
        head = _head_from(head_init, hidden_size, token, revision)
    else:
        config = head_config or default_head_config(hidden_size, head_width)
        if int(config["hidden_size"]) != hidden_size:
            raise ValueError(
                f"Unsloth: head_config hidden_size {config['hidden_size']} != backbone {hidden_size}."
            )
        with torch.random.fork_rng(devices = []):
            torch.manual_seed(random_state)
            head = JointSchemaHead(**config)
    device = next(backbone.parameters()).device
    model = ClefDecisionModel(backbone, head.to(device = device, dtype = torch.float32))
    model.decision_config = {
        "layout": "clef",
        "max_len": max_len,
        "temperature": [1.0] * 3,
        "base_model": str(model_name),
        **({"base_revision": revision} if revision else {}),
        # How the backbone loaded, so a server puts saved adapters back on the same base.
        "load_in_4bit": bool(load_in_4bit),
    }
    _mark_full_finetuning(model, full_finetuning)
    model._unsloth_forced_float32 = bool(getattr(backbone, "_unsloth_forced_float32", False))
    model._unsloth_fast_backbone = fast
    model._saved_temp_tokenizer = processor
    model._unsloth_source_vocab = len(tokenizer)
    source = _source_folder(model_name, token, revision, local_files_only)
    model._unsloth_source_folder = str(source) if source is not None else ""
    _attach_clef_saving(model)
    return model, processor


def freeze_backbone(model):
    # Stage A of the from-LM recipe: train only the decision head on frozen backbone features.
    frozen = [name for name, param in model.encoder.named_parameters() if param.requires_grad]
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    model._unsloth_frozen_backbone = frozen
    return model


def unfreeze_backbone(model):
    names = set(getattr(model, "_unsloth_frozen_backbone", ()))
    for name, param in model.encoder.named_parameters():
        if name in names:
            param.requires_grad_(True)
    model._unsloth_frozen_backbone = []
    return model
