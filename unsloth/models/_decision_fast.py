# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Laya's encoder is launch bound at decision sizes, so long runs compile each layer.
# UNSLOTH_DECISION_COMPILE=0 or 1 turns it off or forces it on.

import contextlib
import functools
import importlib.util
import os
from typing import Optional

import torch
import torch.nn as nn


__all__ = ["compiled_encoder", "pad_length"]

# Shorter runs spend more on a cold compile than it saves.
COMPILE_MIN_FORWARDS = 4000
# Static shapes launch far cheaper than dynamic ones on short decisions, one graph per 64-token bucket
# (G4: full fine-tune 0.528 -> 0.482 s/step at 2 x 16). Past STATIC_MAX_LEN there are too many buckets
# to compile, so long-context data keeps dynamic shapes.
STATIC_MULTIPLE = 64
STATIC_MAX_LEN = 1024


def _encoder_sdpa(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout = 0.0,
    scaling = None,
    **kwargs,
):
    # transformers' SDPA attention without unsloth_zoo's wrappers, which carry a __module__ that
    # torch 2.11's dynamo cannot guard, so the compiled layers fell back to eager on Linux.
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask = attention_mask, dropout_p = dropout, scale = scaling
    )
    return out.transpose(1, 2).contiguous(), None


@contextlib.contextmanager
def _compilable_attention(model, enabled: bool):
    config = model.encoder.config
    original = config._attn_implementation
    if enabled and original == "sdpa":
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask

        AttentionInterface.register("unsloth_decision_sdpa", _encoder_sdpa)
        AttentionMaskInterface.register("unsloth_decision_sdpa", sdpa_mask)
        config._attn_implementation = "unsloth_decision_sdpa"
    try:
        yield
    finally:
        config._attn_implementation = original


def _compile_static(function):
    return torch.compile(function, dynamic = False)


def _training_only(layer, compiled):
    # Evaluation batches have arbitrary lengths, which static graphs would compile one by one.
    def call(*args, **kwargs):
        return compiled(*args, **kwargs) if layer.training else layer._call_impl(*args, **kwargs)

    return call


@contextlib.contextmanager
def _recompile_limit(limit: int):
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    original = getattr(config, name)
    setattr(config, name, max(original, limit))
    try:
        yield
    finally:
        setattr(config, name, original)


def _torch_checkpoint():
    # torch's own checkpoint, also once unsloth_zoo has patched torch.utils.checkpoint.checkpoint.
    module = torch.utils.checkpoint
    for func in (module.checkpoint, getattr(module, "_old_checkpoint", None)):
        if getattr(func, "__module__", None) == module.__name__:
            return func
    return None


def _encoder_layers(model):
    if not isinstance(getattr(model, "head", None), nn.TransformerEncoder):
        return []
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "get_base_model"):
        encoder = encoder.get_base_model()
    layers = getattr(encoder, "layers", None)
    return list(layers) if isinstance(layers, nn.ModuleList) else []


def _on_gpu(model) -> bool:
    return next(model.parameters()).device.type == "cuda"


def _wants_compile(model, forwards: int) -> bool:
    choice = os.environ.get("UNSLOTH_DECISION_COMPILE", "auto")
    if choice in ("0", "1"):
        return choice == "1"
    return (
        _on_gpu(model)
        and forwards >= COMPILE_MIN_FORWARDS
        and importlib.util.find_spec("triton") is not None
    )


def _warm_up(model, amp_dtype) -> None:
    # Fails here, not mid-run, on a platform Inductor cannot serve; RNG and gradients are restored.
    device = next(model.parameters()).device
    vocab = int(getattr(model.encoder.config, "vocab_size", 1000))
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in params]
    training = model.training
    accelerator = device.type != "cpu"
    devices = [device.index or 0] if accelerator else []
    try:
        model.train()
        with torch.random.fork_rng(
            devices = devices, device_type = device.type if accelerator else "cuda"
        ):
            ids = torch.randint(5, min(vocab, 1000), (2, 64), device = device)
            mask = torch.ones_like(ids)
            mask[1, 48:] = 0
            with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
                h = model.encoder(input_ids = ids, attention_mask = mask).last_hidden_state
            if h.requires_grad:
                h.float().sum().backward()
    finally:
        model.train(training)
        for p, g in zip(params, grads):
            p.grad = g


@contextlib.contextmanager
def compiled_encoder(
    model,
    forwards: int,
    amp_dtype = None,
    max_length: Optional[int] = None,
):
    """Compile each Laya encoder layer in place for one training run, then go back to eager.

    In place keeps parameter names, saving and LoRA merging unchanged.
    """
    layers = _encoder_layers(model)
    if not (layers and _wants_compile(model, forwards)):
        layers = []
    static = bool(layers) and max_length is not None and max_length <= STATIC_MAX_LEN
    # Every bucket compiles once with and once without grad (the KL reference forward), plus slack.
    buckets = -(-max_length // STATIC_MULTIPLE) if static else 0
    # Unsloth's reentrant offloaded checkpoint gives compiled bf16 layers wrong gradients (cosine 0.68
    # to eager, 0.30 full fine-tune), so compiled runs checkpoint with torch's own.
    swapped = {}

    def restore():
        for layer in layers:
            layer._compiled_call_impl = None
        for layer, func in swapped.items():
            layer._gradient_checkpointing_func = func
        model.__dict__["_unsloth_pad_multiple"] = 0

    with _compilable_attention(model, bool(layers)), _recompile_limit(2 * buckets + 8):
        if layers:
            try:
                torch_checkpoint = _torch_checkpoint()
                if torch_checkpoint is None:
                    raise RuntimeError("torch's own checkpoint is not reachable")
                for layer in layers:
                    func = getattr(layer, "_gradient_checkpointing_func", None)
                    if func is not None and getattr(func, "func", func) is not torch_checkpoint:
                        swapped[layer] = func
                        layer._gradient_checkpointing_func = functools.partial(
                            torch_checkpoint, use_reentrant = False
                        )
                    if static:
                        layer._compiled_call_impl = _training_only(
                            layer, _compile_static(layer._call_impl)
                        )
                    else:
                        layer.compile(dynamic = True)
                model.__dict__["_unsloth_pad_multiple"] = STATIC_MULTIPLE if static else 0
                _warm_up(model, amp_dtype)
            except Exception as error:
                restore()
                torch._dynamo.reset()
                print(
                    f"Unsloth: compiling the Laya encoder failed ({type(error).__name__}), training eagerly."
                )
                layers, swapped = [], {}
        model.__dict__["_unsloth_decision_compiled"] = bool(layers)
        try:
            yield bool(layers)
        finally:
            restore()


def pad_length(model, inputs: dict) -> dict:
    # Pads a training batch to the static bucket; padded positions are masked keys and never
    # gathered, so the loss is unchanged.
    multiple = model.__dict__.get("_unsloth_pad_multiple", 0)
    extra = -inputs["input_ids"].shape[1] % multiple if multiple and model.training else 0
    if extra:
        pad = model.encoder.config.pad_token_id or 0
        inputs = {
            **inputs,
            "input_ids": torch.nn.functional.pad(inputs["input_ids"], (0, extra), value = pad),
            "attention_mask": torch.nn.functional.pad(inputs["attention_mask"], (0, extra)),
        }
    return inputs
