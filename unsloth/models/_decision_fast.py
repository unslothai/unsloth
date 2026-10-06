# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Laya's encoder is launch bound at decision sizes, so long runs compile each layer.
# UNSLOTH_DECISION_COMPILE=0 or 1 turns it off or forces it on.

import contextlib
import functools
import importlib.util
import os

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint as _torch_checkpoint

__all__ = ["compiled_encoder"]

# Shorter runs spend more on a cold compile than it saves.
COMPILE_MIN_FORWARDS = 4000


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
    ids = torch.randint(5, min(vocab, 1000), (2, 64), device = device)
    mask = torch.ones_like(ids)
    mask[1, 48:] = 0
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in params]
    training = model.training
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        model.train()
        with torch.random.fork_rng(devices = devices):
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
):
    """Compile each Laya encoder layer in place for one training run, then go back to eager.

    In place keeps parameter names, saving and LoRA merging unchanged.
    """
    layers = _encoder_layers(model)
    if not (layers and _wants_compile(model, forwards)):
        layers = []
    # Unsloth's reentrant offloaded checkpoint gives compiled bf16 layers wrong gradients (cosine 0.68
    # to eager, 0.30 full fine-tune), so compiled runs checkpoint with torch's own.
    swapped = {}
    for layer in layers:
        func = getattr(layer, "_gradient_checkpointing_func", None)
        if func is not None and getattr(func, "func", func) is not _torch_checkpoint:
            swapped[layer] = func
            layer._gradient_checkpointing_func = functools.partial(
                _torch_checkpoint, use_reentrant = False
            )
        layer.compile(dynamic = True)

    def restore():
        for layer in layers:
            layer._compiled_call_impl = None
        for layer, func in swapped.items():
            layer._gradient_checkpointing_func = func

    if layers:
        try:
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
