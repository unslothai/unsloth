# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Laya's ModernBERT encoder is launch bound at the decision trainer's sizes (8 rows of ~300 tokens), so
# long runs compile each encoder layer. UNSLOTH_DECISION_COMPILE=0 or 1 turns it off or on.

import contextlib
import importlib.util
import os

import torch.nn as nn

__all__ = ["compiled_encoder"]

# Forward passes a run needs before compiling pays for itself even from a cold cache: 38 ms (LoRA) and
# 16 ms (full) saved per micro-batch of 8 on an L4 against 55-88 s of compiling (22-34 s warm).
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


@contextlib.contextmanager
def compiled_encoder(model, forwards: int):
    """Compile each Laya encoder layer for one training run, and run eagerly again afterwards.

    In place (``nn.Module.compile``), so parameter names, saving and LoRA merging see the same modules;
    calibration and evaluation after training run eagerly instead of compiling a no-grad graph.
    """
    layers = _encoder_layers(model)
    if not (layers and _wants_compile(model, forwards)):
        layers = []
    for layer in layers:
        layer.compile(dynamic = True)
    try:
        yield bool(layers)
    finally:
        for layer in layers:
            layer._compiled_call_impl = None
