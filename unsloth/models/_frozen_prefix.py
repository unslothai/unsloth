# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Skip the frozen bottom layers under gradient checkpointing.

With adapters only on the upper layers (finetune_last_n_layers, layers_to_transform), the embedding output
is made to require grad so reentrant checkpointing gives the adapters a gradient. That also pulls every
frozen layer below into the graph: each one is checkpointed, replayed and backpropagated for a gradient
nothing reads. When every trainable parameter sits at or above layer k of one decoder stack (plus the
output head), layers 0..k-1 run once without a graph and layer k's input becomes the leaf that requires
grad instead. Loss and gradients are unchanged. UNSLOTH_FROZEN_PREFIX_SKIP=0 turns it off."""

import os
import weakref

import torch
import torch.nn as nn

__all__ = ["install_frozen_prefix_skip", "FROZEN_PREFIX_STATS"]

# Engagement counters (tests read them): forwards that skipped a prefix, and prefix layers run without a graph.
FROZEN_PREFIX_STATS = {"forwards": 0, "layers": 0}

# Modules after the decoder stack whose trainable parameters keep the cut valid.
_HEAD_NAMES = ("lm_head", "score", "classifier")


def _enabled():
    return os.environ.get("UNSLOTH_FROZEN_PREFIX_SKIP", "1") != "0"


def _layer_class():
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception:
        return None
    return GradientCheckpointingLayer


def _decoder_stacks(model):
    """Every ModuleList made only of transformers checkpointing layers (the text stack, vision towers)."""
    layer_class = _layer_class()
    if layer_class is None:
        return []
    stacks = []
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            if all(isinstance(layer, layer_class) for layer in module):
                stacks.append(module)
    return stacks


def _head_parameters(model):
    ids = set()
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in _HEAD_NAMES:
            ids.update(id(p) for p in module.parameters())
    try:
        output = model.get_output_embeddings()
    except Exception:
        output = None
    if output is not None:
        ids.update(id(p) for p in output.parameters())
    # A tied head is the input embedding: training it needs the gradient below the cut.
    try:
        embedding = model.get_input_embeddings()
    except Exception:
        embedding = None
    if embedding is not None:
        ids.difference_update(id(p) for p in embedding.parameters())
    return ids


def _plan(model, stacks):
    """(stack, k) when the first k layers of one stack, and everything feeding them, hold nothing trainable."""
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if not trainable:
        return None
    for stack in stacks:
        k = None
        upper = set()
        for index, layer in enumerate(stack):
            ids = {id(p) for p in layer.parameters() if p.requires_grad}
            if ids and k is None:
                k = index
            if k is not None:
                upper.update(ids)
        if not k:
            continue
        if not trainable.issubset(upper | _head_parameters(model)):
            return None
        first = stack[k]
        if not (getattr(first, "gradient_checkpointing", False) and first.training):
            return None
        if any(getattr(layer, "_compiled_call_impl", None) is not None for layer in stack[:k]):
            return None  # hooks would sit inside a compiled call
        return stack, k
    return None


def _detach(value):
    if isinstance(value, torch.Tensor) and value.requires_grad:
        return value.detach()
    if isinstance(value, tuple) and any(
        isinstance(item, torch.Tensor) and item.requires_grad for item in value
    ):
        return tuple(_detach(item) for item in value)
    return value


def _detach_inputs(module, args, kwargs):
    # Nothing below the cut is trainable, so no input of a frozen prefix layer needs a graph.
    return tuple(_detach(arg) for arg in args), {
        key: _detach(value) for key, value in kwargs.items()
    }


def _leaf(hidden):
    return hidden.detach().requires_grad_(True)


def install_frozen_prefix_skip(model):
    """Idempotent. Hooks the outer model; per training forward it decides again from requires_grad."""
    if model is None or not hasattr(model, "register_forward_pre_hook"):
        return model
    if getattr(model, "_unsloth_frozen_prefix", None) is not None:
        return model
    state = {"stacks": None, "active": None, "disabled": False}
    model_ref = weakref.ref(model)

    def _cut_output(module, args, output):
        state["cut"] = True
        if isinstance(output, torch.Tensor):
            return _leaf(output)
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return type(output)((_leaf(output[0]), *output[1:]))
        state["cut"] = False
        return None

    def _restore():
        active = state["active"]
        state["active"] = None
        if active is None:
            return
        for handle in active["handles"]:
            handle.remove()
        for layer, flag in active["flags"]:
            layer.gradient_checkpointing = flag

    def _pre(module, args, kwargs):
        _restore()
        if (
            state["disabled"]
            or not _enabled()
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return None
        embeds = kwargs.get("inputs_embeds")
        if isinstance(embeds, torch.Tensor) and embeds.requires_grad:
            return None  # the caller wants a gradient at the input
        outer = model_ref()
        if outer is None:
            return None
        if state["stacks"] is None:
            state["stacks"] = _decoder_stacks(outer)
        plan = _plan(outer, state["stacks"])
        if plan is None:
            return None
        stack, k = plan
        flags, handles = [], []
        for layer in stack[:k]:
            flags.append((layer, layer.gradient_checkpointing))
            # Off: the hooks then run outside the checkpoint, and the layer runs once.
            layer.gradient_checkpointing = False
            handles.append(layer.register_forward_pre_hook(_detach_inputs, with_kwargs = True))
        handles.append(stack[k - 1].register_forward_hook(_cut_output))
        state["cut"] = False
        state["active"] = {"flags": flags, "handles": handles, "k": k}
        return None

    def _post(module, args, output):
        active = state["active"]
        _restore()
        if active is None or output is None:
            return None  # also the always_call run after a forward that raised
        if not state["cut"]:
            # Layer k would have had an input without grad: its adapters got no gradient this step.
            state["disabled"] = True
            raise RuntimeError(
                "Unsloth: the frozen-prefix skip could not mark the first trainable layer's input "
                "(unexpected layer output). Set UNSLOTH_FROZEN_PREFIX_SKIP=0 and rerun."
            )
        FROZEN_PREFIX_STATS["forwards"] += 1
        FROZEN_PREFIX_STATS["layers"] += active["k"]
        return None

    try:
        handles = (
            model.register_forward_pre_hook(_pre, with_kwargs = True, prepend = True),
            model.register_forward_hook(_post, always_call = True),
        )
    except Exception:
        return model
    try:
        model._unsloth_frozen_prefix = handles
    except Exception:
        for handle in handles:
            handle.remove()
    return model
