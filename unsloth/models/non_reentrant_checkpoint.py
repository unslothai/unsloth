# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The process-wide torch.utils.checkpoint.checkpoint Unsloth installs for non-reentrant gradient checkpointing."""

import inspect

import torch


def non_reentrant_checkpoint(original, offload_hooks = None):
    """`original` (torch's checkpoint) forced to use_reentrant = False, entering `offload_hooks(function)`
    around eager calls when given.

    It replaces torch.utils.checkpoint.checkpoint, so it keeps checkpoint's signature: Dynamo splits a traced
    checkpoint call's keyword arguments into the checkpoint's own and the checkpointed function's by
    inspect.signature(torch.utils.checkpoint.checkpoint) (TagActivationCheckpoint.divide_kwargs). A bare
    (function, *args, **kwargs) signature sends use_reentrant to the function: a checkpoint inside a compiled
    region (DeepSeek-V4.1's chunked sparse attention on T4) failed with "got an unexpected keyword argument
    'use_reentrant'".

    While compiling (Dynamo tracing, and AOT autograd re-entering checkpoint for the checkpoint HOP) the call
    goes straight to `original`: the compiled graph does its own recompute, and the offload hooks' Python
    bookkeeping and side-stream copies are not traceable."""

    def _nonre_checkpoint(function, *args, **kwargs):
        kwargs["use_reentrant"] = False
        if offload_hooks is None or torch.compiler.is_compiling():
            return original(function, *args, **kwargs)
        with offload_hooks(function):
            return original(function, *args, **kwargs)

    try:
        _nonre_checkpoint.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    _nonre_checkpoint._unsloth_original = original
    # Already offloads through offload_hooks: transformers' offload = True would nest it.
    _nonre_checkpoint._unsloth_offloads = offload_hooks is not None
    return _nonre_checkpoint
