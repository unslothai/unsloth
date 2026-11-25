# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a text encoder's input embedding table on the checkpoint's file mapping and gather the prompt's rows on the
host, instead of owning (or pinning) a private copy of the whole table and uploading it to the card per encode.

The rows are the same bytes either way (a gather is a copy), so outputs are bit-identical; only the rows a prompt
names are paged in, and the clean file pages stay reclaimable. Linux only: the check that the table really is a
file view reads ``/proc/self/maps``, and a mapping elsewhere (Windows in particular) would also hold the Hub blob
open for the life of the process. UNSLOTH_DIFFUSION_TE_EMBED_HOST=0 turns it off."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

EMBED_HOST_ENV = "UNSLOTH_DIFFUSION_TE_EMBED_HOST"
ATTR = "_unsloth_te_embed_host"
# The compute device the gathered rows go to, set by the caller per encode.
TARGET_ATTR = "_unsloth_te_embed_target"


def enabled() -> bool:
    return (os.environ.get(EMBED_HOST_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def file_mapped(tensor: Any) -> bool:
    """Whether a contiguous CPU tensor's bytes lie inside one file-backed mapping (Linux ``/proc/self/maps``)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.is_pinned():
            return False
        start = int(tensor.data_ptr())
        end = start + tensor.numel() * tensor.element_size()
        if end <= start:
            return False
        with open("/proc/self/maps", "r", encoding = "utf-8", errors = "replace") as fh:
            for line in fh:
                fields = line.split(None, 5)
                if len(fields) < 6:
                    continue
                lo, _, hi = fields[0].partition("-")
                if int(lo, 16) <= start and end <= int(hi, 16):
                    return fields[4] != "0" and fields[5].strip().startswith("/")
    except (OSError, ValueError, RuntimeError, AttributeError):
        return False
    return False


def is_host_embedding(module: Any) -> bool:
    return bool(getattr(module, "__dict__", {}).get(ATTR))


def keep_on_host(model: Any) -> int:
    """Pin ``model``'s input embedding to its file mapping; returns the table's bytes, 0 when not applicable.

    The ids go to wherever the table is at call time and the rows to the encode device the caller last named with
    ``set_device``. That device is never inferred from another parameter: a layer-streamed encode leaves every decoder
    layer on the host until its own forward starts. Callers that copy or move the encoder's host tensors skip modules
    for which ``is_host_embedding`` holds."""
    if not enabled() or model is None:
        return 0
    try:
        import torch

        emb = model.get_input_embeddings()
        if not isinstance(emb, torch.nn.Embedding):
            return 0
        if is_host_embedding(emb):
            return int(emb.weight.numel() * emb.weight.element_size())
        weight = emb.weight
        if (
            type(weight) not in (torch.nn.Parameter, torch.Tensor)
            or type(weight.data) is not torch.Tensor
        ):
            return 0  # a quantized / wrapped table: not a plain file view
        if not file_mapped(weight.data):
            return 0

        def before(module: Any, args: Any) -> Any:
            home = module.weight.device
            return tuple(a.to(home) if torch.is_tensor(a) and a.device != home else a for a in args)

        def after(module: Any, _args: Any, out: Any) -> Any:
            target = module.__dict__.get(TARGET_ATTR)
            return out if target is None or out.device == target else out.to(target)

        handles = [emb.register_forward_pre_hook(before), emb.register_forward_hook(after)]
        emb.__dict__[ATTR] = handles
        return int(weight.numel() * weight.element_size())
    except Exception:  # noqa: BLE001 - an optimisation: the encoder keeps its usual placement
        return 0


def set_device(model: Any, device: Any) -> bool:
    """Name the device the next encode computes on (rows are sent there); whether the host table is engaged."""
    try:
        import torch

        emb = model.get_input_embeddings()
        if not is_host_embedding(emb):
            return False
        emb.__dict__[TARGET_ATTR] = torch.device(device)
        return True
    except Exception:  # noqa: BLE001
        return False


def release(model: Any) -> bool:
    """Undo ``keep_on_host`` (the table stays where it is); whether it was on."""
    try:
        emb = model.get_input_embeddings()
    except Exception:  # noqa: BLE001
        return False
    getattr(emb, "__dict__", {}).pop(TARGET_ATTR, None)
    handles = getattr(emb, "__dict__", {}).pop(ATTR, None)
    for handle in handles or ():
        handle.remove()
    return handles is not None
