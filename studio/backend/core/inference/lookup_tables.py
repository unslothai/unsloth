# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only embedding tables (per-layer embeddings, Engram) and how llama.cpp places them."""

from __future__ import annotations

from typing import Mapping, Optional

LAZY_READ_TENSORS: dict[str, tuple[str, ...]] = {
    "gemma4": ("per_layer_token_embd.weight",),
    "qwen4exp": ("per_layer_token_embd.weight",),
    "deepseek41": ("per_layer_token_embd.weight",),
}
LAZY_READ_AUTO_MIN_BYTES = 4 * 1024**3
# Archs whose table a build marked by HOST_FORCED_TABLE_MARKER keeps on the host whatever -ot
# says. Studio's prebuilt b11734-mix-c208e5f logs "per_layer_token_embd.weight (105471 MiB
# iq4_nl) must stay in host memory; ignoring placement on CUDA0, using CPU" for the DeepSeek-V4.1
# Engram table; upstream b11870 gathers it in the graph and honours the override.
HOST_FORCED_TABLE_ARCHS = frozenset({"deepseek41"})
HOST_FORCED_TABLE_MARKER = b"must stay in host memory; ignoring placement on"
PER_LAYER_EMBD_PREFIX = "per_layer_token_embd"


def is_per_layer_embedding(name: str) -> bool:
    """Whether ``name`` is a per-layer (PLE / Engram) input embedding table."""
    return str(name).startswith(PER_LAYER_EMBD_PREFIX)


def lazy_read_bytes(
    architecture: Optional[str],
    tensor_bytes: Mapping[str, int],
    mode: str = "auto",
) -> int:
    """Bytes of ``tensor_bytes`` llama.cpp reads lazily under ``mode``."""
    if mode not in ("auto", "on"):
        return 0
    names = LAZY_READ_TENSORS.get(architecture or "", ())
    return sum(
        int(size)
        for name, size in tensor_bytes.items()
        if name in names and (mode == "on" or int(size) > LAZY_READ_AUTO_MIN_BYTES)
    )
