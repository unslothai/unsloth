# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lookup-only embedding tables (per-layer embeddings, Engram) and how llama.cpp places them.

These tables are row-gathered (GGML_OP_GET_ROWS) and llama.cpp pins them to the host as
input-layer tensors, so they never take VRAM. Some architectures also flag theirs
TENSOR_READ_LAZY: under ``--lazy-mode auto`` (the default) a flagged tensor over 4 GiB
stays mmap'd from disk under every load mode and only the rows a token touches are paged
in. Pure data and arithmetic, shared by the launch fit, the offload layout and the
model listing, so the three cannot disagree on which tensors qualify.
"""

from __future__ import annotations

from typing import Mapping, Optional

# TENSOR_READ_LAZY tensors (src/models/gemma4.cpp, qwen4exp.cpp, deepseek41.cpp): under
# --lazy-mode auto one over 4 GiB stays mmap'd under every load mode (lazy_read::add),
# needing no pinned RAM or VRAM. gemma3n stores the same table but does not flag it.
LAZY_READ_TENSORS: dict[str, tuple[str, ...]] = {
    "gemma4": ("per_layer_token_embd.weight",),
    "qwen4exp": ("per_layer_token_embd.weight",),
    "deepseek41": ("per_layer_token_embd.weight",),
}
LAZY_READ_AUTO_MIN_BYTES = 4 * 1024**3
# Architectures whose loader forces the lazy table onto a host buffer type even when an
# override or the fitter names a device for it (llama-model-loader.cpp, deepseek41).
HOST_FORCED_TABLE_ARCHS = frozenset({"deepseek41"})
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
