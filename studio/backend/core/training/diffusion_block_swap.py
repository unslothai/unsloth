# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in DiT block swap for LoRA training on a bf16 base that does not fit the card.

Frozen weights of some repeated blocks live in pinned host RAM and stream to the GPU one block ahead on a side stream
(unsloth_zoo.block_swap.BlockSwap: the host copy is authoritative, so nothing is ever copied back, and the
checkpoint recompute prefetches in reverse). Every block must be checkpointed, so the checkpointing budget is off.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Union

BLOCK_SWAP_ENV = "UNSLOTH_DIT_BLOCK_SWAP"
_OFF = ("", "0", "off", "none", "false", "no")
# Fixed headroom kept free besides the activation estimate: CUDA context, cuBLAS / compile workspaces, fragmentation.
_RESERVE_MIN_BYTES = 3 << 30
_RESERVE_FRACTION = 0.12
# Bytes per (token x hidden) element kept per block under full checkpointing, plus the one block being recomputed.
_SAVED_PER_BLOCK = 2 * 2
_RECOMPUTE_FACTOR = 48

# The swappers installed by the last run, for counters (copies, stall) in benches and tests.
LAST_SWAPPERS: list = []


def block_swap_request(cfg: Any) -> Optional[Union[int, str]]:
    """None (off, the default), "auto", or a block count, from cfg.block_swap or UNSLOTH_DIT_BLOCK_SWAP."""
    raw = getattr(cfg, "block_swap", None)
    if raw is None or str(raw).strip().lower() in _OFF:
        raw = os.environ.get(BLOCK_SWAP_ENV, "")
    raw = str(raw).strip().lower()
    if raw in _OFF:
        return None
    if raw == "auto":
        return "auto"
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"block_swap={raw!r}: use 'off', 'auto' or a number of blocks.") from None
    return n if n > 0 else None


def block_swap_error(
    request,
    base_precision: str,
    device: str,
    gradient_checkpointing: bool,
    integrated: bool = False,
) -> Optional[str]:
    """Why this run cannot swap blocks, or None."""
    if request is None:
        return None
    if device != "cuda":
        return "block_swap needs a CUDA GPU."
    if base_precision != "bf16":
        return (
            f"block_swap streams a dense bf16 base; base_precision={base_precision!r} already fits it on the card. "
            "Use base_precision='bf16' with block_swap, or turn block_swap off."
        )
    if not gradient_checkpointing:
        return "block_swap needs gradient_checkpointing: without it every swapped block stays on the card until backward."
    if integrated:
        return "block_swap has nothing to gain on an integrated GPU, whose VRAM is host RAM; turn block_swap off."
    return None


def _cuda_device_facts() -> tuple[str, bool]:
    """("cuda" | "cpu", integrated) for this host. Never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", False
    except Exception:  # noqa: BLE001 -- no torch: no CUDA
        return "cpu", False
    try:
        from core.inference.host_pinning import device_is_integrated
        return "cuda", bool(device_is_integrated("cuda"))
    except Exception:  # noqa: BLE001 -- unclassifiable: discrete
        return "cuda", False


def block_swap_start_error(cfg: Any, device_facts = None) -> Optional[str]:
    """Why the start route must refuse the REQUEST's block_swap (not the env var) before evicting residents, or None.
    Same gates the DiT trainer applies after the load, plus the families whose trainer has no swap path."""
    raw = getattr(cfg, "block_swap", None)
    raw = "" if raw is None else str(raw).strip().lower()
    if raw in _OFF:
        return None
    if raw != "auto":
        try:
            if int(raw) <= 0:
                return None
        except ValueError:
            return f"block_swap={raw!r}: use 'off', 'auto' or a number of blocks."
    from core.training.diffusion_train_common import _DIT_TRAIN_FAMILIES, dit_fp16_fallback

    family = str(getattr(cfg, "resolved_family", "") or "").strip().lower()
    if family not in _DIT_TRAIN_FAMILIES:
        return (
            f"block_swap is not available for the {family or 'unknown'} family: only the DiT trainer "
            f"({', '.join(sorted(_DIT_TRAIN_FAMILIES))}) streams blocks from host RAM. Turn block_swap off."
        )
    if dit_fp16_fallback(family):
        return "block_swap needs a GPU with bf16: this one trains the nf4 fp16 fallback. Turn block_swap off."
    device, integrated = device_facts if device_facts is not None else _cuda_device_facts()
    return block_swap_error(
        raw,
        str(getattr(cfg, "base_precision", "") or "").strip().lower(),
        device,
        bool(getattr(cfg, "gradient_checkpointing", True)),
        integrated = integrated,
    )


def swap_block_lists(transformer) -> list:
    """(name, ModuleList) for every list of the model's repeated blocks (FLUX: double + single stream)."""
    import torch

    names = set(getattr(transformer, "_repeated_blocks", None) or ())
    out = []
    for name, module in transformer.named_modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and len(module) >= 2
            and all(type(b).__name__ in names for b in module)
        ):
            out.append((name, module))
    return out


def _frozen_bytes(module) -> int:
    return sum(
        p.numel() * p.element_size()
        for n, p in module.named_parameters()
        if not p.requires_grad and "lora_" not in n
    )


def _spread(total: int, n: int) -> list:
    """n evenly spaced indices ending at the last block (as zoo swap_indices "spread")."""
    n = max(0, min(int(n), total))
    return [(k + 1) * total // n - 1 for k in range(n)]


def activation_reserve_bytes(
    total_vram: int,
    tokens: int,
    hidden: int,
    n_blocks: int,
    batch: int = 1,
) -> int:
    """VRAM kept free for one fully checkpointed step: saved block inputs, one block's recompute, fixed headroom."""
    elems = max(1, tokens) * max(1, hidden) * max(1, batch)
    act = elems * (_SAVED_PER_BLOCK * max(1, n_blocks) + 2 * _RECOMPUTE_FACTOR)
    return int(max(_RESERVE_MIN_BYTES, _RESERVE_FRACTION * total_vram) + act)


def plan_swap(
    block_bytes: list,
    request,
    need_bytes: int = 0,
    depth: int = 2,
) -> list:
    """Blocks to swap per list. block_bytes = [[bytes per block] per list]; request = N (spread over the lists by
    length) or "auto" (the smallest equal fraction of every list whose freed bytes, net of the depth + 1 slot pool,
    cover need_bytes). Returns [[indices] per list]."""
    lens = [len(b) for b in block_bytes]
    total = sum(lens)
    if not total:
        return [[] for _ in lens]
    if request != "auto":
        n = min(int(request), total)
        counts = [int(round(n * k / total)) for k in lens]
        # Rounding drift lands on the longest lists.
        order = sorted(range(len(lens)), key = lambda i: -lens[i])
        j = 0
        while sum(counts) != n and j < 4 * len(lens):
            i = order[j % len(order)]
            if sum(counts) < n and counts[i] < lens[i]:
                counts[i] += 1
            elif sum(counts) > n and counts[i] > 0:
                counts[i] -= 1
            j += 1
        return [_spread(k, c) for k, c in zip(lens, counts)]
    if need_bytes <= 0:
        return [[] for _ in lens]
    steps = max(lens)
    for s in range(1, steps + 1):
        plan = [_spread(k, math.ceil(s * k / steps)) for k in lens]
        freed = 0
        for sizes, idx in zip(block_bytes, plan):
            if idx:
                picked = [sizes[i] for i in idx]
                freed += sum(picked) - max(picked) * min(depth + 1, len(picked))
        if freed >= need_bytes:
            return plan
    raise ValueError(
        f"block_swap cannot free {need_bytes / 2**30:.1f} GiB even with every block in host RAM; "
        "use base_precision='int8' or 'nf4', or a lower resolution."
    )


def _place_rest(transformer, swapped, device) -> None:
    """Everything except the swapped blocks' frozen weights to the card (their LoRA params BlockSwap moves)."""
    import torch

    keep = {
        id(p)
        for b in swapped
        for n, p in b.named_parameters()
        if not p.requires_grad and "lora_" not in n
    }
    with torch.no_grad():
        for module in transformer.modules():
            for name, p in module._parameters.items():
                if p is not None and id(p) not in keep and p.device.type != "cuda":
                    p.data = p.data.to(device)
            for name, b in module._buffers.items():
                if b is not None and b.device.type != "cuda":
                    module._buffers[name] = b.to(device)


def install_dit_block_swap(
    transformer,
    device: str,
    request,
    *,
    tokens: int,
    batch: int = 1,
    free_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
    prefetch_depth: Union[int, str] = "auto",
) -> dict:
    """The host-loaded transformer (LoRA attached) goes to the card except the planned blocks, which BlockSwap streams.
    Returns a summary; the swappers are kept in LAST_SWAPPERS and on transformer._unsloth_block_swaps."""
    import torch

    try:
        from unsloth_zoo.block_swap import BlockSwap
    except ImportError as exc:
        raise ValueError(
            f"block_swap needs unsloth_zoo with block_swap ({exc}); update unsloth_zoo."
        ) from exc
    lists = swap_block_lists(transformer)
    if not lists:
        raise ValueError("block_swap: this model has no repeated block list to swap.")
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if free_bytes is None or total_bytes is None:
        free, total = torch.cuda.mem_get_info(dev)
        free_bytes = free if free_bytes is None else free_bytes
        total_bytes = total if total_bytes is None else total_bytes
    block_bytes = [[_frozen_bytes(b) for b in mod] for _, mod in lists]
    n_blocks = sum(len(b) for b in block_bytes)
    hidden = (
        int(getattr(getattr(transformer, "config", None), "num_attention_heads", 0) or 0)
        * int(getattr(getattr(transformer, "config", None), "attention_head_dim", 0) or 0)
        or 3072
    )
    # Optimizer + grads (AdamW: fp32 param, grad, two moments) for every trainable param.
    trainable = sum(p.numel() * 16 for p in transformer.parameters() if p.requires_grad)
    model_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters()) + sum(
        b.numel() * b.element_size() for b in transformer.buffers()
    )
    reserve = activation_reserve_bytes(total_bytes, tokens, hidden, n_blocks, batch)
    need = model_bytes + trainable + reserve - free_bytes
    depth_plan = 2 if prefetch_depth == "auto" else int(prefetch_depth)
    plan = plan_swap(block_bytes, request, need, depth_plan)
    swapped = [mod[i] for (_, mod), idx in zip(lists, plan) for i in idx]
    _place_rest(transformer, swapped, dev)
    swappers = []
    try:
        for (_, mod), idx in zip(lists, plan):
            if idx:
                swappers.append(
                    BlockSwap(
                        mod, idx, prefetch_depth = prefetch_depth, device = dev, model = transformer
                    )
                )
    except Exception as exc:
        for s in swappers:
            s.remove()
        raise ValueError(f"block_swap could not keep the blocks in host RAM: {exc}") from exc
    LAST_SWAPPERS[:] = swappers
    transformer._unsloth_block_swaps = swappers
    host = sum(s.host_bytes() for s in swappers)
    pinned = sum(getattr(s, "pinned_bytes", 0) for s in swappers)
    return {
        "swapped": {name: idx for (name, _), idx in zip(lists, plan)},
        "swapped_blocks": len(swapped),
        "total_blocks": n_blocks,
        "host_gib": round(host / 2**30, 2),
        "pinned_gib": round(pinned / 2**30, 2),
        "need_gib": round(need / 2**30, 2),
        "reserve_gib": round(reserve / 2**30, 2),
    }


def compile_swapped_blocks(
    transformer,
    fullgraph: bool,
    dynamic: bool = True,
) -> int:
    """Regional compile that leaves the swap hooks outside the graph: each repeated block's forward is compiled,
    not its _call_impl (whose pre / post hooks would be a graph break, an error under fullgraph)."""
    import torch

    n = 0
    for _, mod in swap_block_lists(transformer):
        for block in mod:
            block.forward = torch.compile(block.forward, fullgraph = fullgraph, dynamic = dynamic)
            n += 1
    return n


def swap_stats() -> dict:
    """Summed counters of the installed swappers (copies, copy_bytes, stall_ms, ...)."""
    out: dict = {}
    for s in LAST_SWAPPERS:
        try:
            st = s.stats(reset = False) or {}
            for k, v in st.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = out.get(k, 0) + v
            out.setdefault("depths", []).append(st.get("prefetch_depth"))
        except Exception:  # noqa: BLE001 -- counters are diagnostic only
            pass
    return out
