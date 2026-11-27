# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in DiT block swap for LoRA training on a bf16, frozen fp8 or int8 W8A8 base that does not fit the card.

Frozen weights of some repeated blocks live in pinned host RAM and stream to the GPU one block ahead on a side stream
(unsloth_zoo.block_swap.BlockSwap: the host copy is authoritative, so nothing is ever copied back, and the
checkpoint recompute prefetches in reverse). Swapped blocks are always checkpointed; the checkpointing budget plans
the resident blocks into the room the swap freed.

A frozen fp8 / W8A8 base (``diffusion_frozen_fp8`` / ``diffusion_frozen_int8``) swaps its 1-byte codes: the plan is
sized from the converted bytes, each swapped block is quantized Linear by Linear on the card (the same kernels as a
resident layer, so the codes are identical) and its codes go back to host RAM as frozen parameters, which is what
the swapper streams. Their per-row scales stay on the card. UNSLOTH_DIT_BLOCK_SWAP_LOWP=0 keeps the bf16-only gate.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Union

BLOCK_SWAP_ENV = "UNSLOTH_DIT_BLOCK_SWAP"
LOWP_SWAP_ENV = "UNSLOTH_DIT_BLOCK_SWAP_LOWP"
# Mirrors diffusion_dit_trainer.INT8_TRAIN_ENV (that module imports this one).
_INT8_TRAIN_ENV = "UNSLOTH_DIFFUSION_INT8_TRAIN"
_OFF = ("", "0", "off", "none", "false", "no")
# Bytes a frozen low-precision Linear keeps per weight element: 1-byte codes (the fp32 per-row scale stays resident).
_LOWP_CODE_BYTES = 1
# Marks a FrozenFp8Linear / FrozenInt8Linear whose codes stream through a ring slot (diffusion_frozen_fp8 reads it).
SWAPPED_WEIGHT_ATTR = "_studio_swapped_weight"
# Headroom kept free besides the activation estimate: CUDA context, cuBLAS / compile workspaces, fragmentation. The
# fraction is of the VRAM this run can use (free at install), not of the card: another process's share of a 96 GB
# card is not this run's fragmentation, and reserving 12% of it swapped 32 of 34 Z-Image blocks with 16 GB free.
_RESERVE_MIN_BYTES = 3 << 30
_RESERVE_FRACTION = 0.12
# Bytes per (token x hidden) element kept per block under full checkpointing, plus the one block being recomputed.
_SAVED_PER_BLOCK = 2 * 2
_RECOMPUTE_FACTOR = 48

# The swappers installed by the last run, for counters (copies, stall) in benches and tests.
LAST_SWAPPERS: list = []


def block_swap_request(cfg: Any) -> Optional[Union[int, str]]:
    """None (off), "auto", or a block count. An unset cfg.block_swap (None or "") takes UNSLOTH_DIT_BLOCK_SWAP; an
    explicit "off" or 0 is off even when the env asks for a swap."""
    raw = getattr(cfg, "block_swap", None)
    if raw is None or str(raw).strip() == "":
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


def lowp_swap_enabled() -> bool:
    """Block swap may stream a frozen fp8 / int8 W8A8 base; UNSLOTH_DIT_BLOCK_SWAP_LOWP=0 keeps it bf16 only."""
    return os.environ.get(LOWP_SWAP_ENV, "1").strip().lower() not in _OFF


# UNSLOTH_DIFFUSION_INT8_TRAIN values that keep the torchao weight-only int8 base (the pre-W8A8 default).
INT8_WEIGHT_ONLY_VALUES = (
    "weight-only",
    "weight_only",
    "weightonly",
    "wo",
    "torchao",
    "0",
    "off",
    "false",
    "no",
)


def int8_w8a8_env() -> Optional[bool]:
    """The explicit UNSLOTH_DIFFUSION_INT8_TRAIN choice: False for weight-only, True for anything else set, None
    when unset."""
    raw = os.environ.get(_INT8_TRAIN_ENV, "").strip().lower()
    if not raw:
        return None
    return raw not in INT8_WEIGHT_ONLY_VALUES


ROCM_W8A8_TRAIN_ENV = "UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN"


def rocm_w8a8_train_opt_in(mode: Optional[str], cfg: Any = None) -> bool:
    """Opt-in: int8 trains the W8A8 base (torch._int_mm) on ROCm, never weight-only. Off until measured faster there."""
    if (mode or "").strip().lower() != "int8":
        return False
    if os.environ.get(ROCM_W8A8_TRAIN_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    return int8_w8a8_for_run(cfg)


def int8_w8a8_requested() -> bool:
    """int8 trains on the frozen W8A8 base by default (not torchao weight-only, whose tensor subclass cannot stream
    and cannot trace); UNSLOTH_DIFFUSION_INT8_TRAIN=weight-only keeps the weight-only base."""
    explicit = int8_w8a8_env()
    return True if explicit is None else explicit


def int8_w8a8_for_run(cfg: Any = None) -> bool:
    """Whether this run's int8 base is W8A8: the explicit UNSLOTH_DIFFUSION_INT8_TRAIN choice, else the kernel a
    resumed int8 bundle trained on (weight-only for one without the record), else W8A8."""
    explicit = int8_w8a8_env()
    if explicit is not None:
        return explicit
    resume = getattr(cfg, "resume_from_checkpoint", None) if cfg is not None else None
    if resume:
        from core.training.diffusion_checkpoint import recorded_int8_resume_kernel
        kernel = recorded_int8_resume_kernel(resume)
        if kernel is not None:
            return kernel == "w8a8"
    return True


def swap_lowp_mode(base_precision: str, cfg: Any = None) -> Optional[str]:
    """The frozen low-precision base block swap streams for this precision ("fp8" / "int8_w8a8"), or None."""
    if not lowp_swap_enabled():
        return None
    if base_precision == "fp8":
        return "fp8"
    if base_precision == "int8" and int8_w8a8_for_run(cfg):
        return "int8_w8a8"
    return None


def block_swap_explicit_off(cfg: Any) -> bool:
    """cfg.block_swap says "off" or 0 outright (not unset): no swap at all, not even auto's int8-band swap."""
    raw = getattr(cfg, "block_swap", None)
    return raw is not None and str(raw).strip() != "" and str(raw).strip().lower() in _OFF


def block_swap_error(
    request,
    base_precision: str,
    device: str,
    gradient_checkpointing: bool,
    integrated: bool = False,
    cfg: Any = None,
) -> Optional[str]:
    """Why this run cannot swap blocks, or None. ``cfg`` lets a resume keep the int8 kernel its bundle recorded."""
    if request is None:
        return None
    if device != "cuda":
        return "block_swap needs a CUDA GPU."
    if base_precision != "bf16" and swap_lowp_mode(base_precision, cfg) is None:
        if base_precision == "int8" and lowp_swap_enabled():
            return (
                "block_swap streams int8 only as the frozen W8A8 base; the torchao weight-only int8 base "
                "(UNSLOTH_DIFFUSION_INT8_TRAIN=weight-only) cannot stream. Unset it, use base_precision='bf16' or "
                "'fp8', or turn block_swap off."
            )
        allowed = "'bf16'" if not lowp_swap_enabled() else "'bf16', 'fp8' or int8 W8A8"
        return (
            f"block_swap streams a {allowed} base; base_precision={base_precision!r} cannot stream. "
            f"Use base_precision={allowed} with block_swap, or turn block_swap off."
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
        cfg = cfg,
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


def _lowp_weight_bytes(linear) -> int:
    return linear.weight.numel() * _LOWP_CODE_BYTES


def _lowp_scale_bytes(linear) -> int:
    return int(linear.weight.shape[0]) * 4


def _frozen_bytes(module, targets = frozenset()) -> int:
    """Host bytes swapping ``module`` frees: its frozen parameters, plus the codes of an already converted frozen
    fp8 / W8A8 Linear (a buffer until it is swapped). A Linear in ``targets`` (ids) counts as its converted codes."""
    total = 0
    for m in module.modules():
        conv = id(m) in targets
        for n, p in m._parameters.items():
            if p is None or p.requires_grad or "lora_" in n:
                continue
            total += (
                _lowp_weight_bytes(m) if conv and n == "weight" else p.numel() * p.element_size()
            )
        if getattr(m, "_studio_lowp", None) in ("fp8", "int8_w8a8"):
            w = m._buffers.get("weight")
            if w is not None:
                total += w.numel() * w.element_size()
    return total


def _model_bytes(transformer, targets = frozenset()) -> int:
    """Device bytes of the whole base once ``targets`` (Linear ids) are converted: codes plus resident scales."""
    total = 0
    for m in transformer.modules():
        conv = id(m) in targets
        for n, t in (*m._parameters.items(), *m._buffers.items()):
            if t is None:
                continue
            if conv and n == "weight":
                total += _lowp_weight_bytes(m) + _lowp_scale_bytes(m)
            else:
                total += t.numel() * t.element_size()
    return total


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
    """VRAM kept free for one fully checkpointed step: saved block inputs, one block's recompute, fixed headroom.
    ``total_vram`` is the VRAM the run can use (install passes min(free, total))."""
    elems = max(1, tokens) * max(1, hidden) * max(1, batch)
    act = elems * (_SAVED_PER_BLOCK * max(1, n_blocks) + 2 * _RECOMPUTE_FACTOR)
    return int(max(_RESERVE_MIN_BYTES, _RESERVE_FRACTION * total_vram) + act)


def plan_swap(
    block_bytes: list,
    request,
    need_bytes: int = 0,
    depth: int = 2,
) -> list:
    """Returns [[indices] per list]; N is spread by list length, auto takes the smallest equal
    fraction of every list whose freed bytes, net of the depth + 1 slot pool, cover need_bytes."""
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


def _place_rest(
    transformer,
    swapped,
    device,
    move = None,
) -> None:
    """Everything except the swapped blocks' frozen weights to the card (their LoRA params BlockSwap moves). A frozen
    fp8 / W8A8 block's codes are such weights once host_lowp_weight made them parameters; its scales are buffers and
    go to the card. ``move(tensor, device)`` (tests) defaults to ``tensor.to(device)``."""
    import torch

    if move is None:
        move = lambda t, d: t.to(d)  # noqa: E731

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
                    p.data = move(p.data, device)
            for name, b in module._buffers.items():
                if b is not None and b.device.type != "cuda":
                    module._buffers[name] = move(b, device)


def host_lowp_weight(module) -> bool:
    """A converted FrozenFp8Linear / FrozenInt8Linear in a swapped block: its codes move to host RAM as a frozen
    parameter (what the swapper streams; values unchanged) and the module stops using the data_ptr-keyed fp8
    backward cache. The per-row scale buffer stays where it is. Returns whether it changed anything."""
    import torch

    w = module._buffers.get("weight")
    if getattr(module, "_studio_lowp", None) not in ("fp8", "int8_w8a8") or w is None:
        return False
    del module._buffers["weight"]
    module.weight = torch.nn.Parameter(w.to("cpu"), requires_grad = False)
    if module.bias is not None and module.bias.device.type != "cpu":
        module.bias.data = module.bias.data.to("cpu")
    setattr(module, SWAPPED_WEIGHT_ATTR, True)
    return True


def _lowp_convert_blocks(transformer, lowp, swapped, dev) -> tuple[int, int]:
    """Convert ``lowp["targets"]`` (fqns): resident ones streamed onto ``dev`` as before, swapped ones too (identical
    codes) and then handed back to host RAM one Linear at a time, so the bf16 base is never whole on the card."""
    swapped_ids = {id(b) for b in swapped}
    in_swapped = set()
    for fqn in lowp["targets"]:
        parts = fqn.split(".")
        for k in range(1, len(parts)):
            if id(transformer.get_submodule(".".join(parts[:k]))) in swapped_ids:
                in_swapped.add(fqn)
                break
    convert = lowp["convert"]
    done = failed = 0
    for fqn in sorted(in_swapped):
        d, f = convert(transformer, {fqn}, dev)
        done, failed = done + d, failed + f
        parent_name, _, child = fqn.rpartition(".")
        mod = getattr(transformer.get_submodule(parent_name) if parent_name else transformer, child)
        host_lowp_weight(mod)
    d, f = convert(transformer, set(lowp["targets"]) - in_swapped, dev)
    if lowp.get("finish") is not None:
        lowp["finish"](transformer)
    return done + d, failed + f


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
    lowp: Optional[dict] = None,
) -> dict:
    """The host-loaded transformer (LoRA attached) goes to the card except the planned blocks, which BlockSwap streams.
    ``lowp`` = {"mode": "fp8" | "int8_w8a8", "targets": fqns, "convert": fn(model, names, device) -> (done, failed)}
    converts the frozen base first (sized from the converted bytes, see host_lowp_weight). Returns a summary; the
    swappers are kept in LAST_SWAPPERS and on transformer._unsloth_block_swaps."""
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
    targets = frozenset()
    if lowp is not None:
        targets = frozenset(id(transformer.get_submodule(n)) for n in lowp["targets"])
    block_bytes = [[_frozen_bytes(b, targets) for b in mod] for _, mod in lists]
    n_blocks = sum(len(b) for b in block_bytes)
    hidden = (
        int(getattr(getattr(transformer, "config", None), "num_attention_heads", 0) or 0)
        * int(getattr(getattr(transformer, "config", None), "attention_head_dim", 0) or 0)
        or 3072
    )
    # Optimizer + grads (AdamW: fp32 param, grad, two moments) for every trainable param.
    trainable = sum(p.numel() * 16 for p in transformer.parameters() if p.requires_grad)
    model_bytes = _model_bytes(transformer, targets)
    reserve = activation_reserve_bytes(
        min(total_bytes, free_bytes), tokens, hidden, n_blocks, batch
    )
    need = model_bytes + trainable + reserve - free_bytes
    depth_plan = 2 if prefetch_depth == "auto" else int(prefetch_depth)
    plan = plan_swap(block_bytes, request, need, depth_plan)
    swapped = [mod[i] for (_, mod), idx in zip(lists, plan) for i in idx]
    converted = failed = 0
    if lowp is not None:
        converted, failed = _lowp_convert_blocks(transformer, lowp, swapped, dev)
        if not converted:
            raise ValueError(f"block_swap: no eligible linear layer converted to {lowp['mode']}.")
        if failed:
            # A failed conversion stays bf16 (2 bytes/weight, not 1-byte codes): re-size and swap extra blocks.
            block_bytes = [[_frozen_bytes(b) for b in mod] for _, mod in lists]
            model_bytes = _model_bytes(transformer)
            need = model_bytes + trainable + reserve - free_bytes
            if request == "auto":
                plan = [
                    sorted(set(old) | set(new))
                    for old, new in zip(plan, plan_swap(block_bytes, request, need, depth_plan))
                ]
                already = {id(b) for b in swapped}
                swapped = [mod[i] for (_, mod), idx in zip(lists, plan) for i in idx]
                for block in swapped:
                    if id(block) not in already:
                        for m in block.modules():
                            host_lowp_weight(m)
    _place_rest(transformer, swapped, dev)
    from core.training.diffusion_gc_budget import SWAPPED_ATTR

    for block in swapped:
        setattr(block, SWAPPED_ATTR, True)
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
        for block in swapped:
            block.__dict__.pop(SWAPPED_ATTR, None)
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
        "lowp": None if lowp is None else lowp["mode"],
        "lowp_converted": converted,
        "lowp_failed": failed,
    }


def compile_swapped_blocks(
    transformer,
    fullgraph: bool,
    dynamic: bool = True,
    options: Optional[dict] = None,
) -> int:
    """Regional compile that leaves the swap hooks outside the graph: each repeated block's forward is compiled,
    not its _call_impl (whose pre / post hooks would be a graph break, an error under fullgraph)."""
    import torch

    n = 0
    for _, mod in swap_block_lists(transformer):
        for block in mod:
            block.forward = torch.compile(
                block.forward, fullgraph = fullgraph, dynamic = dynamic, options = options
            )
            n += 1
    return n


def reset_block_swaps(transformer) -> int:
    """After an out-of-memory error mid-step: give every slot back and arm the first blocks, so the rerun starts from
    a clean pool (no block left resident or marked as held by the failed graph). Returns the swappers reset."""
    n = 0
    for s in getattr(transformer, "_unsloth_block_swaps", None) or ():
        try:
            s.reset()
            n += 1
        except Exception:  # noqa: BLE001 -- the next forward's steal path still recovers the slots
            pass
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
