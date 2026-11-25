# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-process tensor parallelism of a DiT over two cards (opt-in UNSLOTH_DIFFUSION_TP=1).

Megatron split of every block: q/k/v and the MLP input layers by output rows (each card its heads and its share
of the hidden width), attention out-proj and MLP out by input channels, one all-reduce after each. One Python
thread drives both cards in lockstep (kernel launches are async, so the cards run together). The block shards
live outside the module tree, so moving, hooking or counting the transformer never touches them; the transformer
keeps its prologue / epilogue on the first card. Qwen-Image-2.1 only for now.

INT8 / FP8 ``NativeWeightOnlyLinear``: a column split slices codes and per-row scales (bit-equal outputs); a row
split slices codes by whole ConvRot groups and keeps the per-row scales, and each card quantizes its own input
slice, so the int8 result differs slightly from one card's (its own accuracy gate).
"""

from __future__ import annotations

import copy
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

TP_ENV = "UNSLOTH_DIFFUSION_TP"
TP_COMM_ENV = "UNSLOTH_DIFFUSION_TP_COMM"  # auto | nccl | swap | peer
TP_COMPILE_ENV = (
    "UNSLOTH_DIFFUSION_TP_COMPILE"  # 0 keeps the shards eager when the denoiser compiles
)
GRAPH_UNSAFE_ATTR = "_unsloth_cuda_graph_unsafe"
TP_ATTR = "_unsloth_tp2"
_DITS = frozenset({"QwenImage21Transformer2DModel"})
_COLUMN = ("attn.to_q", "attn.to_k", "attn.to_v", "img_mlp.gate_layer", "img_mlp.proj")
_ROW = ("attn.to_out.0", "img_mlp.out")
_COMMS = ("nccl", "swap", "peer")
_FUSED_ATTRS = ("_unsloth_q21_fused_qkv", "_unsloth_q21_dense_gate_up")


def requested() -> bool:
    return (os.environ.get(TP_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _comm_choice() -> str:
    v = (os.environ.get(TP_COMM_ENV) or "auto").strip().lower()
    return v if v in _COMMS else "auto"


def _get(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def _set(root: Any, path: str, value: Any) -> None:
    head, _, leaf = path.rpartition(".")
    parent = _get(root, head) if head else root
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _is_native(lin: Any) -> bool:
    return type(lin).__name__ == "NativeWeightOnlyLinear" and hasattr(lin, "weight_q")


def refusal(transformer: Any, devices: Sequence[Any]) -> Optional[str]:
    """Why TP-2 cannot run this transformer on these cards, or None."""
    import torch

    name = type(transformer).__name__
    if name not in _DITS:
        return f"{name} is not verified for tensor parallelism"
    if len(devices) != 2:
        return "tensor parallelism needs exactly two cards"
    kinds = {getattr(d, "type", None) for d in devices}
    if len(set(str(d) for d in devices)) != 2 and kinds != {"cpu"}:
        return "the two cards must differ"
    if kinds != {"cuda"} and kinds != {"cpu"}:
        return "tensor parallelism runs on CUDA / ROCm cards only"
    if kinds == {"cuda"}:
        caps = {torch.cuda.get_device_capability(d) for d in devices}
        if len(caps) != 1:
            return "the cards differ in compute capability (lockstep runs at the slower card)"
    if getattr(transformer, TP_ATTR, None) is not None:
        return "already tensor parallel"
    if getattr(transformer, "peft_config", None):
        return "LoRA adapters are loaded (not sharded yet)"
    cfg = transformer.config
    heads = int(cfg.num_attention_heads)
    mlp = int(cfg.num_attention_heads * cfg.attention_head_dim * cfg.mlp_ratio)
    if heads % 2 or mlp % 2:
        return f"{heads} heads / {mlp} MLP width do not split in two"
    for blk in transformer.transformer_blocks:
        for path in _COLUMN + _ROW:
            lin = _get(blk, path)
            if _is_native(lin):
                g = int(getattr(lin, "rot_group", 0) or 0)
                if path in _ROW and g and (lin.in_features // 2) % g:
                    return f"{path}: half of {lin.in_features} input channels is not whole {g}-channel ConvRot groups"
            elif not isinstance(lin, torch.nn.Linear):
                return (
                    f"{path} is a {type(lin).__name__} (dense or native int8 / fp8 Linear expected)"
                )
        if getattr(blk, "_diffusers_hook", None) is not None:
            return "the denoiser carries offload hooks (TP replaces offload)"
    return None


def shard_linear(lin: Any, dim: int, index: int, parts: int, device: Any) -> Any:
    """Card ``index`` of ``parts``: dim 0 = output rows (column parallel), dim 1 = input channels (row parallel)."""
    import torch
    from torch import nn

    if _is_native(lin):
        new = copy.copy(lin)
        new._parameters = OrderedDict(lin._parameters)
        new._buffers = OrderedDict(lin._buffers)
        new._modules = OrderedDict(lin._modules)
        new._forward_hooks = OrderedDict()
        new._forward_pre_hooks = OrderedDict()
        new._int8pack_scales = {}
        wq, ws = lin.weight_q, lin.weight_scale
        if dim == 0:
            k = lin.out_features // parts
            sl = slice(index * k, (index + 1) * k)
            new._buffers["weight_q"] = wq[sl].contiguous().to(device)
            new._buffers["weight_scale"] = ws[sl].contiguous().to(device)
            new.out_features = k
            bias = lin.bias[sl] if lin.bias is not None else None
        else:
            k = lin.in_features // parts
            g = int(getattr(lin, "rot_group", 0) or 0)
            if g and k % g:
                raise ValueError(
                    f"row split of {lin.in_features} channels cuts a {g}-channel ConvRot group"
                )
            new._buffers["weight_q"] = wq[:, index * k : (index + 1) * k].contiguous().to(device)
            new._buffers["weight_scale"] = ws.clone().to(device)
            new.in_features = k
            bias = lin.bias if (lin.bias is not None and index == 0) else None  # added once
        for name in list(new._buffers):
            if name not in ("weight_q", "weight_scale") and new._buffers[name] is not None:
                new._buffers[name] = new._buffers[name].to(device)
        new._parameters.pop("bias", None)
        new.__dict__.pop("bias", None)
        new.bias = (
            None if bias is None else nn.Parameter(bias.detach().to(device), requires_grad = False)
        )
        return new
    w = lin.weight.detach()
    if dim == 0:
        k = lin.out_features // parts
        wk = w[index * k : (index + 1) * k]
        bias = lin.bias[index * k : (index + 1) * k] if lin.bias is not None else None
        out = nn.Linear(lin.in_features, k, bias = bias is not None, device = "meta", dtype = w.dtype)
    else:
        k = lin.in_features // parts
        wk = w[:, index * k : (index + 1) * k]
        bias = lin.bias if (lin.bias is not None and index == 0) else None
        out = nn.Linear(k, lin.out_features, bias = bias is not None, device = "meta", dtype = w.dtype)
    out.weight = nn.Parameter(wk.contiguous().to(device), requires_grad = False)
    if bias is not None:
        out.bias = nn.Parameter(bias.detach().contiguous().to(device), requires_grad = False)
    for attr in ("_unsloth_compute_dtype",):
        if hasattr(lin, attr):
            setattr(out, attr, getattr(lin, attr))
    return out


def shard_block(block: Any, index: int, parts: int, device: Any) -> Any:
    """A copy of ``block`` on ``device`` holding card ``index``'s heads and MLP width; big weights are not copied
    whole (the memo shares them, then the split layers are replaced by their slices)."""
    memo: dict = {}
    for path in _COLUMN + _ROW:
        lin = _get(block, path)
        for t in list(lin.parameters(recurse = False)) + list(lin.buffers(recurse = False)):
            memo[id(t)] = t
    # Attention processors hold configuration only: both cards' shards share the block's own (never copied).
    proc = getattr(block.attn, "processor", None)
    if proc is not None:
        memo[id(proc)] = proc
    # Fused whole-width projections (diffusion_qwenimage21_dense_fused / _fused) are left behind: they hold all
    # heads; the shard runs the stock per-layer path, which those installers may fuse again per card.
    stripped = []
    for sub in (block.attn, block.img_mlp):
        for key in _FUSED_ATTRS + ("forward",):
            if key in sub.__dict__:
                stripped.append((sub, key, sub.__dict__.pop(key)))
    try:
        sb = copy.deepcopy(block, memo)
    finally:
        for sub, key, value in stripped:
            sub.__dict__[key] = value
    for sub in sb.modules():
        # a regional compile wraps the ORIGINAL module's call; the shard runs eager (per-card compile: follow-up)
        if sub.__dict__.get("_compiled_call_impl") is not None:
            sub._compiled_call_impl = None
    for path in _COLUMN:
        _set(sb, path, shard_linear(_get(block, path), 0, index, parts, device))
    for path in _ROW:
        _set(sb, path, shard_linear(_get(block, path), 1, index, parts, device))
    sb.attn.heads = block.attn.heads // parts
    sb.attn.inner_dim = block.attn.inner_dim // parts
    return sb.to(device)


# ------------------------------------------------------------------------------------------------- all-reduce
class HostSwap:
    """All-reduce of two equal tensors through pinned host memory (no P2P needed), chunk-pipelined: each card
    copies its partial D2H on one copy stream; the other card copies it H2D on a second stream once that chunk
    landed, so D2H of chunk i+1 overlaps H2D of chunk i. Both cards then add; x0 + x1 == x1 + x0 bit-for-bit."""

    def __init__(
        self,
        devices: Sequence[Any],
        chunks: int = 4,
    ):
        import torch

        self.devices, self.chunks = list(devices), int(chunks)
        self.d2h = [torch.cuda.Stream(d) for d in self.devices]
        self.h2d = [torch.cuda.Stream(d) for d in self.devices]
        self._host: dict = {}
        self._tmp: dict = {}
        self._read_done: list = [None, None]

    def _bufs(self, n: int, dtype: Any) -> tuple:
        import torch

        key = (n, dtype)
        if key not in self._host:
            self._host[key] = [torch.empty(n, dtype = dtype, pin_memory = True) for _ in self.devices]
            self._tmp[key] = [torch.empty(n, dtype = dtype, device = d) for d in self.devices]
        return self._host[key], self._tmp[key]

    def __call__(self, xs: list) -> list:
        import torch

        n, dt = xs[0].numel(), xs[0].dtype
        host, tmp = self._bufs(n, dt)
        flat = [x.view(-1) for x in xs]
        for c, d in enumerate(self.devices):
            ready = torch.cuda.Event()
            ready.record(
                torch.cuda.current_stream(d)
            )  # partial written; previous add (reads tmp[c]) done
            self.d2h[c].wait_event(ready)
            self.h2d[c].wait_event(ready)
            if self._read_done[c] is not None:  # the other card finished reading host[c] last call
                self.d2h[c].wait_event(self._read_done[c])
        step = -(-n // self.chunks)
        landed: list = [[], []]
        for i in range(self.chunks):
            a, b = i * step, min(n, (i + 1) * step)
            if a >= b:
                break
            for c in (0, 1):
                with torch.cuda.stream(self.d2h[c]):
                    host[c][a:b].copy_(flat[c][a:b], non_blocking = True)
                    ev = torch.cuda.Event()
                    ev.record(self.d2h[c])
                    landed[c].append(ev)
            for c in (0, 1):
                self.h2d[c].wait_event(landed[1 - c][-1])
                with torch.cuda.stream(self.h2d[c]):
                    tmp[c][a:b].copy_(host[1 - c][a:b], non_blocking = True)
        for c, d in enumerate(self.devices):
            done = torch.cuda.Event()
            done.record(self.h2d[c])
            self._read_done[1 - c] = done
            cur = torch.cuda.current_stream(d)
            cur.wait_event(done)
            cur.wait_event(landed[c][-1])  # own D2H read flat[c] before the add overwrites it
            with torch.cuda.device(d):
                flat[c].add_(tmp[c][:n])
        return xs


def make_all_reduce(kind: str, devices: Sequence[Any]) -> Callable[[list], list]:
    if all(getattr(d, "type", None) == "cpu" for d in devices):

        def cpu_sum(xs: list) -> list:  # tests: two "cards" on the CPU
            total = xs[0] + xs[1]
            return [total, total.clone()]

        return cpu_sum
    if kind == "nccl":
        import torch.cuda.nccl as nccl
        def nccl_sum(xs: list) -> list:
            xs = [x.contiguous() for x in xs]
            nccl.all_reduce(xs)
            return xs

        return nccl_sum
    if kind == "swap":
        return HostSwap(devices)
    if kind == "peer":

        def peer_sum(xs: list) -> list:
            a = xs[1].to(devices[0], non_blocking = True)
            b = xs[0].to(devices[1], non_blocking = True)
            return [xs[0] + a, xs[1] + b]

        return peer_sum
    raise ValueError(f"unknown all-reduce {kind!r}")


def pick_all_reduce(
    devices: Sequence[Any],
    numel: int,
    dtype: Any,
    logger: Any = None,
) -> tuple:
    """(name, fn, {name: ms}): the fastest working backend for one ``numel`` all-reduce, or the env's pin."""
    import torch

    forced = _comm_choice()
    names = [forced] if forced != "auto" else list(_COMMS)
    timings: dict = {}
    best = None
    for name in names:
        try:
            fn = make_all_reduce(name, devices)
            xs = [torch.ones(numel, dtype = dtype, device = d) * (i + 1) for i, d in enumerate(devices)]
            ys = fn([x.clone() for x in xs])
            for _ in range(2):
                fn([x.clone() for x in xs])
            for d in devices:
                torch.cuda.synchronize(d)
            if not all(bool((y == 3).all()) for y in ys):
                raise RuntimeError("wrong sum")
            t = time.perf_counter()
            for _ in range(5):
                fn(xs)
            for d in devices:
                torch.cuda.synchronize(d)
            timings[name] = round((time.perf_counter() - t) * 200, 3)  # ms per call
            if best is None or timings[name] < timings[best[0]]:
                best = (name, fn)
        except Exception as exc:  # noqa: BLE001 - a backend that fails here is simply not used
            timings[name] = f"failed: {type(exc).__name__}: {exc}"[:200]
    if best is None:
        raise RuntimeError(f"no all-reduce backend works on {list(map(str, devices))}: {timings}")
    if logger is not None:
        logger.info(
            "diffusion.tp2: all-reduce %s (probe ms per %d elements: %s)", best[0], numel, timings
        )
    return best[0], best[1], timings


# ---------------------------------------------------------------------------------------------------- stack
def _move(t: Any, device: Any) -> Any:
    import torch

    if t is None or isinstance(t, (list, tuple, bool, int, float, str, slice)):
        return t
    if torch.is_tensor(t):
        return t if t.device == device else t.to(device, non_blocking = True)
    to = getattr(t, "to", None)  # flex BlockMask
    return to(device) if callable(to) else t


def _attn_half(blk: Any, h: Any, modulation: Any, ttm: Any, **kw: Any) -> tuple:
    """QwenImage21TransformerBlock.forward up to the attention output (a partial sum on a shard)."""
    mod1, mod2 = modulation.chunk(2, dim = -1)
    x, gate1 = blk._modulate(blk.img_norm1(h), mod1, ttm)
    return blk.attn(hidden_states = x, **kw), gate1, mod2


def _mlp_half(blk: Any, h: Any, mod2: Any, ttm: Any) -> tuple:
    x, gate2 = blk._modulate(blk.img_norm2(h), mod2, ttm)
    return blk.img_mlp(x), gate2


class _CardContext:
    def __init__(self, device: Any):
        self.device = device
        self._ctx = None

    def __enter__(self):
        import torch
        if getattr(self.device, "type", None) == "cuda":
            self._ctx = torch.cuda.device(self.device)
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        return False


def _stack_class():
    from torch import nn
    class TPStack(nn.Module):
        """Stands in for ``transformer_blocks`` (a one-entry ModuleList): runs every block on both cards."""

        def __init__(self, shards: list, devices: list, all_reduce: Callable, comm: str):
            super().__init__()
            # Not registered: .to() / offload / parameters() of the transformer must not see the shards.
            object.__setattr__(self, "_shards", shards)
            object.__setattr__(self, "_devices", list(devices))
            object.__setattr__(self, "_all_reduce", all_reduce)
            self.comm = comm
            self.calls = 0
            self.compiled = False
            object.__setattr__(self, "_stage_attn", _attn_half)
            object.__setattr__(self, "_stage_mlp", _mlp_half)
            self._kv = weakref.WeakKeyDictionary()

        def shards(self) -> list:
            return self._shards

        @property
        def attn(self) -> Any:
            """The model's prefill reads ``block.attn.processor`` to pick the mask form; every shard shares it."""
            return self._shards[0][0].attn

        def _caches(self, layer_cache: Any, mode: Optional[str]) -> Optional[list]:
            if layer_cache is None or mode is None:
                return None
            from diffusers.models.transformers.transformer_qwenimage21 import (
                QwenImage21KVLayerCache,
            )

            per = self._kv.get(layer_cache)
            if per is None or mode == "extract":
                per = [[QwenImage21KVLayerCache() for _ in s] for s in self._shards]
                self._kv[layer_cache] = per
            return per

        def compile(self, *args: Any, **kwargs: Any) -> None:  # noqa: A003 - nn.Module.compile's name
            """Regional compile of a TP denoiser: compile the two per-card stages (attention half, MLP half), the
            all-reduce stays eager between them. Reached through ``compile_repeated_blocks`` (the transformer's
            ``_repeated_blocks`` names this class) or at install when the blocks were compiled already."""
            import torch

            if (os.environ.get(TP_COMPILE_ENV) or "").strip().lower() in (
                "0",
                "off",
                "false",
                "no",
            ):
                return
            kw = dict(kwargs)
            object.__setattr__(self, "_stage_attn", torch.compile(_attn_half, *args, **kw))
            object.__setattr__(self, "_stage_mlp", torch.compile(_mlp_half, *args, **kw))
            self.compiled = True

        def forward(
            self,
            hidden_states: Any,
            modulation: Any,
            rotary_emb: Any = None,
            attention_mask: Any = None,
            target_token_mask: Any = None,
            layer_cache: Any = None,
            kv_cache_mode: Optional[str] = None,
            cache_write_slice: Any = None,
            segments: Any = None,
            key_valid: Any = None,
        ) -> Any:
            import torch

            devs, ar = self._devices, self._all_reduce
            home = hidden_states.device
            hs = [_move(hidden_states, d) for d in devs]
            mods = [_move(modulation, d) for d in devs]
            ropes = [_move(rotary_emb, d) for d in devs]
            masks = [_move(attention_mask, d) for d in devs]
            ttms = [_move(target_token_mask, d) for d in devs]
            kvs = [_move(key_valid, d) for d in devs]
            caches = self._caches(layer_cache, kv_cache_mode)
            for i in range(len(self._shards[0])):
                parts, gates, mod2s = [], [], []
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        a, g1, m2 = self._stage_attn(
                            self._shards[c][i],
                            hs[c],
                            mods[c],
                            ttms[c],
                            attention_mask = masks[c],
                            rotary_emb = ropes[c],
                            layer_cache = None if caches is None else caches[c][i],
                            kv_cache_mode = kv_cache_mode,
                            cache_write_slice = cache_write_slice,
                            segments = segments,
                            key_valid = kvs[c],
                        )
                    parts.append(a)
                    gates.append(g1)
                    mod2s.append(m2)
                parts = ar(parts)
                mlps, gates2 = [], []
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        hs[c] = hs[c] + gates[c].tanh() * parts[c]
                        m, g2 = self._stage_mlp(self._shards[c][i], hs[c], mod2s[c], ttms[c])
                    mlps.append(m)
                    gates2.append(g2)
                mlps = ar(mlps)
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        x = hs[c] + gates2[c].tanh() * mlps[c]
                        hs[c] = x.clip(-65504, 65504) if x.dtype == torch.float16 else x
            self.calls += 1
            out = hs[0]
            return out if out.device == home else out.to(home)

    return TPStack


def install(
    transformer: Any,
    devices: Sequence[Any],
    comm: Optional[str] = None,
    logger: Any = None,
    probe_tokens: int = 4096,
) -> dict:
    """Shard ``transformer``'s blocks over ``devices`` and route its block loop through them. The prologue /
    epilogue (embedders, shared modulation, norm_out, proj_out) move to ``devices[0]``. Returns a status record."""
    import torch
    from torch import nn

    devices = [torch.device(d) for d in devices]
    why = refusal(transformer, devices)
    if why is not None:
        raise RuntimeError(f"tensor parallelism declined: {why}")
    t0 = time.perf_counter()
    blocks = list(transformer.transformer_blocks)
    compile_kw = getattr(transformer, "_unsloth_regional_compile_kwargs", None)
    was_compiled = compile_kw is not None or any(
        b.__dict__.get("_compiled_call_impl") is not None for b in blocks
    )
    shards: list = [[] for _ in devices]
    with torch.no_grad():
        for blk in blocks:
            for c, d in enumerate(devices):
                shards[c].append(shard_block(blk, c, len(devices), d))
            blk.to("meta")  # the whole-block copy is no longer needed
    act_dtype = getattr(transformer, "dtype", None) or torch.float16
    if devices[0].type == "cpu":
        name, fn, probe = "cpu", make_all_reduce("cpu", devices), {}
    elif comm:
        name, fn, probe = comm, make_all_reduce(comm, devices), {}
    else:
        dim = int(transformer.config.num_attention_heads * transformer.config.attention_head_dim)
        name, fn, probe = pick_all_reduce(devices, probe_tokens * dim, act_dtype, logger)
    stack = _stack_class()(shards, devices, fn, name)
    if was_compiled:
        stack.compile(**dict(compile_kw or {}))
    transformer.transformer_blocks = nn.ModuleList([stack])
    transformer.to(devices[0])

    def _local_mib(c: int) -> int:
        seen, total = set(), 0
        for blk in shards[c]:
            for t in list(blk.parameters()) + list(blk.buffers()):
                if id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
        return round(total / 2**20)

    record = {
        "devices": [str(d) for d in devices],
        "comm": name,
        "probe_ms": probe,
        "blocks": len(blocks),
        "shard_mib": [_local_mib(c) for c in range(len(devices))],
        "build_s": round(time.perf_counter() - t0, 1),
    }
    record["stage_compile"] = bool(stack.compiled)
    setattr(transformer, TP_ATTR, record)
    # A later regional compile reaches the stages through TPStack.compile.
    transformer._repeated_blocks = (type(stack).__name__,)
    # One recorded graph cannot hold both cards' launches and the all-reduce between them.
    transformer.__dict__[GRAPH_UNSAFE_ATTR] = "the denoiser is tensor parallel across two cards"
    if logger is not None:
        logger.info("diffusion.tp2: %s", record)
    return record


def is_installed(transformer: Any) -> bool:
    return getattr(transformer, TP_ATTR, None) is not None


def status(transformer: Any) -> Optional[dict]:
    rec = getattr(transformer, TP_ATTR, None)
    if rec is None:
        return None
    out = dict(rec)
    stack = transformer.transformer_blocks[0]
    out["calls"] = int(getattr(stack, "calls", 0))
    out["stage_compiled"] = bool(getattr(stack, "compiled", False))
    return out


def engage_for_plan(
    pipe: Any,
    plan: Any,
    logger: Any = None,
) -> Any:
    """Opt-in (``UNSLOTH_DIFFUSION_TP=1``) step after the multi-GPU plan refinement: shard the denoiser over the
    selection's primary and first secondary card and return the plan with the denoiser resident and no split (the
    shards are placed already; text encoders / VAE keep the plan's policy). Any refusal keeps ``plan``."""
    if not requested():
        return plan
    import dataclasses

    import torch

    def keep(reason: str) -> Any:
        if logger is not None:
            logger.info("diffusion.tp2: not engaged: %s", reason)
        return plan

    try:
        from .diffusion_multigpu import current_selection
        sel = current_selection()
    except Exception as exc:  # noqa: BLE001
        return keep(f"no card selection ({exc})")
    if sel is None or not sel.secondaries:
        return keep("one card selected")
    devices = [
        torch.device("cuda", int(sel.primary)),
        torch.device("cuda", int(sel.secondaries[0])),
    ]
    tr = getattr(pipe, "transformer", None)
    why = refusal(tr, devices)
    if why is not None:
        return keep(why)
    try:
        record = install(
            tr, devices, comm = None if _comm_choice() == "auto" else _comm_choice(), logger = logger
        )
    except Exception as exc:  # noqa: BLE001 - a half-built TP cannot be undone here: surface it
        raise RuntimeError(f"tensor parallelism failed while sharding the denoiser: {exc}") from exc
    pipe._unsloth_tp2 = record
    return dataclasses.replace(
        plan,
        stream_transformer = False,
        # the plan's offload now covers only the encoders (the tier that keeps the denoiser resident)
        stream_text_encoders = bool(plan.stream_text_encoders or plan.offload_policy != "none"),
        resident_transformer_mib = None,
        secondary_ordinal = None,
        secondary_components = None,
        layer_split = None,
        fallback_plan = None,
        reasons = tuple(plan.reasons)
        + (f"denoiser tensor parallel over {', '.join(record['devices'])}",),
    )


def shard_modules(transformer: Any) -> list:
    """Every module of every card's block shards (outside the module tree), for censuses that walk the attention
    modules (INT8 attention routing and its status); empty when ``transformer`` is not tensor parallel."""
    if getattr(transformer, TP_ATTR, None) is None:
        return []
    try:
        stack = transformer.transformer_blocks[0]
        return [m for card in stack.shards() for blk in card for m in blk.modules()]
    except Exception:  # noqa: BLE001 - a census never fails the caller
        return []
