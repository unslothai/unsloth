# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-process tensor parallelism of a DiT over two cards (opt-in UNSLOTH_DIFFUSION_TP=1; Krea 2 also engages
it with the env unset when its Automatic plan would stream the denoiser from host memory, ``=0`` turns that off).

Megatron split of every block: q/k/v and the MLP input layers by output rows (each card its heads and its share
of the hidden width), attention out-proj and MLP out by input channels, one all-reduce after each. One Python
thread drives both cards. Shards live outside the module tree, so moving / hooking the transformer never touches
them. Qwen-Image-2.1 and Krea 2. INT8 row splits quantize per card, so results differ slightly from one card's.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import os
import time
import weakref
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

TP_ENV = "UNSLOTH_DIFFUSION_TP"
TP_COMM_ENV = "UNSLOTH_DIFFUSION_TP_COMM"  # auto | nccl | swap | peer
# 1 compiles each card's stages when the denoiser compiles; default eager (2x T4: no per-step gain, +58 s first image)
TP_COMPILE_ENV = "UNSLOTH_DIFFUSION_TP_COMPILE"
TP_REDUCE_ENV = "UNSLOTH_DIFFUSION_TP_REDUCE"  # fp32 (default on fp16 cards) | native (reduce in the model dtype)
GRAPH_UNSAFE_ATTR = "_unsloth_cuda_graph_unsafe"
# Prefer NCCL unless another backend beats it by more than this (probe noise).
COMM_MARGIN = 0.05
# HostSwap keeps pinned / device buffers for this many (numel, dtype) sizes, least recently used dropped.
HOSTSWAP_SIZES = 2
TP_ATTR = "_unsloth_tp2"
_KREA2 = "Krea2Transformer2DModel"
_DITS = frozenset({"QwenImage21Transformer2DModel", _KREA2})
# Engaged without the env when the load would otherwise stream the denoiser from host memory (``auto_wanted``)
_AUTO_DITS = frozenset({_KREA2})
_COLUMN = ("attn.to_q", "attn.to_k", "attn.to_v", "img_mlp.gate_layer", "img_mlp.proj")
_ROW = ("attn.to_out.0", "img_mlp.out")
# Krea 2: GQA q / k / v and the sigmoid output gate by heads, SwiGLU gate / up by width; out-proj / down by input
_KREA2_COLUMN = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate", "ff.gate", "ff.up")
_KREA2_ROW = ("attn.to_out.0", "ff.down")
_KREA2_PROCESSORS = frozenset({"Krea2AttnProcessor", "Krea2PrefixKVAttnProcessor"})
_COMMS = ("nccl", "swap", "peer")
_FUSED_ATTRS = ("_unsloth_q21_fused_qkv", "_unsloth_q21_dense_gate_up")
_OFF = ("0", "off", "false", "no")


def requested() -> bool:
    return (os.environ.get(TP_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _env_auto() -> bool:
    """The env is unset / ``auto``: neither forced on nor turned off."""
    v = (os.environ.get(TP_ENV) or "").strip().lower()
    return v in ("", "auto")


def _is_krea2_block(block: Any) -> bool:
    return type(block).__name__ == "Krea2TransformerBlock"


def _paths(block: Any) -> tuple:
    """(column-parallel paths, row-parallel paths) of one block."""
    return (_KREA2_COLUMN, _KREA2_ROW) if _is_krea2_block(block) else (_COLUMN, _ROW)


# A plan whose resident share is within this of the whole denoiser holds it whole (planner estimate vs storage).
_WHOLE_SLACK_MIB = 64


def _streams_denoiser(plan: Any, denoiser_mib: Optional[int] = None) -> bool:
    """An Automatic plan that streams part of the denoiser's weights from host memory every step (group or leaf
    streaming with less than the whole denoiser resident). A streaming / group policy that pins the whole denoiser
    (``resident_transformer_mib`` covering it) does not stream it; an unknown denoiser size with a resident share
    set counts as resident."""
    if (getattr(plan, "requested_mode", None) or "auto") != "auto":
        return False  # an explicit memory mode is the user's call
    if getattr(plan, "layer_split", None) is not None:
        return False
    policy = getattr(plan, "offload_policy", "none")
    if policy == "group":
        if getattr(plan, "stream_transformer", True) is False:
            return False
    elif policy != "streaming":
        return False
    resident = getattr(plan, "resident_transformer_mib", None)
    if not resident:
        return True  # every group streams
    if not denoiser_mib:
        return False
    return int(resident) + _WHOLE_SLACK_MIB < int(denoiser_mib)


def _denoiser_mib(transformer: Any) -> Optional[int]:
    try:
        return _storage_mib((*transformer.parameters(), *transformer.buffers()))
    except Exception:  # noqa: BLE001 - unreadable: _streams_denoiser stays conservative
        return None


def auto_wanted(transformer: Any, plan: Any) -> bool:
    """Unset env: TP for a family in ``_AUTO_DITS`` whose Automatic plan streams the denoiser from host memory. Two
    cards then each hold half the blocks resident and run half of every block, instead of one card running all of
    it while the rest streams. A plan that keeps the denoiser on one card (or splits it by layers) is unchanged."""
    return (
        _env_auto()
        and type(transformer).__name__ in _AUTO_DITS
        and _streams_denoiser(plan, _denoiser_mib(transformer))
    )


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


def _is_int8w(lin: Any) -> bool:
    """The small-host load's ``Int8WeightLinear`` (diffusion_small_host): int8 rows, one scale per output row."""
    if type(lin).__name__ != "Int8WeightLinear":
        return False
    q, s = getattr(lin, "qweight", None), getattr(lin, "scale", None)
    return (
        q is not None
        and s is not None
        and q.dim() == 2
        and tuple(s.shape) in ((q.shape[0], 1), (q.shape[0],))
    )


def _linear_refusal(
    lin: Any,
    path: str,
    row: Sequence[str] = _ROW,
    int8w_ok: bool = False,
) -> Optional[str]:
    """Only layer types this module shards explicitly: plain ``nn.Linear`` with a plain floating weight, Studio's
    ``NativeWeightOnlyLinear`` or the small-host ``Int8WeightLinear``. Exact types, not ``isinstance``: a subclass
    (ConvRotLinear rotates its input in ``forward``) or a tensor-subclass weight (torchao) would lose its behaviour in
    a plain sliced Linear."""
    import torch

    if _is_native(lin):
        g = int(getattr(lin, "rot_group", 0) or 0)
        if path in row and g and (lin.in_features // 2) % g:
            return f"{path}: half of {lin.in_features} input channels is not whole {g}-channel ConvRot groups"
        return None
    if int8w_ok and _is_int8w(lin):
        return None
    if type(lin) is not torch.nn.Linear:
        return f"{path} is a {type(lin).__name__} (only nn.Linear and native int8 / fp8 Linears are sharded)"
    w = getattr(lin, "weight", None)
    data = getattr(w, "data", None)
    if type(data) is not torch.Tensor or not data.is_floating_point():
        return f"{path} holds a {type(data).__name__} {getattr(data, 'dtype', '')} weight (plain float expected)"
    return None


def _flex_name(block: Any) -> Optional[str]:
    name = type(getattr(getattr(block, "attn", None), "processor", None)).__name__
    return name if "Flex" in name else None


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
    krea2 = name == _KREA2
    heads = int(cfg.num_attention_heads)
    if krea2:
        kv = int(getattr(cfg, "num_key_value_heads", None) or heads)
        mlp = int(cfg.intermediate_size)
        if heads % 2 or kv % 2 or mlp % 2 or heads % kv:
            return f"{heads} query / {kv} key-value heads / {mlp} MLP width do not split in two"
    else:
        mlp = int(cfg.num_attention_heads * cfg.attention_head_dim * cfg.mlp_ratio)
        if heads % 2 or mlp % 2:
            return f"{heads} heads / {mlp} MLP width do not split in two"
    for blk in transformer.transformer_blocks:
        flex = _flex_name(blk)
        if flex is not None:
            # its BlockMask's mask_mod closes over tensors built on the first card, which no .to() reaches
            return f"{flex} drives flex attention through a BlockMask built on the first card"
        if krea2:
            if not _is_krea2_block(blk):
                return f"a {type(blk).__name__} block (Krea2TransformerBlock expected)"
            proc = type(getattr(blk.attn, "processor", None)).__name__
            if proc not in _KREA2_PROCESSORS:
                # the shard halves num_heads / num_kv_heads; only these processors are known to read them
                return f"{proc} is not a verified Krea 2 attention processor"
        column, row = _paths(blk)
        for path in column + row:
            lin = _get(blk, path)
            why = _linear_refusal(lin, path, row, int8w_ok = krea2)
            if why is not None:
                return why
        for sub_name, sub in blk.named_modules():
            if getattr(sub, "_diffusers_hook", None) is not None:
                # block level: offload hooks (TP replaces offload); a leaf: a layerwise compute cast (the small-host
                # load keeps bf16 storage and casts each Linear to fp16) that a plain shard would not carry
                where = f" on {sub_name}" if sub_name else ""
                return f"the denoiser carries diffusers hooks{where} (offload or layerwise casting)"
            if getattr(sub, "_unsloth_mmap_streamed", False):
                return f"{sub_name or 'a block'} streams its weights with a layerwise cast (small-host load)"
    return None


def reduce_fp32() -> bool:
    return (os.environ.get(TP_REDUCE_ENV) or "fp32").strip().lower() != "native"


_MM_OUT_DTYPE: dict = {}


def _mm_fp32(x2: Any, w: Any) -> Any:
    """``x2 @ w.T`` with float32 output from half inputs: one GEMM with ``out_dtype`` where torch has it for this
    device, else the float32 product."""
    import torch

    key = (x2.device.type, x2.dtype)
    if _MM_OUT_DTYPE.get(key, True):
        try:
            return torch.mm(x2, w.t(), out_dtype = torch.float32)
        except torch.OutOfMemoryError:
            raise  # a full card is not a missing overload: never cache a decline for it
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_OUT_DTYPE[key] = False
    return torch.mm(x2.float(), w.float().t())


def _own(t: Any, device: Any) -> Any:
    """``t`` on ``device`` in a fresh allocation of exactly its size: a slice ``.contiguous().to()`` on the same
    device can return a view that keeps the whole original weight alive."""
    import torch

    out = torch.empty(t.shape, dtype = t.dtype, device = device)
    out.copy_(t)
    return out


def _is_compile_failure(exc: BaseException) -> bool:
    try:
        from .diffusion_speed import is_compile_failure
    except Exception:  # noqa: BLE001 - Studio's predicate unavailable: dynamo's own exception family
        import torch

        exc_mod = getattr(getattr(torch, "_dynamo", None), "exc", None)
        base = getattr(exc_mod, "TorchDynamoException", None)
        return base is not None and isinstance(exc, base)
    return is_compile_failure(exc)


def _autocast_fp16(x: Any) -> bool:
    """True inside an fp16 autocast region for ``x``'s device."""
    import torch
    try:
        kind = x.device.type
        return (
            bool(torch.is_autocast_enabled(kind))
            and torch.get_autocast_dtype(kind) == torch.float16
        )
    except Exception:  # noqa: BLE001 - a torch without the per-device query: no autocast assumed
        return False


def _stage_compile_failure(exc: BaseException) -> bool:
    """Studio's compile-time failure predicate, plus Dynamo's recompile limit under ``fullgraph=True``: the stages
    trace per token length / cache mode / card, and hitting the limit raises before anything ran."""
    if _is_compile_failure(exc):
        return True
    try:
        import torch
        hit = getattr(getattr(torch._dynamo, "exc", None), "FailOnRecompileLimitHit", None)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(hit, type) and isinstance(exc, hit)


@lru_cache(maxsize = 1)
def _row_parallel_class():
    """Built once: a fresh class per shard would make every compiled stage retrace on the type guard."""
    from torch import nn

    class RowParallelOut(nn.Module):
        """A row-parallel shard whose partial sum leaves in float32 on half-precision activations: two partials of
        +6e4 and -6e4 overflow to +Inf / -Inf in fp16 and reduce to NaN, where the unsplit layer gives a finite
        result. The caller casts after the all-reduce (one rounding, as the unsplit layer)."""

        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._modules["inner"], name)

        def forward(self, x: Any) -> Any:
            import torch

            inner = self.inner
            # fp16 autocast turns a bf16 / fp32 activation into fp16 partials inside inner(x): same overflow
            cast16 = x.dtype != torch.float16 and _autocast_fp16(x)
            if (x.dtype != torch.float16 and not cast16) or not reduce_fp32():
                return inner(x)
            if _is_native(inner) or _is_int8w(inner):
                # autocast stays off for the partial: under fp16 autocast its GEMM would return fp16 again
                with torch.autocast(device_type = x.device.type, enabled = False):
                    return inner(x.float())
            x2 = x.reshape(-1, x.shape[-1])
            if cast16:
                # what autocast would feed the unsplit GEMM (fp16 operands), with a float32 partial out
                with torch.autocast(device_type = x.device.type, enabled = False):
                    y = _mm_fp32(x2.to(torch.float16), inner.weight.to(torch.float16))
                    if inner.bias is not None:
                        y = y + inner.bias.float()
            else:
                y = _mm_fp32(x2, inner.weight)
                if inner.bias is not None:
                    y = y + inner.bias.float()
            return y.reshape(*x.shape[:-1], y.shape[-1])

    return RowParallelOut


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
            new._buffers["weight_q"] = _own(wq[sl], device)
            new._buffers["weight_scale"] = _own(ws[sl], device)
            new.out_features = k
            bias = lin.bias[sl] if lin.bias is not None else None
        else:
            k = lin.in_features // parts
            g = int(getattr(lin, "rot_group", 0) or 0)
            if g and k % g:
                raise ValueError(
                    f"row split of {lin.in_features} channels cuts a {g}-channel ConvRot group"
                )
            new._buffers["weight_q"] = _own(wq[:, index * k : (index + 1) * k], device)
            new._buffers["weight_scale"] = _own(ws, device)
            new.in_features = k
            bias = lin.bias if (lin.bias is not None and index == 0) else None  # added once
        for name in list(new._buffers):
            if name not in ("weight_q", "weight_scale") and new._buffers[name] is not None:
                new._buffers[name] = _own(new._buffers[name], device)
        new._parameters.pop("bias", None)
        new.__dict__.pop("bias", None)
        new.bias = (
            None if bias is None else nn.Parameter(_own(bias.detach(), device), requires_grad = False)
        )
        return new
    if _is_int8w(lin):
        # per-output-row scales: a row split keeps the rows' scales whole, a column split slices them with the rows
        q, s = lin.qweight.detach(), lin.scale.detach()
        if dim == 0:
            k = lin.out_features // parts
            sl = slice(index * k, (index + 1) * k)
            qk, sk, ins, outs = q[sl], s[sl], lin.in_features, k
            bias = lin.bias[sl] if lin.bias is not None else None
        else:
            k = lin.in_features // parts
            qk, sk, ins, outs = q[:, index * k : (index + 1) * k], s, k, lin.out_features
            bias = lin.bias if (lin.bias is not None and index == 0) else None
        new = type(lin)(
            _own(qk, device),
            _own(sk, device),
            None if bias is None else _own(bias.detach(), device),
            ins,
            outs,
        )
        new.act_int8 = bool(getattr(lin, "act_int8", False))
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
    out.weight = nn.Parameter(_own(wk, device), requires_grad = False)
    if bias is not None:
        out.bias = nn.Parameter(_own(bias.detach(), device), requires_grad = False)
    for attr in ("_unsloth_compute_dtype",):
        if hasattr(lin, attr):
            setattr(out, attr, getattr(lin, attr))
    return out


def shard_block(block: Any, index: int, parts: int, device: Any) -> Any:
    """A copy of ``block`` on ``device`` holding card ``index``'s heads and MLP width; big weights are not copied
    whole (the memo shares them, then the split layers are replaced by their slices)."""
    memo: dict = {}
    column, row = _paths(block)
    krea2 = _is_krea2_block(block)
    for path in column + row:
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
    for sub in (block.attn, block.ff if krea2 else block.img_mlp):
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
    for path in column:
        _set(sb, path, shard_linear(_get(block, path), 0, index, parts, device))
    row_cls = _row_parallel_class()
    for path in row:
        _set(sb, path, row_cls(shard_linear(_get(block, path), 1, index, parts, device)))
    if krea2:
        # GQA: card c holds query heads [c*H/2, (c+1)*H/2), which read exactly key-value heads [c*KV/2, (c+1)*KV/2)
        sb.attn.num_heads = block.attn.num_heads // parts
        sb.attn.num_kv_heads = block.attn.num_kv_heads // parts
        if "heads" in block.attn.__dict__:
            sb.attn.heads = block.attn.heads // parts
    else:
        sb.attn.heads = block.attn.heads // parts
        sb.attn.inner_dim = block.attn.inner_dim // parts
    return sb.to(device)


def _evict(host: Any, tmp: Any, keep: int) -> None:
    """Drop the least recently used sizes until ``keep`` remain. Safe while copies are in flight: the pinned
    allocator holds a freed block until its recorded copies finish, and every later use of a device scratch
    buffer on the compute stream was ordered after its H2D writes."""
    while len(host) > max(0, keep):
        key = next(iter(host))
        host.pop(key, None)
        tmp.pop(key, None)


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
        self._host: OrderedDict = OrderedDict()
        self._tmp: OrderedDict = OrderedDict()
        self._read_done: list = [None, None]
        # the add that last read tmp[c], recorded on the stream that ran it (the caller's stream can change)
        self._add_done: list = [None, None]

    def _bufs(self, n: int, dtype: Any) -> tuple:
        import torch

        key = (n, dtype)
        if key in self._host:
            self._host.move_to_end(key)
            self._tmp.move_to_end(key)
        else:
            _evict(self._host, self._tmp, HOSTSWAP_SIZES - 1)
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
            if (
                self._add_done[c] is not None
            ):  # the previous add read tmp[c], maybe on another stream
                self.h2d[c].wait_event(self._add_done[c])
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
            added = torch.cuda.Event()
            added.record(cur)
            self._add_done[c] = added
            # the scratch was written on h2d and read on this stream: an evicted size must not be reused early
            tmp[c].record_stream(cur)
            tmp[c].record_stream(self.h2d[c])
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


def _sync(devices: Sequence[Any]) -> None:
    import torch
    for d in devices:
        if getattr(d, "type", None) == "cuda":
            torch.cuda.synchronize(d)


def choose_comm(timings: dict) -> Optional[str]:
    """NCCL unless a working backend is more than ``COMM_MARGIN`` faster (a single-size probe is noisy)."""
    ok = {k: v for k, v in timings.items() if isinstance(v, (int, float))}
    if not ok:
        return None
    best = min(ok, key = ok.get)
    if "nccl" in ok and best != "nccl" and ok[best] > ok["nccl"] * (1.0 - COMM_MARGIN):
        return "nccl"
    return best


def pick_all_reduce(
    devices: Sequence[Any],
    numel: int,
    dtype: Any,
    logger: Any = None,
    only: Optional[str] = None,
) -> tuple:
    """(name, fn, {name: ms}): the fastest working backend for one ``numel`` all-reduce. A pin (``only`` or the
    env) is probed the same way (one real collective, result checked) and raises when it does not work, so a build
    without NCCL (native Windows) never publishes a load that fails on the first denoise step. Picked once per load
    at one size: on 2x T4 the all-reduce is 4-7 % of a TP block, so per-shape picks gain little."""
    import torch

    forced = only or _comm_choice()
    names = [forced] if forced != "auto" else list(_COMMS)
    timings: dict = {}
    fns: dict = {}
    for name in names:
        try:
            fn = make_all_reduce(name, devices)
            xs = [torch.ones(numel, dtype = dtype, device = d) * (i + 1) for i, d in enumerate(devices)]
            ys = fn([x.clone() for x in xs])
            for _ in range(2):
                fn([x.clone() for x in xs])
            _sync(devices)
            if not all(bool((y == 3).all()) for y in ys):
                raise RuntimeError("wrong sum")
            t = time.perf_counter()
            for _ in range(10):
                fn(xs)
            _sync(devices)
            timings[name] = round((time.perf_counter() - t) * 1000 / 10, 3)  # ms per call
            fns[name] = fn
        except torch.OutOfMemoryError:
            raise  # a full card is not a broken backend: the caller's OOM handling decides
        except Exception as exc:  # noqa: BLE001 - a backend that fails here is simply not used
            timings[name] = f"failed: {type(exc).__name__}: {exc}"[:200]
    choice = choose_comm(timings)
    if choice is None:
        raise RuntimeError(f"no all-reduce backend works on {list(map(str, devices))}: {timings}")
    best = (choice, fns[choice])
    if logger is not None:
        logger.info(
            "diffusion.tp2: all-reduce %s (probe ms per %d elements: %s)", best[0], numel, timings
        )
    return best[0], best[1], timings


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
            """Opt-in (``UNSLOTH_DIFFUSION_TP_COMPILE=1``) compile of the two per-card stages; the all-reduce stays
            eager. Reached via ``compile_repeated_blocks`` or at install when the blocks were compiled already."""
            import torch

            if (os.environ.get(TP_COMPILE_ENV) or "").strip().lower() not in (
                "1",
                "on",
                "true",
                "yes",
            ):
                return
            kw = dict(kwargs)
            object.__setattr__(
                self,
                "_stage_attn",
                self._guarded(torch.compile(_attn_half, *args, **kw), _attn_half),
            )
            object.__setattr__(
                self, "_stage_mlp", self._guarded(torch.compile(_mlp_half, *args, **kw), _mlp_half)
            )
            self.compiled = True

        def _guarded(self, compiled: Any, eager: Any) -> Callable:
            """Studio's compile recovery for the stages: a failure while BUILDING the graph (``is_compile_failure``)
            puts both stages back on eager for this and every later call; any other error propagates."""
            stack = self

            def call(*args: Any, **kwargs: Any) -> Any:
                if not stack.compiled:
                    return eager(*args, **kwargs)
                try:
                    return compiled(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - reraised unless a compile-time failure
                    if not _stage_compile_failure(exc):
                        raise
                    stack.compile_error = f"{type(exc).__name__}: {exc}"[:300]
                    object.__setattr__(stack, "_stage_attn", _attn_half)
                    object.__setattr__(stack, "_stage_mlp", _mlp_half)
                    stack.compiled = False
                return eager(*args, **kwargs)

            return call

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
            if attention_mask is not None and not torch.is_tensor(attention_mask):
                # a flex BlockMask (processor switched after install): its mask_mod captures first-card tensors
                raise RuntimeError(
                    "tensor parallelism cannot run a flex-attention BlockMask on the second card; "
                    f"reload without {TP_ENV} or with the default attention processor"
                )
            hs = [_move(hidden_states, d) for d in devs]
            mods = [_move(modulation, d) for d in devs]
            ropes = [_move(rotary_emb, d) for d in devs]
            masks = [_move(attention_mask, d) for d in devs]
            ttms = [_move(target_token_mask, d) for d in devs]
            kvs = [_move(key_valid, d) for d in devs]
            caches = self._caches(layer_cache, kv_cache_mode)
            compact = (
                _kv_compactor() if caches is not None and cache_write_slice is not None else None
            )
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
                    if compact and caches is not None and cache_write_slice is not None:
                        compact(caches[c][i])
                parts = ar(parts)
                mlps, gates2 = [], []
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        hs[c] = hs[c] + gates[c].tanh() * parts[c].to(hs[c].dtype)
                        m, g2 = self._stage_mlp(self._shards[c][i], hs[c], mod2s[c], ttms[c])
                    mlps.append(m)
                    gates2.append(g2)
                mlps = ar(mlps)
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        x = hs[c] + gates2[c].tanh() * mlps[c].to(hs[c].dtype)
                        hs[c] = x.clip(-65504, 65504) if x.dtype == torch.float16 else x
            self.calls += 1
            out = hs[0]
            return out if out.device == home else out.to(home)

    return TPStack


def _krea2_attn_half(blk: Any, h: Any, temb: Any, rope: Any, mask: Any) -> tuple:
    """``Krea2TransformerBlock.forward`` up to the attention output (a partial sum on a shard), stock arithmetic."""
    modulation = temb.unflatten(-1, (6, -1)) + blk.scale_shift_table
    prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)
    a = blk.attn(
        (1.0 + prescale) * blk.norm1(h) + preshift,
        attention_mask = mask,
        image_rotary_emb = rope,
    )
    return a, pregate, (postscale, postshift, postgate)


def _krea2_mlp_half(blk: Any, h: Any, post: tuple) -> tuple:
    postscale, postshift, postgate = post
    return blk.ff((1.0 + postscale) * blk.norm2(h) + postshift), postgate


def _move_any(t: Any, device: Any) -> Any:
    """``_move`` that also reaches into a tuple (Krea 2's rotary embedding is a ``(cos, sin)`` pair)."""
    if isinstance(t, tuple):
        return tuple(_move(x, device) for x in t)
    return _move(t, device)


def _krea2_stack_class():
    base = _stack_class()

    class TPStack(base):
        """Krea 2's block loop on both cards: ``(hidden_states, temb, image_rotary_emb, attention_mask)`` as the
        model calls each block. Eager: regional compile (``compile_repeated_blocks``) is a no-op on a TP load (the
        opt-in per-card stage compile covers Qwen-Image-2.1 only); an outer ``torch.compile`` of the whole model
        traces it without extra recompiles."""

        def compile(self, *args: Any, **kwargs: Any) -> None:  # noqa: A003 - nn.Module.compile's name
            return None

        def forward(
            self,
            hidden_states: Any,
            temb: Any,
            image_rotary_emb: Any = None,
            attention_mask: Any = None,
        ) -> Any:
            import torch

            devs, ar = self._devices, self._all_reduce
            # a decode that never reached its restore (an exception between lend and restore) left them lent
            _return_lent(self)
            home = hidden_states.device
            hs = [_move(hidden_states, d) for d in devs]
            tembs = [_move(temb, d) for d in devs]
            ropes = [_move_any(image_rotary_emb, d) for d in devs]
            masks = [_move(attention_mask, d) for d in devs]
            for i in range(len(self._shards[0])):
                parts, gates, posts = [], [], []
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        a, g1, post = _krea2_attn_half(
                            self._shards[c][i], hs[c], tembs[c], ropes[c], masks[c]
                        )
                    parts.append(a)
                    gates.append(g1)
                    posts.append(post)
                parts = ar(parts)
                mlps, gates2 = [], []
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        hs[c] = hs[c] + gates[c] * parts[c].to(hs[c].dtype)
                        m, g2 = _krea2_mlp_half(self._shards[c][i], hs[c], posts[c])
                    mlps.append(m)
                    gates2.append(g2)
                mlps = ar(mlps)
                for c, d in enumerate(devs):
                    with _CardContext(d):
                        hs[c] = hs[c] + gates2[c] * mlps[c].to(hs[c].dtype)
            if not torch.compiler.is_compiling():
                # eager only: a module int mutated under an outer Dynamo trace is a guard that fails every call
                self.calls += 1
            out = hs[0]
            return out if out.device == home else out.to(home)

    return TPStack


def _kv_compactor() -> Optional[Callable]:
    """Studio's prefix K/V compaction (diffusion_qwenimage21): a compiled extract step stores views of the full K/V,
    which would pin the whole prefill through decode; the per-card caches get the same copy-down."""
    try:
        from .diffusion_qwenimage21 import _compact_layer_cache, compact_kv_enabled
    except Exception:  # noqa: BLE001 - no Studio compaction available: keep the entries as stored
        return None
    return _compact_layer_cache if compact_kv_enabled() else None


DECODE_LEND_ENV = (
    "UNSLOTH_DIFFUSION_TP_DECODE_LEND"  # 0 keeps the first card's shards in place for the decode
)
_LENT_ATTR = "_unsloth_tp2_lent"
# Room the second card keeps free beside the lent shards.
_LEND_RESERVE_MIB = 1024


def _shard_tensors(card: list) -> list:
    """(module, kind, name) for every own parameter / buffer of one card's block shards."""
    out = []
    for blk in card:
        for m in blk.modules():
            out += [(m, "p", n) for n, t in m._parameters.items() if t is not None]
            out += [(m, "b", n) for n, t in m._buffers.items() if t is not None]
    return out


def _move_card(entries: list, device: Any) -> None:
    for m, kind, name in entries:
        if kind == "p":
            p = m._parameters[name]
            p.data = p.data.to(device, non_blocking = True)
        else:
            m._buffers[name] = m._buffers[name].to(device, non_blocking = True)


def _return_lent(stack: Any) -> None:
    """Bring every lent tensor home; the record is dropped only once all of them are back, so a failure here raises
    and leaves it for the next call (never a silent mixed placement)."""
    lent = stack.__dict__.get(_LENT_ATTR)
    if lent is None:
        return
    import torch

    entries, home = lent
    with torch.no_grad():
        _move_card(entries, home)
    stack.__dict__.pop(_LENT_ATTR, None)


def _lend_card(
    stack: Any,
    entries: list,
    home: Any,
    dst: Any,
    logger: Any = None,
) -> bool:
    """Move ``entries`` from ``home`` to ``dst``; True when all moved. The lend is recorded BEFORE the first move,
    so a failure part way is always recoverable: the rollback runs through ``_return_lent``, and if the rollback
    fails too the record stays and the next denoiser call retries it (raising there if it still cannot)."""
    import torch

    stack.__dict__[_LENT_ATTR] = (entries, home)
    try:
        with torch.no_grad():
            _move_card(entries, dst)
        return True
    except Exception as exc:  # noqa: BLE001 - an OOM or a failed copy: put everything back
        try:
            _return_lent(stack)
        except Exception as back:  # noqa: BLE001 - kept recorded: the next denoiser call retries
            if logger is not None:
                logger.warning(
                    "diffusion.tp2: decode lend failed (%s) and so did returning the shards to %s (%s); the "
                    "next denoiser call retries the return",
                    exc,
                    home,
                    back,
                )
            return False
        if logger is not None and not isinstance(exc, torch.OutOfMemoryError):
            logger.warning("diffusion.tp2: decode lend skipped: %s", exc)
        return False


def lend_for_decode(pipe: Any, logger: Any = None) -> Optional[Callable[[], None]]:
    """Krea 2 under TP: the first card holds its half of the blocks, the prologue and the VAE, and the VAE decode
    (fp32 on T4, ~6 GiB at 1024 px) does not fit beside them, so its OOM retry moved models off the card or tiled
    (output then depends on free memory). The second card is idle during the decode: the first card's shards move
    there for it and come back after (``restore``; also on the next denoiser call). Weights move unchanged, so the
    images are bit-identical. None when TP is not installed, the denoiser is not Krea 2, or the second card lacks
    the room."""
    tr = getattr(pipe, "transformer", None)
    if type(tr).__name__ != _KREA2 or not is_installed(tr):
        return None
    if (os.environ.get(DECODE_LEND_ENV) or "").strip().lower() in _OFF:
        return None
    stack = _stack(tr)
    if stack is None or stack.__dict__.get(_LENT_ATTR) is not None:
        return None
    devs = list(getattr(stack, "_devices", ()))
    if len(devs) != 2 or any(getattr(d, "type", None) != "cuda" for d in devs):
        return None
    import torch

    try:
        entries = _shard_tensors(stack.shards()[0])
        need = _storage_mib(
            (m._parameters[n] if k == "p" else m._buffers[n]) for m, k, n in entries
        )
        free, _total = _card_free_total_mib(devs[1])
        cached = (
            int(torch.cuda.memory_reserved(devs[1])) - int(torch.cuda.memory_allocated(devs[1]))
        ) >> 20
        if free + max(0, cached) < need + _LEND_RESERVE_MIB:
            if logger is not None:
                logger.info(
                    "diffusion.tp2: decode keeps the shards on %s (%s has %d MiB of the %d MiB needed)",
                    devs[0],
                    devs[1],
                    free + max(0, cached),
                    need + _LEND_RESERVE_MIB,
                )
            return None
    except Exception as exc:  # noqa: BLE001 - nothing moved yet: the decode runs with the shards in place
        if logger is not None:
            logger.warning("diffusion.tp2: decode lend skipped: %s", exc)
        return None
    if not _lend_card(stack, entries, devs[0], devs[1], logger):
        return None
    stack.lends = int(getattr(stack, "lends", 0)) + 1
    rec = getattr(tr, TP_ATTR, None)
    if isinstance(rec, dict):
        rec["decode_lends"] = stack.lends
    if stack.lends == 1 and logger is not None:
        logger.info(
            "diffusion.tp2: the VAE decode borrows %s: its %d MiB of shards wait on %s and return after",
            devs[0],
            need,
            devs[1],
        )
    torch.cuda.synchronize(
        devs[0]
    )  # the copies read the first card's storage before the decode reuses it
    with torch.cuda.device(devs[0]):
        torch.cuda.empty_cache()

    def restore() -> None:
        _return_lent(stack)

    return restore


def _free_cards(devices: Sequence[Any]) -> None:
    gc.collect()
    try:
        import torch
        for d in devices:
            if getattr(d, "type", None) == "cuda":
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - only a memory hint
        pass


def release_kv(transformer: Any) -> int:
    """Drop every per-card prefix K/V the TP stack holds (bytes freed); Studio's final-step release calls it, since
    the pipeline's own cache objects stay empty under TP. Only safe after the last denoiser call."""
    import torch

    if getattr(transformer, TP_ATTR, None) is None:
        return 0
    try:
        stack = transformer.transformer_blocks[0]
        per_render = list(stack._kv.values())
        stack._kv.clear()
    except Exception:  # noqa: BLE001
        return 0
    freed = 0
    for per in per_render:
        for card in per:
            for layer in card:
                for name in ("k", "v"):
                    t = getattr(layer, name, None)
                    if isinstance(t, torch.Tensor):
                        freed += t.untyped_storage().nbytes()
                        setattr(layer, name, None)
    return freed


def install(
    transformer: Any,
    devices: Sequence[Any],
    comm: Optional[str] = None,
    logger: Any = None,
    probe_tokens: int = 4096,
    retire: bool = True,
) -> dict:
    """Shard ``transformer``'s blocks over ``devices`` and route its block loop through them. The prologue /
    epilogue (embedders, shared modulation, norm_out, proj_out) move to ``devices[0]``. Returns a status record.
    ``retire=False`` keeps the original blocks (where they were) for ``uninstall`` until ``commit``."""
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
    act_dtype = getattr(transformer, "dtype", None) or torch.float16
    # the dtype the row-parallel partials travel in (float32 on fp16 activations, see RowParallelOut)
    red_dtype = torch.float32 if (act_dtype == torch.float16 and reduce_fp32()) else act_dtype
    # 1. the all-reduce first: nothing of the model has changed yet
    if devices[0].type == "cpu" and comm in (None, "cpu"):
        name, fn, probe = "cpu", make_all_reduce("cpu", devices), {}
    else:
        dim = int(transformer.config.num_attention_heads * transformer.config.attention_head_dim)
        name, fn, probe = pick_all_reduce(devices, probe_tokens * dim, red_dtype, logger, only = comm)
    # 2. shards next to the untouched blocks; 3. swap; the originals go only once everything is placed
    shards: list = [[] for _ in devices]
    original_blocks = transformer.transformer_blocks
    home = next((p.device for p in transformer.parameters()), None)
    swapped = False
    try:
        with torch.no_grad():
            for blk in blocks:
                for c, d in enumerate(devices):
                    shards[c].append(shard_block(blk, c, len(devices), d))
        stack_cls = _krea2_stack_class() if type(transformer).__name__ == _KREA2 else _stack_class()
        stack = stack_cls(shards, devices, fn, name)
        if was_compiled:
            stack.compile(**dict(compile_kw or {}))
        transformer.transformer_blocks = nn.ModuleList([stack])
        swapped = True
        transformer.to(devices[0])
    except BaseException:
        if swapped:
            transformer.transformer_blocks = original_blocks
            if home is not None:
                with contextlib.suppress(Exception):
                    transformer.to(home)
        shards.clear()
        _free_cards(devices)
        raise
    setattr(transformer, TP_ATTR, {"devices": [str(d) for d in devices], "comm": name})

    def _local_mib(c: int) -> int:
        # storage bytes, not logical ones: a view would keep (and should report) its whole storage
        seen, total = set(), 0
        for blk in shards[c]:
            for t in list(blk.parameters()) + list(blk.buffers()):
                st = t.untyped_storage()
                if st.data_ptr() not in seen:
                    seen.add(st.data_ptr())
                    total += st.nbytes()
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
    object.__setattr__(stack, "_originals", (original_blocks, home))
    if retire:
        commit(transformer, logger)
    transformer._repeated_blocks = (type(stack).__name__,)
    # One recorded graph cannot hold both cards' launches and the all-reduce between them.
    transformer.__dict__[GRAPH_UNSAFE_ATTR] = "the denoiser is tensor parallel across two cards"
    if logger is not None:
        logger.info("diffusion.tp2: %s", record)
    return record


class PlacementError(RuntimeError):
    """The placement could not keep the TP denoiser resident (the encoders' group offload failed); the load must
    uninstall TP and place the unsplit denoiser instead of letting a fallback hook or stream the TP stack."""


def _stack(transformer: Any) -> Any:
    try:
        stack = transformer.transformer_blocks[0]
    except Exception:  # noqa: BLE001
        return None
    return stack if type(stack).__name__ == "TPStack" else None


def commit(transformer: Any, logger: Any = None) -> None:
    """Free the original blocks (to meta) once TP is placed; TP stays complete whatever happens here."""
    import torch

    stack = _stack(transformer)
    originals = getattr(stack, "_originals", None) if stack is not None else None
    if originals is None:
        return
    object.__setattr__(stack, "_originals", None)
    try:
        with torch.no_grad():
            for blk in originals[0]:
                blk.to("meta")
    except Exception as exc:  # noqa: BLE001 - they stay in host memory, TP itself is intact
        rec = getattr(transformer, TP_ATTR, None)
        if isinstance(rec, dict):
            rec["originals_kept"] = f"{type(exc).__name__}: {exc}"[:200]
        if logger is not None:
            logger.warning("diffusion.tp2: original blocks kept after setup: %s", exc)


def uninstall(transformer: Any, logger: Any = None) -> bool:
    """Put the original blocks back (before ``commit`` only) and drop the shards. False when they are gone."""
    stack = _stack(transformer)
    originals = getattr(stack, "_originals", None) if stack is not None else None
    if originals is None:
        return False
    blocks, home = originals
    devices = list(getattr(stack, "_devices", ()))
    object.__setattr__(stack, "_originals", None)
    transformer.transformer_blocks = blocks
    if home is not None:
        with contextlib.suppress(Exception):
            transformer.to(home)
    for key in (TP_ATTR, GRAPH_UNSAFE_ATTR, "_repeated_blocks"):
        transformer.__dict__.pop(key, None)
    with contextlib.suppress(Exception):
        stack._shards.clear()
        stack._kv.clear()
    _free_cards(devices)
    if logger is not None:
        logger.info("diffusion.tp2: uninstalled, the unsplit denoiser is placed instead")
    return True


def place_atomically(
    pipe: Any,
    plan: Any,
    plan_without_tp: Any,
    apply: Callable[[Any], Any],
    logger: Any = None,
) -> tuple:
    """``(apply(plan), plan)`` with TP committed, or, when the placement cannot keep the TP denoiser resident
    (``PlacementError``), TP uninstalled and ``(apply(plan_without_tp), plan_without_tp)``. Any other failure
    uninstalls TP and propagates. Without TP: ``(apply(plan), plan)``."""
    tr = getattr(pipe, "transformer", None)
    if not is_installed(tr):
        return apply(plan), plan
    try:
        out = apply(plan)
    except PlacementError as exc:
        if logger is not None:
            logger.warning(
                "diffusion.tp2: placement failed (%s); the normal placement runs instead", exc
            )
        uninstall(tr, logger)
        with contextlib.suppress(Exception):
            delattr(pipe, "_unsloth_tp2")
        return apply(plan_without_tp), plan_without_tp
    except BaseException:
        if uninstall(tr, logger):
            with contextlib.suppress(Exception):
                delattr(pipe, "_unsloth_tp2")
        raise
    commit(tr, logger)
    return out, plan


def shard_named_modules(transformer: Any) -> list:
    """``(name, module)`` for every card's block shards, named as the unsplit tree would name them
    (``transformer_blocks.<i>.<sub>``), so per-block rules (INT8 attention's native block 0) reach the shards."""
    stack = _stack(transformer) if getattr(transformer, TP_ATTR, None) is not None else None
    if stack is None:
        return []
    out = []
    for card in stack.shards():
        for i, blk in enumerate(card):
            for sub, m in blk.named_modules():
                out.append((f"transformer_blocks.{i}" + (f".{sub}" if sub else ""), m))
    return out


def is_installed(transformer: Any) -> bool:
    return getattr(transformer, TP_ATTR, None) is not None


def status(transformer: Any) -> Optional[dict]:
    rec = getattr(transformer, TP_ATTR, None)
    if rec is None:
        return None
    out = dict(rec)
    stack = transformer.transformer_blocks[0]
    out["calls"] = int(getattr(stack, "calls", 0))
    out["decode_lends"] = int(getattr(stack, "lends", 0))
    out["stage_compiled"] = bool(getattr(stack, "compiled", False))
    return out


def _card_free_total_mib(device: Any) -> tuple:
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return int(free // 2**20), int(total // 2**20)


def _storage_mib(tensors: Any) -> int:
    seen, total = set(), 0
    for t in tensors:
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            total += st.nbytes()
    return -(-total // 2**20)


def fit_refusal(transformer: Any, devices: Sequence[Any]) -> Optional[str]:
    """Per-card render headroom: each card holds half the blocks plus Studio's denoiser working set (twice on the
    first card, which also runs the embedders, encode and decode) and the reserve a CFG replica keeps; the first card
    also holds the prologue / epilogue. None when both fit. VAE decode is not priced: Studio's decode OOM recovery
    owns it."""
    try:
        blocks = list(transformer.transformer_blocks)
        block_mib = _storage_mib(t for b in blocks for t in (*b.parameters(), *b.buffers()))
        all_mib = _storage_mib((*transformer.parameters(), *transformer.buffers()))
    except Exception as exc:  # noqa: BLE001
        return f"denoiser size unreadable ({exc})"
    prologue = max(0, all_mib - block_mib)
    shard = -(-block_mib // len(devices))
    try:
        from .diffusion_multigpu import _replica_reserve_mib, replica_working_mib
        working = int(replica_working_mib(transformer))
    except Exception:  # noqa: BLE001
        working, _replica_reserve_mib = 1024, None
    for i, d in enumerate(devices):
        try:
            free, total = _card_free_total_mib(d)
        except Exception as exc:  # noqa: BLE001
            return f"{d}: free memory unreadable ({exc})"
        reserve = _replica_reserve_mib(free, total) if _replica_reserve_mib is not None else 1024
        need = shard + working * (2 if i == 0 else 1) + reserve + (prologue if i == 0 else 0)
        if free < need:
            return f"{d} needs {need} MiB for its shard, working set and reserve; {free} MiB free"
    return None


def _tp_plan(plan: Any, record: dict) -> Any:
    """The plan with the denoiser resident on its shards: no split, no denoiser offload. A plan that offloads keeps
    offloading the encoders through group offload (the only tier whose denoiser can stay resident); the others
    (whole-module, sequential, streaming) would move or hook the denoiser."""
    import dataclasses

    policy = getattr(plan, "offload_policy", "none")
    offloads = policy != "none"
    return dataclasses.replace(
        plan,
        offload_policy = "group" if offloads else "none",
        stream_transformer = False,
        stream_text_encoders = bool(getattr(plan, "stream_text_encoders", False) or offloads),
        resident_transformer_mib = None,
        secondary_ordinal = None,
        secondary_components = None,
        layer_split = None,
        fallback_plan = None,
        reasons = tuple(plan.reasons)
        + (f"denoiser tensor parallel over {', '.join(record['devices'])}",),
    )


def engage_for_plan(
    pipe: Any,
    plan: Any,
    logger: Any = None,
    loras: Any = None,
) -> Any:
    """Opt-in (``UNSLOTH_DIFFUSION_TP=1``, or ``auto_wanted`` with the env unset) step after the multi-GPU plan
    refinement: shard the denoiser over the selection's primary and the secondary with the most headroom that fits,
    and return the plan with the denoiser resident and no split (``_tp_plan``). Any refusal or failure keeps
    ``plan``."""
    if not (requested() or auto_wanted(getattr(pipe, "transformer", None), plan)):
        return plan
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
    if any(w for (_i, w) in _lora_pairs(loras)):
        # PEFT target discovery cannot see the shards: a load that names adapters keeps the normal plan
        return keep("the load names LoRA adapters")
    if (
        getattr(plan, "secondary_components", None)
        and getattr(plan, "offload_policy", None) == "none"
    ):
        # the plan was sized with those components on the other card; TP would bring them back to the first
        return keep(f"the plan places {', '.join(plan.secondary_components)} on another card")
    tr = getattr(pipe, "transformer", None)
    primary = torch.device("cuda", int(sel.primary))
    reasons, devices = [], None
    for ordinal in sel.secondaries:
        cand = [primary, torch.device("cuda", int(ordinal))]
        why = refusal(tr, cand) or fit_refusal(tr, cand)
        if why is None:
            devices = cand
            break
        reasons.append(why)
    if devices is None:
        return keep("; ".join(reasons))
    try:
        record = install(
            tr,
            devices,
            comm = None if _comm_choice() == "auto" else _comm_choice(),
            logger = logger,
            retire = False,  # the originals go only once placement succeeded (place_atomically)
        )
    except torch.OutOfMemoryError:
        raise  # install restored the blocks; Studio's OOM handling owns the retry
    except Exception as exc:  # noqa: BLE001 - install restored the blocks: the normal plan still loads
        if logger is not None:
            logger.warning(
                "diffusion.tp2: setup failed, the normal placement runs instead: %s", exc
            )
        return plan
    pipe._unsloth_tp2 = record
    return _tp_plan(plan, record)


def _lora_pairs(loras: Any) -> list:
    out = []
    for entry in loras or ():
        try:
            lora_id, weight = entry
            out.append((lora_id, float(weight)))
        except Exception:  # noqa: BLE001 - an unrecognised entry names no adapter
            continue
    return out


def lora_block_reason(pipe: Any) -> Optional[str]:
    """Why adapters cannot attach to this load (TP shards are invisible to PEFT's target discovery), or None."""
    if getattr(getattr(pipe, "transformer", None), TP_ATTR, None) is None:
        return None
    return (
        "LoRA is not available on this load: the denoiser runs tensor parallel across two cards "
        f"({TP_ENV}), which cannot carry adapters yet. Reload with {TP_ENV}=0, or name the LoRA in the load "
        "request (Unsloth then keeps the normal placement)."
    )


def shard_modules(transformer: Any, first_card_only: bool = False) -> list:
    """Every module of every card's block shards (outside the module tree), for censuses that walk the attention
    modules (INT8 attention routing and its status); empty when ``transformer`` is not tensor parallel."""
    if getattr(transformer, TP_ATTR, None) is None:
        return []
    try:
        stack = transformer.transformer_blocks[0]
        cards = stack.shards()[:1] if first_card_only else stack.shards()
        return [m for card in cards for blk in card for m in blk.modules()]
    except Exception:  # noqa: BLE001 - a census never fails the caller
        return []
