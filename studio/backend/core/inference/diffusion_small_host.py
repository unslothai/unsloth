# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host load route: keep a pipeline load inside host RAM when the compute dtype differs from the stored one.

Root cause it addresses (measured, FLUX.1-schnell on an sm75 card, which computes in fp16): a bf16 repo loaded with
``torch_dtype=float16`` materialises every converted weight in anonymous host memory (transformer 22.7 GB, T5
11.0 GB, 35 GB peak), while the same load at the stored bf16 stays file-backed (0.6 GB anonymous). On a 31 GB host the
fp16 load is OOM-killed inside ``from_pretrained`` before any planner or offload code runs.

The route, engaged only when the dense converted load would not fit the host's available RAM (or forced):
  * big bf16 components load at their stored dtype, so their weights stay memory-mapped;
  * streamed text encoders get diffusers layerwise casting (bf16 storage, compute-dtype forward) under leaf-level
    group offload, unpinned, so their host bytes stay in the page cache and every forward computes in the same
    compute dtype the dense load used (bf16 -> fp16 is exact, so the encoder output is bit-identical);
  * a denoiser too large to stay resident in the compute dtype is stored as int8 per-output-channel weights
    (quantised on the GPU one Linear at a time, fp16 dequant in forward), halving its host and device bytes;
  * a denoiser that stays resident is cast on the device (exact).

Kill switch ``UNSLOTH_DIFFUSION_SMALL_HOST=0`` restores the dense load; ``=1`` forces the route for testing.
``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` skips the pre-load refusal when even the route cannot fit.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SMALL_HOST_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST"
HOST_RAM_CHECK_ENV = "UNSLOTH_DIFFUSION_HOST_RAM_CHECK"
SMALL_HOST_ATTR = "_unsloth_small_host"

_MIB = 1024 * 1024
# Components below this load converted as before: their anonymous bytes are noise next to the reserve.
_STREAM_MIN_MIB = 512
# Host RAM kept free beyond the predicted bytes: the CUDA context, torch, the VAE and the decode buffers.
_HOST_RESERVE_MIN_MIB = 3072
_HOST_RESERVE_FRACTION = 0.10
# fp16 T5 keeps its ``wo`` projections in fp32 (transformers ``_keep_in_fp32_modules``): 9.5 GB stored, 11.0 GB loaded.
_CONVERT_MARGIN = 1.15
# Linears this small stay dense in the compute dtype (embedders, final projections, modulation heads of small DiTs).
_INT8_MIN_ELEMENTS = 1 << 22
_DENOISER_NAMES = ("transformer", "transformer_2", "unconditional_transformer", "unet")


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def small_host_disabled() -> bool:
    return _env(SMALL_HOST_ENV) in ("0", "off", "false", "no")


def small_host_forced() -> bool:
    return _env(SMALL_HOST_ENV) in ("1", "on", "true", "yes", "force")


def host_ram_check_disabled() -> bool:
    return _env(HOST_RAM_CHECK_ENV) in ("0", "off", "false", "no")


_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _safetensors_bytes_by_dtype(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if not 0 < n <= 256 * _MIB:
                return out
            header = json.loads(fh.read(n))
    except Exception:  # noqa: BLE001 - unreadable header: no claim about this file
        return out
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = meta["data_offsets"]
            key = _ST_DTYPES.get(str(meta.get("dtype")), str(meta.get("dtype")).lower())
            out[key] = out.get(key, 0) + int(end) - int(start)
        except Exception:  # noqa: BLE001
            continue
    return out


@dataclass(frozen = True)
class StoredComponent:
    name: str
    mib: int
    dtype: str  # dominant stored float dtype ("bfloat16", "float16", "float32", ...)


def stored_components(snapshot: Any) -> dict[str, StoredComponent]:
    """Per pipeline component: stored MiB and dominant dtype, from the safetensors headers alone (no weight read)."""
    out: dict[str, StoredComponent] = {}
    try:
        root = Path(snapshot)
        if not root.is_dir():
            return out
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            totals: dict[str, int] = {}
            for f in sub.glob("*.safetensors"):
                for k, v in _safetensors_bytes_by_dtype(f).items():
                    totals[k] = totals.get(k, 0) + v
            if not totals:
                continue
            dtype = max(totals.items(), key = lambda kv: kv[1])[0]
            out[sub.name] = StoredComponent(sub.name, -(-sum(totals.values()) // _MIB), dtype)
    except Exception:  # noqa: BLE001 - no claim
        return {}
    return out


def resolve_snapshot_dir(repo_or_dir: Any, cache_dir: Any = None) -> Optional[Path]:
    """The local diffusers directory for a repo id (cache only, never the network) or a local path."""
    try:
        p = Path(str(repo_or_dir)).expanduser()
        if p.is_dir():
            return p
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(str(repo_or_dir), "model_index.json", cache_dir = cache_dir)
        if isinstance(hit, str):
            return Path(hit).parent
    except Exception:  # noqa: BLE001
        return None
    return None


def host_ram_mib() -> tuple[Optional[int], Optional[int]]:
    """(total, available) host MiB, both capped by an enforcing cgroup."""
    from .diffusion_memory import (
        _available_system_memory_mib,
        _cgroup_memory_limit_mib,
        _system_memory_mib,
    )

    total, _ = _system_memory_mib()
    available = _available_system_memory_mib()
    limit = _cgroup_memory_limit_mib()
    if total is not None and limit is not None:
        total = min(int(total), int(limit))
    return total, available


def host_reserve_mib(total_mib: Optional[int]) -> int:
    return max(_HOST_RESERVE_MIN_MIB, int((total_mib or 0) * _HOST_RESERVE_FRACTION))


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _itemsize(name: str) -> int:
    return {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}.get(
        name, 2
    )


@dataclass(frozen = True)
class SmallHostDecision:
    engaged: bool
    reason: str
    available_mib: Optional[int] = None
    total_mib: Optional[int] = None
    dense_host_mib: int = 0
    route_host_mib: int = 0
    # components loaded at their stored dtype (memory-mapped) instead of the compute dtype
    storage_dtypes: dict[str, str] = field(default_factory = dict)
    refuse: Optional[str] = None


def decide_small_host(
    components: dict[str, StoredComponent],
    compute_dtype: Any,
    *,
    device: str,
    host_total_mib: Optional[int],
    host_available_mib: Optional[int],
    lora_active: bool = False,
) -> SmallHostDecision:
    """Whether this pipeline load takes the small-host route. Pure, so the decision table is testable.

    Engages only for CUDA loads whose big components are stored in a different float dtype than the compute dtype
    (bf16 repo on an fp16-only card, or promoted to fp32): there, and only there, ``from_pretrained`` materialises the converted copy in
    anonymous host memory. Same-dtype loads stay memory-mapped and are never touched."""
    if small_host_disabled():
        return SmallHostDecision(False, f"{SMALL_HOST_ENV}=0")
    if str(device) != "cuda":
        return SmallHostDecision(False, "not a CUDA load")
    compute = _dtype_name(compute_dtype)
    if compute not in ("float16", "float32"):
        # bf16 compute loads a bf16 repo memory-mapped already
        return SmallHostDecision(False, f"compute dtype {compute} needs no conversion")
    converted = {
        name: comp
        for name, comp in components.items()
        if comp.dtype == "bfloat16" and comp.mib >= _STREAM_MIN_MIB
    }
    if not converted:
        return SmallHostDecision(False, "no large bf16 component to convert")
    dense = int(
        sum(c.mib * _itemsize(compute) / _itemsize(c.dtype) for c in converted.values())
        * _CONVERT_MARGIN
    )
    # Route: encoders stay memory-mapped; a denoiser may land on the host as int8 (half its stored bf16 bytes).
    route = sum(c.mib // 2 for n, c in converted.items() if n in _DENOISER_NAMES)
    reserve = host_reserve_mib(host_total_mib)
    storage = {name: comp.dtype for name, comp in converted.items()}
    forced = small_host_forced()
    if host_available_mib is None and not forced:
        return SmallHostDecision(False, "host RAM unreadable", dense_host_mib = dense)
    fits = host_available_mib is not None and dense + reserve <= int(host_available_mib)
    if fits and not forced:
        return SmallHostDecision(
            False,
            f"dense {compute} load ({dense} MiB) fits {host_available_mib} MiB available host RAM",
            host_available_mib,
            host_total_mib,
            dense,
            route,
        )
    if lora_active and not forced:
        # The int8 denoiser cannot carry adapters: keep the dense load (it may still fit with swap or page reclaim).
        return SmallHostDecision(
            False,
            f"LoRA adapters need the dense denoiser (dense load ~{dense} MiB, {host_available_mib} MiB available)",
            host_available_mib,
            host_total_mib,
            dense,
            route,
        )
    refuse = None
    if (
        host_available_mib is not None
        and route + reserve > int(host_available_mib)
        and not host_ram_check_disabled()
    ):
        refuse = _refusal(route, reserve, host_available_mib, "even the low-memory route")
    why = (
        f"{SMALL_HOST_ENV}=1"
        if forced and fits
        else f"dense {compute} load needs ~{dense} MiB + {reserve} MiB reserve of host RAM, "
        f"{host_available_mib} MiB available"
    )
    return SmallHostDecision(
        True, why, host_available_mib, host_total_mib, dense, route, storage, refuse
    )


def _refusal(need: int, reserve: int, available: Optional[int], what: str) -> str:
    return (
        f"Not enough system RAM to load this model: {what} needs about {(need + reserve) / 1024:.1f} GB "
        f"of host memory and {((available or 0)) / 1024:.1f} GB is available. Close other applications, "
        f"pick a smaller or GGUF model, or set {HOST_RAM_CHECK_ENV}=0 to try anyway."
    )


def torch_dtype_map(decision: SmallHostDecision, compute_dtype: Any) -> Any:
    """``torch_dtype`` for ``DiffusionPipeline.from_pretrained``: stored dtype for the routed components."""
    if not decision.engaged or not decision.storage_dtypes:
        return compute_dtype
    import torch

    mapping: dict[str, Any] = {"default": compute_dtype}
    for name, dtype in decision.storage_dtypes.items():
        mapping[name] = getattr(torch, dtype)
    return mapping


# --------------------------------------------------------------------------------------------- int8 weight storage
def _int8_linear_class():
    import torch
    import torch.nn.functional as F

    class Int8WeightLinear(torch.nn.Module):
        """Linear with int8 per-output-channel weights, dequantised to the input dtype for each forward."""

        def __init__(self, qweight, scale, bias, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.qweight = torch.nn.Parameter(qweight, requires_grad = False)
            self.scale = torch.nn.Parameter(scale, requires_grad = False)
            self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad = False)

        def forward(self, x):
            scale = self.scale
            if scale.dtype == x.dtype:
                # One elementwise pass: int8 * float promotes to the float dtype, so the cast and the per-row scale
                # run in a single kernel. Same op math as cast-then-multiply (int8 -> fp16/fp32 is exact), so the
                # weight is bit-identical; the separate multiply was a second full pass over the weight.
                w = torch.mul(self.qweight, scale)
            else:
                w = self.qweight.to(x.dtype) * scale.to(x.dtype)
            return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, int8 weight"

    return Int8WeightLinear


_INT8_CLS = None


def int8_linear_class():
    global _INT8_CLS
    if _INT8_CLS is None:
        _INT8_CLS = _int8_linear_class()
    return _INT8_CLS


def quantize_int8_weight_(
    module: Any,
    *,
    compute_dtype: Any,
    work_device: Any,
    keep_device: Any = "cpu",
) -> dict[str, int]:
    """Replace every large ``nn.Linear`` of ``module`` with int8 weight storage, in place, one Linear at a time.

    Each weight is read once (memory-mapped bf16), quantised on ``work_device`` with a per-row absmax scale and
    stored on ``keep_device``; every other parameter / buffer is cast to ``compute_dtype``. Host peak = the int8
    result plus one weight in flight."""
    import torch

    cls = int8_linear_class()
    work = torch.device(work_device)
    keep = torch.device(keep_device)
    stats = {"linears": 0, "dense_linears": 0, "int8_bytes": 0}
    targets = [
        (name, child) for name, child in module.named_modules() if type(child) is torch.nn.Linear
    ]
    for name, lin in targets:
        if lin.weight.numel() < _INT8_MIN_ELEMENTS:
            stats["dense_linears"] += 1
            continue
        with torch.no_grad():
            w = lin.weight.detach().to(work, non_blocking = False).float()
            scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-12) / 127.0
            q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
            bias = None if lin.bias is None else lin.bias.detach().to(keep, compute_dtype)
            new = cls(
                q.to(keep),
                scale.to(compute_dtype).to(keep),
                bias,
                lin.in_features,
                lin.out_features,
            )
            del w, q, scale
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, new)
        stats["linears"] += 1
        stats["int8_bytes"] += new.qweight.numel()
    # What is left (norms, embedders, the small Linears) converts to the compute dtype like the dense load would.
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, cls):
                continue
            for pname, p in list(sub.named_parameters(recurse = False)):
                if p.dtype == torch.bfloat16 and p.dtype != compute_dtype:
                    p.data = p.data.to(keep, compute_dtype)
                elif p.device != keep:
                    p.data = p.data.to(keep)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is not None and b.dtype == torch.bfloat16 and b.dtype != compute_dtype:
                    sub._buffers[bname] = b.to(keep, compute_dtype)
    if work.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def _keep_fp32_patterns(module: Any) -> tuple[str, ...]:
    pats = getattr(module, "_keep_in_fp32_modules", None) or ()
    strict = getattr(module, "_keep_in_fp32_modules_strict", None) or ()
    return tuple(str(p) for p in (*pats, *strict))


def cast_resident_(module: Any, device: Any, compute_dtype: Any) -> None:
    """Move ``module`` onto ``device`` one tensor at a time, casting stored floats to the dense load's dtypes
    (compute dtype, fp32 for ``_keep_in_fp32_modules``): exact, and no host copy."""
    import torch

    dev = torch.device(device)
    keep = _keep_fp32_patterns(module)
    with torch.no_grad():
        for mname, sub in module.named_modules():
            want = torch.float32 if keep and any(k in mname for k in keep) else compute_dtype
            for pname, p in list(sub.named_parameters(recurse = False)):
                dt = want if p.dtype == torch.bfloat16 else p.dtype
                p.data = p.data.to(dev).to(dt)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is None:
                    continue
                dt = want if b.dtype == torch.bfloat16 else b.dtype
                sub._buffers[bname] = b.to(dev).to(dt)


def prepare_streamed_encoder_(module: Any, compute_dtype: Any) -> int:
    """Layerwise-cast a memory-mapped bf16 encoder: Linear and Embedding weights stay in their stored dtype on the
    host (and in the page cache), each layer computes in ``compute_dtype`` (fp32 for ``_keep_in_fp32_modules``, as the
    dense load keeps them). Norms and other small parameters convert now, and so does whatever owns the first float
    parameter, because ``module.dtype`` (first float parameter) is what the pipeline casts prompt embeddings to.
    Returns the MiB still stored memory-mapped."""
    import torch
    from diffusers.hooks import apply_layerwise_casting

    keep = _keep_fp32_patterns(module)
    streamable = (torch.nn.Linear, torch.nn.Embedding)

    def _kept(mname: str) -> bool:
        return bool(keep) and any(k in mname for k in keep)

    def _want(mname: str) -> Any:
        return torch.float32 if _kept(mname) else compute_dtype

    def _convert(sub: Any, want: Any) -> None:
        # Only what the stored-dtype load changed: an fp32 buffer (RoPE ``inv_freq``) stays fp32, as in the dense load.
        for pname, p in list(sub.named_parameters(recurse = False)):
            if p.dtype == torch.bfloat16 and p.dtype != want:
                p.data = p.data.to(want)
        for bname, b in list(sub.named_buffers(recurse = False)):
            if b is not None and b.dtype == torch.bfloat16 and b.dtype != want:
                sub._buffers[bname] = b.to(want)

    first_owner = None
    for pname, p in module.named_parameters():
        if p.is_floating_point():
            first_owner = pname.rpartition(".")[0]
            break
    streamed = 0
    with torch.no_grad():
        for mname, sub in module.named_modules():
            want = _want(mname)
            # A kept-fp32 Linear converts now: its parent reads ``weight.dtype`` BEFORE the layer runs (T5's ``wo``
            # cast) and would cast its input to the stored bf16. 4 GB for T5-XXL, the bytes the dense load holds too.
            if (
                type(sub) in streamable
                and not _kept(mname)
                and mname != first_owner
                and sub.weight.dtype != want
            ):
                storage = sub.weight.dtype
                streamed += sub.weight.numel() * sub.weight.element_size()
                # No skip pattern / class: the hook lands on this layer itself.
                apply_layerwise_casting(
                    sub,
                    storage_dtype = storage,
                    compute_dtype = want,
                    skip_modules_pattern = None,
                    skip_modules_classes = None,
                )
                continue
            _convert(sub, want)
    return streamed // _MIB


def module_dtype_ok(module: Any, compute_dtype: Any) -> bool:
    try:
        return getattr(module, "dtype", compute_dtype) == compute_dtype
    except Exception:  # noqa: BLE001
        return True


def mark(pipe: Any, info: dict) -> None:
    try:
        setattr(pipe, SMALL_HOST_ATTR, dict(info))
    except Exception:  # noqa: BLE001
        pass


def engaged_on(pipe: Any) -> Optional[dict]:
    return getattr(pipe, SMALL_HOST_ATTR, None)


# ------------------------------------------------------------------------------------ streamed-encoder prefetch
ENCODER_PREFETCH_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_PREFETCH"
ENCODER_PREFETCH_ATTR = "_unsloth_encoder_prefetch"
# Pinned staging ring: the only host bytes this adds (diffusers pins a fresh copy of every tensor per onload instead).
_STAGE_SLOT_BYTES = 32 * _MIB
_STAGE_SLOTS = 4
# Device bytes copied ahead of the encoder's forward: a few groups (a T5-XXL MLP weight is 80 MiB, a Qwen2.5-VL one
# 130 MiB) keep the copy ahead of the compute.
_PREFETCH_MAX_BYTES = 384 * _MIB
_PREFETCH_MIN_BYTES = 128 * _MIB


def encoder_prefetch_disabled() -> bool:
    return _env(ENCODER_PREFETCH_ENV) in ("0", "off", "false", "no")


def _group_tensors(group: Any) -> list:
    """(tensor, host source) pairs of a diffusers offload group, in the order diffusers onloads them."""
    out: list = []
    seen: set[int] = set()
    cpu = getattr(group, "cpu_param_dict", None) or {}
    for t in (
        [p for m in group.modules for p in m.parameters()]
        + [b for m in group.modules for b in m.buffers()]
        + list(getattr(group, "parameters", None) or [])
        + list(getattr(group, "buffers", None) or [])
    ):
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append((t, cpu.get(t, t)))
    return out


class _EncoderPrefetcher:
    """Streams a memory-mapped text encoder's offload groups ahead of its forward.

    diffusers' leaf-level stream pins a fresh host copy of every weight inside each group's ``onload_`` (a
    single-threaded page-fault-and-copy on the calling thread) and then waits for that one copy, so the encoder
    runs at the host's copy speed with the GPU and the PCIe link mostly idle. Here a worker thread copies the groups,
    in the order the previous forward ran them, through a small reusable pinned ring into device tensors on a side
    stream, up to a bounded number of bytes ahead; each group's ``onload_`` only makes the compute stream wait for its
    copy. Bytes, dtypes and ops are unchanged, so the encoder output is bit-identical. A group the order did not
    predict, or any worker error, falls back to a synchronous copy of that group."""

    def __init__(self, module: Any, groups: list, device: Any):
        import threading

        import torch

        self.module = module
        self.groups = groups
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.stream = None  # side stream, created on the first copy
        self.compute = None  # the stream the encoder runs on, read when a forward begins
        self.order: list = []  # group ids in the last forward's onload order
        self.seen: list = []
        self.active = False
        self.cond = threading.Condition()
        self.ready: dict = {}  # id(group) -> (device tensors, event)
        self.inflight = 0
        self.budget = _PREFETCH_MIN_BYTES
        self.pos = 0
        self.stop = False
        self.worker = None
        self.error: Optional[BaseException] = None
        self.stage: list = []
        self.stage_events: list = []
        self.slot = 0
        self.stats = {"prefetched": 0, "sync": 0, "passes": 0}
        self.by_id = {id(g): g for g in groups}

    # -- copies ------------------------------------------------------------------------------------------------------
    def _side_stream(self) -> Any:
        import torch

        if self.stream is None:
            self.stream = torch.cuda.Stream(device = self.device)
        return self.stream

    def _ensure_stage(self) -> None:
        import torch

        if not self.stage:
            self.stage = [
                torch.empty(_STAGE_SLOT_BYTES, dtype = torch.uint8, pin_memory = True)
                for _ in range(_STAGE_SLOTS)
            ]
            self.stage_events = [None] * _STAGE_SLOTS

    def _copy_group(self, group: Any) -> tuple:
        """Device copies of ``group``'s host tensors, issued on the side stream. Returns (tensors, event, bytes)."""
        import torch

        self._ensure_stage()
        stream = self._side_stream()
        compute = self.compute or torch.cuda.current_stream(self.device)
        moved: list = []
        nbytes = 0
        with torch.cuda.stream(stream):
            for _t, src in _group_tensors(group):
                n = int(src.numel()) * int(src.element_size())
                if src.device.type != "cpu" or n == 0 or not src.is_contiguous() or src.is_pinned():
                    # Device-resident, empty, strided or already pinned: a plain copy on the compute stream.
                    with torch.cuda.stream(compute):
                        moved.append(src.to(self.device, non_blocking = src.device.type != "cpu" or src.is_pinned()))
                    continue
                nbytes += n
                # Allocated from the compute stream's pool (a side-stream block would stay cached where the denoise
                # cannot reuse it); the copy waits for everything already queued there, which covers whatever last used
                # a freed block it receives.
                with torch.cuda.stream(compute):
                    dst = torch.empty(src.shape, dtype = src.dtype, device = self.device)
                    queued = torch.cuda.Event()
                    queued.record(compute)
                stream.wait_event(queued)
                sb = src.reshape(-1).view(torch.uint8)
                db = dst.reshape(-1).view(torch.uint8)
                off = 0
                while off < n:
                    k = min(_STAGE_SLOT_BYTES, n - off)
                    i = self.slot
                    self.slot = (i + 1) % _STAGE_SLOTS
                    ev = self.stage_events[i]
                    if ev is not None:
                        ev.synchronize()
                    buf = self.stage[i][:k]
                    buf.copy_(sb[off : off + k])
                    db[off : off + k].copy_(buf, non_blocking = True)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    self.stage_events[i] = ev
                    off += k
                moved.append(dst)
            done = torch.cuda.Event()
            done.record(stream)
        return moved, done, nbytes

    def _attach(self, group: Any, moved: list, done: Any) -> None:
        import torch

        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(done)
        for (t, _src), dev in zip(_group_tensors(group), moved):
            t.data = dev
            dev.record_stream(cur)

    # -- worker ------------------------------------------------------------------------------------------------------
    def _run(self, order: list) -> None:
        import torch

        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            for gid in order:
                with self.cond:
                    # A group the forward onloads twice waits for its first copy to be taken.
                    while not self.stop and (
                        (self.inflight > 0 and self.inflight >= self.budget) or gid in self.ready
                    ):
                        self.cond.wait(0.05)
                    if self.stop:
                        return
                group = self.by_id.get(gid)
                if group is None:
                    continue
                moved, done, n = self._copy_group(group)
                with self.cond:
                    if self.stop:
                        return
                    self.ready[gid] = (moved, done, n)
                    self.inflight += n
                    self.cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - every group falls back to a synchronous copy
            with self.cond:
                self.error = exc
                self.cond.notify_all()

    def _halt(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        if self.worker is not None:
            self.worker.join()
        self.worker = None
        with self.cond:
            self.ready.clear()
            self.inflight = 0

    # -- hooks -------------------------------------------------------------------------------------------------------
    def begin(self) -> None:
        import threading

        import torch

        self._halt()
        self.stop = False
        self.error = None
        self.seen = []
        self.pos = 0
        self.active = True
        self.stats["passes"] += 1
        self.compute = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if not self.order:
            return  # first forward: synchronous copies, and it records the order
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
            self.budget = max(_PREFETCH_MIN_BYTES, min(_PREFETCH_MAX_BYTES, int(free) // 8))
        except Exception:  # noqa: BLE001
            self.budget = _PREFETCH_MIN_BYTES
        self._order_after_compute()
        self.worker = threading.Thread(
            target = self._run, args = (list(self.order),), daemon = True, name = "unsloth-encoder-prefetch"
        )
        self.worker.start()

    def _order_after_compute(self) -> None:
        """The worker's copies must not overtake work the compute stream already queued (the prompt's input ids)."""
        import torch

        self._side_stream().wait_stream(torch.cuda.current_stream(self.device))

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        self._halt()
        if self.seen:
            self.order = list(self.seen)

    def onload(self, group: Any) -> None:
        gid = id(group)
        if self.active:
            self.seen.append(gid)
        entry = None
        if self.worker is not None:
            expected = self.order[self.pos] if self.pos < len(self.order) else None
            if expected == gid:
                self.pos += 1
                with self.cond:
                    while gid not in self.ready and self.error is None and self.worker.is_alive():
                        self.cond.wait(0.05)
                    entry = self.ready.pop(gid, None)
                    if entry is not None:
                        self.inflight -= entry[2]
                        self.cond.notify_all()
            else:
                # Off the recorded order: stop prefetching for this forward; the next one re-records the order.
                self._halt()
        if entry is None:
            moved, done, _n = self._copy_group(group)
            self.stats["sync"] += 1
        else:
            moved, done, _n = entry
            self.stats["prefetched"] += 1
        self._attach(group, moved, done)


def install_encoder_prefetch(module: Any, device: Any, logger: Any = None) -> int:
    """Swap each streamed offload group's ``onload_`` of a small-host text encoder for the prefetching copy. Groups made
    resident are left alone. Returns the number of groups covered (0: unchanged, e.g. kill switch or no CUDA)."""
    if encoder_prefetch_disabled():
        return 0
    try:
        import torch

        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return 0
        from .diffusion_memory import _offload_groups

        groups = [
            g
            for g in _offload_groups(module)
            if not getattr(g, "_unsloth_resident", False)
            and getattr(g, "stream", None) is not None
            and not getattr(g, "offload_to_disk_path", None)
        ]
        if not groups:
            return 0
        pf = _EncoderPrefetcher(module, groups, device)
        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        for group in groups:

            def onload_(*_a: Any, _g: Any = group, **_k: Any) -> None:
                pf.onload(_g)

            group.onload_ = disable(onload_) if callable(disable) else onload_
        module.register_forward_pre_hook(lambda *_a, **_k: pf.begin())
        module.register_forward_hook(lambda *_a, **_k: pf.end(), always_call = True)
        setattr(module, ENCODER_PREFETCH_ATTR, pf)
        if logger is not None:
            logger.info(
                "diffusion.small_host: %s streams %d offload groups through a prefetching copy",
                type(module).__name__,
                len(groups),
            )
        return len(groups)
    except Exception as exc:  # noqa: BLE001 - keep diffusers' own onload
        if logger is not None:
            logger.warning("diffusion.small_host: encoder prefetch unavailable (%s)", exc)
        return 0
