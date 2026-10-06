# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host load route: a bf16 repo loaded at fp16/fp32 materialises every converted weight in anonymous host RAM
(FLUX.1 at fp16: 35 GB, OOM-killed on a 31 GB host before the planner runs). Engaged only when that dense load would
not fit: big components load at their stored bf16 (memory-mapped), encoders stream with layerwise casting, a streamed
denoiser is stored as int8 per-channel weights. ``UNSLOTH_DIFFUSION_SMALL_HOST=0|1`` disables / forces it;
``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` skips the pre-load refusal.
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
_STREAM_MIN_MIB = 512
_HOST_RESERVE_MIN_MIB = 3072
_HOST_RESERVE_FRACTION = 0.10
# fp16 T5 keeps ``wo`` in fp32 (``_keep_in_fp32_modules``): 9.5 GB stored, 11.0 GB loaded.
_CONVERT_MARGIN = 1.15
_INT8_MIN_ELEMENTS = 1 << 22
# DiT denoisers the route stores as int8; a UNet (mostly convs) converts dense like any other non-encoder component
INT8_DENOISER_NAMES = ("transformer", "transformer_2", "unconditional_transformer")


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def small_host_disabled() -> bool:
    return _env(SMALL_HOST_ENV) in ("0", "off", "false", "no")


def small_host_forced() -> bool:
    return _env(SMALL_HOST_ENV) in ("1", "on", "true", "yes", "force")


def host_ram_check_disabled() -> bool:
    return _env(HOST_RAM_CHECK_ENV) in ("0", "off", "false", "no")


_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _safetensors_bytes_by_dtype(path: Path, keep: tuple[str, ...] = ()) -> dict[str, int]:
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
            if keep and any(f".{k}." in f".{name}." for k in keep):
                out["kept"] = out.get("kept", 0) + int(end) - int(start)
        except Exception:  # noqa: BLE001
            continue
    return out


@dataclass(frozen = True)
class StoredComponent:
    name: str
    mib: int
    dtype: str  # dominant stored float dtype ("bfloat16", "float16", "float32", ...)
    # stored MiB of ``_keep_in_fp32_modules`` tensors: they convert to fp32 on the host even on the route
    kept_fp32_mib: int = 0


def _keep_fp32_from_config(sub: Path) -> tuple[str, ...]:
    try:
        arch = (
            json.loads((sub / "config.json").read_text(encoding = "utf-8")).get("architectures")
            or [None]
        )[0]
        if not arch:
            return ()
        import transformers

        return _keep_fp32_patterns(getattr(transformers, str(arch), None))
    except Exception:  # noqa: BLE001
        return ()


def _weight_files(sub: Path) -> list[Path]:
    files = sorted(sub.glob("*.safetensors"))
    # ``model.fp16.safetensors`` next to ``model.safetensors`` is a variant from_pretrained does not read
    plain = [f for f in files if "." not in f.name[: -len(".safetensors")]]
    return plain or files


def stored_components(snapshot: Any) -> dict[str, StoredComponent]:
    """Per component: stored MiB and dominant dtype, from safetensors headers only."""
    out: dict[str, StoredComponent] = {}
    try:
        root = Path(snapshot)
        if not root.is_dir():
            return out
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            totals: dict[str, int] = {}
            keep = _keep_fp32_from_config(sub) if sub.name.startswith("text_encoder") else ()
            for f in _weight_files(sub):
                for k, v in _safetensors_bytes_by_dtype(f, keep).items():
                    totals[k] = totals.get(k, 0) + v
            kept = totals.pop("kept", 0)
            if not totals:
                continue
            dtype = max(totals.items(), key = lambda kv: kv[1])[0]
            out[sub.name] = StoredComponent(
                sub.name, -(-sum(totals.values()) // _MIB), dtype, -(-kept // _MIB)
            )
    except Exception:  # noqa: BLE001 - no claim
        return {}
    return out


def resolve_snapshot_dir(repo_or_dir: Any, cache_dir: Any = None) -> Optional[Path]:
    """Local diffusers dir for a repo id (cache only) or path."""
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
    from .diffusion_memory import _available_system_memory_mib, _host_ram_capacity_mib
    return _host_ram_capacity_mib(), _available_system_memory_mib()


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
    # component -> stored dtype it loads at (memory-mapped)
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
    """Whether this pipeline load takes the small-host route. Only CUDA loads converting large bf16 components
    (fp16 / fp32 compute) change; same-dtype loads already stay memory-mapped."""
    if small_host_disabled():
        return SmallHostDecision(False, f"{SMALL_HOST_ENV}=0")
    if str(device) != "cuda":
        return SmallHostDecision(False, "not a CUDA load")
    compute = _dtype_name(compute_dtype)
    if compute not in ("float16", "float32"):
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
    # encoders stay memory-mapped except fp32-kept layers; a DiT lands as int8
    route = sum(
        c.mib // 2
        if n in INT8_DENOISER_NAMES
        else c.kept_fp32_mib * 4 // _itemsize(c.dtype)
        if n.startswith("text_encoder")
        else c.mib * _itemsize(compute) // _itemsize(c.dtype)
        for n, c in converted.items()
    )
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
        # the int8 denoiser cannot carry adapters
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


INT8_ACT_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_INT8_ACT"
# Measured on a T4 (LPIPS inside the fp32-vs-bf16 spread); qwen-image-edit shares the DiT but is unmeasured.
INT8_ACT_FAMILIES = frozenset({"qwen-image"})
# torch._int_mm: rows M > 16, K and N multiples of 8.
_INT8_ACT_MIN_ROWS = 17
_INT8_ACT_ALIGN = 8
_INT8_ACT_DEVICE_OK: dict[int, bool] = {}
_INT8_ACT_COUNTS = {"int8_act": 0, "dequant": 0}


def int8_act_disabled() -> bool:
    return _env(INT8_ACT_ENV) in ("0", "off", "false", "no")


def int8_act_family(fam: Any) -> bool:
    """Whether ``fam``'s int8-stored denoiser may run W8A8 (still gated per device and per call)."""
    if int8_act_disabled():
        return False
    name = str(getattr(fam, "name", fam) or "").strip().lower()
    return name in INT8_ACT_FAMILIES


def int8_act_counts() -> dict[str, int]:
    return dict(_INT8_ACT_COUNTS)


def int8_act_device_ok(device: Any) -> bool:
    """sm_75 CUDA only (fp32-promoted, int8 tensor cores; on sm_80+ bf16 eager W8A8 does not win; ROCm capabilities
    differ), and ``torch._int_mm`` must return the exact integer product there."""
    import torch

    try:
        dev = torch.device(device)
        if dev.type != "cuda" or getattr(torch.version, "hip", None):
            return False
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        return False
    hit = _INT8_ACT_DEVICE_OK.get(idx)
    if hit is not None:
        return hit
    ok = False
    try:
        if tuple(torch.cuda.get_device_capability(idx)) == (7, 5) and hasattr(torch, "_int_mm"):
            g = torch.Generator(device = "cpu").manual_seed(0)
            a = torch.randint(-127, 128, (32, 64), generator = g, dtype = torch.int8)
            b = torch.randint(-127, 128, (48, 64), generator = g, dtype = torch.int8)
            want = a.to(torch.int64) @ b.to(torch.int64).t()
            got = torch._int_mm(a.to(dev), b.to(dev).t())
            ok = bool(torch.equal(got.cpu().to(torch.int64), want))
    except Exception:  # noqa: BLE001 - no usable int8 GEMM: keep the dequantised path
        ok = False
    _INT8_ACT_DEVICE_OK[idx] = ok
    return ok


def _int8_linear_class():
    import torch
    import torch.nn.functional as F

    class Int8WeightLinear(torch.nn.Module):
        """Linear with int8 per-output-channel weights, dequantised to the input dtype for each forward.

        ``act_int8``: eligible float32 CUDA calls quantise activations per row and run ``torch._int_mm`` instead."""

        def __init__(self, qweight, scale, bias, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.qweight = torch.nn.Parameter(qweight, requires_grad = False)
            self.scale = torch.nn.Parameter(scale, requires_grad = False)
            self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad = False)
            self.act_int8 = False

        def _int8_act_ok(self, x) -> bool:
            if not self.act_int8 or x.dtype is not torch.float32 or not x.is_cuda:
                return False
            if self.in_features % _INT8_ACT_ALIGN or self.out_features % _INT8_ACT_ALIGN:
                return False
            if x.numel() // max(1, self.in_features) < _INT8_ACT_MIN_ROWS:
                return False
            return int8_act_device_ok(x.device)

        def _forward_int8_act(self, x):
            x2 = x.reshape(-1, self.in_features)
            xs = torch.linalg.vector_norm(x2, ord = float("inf"), dim = 1, keepdim = True)
            xs = xs.clamp_min_(1e-12).div_(127.0)
            xq = torch.div(x2, xs).round_().clamp_(-127, 127).to(torch.int8)
            acc = torch._int_mm(xq, self.qweight.t())
            y = torch.mul(acc, xs).mul_(self.scale.reshape(1, -1).to(torch.float32))
            if self.bias is not None:
                y.add_(self.bias.to(torch.float32))
            return y.reshape(*x.shape[:-1], self.out_features)

        def forward(self, x):
            if self._int8_act_ok(x):
                _INT8_ACT_COUNTS["int8_act"] += 1
                return self._forward_int8_act(x)
            if self.act_int8:
                _INT8_ACT_COUNTS["dequant"] += 1
            scale = self.scale
            if scale.dtype == x.dtype:
                # int8 * float promotes: cast + scale in one pass, bit-identical (int8 -> fp16/fp32 is exact).
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
    act_int8: bool = False,
) -> dict[str, int]:
    """Replace every large ``nn.Linear`` with per-row int8 weights in place, one Linear at a time (quantised on
    ``work_device``, stored on ``keep_device``); everything else casts to ``compute_dtype``. ``act_int8`` lets the
    new Linears run W8A8 where ``Int8WeightLinear`` allows it."""
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
            new.act_int8 = bool(act_int8)
            del w, q, scale
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, new)
        stats["linears"] += 1
        stats["int8_bytes"] += new.qweight.numel()
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
    """Move ``module`` to ``device`` one tensor at a time, cast as the dense load would (fp32 for kept modules)."""
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
    """Layerwise-cast a memory-mapped bf16 encoder: Linear / Embedding weights keep their stored dtype, everything
    else converts now, including the owner of the first float parameter (``module.dtype`` sets the prompt-embedding
    dtype). Returns the MiB left memory-mapped."""
    import torch
    from diffusers.hooks import apply_layerwise_casting

    keep = _keep_fp32_patterns(module)
    streamable = (torch.nn.Linear, torch.nn.Embedding)

    def _kept(mname: str) -> bool:
        return bool(keep) and any(k in mname for k in keep)

    def _want(mname: str) -> Any:
        return torch.float32 if _kept(mname) else compute_dtype

    def _convert(sub: Any, want: Any) -> None:
        # fp32 buffers (RoPE ``inv_freq``) stay fp32, as in the dense load
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
            # kept-fp32 Linears convert now: T5 casts its input to ``wo.weight.dtype`` before the hook runs
            if (
                type(sub) in streamable
                and not _kept(mname)
                and mname != first_owner
                and sub.weight.dtype != want
            ):
                storage = sub.weight.dtype
                streamed += sub.weight.numel() * sub.weight.element_size()
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


ENCODER_PREFETCH_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_PREFETCH"
ENCODER_PREFETCH_ATTR = "_unsloth_encoder_prefetch"
# Pinned staging ring: the only host bytes this adds.
_STAGE_SLOT_BYTES = 32 * _MIB
_STAGE_SLOTS = 4
# Device bytes copied ahead: a few groups (T5-XXL MLP weight 80 MiB, Qwen2.5-VL 130 MiB).
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
    """Copies a memory-mapped encoder's offload groups ahead of its forward, in the previous forward's order, on a
    worker thread through a pinned ring (diffusers pins each group on the calling thread). Off-order groups and worker
    errors fall back to a synchronous copy."""

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

    def _side_stream(self) -> Any:
        import torch
        if self.stream is None:
            self.stream = torch.cuda.Stream(device = self.device)
        return self.stream

    def _ensure_stage(self) -> None:
        import torch
        if not self.stage:
            # inference_mode is thread-local: the worker cannot write inference tensors made on the forward thread.
            with torch.inference_mode(False):
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
                    with torch.cuda.stream(compute):
                        moved.append(
                            src.to(
                                self.device,
                                non_blocking = src.device.type != "cpu" or src.is_pinned(),
                            )
                        )
                    continue
                nbytes += n
                # Compute-stream pool (a side-stream block stays cached away from the denoise); the copy waits for
                # work queued there, covering the last user of a freed block.
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
                # A dropped entry (halt, exception) frees dst before the copy lands.
                dst.record_stream(stream)
                moved.append(dst)
            done = torch.cuda.Event()
            done.record(stream)
        return moved, done, nbytes

    def _attach(self, group: Any, moved: list, done: Any) -> None:
        import torch

        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(done)
        if getattr(group, "stream", None) is not None:
            # diffusers fences its own-stream prefetch only in the next group's onload_, which this replaces.
            cur.wait_stream(group.stream)
        for (t, _src), dev in zip(_group_tensors(group), moved):
            t.data = dev
            dev.record_stream(cur)

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
        self.compute = (
            torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        )
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


def install_encoder_prefetch(
    module: Any,
    device: Any,
    logger: Any = None,
) -> int:
    """Swap each streamed offload group's ``onload_`` of a small-host text encoder for the prefetching copy. Groups made
    resident are left alone. Returns the number of groups covered (0: unchanged, e.g. kill switch or no CUDA)."""
    if encoder_prefetch_disabled():
        return 0
    try:
        import torch

        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return 0
        from .diffusion_memory import GROUP_OFFLOAD_PIN_ENV, _offload_groups

        if _env(GROUP_OFFLOAD_PIN_ENV) in ("0", "off", "false", "no"):
            return 0  # the ring is pinned memory: the user's "pin nothing" override wins

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
        pf._ensure_stage()  # a pinned allocation failing here keeps diffusers' onload, not a failed render
        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        for group in groups:

            def onload_(
                *_a: Any,
                _g: Any = group,
                **_k: Any,
            ) -> None:
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
