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
    return {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}.get(name, 2)


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
        refuse = None
        if not host_ram_check_disabled():
            refuse = _refusal(dense, reserve, host_available_mib, "LoRA adapters need the dense load")
        return SmallHostDecision(
            False,
            "LoRA adapters need the dense denoiser",
            host_available_mib,
            host_total_mib,
            dense,
            route,
            refuse = refuse,
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
            w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
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
        (name, child)
        for name, child in module.named_modules()
        if type(child) is torch.nn.Linear
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
                if p.is_floating_point() and p.dtype != compute_dtype:
                    p.data = p.data.to(keep, compute_dtype)
                elif p.device != keep:
                    p.data = p.data.to(keep)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is not None and b.is_floating_point() and b.dtype != compute_dtype:
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
                dt = want if p.is_floating_point() else p.dtype
                p.data = p.data.to(dev).to(dt)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is None:
                    continue
                dt = want if b.is_floating_point() else b.dtype
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

    def _want(mname: str) -> Any:
        return torch.float32 if keep and any(k in mname for k in keep) else compute_dtype

    def _convert(sub: Any, want: Any) -> None:
        for pname, p in list(sub.named_parameters(recurse = False)):
            if p.is_floating_point() and p.dtype != want:
                p.data = p.data.to(want)
        for bname, b in list(sub.named_buffers(recurse = False)):
            if b is not None and b.is_floating_point() and b.dtype != want:
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
                and want is compute_dtype
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
