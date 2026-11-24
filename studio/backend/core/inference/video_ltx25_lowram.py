# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 on a GPU without native bfloat16 (T4, V100) whose host cannot hold the float32 load: the low-RAM route.

The load reads what a bfloat16 card reads (the hosted int8 DiT as native Linears, the bf16 companions, the hosted int8
Gemma 4 in the encoder worker) and runs it in float32: Linear weights stay at their stored bf16 and are cast as each
layer runs, everything else converts at load. ``UNSLOTH_DIFFUSION_SMALL_HOST=0`` keeps the float32 load (and its
refusal); ``=1`` takes the route on any fp16-only card. Must import nothing heavy at module level."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

# Companions converted to float32 at load (convolutions, no per-layer cast): video VAE, latent upsampler, audio VAE +
# vocoder, decimal GB at bf16; their float32 copies are anonymous memory. The 6.34 GB connectors are Linears and stay
# bf16, memory-mapped, read only while a prompt is encoded.
_WIDENED_COMPANIONS_GB = 1.45 + 1.0 + 0.37
_CONNECTORS_GB = 6.34
# Host GB beside the weights: two torch + CUDA processes (Studio and the encoder worker), the offload staging and the
# page cache the kernel keeps. Calibrated on a Colab T4 High-RAM host: 42.8 GiB peak in use (MemTotal - MemAvailable)
# rendering 768x512x121, which this prices at 42.7 GiB.
_RUNTIME_GB = 10.5
_FALLBACK_INT8_DIT_GB = 25.11
# Turing: the floor of the fp16-card int8 path (diffusion_transformer_quant._FP16_CARD_INT8_CAPS).
_MIN_CAPABILITY = (7, 5)


@dataclasses.dataclass(frozen = True)
class Ltx25LowRam:
    engaged: bool
    reason: str
    need_mib: int = 0
    total_mib: Optional[int] = None
    fp32_need_mib: int = 0
    refusal: Optional[str] = None


def lowram_ineligible(fam: Any, target: Any, *, model_kind: str) -> Optional[str]:
    """Why the route cannot serve this load whatever the host holds, else None; known before any download."""
    try:
        import torch

        from .diffusion_fp16_guard import fp16_promotes_to_fp32
        from .diffusion_small_host import small_host_disabled
        from .diffusion_transformer_quant import _capability, torch_is_rocm

        if getattr(fam, "name", None) != "ltx-2.5":
            return "not LTX-2.5"
        if model_kind != "pipeline":
            return f"a {model_kind} load"
        if getattr(target, "device", None) != "cuda":
            return "not a CUDA load"
        if getattr(target, "dtype", None) is not torch.float16:
            return "the GPU runs bfloat16"
        if not fp16_promotes_to_fp32(fam):
            return "float16 is not promoted"
        if torch_is_rocm():
            return "ROCm (the hosted int8 DiT reads through the NVIDIA build)"
        cap = _capability(getattr(target, "ordinal", None))
        if cap is None or cap < _MIN_CAPABILITY:
            # The W8A8 GEMM (cuBLASLt ``torch._int_mm``) needs int8 tensor cores, as the fp16-card int8 path does.
            return f"compute capability {cap} has no int8 tensor cores (Turing or newer needed)"
        if small_host_disabled():
            return "UNSLOTH_DIFFUSION_SMALL_HOST=0"
        return None
    except Exception as exc:  # noqa: BLE001 -- unanswerable: the float32 load, as before
        return f"probe failed: {type(exc).__name__}"


def _int8_dit_gb(fam: Any) -> float:
    for scheme, gb in getattr(fam, "prequant_resident_gb_by_scheme", None) or ():
        if scheme == "int8":
            return float(gb)
    return _FALLBACK_INT8_DIT_GB


def lowram_need_gb(
    fam: Any,
    text_encoder_quant: Optional[str],
    base: Optional[str],
    *,
    hosted_dit: bool = True,
) -> Optional[float]:
    """Host GB the route needs: what it holds as anonymous memory (float32 copies of the convolution companions and of
    the encoder's dense tensors, the two processes' runtime) plus the larger of its two memory-mapped working sets,
    the int8 DiT file while denoising or the encoder's int8 codes and the connectors while encoding. A mapped set
    that is idle can be paged out and read back from its file."""
    sizes = tuple(float(x) for x in (getattr(fam, "bf16_components_gb", None) or ()))
    if len(sizes) != 3:
        return None
    from .video_ltx25 import LTX25_TE_BF16_GB, LTX25_TE_INT8_GB, ltx25_te_host_gb

    te_gb = ltx25_te_host_gb(
        text_encoder_quant, base or getattr(fam, "base_repo", None), bf16_gb = sizes[1], scale = 2.0
    )
    if te_gb < sizes[1] * 2.0:
        # hosted int8 encoder: codes stay mapped, the dense tensors widen to float32
        dense = max(0.0, 2.0 * LTX25_TE_INT8_GB - LTX25_TE_BF16_GB)
        te_mapped, te_anon = LTX25_TE_INT8_GB - dense, dense * 2.0
    else:
        te_mapped, te_anon = 0.0, te_gb
    anon = _WIDENED_COMPANIONS_GB * 2.0 + te_anon + _RUNTIME_GB
    # the hosted int8 file, else the dense bf16 shards (explicit "none", LoRAs, the hosted read off)
    dit = _int8_dit_gb(fam) if hosted_dit else sizes[0]
    return anon + max(dit, te_mapped + _CONNECTORS_GB)


def fp32_need(
    fam: Any, text_encoder_quant: Optional[str], base: Optional[str]
) -> Optional[tuple[int, float]]:
    """(MiB, encoder GB) of the float32 load: DiT and companions widened, plus the encoder worker at float32."""
    sizes = tuple(float(x) for x in (getattr(fam, "bf16_components_gb", None) or ()))
    if len(sizes) != 3:
        return None
    from .video_ltx25 import ltx25_te_host_gb

    te_gb = ltx25_te_host_gb(
        text_encoder_quant, base or getattr(fam, "base_repo", None), bf16_gb = sizes[1], scale = 2.0
    )
    return int(((sizes[0] + sizes[2]) * 2.0 + te_gb) * 1e9 / 1048576), te_gb


def decide_lowram(
    fam: Any,
    target: Any,
    *,
    model_kind: str,
    text_encoder_quant: Optional[str] = None,
    base: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    speed_mode: Optional[str] = None,
    loras_active: bool = False,
    resolve_te: bool = True,
) -> Ltx25LowRam:
    """Whether this LTX-2.5 load takes the low-RAM route; ``refusal`` when even the route cannot fit. Never raises.

    Priced at what the load will read: the encoder the loader resolves (``resolve_te``: from the raw request, as the
    loader does, Speed "off" keeping the dense one) and the hosted int8 DiT only where the loader reads it."""
    skip = lowram_ineligible(fam, target, model_kind = model_kind)
    if skip:
        return Ltx25LowRam(False, skip)
    base = base or getattr(fam, "base_repo", None)
    try:
        from .diffusion_small_host import host_ram_check_disabled, host_ram_mib, small_host_forced
        from .video import _ltx25_lowram_hosted_dit, video_te_quant_request
        from .video_ltx25 import ltx25_resolve_te_request

        if resolve_te:
            text_encoder_quant = video_te_quant_request(
                fam, text_encoder_quant, base = base, speed_mode = speed_mode
            )[0]
            text_encoder_quant = ltx25_resolve_te_request(text_encoder_quant, base)
        hosted = _ltx25_lowram_hosted_dit(
            fam,
            storage_target(target),
            transformer_quant = transformer_quant,
            loras_active = loras_active,
            base = base,
            speed_mode = speed_mode,
        )
        fp32 = fp32_need(fam, text_encoder_quant, base)
        route_gb = lowram_need_gb(fam, text_encoder_quant, base, hosted_dit = hosted)
        if fp32 is None or route_gb is None:
            return Ltx25LowRam(False, "no component sizes")
        need = int(route_gb * 1e9 / 1048576)
        total = host_ram_mib()[0]
        forced = small_host_forced()
        if not forced and (not total or fp32[0] <= int(total)):
            return Ltx25LowRam(
                False,
                "host RAM unreadable" if not total else "the float32 load fits host RAM",
                need,
                total,
                fp32[0],
            )
        if total and need > int(total) and not host_ram_check_disabled():
            label = getattr(fam, "label", None) or "LTX-2.5"
            refusal = (
                f"Not enough system RAM to load this model: {label} on this GPU (no native bfloat16) needs about "
                f"{need / 1024:.1f} GB of host memory even on the low-memory route (float32 load: "
                f"{fp32[0] / 1024:.1f} GB), and {int(total) / 1024:.1f} GB is available. Load the {label} GGUF "
                f"build instead, use a bfloat16 GPU (NVIDIA Ampere or newer), or set "
                f"UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0 to try anyway."
            )
            return Ltx25LowRam(
                False, "even the low-memory route does not fit", need, total, fp32[0], refusal
            )
        why = (
            "UNSLOTH_DIFFUSION_SMALL_HOST=1"
            if forced
            else f"float32 load needs ~{fp32[0]} MiB of host RAM, {total} MiB total"
        )
        return Ltx25LowRam(True, why, need, total, fp32[0])
    except Exception as exc:  # noqa: BLE001 -- unanswerable: the float32 load, as before
        return Ltx25LowRam(False, f"probe failed: {type(exc).__name__}")


def storage_target(target: Any) -> Any:
    """``target`` reading as a bfloat16 card: what the route plans and loads with (it computes in float32)."""
    import torch
    try:
        return dataclasses.replace(target, dtype = torch.bfloat16)
    except TypeError:
        import copy

        out = copy.copy(target)
        out.dtype = torch.bfloat16
        return out


def eager_target(target: Any) -> Any:
    """The real card with torch.compile off: the route's per-layer casts are Python hooks, and its float32 compute
    is unmeasured under compile. ``compile_eligible`` already declines the fp16-incompatible family on an fp16 card;
    this keeps that true if the family flag or the card ever change."""
    try:
        return dataclasses.replace(target, supports_default_torch_compile = False)
    except TypeError:
        import copy

        out = copy.copy(target)
        out.supports_default_torch_compile = False
        return out


def compute_components(pipe: Any) -> list[tuple[str, Any]]:
    """(name, module) of every LTX-2.5 component this process runs (the encoder runs in its worker)."""
    import torch

    out = []
    for name in ("transformer", "connectors", "vae", "audio_vae", "vocoder"):
        mod = getattr(pipe, name, None)
        if isinstance(mod, torch.nn.Module):
            out.append((name, mod))
    upsampler = getattr(pipe, "_unsloth_ltx25_upsampler", None)
    if isinstance(upsampler, torch.nn.Module):
        out.append(("latent_upsampler", upsampler))
    return out


_CAST_HOOK = "layerwise_casting"


def _storage_cast_hook_class():
    """A diffusers layerwise-casting hook (``module.dtype`` reports its compute dtype) that casts this module's bf16
    parameters for the call only and puts the stored tensors back in a ``finally``: an exception never leaves a
    float32 weight behind, and a memory-mapped weight keeps its mapping instead of being replaced by a copy."""
    from diffusers.hooks.layerwise_casting import LayerwiseCastingHook

    class StorageCastHook(LayerwiseCastingHook):
        def initialize_hook(self, module: Any) -> Any:
            return module  # already at its storage dtype; no copy

        def pre_forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
            return args, kwargs

        def post_forward(self, module: Any, output: Any) -> Any:
            return output

        def new_forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
            stored = [
                (p, p.data)
                for p in module.parameters(recurse = False)
                if p.dtype == self.storage_dtype
            ]
            try:
                for p, data in stored:
                    p.data = data.to(self.compute_dtype, non_blocking = self.non_blocking)
                return self.fn_ref.original_forward(*args, **kwargs)
            finally:
                for p, data in stored:
                    p.data = data

    return StorageCastHook


def upcast_compute_(pipe: Any, compute_dtype: Any) -> dict[str, int]:
    """Run the bf16-stored pipeline in ``compute_dtype``. Linear and convolution weights keep bf16 and are cast for
    each call (``module.dtype`` reports ``compute_dtype``); int8 codes keep their storage and are never read here
    (a native int8 Linear's ``weight`` property dequantizes); every other bf16 tensor (norms, embeddings, loose
    parameters, buffers) converts now. Returns the MiB per component left at bf16. The route stays eager: the casts
    are Python hooks, and ``compile_eligible`` declines an fp16-incompatible family on the fp16 card."""
    import torch
    from diffusers.hooks import HookRegistry

    nn = torch.nn
    streamable = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    hook_cls = _storage_cast_hook_class()
    out: dict[str, int] = {}
    with torch.no_grad():
        for name, module in compute_components(pipe):
            kept = 0
            for sub in module.modules():
                # the type first: never touch ``weight`` on anything else
                if isinstance(sub, streamable):
                    own = [t for t in sub.parameters(recurse = False) if t.dtype == torch.bfloat16]
                    if any(t is sub._parameters.get("weight") for t in own):
                        kept += sum(t.numel() * t.element_size() for t in own)
                        registry = HookRegistry.check_if_exists_or_initialize(sub)
                        registry.register_hook(
                            hook_cls(torch.bfloat16, compute_dtype, False), _CAST_HOOK
                        )
                        continue
                for _, p in list(sub.named_parameters(recurse = False)):
                    if p.dtype == torch.bfloat16:
                        p.data = p.data.to(compute_dtype)
                for bname, b in list(sub.named_buffers(recurse = False)):
                    if b is not None and b.dtype == torch.bfloat16:
                        sub._buffers[bname] = b.to(compute_dtype)
            out[name] = kept // (1024 * 1024)
    return out
