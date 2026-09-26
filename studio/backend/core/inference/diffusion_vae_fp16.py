# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp16 VAE decode for U-Net (SDXL) pipelines whose VAE sets ``force_upcast``.

The SDXL VAE overflows fp16 in its last decoder stages (activations reach ~1.2e6), so SDXL pipelines upcast the whole
VAE to fp32 for every decode on fp16 GPUs (T4: 2.75 s vs 1.01 s at 1024). Dividing the residual stream after the
first upsampler by 2**8 (and the eps of each GroupNorm reading it by 2**16) is an identity in exact arithmetic, since
GroupNorm is scale invariant, and keeps every activation inside fp16 range. Weights are rescaled in place by a power of
two, so an fp32 decode stays exact up to the fp16 rounding of the smallest rescaled weights.

Measured on 72 real SDXL / SDXL-Turbo latents vs the fp32 decode: PSNR 61.0-63.2 dB min / 67.8-68.1 dB mean, LPIPS at
most 5e-5, at most 6 pixels per 1024x1024 frame more than 8/255 off, no non-finite frame (the stock fp16 decode is NaN
on every one; bf16 is 49.4 dB). A non-finite decode still reruns in fp32 and hands every later decode back to the
pipeline's fp32 upcast. The encode keeps its fp32 math.
"""

from __future__ import annotations

import functools
import types
from typing import Any, Optional

_SCALE_LOG2 = 8


def _scale_plan(decoder: Any) -> Optional[tuple]:
    """(convs, norms) that divide the decoder's residual stream after ``up_blocks[0]``'s upsampler.

    ``convs`` are (conv, divide_weight) pairs: a conv fed by an unscaled tensor (the entry upsampler, every post-norm
    ``conv2``) divides weight and bias; one whose input is already scaled (shortcut, later upsamplers) divides only its
    bias. ``norms`` read the scaled stream, so their eps shrinks by scale**2 and they stay exact. None when the decoder
    is not the plain GroupNorm AutoencoderKL layout this relies on (an attention or spatial norm in an up block, a time
    embedding)."""
    import torch

    up_blocks = list(getattr(decoder, "up_blocks", None) or ())
    if len(up_blocks) < 2 or not isinstance(
        getattr(decoder, "conv_norm_out", None), torch.nn.GroupNorm
    ):
        return None
    entry = list(getattr(up_blocks[0], "upsamplers", None) or ())
    if len(entry) != 1 or not isinstance(getattr(entry[0], "conv", None), torch.nn.Conv2d):
        return None
    plan = [(entry[0].conv, True)]
    norms = [decoder.conv_norm_out]
    for block in up_blocks[1:]:
        if getattr(block, "attentions", None):
            return None
        for resnet in getattr(block, "resnets", None) or ():
            if not (
                isinstance(getattr(resnet, "norm1", None), torch.nn.GroupNorm)
                and isinstance(getattr(resnet, "norm2", None), torch.nn.GroupNorm)
                and isinstance(getattr(resnet, "conv2", None), torch.nn.Conv2d)
                and getattr(resnet, "time_emb_proj", None) is None
                and getattr(resnet, "upsample", None) is None
                and getattr(resnet, "downsample", None) is None
            ):
                return None
            plan.append((resnet.conv2, True))
            norms.append(resnet.norm1)
            shortcut = getattr(resnet, "conv_shortcut", None)
            if shortcut is not None:
                if not isinstance(shortcut, torch.nn.Conv2d):
                    return None
                plan.append((shortcut, False))
        for upsampler in getattr(block, "upsamplers", None) or ():
            if not isinstance(getattr(upsampler, "conv", None), torch.nn.Conv2d):
                return None
            plan.append((upsampler.conv, False))
    return plan, norms


def enable_fp16_vae_decode(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> bool:
    """Decode a force_upcast fp16 AutoencoderKL in fp16 on a U-Net pipeline. True when engaged."""
    vae = getattr(pipe, "vae", None)
    if (
        vae is None
        or getattr(pipe, "unet", None) is None
        or getattr(target, "device", None) != "cuda"
    ):
        return False
    if getattr(vae, "_unsloth_fp16_decode", False):
        return True
    config = getattr(vae, "config", None)
    if type(vae).__name__ != "AutoencoderKL" or not getattr(config, "force_upcast", False):
        return False
    # A decode compiled before this would sit inside the check, and its eager fallback would unwrap it.
    if getattr(vae, "_unsloth_compiled_decode", False):
        return False
    try:
        import torch

        if getattr(vae, "dtype", None) is not torch.float16:
            return False
        found = _scale_plan(getattr(vae, "decoder", None))
        post_quant_conv = getattr(vae, "post_quant_conv", None)
        encode, decode = getattr(vae, "encode", None), getattr(vae, "decode", None)
        if found is None or not callable(encode) or not callable(decode):
            return False
        plan, norms = found
        scale = float(2**-_SCALE_LOG2)
        with torch.no_grad():
            for conv, divide_weight in plan:
                if divide_weight:
                    conv.weight.mul_(scale)
                if conv.bias is not None:
                    conv.bias.mul_(scale)
        for norm in norms:
            norm.eps = norm.eps * scale * scale
        decode_parts = [m for m in (post_quant_conv, vae.decoder) if m is not None]
        encode_parts = [
            m
            for m in (getattr(vae, "encoder", None), getattr(vae, "quant_conv", None))
            if m is not None
        ]
        fell_back: list = []
        # Studio's VAE decode compile lands in this slot, inside the non-finite check: a data-dependent check inside
        # the compiled region would graph-break, and a compile failure restores the slot, never unwrapping the check.
        slot = types.SimpleNamespace(decode = decode)

        def _fp32_call(fn: Any, parts: list, x: Any, *args: Any, **kwargs: Any) -> Any:
            for part in parts:
                part.to(torch.float32)
            try:
                return fn(x.to(torch.float32), *args, **kwargs)
            finally:
                for part in parts:
                    part.to(torch.float16)

        @functools.wraps(decode)
        def fp16_decode(z: Any, *args: Any, **kwargs: Any) -> Any:
            if fell_back or not torch.is_tensor(z) or z.dtype is not torch.float16:
                return slot.decode(z, *args, **kwargs)
            out = slot.decode(z, *args, **kwargs)
            sample = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
            if not torch.is_tensor(sample) or bool(torch.isfinite(sample).all()):
                return out
            if logger is not None:
                logger.warning(
                    "diffusion.vae: fp16 decode was not finite; decoding in fp32 from now on"
                )
            fell_back.append(True)
            vae.register_to_config(force_upcast = True)
            return _fp32_call(slot.decode, decode_parts, z, *args, **kwargs)

        # The pipelines upcast the whole VAE for an encode too; keep that math (encoder only) now the flag is off.
        @functools.wraps(encode)
        def fp32_encode(x: Any, *args: Any, **kwargs: Any) -> Any:
            if fell_back or not torch.is_tensor(x) or x.dtype is not torch.float16:
                return encode(x, *args, **kwargs)
            return _fp32_call(encode, encode_parts, x, *args, **kwargs)

        fp16_decode._unsloth_decode_slot = slot
        vae.decode = fp16_decode
        vae.encode = fp32_encode
        vae.register_to_config(force_upcast = False)
        vae._unsloth_fp16_decode = True
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.vae: fp16 decode not enabled: %s", exc)
        return False
