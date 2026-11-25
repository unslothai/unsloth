# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""16-bit VAE decode safety net: a non-finite decode from finite latents is redone (and stays) fp32; GPUs without
tensor cores decode/encode in fp32 from the start. Weights stay stored in 16 bits.
UNSLOTH_DIFFUSION_VAE_FP32: auto (default) | 1 = always fp32 | 0 = off (stock 16-bit decode, no retry)."""

from __future__ import annotations

import functools
import os
import re
import types
from typing import Any, Optional

from .diffusion_vae_upcast import fp32_keeping_storage

VAE_FP32_ENV = "UNSLOTH_DIFFUSION_VAE_FP32"
MODE_OFF = "off"
MODE_AUTO = "auto"
MODE_ALWAYS = "always"
_MARK = "_unsloth_finite_decode"
_ON = ("1", "on", "true", "yes", "always", "fp32", "force")
_OFF = ("0", "off", "false", "no", "never")

NONFINITE_LATENTS_MESSAGE = (
    "The denoiser produced NaN or infinite latents before the VAE decode, so the image cannot be decoded. The "
    "transformer overflowed at this resolution and precision on this GPU; try a smaller resolution or another quant "
    "of this model."
)

# TU116/TU117 lack tensor cores but report capability 7.5 like RTX 20xx / T4, so only the name tells them apart.
_NO_TENSOR_CORE_NAME = re.compile(
    r"\bGTX\s*16\d\d\b|\bMX\s*[45]\d0\b|\b(?:Quadro\s+)?T(?:400|500|550|600|1000|1200|2000)\b",
    re.IGNORECASE,
)

STATS = {"retries": 0, "fp32_decodes": 0, "nonfinite_latents": 0}


def vae_fp32_mode() -> str:
    raw = (os.environ.get(VAE_FP32_ENV) or "").strip().lower()
    if raw in _OFF:
        return MODE_OFF
    if raw in _ON:
        return MODE_ALWAYS
    return MODE_AUTO


def lacks_tensor_cores(name: Optional[str], capability: Optional[tuple]) -> bool:
    try:
        major, minor = int(capability[0]), int(capability[1])
    except Exception:  # noqa: BLE001 - unknown capability: keep the 16-bit decode (the retry still guards it)
        return False
    if major < 7:
        return True
    if (major, minor) != (7, 5):
        return False
    return bool(_NO_TENSOR_CORE_NAME.search(str(name or "")))


def target_lacks_tensor_cores(target: Any) -> bool:
    if getattr(target, "device", None) != "cuda" or getattr(target, "backend", "cuda") != "cuda":
        return False
    try:
        import torch

        ordinal = getattr(target, "ordinal", None)
        index = torch.cuda.current_device() if ordinal is None else int(ordinal)
        return lacks_tensor_cores(
            torch.cuda.get_device_name(index), tuple(torch.cuda.get_device_capability(index))
        )
    except Exception:  # noqa: BLE001
        return False


def prefers_fp32_decode(target: Any) -> bool:
    mode = vae_fp32_mode()
    return mode == MODE_ALWAYS or (mode == MODE_AUTO and target_lacks_tensor_cores(target))


def _first_dtype(module: Any) -> Any:
    try:
        return next(iter(module.parameters())).dtype
    except Exception:  # noqa: BLE001
        return None


def _cast_floats(
    obj: Any,
    dtype: Any,
    depth: int = 0,
) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if depth > 2:
        return obj
    if isinstance(obj, (tuple, list)):
        return type(obj)(_cast_floats(v, dtype, depth + 1) for v in obj)
    if isinstance(
        obj, dict
    ):  # diffusers BaseOutput is an OrderedDict that mirrors its fields as attributes
        for k in list(obj.keys()):
            v = _cast_floats(obj[k], dtype, depth + 1)
            obj[k] = v
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj
    fields = getattr(obj, "__dict__", None)
    if isinstance(fields, dict):
        for k, v in list(fields.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                fields[k] = v.to(dtype)
    return obj


def _sample_of(out: Any) -> Any:
    if isinstance(out, tuple):
        return out[0] if out else None
    return getattr(out, "sample", out)


def install_finite_decode(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> Optional[str]:
    """Returns the mode engaged ("fp32" or "retry"), None when not installed."""
    mode = vae_fp32_mode()
    vae = getattr(pipe, "vae", None)
    if mode == MODE_OFF or vae is None:
        return None
    if getattr(vae, _MARK, None):
        return getattr(vae, _MARK)
    # The SDXL fp16 decode and the video half decode carry their own non-finite fallback.
    if getattr(vae, "_unsloth_fp16_decode", False) or getattr(vae, "_unsloth_half_decode", False):
        return None
    decode, encode = getattr(vae, "decode", None), getattr(vae, "encode", None)
    if not callable(decode):
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    half = (torch.float16, torch.bfloat16)
    always = prefers_fp32_decode(target)
    decode_parts = [
        m
        for m in (getattr(vae, "post_quant_conv", None), getattr(vae, "decoder", None))
        if isinstance(m, torch.nn.Module)
    ] or None
    encode_parts = [
        m
        for m in (getattr(vae, "encoder", None), getattr(vae, "quant_conv", None))
        if isinstance(m, torch.nn.Module)
    ] or None
    state = {"fp32": always}
    # Decode compile goes in this slot: the check stays eager (no graph break) and outlives a fallback.
    slot = types.SimpleNamespace(decode = decode)

    def _fp32_decode(z: Any, *args: Any, **kwargs: Any) -> Any:
        STATS["fp32_decodes"] += 1
        with fp32_keeping_storage(vae, decode_parts):
            return slot.decode(z.to(torch.float32), *args, **kwargs)

    @functools.wraps(decode)
    def finite_decode(z: Any, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_tensor(z) or not z.is_floating_point() or z.device.type == "meta":
            return slot.decode(z, *args, **kwargs)
        # Read per call: the VAE may be cast to its 16-bit dtype after this layer is installed.
        if _first_dtype(vae) not in half:
            return slot.decode(z, *args, **kwargs)
        if state["fp32"]:
            return _fp32_decode(z, *args, **kwargs)
        out = slot.decode(z, *args, **kwargs)
        sample = _sample_of(out)
        if not torch.is_tensor(sample) or bool(torch.isfinite(sample).all()):
            return out
        if not bool(torch.isfinite(z).all()):
            STATS["nonfinite_latents"] += 1
            raise RuntimeError(NONFINITE_LATENTS_MESSAGE)
        del out, sample
        STATS["retries"] += 1
        if logger is not None:
            logger.warning(
                "diffusion.vae: %s decode was not finite; decoding again in fp32 from the same latents, and in "
                "fp32 from now on (%s=1 forces it from the start)",
                str(z.dtype).replace("torch.", ""),
                VAE_FP32_ENV,
            )
        state["fp32"] = True
        return _fp32_decode(z, *args, **kwargs)

    finite_decode._unsloth_decode_slot = slot
    vae.decode = finite_decode
    if always and callable(encode) and encode_parts is not None:

        @functools.wraps(encode)
        def fp32_encode(x: Any, *args: Any, **kwargs: Any) -> Any:
            if not torch.is_tensor(x) or not x.is_floating_point() or _first_dtype(vae) not in half:
                return encode(x, *args, **kwargs)
            dtype = _first_dtype(vae)
            with fp32_keeping_storage(vae, encode_parts):
                out = encode(x.to(torch.float32), *args, **kwargs)
            # img2img feeds the latents to a 16-bit denoiser.
            return _cast_floats(out, dtype)

        vae.encode = fp32_encode
    engaged = "fp32" if always else "retry"
    setattr(vae, _MARK, engaged)
    if logger is not None and always:
        logger.info(
            "diffusion.vae: decoding in fp32 (%s)",
            "forced by %s" % VAE_FP32_ENV
            if mode == MODE_ALWAYS
            else "this GPU has no tensor cores",
        )
    return engaged


__all__ = [
    "NONFINITE_LATENTS_MESSAGE",
    "STATS",
    "VAE_FP32_ENV",
    "install_finite_decode",
    "lacks_tensor_cores",
    "prefers_fp32_decode",
    "target_lacks_tensor_cores",
    "vae_fp32_mode",
]
