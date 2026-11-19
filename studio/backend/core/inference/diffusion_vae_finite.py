# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""16-bit image VAE decode safety net. A decode that comes back NaN / Inf from finite latents is decoded again in
fp32 from the same latents, and every later decode of that VAE stays fp32; non-finite latents raise a message that
names the denoiser instead. GPUs without tensor cores (GTX 16xx, MX450 / MX550, TU117 Quadro / T-series, pre-Volta)
decode and encode in fp32 from the start: fp16 convolutions there are widely reported to produce NaN / black images.
The weights stay stored in 16 bits (cast per call), so memory placement and offload are unchanged.
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

# Turing parts built without tensor cores: TU116 (GTX 1650 Super / 1660 / Ti / Super) and TU117 (GTX 1630 / 1650 /
# Ti, MX450, MX550, Quadro T400 / T600 / T1000 / T1200 / T2000, T500 / T550 / T600 / T1000 laptop). They report
# compute capability 7.5 like the RTX 20xx and the Tesla T4, which do have tensor cores, so only the name tells them
# apart. Pre-Volta (capability below 7.0) has none either.
_NO_TENSOR_CORE_NAME = re.compile(
    r"\bGTX\s*16\d\d\b|\bMX\s*[45]\d0\b|\b(?:Quadro\s+)?T(?:400|500|550|600|1000|1200|2000)\b",
    re.IGNORECASE,
)

# Engagement counters (tests and evidence): renders whose 16-bit decode was retried, decodes run in fp32.
STATS = {"retries": 0, "fp32_decodes": 0, "nonfinite_latents": 0}


def vae_fp32_mode() -> str:
    raw = (os.environ.get(VAE_FP32_ENV) or "").strip().lower()
    if raw in _OFF:
        return MODE_OFF
    if raw in _ON:
        return MODE_ALWAYS
    return MODE_AUTO


def lacks_tensor_cores(name: Optional[str], capability: Optional[tuple]) -> bool:
    """True for an NVIDIA GPU with no tensor cores, from its name and compute capability."""
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
    """Whether this load decodes in fp32 from the start (forced, or a GPU without tensor cores)."""
    mode = vae_fp32_mode()
    return mode == MODE_ALWAYS or (mode == MODE_AUTO and target_lacks_tensor_cores(target))


def _first_dtype(module: Any) -> Any:
    try:
        return next(iter(module.parameters())).dtype
    except Exception:  # noqa: BLE001
        return None


def _sample_of(out: Any) -> Any:
    if isinstance(out, tuple):
        return out[0] if out else None
    return getattr(out, "sample", out)


def install_finite_decode(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> Optional[str]:
    """Wrap ``pipe.vae.decode``; returns the mode engaged ("fp32" or "retry"), None when not installed. A VAE whose
    weights are fp32 at call time passes straight through."""
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
            with fp32_keeping_storage(vae, encode_parts):
                return encode(x.to(torch.float32), *args, **kwargs)

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
