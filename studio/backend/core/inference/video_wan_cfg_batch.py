# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan classifier-free guidance as one batch-2 denoiser call on fp16 GPUs (T4 and other pre-Ampere cards).

``WanPipeline`` calls the denoiser twice per step at batch 1, once with the prompt and once with the negative prompt,
on the same latents and timestep. When the denoiser streams its weights from host RAM (a 10 GB Wan2.2-TI2V-5B on a
15 GB T4), every step moves the whole DiT over PCIe twice, and attention at batch 1 runs ~12% more cycles than one
batch-2 call.

The pipeline only reads the conditional prediction after the unconditional call returns, so the conditional call
returns an empty output tensor and records its inputs; the unconditional call that follows (same latent and timestep
objects, same embedding shape) runs both rows as one batch-2 forward, writes row 0 into that tensor and returns row 1.
Anything else falls back to the stock order: a following call that is not the matching unconditional one first
computes the pending conditional call alone, and an out-of-memory batch reruns both rows one at a time.

Batch 2 is not bit-identical to two batch-1 calls (cuBLAS may pick different GEMM tiles for twice the rows); the gain
and the drift were measured on a T4 before this was enabled, so it is scoped to fp16 loads only (bf16 GPUs keep the
stock two calls). Armed only inside the denoising loop of a ``WanPipeline`` whose ``__call__`` still has the shape this
relies on, with CFG on, no step cache on the denoiser and no gradients. Kill switch: ``UNSLOTH_DIFFUSION_WAN_CFG_BATCH=0``.
"""

from __future__ import annotations

import inspect
import os
import threading
from typing import Any, Optional

WAN_CFG_BATCH_ENV = "UNSLOTH_DIFFUSION_WAN_CFG_BATCH"
_ATTR = "_unsloth_wan_cfg_batch"
# Source tokens the deferral relies on, checked in the pipeline class before installing.
_PIPELINE_TOKENS = (
    'current_model.cache_context("cond"',
    'current_model.cache_context("uncond"',
    "noise_uncond = current_model(",
    "encoder_hidden_states=negative_prompt_embeds",
    "noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)",
    "self._current_timestep = t",
    "self._current_timestep = None",
)
_SUPPORTED: dict = {}
_COUNTS = {"batched": 0, "single": 0, "fallback": 0}
_LOCK = threading.Lock()


def disabled() -> bool:
    return (os.environ.get(WAN_CFG_BATCH_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _pipeline_supported(cls: Any) -> bool:
    if cls in _SUPPORTED:
        return _SUPPORTED[cls]
    try:
        src = inspect.getsource(inspect.unwrap(cls.__call__))
        ok = all(tok in src for tok in _PIPELINE_TOKENS)
    except (OSError, TypeError):
        ok = False
    _SUPPORTED[cls] = ok
    return ok


class _Pending:
    __slots__ = ("kwargs", "out")

    def __init__(self, kwargs: dict, out: Any):
        self.kwargs = kwargs
        self.out = out


_ARG_NAMES = (
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "encoder_hidden_states_image",
    "return_dict",
    "attention_kwargs",
)


def _as_kwargs(args: tuple, kwargs: dict) -> Optional[dict]:
    if len(args) > len(_ARG_NAMES):
        return None
    call = dict(zip(_ARG_NAMES, args))
    for key, value in kwargs.items():
        if key in call or key not in _ARG_NAMES:
            return None
        call[key] = value
    return call


def _is_oom(exc: BaseException) -> bool:
    try:
        from .diffusion_batched import is_oom_error

        return bool(is_oom_error(exc))
    except Exception:  # noqa: BLE001
        return "out of memory" in str(exc).lower()


class _CfgBatcher:
    """Instance-level wrapper around one Wan denoiser's forward (outermost, so offload hooks still run per call)."""

    def __init__(self, pipe: Any, transformer: Any, inner: Any):
        import weakref

        self._pipe = weakref.ref(pipe)
        self._inner = inner
        self._pending: Optional[_Pending] = None
        self.__wrapped__ = inner

    # -- arming ---------------------------------------------------------------------------------------------------
    def _armed(self, call: dict) -> bool:
        import torch

        pipe = self._pipe()
        if pipe is None or disabled():
            return False
        if getattr(pipe, "_current_timestep", None) is None or getattr(pipe, "_interrupt", False):
            return False
        try:
            if not pipe.do_classifier_free_guidance:
                return False
        except Exception:  # noqa: BLE001
            return False
        if call.get("return_dict", True) is not False or call.get("encoder_hidden_states_image") is not None:
            return False
        x, t, enc = call.get("hidden_states"), call.get("timestep"), call.get("encoder_hidden_states")
        if not all(torch.is_tensor(v) for v in (x, t, enc)):
            return False
        if x.dtype is not torch.float16 or not x.is_cuda or x.shape[0] != 1 or enc.shape[0] != 1:
            return False
        if t.dim() not in (1, 2) or t.shape[0] != 1:
            return False
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return False
        return True

    @staticmethod
    def _pairs(pending: _Pending, call: dict) -> bool:
        a, b = pending.kwargs, call
        if b.get("hidden_states") is not a["hidden_states"] or b.get("timestep") is not a["timestep"]:
            return False
        if b.get("return_dict", True) is not False or b.get("encoder_hidden_states_image") is not None:
            return False
        if b.get("attention_kwargs") is not a.get("attention_kwargs"):
            return False
        ea, eb = a["encoder_hidden_states"], b.get("encoder_hidden_states")
        return eb is not None and eb.shape == ea.shape and eb.dtype == ea.dtype and eb.device == ea.device

    # -- execution ------------------------------------------------------------------------------------------------
    def _single(self, call: dict) -> Any:
        return self._inner(**call)

    def _fill(self, pending: _Pending) -> None:
        out = self._single(pending.kwargs)[0]
        pending.out.copy_(out)
        _COUNTS["single"] += 1

    def _batched(self, pending: _Pending, call: dict) -> Any:
        import torch

        a = pending.kwargs
        both = dict(a)
        both["hidden_states"] = torch.cat([a["hidden_states"], a["hidden_states"]])
        both["timestep"] = torch.cat([a["timestep"], a["timestep"]])
        both["encoder_hidden_states"] = torch.cat([a["encoder_hidden_states"], call["encoder_hidden_states"]])
        out = self._inner(**both)[0]
        pending.out.copy_(out[:1])
        _COUNTS["batched"] += 1
        return (out[1:].contiguous(),)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        call = _as_kwargs(args, kwargs)
        pending, self._pending = self._pending, None
        if pending is not None:
            if call is not None and self._pairs(pending, call):
                try:
                    return self._batched(pending, call)
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    _COUNTS["fallback"] += 1
                    torch.cuda.empty_cache()
                    self._fill(pending)
                    return self._inner(*args, **kwargs)
            # not the unconditional call this was waiting for: compute the conditional one alone first, and run this
            # one as it came (it follows a deferred call, so it is never deferred itself)
            self._fill(pending)
            return self._inner(*args, **kwargs)
        if call is not None and self._armed(call):
            x = call["hidden_states"]
            self._pending = _Pending(call, torch.empty_like(x, memory_format = torch.contiguous_format))
            return (self._pending.out,)
        return self._inner(*args, **kwargs)

    def flush(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            self._fill(pending)


def wanted(dtype: Any, device: Any = "cuda") -> bool:
    if disabled():
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return dtype is torch.float16 and str(device).startswith("cuda") and not getattr(torch.version, "hip", None)


def install_for_pipe(
    pipe: Any,
    dtype: Any,
    device: Any = "cuda",
    *,
    cache_engaged: Any = None,
    logger: Any = None,
) -> bool:
    """Wrap the Wan denoiser(s) of an fp16 ``WanPipeline`` load; False (nothing wrapped) otherwise."""
    if not wanted(dtype, device) or cache_engaged:
        return False
    if type(pipe).__name__ != "WanPipeline" or not _pipeline_supported(type(pipe)):
        return False
    engaged = False
    with _LOCK:
        for name in ("transformer", "transformer_2"):
            dit = getattr(pipe, name, None)
            if dit is None or type(dit).__name__ != "WanTransformer3DModel":
                continue
            if isinstance(dit.__dict__.get("forward"), _CfgBatcher):
                engaged = True
                continue
            batcher = _CfgBatcher(pipe, dit, dit.forward)
            dit.forward = batcher
            setattr(dit, _ATTR, batcher)
            engaged = True
    if engaged and logger is not None:
        try:
            logger.info("video.wan_cfg_batch: conditional and unconditional denoiser calls run as one batch-2 call")
        except Exception:  # noqa: BLE001
            pass
    return engaged


def uninstall(pipe: Any) -> None:
    for name in ("transformer", "transformer_2"):
        dit = getattr(pipe, name, None)
        if dit is None:
            continue
        batcher = dit.__dict__.get(_ATTR)
        if batcher is None:
            continue
        batcher.flush()
        if dit.__dict__.get("forward") is batcher:
            dit.forward = batcher._inner
        try:
            delattr(dit, _ATTR)
        except AttributeError:
            pass


def counts() -> dict:
    return dict(_COUNTS)
