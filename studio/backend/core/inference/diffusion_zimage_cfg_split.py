# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Z-Image guidance branches as two calls at the guidance-off batch, so turning CFG on compiles and records nothing.

``ZImagePipeline`` (and its img2img / inpaint variants) batches a guided step into ONE transformer call of twice the
samples: the latents repeated, the prompt embeddings followed by the negative ones. A session that rendered guidance 0
first (Turbo's default) compiled the repeated block at one sample with no attention mask; the doubled call is a new
batch (dynamo specialises size 1) and, whenever the two captions pad to different 32-token buckets, a mask that was None
before, so the first guided render recompiled the block three times (one graph per block kind: noise refiner, context
refiner, main layers) and recorded a new whole-step CUDA graph. Later guided prompts kept recompiling as their caption
lengths moved between "same bucket as the negative" (no mask) and "other bucket" (mask).

Here a guided call of ``2 * B`` samples runs as the conditional ``B`` then the unconditional ``B`` through the module's
normal call (every forward layer, hook and graph sees an ordinary guidance-off call), and the two output lists are
joined in the pipeline's order. Same math: the batched call masks the padding, the split one has none to mask. Not
bit-exact: a single sample with no mask can pick other attention / GEMM tiles than the masked pair.

Engaged per render (``render``: Studio knows the images per call, so ``2 * B`` samples is a guided call, ``B`` is not),
only on a resident denoiser: a block-streamed one would upload its weights twice per step, which costs more than the
recompile. Declined per call under a step cache and a static step skip that cannot key a step's two calls apart.
Under a CFG-parallel router the halves go to ``run_pair`` (identity pairing never matches fresh tensors). Kill switch: ``UNSLOTH_DIFFUSION_ZIMAGE_CFG_SPLIT=0``.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, Optional

ZIMAGE_CFG_SPLIT_ENV = "UNSLOTH_DIFFUSION_ZIMAGE_CFG_SPLIT"
_TRANSFORMER_CLASS = "ZImageTransformer2DModel"
_STREAMED_POLICIES = ("group", "streaming", "sequential")
# Instance attributes: armed at load, images per guidance-off call during a render, per-load counters.
_ARMED_ATTR = "_unsloth_zimage_cfg_split"
_SAMPLES_ATTR = "_unsloth_zimage_cfg_samples"
_STATS_ATTR = "_unsloth_zimage_cfg_split_stats"
# Keyword arguments a split call may carry (anything else, e.g. siglip / controlnet / noise masks, runs batched).
_PLAIN_KWARGS = frozenset({"return_dict", "joint_attention_kwargs", "patch_size", "f_patch_size"})

_LOCK = threading.Lock()
_PATCHED: dict = {}


def zimage_cfg_split_disabled() -> bool:
    return (os.environ.get(ZIMAGE_CFG_SPLIT_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _is_zimage(module: Any) -> bool:
    return type(module).__name__ == _TRANSFORMER_CLASS


def _skip_keys_calls_apart(skip: Any) -> bool:
    """Whether a static step skip keys cond / uncond apart: without a step signal one shared counter advances twice."""
    if not hasattr(skip, "plan"):
        return False
    if not skip.plan:
        return True
    return getattr(skip, "context", None) is None and bool(getattr(skip, "step_signal", False))


def _declined_now(module: Any) -> Optional[str]:
    slots = getattr(module, "__dict__", {})
    # FBCache keeps one residual per cache_context; two calls in one context would compare cond against uncond.
    if getattr(module, "_unsloth_step_cache", None):
        return "step cache"
    # A CFG-parallel router installed after the skip takes the forward slot.
    for skip in (slots.get("_unsloth_static_skip"), slots.get("forward")):
        if getattr(skip, "_unsloth_outer_forward", False) and not _skip_keys_calls_apart(skip):
            return "static step skip without a per-step call order"
    if getattr(module, "training", False):
        return "training"
    return None


def _halves(samples: int, args: tuple, kwargs: dict) -> Optional[tuple]:
    """``(cond_args, uncond_args, kwargs)`` when this is a guided call of ``2 * samples`` plain samples, else None."""
    if samples < 1 or len(args) != 3 or not set(kwargs) <= _PLAIN_KWARGS:
        return None
    x, t, cap = args
    n = 2 * samples
    if not isinstance(x, list) or not isinstance(cap, list) or len(x) != n or len(cap) != n:
        return None
    # Omni inputs are lists of lists per sample; this splits plain tensors only.
    if not all(hasattr(v, "shape") for v in x) or not all(hasattr(v, "shape") for v in cap):
        return None
    shape = getattr(t, "shape", None)
    if shape is None or len(shape) != 1 or shape[0] != n:
        return None
    return (
        (x[:samples], t[:samples], cap[:samples]),
        (x[samples:], t[samples:], cap[samples:]),
        kwargs,
    )


def _join(a: Any, b: Any) -> Any:
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and a:
        return (list(a[0]) + list(b[0]),) + tuple(a[1:])
    sample_a, sample_b = getattr(a, "sample", None), getattr(b, "sample", None)
    if isinstance(sample_a, list) and isinstance(sample_b, list):
        return type(a)(sample = sample_a + sample_b)
    raise TypeError(f"unexpected Z-Image output {type(a).__name__}")


def _make_call(orig_call: Any) -> Any:
    def __call__(self, *args, **kwargs):
        slots = getattr(self, "__dict__", {})
        samples = slots.get(_SAMPLES_ATTR)
        if not samples or not slots.get(_ARMED_ATTR):
            return orig_call(self, *args, **kwargs)
        parts = _halves(int(samples), args, kwargs)
        stats = slots.get(_STATS_ATTR)
        if parts is None:
            return orig_call(self, *args, **kwargs)
        if _declined_now(self) is not None:
            if stats is not None:
                stats["batched"] += 1
            return orig_call(self, *args, **kwargs)
        cond, uncond, kw = parts
        router = slots.get("_unsloth_cfg_router")
        pair = None
        if router is not None and slots.get("forward") is router:
            pair = router.run_pair(lambda: orig_call(self, *cond, **kw), uncond, dict(kw))
        if pair is not None:
            out_cond, out_uncond = pair
            if out_uncond is None:
                out_uncond = orig_call(self, *uncond, **kw)
            elif stats is not None:
                stats["parallel"] = stats.get("parallel", 0) + 1
        else:
            out_cond = orig_call(self, *cond, **kw)
            out_uncond = orig_call(self, *uncond, **kw)
        if stats is not None:
            stats["split"] += 1
        return _join(out_cond, out_uncond)

    __call__.__unsloth_zimage_cfg_split__ = True
    return __call__


def _patch_class(cls: Any) -> bool:
    with _LOCK:
        if cls in _PATCHED:
            return True
        own = cls.__dict__.get("__call__")
        if getattr(own, "__unsloth_zimage_cfg_split__", False):
            return True
        # (the class's own __call__ or None when inherited, the patched one)
        patched = _make_call(cls.__call__)
        cls.__call__ = patched
        _PATCHED[cls] = (own, patched)
        return True


def uninstall_zimage_cfg_split(pipe: Any = None) -> None:
    """Disarm ``pipe.transformer`` and restore the class ``__call__`` (model unload). Idempotent."""
    denoiser = getattr(pipe, "transformer", None) if pipe is not None else None
    slots = getattr(denoiser, "__dict__", None)
    if isinstance(slots, dict):
        for name in (_ARMED_ATTR, _SAMPLES_ATTR, _STATS_ATTR):
            slots.pop(name, None)
    with _LOCK:
        for cls, (own, patched) in list(_PATCHED.items()):
            if cls.__dict__.get("__call__") is patched:
                if own is None:
                    delattr(cls, "__call__")
                else:
                    cls.__call__ = own
            _PATCHED.pop(cls, None)


def install_zimage_cfg_split(
    pipe: Any,
    *,
    offload_policy: str,
    logger: Any = None,
) -> Optional[str]:
    """Arm the split on ``pipe.transformer``; the decline reason, or None when armed. Never fails a load."""
    try:
        denoiser = getattr(pipe, "transformer", None)
        if denoiser is None or not _is_zimage(denoiser):
            return "not a Z-Image denoiser"
        if zimage_cfg_split_disabled():
            reason = f"{ZIMAGE_CFG_SPLIT_ENV}=0"
        elif offload_policy in _STREAMED_POLICIES:
            reason = (
                f"denoiser is streamed ({offload_policy}): two calls would upload it twice per step"
            )
        else:
            reason = None
        if reason is None:
            _patch_class(type(denoiser))
            denoiser.__dict__[_ARMED_ATTR] = True
            denoiser.__dict__[_STATS_ATTR] = {"split": 0, "batched": 0}
        else:
            denoiser.__dict__.pop(_ARMED_ATTR, None)
        if logger is not None:
            logger.info(
                "diffusion.zimage_cfg_split: %s",
                "armed (guided steps run as two calls at the guidance-off batch)"
                if reason is None
                else f"not armed ({reason})",
            )
        return reason
    except Exception as exc:  # noqa: BLE001 - an optional lever
        if logger is not None:
            logger.info("diffusion.zimage_cfg_split: not armed (%s: %s)", type(exc).__name__, exc)
        return f"{type(exc).__name__}: {exc}"


def zimage_cfg_split_status(pipe: Any) -> Optional[dict]:
    denoiser = getattr(pipe, "transformer", None)
    if denoiser is None or not _is_zimage(denoiser):
        return None
    slots = getattr(denoiser, "__dict__", {})
    return {"armed": bool(slots.get(_ARMED_ATTR)), **dict(slots.get(_STATS_ATTR) or {})}


@contextlib.contextmanager
def render(pipe: Any, samples: int):
    """For one pipeline call whose guidance-off transformer call carries ``samples`` images."""
    denoiser = getattr(pipe, "transformer", None)
    armed = denoiser is not None and bool(getattr(denoiser, "__dict__", {}).get(_ARMED_ATTR))
    if not armed:
        yield
        return
    denoiser.__dict__[_SAMPLES_ATTR] = int(samples)
    try:
        yield
    finally:
        denoiser.__dict__.pop(_SAMPLES_ATTR, None)
