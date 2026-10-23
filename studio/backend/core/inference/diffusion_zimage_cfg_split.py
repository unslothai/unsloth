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

Studio arms it ``compiled_only``: a render splits only while the denoiser runs compiled code or replays CUDA graphs,
checked per render since a deferred compile can engage after the load. An eager denoiser has no graph to keep, and
there the two serial calls cost more than one batched guided call (B200, eager, random weights: ~2x at 256px,
~1.3-1.5x at 512px, parity at 1024px). An eager guided call that runs out of memory batched retries as the two
calls (half the activations) for the rest of that render. An active CFG-parallel router keeps the split eager too:
its replica runs the guidance-off half concurrently, and it cannot pair one batched call. Graphs are read from the
denoiser too: an img2img / inpaint pipeline built by ``from_pipe`` shares the transformer and its graph wrapper but not the root pipe's handles.
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
_COMPILED_ONLY_ATTR = "_unsloth_zimage_cfg_split_compiled_only"
# During an eager compiled_only render: the guided call runs batched and splits only once it ran out of memory.
_EAGER_ATTR = "_unsloth_zimage_cfg_eager"
_OOM_ATTR = "_unsloth_zimage_cfg_oom"
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


def _live_graph_wrapper(denoiser: Any) -> bool:
    """Whether an enabled CUDA-graph wrapper sits on ``denoiser`` or one of its blocks. Read from the module, not a
    pipeline: an img2img / inpaint pipeline built by ``from_pipe`` shares the transformer but not the handle tuple."""
    from .diffusion_cuda_graph import _LIVE_WRAPPERS

    owned = {id(m) for m in denoiser.modules()}
    return any(
        getattr(w, "enabled", False)
        and not getattr(w, "poisoned", False)
        and id(getattr(w, "module", None)) in owned
        for w in tuple(_LIVE_WRAPPERS)
    )


def _graph_handle_live(handle: Any) -> bool:
    """A pipe graph handle that can still replay: enabled, and not poisoned unless a per-block fallback remains."""
    if not getattr(handle, "enabled", True):
        return False
    if not getattr(handle, "poisoned", False):
        return True
    if getattr(handle, "fallback", None) is not None:
        return True
    fallback = getattr(handle, "fallback_handle", None)
    return fallback is not None and _graph_handle_live(fallback)


def _graphs_engaged(pipe: Any, denoiser: Any) -> bool:
    """Whether ``denoiser`` runs compiled code or replays CUDA graphs now, the only case a split saves anything."""
    try:
        from .diffusion_bg_compile import eager_forced

        # A render recording for a pending background compile bypasses compiled calls and graphs alike.
        if eager_forced():
            return False
    except Exception:  # noqa: BLE001 - unknown: read the denoiser
        pass
    try:
        from .diffusion_cfg_batch import compiled_denoiser
    except Exception:  # noqa: BLE001 - unknown: keep the split
        return True
    try:
        # A capture that failed for good leaves its poisoned handle in the tuple: it replays nothing.
        if any(
            _graph_handle_live(h) for h in tuple(getattr(pipe, "_unsloth_cuda_graphs", None) or ())
        ):
            return True
        if _live_graph_wrapper(denoiser):
            return True
        # A compile that failed at its first forward left marks on the transformer but runs eager.
        guard = getattr(denoiser, "__dict__", {}).get("_unsloth_compile_guard")
        if guard is not None and getattr(guard, "error", None):
            return False
        if compiled_denoiser(denoiser) is not None:
            return True
        return any(
            hasattr(getattr(sub, "__dict__", {}).get("forward"), "_torchdynamo_orig_callable")
            for sub in denoiser.modules()
        )
    except Exception:  # noqa: BLE001 - unknown: keep the split
        return True


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


def _is_oom(exc: BaseException) -> bool:
    try:
        from .diffusion_batched import is_oom_error
        return bool(is_oom_error(exc))
    except Exception:  # noqa: BLE001
        return "out of memory" in str(exc).lower()


def _drop_tracebacks(exc: BaseException) -> None:
    try:
        from .diffusion_vae_oom import drop_tracebacks
        drop_tracebacks(exc)
    except Exception:  # noqa: BLE001 - best effort
        pass


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _router_declines_now() -> bool:
    """A CFG-parallel router runs no pair under a background compile's eager or capture gate."""
    try:
        from .diffusion_multigpu import _background_compile_call
        return bool(_background_compile_call())
    except Exception:  # noqa: BLE001
        return False


def _static_skip_state(slots: dict) -> Optional[tuple]:
    skip = slots.get("_unsloth_static_skip")
    if skip is None or not hasattr(skip, "ordinal"):
        return None
    counters = getattr(skip, "counters", None)
    stats = getattr(skip, "stats", None)
    return (
        skip,
        skip.ordinal,
        dict(counters) if isinstance(counters, dict) else None,
        dict(stats) if isinstance(stats, dict) else None,
    )


def _restore_static_skip(slots: dict, state: Optional[tuple]) -> None:
    if state is None:
        return
    skip, ordinal, counters, stats = state
    if slots.get("_unsloth_static_skip") is not skip:
        return
    skip.ordinal = ordinal
    if counters is not None:
        skip.counters = counters
    if stats is not None:
        skip.stats = stats


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
        router = slots.get("_unsloth_cfg_router")
        routed = (
            router is not None
            and slots.get("forward") is router
            and getattr(router, "disabled_reason", None) is None
            and not _router_declines_now()
        )
        # A CFG-parallel router needs the halves even eager: it cannot pair one batched call.
        if slots.get(_EAGER_ATTR) and not slots.get(_OOM_ATTR) and not routed:
            # Eager: one batched call is faster; two calls at half the activations only when it does not fit.
            skip_state = _static_skip_state(slots)
            try:
                return orig_call(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - reraised unless an OOM
                if not _is_oom(exc):
                    raise
                _drop_tracebacks(exc)
                # The failed call consumed a static-skip ordinal; the halves must land under this step's keys.
                _restore_static_skip(slots, skip_state)
            # Outside the handler, so the failed call's frames are gone before the split runs.
            self.__dict__[_OOM_ATTR] = True
            _empty_cache()
            if stats is not None:
                stats["oom_split"] = stats.get("oom_split", 0) + 1
        cond, uncond, kw = parts
        pair = None
        if routed:
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
        for name in (
            _ARMED_ATTR,
            _SAMPLES_ATTR,
            _STATS_ATTR,
            _COMPILED_ONLY_ATTR,
            _EAGER_ATTR,
            _OOM_ATTR,
        ):
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
    compiled_only: bool = False,
) -> Optional[str]:
    """Arm the split on ``pipe.transformer``; the decline reason, or None when armed. Never fails a load.
    ``compiled_only``: a render splits only while the denoiser is compiled or replays CUDA graphs."""
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
            denoiser.__dict__[_COMPILED_ONLY_ATTR] = bool(compiled_only)
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
    slots = getattr(denoiser, "__dict__", {}) if denoiser is not None else {}
    armed = bool(slots.get(_ARMED_ATTR))
    if not armed:
        yield
        return
    # Eager (compiled_only, nothing compiled or graphed): batched, split only after the batched call ran out of memory.
    eager = bool(slots.get(_COMPILED_ONLY_ATTR)) and not _graphs_engaged(pipe, denoiser)
    denoiser.__dict__[_SAMPLES_ATTR] = int(samples)
    if eager:
        denoiser.__dict__[_EAGER_ATTR] = True
    try:
        yield
    finally:
        for name in (_SAMPLES_ATTR, _EAGER_ATTR, _OOM_ATTR):
            denoiser.__dict__.pop(name, None)
