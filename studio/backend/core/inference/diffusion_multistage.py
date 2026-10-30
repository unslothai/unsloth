# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-stage render helpers keyed on pipeline attributes: encode once, hold denoiser under offload."""

from __future__ import annotations

import contextlib
from typing import Any, Iterable

# Headroom kept free beside a held denoiser while the modules between two calls run next to it.
HOLD_MARGIN_BYTES = 2 * 2**30


def module_bytes(module: Any) -> int:
    try:
        return sum(p.numel() * p.element_size() for p in module.parameters())
    except Exception:  # noqa: BLE001 -- not a module (tests / absent component)
        return 0


def _denoisers(pipe: Any) -> list:
    try:
        from .diffusion_cuda_graph import _denoiser_modules
        return list(_denoiser_modules(pipe))
    except Exception:  # noqa: BLE001 -- no denoiser found: nothing to hold
        return []


def companions_fit_beside_denoiser(pipe: Any, companions: Iterable[Any]) -> bool:
    """Whether the denoiser a call onloaded can stay while ``companions`` (the modules that run before it is needed
    again) are onloaded beside it: free memory plus the allocator's idle cache (minus CUDA graph pools) covers them
    with a margin."""
    try:
        import torch

        dits = _denoisers(pipe)
        param = next(dits[0].parameters()) if dits else None
        if param is None or param.device.type != "cuda":
            return False
        free, total = torch.cuda.mem_get_info(param.device)
        from .diffusion_memory import _user_budget_reserve_mib, cuda_reusable_cache_bytes

        idle = cuda_reusable_cache_bytes(param.device)

        reserve = _user_budget_reserve_mib(int(total) >> 20) << 20
    except Exception:  # noqa: BLE001 -- no CUDA / no denoiser: keep the stock offload
        return False
    need = sum(module_bytes(m) for m in companions if m is not None) + HOLD_MARGIN_BYTES
    return free + max(0, idle) - reserve >= need


@contextlib.contextmanager
def hold_denoiser_between_calls(
    pipe: Any,
    runner: Any,
    companions: Iterable[Any] = (),
):
    """Around a pipeline call that another call of the same request follows. ``pipe`` owns the offload hooks
    (``enable_model_cpu_offload``); ``runner`` is the pipeline actually called (``pipe`` or a pipeline derived from
    its modules, which has no hook list of its own and so never offloads at its end). Without hooks: a no-op.

    runner is pipe: its end-of-call ``maybe_free_model_hooks`` keeps the stateful-cache reset and skips the offload
    when the companions fit. Derived runner: pipe's hooks are freed after the call only when they do not fit."""
    companions = tuple(companions)
    if not getattr(pipe, "_all_hooks", None):
        yield
        return
    if runner is not pipe:
        try:
            yield
        finally:
            if not companions_fit_beside_denoiser(pipe, companions):
                pipe.maybe_free_model_hooks()
        return
    original = pipe.maybe_free_model_hooks
    had_own = "maybe_free_model_hooks" in getattr(pipe, "__dict__", {})

    def at_call_end() -> None:
        if not companions_fit_beside_denoiser(pipe, companions):
            original()
            return
        for component in getattr(pipe, "components", {}).values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

    pipe.maybe_free_model_hooks = at_call_end
    try:
        yield
    finally:
        if had_own:
            pipe.maybe_free_model_hooks = original
        else:
            del pipe.maybe_free_model_hooks


@contextlib.contextmanager
def encode_once(runner: Any):
    """For the span of one request: identical keyword-only ``encode_prompt`` calls (what the diffusers pipelines make)
    after the first get a device copy of the first one's outputs (taken right after it, so its in-place edits never
    reach them) instead of encoding or uploading again. Later calls share that copy: diffusers' pipelines build new
    tensors from it (cat / repeat) and never write it. Calls with tensors or positional arguments pass through.
    Restores the attribute it found."""
    encode = getattr(runner, "encode_prompt", None)
    if not callable(encode):
        yield
        return
    had_own = "encode_prompt" in getattr(runner, "__dict__", {})
    memo: dict[str, Any] = {}

    def encode_prompt(*args: Any, **kwargs: Any) -> Any:
        import torch

        if args or any(isinstance(v, torch.Tensor) for v in kwargs.values()):
            return encode(*args, **kwargs)
        key = repr(sorted((k, repr(v)) for k, v in kwargs.items()))
        if key in memo:
            return memo[key]
        out = encode(**kwargs)
        if isinstance(out, tuple):
            # A device copy: the first call may edit its own tensors in place.
            memo[key] = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in out)
        return out

    runner.encode_prompt = encode_prompt
    try:
        yield
    finally:
        memo.clear()
        if had_own:
            runner.encode_prompt = encode
        else:
            try:
                del runner.encode_prompt
            except AttributeError:
                pass
