# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streamed denoiser: each block runs both guidance branches while its weights are on the card, so a step uploads
them once instead of once per branch. Every block call keeps its plain inputs and shapes, so outputs are
bit-identical. UNSLOTH_DIFFUSION_CFG_PAIR=0 streams each branch on its own."""

from __future__ import annotations

import os
from typing import Any, Optional

CFG_PAIR_ENV = "UNSLOTH_DIFFUSION_CFG_PAIR"

# class -> (block list attribute, the block output's names in the next block's keyword arguments)
PAIRED_CLASSES = {
    "QwenImageTransformer2DModel": (
        "transformer_blocks",
        ("encoder_hidden_states", "hidden_states"),
    ),
}

_OFF = ("0", "off", "false", "no", "disable", "disabled")


def cfg_pair_enabled() -> bool:
    return (os.environ.get(CFG_PAIR_ENV) or "").strip().lower() not in _OFF


class NotPairable(Exception):
    pass


class _Captured(Exception):
    pass


def _bg() -> Any:
    from . import diffusion_block_graph as bg
    return bg


def _hook_refs(module: Any) -> list:
    return list(getattr(getattr(module, "_diffusers_hook", None), "_fn_refs", None) or ())


def below_hook_ref(block: Any) -> Any:
    """The offload hook reference whose ``forward`` is the block's own compute (compiled below the hook or not)."""
    bg = _bg()
    if bg._group_offload_hook(block) is None:
        return None
    refs = _hook_refs(block)
    target = getattr(block, "_unsloth_below_hook_ref", None)
    if any(r is target for r in refs):
        return target
    for r in refs:
        if bg._own_forward(getattr(r, "forward", None), block) is not None:
            return r
    return None


def root_forward(module: Any, fallback: Any = None) -> Any:
    """The forward below the denoiser's top-level offload hook, else ``fallback`` if that is the plain forward."""
    bg = _bg()
    for r in _hook_refs(module):
        fn = getattr(r, "forward", None)
        if bg._own_forward(fn, module) is not None:
            return fn
    if not _hook_refs(module) and bg._own_forward(fallback, module) is not None:
        return fallback
    return None


def _spec(module: Any) -> Optional[tuple]:
    cls = type(module)
    if not cls.__module__.startswith("diffusers."):
        return None
    return PAIRED_CLASSES.get(cls.__name__)


def _streams_a_block(module: Any) -> bool:
    try:
        from .diffusion_memory import _offload_groups
        return any(
            not getattr(g, "_unsloth_resident", False)
            and getattr(g, "offload_leader", None) is not module
            for g in _offload_groups(module)
        )
    except Exception:  # noqa: BLE001 - unreadable layout: treat as streamed, the block checks still apply
        return True


def pair_decline(module: Any, fallback_forward: Any = None) -> Optional[str]:
    if not cfg_pair_enabled():
        return f"{CFG_PAIR_ENV}=0"
    spec = _spec(module)
    if spec is None:
        return f"{type(module).__name__} has no paired block loop"
    import torch

    blocks = getattr(module, "_modules", {}).get(spec[0])
    if not isinstance(blocks, torch.nn.ModuleList) or not len(blocks):
        return "no block list"
    if any(below_hook_ref(b) is None for b in blocks):
        return "a block runs without an offload hook to pair below"
    inside = {id(m) for b in blocks for m in b.modules()}
    for name, sub in module.named_modules():
        if sub is module or id(sub) in inside:
            continue
        if (
            getattr(sub, "_diffusers_hook", None) is not None
            or getattr(sub, "_hf_hook", None) is not None
        ):
            return f"{name} carries its own hook"
    if root_forward(module, fallback_forward) is None:
        return "the denoiser's own forward is not reachable below its hooks"
    if not _streams_a_block(module):
        return "every block stays resident"
    return None


def _modules() -> tuple:
    import torch

    class _Probe(torch.nn.Module):
        call: Any = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            self.call = (args, kwargs)
            raise _Captured

    class _Inject(torch.nn.Module):
        out: Any = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.out

    return _Probe, _Inject


class PairedForward:
    """Second branch rides the first's block calls; its prologue/epilogue run the stock forward on a one-block list."""

    def __init__(self, module: Any) -> None:
        import torch

        probe_cls, inject_cls = _modules()
        self.probe, self.inject = probe_cls(), inject_cls()
        self.probe_list = torch.nn.ModuleList([self.probe])
        self.inject_list = torch.nn.ModuleList([self.inject])
        self.blocks_attr, self.carry = _spec(module)

    def run(self, module: Any, outer: Any, root_fn: Any, first: tuple, second: tuple) -> tuple:
        """(first output, second output). ``outer`` is the hooked forward a plain first-branch call goes through."""
        (args1, kwargs1), (args2, kwargs2) = first, second
        if args1 or args2:
            raise NotPairable("positional arguments")
        for kw in (kwargs1, kwargs2):
            if kw.get("controlnet_block_samples") is not None:
                raise NotPairable("ControlNet residuals")
            att = kw.get("attention_kwargs") or {}
            if isinstance(att, dict) and att.get("scale", 1.0) != 1.0:
                raise NotPairable("a LoRA scale in attention_kwargs")
        mods = module._modules
        blocks = mods[self.blocks_attr]
        refs = [below_hook_ref(b) for b in blocks]
        if any(r is None for r in refs):
            raise NotPairable("a block lost its offload hook")
        last = len(blocks) - 1
        state: dict = {}

        def nested(stand_in: Any) -> Any:
            saved = mods[self.blocks_attr]
            mods[self.blocks_attr] = stand_in
            try:
                return root_fn(**kwargs2)
            finally:
                mods[self.blocks_attr] = saved

        def paired(index: int, compute: Any) -> Any:
            def call(*a: Any, **k: Any) -> Any:
                out = compute(*a, **k)
                if index == 0:
                    try:
                        nested(self.probe_list)
                    except _Captured:
                        pass
                    p_args, p_kwargs = self.probe.call
                    self.probe.call = None
                    if p_args:
                        raise NotPairable("positional block arguments")
                    state["kwargs"] = dict(p_kwargs)
                second_out = compute(**state["kwargs"])
                state["kwargs"].update(zip(self.carry, second_out))
                if index == last:
                    self.inject.out = second_out
                    try:
                        state["out"] = nested(self.inject_list)
                    finally:
                        self.inject.out = None
                return out

            return call

        saved = [r.forward for r in refs]
        try:
            for i, r in enumerate(refs):
                r.forward = paired(i, saved[i])
            mine = outer(*args1, **kwargs1)
        finally:
            for r, fn in zip(refs, saved):
                r.forward = fn
        if "out" not in state:
            raise NotPairable("the forward never reached its last block")
        return mine, state["out"]
