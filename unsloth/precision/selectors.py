# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""High-precision layer selection by structure: output heads (and weights tied to them), embeddings, MoE
routers and the first / last decoder layers. Names are a fallback only: a router called `wg` or `gate_proj`,
or one in remote code, is found by its shape and its place beside the experts."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .recipes import HPLayerPolicy

__all__ = [
    "num_experts_of",
    "find_routers",
    "find_output_heads",
    "find_embeddings",
    "decoder_layers",
    "high_precision_layers",
    "hp_filter",
]

_EXPERT_COUNT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_routed_experts",
)
_SUBCONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder")


def num_experts_of(config) -> Optional[int]:
    """Number of routed experts from a (possibly nested) HF config, None for a dense model."""
    seen = set()
    stack = [config]
    while stack:
        c = stack.pop()
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        for key in _EXPERT_COUNT_KEYS:
            v = c.get(key) if isinstance(c, dict) else getattr(c, key, None)
            if isinstance(v, int) and not isinstance(v, bool) and v > 1:
                return v
        for key in _SUBCONFIG_KEYS:
            stack.append(c.get(key) if isinstance(c, dict) else getattr(c, key, None))
    return None


def _experts_count(module) -> Optional[int]:
    """Expert count of an experts container: ModuleList length, `num_experts` attribute, or the leading dim
    of its 3-D parameters (stacked expert weights)."""
    if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
        return len(module)
    for key in _EXPERT_COUNT_KEYS:
        v = getattr(module, key, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 1:
            return v
    for p in module.parameters(recurse = False):
        if p.dim() == 3 and p.shape[0] > 1:
            return int(p.shape[0])
    return None


def _out_features(module) -> Optional[int]:
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return int(w.shape[0])
    return getattr(module, "out_features", None)


def find_routers(model) -> Dict[str, str]:
    """{fqn: reason} for every module that scores tokens over experts.

    A module is a router when its output width equals the expert count and it sits in the same block as an
    experts container (a child whose name contains 'expert', excluding shared experts). With no experts
    container in its block, the config's expert count plus a 2-D weight whose width matches is enough only
    for a module whose leaf name is a known router name."""
    config = getattr(model, "config", None)
    n_cfg = num_experts_of(config) if config is not None else None
    found = {}
    for parent_name, parent in model.named_modules():
        # Inside an expert or shared expert (`experts.3`, `shared_expert`): its projections can be as wide as
        # the expert count (Qwen3-Next: 512 experts, moe_intermediate_size 512) but never route.
        if any("expert" in part.lower() for part in parent_name.split(".")):
            continue
        experts = [
            (n, c)
            for n, c in parent.named_children()
            if "expert" in n.lower() and "shared" not in n.lower()
        ]
        counts = {_experts_count(c) for _, c in experts} - {None}
        if not counts and n_cfg is None:
            continue
        widths = counts or {n_cfg}
        for child_name, child in parent.named_children():
            if any(child is c for _, c in experts) or "shared" in child_name.lower():
                continue
            out = _out_features(child)
            if out is None or out not in widths:
                continue
            # No experts container beside it: only a router name counts; `gate_proj` is excluded here
            # because a dense MLP's gate_proj can be as wide as the expert count.
            if not counts and child_name.lower() not in (
                "gate",
                "router",
                "wg",
                "router_proj",
                "score",
            ):
                continue
            fqn = f"{parent_name}.{child_name}" if parent_name else child_name
            found[fqn] = (
                f"router: {out} outputs = expert count, beside {experts[0][0] if experts else 'config'}"
            )
    return found


def find_output_heads(model) -> Dict[str, str]:
    """Output embedding modules, and every Linear sharing its weight storage."""
    heads = {}
    out = None
    try:
        out = model.get_output_embeddings()
    except Exception:
        out = None
    ptrs = set()
    if out is not None and isinstance(getattr(out, "weight", None), torch.Tensor):
        try:
            ptrs.add(out.weight.data_ptr())
        except Exception:
            pass
    for name, module in model.named_modules():
        if out is not None and module is out:
            heads[name] = "output head (get_output_embeddings)"
            continue
        w = getattr(module, "weight", None)
        if isinstance(module, torch.nn.Linear) and isinstance(w, torch.Tensor) and ptrs:
            try:
                if not w.is_meta and w.data_ptr() in ptrs:
                    heads[name] = "weight tied to the output head"
            except Exception:
                pass
    return heads


def find_embeddings(model) -> Dict[str, str]:
    emb = {}
    inp = None
    try:
        inp = model.get_input_embeddings()
    except Exception:
        inp = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) or (inp is not None and module is inp):
            emb[name] = "embedding"
    return emb


def decoder_layers(model):
    """(fqn prefix of the decoder layer list, number of layers): the longest ModuleList of same-type blocks
    that hold Linear layers. (None, 0) when there is none."""
    best = (None, 0)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        types = {type(m) for m in module}
        if len(types) != 1:
            continue
        if not any(isinstance(m, torch.nn.Linear) for m in module[0].modules()):
            continue
        # experts lists (ModuleList of MLPs inside a block) are shorter than the layer list on real models, but
        # a tie goes to the shallower list.
        if len(module) > best[1] or (
            len(module) == best[1] and best[0] is not None and name.count(".") < best[0].count(".")
        ):
            best = (name, len(module))
    return best


def high_precision_layers(model, policy: HPLayerPolicy) -> Dict[str, str]:
    """{fqn: reason} of modules the policy keeps in high precision. A prefix entry (decoder layer) covers
    every module under it."""
    keep = {}
    if policy.output_head:
        keep.update(find_output_heads(model))
    if policy.embeddings:
        keep.update(find_embeddings(model))
    if policy.routers:
        keep.update(find_routers(model))
    if policy.first_n or policy.last_n or policy.last_frac:
        prefix, n = decoder_layers(model)
        if prefix is not None and n:
            import math
            last = max(policy.last_n, math.ceil(policy.last_frac * n) if policy.last_frac else 0)
            for i in range(n):
                if i < policy.first_n:
                    keep[f"{prefix}.{i}"] = f"first {policy.first_n} decoder layers"
                elif i >= n - last:
                    keep[f"{prefix}.{i}"] = f"last {last} of {n} decoder layers"
    if policy.shared_experts:
        for name, _ in model.named_modules():
            if "shared_expert" in name.rsplit(".", 1)[-1]:
                keep[name] = "shared expert"
    return keep


def hp_filter(model, policy: HPLayerPolicy):
    """Predicate (module, fqn) -> True when the module (or a block containing it) stays in high precision."""
    keep = high_precision_layers(model, policy)
    exact = set(keep)

    def is_hp(module, fqn: str) -> bool:
        if fqn in exact:
            return True
        parts = fqn.split(".")
        return any(".".join(parts[:i]) in exact for i in range(1, len(parts)))

    is_hp.keep = keep
    return is_hp
