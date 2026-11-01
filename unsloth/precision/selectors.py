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
    """Expert count of an experts container: ModuleList / ModuleDict length (SwitchTransformers keeps its
    experts in a ModuleDict), `num_experts` attribute, or the leading dim of its 3-D parameters (stacked
    expert weights)."""
    if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)) and len(module) > 1:
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


def _router_width(module) -> Optional[int]:
    """Output width of a router: its own 2-D weight, else the one 2-D weight inside it (a gating module that
    wraps a Linear, GraniteMoE's `router.layer`)."""
    out = _out_features(module)
    if out is not None:
        return out
    inner = [p for p in module.parameters() if p.dim() == 2]
    return int(inner[0].shape[0]) if len(inner) == 1 else None


def _is_stacked_experts(module, n_cfg) -> bool:
    """A container holding all experts' weights as 3-D parameters [num_experts, ...] (GraniteMoE's
    ParallelExperts `input_linear`), whatever it is called. Convolutions (Mamba's conv1d) are 3-D too."""
    if not n_cfg or isinstance(module, torch.nn.modules.conv._ConvNd):
        return False
    return any(p.dim() == 3 and p.shape[0] == n_cfg for p in module.parameters(recurse = False))


def _is_experts_child(name, module, n_cfg) -> bool:
    return ("expert" in name.lower() and "shared" not in name.lower()) or _is_stacked_experts(
        module, n_cfg
    )


def _expert_subtrees(model, n_cfg) -> set:
    """ids of modules inside an individual expert or a shared expert: the subtree of every experts child
    (`experts.3`, `experts.expert_0`) and of every shared-expert MLP. An MoE container that owns stacked
    expert weights in its children, and so routes over them itself (JetMoE's attention `experts`, whose
    router is `experts.router`), is not a collection of experts and stays inspectable."""
    inside = set()
    for _, parent in model.named_modules():
        for name, child in parent.named_children():
            if "shared" not in name.lower() and not _is_experts_child(name, child, n_cfg):
                continue
            if "shared" not in name.lower() and any(
                _is_stacked_experts(c, n_cfg) for c in child.children()
            ):
                continue
            inside.update(id(m) for m in child.modules())
    return inside


def find_routers(model) -> Dict[str, str]:
    """{fqn: reason} for every module that scores tokens over experts.

    A module is a router when its output width equals the expert count and it sits in the same block as an
    experts container (a child whose name contains 'expert', excluding shared experts, or one holding stacked
    3-D expert weights). With no experts container in its block, the config's expert count plus a 2-D weight
    whose width matches is enough only for a module whose leaf name is a known router name."""
    config = getattr(model, "config", None)
    n_cfg = num_experts_of(config) if config is not None else None
    found = {}
    inside_experts = _expert_subtrees(model, n_cfg)
    for parent_name, parent in model.named_modules():
        # Inside an expert or shared expert (`experts.3`, `shared_expert`): its projections can be as wide as
        # the expert count (Qwen3-Next: 512 experts, moe_intermediate_size 512) but never route.
        if id(parent) in inside_experts:
            continue
        experts = [(n, c) for n, c in parent.named_children() if _is_experts_child(n, c, n_cfg)]
        counts = {_experts_count(c) for _, c in experts} - {None}
        if not counts and n_cfg is None:
            continue
        widths = counts or {n_cfg}
        for child_name, child in parent.named_children():
            if any(child is c for _, c in experts) or "shared" in child_name.lower():
                continue
            out = _router_width(child) if counts else _out_features(child)
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
    # No output embedding (sequence classification, reward / value heads): the head is every Linear outside
    # the base model (`score`, `classifier`).
    base = getattr(model, "base_model", None) if out is None else None
    outside = set()
    if isinstance(base, torch.nn.Module) and base is not model:
        inside = {id(m) for m in base.modules()}
        outside = {id(m) for m in model.modules() if id(m) not in inside}
    for name, module in model.named_modules():
        if out is not None and module is out:
            heads[name] = "output head (get_output_embeddings)"
            continue
        if id(module) in outside and isinstance(module, torch.nn.Linear):
            heads[name] = "task head outside the base model"
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


_NON_DECODER_PARTS = (
    "vision",
    "visual",
    "image",
    "audio",
    "speech",
    "encoder",
    "tower",
    "projector",
)


def _is_block_stack(module) -> bool:
    """A non-empty ModuleList whose entries each hold a Linear inside them (not a list of bare Linears such as
    Gemma3n's `altup_projections`). Hybrid stacks (Jamba, Zamba) mix block classes, which is fine."""
    if not isinstance(module, torch.nn.ModuleList) or len(module) < 1:
        return False
    return all(
        any(isinstance(m, torch.nn.Linear) for m in block.modules() if m is not block)
        for block in module
    )


def _language_roots(model):
    """(fqn, module) of the language decoder: the innermost of HF `get_decoder()` (when it names a proper
    submodule) and the nearest ancestor of the input embedding that holds a block stack; else the model."""
    modules = dict(model.named_modules())
    roots = []
    get_decoder = getattr(model, "get_decoder", None)
    try:
        decoder = get_decoder() if callable(get_decoder) else None
    except Exception:
        decoder = None
    if isinstance(decoder, torch.nn.Module) and decoder is not model:
        roots += [(n, m) for n, m in modules.items() if m is decoder][:1]
    try:
        inp = model.get_input_embeddings()
    except Exception:
        inp = None
    emb = next((n for n, m in modules.items() if inp is not None and m is inp), None)
    if emb:
        parts = emb.split(".")
        for i in range(len(parts) - 1, -1, -1):
            name = ".".join(parts[:i])
            if any(_is_block_stack(m) for m in modules[name].modules()):
                roots.append((name, modules[name]))
                break
    # Keep a root only when no other candidate lies inside it (Idefics3: get_decoder() is the whole
    # multimodal model, the embedding's ancestor is its text model). Unrelated roots: get_decoder() wins
    # (T5: the shared embedding sits above both stacks).
    roots = [
        (n, m)
        for n, m in roots
        if not any(o != n and (not n or o.startswith(n + ".")) for o, _ in roots)
    ]
    return roots[0] if roots else ("", model)


def decoder_layers(model):
    """(fqn prefix of the decoder layer list, number of layers): the longest ModuleList of blocks that each
    hold Linear layers inside the language decoder (the stack beside the input embedding, or HF
    `get_decoder()`), ignoring experts lists and any list nested inside another such list (an MoE model can
    have more experts per block than blocks). Searching the whole model, vision / audio / encoder stacks
    lose to any other stack; an encoder-only model (BERT) gets its single stack. One block is a stack.
    (None, 0) when there is none."""
    root, decoder = _language_roots(model)
    candidates = []
    for name, module in decoder.named_modules():
        name = f"{root}.{name}" if root and name else (root or name)
        if not _is_block_stack(module):
            continue
        if any("expert" in part.lower() for part in name.split(".")):
            continue
        candidates.append((name, len(module)))
    # A list inside a block of another candidate list (`layers.0.<...>`) is part of that block, not the stack.
    outer = [
        (name, n)
        for name, n in candidates
        if not any(o != name and o and name.startswith(o + ".") for o, _ in candidates)
    ]
    if not root:
        language = [
            (name, n)
            for name, n in outer
            if not any(p.startswith(_NON_DECODER_PARTS) for p in name.lower().split("."))
        ]
        outer = language or outer
    best = (None, 0)
    for name, n in outer:
        # A tie goes to the shallower list.
        if n > best[1] or (
            n == best[1] and best[0] is not None and name.count(".") < best[0].count(".")
        ):
            best = (name, n)
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
