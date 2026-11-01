# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-stage loading for pipeline parallelism: each rank reads only the decoder layers its stages keep.

Inside `pipeline_sharded_loading(pp_mesh, schedule, config)` every transformers `from_pretrained` builds the
model on meta, derives the split apply_pipeline_parallel will use (weights only, so later LoRA does not move
it) and reads this rank's decoder layers plus the non-layer modules; the embedding only on the first stage's
rank, the head only on the last's (a tied pair where either end lives). Every other weight is a storage-free
zero-stride placeholder, so a model larger than one GPU can be split. apply_pipeline_parallel must use the
same split (checked); a forward before it is refused. Weight-derived trainable tensors sync only from ranks
that read the weight; weight-reading LoRA inits (PiSSA, OLoRA, MiCA, LoftQ, CorDA, LoRA-GA) skip placeholders
and calibration-forward inits are refused. 16-bit checkpoints only.

    mesh = init_device_mesh("cuda", (pp,), mesh_dim_names = ("pp",))
    with pipeline_sharded_loading(mesh["pp"], schedule = "1f1b"):
        model, tokenizer = FastLanguageModel.from_pretrained(name, load_in_4bit = False, ...)
    model = FastLanguageModel.get_peft_model(model, ...)
    pp = apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 4, schedule = "1f1b")
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

__all__ = [
    "pipeline_sharded_loading",
    "load_pipeline_parallel",
    "check_preloaded_split",
]

_MARK = "_unsloth_pp_preload"
_GUARDS = "_unsloth_pp_preload_guards"
_PLACEHOLDER = "_unsloth_pp_placeholder"
_QUANTIZED_HINT = (
    "4-bit / 8-bit QLoRA loads the full quantized model and is split by apply_pipeline_parallel: drop "
    "pipeline_sharded_loading for it, or pass load_in_4bit = False."
)


def _placeholder_param(old, device):
    value = torch.zeros((), dtype = old.dtype, device = device).expand(old.shape)
    new = nn.Parameter(value, requires_grad = old.requires_grad)
    setattr(new, _PLACEHOLDER, True)
    return new


def is_placeholder(t):
    return bool(getattr(t, _PLACEHOLDER, False))


def _refuse_unsplit_forward(
    module,
    args,
    kwargs = None,
):
    raise RuntimeError(
        "Unsloth: this model was loaded with pipeline_sharded_loading (each rank holds only its stages' "
        "layers); call apply_pipeline_parallel(model, pp_mesh, ...) with the same mesh, schedule and split "
        "after get_peft_model and before running it. LoRA initializations that run calibration forwards "
        "(init_lora_weights = 'corda' / 'eva' / 'lora_ga') need the whole model: load it without "
        "pipeline_sharded_loading, then apply_pipeline_parallel."
    )


# PEFT LoraLayer initializers that read the base weight (and most rewrite it with a dense residual).
_WEIGHT_INITS = (
    "pissa_init",
    "olora_init",
    "mica_init",
    "corda_init",
    "loftq_init",
    "lora_ga_init",
)
_SKIPS_PLACEHOLDER = "_unsloth_pp_skips_placeholder"


def base_is_placeholder(lora_layer):
    weight = getattr(lora_layer.get_base_layer(), "weight", None)
    return weight is not None and is_placeholder(weight)


def _placeholder_aware(name, original):
    def init(self, adapter_name, *args, **kwargs):
        if not base_is_placeholder(self):
            return original(self, adapter_name, *args, **kwargs)
        # Another rank owns this layer: computing on the zero placeholder would densify it (fp32 copy plus a
        # full residual) for an adapter the split drops. Keep the global RNG in step with a full load:
        # PEFT's pissa_niter_N draws one sketch through torch.svd_lowrank (the other inits draw nothing).
        init_lora_weights = args[0] if args else kwargs.get("init_lora_weights")
        if (
            name == "pissa_init"
            and isinstance(init_lora_weights, str)
            and "_niter_" in init_lora_weights
        ):
            weight = self.get_base_layer().weight
            torch.randn(
                min(weight.shape[-2:]),
                self.r[adapter_name],
                dtype = torch.float32,
                device = weight.device,
            )
        return None

    init.__name__ = init.__qualname__ = name
    init.__wrapped__ = original
    setattr(init, _SKIPS_PLACEHOLDER, True)
    return init


def install_placeholder_aware_lora_init():
    """Wrap PEFT's weight-dependent LoraLayer initializers to skip pipeline_sharded_loading placeholders (a no-op
    for every other layer). Idempotent; Unsloth's fast PiSSA / MiCA swap checks placeholders itself."""
    try:
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        return
    for name in _WEIGHT_INITS:
        original = LoraLayer.__dict__.get(name)
        if original is None or getattr(original, _SKIPS_PLACEHOLDER, False):
            continue
        setattr(LoraLayer, name, _placeholder_aware(name, original))


def _split_for(model, mesh, schedule, config):
    from .pp_native import PPConfig, _find_parts, _refuse_unsupported, derive_pp_split
    from .pp_schedules import resolve_stages_per_rank, schedule_spec, stage_indices

    config = config or PPConfig()
    spec = schedule_spec(schedule)
    v = resolve_stages_per_rank(spec, config.stages_per_rank)
    pp, rank = mesh.size(), mesh.get_local_rank()
    parts = _find_parts(model)
    # The structural refusals (MoE, MLA, ...) before anything is read. Every weight of a fresh load is
    # trainable until get_peft_model freezes it: those refusals run again at apply_pipeline_parallel.
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        _refuse_unsupported(model, parts)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    bounds = derive_pp_split(model, pp * v, config)
    local = stage_indices(spec, rank, pp, v)
    return parts, spec, v, bounds, tuple(local)


def _materialize(model, name, kwargs, mesh, plan, device):
    from .tp_native import _fill_from_checkpoint

    schedule, config = plan
    parts, spec, v, bounds, local = _split_for(model, mesh, schedule, config)
    total = len(bounds)
    owned = {i for k in local for i in range(*bounds[k])}
    holds_first, holds_last = 0 in local, total - 1 in local
    skip = set()
    for i, layer in enumerate(parts.layers):
        if i not in owned:
            skip.update(id(p) for p in layer.parameters())
    keep = set()
    if holds_first:
        keep.update(id(p) for p in parts.embed.parameters())
    if holds_last:
        keep.update(id(p) for p in parts.lm_head.parameters())
    for module in (parts.embed, parts.lm_head):
        skip.update(id(p) for p in module.parameters() if id(p) not in keep)
    placeholders = []

    def read_local(old, reader, key):
        if id(old) not in skip:
            return None
        new = _placeholder_param(old, device)
        placeholders.append(new)
        return new

    reader, skipped_numel, full_numel = _fill_from_checkpoint(
        model,
        name,
        kwargs,
        device,
        read_local,
        label = "pipeline parallel",
        fallback = "a full load, then apply_pipeline_parallel",
    )
    install_placeholder_aware_lora_init()
    guards = [model.register_forward_pre_hook(_refuse_unsplit_forward, with_kwargs = True)]
    for i, layer in enumerate(parts.layers):
        if i not in owned:
            guards.append(
                layer.register_forward_pre_hook(_refuse_unsplit_forward, with_kwargs = True)
            )
    model.__dict__[_GUARDS] = guards
    stats = {
        "pp": mesh.size(),
        "rank": mesh.get_local_rank(),
        "ranks": mesh.mesh.tolist(),
        "schedule": spec.name,
        "stages_per_rank": v,
        "bounds": [tuple(b) for b in bounds],
        "local_stages": local,
        "placeholder_tensors": len(placeholders),
        "placeholder_numel": skipped_numel,
        "loaded_numel": full_numel,
        "read_bytes": reader.full_bytes,
    }
    model.__dict__[_MARK] = stats
    return stats


def _preload_info(model):
    from .pp_native import _model_chain
    for level in _model_chain(model) if isinstance(model, nn.Module) else ():
        for module in level.modules():
            info = module.__dict__.get(_MARK)
            if info is not None:
                return module, info
    return None, None


def check_preloaded_split(model, pp_mesh, schedule, stages_per_rank, bounds, local_stages, parts):
    """A pipeline_sharded_loading model must be split as loaded (other layers are placeholders). Returns the
    loaded module or None."""
    holder, info = _preload_info(model)
    if info is None:
        return None
    want = {
        "ranks": pp_mesh.mesh.tolist(),
        "schedule": schedule,
        "stages_per_rank": stages_per_rank,
        "bounds": [tuple(b) for b in bounds],
        "local_stages": tuple(local_stages),
    }
    got = {k: info[k] for k in want}
    if got != want:
        diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        raise ValueError(
            "Unsloth: this model was loaded with pipeline_sharded_loading for another split (loaded, now): "
            f"{diff}. Pass apply_pipeline_parallel the mesh, schedule and PPConfig pipeline_sharded_loading "
            "used; the layers other stages own were never read."
        )
    return holder


def loaded_by(parts, spec, pp, stages_per_rank, bounds):
    """{id(parameter): pp ranks that read it} for parameters under a layer, the embedding or the head (the rest
    was read everywhere). Weight-derived trainable tensors are synced only among these ranks, never from a
    placeholder copy."""
    from .pp_schedules import stage_indices

    per_rank = [stage_indices(spec, r, pp, stages_per_rank) for r in range(pp)]
    owners = {}

    def add(module, ranks):
        for p in module.parameters():
            owners.setdefault(id(p), set()).update(ranks)

    for i, layer in enumerate(parts.layers):
        add(
            layer,
            {
                r
                for r, local in enumerate(per_rank)
                if any(bounds[k][0] <= i < bounds[k][1] for k in local)
            },
        )
    # A tied embedding / head is one parameter: read where either end lives (the union).
    add(parts.embed, {r for r, local in enumerate(per_rank) if 0 in local})
    add(parts.lm_head, {r for r, local in enumerate(per_rank) if len(bounds) - 1 in local})
    return {k: frozenset(v) for k, v in owners.items()}


def release_preload_guards(holder):
    for handle in holder.__dict__.pop(_GUARDS, ()):
        handle.remove()


@contextlib.contextmanager
def pipeline_sharded_loading(
    pp_mesh = None,
    schedule = "gpipe",
    config = None,
):
    """Every transformers `from_pretrained` inside reads only this rank's pipeline stages (see the module
    docstring). Pass apply_pipeline_parallel the same pp_mesh, schedule and config afterwards."""
    from .pp_schedules import schedule_spec
    from .tp_native import _sharded_from_pretrained

    schedule_spec(schedule)  # refuse an unknown schedule before loading anything
    if pp_mesh is None:
        pp_mesh = _default_load_mesh()
    if pp_mesh.ndim != 1:
        raise ValueError("Unsloth: pipeline_sharded_loading takes a 1-D pp mesh (pass mesh['pp']).")
    state = {
        "mesh": pp_mesh,
        "plan": (schedule, config),
        "materialize": _materialize,
        "label": "pipeline parallel",
        "quantized_hint": _QUANTIZED_HINT,
        "prequantized_hint": _QUANTIZED_HINT,
        "mesh_hint": "Unsloth: pipeline_sharded_loading takes a 1-D pp mesh (pass mesh['pp']).",
    }
    with _sharded_from_pretrained(state) as live:
        yield live


def _default_load_mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        raise RuntimeError(
            "Unsloth: pipeline_sharded_loading needs an initialized process group (torchrun) or a pp mesh."
        )
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (dist.get_world_size(),), mesh_dim_names = ("pp",))


def load_pipeline_parallel(
    model_name,
    pp_mesh = None,
    schedule = "gpipe",
    config = None,
    loader = None,
    **kwargs,
):
    """`loader(model_name, **kwargs)` under pipeline_sharded_loading (default FastLanguageModel.from_pretrained,
    16-bit). Returns what the loader returns; raises if no model went through the sharded path."""
    if loader is None:
        from unsloth import FastLanguageModel
        loader = FastLanguageModel.from_pretrained
        kwargs.setdefault("load_in_4bit", False)
    with pipeline_sharded_loading(pp_mesh, schedule, config) as state:
        out = loader(model_name, **kwargs)
    if not state["models"]:
        raise RuntimeError(
            "Unsloth: load_pipeline_parallel's loader did not call transformers from_pretrained; nothing was "
            "loaded sharded."
        )
    return out
