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
and calibration-forward inits are refused. 16-bit and pre-quantized bitsandbytes 4-bit checkpoints (a 4-bit
weight another rank owns is a storage-free Params4bit with its real QuantState shape).

Quantizing on load (a bitsandbytes 4-bit BitsAndBytesConfig as quantization_config, transformers 5): transformers
loads and quantizes every tensor as in a full load, and each tensor of a layer this rank's stages do not own (the
embedding / head likewise) becomes a storage-free placeholder right after it is set, so a rank holds its own
stages plus the tensor being converted. Owned weights and QuantStates are the full load's bytes. Under
UNSLOTH_PP_SIZE it runs only when the whole 4-bit load would not fit the GPU (UNSLOTH_PP_SHARDED_LOAD=1 forces it,
=0 keeps the full load).

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
    "Quantizing on load (load_in_4bit / 8-bit of a 16-bit checkpoint) loads the full model and is split by "
    "apply_pipeline_parallel: drop pipeline_sharded_loading for it, or load a pre-quantized bnb-4bit checkpoint."
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
        # Another rank owns this layer: a zero placeholder would densify it; keep the global RNG in
        # step with a full load (PEFT pissa_niter_N draws one sketch via torch.svd_lowrank).
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
    from .pp_native import (
        PPConfig,
        _find_parts,
        _refuse_unsupported,
        check_multi_stage_dp,
        derive_pp_split,
    )
    from .pp_schedules import resolve_stages_per_rank, schedule_spec, stage_indices

    config = config or PPConfig()
    spec = schedule_spec(schedule)
    v = resolve_stages_per_rank(spec, config.stages_per_rank)
    check_multi_stage_dp(mesh, v)  # apply_pipeline_parallel would refuse this split after the read
    pp, rank = mesh.size(), mesh.get_local_rank()
    parts = _find_parts(model)
    if parts.ple is not None:
        raise ValueError(
            "Unsloth: pipeline_sharded_loading does not split per-layer input tables (Gemma 4 E2B / E4B) yet; load "
            "the whole model, then apply_pipeline_parallel."
        )
    # The structural refusals (MoE, MLA, ...) before anything is read. Every weight of a fresh load is
    # trainable until get_peft_model freezes it: those refusals run again at apply_pipeline_parallel.
    # at_load: a VLM's text-only decision waits for the split (Trainer / UNSLOTH_PP_TEXT_ONLY); towers load whole.
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        _refuse_unsupported(model, parts, at_load = True, allow_moe = False)
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
        module = reader.quant.module(old) if reader.quant is not None else None
        if module is not None:
            new = reader.quant.placeholder(key, module, device)
            setattr(new, _PLACEHOLDER, True)
        else:
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
    """{id(parameter): pp ranks that read it} for params under a layer, embedding or head;
    weight-derived trainable tensors sync only among those ranks, never from a placeholder."""
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
    with _pipeline_loading(pp_mesh, schedule, config) as live:
        yield live


@contextlib.contextmanager
def _pipeline_loading(
    pp_mesh,
    schedule,
    config,
    auto = False,
):
    """pipeline_sharded_loading; `auto`: only checkpoints larger than the GPU (UNSLOTH_PP_SIZE), `pp_mesh` then
    a callable returning it (read at the first from_pretrained)."""
    from .pp_schedules import schedule_spec
    from .tp_native import _sharded_from_pretrained, refuse_hf_distributed_kwargs

    spec = schedule_spec(schedule)  # refuse an unknown schedule before loading anything
    if pp_mesh is None:
        pp_mesh = _default_load_mesh()
    if not callable(pp_mesh) and pp_mesh.ndim != 1:
        raise ValueError("Unsloth: pipeline_sharded_loading takes a 1-D pp mesh (pass mesh['pp']).")
    from .pp_native import PPConfig, check_multi_stage_dp
    from .pp_schedules import resolve_stages_per_rank

    stages = resolve_stages_per_rank(spec, (config or PPConfig()).stages_per_rank)
    if not callable(pp_mesh):
        check_multi_stage_dp(pp_mesh, stages)
    state = {
        "mesh": pp_mesh,
        "plan": (schedule, config),
        "materialize": _materialize,
        "label": "pipeline parallel",
        "quantized_hint": _QUANTIZED_HINT,
        "prequantized_hint": _QUANTIZED_HINT,
        "gguf_hint": (
            "Drop pipeline_sharded_loading: apply_pipeline_parallel (or UNSLOTH_PP_SIZE) then splits the fully "
            "loaded model, gguf_packed = True included."
        ),
        "mesh_hint": "Unsloth: pipeline_sharded_loading takes a 1-D pp mesh (pass mesh['pp']).",
        "auto": auto,
        # Pre-quantized bnb 4-bit (Inkling 2) converted on load; per-stage as quantize-on-load.
        "prequantized_fallback": lambda original,
        cls,
        name,
        args,
        kwargs,
        mesh,
        live: _load_quantizing(original, cls, name, args, kwargs, live),
    }
    from transformers import PreTrainedModel

    real = PreTrainedModel.__dict__[
        "from_pretrained"
    ].__func__  # transformers' own (before any swap)
    with _sharded_from_pretrained(state) as live:
        delegate = PreTrainedModel.__dict__["from_pretrained"]
        quantizing = {"active": False}

        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            if not live["active"] and not quantizing["active"]:
                refuse_hf_distributed_kwargs(kwargs, "pipeline parallel", "apply_pipeline_parallel")
                if stages > 1 and callable(live["mesh"]):
                    # A mesh factory: resolve it now, before the checkpoint is sized or read.
                    mesh = live["mesh"]()
                    check_multi_stage_dp(mesh, stages)
                    live["mesh"] = mesh
            if (
                not live["active"]
                and not quantizing["active"]
                and _bnb4_on_load(cls, pretrained_model_name_or_path, kwargs)
                and _quantize_sharded(live, pretrained_model_name_or_path, kwargs)
            ):
                quantizing["active"] = True
                try:
                    return _load_quantizing(
                        real, cls, pretrained_model_name_or_path, args, kwargs, live
                    )
                finally:
                    quantizing["active"] = False
            return delegate.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

        PreTrainedModel.from_pretrained = classmethod(from_pretrained)
        try:
            yield live
        finally:
            PreTrainedModel.from_pretrained = delegate


# ------------------------------------------------------------------------- quantize on load, per stage


def _bnb4_on_load(cls, name, kwargs):
    """This from_pretrained quantizes a not pre-quantized checkpoint to bitsandbytes 4-bit through a 4-bit
    BitsAndBytesConfig (what Unsloth's loaders pass) on a transformers that sets every loaded tensor through
    core_model_loading.set_param_for_module (5.x). Bare load_in_4bit / 8-bit kwargs and transformers 4.x keep
    the previous behaviour (whole load under UNSLOTH_PP_SIZE, the refusal under pipeline_sharded_loading)."""
    if kwargs.get("load_in_8bit") or kwargs.get("load_in_4bit"):
        return False
    asked = kwargs.get("quantization_config")
    if asked is None:
        return False
    try:
        from transformers import core_model_loading as loading
    except Exception:
        return False
    if not callable(getattr(loading, "set_param_for_module", None)):
        return False
    from .bnb_sharded_load import _as_dict, checkpoint_quantization

    try:
        qc = _as_dict(asked) or {}
    except Exception:
        return False
    method = qc.get("quant_method", "")
    method = str(getattr(method, "value", method)).lower()
    if not (
        method == "bitsandbytes"
        and (qc.get("load_in_4bit") or qc.get("_load_in_4bit"))
        and not (qc.get("load_in_8bit") or qc.get("_load_in_8bit"))
    ):
        return False
    try:
        _, prequantized = checkpoint_quantization(cls, name, dict(kwargs))
    except Exception:
        prequantized = None
    return prequantized is None  # a pre-quantized bnb-4bit checkpoint takes the sliced read instead


def _quantized_load_bytes(name, kwargs):
    """Bytes of a whole bitsandbytes 4-bit load of name: 2-D linear and 3-D expert weights ~0.53
    bytes/param (codes + absmax) unless config skips them; packed/fp8 as stored, rest 16-bit."""
    import collections
    import math

    from safetensors import safe_open

    from ..parallel_auto import _DTYPE_BYTES, _is_quantizable
    from .bnb_sharded_load import _as_dict
    from .tp_native import _checkpoint_index

    try:
        qc = _as_dict(kwargs.get("quantization_config")) or {}
    except Exception:
        qc = {}
    skip = [m for m in (qc.get("llm_int8_skip_modules") or ()) if isinstance(m, str)]
    dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
    dense = 4 if dtype in (torch.float32, "float32") else 2
    by_file = collections.defaultdict(list)
    for key, path in _checkpoint_index(name, kwargs).items():
        by_file[path].append(key)
    total = 0
    for path, keys in by_file.items():
        with safe_open(path, framework = "pt", device = "cpu") as f:
            for key in keys:
                piece = f.get_slice(key)
                dt, shape = str(piece.get_dtype()).upper(), tuple(piece.get_shape())
                numel = math.prod(shape) if shape else 1
                if not dt.startswith(("F", "BF")) or dt.startswith("F8"):
                    total += numel * _DTYPE_BYTES.get(dt, 1)
                    continue
                module = key[: -len(".weight")] if key.endswith(".weight") else key
                stacked = len(shape) == 3 and "expert" in key.lower()
                if (_is_quantizable(key, shape) or stacked) and not any(m in module for m in skip):
                    total += int(numel * 0.53)
                else:
                    total += numel * dense
    return total


def _quantize_sharded(state, name, kwargs):
    """Whether this 4-bit quantize-on-load reads per stage: always under pipeline_sharded_loading (unless
    UNSLOTH_PP_SHARDED_LOAD=0); under UNSLOTH_PP_SIZE (auto) when UNSLOTH_PP_SHARDED_LOAD=1, or when the whole
    4-bit load exceeds 90% of the free memory of some rank's GPU (a collective over the pp mesh)."""
    import os

    flag = str(os.environ.get("UNSLOTH_PP_SHARDED_LOAD", "")).strip()
    if flag == "0":
        return False
    if callable(state["mesh"]):
        state["mesh"] = state["mesh"]()
    mesh = state["mesh"] if state["mesh"] is not None else _default_load_mesh()
    state["mesh"] = mesh
    if mesh.ndim != 1:
        return False
    if flag == "1" or not state.get("auto"):
        return True
    from .tp_native import _load_device

    device = _load_device(kwargs.get("device_map"), mesh)
    if device.type != "cuda":
        return False
    try:
        nbytes = _quantized_load_bytes(name, kwargs)
    except Exception:
        nbytes = 0
    free = torch.cuda.mem_get_info(device)[0]
    exceeds = torch.tensor([int(nbytes > 0.9 * free)], device = device)
    import torch.distributed as dist

    dist.all_reduce(exceeds, op = dist.ReduceOp.MAX, group = mesh.get_group())
    return bool(exceeds.item())


def _zero_like(t):
    return torch.zeros((), dtype = t.dtype, device = t.device).expand(t.shape)


def _quantized_placeholder(p, module):
    """Zero-stride bytes/absmax, real QuantState; split costs, LoRA shapes match the full load."""
    import copy

    new = torch.Tensor._make_subclass(type(p), _zero_like(p.data), False)
    for attr, value in p.__dict__.items():
        new.__dict__[attr] = value
    qs = getattr(p, "quant_state", None)
    if qs is not None:
        qs = copy.copy(qs)
        if isinstance(getattr(qs, "absmax", None), torch.Tensor):
            qs.absmax = _zero_like(qs.absmax)
        state2 = getattr(qs, "state2", None)
        if state2 is not None:
            state2 = copy.copy(state2)
            if isinstance(getattr(state2, "absmax", None), torch.Tensor):
                state2.absmax = _zero_like(state2.absmax)
            qs.state2 = state2
        new.quant_state = qs
        if module is not None and hasattr(module, "quant_state"):
            module.quant_state = qs
    return new


def _ties_embeddings(model, parts, names):
    """Whether the lm_head shares the input embedding's weight once the load finishes. The quantize-on-load model is
    built with tying off and set_param may already have replaced one end, so the parameter identity alone cannot tell:
    the config's tie_word_embeddings plus the class's tied-weight declaration naming the head decide
    (``_tied_weights_keys``: a {target: source} dict on transformers 5.x, a list of names on 4.x)."""
    weight = getattr(parts.embed, "weight", None)
    if weight is not None and weight is getattr(parts.lm_head, "weight", None):
        return True
    config = getattr(model, "config", None)
    flags = [getattr(config, "tie_word_embeddings", None)]
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        try:
            flags.append(getattr(get_text(), "tie_word_embeddings", None))
        except Exception:
            pass
    if not any(flag is True for flag in flags):
        return False
    head = names.get(id(parts.lm_head), "") or "lm_head"
    head_leaf = head.rsplit(".", 1)[-1]
    for cls in type(model).__mro__:
        keys = cls.__dict__.get("_tied_weights_keys")
        if keys is None:
            continue
        if isinstance(keys, dict):
            keys = list(keys)
        for key in keys or ():
            key = str(key)
            if (
                key.startswith(head + ".")
                or key.split(".")[0] == head_leaf
                or f"{head_leaf}." in key
            ):
                return True
        return False  # the most derived declaration wins (an empty one: never tied)
    return False


def _module_names(root):
    return {id(m): name for name, m in root.named_modules(remove_duplicate = False)}


def _load_quantizing(real, cls, name, args, kwargs, state):
    """Bnb 4-bit quantize-on-load per stage; off-stage tensors become placeholders when set."""
    from transformers import core_model_loading as loading

    mesh = state["mesh"]
    schedule, config = state["plan"]
    set_param = loading.set_param_for_module
    plan = {}
    counts = {"loaded_bytes": 0, "kept_bytes": 0, "placeholder_tensors": 0, "placeholder_numel": 0}

    def stage_plan(model):
        # The meta model as transformers built it (bitsandbytes Linear4bit already swapped in): the split
        # apply_pipeline_parallel derives later (QuantState shapes give the same layer costs). Pure: the
        # caller decides which plan it fills.
        parts, spec, v, bounds, local = _split_for(model, mesh, schedule, config)
        owned = {i for k in local for i in range(*bounds[k])}
        holds_first, holds_last = 0 in local, len(bounds) - 1 in local
        names = _module_names(model)
        prefixes = [
            names[id(layer)] + "." for i, layer in enumerate(parts.layers) if i not in owned
        ]
        tied = _ties_embeddings(model, parts, names)
        for module, holds in ((parts.embed, holds_first), (parts.lm_head, holds_last)):
            if not (holds or (tied and (holds_first or holds_last))):
                prefixes.append(names[id(module)] + ".")
        return dict(
            model = model,
            parts = parts,
            spec = spec,
            v = v,
            bounds = bounds,
            local = local,
            owned = owned,
            prefixes = tuple(p for p in prefixes if p != "."),
        )

    def set_and_drop(model, target_name, param_value, *rest, **more):
        set_param(model, target_name, param_value, *rest, **more)
        if plan.get("model") is not model:
            # Only the setter fills `plan`: the post-load check below reads it as proof a tensor went through here.
            plan.clear()
            plan.update(stage_plan(model))
        module_path, _, pname = target_name.rpartition(".")
        try:
            module = model.get_submodule(module_path) if module_path else model
        except AttributeError:
            return
        p = module._parameters.get(pname)
        if not isinstance(p, torch.Tensor) or p.is_meta:
            return
        nbytes = p.numel() * p.element_size()
        counts["loaded_bytes"] += nbytes
        if not target_name.startswith(plan["prefixes"]):
            counts["kept_bytes"] += nbytes
            return
        if any(c.__name__ == "Params4bit" for c in type(p).__mro__):
            new = _quantized_placeholder(p, module)
        else:
            new = _placeholder_param(p, p.device)
            for attr in ("_is_hf_initialized",):
                if hasattr(p, attr):
                    setattr(new, attr, getattr(p, attr))
        setattr(new, _PLACEHOLDER, True)
        module._parameters[pname] = new
        counts["placeholder_tensors"] += 1
        counts["placeholder_numel"] += p.numel()

    warm = {}

    def keep(model, target_name):
        # Warmup sees the meta model; its own plan dict stops it passing the post-load check.
        if warm.get("model") is not model:
            warm.clear()
            warm.update(stage_plan(model))
        return not target_name.startswith(warm["prefixes"])

    from .fsdp2_load import sharded_load_memory

    loading.set_param_for_module = set_and_drop
    state["active"] = True
    try:
        with sharded_load_memory(mesh.size(), keep):
            loaded = real(cls, name, *args, **kwargs)
    finally:
        loading.set_param_for_module = set_param
        state["active"] = False
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    if plan.get("model") is not model:
        raise RuntimeError(
            "Unsloth: pipeline parallel quantize-on-load: transformers set no tensor through "
            "core_model_loading.set_param_for_module; load the whole model, then apply_pipeline_parallel."
        )
    parts, owned = plan["parts"], plan["owned"]
    # Placeholders that transformers re-set / tied away do not count.
    live = [p for p in model.parameters() if is_placeholder(p)]
    install_placeholder_aware_lora_init()
    guards = [model.register_forward_pre_hook(_refuse_unsplit_forward, with_kwargs = True)]
    for i, layer in enumerate(parts.layers):
        if i not in owned:
            guards.append(
                layer.register_forward_pre_hook(_refuse_unsplit_forward, with_kwargs = True)
            )
    model.__dict__[_GUARDS] = guards
    model.__dict__[_MARK] = {
        "pp": mesh.size(),
        "rank": mesh.get_local_rank(),
        "ranks": mesh.mesh.tolist(),
        "schedule": plan["spec"].name,
        "stages_per_rank": plan["v"],
        "bounds": [tuple(b) for b in plan["bounds"]],
        "local_stages": plan["local"],
        "quantized_on_load": True,
        "placeholder_tensors": len(live),
        "placeholder_numel": counts["placeholder_numel"],
        "loaded_numel": None,
        "read_bytes": counts["loaded_bytes"],
        "kept_bytes": counts["kept_bytes"],
    }
    state["models"].append(model)
    return loaded


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
