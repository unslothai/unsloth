# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallel, one or more stages per rank (every torch schedule, pp_schedules.py); LoRA first; order
pp outermost then dp then tp, split before parallelizing; save_adapter is collective, rank 0 writes; param
names unchanged."""

from __future__ import annotations

import ast
import collections
import dataclasses
import inspect
import math
import re
import sys
import textwrap
import types
import warnings
from typing import Optional

import torch
import torch.nn as nn

from .pp_schedules import (
    SCHEDULE_NAMES,
    allow_split_backward,
    compiled_blocks,
    non_reentrant_checkpointing_only,
    split_backward_keeps_function_nodes,
    single_backward_kernels,
    resolve_stages_per_rank,
    schedule_spec,
    stage_indices,
    torch_schedule_class,
    validate_schedule,
)

__all__ = [
    "PPConfig",
    "PipelineParallel",
    "apply_pipeline_parallel",
    "check_mesh_composition",
    "check_pp_dp_composition",
    "derive_pp_split",
    "is_pipeline_parallel",
    "pipeline_stage_of",
    "split_layers",
]

_PP_FLAG = "_unsloth_pp"
_SCHEDULES = SCHEDULE_NAMES
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# Layers reading another layer's state, or extra per-layer inputs: a stage boundary would cut them.
_CROSS_LAYER_CONFIG_KEYS = (
    "num_kv_shared_layers",
    "altup_num_inputs",
    "hidden_size_per_layer_input",
    "kv_lora_rank",
)
_ATTENTION_LAYER_TYPES = {"full_attention", "sliding_attention", "chunked_attention", "attention"}
_STEP_KWARGS = ("attention_mask", "position_ids")
_MULTI_STAGE_DP = (
    "Unsloth: native FSDP2 data parallel over a pipeline with several stages per rank is not supported "
    "(each FSDP2 root must be one stage's module); use one stage per rank (schedule = 'gpipe' / '1f1b') "
    "with a dp mesh, or no dp mesh."
)
_SIMPLE = (bool, int, float, str, type(None))


@dataclasses.dataclass
class PPConfig:
    # Decoder layers per stage, e.g. [10, 6]; None = balanced by weight elements (lm_head on the last stage).
    layers_per_stage: Optional[list] = None
    # lm_head + loss cost relative to a decoder weight element (split balance only).
    lm_head_cost: float = 1.0
    # dtype of the hidden states sent between stages; None = the input embedding dtype.
    activation_dtype: Optional[torch.dtype] = None
    # Broadcast trainable parameters from pp rank 0 before splitting (every stage must start from one adapter).
    sync_trainable: bool = True
    # (micro-batch size, sequence length) shapes whose stage / schedule stay built.
    max_cached_shapes: int = 4
    # Stages per rank for multi-stage schedules (interleaved1f1b, loopedbfs, ...); None = the schedule's
    # default (1 for gpipe / 1f1b, 2 otherwise). The decoder is cut into pp x stages_per_rank stages.
    stages_per_rank: Optional[int] = None


class _StageOutput(BaseException):
    """Raised by the pass-through after a non-last stage's last layer: the stage output. A BaseException,
    so no `except Exception` inside a model forward swallows it."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden


def _unwrap(model):
    inner = model
    if hasattr(inner, "get_base_model"):
        inner = inner.get_base_model()
    return inner


def _text_config(model):
    config = getattr(_unwrap(model), "config", None)
    if config is None:
        return None
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else config


def _decoder_of(inner):
    get_decoder = getattr(inner, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder()
        except Exception:
            decoder = None
        if isinstance(decoder, nn.Module) and decoder is not inner:
            return decoder
    prefix = getattr(inner, "base_model_prefix", None)
    decoder = getattr(inner, prefix, None) if isinstance(prefix, str) else None
    return decoder if isinstance(decoder, nn.Module) and decoder is not inner else None


def _owner(root, target):
    """(parent module, attribute name) holding `target` inside `root`."""
    for name, module in root.named_modules():
        for attr, child in module._modules.items():
            if child is target:
                return module, attr
    return None, None


def _model_chain(model):
    """`model` and every wrapper level below it (PeftModel -> LoraModel -> CausalLM -> decoder)."""
    chain, m = [], model
    while isinstance(m, nn.Module) and all(m is not c for c in chain) and len(chain) < 8:
        chain.append(m)
        modules = m.__dict__.get("_modules") or {}
        m = modules.get("base_model") or modules.get("model")
    inner = _unwrap(model)
    for extra in (inner, _decoder_of(inner)):
        if isinstance(extra, nn.Module) and all(extra is not c for c in chain):
            chain.append(extra)
    return chain


def is_pipeline_parallel(model) -> bool:
    """True when apply_pipeline_parallel split `model` or a wrapper level below it."""
    if not isinstance(model, nn.Module):
        return False
    try:
        chain = _model_chain(model)
    except Exception:
        chain = [model]
    return any(m.__dict__.get(_PP_FLAG) is not None for m in chain)


def pipeline_stage_of(model):
    """The PipelineParallel of `model` (any wrapper level below it), else None."""
    if not isinstance(model, nn.Module):
        return None
    for level in _model_chain(model):
        stage = level.__dict__.get("_unsloth_pp_stage")
        if stage is not None:
            return stage
    return None


# ------------------------------------------------------------------------------- composed meshes


def _root_mesh(mesh):
    get = getattr(mesh, "_get_root_mesh", None)
    if callable(get):
        try:
            return get()
        except Exception:
            pass
    try:
        from torch.distributed.device_mesh import _mesh_resources
        return _mesh_resources.get_root_mesh(mesh)
    except Exception:
        return None


def check_mesh_composition(meshes, what = "pipeline parallel composition"):
    """`meshes` ({name: 1-D DeviceMesh}) must be dimensions of one init_device_mesh covering every rank:
    1-D, meeting only at this rank (orthogonal) and with sizes multiplying to the world size. Returns the
    world size covered; ValueError otherwise (nothing is changed)."""
    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    me = dist.get_rank() if dist.is_initialized() else 0
    names = sorted(meshes)
    sizes = {name: meshes[name].size() for name in names}
    for name in names:
        if meshes[name].ndim != 1:
            raise ValueError(
                f"Unsloth: {what} takes 1-D meshes (mesh['{name}']), got a {meshes[name].ndim}-D one."
            )
    ranks = {name: set(meshes[name].mesh.flatten().tolist()) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if ranks[a] & ranks[b] != {me}:
                raise ValueError(
                    f"Unsloth: {what}: the {a} mesh (ranks {sorted(ranks[a])}) and the {b} mesh (ranks "
                    f"{sorted(ranks[b])}) must be two dimensions of one init_device_mesh, meeting only at "
                    f"this rank ({me})."
                )
    roots = [_root_mesh(meshes[name]) for name in names]
    if all(r is not None for r in roots) and any(r is not roots[0] for r in roots):
        raise ValueError(
            f"Unsloth: {what}: the {', '.join(names)} meshes must come from one init_device_mesh(..., "
            f"mesh_dim_names = (...)) (e.g. mesh['pp'] and mesh['dp'] of the same mesh)."
        )
    covered = math.prod(sizes.values())
    if covered != world:
        raise ValueError(
            f"Unsloth: {what}: {' x '.join(f'{n} {sizes[n]}' for n in names)} = {covered} ranks, but the world "
            f"has {world}; every rank must be in one pipeline stage of one data / tensor parallel replica."
        )
    return covered


def check_pp_dp_composition(model, dp_mesh):
    """shard_model(model, FSDP2Config(mesh = dp_mesh)) on a model split by apply_pipeline_parallel: valid on
    the model apply_pipeline_parallel was given, once, with dp_mesh orthogonal to the pp mesh (and to the tp
    mesh, applied before) and pp x dp (x tp) covering the world. Returns the PipelineParallel; ValueError
    before anything is sharded otherwise."""
    stage = pipeline_stage_of(model)
    if stage is None:
        raise ValueError("Unsloth: this model is not split by apply_pipeline_parallel.")
    if stage.model is not model:
        raise ValueError(
            "Unsloth: native FSDP2 over a pipeline parallel model needs the model apply_pipeline_parallel was "
            f"given ({type(stage.model).__name__}), not a level inside it ({type(model).__name__})."
        )
    if stage.dp_mesh is not None:
        raise ValueError(
            "Unsloth: this pipeline stage is already sharded over a data parallel mesh."
        )
    if stage.stages_per_rank > 1:
        raise ValueError(_MULTI_STAGE_DP)
    if dp_mesh is None:
        raise ValueError(
            "Unsloth: native FSDP2 over a pipeline parallel model needs FSDP2Config(mesh = mesh['dp']), the "
            "data parallel dimension of the mesh whose mesh['pp'] apply_pipeline_parallel used."
        )
    meshes = {"pp": stage.mesh, "dp": dp_mesh}
    if stage.tp_mesh is not None:
        meshes["tp"] = stage.tp_mesh
    try:
        check_mesh_composition(meshes, "pipeline parallel x native FSDP2 data parallel")
    except ValueError as exc:
        root = _root_mesh(stage.mesh)
        names = getattr(root, "mesh_dim_names", None) or ()
        hint = ""
        if "tp" in names and "tp" not in meshes:
            hint = " Apply apply_tensor_parallel(model, mesh['tp']) before shard_model."
        raise ValueError(f"{exc}{hint}") from None
    return stage


# ----------------------------------------------------------------------------------------------- parts


@dataclasses.dataclass
class _Parts:
    inner: nn.Module  # the CausalLM
    decoder: nn.Module  # the base model holding embed / layers / norm
    layers: nn.ModuleList
    layers_name: str  # attribute of `decoder`
    embed: nn.Module
    lm_head: nn.Module
    norms: list  # [(name, module)]: other parameter-holding decoder children (final norm)
    source: str  # "transformers" (base_model_pp_plan) / "structural"
    tied: bool


def _has_params(module):
    return next(module.parameters(), None) is not None


def _find_parts(model) -> _Parts:
    inner = _unwrap(model)
    decoder = _decoder_of(inner)
    if decoder is None:
        raise ValueError(
            f"Unsloth: pipeline parallelism found no decoder inside {type(inner).__name__} "
            "(no get_decoder() / base_model_prefix module)."
        )
    text = _text_config(model)
    n = getattr(text, "num_hidden_layers", None)
    # A decoder-only use of an encoder-decoder config (BartForCausalLM): num_hidden_layers is the encoder's.
    counts = {n, getattr(text, "decoder_layers", None)} - {None}
    pp_plan = getattr(text, "base_model_pp_plan", None) or getattr(
        getattr(inner, "config", None), "base_model_pp_plan", None
    )
    layers_name, source = None, "structural"
    named = dict(decoder.named_children())
    if isinstance(pp_plan, dict) and pp_plan:
        missing = [k for k in pp_plan if k not in named]
        lists = [k for k in pp_plan if isinstance(named.get(k), nn.ModuleList)]
        if not missing and len(lists) == 1:
            layers_name, source = lists[0], "transformers"
    if layers_name is None:
        for name, child in named.items():
            if (
                isinstance(child, nn.ModuleList)
                and len(child)
                and (n is None or len(child) in counts)
            ):
                layers_name = name
                break
    if layers_name is None:
        raise ValueError(
            f"Unsloth: pipeline parallelism found no decoder layer list in {type(decoder).__name__} "
            f"(no direct ModuleList child of num_hidden_layers = {n} layers)."
        )
    layers = named[layers_name]
    embed = inner.get_input_embeddings() if hasattr(inner, "get_input_embeddings") else None
    lm_head = inner.get_output_embeddings() if hasattr(inner, "get_output_embeddings") else None
    if not isinstance(embed, nn.Module) or _owner(decoder, embed)[0] is None:
        raise ValueError(
            "Unsloth: pipeline parallelism needs the input embedding inside the decoder "
            f"({type(decoder).__name__})."
        )
    if not isinstance(lm_head, nn.Module) or _owner(inner, lm_head)[0] is None:
        raise ValueError(
            f"Unsloth: pipeline parallelism needs a causal LM head ({type(inner).__name__} has no "
            "get_output_embeddings module)."
        )
    if _owner(decoder, lm_head)[0] is not None:
        raise ValueError("Unsloth: pipeline parallelism v1 needs the lm_head outside the decoder.")
    norms = []
    for name, child in named.items():
        if child is layers or child is embed or not _has_params(child):
            continue
        if _owner(child, embed)[0] is not None or child is embed:
            continue
        norms.append((name, child))
    hidden = getattr(text, "hidden_size", None)
    vocab = getattr(text, "vocab_size", None)
    order = list(decoder._modules)
    others = []
    for name, child in norms:
        base = getattr(child, "original_module", child)  # PEFT modules_to_save wraps it
        norm_like = "norm" in name.lower() or "Norm" in type(base).__name__
        too_big = bool(hidden) and sum(p.numel() for p in base.parameters()) > 4 * hidden
        # A learned position / token-type table before the layers (GPT-2 wpe, OPT / BART / Whisper
        # embed_positions): indexed by positions, not tokens, so later stages (whose pre-layer result is
        # replaced by the received hidden states anyway) need nothing from it. A token-indexed table could
        # feed the layers a side input and stays refused.
        positional = (
            isinstance(base, nn.Embedding)
            and isinstance(vocab, int)
            and base.num_embeddings != vocab
            and order.index(name) < order.index(layers_name)
        )
        if positional:
            continue
        if not norm_like or too_big:
            others.append(name)
    if others:
        raise ValueError(
            "Unsloth: pipeline parallelism v1 only splits embed / layers / final norm / lm_head, but "
            f"{type(decoder).__name__} also holds {others} (side inputs or per-layer embeddings)."
        )
    for name, child in inner.named_children():
        if child is decoder or child is lm_head or not _has_params(child):
            continue
        if _owner(child, lm_head)[0] is not None:
            continue
        if _owner(child, decoder)[0] is not None:
            # A wrapper around the decoder (OPTModel.decoder, BartDecoderWrapper.decoder): fine when it holds
            # nothing else.
            inside = {id(p) for p in decoder.parameters()}
            if all(id(p) in inside for p in child.parameters()):
                continue
        raise ValueError(
            f"Unsloth: pipeline parallelism v1 found an extra module {name} ({type(child).__name__}) "
            f"next to the decoder of {type(inner).__name__} (multimodal towers / extra heads are v2)."
        )
    e_w = getattr(embed, "weight", None)
    h_w = getattr(lm_head, "weight", None)
    tied = e_w is not None and e_w is h_w
    return _Parts(inner, decoder, layers, layers_name, embed, lm_head, norms, source, tied)


def _refuse_shared_across_stages(parts, bounds):
    """A trainable parameter used by layers of two stages (a block shared by several layers, Zamba2 /
    ALBERT style) would be trained as two independent copies."""
    stage_of = {i: k for k, (start, end) in enumerate(bounds) for i in range(start, end)}
    owners = {}
    for i, layer in enumerate(parts.layers):
        for name, p in layer.named_parameters():
            stages, first, param = owners.setdefault(
                id(p), (set(), f"{parts.layers_name}.{i}.{name}", p)
            )
            stages.add(stage_of[i])
    shared = [first for stages, first, p in owners.values() if len(stages) > 1 and p.requires_grad]
    if shared:
        raise ValueError(
            f"Unsloth: pipeline parallelism v1 cannot split layers that share trainable parameters across "
            f"stages ({shared[:3]}: a block reused by several layers). Keep the sharing layers on one stage "
            "(layers_per_stage, when they are adjacent), or use native FSDP2 (UNSLOTH_FSDP2=1)."
        )


def _check_loss_normalization(inner):
    """PipelineParallel sums the micro-batch losses, each normalised by the token count of the whole batch
    (num_items_in_batch). A causal LM forward that drops num_items_in_batch takes a per-micro-batch mean, so the
    sum would be n_microbatches times the one-process loss (BART / Whisper / TrOCR decoders)."""
    try:
        source = inspect.getsource(_forward_function(inner))
    except (OSError, TypeError):
        source = ""
    passes_kwargs = re.search(r"loss_function\([^()]*(?:\([^()]*\)[^()]*)*\*\*\s*\w+", source)
    if "num_items_in_batch" in source or passes_kwargs:
        return
    raise ValueError(
        f"Unsloth: pipeline parallelism v1 needs the {type(inner).__name__} loss to take num_items_in_batch "
        "(self.loss_function(..., **kwargs)); this forward computes a per-micro-batch mean, so micro-batches "
        "cannot be summed like gradient accumulation."
    )


def _refuse_unsupported(model, parts):
    text = _text_config(model)
    inner = parts.inner
    config = getattr(inner, "config", None)
    for key in ("vision_config", "audio_config", "speech_config"):
        if getattr(config, key, None) is not None:
            raise ValueError(
                f"Unsloth: pipeline parallelism v1 supports text decoders only ({type(inner).__name__} has "
                f"{key}). Use native FSDP2 (UNSLOTH_FSDP2=1) instead."
            )
    for key in _MOE_CONFIG_KEYS:
        value = getattr(text, key, None)
        if isinstance(value, int) and value > 0:
            raise ValueError(
                f"Unsloth: pipeline parallelism v1 does not support mixture-of-experts models ({key} = "
                f"{value}): the router auxiliary loss is not carried between stages. Use native FSDP2 "
                "(UNSLOTH_FSDP2=1) instead."
            )
    for key in _CROSS_LAYER_CONFIG_KEYS:
        value = getattr(text, key, None)
        if value:
            what = "multi-head latent attention" if key == "kv_lora_rank" else "cross-layer state"
            raise ValueError(
                f"Unsloth: pipeline parallelism v1 does not support {what} ({key} = {value})."
            )
    layer_types = getattr(text, "layer_types", None)
    if isinstance(layer_types, (list, tuple)):
        odd = sorted({t for t in layer_types if t not in _ATTENTION_LAYER_TYPES})
        if odd:
            raise ValueError(
                f"Unsloth: pipeline parallelism v1 does not support hybrid layer stacks (layer_types {odd})."
            )
    classes = {type(layer).__name__ for layer in parts.layers}
    if len(classes) > 1:
        raise ValueError(
            "Unsloth: pipeline parallelism v1 needs one decoder layer class, got "
            f"{sorted(classes)} (non-homogeneous stacks are v2)."
        )
    for name, module in parts.layers.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        cls = type(module).__name__
        if leaf == "experts" or "Experts" in cls or "SparseMoe" in cls or "MoE" in cls:
            raise ValueError(
                f"Unsloth: pipeline parallelism v1 does not support mixture-of-experts layers ({name}: {cls})."
            )
    for m in _model_chain(model):
        if m.__dict__.get("_unsloth_tp") is not None:
            raise ValueError(
                "Unsloth: apply_pipeline_parallel must come before apply_tensor_parallel (each pipeline "
                "stage is tensor parallel inside): load the model again, call apply_pipeline_parallel(model, "
                "mesh['pp']), then apply_tensor_parallel(model, mesh['tp'])."
            )
    if any(type(m).__name__.startswith("FSDP") for m in model.modules()):
        raise ValueError(
            "Unsloth: apply_pipeline_parallel must come before FSDP2 (each pipeline stage is sharded over the "
            "data parallel mesh): call apply_pipeline_parallel(model, mesh['pp']) first, then "
            "shard_model(model, FSDP2Config(mesh = mesh['dp']))."
        )
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    if (
        any(isinstance(p, DTensor) for p in model.parameters())
        or (getattr(model, "tp_size", None) or 1) > 1
    ):
        raise ValueError(
            "Unsloth: this model is already distributed (DTensor parameters or a transformers tp_plan); "
            "load it without tp_plan before apply_pipeline_parallel."
        )
    if parts.tied and parts.embed.weight.requires_grad:
        raise ValueError(
            "Unsloth: pipeline parallelism v1 cannot train tied input / output embeddings (the first and "
            "last stage would each update their own copy). Freeze them, or untie them before splitting."
        )
    _check_loss_normalization(inner)


# ---------------------------------------------------------------------------------- layer call format


def _forward_function(module):
    forward = module.__dict__.get("forward") or type(module).forward
    forward = getattr(forward, "__func__", forward)
    seen = 0
    while hasattr(forward, "__wrapped__") and seen < 8:
        forward = forward.__wrapped__
        seen += 1
    return forward


def _layer_call_format(decoder, layers_name):
    """'tuple' when the decoder forward reads `out = layer(...); hidden = out[0]` (Unsloth fast forwards,
    transformers 4.x), 'tensor' when it reads `hidden = layer(...)` (transformers 5.x)."""
    try:
        source = textwrap.dedent(inspect.getsource(_forward_function(decoder)))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise ValueError(
            f"Unsloth: pipeline parallelism cannot read the {type(decoder).__name__} forward ({exc})."
        ) from None
    formats = set()
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or f"self.{layers_name}" not in ast.unparse(loop.iter):
            continue
        target = loop.target
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[-1]
        if not isinstance(target, ast.Name):
            continue
        var = target.id
        body = ast.Module(body = loop.body, type_ignores = [])
        text = ast.unparse(body)
        for node in ast.walk(body):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not (isinstance(call.func, ast.Name) and call.func.id == var):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            out = node.targets[0].id
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Name) and first.id == out:
                formats.add("tensor")
            elif f"{out}[0]" in text:
                formats.add("tuple")
        # A layer handed to a checkpoint function would run the pass-through inside the checkpoint.
        for node in ast.walk(body):
            if (
                isinstance(node, ast.Call)
                and any(var in ast.unparse(a) for a in node.args)
                and "checkpoint" in ast.unparse(node.func).lower()
            ):
                if "GradientCheckpointingLayer" not in text:
                    formats.add("checkpointed")
    if "checkpointed" in formats:
        if getattr(decoder, "gradient_checkpointing", False):
            raise ValueError(
                f"Unsloth: the {type(decoder).__name__} forward wraps every layer in a checkpoint call; "
                "pipeline parallelism v1 needs per-layer checkpointing (GradientCheckpointingLayer)."
            )
        formats.discard("checkpointed")
    if len(formats) != 1:
        raise ValueError(
            f"Unsloth: pipeline parallelism cannot tell how {type(decoder).__name__}.forward calls its "
            f"{layers_name} (found {sorted(formats) or 'no layer call'})."
        )
    return formats.pop()


# ----------------------------------------------------------------------------------------- the split


def _weight_elements(p):
    shape = getattr(getattr(p, "quant_state", None), "shape", None)
    if shape is not None:
        return int(math.prod(shape))
    return p.numel()


def _layer_cost(layer):
    total = sum(_weight_elements(p) for name, p in layer.named_parameters() if "lora_" not in name)
    return max(total, 1)


def split_layers(
    costs,
    num_stages,
    last_extra = 0.0,
    first_extra = 0.0,
):
    """Contiguous [start, end) ranges (>= 1 layer each) minimizing the largest stage cost."""
    n = len(costs)
    if num_stages < 1 or num_stages > n:
        raise ValueError(
            f"Unsloth: cannot split {n} decoder layers into {num_stages} pipeline stages (at least one "
            "layer per stage)."
        )
    prefix = [0.0]
    for c in costs:
        prefix.append(prefix[-1] + float(c))

    def cost(i, j, k):
        c = prefix[j] - prefix[i]
        if k == 0:
            c += first_extra
        if k == num_stages - 1:
            c += last_extra
        return c

    inf = float("inf")
    # best[k][j]: smallest max cost of stages 0..k covering layers [0, j)
    best = [[inf] * (n + 1) for _ in range(num_stages)]
    cut = [[0] * (n + 1) for _ in range(num_stages)]
    for j in range(1, n + 1):
        best[0][j] = cost(0, j, 0)
    for k in range(1, num_stages):
        for j in range(k + 1, n + 1):
            for i in range(k, j):
                value = max(best[k - 1][i], cost(i, j, k))
                if value < best[k][j] - 1e-9:
                    best[k][j], cut[k][j] = value, i
    bounds, j = [], n
    for k in range(num_stages - 1, 0, -1):
        i = cut[k][j]
        bounds.append((i, j))
        j = i
    bounds.append((0, j))
    return bounds[::-1]


def derive_pp_split(
    model,
    num_stages,
    config: Optional[PPConfig] = None,
):
    """[(start, end)] decoder layer ranges per stage (dry run: no change to the model)."""
    config = config or PPConfig()
    parts = _find_parts(model)
    n = len(parts.layers)
    if config.layers_per_stage is not None:
        sizes = [int(s) for s in config.layers_per_stage]
        if len(sizes) != num_stages or sum(sizes) != n or min(sizes) < 1:
            raise ValueError(
                f"Unsloth: layers_per_stage {sizes} must give {num_stages} stages >= 1 layer summing to {n}."
            )
        bounds, start = [], 0
        for s in sizes:
            bounds.append((start, start + s))
            start += s
        return bounds
    costs = [_layer_cost(layer) for layer in parts.layers]
    head = parts.lm_head.weight if hasattr(parts.lm_head, "weight") else None
    last_extra = float(config.lm_head_cost) * (_weight_elements(head) if head is not None else 0)
    return split_layers(costs, num_stages, last_extra = last_extra)


# ------------------------------------------------------------------------------- stage replacements


def _simple_attrs(module):
    out = {}
    for key, value in module.__dict__.items():
        if key.startswith("_") or key in ("training", "gradient_checkpointing"):
            continue
        if isinstance(value, _SIMPLE) or (
            isinstance(value, (tuple, list)) and all(isinstance(v, _SIMPLE) for v in value)
        ):
            out[key] = value
    return out


try:
    from transformers.modeling_layers import GradientCheckpointingLayer as _CheckpointLayer
except (
    Exception
):  # transformers without GradientCheckpointingLayer: forwards checkpoint by isinstance
    _CheckpointLayer = nn.Module


class _PPPassThrough(_CheckpointLayer):
    """A decoder layer another stage owns. Never checkpointed (GradientCheckpointingLayer with the flag
    pinned off), so the hidden states it hands out stay in this stage's autograd graph."""

    gradient_checkpointing = property(lambda self: False, lambda self, value: None)

    def __init__(self, index, ctx, source):
        nn.Module.__init__(self)
        for key, value in _simple_attrs(source).items():
            self.__dict__[key] = value
        for sub in ("self_attn", "attention", "attn", "mlp"):
            child = source._modules.get(sub)
            if child is not None:
                self.__dict__[sub] = types.SimpleNamespace(**_simple_attrs(child))
        self.__dict__["_pp_index"] = index
        self.__dict__["_pp_ctx"] = ctx
        self.__dict__["_unsloth_skip_checkpoint"] = True

    def forward(self, *args, **kwargs):
        return self._pp_ctx.pass_through(self._pp_index, args, kwargs)

    def extra_repr(self):
        return f"index={self._pp_index}, owned by another pipeline stage"


def _placeholder(weight):
    """Same shape / dtype / device as `weight`, no storage (zero stride)."""
    return torch.zeros((), dtype = weight.dtype, device = weight.device).expand(weight.shape)


class _PPEmbeddingStub(nn.Module):
    """The input embedding on stages > 0: zeros in its place (replaced before the first owned layer)."""

    def __init__(self, source):
        super().__init__()
        for key, value in _simple_attrs(source).items():
            self.__dict__[key] = value
        weight = source.weight
        self.__dict__["weight"] = _placeholder(weight)
        scale = getattr(source, "embed_scale", None)
        if scale is not None:
            self.__dict__["embed_scale"] = scale
        self.__dict__["_pp_width"] = weight.shape[-1]

    def forward(self, input_ids, *args, **kwargs):
        return self.weight.new_zeros(*input_ids.shape, self._pp_width)


class _PPHeadStub(nn.Module):
    """lm_head on stages before the last (never called: those stages stop after their last layer)."""

    def __init__(self, source):
        super().__init__()
        for key, value in _simple_attrs(source).items():
            self.__dict__[key] = value
        self.__dict__["weight"] = _placeholder(source.weight)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Unsloth: lm_head lives on the last pipeline stage.")


class _StageContext:
    def __init__(self, start, end, is_first, is_last, tuple_format):
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last
        self.tuple_format = tuple_format
        self.active = False
        self.stage_input = None
        # The step's token count, given to every stage forward from here rather than as a schedule kwarg:
        # a non-tensor kwarg reaches torch's split backward (zero bubble) as a stage input on torch < 2.13.
        self.num_items_in_batch = None

    def configure(self, start, end, is_first, is_last):
        """The stage about to run (several stages of one rank share this context, one runs at a time)."""
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last

    def pass_through(self, index, args, kwargs):
        if not self.active:
            raise RuntimeError(
                "Unsloth: this model is split for pipeline parallelism (apply_pipeline_parallel); "
                "train it with PipelineParallel.step, or save the adapter and use it on one GPU."
            )
        if index < self.start:
            hidden = self.stage_input
        else:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not self.is_last:
                raise _StageOutput(hidden)
        return (hidden,) if self.tuple_format else hidden


class _PPStageModule(nn.Module):
    """What PipelineStage runs: the whole model's forward on this stage's slice.

    With several stages per rank, `stage` = (start, end, is_first, is_last, swaps): the context is set to this
    stage and, for the duration of the forward, `swaps` [(container, key, stand-in)] replace this rank's
    modules that belong to its other stages (their layers by pass-throughs, the embedding by its stub)."""

    def __init__(
        self,
        model,
        ctx,
        dtype,
        hidden_size,
        stage = None,
    ):
        super().__init__()
        self.model = model
        self.__dict__["_pp_ctx"] = ctx
        self.__dict__["_pp_dtype"] = dtype
        self.__dict__["_pp_hidden"] = hidden_size
        self.__dict__["_pp_warned"] = False
        self.__dict__["_pp_stage"] = stage

    def forward(self, x, **kwargs):
        stage = self._pp_stage
        if stage is None:
            return self._stage_forward(x, **kwargs)
        start, end, is_first, is_last, swaps = stage
        self._pp_ctx.configure(start, end, is_first, is_last)
        originals = [(container, key, container._modules[key]) for container, key, _ in swaps]
        for container, key, stand_in in swaps:
            container._modules[key] = stand_in
        try:
            return self._stage_forward(x, **kwargs)
        finally:
            for container, key, original in originals:
                container._modules[key] = original

    def _stage_forward(self, x, **kwargs):
        ctx = self._pp_ctx
        if ctx.is_first:
            input_ids = x
        else:
            input_ids = torch.zeros(x.shape[:2], dtype = torch.long, device = x.device)
            ctx.stage_input = x
        ctx.active = True
        try:
            # Pass-through layers return hidden states only: no per-layer attentions / hidden states.
            kwargs.update(output_attentions = False, output_hidden_states = False)
            if ctx.num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = ctx.num_items_in_batch
            out = self.model(input_ids = input_ids, use_cache = False, **kwargs)
        except _StageOutput as stop:
            hidden = stop.hidden
            if hidden.shape[-1] != self._pp_hidden or hidden.dim() != 3:
                raise RuntimeError(
                    f"Unsloth: pipeline stage output has shape {tuple(hidden.shape)}, expected "
                    f"[micro-batch, seq, {self._pp_hidden}]."
                ) from None
            if hidden.dtype != self._pp_dtype:
                if not self._pp_warned:
                    self.__dict__["_pp_warned"] = True
                    warnings.warn(
                        f"Unsloth: pipeline stage hidden states are {hidden.dtype}, sent as "
                        f"{self._pp_dtype} (PPConfig.activation_dtype).",
                        stacklevel = 2,
                    )
                hidden = hidden.to(self._pp_dtype)
            return hidden
        finally:
            ctx.active = False
            ctx.stage_input = None
        if not ctx.is_last:
            raise RuntimeError(
                "Unsloth: a pipeline stage finished its forward without reaching its boundary."
            )
        loss = getattr(out, "loss", None)
        if loss is None and isinstance(out, (tuple, list)):
            loss = out[0]
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            raise RuntimeError("Unsloth: the last pipeline stage needs labels to return a loss.")
        return loss


# ---------------------------------------------------------------------------------- 1F1B safety


def _gradient_checkpointing_on(model):
    return any(
        getattr(m, "gradient_checkpointing", False) is True for m in model.modules()
    ) or bool(getattr(_unwrap(model), "is_gradient_checkpointing", False))


def _zoo_slot_lease_present():
    """unsloth_zoo smart gradient checkpointing leases host offload slots per checkpoint (zoo PR #2071):
    without it, interleaved forwards / backwards (1F1B) overwrite offloaded activations."""
    module = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    if module is None:
        return None  # smart gradient checkpointing not loaded
    return hasattr(module, "_CpuSlotLease")


def _check_torch_pipelining():
    """The schedule calls pass scale_grads (torch >= 2.7) and step(return_outputs = ...) (torch >= 2.10).
    Probe the signatures, not the version string: on older torch the first step fails (2.6) or passes
    return_outputs into the model as a keyword (2.7 - 2.9) after this rank's stage was already cut out."""
    try:
        from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe
    except ImportError as e:
        raise RuntimeError(
            f"Unsloth: pipeline parallelism needs torch.distributed.pipelining (torch >= 2.10); "
            f"torch {torch.__version__} has none ({e})."
        ) from None
    for cls in (ScheduleGPipe, Schedule1F1B):
        init = inspect.signature(cls.__init__).parameters
        step = inspect.signature(cls.step).parameters
        if "scale_grads" not in init or "return_outputs" not in step:
            raise RuntimeError(
                f"Unsloth: pipeline parallelism needs torch >= 2.10 (Schedule scale_grads and "
                f"step(return_outputs)); torch {torch.__version__} lacks them. The model was not modified."
            )


def _check_schedule(model, spec, props):
    """Refusals for the schedule `spec` whose compute order has `props` (pp_schedules.validate_schedule)."""
    if (
        props["interleaved"]
        and _gradient_checkpointing_on(model)
        and _zoo_slot_lease_present() is False
    ):
        raise ValueError(
            f"Unsloth: the {spec.name} pipeline schedule interleaves forwards and backwards, which this "
            "unsloth_zoo's smart gradient checkpointing does not support (offloaded activations of a pending "
            "micro-batch would be overwritten). Use schedule = 'gpipe' (or 'loopedbfs'), or update unsloth_zoo "
            "(host slot leases, unsloth-zoo PR #2071)."
        )
    fused = single_backward_kernels(model) if props["split_backward"] else []
    if fused:
        raise ValueError(
            f"Unsloth: the {spec.name} pipeline schedule splits each backward into an input and a weight pass, "
            f"which runs the backward of Unsloth's fused LoRA kernels ({', '.join(fused)}) twice; they can run it "
            "once. Use schedule = 'interleaved1f1b' (or 'loopedbfs'), or load the model without the fused LoRA "
            "path."
        )
    if props["split_backward"] and not split_backward_keeps_function_nodes():
        compiled = compiled_blocks(model)
        if compiled:
            shown = ", ".join(compiled[:3]) + (
                f" and {len(compiled) - 3} more" if len(compiled) > 3 else ""
            )
            raise ValueError(
                f"Unsloth: the {spec.name} pipeline schedule splits each backward into an input and a weight "
                f"pass, which torch {torch.__version__} cannot run over torch.compile'd blocks ({shown}): its "
                "weight pass reads _input_metadata of the compiled (AOTAutograd) backward nodes after their graph "
                "root is released, which needs torch >= 2.13. Use schedule = 'interleaved1f1b' (or 'loopedbfs', "
                "a non zero-bubble schedule), disable the compiled blocks (UNSLOTH_BLOCK_COMPILE=0, UNSLOTH_COMPILE_DISABLE=1, no "
                "module.compile()), or upgrade torch. The model was not modified."
            )
    if props["split_backward"] and non_reentrant_checkpointing_only(model) is False:
        raise ValueError(
            f"Unsloth: the {spec.name} pipeline schedule splits each backward into an input and a weight pass "
            "(torch.autograd.grad), which reentrant gradient checkpointing (transformers' default, Unsloth "
            "smart checkpointing) cannot run. Turn gradient checkpointing off, use torch's non-reentrant "
            "checkpointing (gradient_checkpointing_kwargs = {'use_reentrant': False}), or use "
            "schedule = 'interleaved1f1b'."
        )


# ------------------------------------------------------------------------------------- the stage


def _refuse_generate(*args, **kwargs):
    raise RuntimeError(
        "Unsloth: generation is not supported on a pipeline parallel model in v1; save the adapter "
        "(PipelineParallel.save_adapter) and generate on one GPU."
    )


class PipelineParallel:
    """This rank's pipeline stage of a model split by apply_pipeline_parallel."""

    def __init__(
        self,
        model,
        mesh,
        parts,
        bounds,
        n_microbatches,
        schedule,
        config,
        tuple_format,
        dtype,
        stages_per_rank = 1,
    ):
        import torch.distributed as dist

        self.model = model
        self.mesh = mesh
        self.group = mesh.get_group()
        self.num_stages = mesh.size()  # pp ranks
        self.stage_index = (
            mesh.get_local_rank()
        )  # this rank's pp index (its stage with one stage per rank)
        self.bounds = bounds  # [start, end) decoder layers of every stage (pp x stages_per_rank)
        self.spec = schedule_spec(schedule)
        self.stages_per_rank = int(stages_per_rank)
        self.total_stages = self.num_stages * self.stages_per_rank
        if len(bounds) != self.total_stages:
            raise ValueError(
                f"Unsloth: {len(bounds)} layer ranges for {self.total_stages} pipeline stages."
            )
        self.local_stages = stage_indices(
            self.spec, self.stage_index, self.num_stages, self.stages_per_rank
        )
        self.start, self.end = bounds[self.local_stages[0]]
        self.n_microbatches = n_microbatches
        self.schedule = self.spec.name
        self.config = config
        self.parts = parts
        self._foreign_prefixes = ()  # set by apply_pipeline_parallel
        self.activation_dtype = dtype
        self.ctx = _StageContext(self.start, self.end, self.is_first, self.is_last, tuple_format)
        hidden = getattr(_text_config(model), "hidden_size", None) or parts.embed.weight.shape[-1]
        self.hidden_size = int(hidden)
        self.module = _PPStageModule(model, self.ctx, dtype, self.hidden_size)
        self.stage_modules = [self.module]  # one per local stage (set by apply_pipeline_parallel)
        self._global_ranks = [dist.get_global_rank(self.group, i) for i in range(self.num_stages)]
        last_owner = next(
            r
            for r in range(self.num_stages)
            if self.total_stages - 1
            in stage_indices(self.spec, r, self.num_stages, self.stages_per_rank)
        )
        self._last_global_rank = self._global_ranks[last_owner]
        self._shapes = collections.OrderedDict()
        self._scaler = None
        self._step_scaler = (
            None  # the enabled GradScaler of the last step (clip_grad_norm_ checks it)
        )
        self._mb_losses = []
        self._loss_multiplier = 1
        self.dp_mesh = None  # set by shard_model (FSDP2 over the data parallel dimension)

    # composition with FSDP2 (dp) / tensor parallelism (tp): every other mesh dimension must be applied
    @property
    def tp_mesh(self):
        for level in _model_chain(self.model):
            mesh = level.__dict__.get("_unsloth_tp_mesh")
            if mesh is not None:
                return mesh
        return None

    @property
    def composed(self):
        """True when the stage spans more ranks than its pp group (dp and / or tp)."""
        import torch.distributed as dist
        return dist.is_initialized() and dist.get_world_size() != self.num_stages

    def _meshes(self):
        meshes = {"pp": self.mesh}
        if self.dp_mesh is not None:
            meshes["dp"] = self.dp_mesh
        if self.tp_mesh is not None:
            meshes["tp"] = self.tp_mesh
        return meshes

    def _check_composed(self):
        if not self.composed:
            return
        meshes = self._meshes()
        try:
            check_mesh_composition(meshes)
        except ValueError as exc:
            names = getattr(_root_mesh(self.mesh), "mesh_dim_names", None) or ("tp", "dp")
            missing = (
                " then ".join(
                    call
                    for name, call in (
                        ("tp", "apply_tensor_parallel(model, mesh['tp'])"),
                        ("dp", "shard_model(model, FSDP2Config(mesh = mesh['dp']))"),
                    )
                    if name not in meshes and (name in names or not set(names) & {"tp", "dp"})
                )
                or "apply_tensor_parallel over the tp dimension / shard_model over the dp dimension"
            )
            raise RuntimeError(
                f"Unsloth: this pipeline stage spans {', '.join(meshes)} only ({exc}). Apply the other mesh "
                f"dimensions before training: {missing}."
            ) from None

    def _attach_data_parallel(
        self,
        dp_mesh,
        mp_policy = None,
    ):
        """shard_model(model, FSDP2Config(mesh = dp_mesh)) sharded this stage: make the PipelineStage's module
        the outer FSDP2 root (no parameters of its own), so torch's PipelineStage defers the gradient
        reduce-scatter to the end of the step (set_requires_gradient_sync(False) per micro-batch backward,
        one post_backward in perform_reduce_grad), as torchtitan's stage submodule does."""
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if self.stages_per_rank > 1:
            raise ValueError(_MULTI_STAGE_DP)
        policy = MixedPrecisionPolicy(
            param_dtype = None,
            reduce_dtype = getattr(mp_policy, "reduce_dtype", None),
            cast_forward_inputs = False,  # the received hidden states keep PPConfig.activation_dtype
        )
        fully_shard(self.module, mesh = dp_mesh, mp_policy = policy, reshard_after_forward = False)
        self.dp_mesh = dp_mesh
        self._shapes.clear()  # stages built before hold the unwrapped module

    @property
    def is_first(self):
        """This rank holds the first stage (the input embedding)."""
        return 0 in self.local_stages

    @property
    def is_last(self):
        """This rank holds the last stage (final norm, lm_head and the loss)."""
        return self.total_stages - 1 in self.local_stages

    @property
    def device(self):
        for p in self.module.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def parameters(self):
        """This stage's trainable parameters (disjoint across stages)."""
        return [p for p in self.module.parameters() if p.requires_grad]

    def __repr__(self):
        if self.stages_per_rank > 1:
            ranges = ", ".join(
                f"{k}: [{self.bounds[k][0]}, {self.bounds[k][1]})" for k in self.local_stages
            )
            return (
                f"PipelineParallel(rank {self.stage_index} / {self.num_stages}, stages {{{ranges}}} of "
                f"{self.total_stages}, {self.schedule}, n_microbatches = {self.n_microbatches})"
            )
        return (
            f"PipelineParallel(stage {self.stage_index} / {self.num_stages}, layers [{self.start}, "
            f"{self.end}), {self.schedule}, n_microbatches = {self.n_microbatches})"
        )

    # schedule per (micro-batch size, sequence length): static stage metadata
    def _loss_fn(self, output, target):
        self._mb_losses.append(output.detach())
        if self._loss_multiplier != 1:
            # FSDP2 averages gradients over dp: dp x (sum of this replica's token losses / global tokens)
            # averages to the global batch gradient.
            output = output * self._loss_multiplier
        return output if self._scaler is None else self._scaler.scale(output)

    def _schedule_for(self, mb, seq):
        from torch.distributed.pipelining import PipelineStage

        key = (mb, seq)
        if key in self._shapes:
            self._shapes.move_to_end(key)
            return self._shapes[key]
        device = self.device
        hidden = (mb, seq, self.hidden_size)
        stages = []
        for index, module in zip(self.local_stages, self.stage_modules):
            if index == 0:
                example_in = torch.empty((mb, seq), dtype = torch.long, device = "meta")
            else:
                # requires_grad is part of the static metadata: received hidden states need their grad.
                example_in = torch.empty(
                    hidden, dtype = self.activation_dtype, device = "meta", requires_grad = True
                )
            if index == self.total_stages - 1:
                example_out = torch.empty(
                    (), dtype = torch.float32, device = "meta", requires_grad = True
                )
            else:
                example_out = torch.empty(
                    hidden, dtype = self.activation_dtype, device = "meta", requires_grad = True
                )
            stages.append(
                PipelineStage(
                    module,
                    index,
                    self.total_stages,
                    device,
                    input_args = example_in,
                    output_args = example_out,
                    group = self.group,
                )
            )
        cls = torch_schedule_class(self.spec)
        schedule = cls(
            stages if self.spec.multi else stages[0],
            self.n_microbatches,
            loss_fn = self._loss_fn,
            scale_grads = False,
        )
        self._shapes[key] = schedule
        while len(self._shapes) > max(1, int(self.config.max_cached_shapes)):
            self._shapes.popitem(last = False)
        return schedule

    def step(
        self,
        input_ids,
        labels,
        num_items_in_batch = None,
        scaler = None,
        **kwargs,
    ):
        """Forward + backward of one optimizer step's batch (the same batch on every pp rank), split into
        n_microbatches micro-batches. Gradients accumulate into this stage's parameters (call
        optimizer.zero_grad between steps). Returns the batch loss (sum over tokens / num_items_in_batch,
        unscaled) on every rank.

        With FSDP2 over a dp mesh, each dp replica passes its own slice of the global batch (pp / tp peers the
        same one); num_items_in_batch is the token count of the whole global batch (default: this slice's
        count summed over dp) and the returned loss is the global batch loss."""
        import torch.distributed as dist

        self._check_composed()
        unknown = sorted(set(kwargs) - set(_STEP_KWARGS))
        if unknown:
            raise ValueError(
                f"Unsloth: PipelineParallel.step takes input_ids, labels, {', '.join(_STEP_KWARGS)}; got "
                f"{unknown} (packed sequences / extra inputs are v2)."
            )
        if labels is None:
            raise ValueError(
                "Unsloth: PipelineParallel.step needs labels (the last stage computes the loss)."
            )
        if input_ids.dim() != 2 or tuple(labels.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"Unsloth: PipelineParallel.step takes [batch, seq] input_ids and labels, got "
                f"{tuple(input_ids.shape)} / {tuple(labels.shape)}."
            )
        bsz, seq = input_ids.shape
        m = self.n_microbatches
        if bsz % m:
            raise ValueError(
                f"Unsloth: the batch size {bsz} must be a multiple of n_microbatches = {m}."
            )
        device = self.device
        dp_group = self.dp_mesh.get_group() if self.dp_mesh is not None else None
        if num_items_in_batch is None:
            # Tokens the loss averages over in one process: shifted labels that are not ignored.
            count = (labels[:, 1:] != -100).sum()
            if dp_group is not None:
                # ... over the whole global batch: every dp replica's slice.
                count = count.to(device = device, dtype = torch.int64)
                dist.all_reduce(count, group = dp_group)
            num_items_in_batch = int(count)
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = int(
                num_items_in_batch
            )  # a non-tensor kwarg is replicated per micro-batch
        num_items_in_batch = max(num_items_in_batch, 1)
        self._loss_multiplier = self.dp_mesh.size() if self.dp_mesh is not None else 1
        schedule = self._schedule_for(bsz // m, seq)
        step_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0 or value.shape[0] not in (1, bsz):
                    raise ValueError(
                        f"Unsloth: PipelineParallel.step {key} must have the batch dimension first "
                        f"({bsz} or 1), got {tuple(value.shape)}."
                    )
                if value.shape[0] == 1 and bsz > 1:
                    value = value.expand(bsz, *value.shape[1:])  # split per micro-batch on dim 0
                value = value.to(device)
            step_kwargs[key] = value
        step_kwargs["labels"] = labels.to(device)
        self.ctx.num_items_in_batch = num_items_in_batch
        if scaler is not None and scaler.is_enabled():
            scaler.scale(
                torch.zeros((), device = device)
            )  # every stage's scaler holds the scale (unscale_)
            self._scaler = scaler
            self._bind_scaler(scaler)
        self._step_scaler = scaler if scaler is not None and scaler.is_enabled() else None
        self._mb_losses = []
        args = (input_ids.to(device),) if self.is_first else ()
        target = step_kwargs["labels"] if self.is_last else None
        try:
            schedule.step(*args, target = target, return_outputs = False, **step_kwargs)
        finally:
            self._scaler = None
            self.ctx.num_items_in_batch = None
        if self.dp_mesh is not None:
            self._average_replicated_grads()
        if self.is_last:
            if len(self._mb_losses) != m:
                raise RuntimeError(
                    f"Unsloth: expected {m} micro-batch losses, got {len(self._mb_losses)}."
                )
            total = torch.stack([l.float() for l in self._mb_losses]).sum().reshape(1).to(device)
            self._mb_losses = []
        else:
            total = torch.zeros(1, dtype = torch.float32, device = device)
        if self.num_stages > 1:
            dist.broadcast(total, src = self._last_global_rank, group = self.group)
        if dp_group is not None:
            dist.all_reduce(total, group = dp_group)  # each replica's sum / global tokens
        return total[0]

    def _average_replicated_grads(self):
        """Trainable parameters FSDP2 keeps replicated (Unsloth averages them over dp like DDP from an
        autograd callback, deferred while gradient sync is off): the pipeline schedule runs every micro-batch
        backward with sync off and reduces FSDP2's own groups outside autograd, so average them here once.
        Averaging equal values is a no-op, so an earlier per-backward average stays correct."""
        params = self.model.__dict__.get("_unsloth_replicated_grad_params")
        if not params:
            return
        import torch.distributed as dist

        from .fsdp2_save import _sync_replicated_grads, model_mesh

        state = self.model.__dict__.setdefault("_unsloth_replicated_grad_state", {"queued": False})
        _sync_replicated_grads(
            params, dist, model_mesh(self.model) or self.dp_mesh, state.setdefault("buffers", {})
        )

    def _grad_copies(self, grad):
        """(local tensor, number of ranks holding the same values) for this rank's gradient. Stages hold
        disjoint parameters; inside a stage plain gradients are replicated over its world // pp ranks and
        a DTensor's local shard repeats over that count divided by its shard dimensions."""
        stage_ranks = torch.distributed.get_world_size() // self.num_stages if self.composed else 1
        try:
            from torch.distributed.tensor import DTensor, Replicate
        except ImportError:
            DTensor = None
        if DTensor is None or not isinstance(grad, DTensor):
            return grad, grad, stage_ranks
        if any(pl.is_partial() for pl in grad.placements):
            grad = grad.redistribute(
                grad.device_mesh,
                [Replicate() if pl.is_partial() else pl for pl in grad.placements],
            )
        mesh = grad.device_mesh
        shards = math.prod(
            mesh.size(i)
            for i, pl in enumerate(grad.placements)
            if pl.is_shard() or "StridedShard" in type(pl).__name__
        )
        if stage_ranks % mesh.size() or stage_ranks % shards:
            raise ValueError(
                f"Unsloth: a gradient on a {mesh.size()}-rank mesh does not fit this stage's {stage_ranks} ranks."
            )
        local = getattr(grad, "_local_tensor", None)
        return grad, (grad.to_local() if local is None else local), stage_ranks // shards

    def clip_grad_norm_(
        self,
        max_norm,
        parameters = None,
        eps = 1e-6,
    ):
        """Clip by the global 2-norm over every stage (squares summed over pp). Returns the norm.
        Composed with dp / tp: every gradient element counted once over the world (a local square sum
        divided by the ranks holding a copy of it, summed over every rank)."""
        import torch.distributed as dist

        self._check_unscaled()
        params = self.parameters() if parameters is None else list(parameters)
        params = [p for p in params if p.grad is not None]
        device = self.device
        sq = torch.zeros((), dtype = torch.float64, device = device)
        grads = []
        for p in params:
            full, local, copies = self._grad_copies(p.grad)
            if full is not p.grad:
                p.grad = full
            grads.append(local)
            part = local.detach().double().square().sum().to(device)
            sq = sq + (part / copies if copies != 1 else part)
        if self.composed:
            dist.all_reduce(sq)  # the meshes cover the world (checked by step)
        elif self.num_stages > 1:
            dist.all_reduce(sq, group = self.group)
        total = sq.sqrt()
        coef = torch.clamp(float(max_norm) / (total + eps), max = 1.0)
        for g in grads:
            g.mul_(coef.to(g.dtype))
        return total.float()

    def unscale_(self, scaler, optimizer):
        """scaler.unscale_(optimizer), then found_inf is the MAX over every stage, so all stages skip the
        same steps (scaler.step / scaler.update after this)."""
        scaler.unscale_(optimizer)
        if not scaler.is_enabled():
            return
        self._agree_found_inf(scaler, optimizer)

    def _bind_scaler(self, scaler):
        """Make scaler.unscale_ / scaler.step agree found_inf over every stage, or one stage steps while another skips.
        Bound until scaler.update(), so the scaler pickles again where torch.amp allows it and holds no pipeline."""
        import weakref

        bound = scaler.__dict__.get("_unsloth_pp_agree")
        if bound is not None and bound() is self:
            return
        cls = type(scaler)
        unscale = cls.unscale_.__get__(scaler)
        update = cls.update.__get__(scaler)
        pipeline = weakref.ref(self)

        def unscale_(optimizer):
            unscale(optimizer)
            pp = pipeline()
            if pp is not None and scaler.is_enabled():
                pp._agree_found_inf(scaler, optimizer)

        def update_(new_scale = None):
            try:
                return update(new_scale)
            finally:
                for name in ("unscale_", "update", "_unsloth_pp_agree"):
                    scaler.__dict__.pop(name, None)
                pp = pipeline()
                if pp is not None and pp._step_scaler is scaler:
                    pp._step_scaler = (
                        None  # this step's gradients were unscaled (update follows step)
                    )

        scaler.unscale_ = unscale_
        scaler.update = update_
        scaler._unsloth_pp_agree = pipeline

    def _check_unscaled(self):
        """clip_grad_norm_ on gradients still multiplied by the loss scale clips the wrong norm."""
        scaler = self._step_scaler
        if scaler is None or not scaler.is_enabled():
            return
        try:
            from torch.amp.grad_scaler import OptState
        except ImportError:
            return
        states = getattr(scaler, "_per_optimizer_states", {}) or {}
        if any(
            state.get("stage") in (OptState.UNSCALED, OptState.STEPPED) for state in states.values()
        ):
            return  # scaler.step unscales first: STEPPED gradients are unscaled too
        raise ValueError(
            "Unsloth: PipelineParallel.clip_grad_norm_ got gradients still multiplied by the float16 "
            "loss scale. Call scaler.unscale_(optimizer) (or pp.unscale_(scaler, optimizer)) first, as "
            "torch.amp requires before clipping."
        )

    def _agree_found_inf(self, scaler, optimizer):
        import torch.distributed as dist

        state = scaler._per_optimizer_states[id(optimizer)]
        found = state["found_inf_per_device"]
        device = self.device
        total = torch.zeros(1, dtype = torch.float32, device = device)
        for t in found.values():
            total = torch.maximum(total, t.reshape(1).float().to(device))
        if self.composed:
            dist.all_reduce(total, op = dist.ReduceOp.MAX)  # every stage of every dp / tp replica
        elif self.num_stages > 1:
            dist.all_reduce(total, op = dist.ReduceOp.MAX, group = self.group)
        if not found:
            found[device] = total
        for dev, t in found.items():
            t.copy_(total.to(t.device).reshape(t.shape))

    def _adapters(self, kwargs):
        adapters = kwargs.get("selected_adapters") or list(
            getattr(self.model, "peft_config", {}) or {}
        )
        if not adapters:
            raise ValueError("Unsloth: PipelineParallel.save_adapter needs a PEFT model.")
        return list(adapters)

    def adapter_state_dict(self, **kwargs):
        """{adapter: PEFT adapter state dict (save_pretrained keys)} of every stage, merged on pp rank 0
        (collective). None on the other ranks."""
        import torch.distributed as dist
        from peft import get_peft_model_state_dict

        local = {}
        raw = gather = None
        if self.composed or self.dp_mesh is not None or self.tp_mesh is not None:
            # FSDP2 / tensor parallel DTensors: gathered whole (collective over the stage's dp / tp ranks, the
            # same keys in the same order on each), fused tp layouts back in their 1-GPU order.
            raw = self.model.state_dict(keep_vars = True)
            from .tp_native import _full, _param_layouts

            layouts = _param_layouts(self.model)

            def gather(v):
                return _full(v, v, layouts.get(id(v)))

        with torch.no_grad():
            for adapter in self._adapters(kwargs):
                # PEFT selects per stage: only this stage's tuner layers exist here.
                selected = get_peft_model_state_dict(
                    self.model,
                    state_dict = raw,
                    adapter_name = adapter,
                    save_embedding_layers = kwargs.get("save_embedding_layers", "auto"),
                )
                local[adapter] = {
                    k: (gather(v) if gather is not None else v).detach().cpu().clone()
                    for k, v in selected.items()
                    if not k.startswith(self._foreign_prefixes)
                }
        if self.num_stages == 1:
            return local if not self.composed or dist.get_rank() == 0 else None
        first = self.stage_index == 0
        gathered = [None] * self.num_stages if first else None
        dist.gather_object(local, gathered, dst = self._global_ranks[0], group = self.group)
        if not first or (self.composed and dist.get_rank() != 0):
            return None  # one pp group (the one holding global rank 0) writes
        merged = {adapter: {} for adapter in local}
        for part in gathered:
            for adapter, state in part.items():
                merged[adapter].update(state)
        return merged

    def save_adapter(self, save_directory, **kwargs):
        """Write the standard 1-GPU PEFT adapter from pp rank 0 (collective over the pp ranks)."""
        import torch.distributed as dist

        merged = self.adapter_state_dict(**kwargs)
        if merged is not None:
            import peft.peft_model as peft_model

            selection = peft_model.get_peft_model_state_dict

            def gathered(
                model,
                state_dict = None,
                adapter_name = "default",
                **unused,
            ):
                return merged[adapter_name]

            kwargs.pop("state_dict", None)
            kwargs.setdefault("is_main_process", True)
            # PEFT's own writer (files, config, card), fed the merged selection of every stage.
            peft_model.get_peft_model_state_dict = gathered
            try:
                peft_model.PeftModel.save_pretrained(self.model, save_directory, **kwargs)
            finally:
                peft_model.get_peft_model_state_dict = selection
        if self.composed:
            dist.barrier()
        elif self.num_stages > 1:
            dist.barrier(group = self.group)


def _local_stage_modules(pp_stage, parts, owned):
    """One _PPStageModule per local stage of a multi-stage rank. Every one runs the same model; while stage k
    runs, this rank's layers of its other stages are swapped for pass-throughs and, unless k is the first
    stage, the embedding for its stub (so it is not looked up for nothing)."""
    layers = parts.layers
    stand_ins = {i: _PPPassThrough(i, pp_stage.ctx, layers[i]) for i in owned}
    embed_owner = _owner(parts.decoder, parts.embed) if pp_stage.is_first else None
    embed_stub = _PPEmbeddingStub(parts.embed) if pp_stage.is_first else None
    modules = []
    for k in pp_stage.local_stages:
        start, end = pp_stage.bounds[k]
        swaps = [(layers, str(i), stand_ins[i]) for i in owned if not (start <= i < end)]
        if k != 0 and embed_owner is not None:
            swaps.append((embed_owner[0], embed_owner[1], embed_stub))
        stage = (start, end, k == 0, k == pp_stage.total_stages - 1, swaps)
        modules.append(
            _PPStageModule(
                pp_stage.model,
                pp_stage.ctx,
                pp_stage.activation_dtype,
                pp_stage.hidden_size,
                stage = stage,
            )
        )
    return modules


def _sync_trainable(
    model,
    group,
    src,
    owners = None,
):
    """Broadcast every trainable parameter from `src`. With `owners` ({id: pp ranks that loaded it}, from a
    pipeline_sharded_loading model) a parameter listed there is synced only among its owners (from pp rank 0 if
    it is one, else the lowest): a non-owner's copy may be built from a zero placeholder, and the split drops it
    anyway. A single owner keeps its own copy."""
    import torch.distributed as dist

    me = dist.get_rank(group) if owners is not None else None
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            held = owners.get(id(param)) if owners is not None else None
            if held is None:
                dist.broadcast(param.data, src = src, group = group)
                continue
            if len(held) < 2:
                continue
            root = 0 if 0 in held else min(held)
            # Non-owners receive into scratch: their copy may be a zero-stride placeholder.
            buf = (
                param.data
                if me in held
                else torch.empty(param.shape, dtype = param.dtype, device = param.device)
            )
            dist.broadcast(buf, src = dist.get_global_rank(group, root), group = group)


def _default_mesh(model):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        raise RuntimeError(
            "Unsloth: apply_pipeline_parallel needs an initialized process group (torchrun) or a pp_mesh."
        )
    device = next(model.parameters()).device
    return init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names = ("pp",))


def apply_pipeline_parallel(
    model,
    pp_mesh = None,
    n_microbatches = None,
    schedule = "gpipe",
    config: Optional[PPConfig] = None,
) -> PipelineParallel:
    """Keep this rank's stage of `model` (in place: other stages' layers are dropped) and return its
    PipelineParallel. Apply LoRA first; every rank must hold the same model."""
    import torch.distributed as dist

    existing = None
    for level in _model_chain(model) if isinstance(model, nn.Module) else ():
        existing = existing or level.__dict__.get("_unsloth_pp_stage")
    if existing is not None:
        return existing
    _check_torch_pipelining()
    config = config or PPConfig()
    spec = schedule_spec(schedule)
    schedule = spec.name
    stages_per_rank = resolve_stages_per_rank(spec, config.stages_per_rank)
    if pp_mesh is None:
        pp_mesh = _default_mesh(model)
    if pp_mesh.ndim != 1:
        raise ValueError("Unsloth: apply_pipeline_parallel takes a 1-D pp mesh (pass mesh['pp']).")
    pp = pp_mesh.size()
    if dist.is_initialized() and pp != dist.get_world_size():
        # pp x dp / pp x tp: the pp mesh must be one dimension of a mesh covering every rank; the other
        # dimensions are applied after the split (apply_tensor_parallel, shard_model), checked again by step.
        root = _root_mesh(pp_mesh)
        if (
            root is None
            or root is pp_mesh
            or root.ndim < 2
            or root.size() != dist.get_world_size()
            or dist.get_world_size() % pp
        ):
            raise ValueError(
                f"Unsloth: the pipeline parallel mesh has {pp} of {dist.get_world_size()} ranks. Pass mesh['pp'] of "
                "one init_device_mesh(..., (pp, dp), mesh_dim_names = ('pp', 'dp')) (or ('pp', 'tp') / ('pp', "
                "'dp', 'tp')) covering every rank, then shard_model(model, FSDP2Config(mesh = mesh['dp'])) / "
                "apply_tensor_parallel(model, mesh['tp']); or a pp mesh over all ranks."
            )
    n_microbatches = int(n_microbatches or pp)
    if n_microbatches < 1:
        raise ValueError("Unsloth: n_microbatches must be >= 1.")
    if schedule == "1f1b" and n_microbatches < pp:
        raise ValueError(
            f"Unsloth: the 1F1B schedule needs n_microbatches >= stages ({n_microbatches} < {pp})."
        )
    parts = _find_parts(model)
    _refuse_unsupported(model, parts)
    total_stages = pp * stages_per_rank
    rank = pp_mesh.get_local_rank()
    if stages_per_rank > 1:
        root = _root_mesh(pp_mesh)
        names = tuple(getattr(root, "mesh_dim_names", None) or ()) if root is not None else ()
        if "dp" in names:
            raise ValueError(_MULTI_STAGE_DP)
    # The torch schedule's own checks, on placeholder stages: before anything is cut.
    props = validate_schedule(spec, pp_mesh.get_group(), pp, stages_per_rank, n_microbatches, rank)
    _check_schedule(model, spec, props)
    tuple_format = _layer_call_format(parts.decoder, parts.layers_name) == "tuple"
    bounds = derive_pp_split(model, total_stages, config)
    _refuse_shared_across_stages(parts, bounds)
    from .pp_load import check_preloaded_split, loaded_by, release_preload_guards

    # pipeline_sharded_loading read only this rank's stages: the split must be the one it loaded.
    preloaded = check_preloaded_split(
        model,
        pp_mesh,
        schedule,
        stages_per_rank,
        bounds,
        stage_indices(spec, rank, pp, stages_per_rank),
        parts,
    )
    dtype = config.activation_dtype or parts.embed.weight.dtype
    if config.sync_trainable and pp > 1:
        owners = None
        if preloaded is not None:
            owners = loaded_by(parts, spec, pp, stages_per_rank, bounds)
        _sync_trainable(
            model, pp_mesh.get_group(), dist.get_global_rank(pp_mesh.get_group(), 0), owners
        )

    donated_off = props["split_backward"] and allow_split_backward()
    pp_stage = PipelineParallel(
        model,
        pp_mesh,
        parts,
        bounds,
        n_microbatches,
        schedule,
        config,
        tuple_format,
        dtype,
        stages_per_rank = stages_per_rank,
    )
    local = pp_stage.local_stages
    owned = sorted({i for k in local for i in range(*bounds[k])})
    start, end = bounds[local[0]]
    is_first, is_last = pp_stage.is_first, pp_stage.is_last
    layers = parts.layers
    for i in range(len(layers)):
        if i not in owned:
            layers[i] = _PPPassThrough(i, pp_stage.ctx, layers[i])
    embed, lm_head = parts.embed, parts.lm_head
    if not is_first:
        parent, attr = _owner(parts.decoder, parts.embed)
        embed = _PPEmbeddingStub(parts.embed)
        setattr(parent, attr, embed)
    if not is_last:
        parent, attr = _owner(parts.inner, parts.lm_head)
        lm_head = _PPHeadStub(parts.lm_head)
        setattr(parent, attr, lm_head)
    # A norm registered before the layer list (Bloom's word_embeddings_layernorm) feeds the first stage; one
    # after it (the final norm) the last. Other stages keep it frozen, so it trains on its owner only.
    order = list(parts.decoder._modules)
    foreign = []
    for name, norm in parts.norms:
        input_side = order.index(name) < order.index(parts.layers_name)
        if not (is_first if input_side else is_last):
            for p in norm.parameters():
                p.requires_grad_(False)
            foreign.append(norm)
    # Adapter keys of norms another stage trains (modules_to_save): that stage's copy is the one saved.
    pp_stage._foreign_prefixes = tuple(
        name + "." for name, module in model.named_modules() if any(module is f for f in foreign)
    )
    # The stage object must not pin a dropped embedding / lm_head (vocabulary-sized) alive.
    parts = pp_stage.parts = dataclasses.replace(parts, embed = embed, lm_head = lm_head)
    if stages_per_rank > 1:
        pp_stage.stage_modules = _local_stage_modules(pp_stage, parts, owned)
        pp_stage.module = pp_stage.stage_modules[0]
    if preloaded is not None:
        release_preload_guards(preloaded)  # the placeholder layers / embedding / head are gone now
    info = {
        "size": pp,
        "rank": rank,
        "layers": (start, end),
        "bounds": bounds,
        "source": parts.source,
        "schedule": schedule,
        "stages_per_rank": stages_per_rank,
        "local_stages": tuple(local),
    }
    inner = parts.inner
    for level in _model_chain(model):
        level.__dict__[_PP_FLAG] = info
        level.__dict__["_unsloth_pp_stage"] = pp_stage
        if callable(getattr(level, "generate", None)):
            level.__dict__["generate"] = _refuse_generate
    del inner
    import gc

    gc.collect()  # the dropped layers sit in reference cycles: free their weights now, not at a later collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (dist.get_rank() if dist.is_initialized() else rank) == 0:
        print(
            f"Unsloth: pipeline parallel over {pp} ranks x {stages_per_rank} stages ({schedule}, "
            f"{n_microbatches} micro-batches, {parts.source} parts): decoder layers per stage "
            f"{[e - s for s, e in bounds]}"
            + (", tied embeddings kept on the first and last stage" if parts.tied else "")
            + (
                "; compiled backward graphs built without donated buffers (zero bubble runs autograd.grad "
                "with retain_graph)"
                if donated_off
                else ""
            )
            + "."
        )
    return pp_stage
