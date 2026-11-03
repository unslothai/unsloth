# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Split after LoRA, then apply orthogonal dp/tp; adapter saves are collective."""

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
_SCHEDULES = ("gpipe", "1f1b")
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# Shared or cross-layer state cannot cross a stage boundary.
_CROSS_LAYER_CONFIG_KEYS = (
    "num_kv_shared_layers",
    "altup_num_inputs",
    "hidden_size_per_layer_input",
    "kv_lora_rank",
)
_ATTENTION_LAYER_TYPES = {"full_attention", "sliding_attention", "chunked_attention", "attention"}
_STEP_KWARGS = ("attention_mask", "position_ids")
_SIMPLE = (bool, int, float, str, type(None))


@dataclasses.dataclass
class PPConfig:
    # None balances stages, including the final lm_head cost.
    layers_per_stage: Optional[list] = None

    lm_head_cost: float = 1.0
    # None uses the input embedding dtype.
    activation_dtype: Optional[torch.dtype] = None

    sync_trainable: bool = True

    max_cached_shapes: int = 4


class _StageOutput(BaseException):
    """BaseException prevents model-level except Exception handlers from swallowing stage exits."""

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
    for name, module in root.named_modules():
        for attr, child in module._modules.items():
            if child is target:
                return module, attr
    return None, None


def _model_chain(model):
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
    if not isinstance(model, nn.Module):
        return False
    try:
        chain = _model_chain(model)
    except Exception:
        chain = [model]
    return any(m.__dict__.get(_PP_FLAG) is not None for m in chain)


def pipeline_stage_of(model):
    if not isinstance(model, nn.Module):
        return None
    for level in _model_chain(model):
        stage = level.__dict__.get("_unsloth_pp_stage")
        if stage is not None:
            return stage
    return None


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
    """Validate orthogonal 1-D meshes that jointly cover the world before changing the model."""
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
    """Validate pp-first composition on the original model with orthogonal dp/tp meshes."""
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


@dataclasses.dataclass
class _Parts:
    inner: nn.Module
    decoder: nn.Module
    layers: nn.ModuleList
    layers_name: str
    embed: nn.Module
    lm_head: nn.Module
    norms: list
    source: str
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
    # Decoder-only encoder-decoder configs report encoder depth here.
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
        base = getattr(child, "original_module", child)
        norm_like = "norm" in name.lower() or "Norm" in type(base).__name__
        too_big = bool(hidden) and sum(p.numel() for p in base.parameters()) > 4 * hidden
        # Later stages replace positional-table output; token-indexed side inputs remain unsupported.

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
    """Refuse cross-stage trainable sharing, which would create independently trained copies."""
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
    """Micro-batch means overcount loss; every loss must use the global valid-token count."""
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


def _forward_function(module):
    forward = module.__dict__.get("forward") or type(module).forward
    forward = getattr(forward, "__func__", forward)
    seen = 0
    while hasattr(forward, "__wrapped__") and seen < 8:
        forward = forward.__wrapped__
        seen += 1
    return forward


def _layer_call_format(decoder, layers_name):
    """Match the decoder's layer output convention: tuple on legacy forwards, tensor on newer ones."""
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
        # Pass-through stages must not be checkpointed.
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
    # best[k][j] is the minimum peak cost for stages 0..k covering layers [0, j).
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
except Exception:  # Legacy transformers checkpoint by isinstance.
    _CheckpointLayer = nn.Module


class _PPPassThrough(_CheckpointLayer):
    """Keep the pass-through in autograd without checkpointing another stage's layer."""

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
    return torch.zeros((), dtype = weight.dtype, device = weight.device).expand(weight.shape)


class _PPEmbeddingStub(nn.Module):
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
    def __init__(self, model, ctx, dtype, hidden_size):
        super().__init__()
        self.model = model
        self.__dict__["_pp_ctx"] = ctx
        self.__dict__["_pp_dtype"] = dtype
        self.__dict__["_pp_hidden"] = hidden_size
        self.__dict__["_pp_warned"] = False

    def forward(self, x, **kwargs):
        ctx = self._pp_ctx
        if ctx.is_first:
            input_ids = x
        else:
            input_ids = torch.zeros(x.shape[:2], dtype = torch.long, device = x.device)
            ctx.stage_input = x
        ctx.active = True
        try:
            kwargs.update(output_attentions = False, output_hidden_states = False)
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


def _gradient_checkpointing_on(model):
    return any(
        getattr(m, "gradient_checkpointing", False) is True for m in model.modules()
    ) or bool(getattr(_unwrap(model), "is_gradient_checkpointing", False))


def _zoo_slot_lease_present():
    """Interleaved 1F1B needs per-checkpoint offload-slot leases (zoo #2071)."""
    module = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    if module is None:
        return None
    return hasattr(module, "_CpuSlotLease")


def _check_torch_pipelining():
    """Probe pipelining signatures before splitting: older APIs fail or leak kwargs into forward."""
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


def _check_schedule(model, schedule):
    if schedule not in _SCHEDULES:
        raise ValueError(
            f"Unsloth: pipeline schedule must be one of {_SCHEDULES}, got {schedule!r}."
        )
    if (
        schedule == "1f1b"
        and _gradient_checkpointing_on(model)
        and _zoo_slot_lease_present() is False
    ):
        raise ValueError(
            "Unsloth: the 1F1B pipeline schedule interleaves forwards and backwards, which this "
            "unsloth_zoo's smart gradient checkpointing does not support (offloaded activations of a pending "
            "micro-batch would be overwritten). Use schedule = 'gpipe', or update unsloth_zoo (host slot "
            "leases, unsloth-zoo PR #2071)."
        )


def _refuse_generate(*args, **kwargs):
    raise RuntimeError(
        "Unsloth: generation is not supported on a pipeline parallel model in v1; save the adapter "
        "(PipelineParallel.save_adapter) and generate on one GPU."
    )


class PipelineParallel:
    def __init__(
        self, model, mesh, parts, bounds, n_microbatches, schedule, config, tuple_format, dtype
    ):
        import torch.distributed as dist

        self.model = model
        self.mesh = mesh
        self.group = mesh.get_group()
        self.num_stages = mesh.size()
        self.stage_index = mesh.get_local_rank()
        self.bounds = bounds
        self.start, self.end = bounds[self.stage_index]
        self.n_microbatches = n_microbatches
        self.schedule = schedule
        self.config = config
        self.parts = parts
        self._foreign_prefixes = ()
        self.activation_dtype = dtype
        self.ctx = _StageContext(self.start, self.end, self.is_first, self.is_last, tuple_format)
        hidden = getattr(_text_config(model), "hidden_size", None) or parts.embed.weight.shape[-1]
        self.hidden_size = int(hidden)
        self.module = _PPStageModule(model, self.ctx, dtype, self.hidden_size)
        self._global_ranks = [dist.get_global_rank(self.group, i) for i in range(self.num_stages)]
        self._shapes = collections.OrderedDict()
        self._scaler = None
        self._step_scaler = None
        self._mb_losses = []
        self._loss_multiplier = 1
        self.dp_mesh = None

    @property
    def tp_mesh(self):
        for level in _model_chain(self.model):
            mesh = level.__dict__.get("_unsloth_tp_mesh")
            if mesh is not None:
                return mesh
        return None

    @property
    def composed(self):
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
        """The outer FSDP root defers all micro-batch reductions until pipeline finalization."""
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        policy = MixedPrecisionPolicy(
            param_dtype = None,
            reduce_dtype = getattr(mp_policy, "reduce_dtype", None),
            cast_forward_inputs = False,
        )
        fully_shard(self.module, mesh = dp_mesh, mp_policy = policy, reshard_after_forward = False)
        from .fsdp2_save import _install_replicated_grad_reset

        _install_replicated_grad_reset(self.module)
        self.dp_mesh = dp_mesh
        self._shapes.clear()

    @property
    def is_first(self):
        return self.stage_index == 0

    @property
    def is_last(self):
        return self.stage_index == self.num_stages - 1

    @property
    def device(self):
        for p in self.module.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def __repr__(self):
        return (
            f"PipelineParallel(stage {self.stage_index} / {self.num_stages}, layers [{self.start}, "
            f"{self.end}), {self.schedule}, n_microbatches = {self.n_microbatches})"
        )

    def _loss_fn(self, output, target):
        self._mb_losses.append(output.detach())
        if self._loss_multiplier != 1:
            # FSDP averages dp gradients, so scale each replica's global-token loss by dp size.

            output = output * self._loss_multiplier
        return output if self._scaler is None else self._scaler.scale(output)

    def _schedule_for(self, mb, seq):
        from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

        key = (mb, seq)
        if key in self._shapes:
            self._shapes.move_to_end(key)
            return self._shapes[key]
        device = self.device
        if self.is_first:
            example_in = torch.empty((mb, seq), dtype = torch.long, device = "meta")
        else:
            # Received hidden states must require gradients in the static stage metadata.
            example_in = torch.empty(
                (mb, seq, self.hidden_size),
                dtype = self.activation_dtype,
                device = "meta",
                requires_grad = True,
            )
        if self.is_last:
            example_out = torch.empty((), dtype = torch.float32, device = "meta", requires_grad = True)
        else:
            example_out = torch.empty(
                (mb, seq, self.hidden_size),
                dtype = self.activation_dtype,
                device = "meta",
                requires_grad = True,
            )
        stage = PipelineStage(
            self.module,
            self.stage_index,
            self.num_stages,
            device,
            input_args = example_in,
            output_args = example_out,
            group = self.group,
        )
        cls = ScheduleGPipe if self.schedule == "gpipe" else Schedule1F1B
        schedule = cls(stage, self.n_microbatches, loss_fn = self._loss_fn, scale_grads = False)
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
        """Accumulate one batch; pp/tp peers share inputs, dp ranks pass slices, and loss is global."""
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
            count = (labels[:, 1:] != -100).sum()
            if dp_group is not None:
                count = count.to(device = device, dtype = torch.int64)
                dist.all_reduce(count, group = dp_group)
            num_items_in_batch = int(count)
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = int(num_items_in_batch)
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
                    value = value.expand(bsz, *value.shape[1:])
                value = value.to(device)
            step_kwargs[key] = value
        step_kwargs["labels"] = labels.to(device)
        step_kwargs["num_items_in_batch"] = num_items_in_batch
        if scaler is not None and scaler.is_enabled():
            scaler.scale(torch.zeros((), device = device))
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
            dist.broadcast(total, src = self._global_ranks[-1], group = self.group)
        if dp_group is not None:
            dist.all_reduce(total, group = dp_group)
        return total[0]

    def _average_replicated_grads(self):
        params = self.model.__dict__.get("_unsloth_replicated_grad_params")
        if not params:
            return
        import torch.distributed as dist

        from .fsdp2_save import _sync_replicated_grads, model_mesh

        state = self.model.__dict__.setdefault("_unsloth_replicated_grad_state", {"queued": False})
        _sync_replicated_grads(
            params,
            dist,
            model_mesh(self.model) or self.dp_mesh,
            state.setdefault("buffers", {}),
            state.pop("pending", None),
        )

    def _grad_copies(self, grad):
        """Count each stage gradient once, accounting for replicated and TP-sharded copies."""
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
        """Clip the global norm across pp/dp/tp without double-counting replicas."""
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
            dist.all_reduce(sq)
        elif self.num_stages > 1:
            dist.all_reduce(sq, group = self.group)
        total = sq.sqrt()
        coef = torch.clamp(float(max_norm) / (total + eps), max = 1.0)
        for g in grads:
            g.mul_(coef.to(g.dtype))
        return total.float()

    def unscale_(self, scaler, optimizer):
        """Reduce found_inf over every stage so all optimizers skip the same step."""
        scaler.unscale_(optimizer)
        if not scaler.is_enabled():
            return
        self._agree_found_inf(scaler, optimizer)

    def _bind_scaler(self, scaler):
        """Bind scaler agreement only until update so the scaler can be pickled afterward."""
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
                    pp._step_scaler = None

        scaler.unscale_ = unscale_
        scaler.update = update_
        scaler._unsloth_pp_agree = pipeline

    def _check_unscaled(self):
        """Refuse clipping scaled gradients, which would use the wrong norm."""
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
            return
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
            dist.all_reduce(total, op = dist.ReduceOp.MAX)
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
        import torch.distributed as dist
        from peft import get_peft_model_state_dict

        local = {}
        raw = gather = None
        if self.composed or self.dp_mesh is not None or self.tp_mesh is not None:
            # Every dp/tp rank must gather the same keys in the same order.

            raw = self.model.state_dict(keep_vars = True)
            from .tp_native import _full, _param_layouts

            layouts = _param_layouts(self.model)

            def gather(v):
                return _full(v, v, layouts.get(id(v)))

        with torch.no_grad():
            for adapter in self._adapters(kwargs):
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
        gathered = [None] * self.num_stages if self.is_first else None
        dist.gather_object(local, gathered, dst = self._global_ranks[0], group = self.group)
        if not self.is_first or (self.composed and dist.get_rank() != 0):
            return None
        merged = {adapter: {} for adapter in local}
        for part in gathered:
            for adapter, state in part.items():
                merged[adapter].update(state)
        return merged

    def save_adapter(self, save_directory, **kwargs):
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

            peft_model.get_peft_model_state_dict = gathered
            try:
                peft_model.PeftModel.save_pretrained(self.model, save_directory, **kwargs)
            finally:
                peft_model.get_peft_model_state_dict = selection
        if self.composed:
            dist.barrier()
        elif self.num_stages > 1:
            dist.barrier(group = self.group)


def _sync_trainable(model, group, src):
    import torch.distributed as dist
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                dist.broadcast(param.data, src = src, group = group)


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
    """Split in place after LoRA; every rank must begin with the same model."""
    import torch.distributed as dist

    existing = None
    for level in _model_chain(model) if isinstance(model, nn.Module) else ():
        existing = existing or level.__dict__.get("_unsloth_pp_stage")
    if existing is not None:
        return existing
    _check_torch_pipelining()
    config = config or PPConfig()
    schedule = str(schedule).lower().replace("-", "").replace("_", "")
    if pp_mesh is None:
        pp_mesh = _default_mesh(model)
    if pp_mesh.ndim != 1:
        raise ValueError("Unsloth: apply_pipeline_parallel takes a 1-D pp mesh (pass mesh['pp']).")
    pp = pp_mesh.size()
    if dist.is_initialized() and pp != dist.get_world_size():
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
    _check_schedule(model, schedule)
    tuple_format = _layer_call_format(parts.decoder, parts.layers_name) == "tuple"
    bounds = derive_pp_split(model, pp, config)
    _refuse_shared_across_stages(parts, bounds)
    dtype = config.activation_dtype or parts.embed.weight.dtype
    rank = pp_mesh.get_local_rank()
    if config.sync_trainable and pp > 1:
        _sync_trainable(model, pp_mesh.get_group(), dist.get_global_rank(pp_mesh.get_group(), 0))

    pp_stage = PipelineParallel(
        model, pp_mesh, parts, bounds, n_microbatches, schedule, config, tuple_format, dtype
    )
    start, end = bounds[rank]
    is_first, is_last = rank == 0, rank == pp - 1
    layers = parts.layers
    for i in range(len(layers)):
        if not (start <= i < end):
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
    # Norms before/after the layer list belong only to the first/last stage.

    order = list(parts.decoder._modules)
    foreign = []
    for name, norm in parts.norms:
        input_side = order.index(name) < order.index(parts.layers_name)
        if not (is_first if input_side else is_last):
            for p in norm.parameters():
                p.requires_grad_(False)
            foreign.append(norm)

    pp_stage._foreign_prefixes = tuple(
        name + "." for name, module in model.named_modules() if any(module is f for f in foreign)
    )
    # Do not retain vocabulary-sized weights after dropping their stage.
    parts = pp_stage.parts = dataclasses.replace(parts, embed = embed, lm_head = lm_head)
    info = {
        "size": pp,
        "rank": rank,
        "layers": (start, end),
        "bounds": bounds,
        "source": parts.source,
        "schedule": schedule,
    }
    inner = parts.inner
    for level in _model_chain(model):
        level.__dict__[_PP_FLAG] = info
        level.__dict__["_unsloth_pp_stage"] = pp_stage
        if callable(getattr(level, "generate", None)):
            level.__dict__["generate"] = _refuse_generate
    del inner
    import gc

    gc.collect()  # Collect dropped-layer reference cycles immediately to release their weights.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (dist.get_rank() if dist.is_initialized() else rank) == 0:
        print(
            f"Unsloth: pipeline parallel over {pp} stages ({schedule}, {n_microbatches} micro-batches, "
            f"{parts.source} parts): decoder layers per stage {[e - s for s, e in bounds]}"
            + (", tied embeddings kept on the first and last stage" if parts.tied else "")
            + "."
        )
    return pp_stage
