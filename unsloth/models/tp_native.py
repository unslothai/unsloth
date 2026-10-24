# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallelism (column / row sharding of attention and MLP) for any dense decoder, on DTensor.

    model = FastLanguageModel.get_peft_model(model, ...)  # LoRA first: same init as 1 GPU
    apply_tensor_parallel(model, init_device_mesh("cuda", (tp,), mesh_dim_names = ("tp",))["tp"])

v1: LoRA / QLoRA training forward on dense decoders, tp ranks only (no Trainer / FSDP2 / generate).
Custom loop (torchrun --nproc_per_node tp):

    params = [p for p in model.parameters() if p.requires_grad]  # DTensors (keep plain ones in their own group)
    optimizer = torch.optim.AdamW(params, lr = 2e-4)
    loss = model(input_ids = ids, labels = ids).loss  # same batch on every tp rank
    loss.backward()
    tp_clip_grad_norm_(params, 1.0)
    optimizer.step(); optimizer.zero_grad()
    save_tp_adapter(model, "out")  # collective: the standard 1-GPU PEFT adapter, written by rank 0
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import re
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TPConfig",
    "TPPlan",
    "apply_tensor_parallel",
    "derive_tp_plan",
    "is_tensor_parallel",
    "shard_slice",
    "tp_clip_grad_norm_",
    "tp_full_state_dict",
    "save_tp_adapter",
]

_TP_FLAG = "_unsloth_tp"
COLWISE = "colwise"
ROWWISE = "rowwise"
HEAD_NORM = "replicated_with_grad_allreduce"

# transformers TP styles (4.5x and 5.x names) -> what this module does with them.
_STYLES = {
    "colwise": COLWISE,
    "local_colwise": COLWISE,
    "colwise_rep": COLWISE,
    "colwise_gather_output": COLWISE,
    "packed_colwise": COLWISE,
    "local_packed_colwise": COLWISE,
    "rowwise": ROWWISE,
    "local_rowwise": ROWWISE,
    "rowwise_rep": ROWWISE,
    "rowwise_split_input": ROWWISE,
    "replicated_with_grad_allreduce": HEAD_NORM,
}
# Outside the decoder regions (embeddings, lm_head, final norm): replicated in v1.
_REPLICATED_STYLES = {
    "embedding_rowwise",
    "gather",
    "sequence_parallel",
    "all_reduce",
    "replicate",
    "isolated",
}
_MOE_STYLES = {
    "grouped_gemm",
    "ep_router",
    "megamoe_router",
    "moe_tp_experts",
    "megamoe_experts",
    "moe_identity_expert",
    "packed_rowwise",
    "local_packed_rowwise",
}
_MLA_STYLES = {"mla_kv_a_proj"}

_ATTN_NAMES = ("self_attn", "attention", "attn", "self_attention")
_MLP_NAMES = ("mlp", "feed_forward", "ffn")
_Q, _K, _V = ("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")
_FUSED_QKV = ("qkv_proj",)  # [q, k, v] along the output dim (Phi3)
_ATTN_ROW = ("o_proj", "out_proj", "dense")
_MLP_COL = ("gate_proj", "up_proj", "w1", "w3", "fc1", "c_fc")
_MLP_PACKED = ("gate_up_proj",)  # [gate, up] along the output dim
_MLP_ROW = ("down_proj", "w2", "fc2")
_HEAD_NORMS = ("q_norm", "k_norm", "query_norm", "key_norm")
_MLA_NAMES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm")
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# An attention forward reading these sizes off the config would reshape to the global head count.
_GLOBAL_HEAD_READS = re.compile(r"config\.(num_attention_heads|num_key_value_heads)\b")
_GLOBAL_WIDTH_READS = re.compile(r"(config|self)\.hidden_size\b")


@dataclasses.dataclass
class TPConfig:
    # {"layers.*.self_attn.q_proj": "colwise", ...}; None = the model's TP plan, else structural discovery.
    plan: Optional[dict] = None
    # Broadcast trainable parameters from tp rank 0 first (every rank must start from the same adapter).
    sync_trainable: bool = True


@dataclasses.dataclass
class _Proj:
    name: str
    module: nn.Module
    style: str  # COLWISE / ROWWISE
    parts: tuple  # output (colwise) part sizes; one part unless fused / packed
    from_plan: bool = False


@dataclasses.dataclass
class _Region:
    path: str
    module: nn.Module
    kind: str  # "attention" / "mlp"
    projs: list
    norms: list  # per-head norms inside the region: replicated, gradients summed over tp
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0


@dataclasses.dataclass
class TPPlan:
    tp_size: int
    regions: list
    source: str  # "user" / "transformers" / "structural"
    replicated_layers: int = (
        0  # decoder layers with no region (e.g. linear attention): run replicated
    )

    def styles(self):
        """{projection path: style} (the plan in transformers form, local paths)."""
        return {f"{r.path}.{p.name}": p.style for r in self.regions for p in r.projs}


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


def _hf_plan(model):
    inner = _unwrap(model)
    plan = {}
    text = _text_config(model)
    for source in (getattr(text, "base_model_tp_plan", None), getattr(inner, "_tp_plan", None)):
        if isinstance(source, dict):
            plan.update(source)
    return plan


def _pattern(key):
    return re.compile(r"(^|\.)" + re.escape(key).replace(r"\*", r"\d+") + r"$")


def _style_for(path, compiled):
    for key, regex, style in compiled:
        if regex.search(path):
            return key, style
    return None, None


def _weight(module):
    base = module._modules.get("base_layer", module)
    return base._parameters.get("weight", getattr(base, "weight", None)), base


def _has_params(module):
    return next(module.parameters(), None) is not None


def _decoder_layers(model):
    """The ModuleList of decoder layers (len == text config num_hidden_layers)."""
    inner = _unwrap(model)
    decoder = inner
    get_decoder = getattr(inner, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder() or inner
        except Exception:
            decoder = inner
    text = _text_config(model)
    n = getattr(text, "num_hidden_layers", None)
    prefix = ""
    for name, module in inner.named_modules():
        if module is decoder:
            prefix = name
            break
    for name, module in decoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) and (n is None or len(module) == n):
            path = ".".join(x for x in (prefix, name) if x)
            return path, module
    raise ValueError(
        "Unsloth: tensor parallelism found no decoder layer list "
        f"({type(inner).__name__}: no ModuleList of num_hidden_layers = {n} layers)."
    )


_ATTENTION_DROPOUT_KEYS = ("attention_dropout", "attn_pdrop")


def _refuse_dropout(model, layers):
    """v1 has no TP-aware RNG (masks must differ per shard, match when replicated): refuse dropout."""
    text = _text_config(model)
    for key in _ATTENTION_DROPOUT_KEYS:
        value = getattr(text, key, None)
        if isinstance(value, (int, float)) and value > 0:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support attention dropout ({key} = {value}); set it to 0."
            )
    for name, module in layers.named_modules():
        if "lora_dropout" in name or not isinstance(module, nn.modules.dropout._DropoutNd):
            continue
        if module.p > 0:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support dropout inside decoder layers ({name}: p = "
                f"{module.p}); set it to 0."
            )


def _refuse_moe_mla(model, layers):
    text = _text_config(model)
    for key in _MOE_CONFIG_KEYS:
        value = getattr(text, key, None)
        if isinstance(value, int) and value > 0:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support mixture-of-experts models ({key} = {value}): "
                "expert sharding is not implemented. Use native FSDP2 (UNSLOTH_FSDP2=1) instead."
            )
    if getattr(text, "kv_lora_rank", None):
        raise ValueError(
            "Unsloth: tensor parallelism v1 does not support multi-head latent attention (MLA, kv_lora_rank "
            f"= {text.kv_lora_rank}). Use native FSDP2 (UNSLOTH_FSDP2=1) instead."
        )
    for name, module in layers.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        cls = type(module).__name__
        if leaf == "experts" or "Experts" in cls or "SparseMoe" in cls or "MoE" in cls:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support mixture-of-experts layers ({name}: {cls})."
            )
        if leaf in _MLA_NAMES:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support multi-head latent attention ({name})."
            )


def _classify(region_path, name, child, kind, compiled):
    """(style, parts kind) for one child of an attention / MLP region, from the plan or its name."""
    key, style = _style_for(f"{region_path}.{name}", compiled) if compiled else (None, None)
    if style is not None:
        if style in _MOE_STYLES:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support the MoE style '{style}' ({key})."
            )
        if style in _MLA_STYLES:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support the MLA style '{style}' ({key})."
            )
        if style in _REPLICATED_STYLES:
            raise ValueError(
                f"Unsloth: '{key}': '{style}' inside a decoder {kind} is not supported by tensor parallelism v1."
            )
        if style not in _STYLES:
            raise ValueError(f"Unsloth: unknown tensor parallel style '{style}' for '{key}'.")
        raw, style = style, _STYLES[style]
        # A fused qkv is [q, kv, kv] (GQA: unequal parts) whatever the plan calls it; "packed" means equal halves.
        if style == COLWISE and kind == "attention" and name in _FUSED_QKV:
            return COLWISE, "qkv:plan"
        if style == COLWISE and (raw.endswith("packed_colwise") or name in _MLP_PACKED):
            if kind == "attention":
                raise ValueError(
                    f"Unsloth: '{key}': '{raw}' on the attention projection '{name}' has no known layout "
                    "(tensor parallelism v1 knows fused qkv_proj); pass plan = {...} with 'colwise'."
                )
            return COLWISE, "packed:plan"
        return style, "plan"
    if kind == "attention":
        if name in _Q + _K + _V:
            return COLWISE, None
        if name in _FUSED_QKV:
            return COLWISE, "qkv"
        if name in _ATTN_ROW:
            return ROWWISE, None
        if name in _HEAD_NORMS:
            return HEAD_NORM, None
    else:
        if name in _MLP_COL:
            return COLWISE, None
        if name in _MLP_PACKED:
            return COLWISE, "packed"
        if name in _MLP_ROW:
            return ROWWISE, None
    return None, None


def _attention_sizes(module, projs, text):
    """(heads, kv_heads, head_dim) of one attention module, from its own projections."""
    heads_cfg = getattr(text, "num_attention_heads", None)
    kv_cfg = getattr(text, "num_key_value_heads", None) or heads_cfg
    head_dim = getattr(module, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = getattr(text, "head_dim", None) or (text.hidden_size // heads_cfg)
    by_name = {p.name: p for p in projs}
    q = next((by_name[n] for n in _Q if n in by_name), None)
    k = next((by_name[n] for n in _K if n in by_name), None)
    if q is not None and k is not None:
        heads = _logical_shape(*_weight(q.module))[0] // head_dim
        kv_heads = _logical_shape(*_weight(k.module))[0] // head_dim
    else:
        heads, kv_heads = heads_cfg, kv_cfg
    return heads, kv_heads, head_dim


def _check_divisible(what, value, tp, path):
    if value % tp != 0:
        raise ValueError(
            f"Unsloth: tensor parallel size {tp} does not divide {what} = {value} ({path}). "
            f"Choose a tp size that divides it."
        )


def derive_tp_plan(
    model,
    tp_size,
    plan = None,
) -> TPPlan:
    """The regions to shard, validated for tp_size; ValueError for anything v1 cannot shard."""
    tp_size = int(tp_size)
    if tp_size < 1:
        raise ValueError(f"Unsloth: tensor parallel size must be >= 1, got {tp_size}.")
    user = bool(plan)
    plan = dict(plan) if plan else _hf_plan(model)
    compiled = [(k, _pattern(k), v) for k, v in plan.items()]
    layers_path, layers = _decoder_layers(model)
    # Config / module MoE and MLA first: older transformers plans name expert styles v1 does not know.
    _refuse_moe_mla(model, layers)
    _refuse_dropout(model, layers)
    for key, _, style in compiled:
        if style in _MOE_STYLES:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support mixture-of-experts layers ('{key}': '{style}')."
            )
        if style in _MLA_STYLES:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support multi-head latent attention ('{key}')."
            )
        if style not in _STYLES and style not in _REPLICATED_STYLES:
            raise ValueError(f"Unsloth: unknown tensor parallel style '{style}' for '{key}'.")
    text = _text_config(model)
    regions, replicated = [], 0
    for index, layer in enumerate(layers):
        found = 0
        for child_name, child in layer.named_children():
            kind = (
                "attention"
                if child_name in _ATTN_NAMES
                else ("mlp" if child_name in _MLP_NAMES else None)
            )
            if kind is None or not _has_params(child):
                continue
            path = f"{layers_path}.{index}.{child_name}"
            regions.append(_region(path, child, kind, compiled, text, tp_size))
            found += 1
        replicated += found == 0
    if not regions:
        raise ValueError(
            "Unsloth: tensor parallelism found no attention / MLP modules to shard in the decoder layers."
        )
    if user:
        source = "user"
    else:
        source = (
            "transformers" if any(p.from_plan for r in regions for p in r.projs) else "structural"
        )
    return TPPlan(tp_size = tp_size, regions = regions, source = source, replicated_layers = replicated)


def _region(path, module, kind, compiled, text, tp):
    projs, norms = [], []
    for name, param in module.named_parameters(recurse = False):
        raise ValueError(
            f"Unsloth: tensor parallelism cannot shard '{path}': it owns the parameter '{name}' directly "
            "(not inside a projection), which v1 has no rule for."
        )
    for name, child in module.named_children():
        if not _has_params(child):
            continue
        style, packing = _classify(path, name, child, kind, compiled)
        if style is None:
            raise ValueError(
                f"Unsloth: tensor parallelism has no rule for '{path}.{name}' ({type(child).__name__}); "
                "every module with parameters inside a sharded attention / MLP must be in the plan. Pass "
                "plan = {...} to apply_tensor_parallel."
            )
        if style == HEAD_NORM:
            norms.append((name, child))
            continue
        weight, _ = _weight(child)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            raise ValueError(
                f"Unsloth: tensor parallelism expects a Linear at '{path}.{name}' "
                f"(got {type(child).__name__})."
            )
        from_plan = packing is not None and packing.endswith("plan")
        packing = packing.split(":")[0] if packing and packing != "plan" else None
        projs.append(_Proj(name, child, style, packing, from_plan))
    if not any(p.style == COLWISE for p in projs) or not any(p.style == ROWWISE for p in projs):
        raise ValueError(
            f"Unsloth: '{path}' needs column and row sharded projections for tensor parallelism; found "
            f"{[(p.name, p.style) for p in projs]}."
        )
    region = _Region(path, module, kind, projs, norms)
    if kind == "attention":
        heads, kv_heads, head_dim = _attention_sizes(module, projs, text)
        region.heads, region.kv_heads, region.head_dim = heads, kv_heads, head_dim
        _check_divisible("num_attention_heads", heads, tp, path)
        if kv_heads < tp:
            raise ValueError(
                f"Unsloth: tensor parallel size {tp} is larger than num_key_value_heads = {kv_heads} "
                f"({path}); replicating KV heads is not supported in v1. Use tp <= {kv_heads}."
            )
        _check_divisible("num_key_value_heads", kv_heads, tp, path)
        for name, norm in norms:
            numel = sum(p.numel() for p in norm.parameters())
            if numel != head_dim:
                raise ValueError(
                    f"Unsloth: '{path}.{name}' normalizes {numel} features, not one head ({head_dim}): a norm "
                    "across all heads needs the full projection, which tensor parallelism v1 does not gather."
                )
        _check_attention_source(module, path)
    elif norms:
        raise ValueError(f"Unsloth: unexpected per-head norm in MLP '{path}'.")
    for proj in projs:
        weight, base = _weight(proj.module)
        out_features, in_features = _logical_shape(weight, base)
        if proj.style == ROWWISE:
            _check_divisible(f"{proj.name} in_features", in_features, tp, path)
            proj.parts = (out_features,)
            continue
        if proj.parts == "qkv":
            q, kv = region.heads * region.head_dim, region.kv_heads * region.head_dim
            if q + 2 * kv != out_features:
                raise ValueError(
                    f"Unsloth: fused '{path}.{proj.name}' has {out_features} outputs, not q + 2 kv = "
                    f"{q + 2 * kv}; its layout is unknown to tensor parallelism v1."
                )
            proj.parts = (q, kv, kv)
        elif proj.parts == "packed":
            if out_features % 2:
                raise ValueError(f"Unsloth: packed '{path}.{proj.name}' has an odd output size.")
            proj.parts = (out_features // 2, out_features // 2)
        else:
            proj.parts = (out_features,)
        for size in proj.parts:
            what = "intermediate_size" if kind == "mlp" else f"{proj.name} out_features"
            _check_divisible(what, size, tp, path)
    return region


def _logical_shape(weight, base):
    state = getattr(weight, "quant_state", None)
    if state is not None and getattr(state, "shape", None) is not None:
        return tuple(state.shape)
    return tuple(weight.shape)


def _forward_source(module):
    forward = module.forward
    for _ in range(4):
        forward = getattr(forward, "__func__", forward)
        forward = inspect.unwrap(forward)
        # torch.compile'd forwards (unsloth_zoo temporary patches) keep the original here.
        inner = getattr(forward, "_torchdynamo_orig_callable", None)
        if inner is None:
            break
        forward = inner
    return inspect.getsource(forward)


def _check_attention_source(module, path):
    try:
        source = _forward_source(module)
    except (OSError, TypeError):
        raise ValueError(
            f"Unsloth: tensor parallelism cannot read the forward of '{path}' ({type(module).__name__}) to "
            "check it uses this rank's head count."
        )
    if _GLOBAL_WIDTH_READS.search(source) and "_unsloth_tp_heads" not in source:
        raise ValueError(
            f"Unsloth: the attention forward of '{path}' ({type(module).__name__}) reshapes with hidden_size, "
            "the global width; tensor parallelism v1 needs it to follow the local heads."
        )


def shard_slice(tensor, dim, parts, rank, tp):
    """This rank's slice of `tensor` along `dim`: the rank-th 1/tp of every part, concatenated (a copy)."""
    pieces, offset = [], 0
    for size in parts:
        step = size // tp
        pieces.append(tensor.narrow(dim, offset + rank * step, step))
        offset += size
    return torch.cat(pieces, dim = dim)


def _unpermute(tensor, dim, parts, tp):
    """Inverse of concatenating every rank's shard_slice: the original part order."""
    if len(parts) == 1:
        return tensor
    local = sum(parts) // tp
    chunks = tensor.split(local, dim = dim)  # one per rank
    pieces = []
    for i, size in enumerate(parts):
        step = size // tp
        start = sum(p // tp for p in parts[:i])
        pieces.extend(chunk.narrow(dim, start, step) for chunk in chunks)
    return torch.cat(pieces, dim = dim)


def _dtensor_api():
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
    return DTensor, Partial, Replicate, Shard


def _is_dtensor(t):
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(t, DTensor)


def _local_complete(t):
    """A local view whose gradient is already complete (e.g. row bias): never summed over tp again."""
    DTensor, _, Replicate, _ = _dtensor_api()
    if not isinstance(t, DTensor):
        return t
    if t.requires_grad and any(p.is_replicate() for p in t.placements):
        return t.to_local(grad_placements = [Replicate()])
    return t.to_local()


def _local(t):
    """A local view of a DTensor for local compute; replicated trainable tensors return Partial grads."""
    DTensor, Partial, _, _ = _dtensor_api()
    if not isinstance(t, DTensor):
        return t
    if t.requires_grad and any(p.is_replicate() for p in t.placements):
        return t.to_local(grad_placements = [Partial()])
    return t.to_local()


def _tp_lora_parameters(proj, bias = False):
    """get_lora_parameters(_bias) on local shards; row bias is None (the region adds it after the reduce)."""
    weight, base = _weight(proj)
    W = _local(weight)
    W_quant = getattr(W, "quant_state", None)
    b = None
    if bias and proj.__dict__.get("_unsloth_tp_style") == COLWISE and base.bias is not None:
        b = _local(base.bias)
    lora_A = proj._modules.get("lora_A")
    adapter = None
    if lora_A is not None and not (
        getattr(proj, "disable_adapters", True) or getattr(proj, "merged", False)
    ):
        adapters = _active_adapter(proj)
        if len(adapters) > 1:
            raise ValueError("Unsloth: LoRA parameter extraction requires a single active adapter.")
        if adapters and adapters[0] in lora_A:
            adapter = adapters[0]
    if adapter is None:
        out = (W, W_quant, None, None, None)
    else:
        out = (
            W,
            W_quant,
            _local(lora_A[adapter].weight),
            _local(proj.lora_B[adapter].weight),
            proj.scaling[adapter],
        )
    return out + (b,) if bias else out


def _replicate_grad(grad):
    # The Partial gradient of a replicated factor becomes its tp sum here (one all-reduce per backward).
    if _is_dtensor(grad) and any(p.is_partial() for p in grad.placements):
        _, _, Replicate, _ = _dtensor_api()
        return grad.redistribute(grad.device_mesh, [Replicate()])
    return grad


def _is_params4bit(t):
    return any(cls.__name__ == "Params4bit" for cls in type(t).__mro__)


def _refuse_base(path, proj_module, weight, base):
    cls = type(base).__name__
    if weight is None:
        raise ValueError(f"Unsloth: '{path}' ({cls}) has no dense weight (GPTQ / AWQ?).")
    if getattr(base, "qweight", None) is not None or cls.startswith(("QuantLinear", "WQLinear")):
        raise ValueError(
            f"Unsloth: tensor parallelism v1 does not support GPTQ / AWQ weights ({path})."
        )
    if cls == "Linear8bitLt" or type(weight).__name__ == "Int8Params":
        raise ValueError(
            f"Unsloth: tensor parallelism v1 does not support bitsandbytes 8-bit weights ({path}); load in "
            "4-bit or 16-bit."
        )
    fp8 = {
        getattr(torch, n, None)
        for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    }
    if weight.dtype in fp8 or getattr(base, "weight_scale", None) is not None:
        raise ValueError(f"Unsloth: tensor parallelism v1 does not support fp8 weights ({path}).")
    if getattr(type(base), "_unsloth_mxfp4_packed_linear", False):
        raise ValueError(
            f"Unsloth: tensor parallelism v1 does not support packed MXFP4 weights ({path})."
        )
    for module in proj_module.modules():
        forward_qat = any("FakeQuantized" in c.__name__ for c in type(module).__mro__)
        if (
            forward_qat
            or getattr(module, "weight_fake_quantizer", None) is not None
            or getattr(module, "activation_fake_quantizer", None) is not None
        ):
            # The TP projection forward and get_lora_parameters run plain matmuls on the local shards.
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support QAT fake quantization ({path}: "
                f"{type(module).__name__})."
            )
    if not _is_params4bit(weight) and type(weight) not in (nn.Parameter, torch.Tensor):
        raise ValueError(
            f"Unsloth: tensor parallelism v1 cannot shard '{path}' weights of type {type(weight).__name__}."
        )
    if not _is_params4bit(weight) and not weight.is_floating_point():
        raise ValueError(
            f"Unsloth: tensor parallelism v1 cannot shard {weight.dtype} weights ({path})."
        )
    if weight.requires_grad:
        raise ValueError(
            f"Unsloth: tensor parallelism v1 shards LoRA / QLoRA training only; '{path}' has a trainable base "
            "weight (full finetuning). Use native FSDP2 for full finetuning."
        )


def _active_adapter(proj):
    """Active adapter names, resolved like kernels/utils.py get_lora_parameters; empty stays empty."""
    adapters = getattr(proj, "active_adapters", None)
    if adapters is None:
        adapters = getattr(proj, "active_adapter", "default")
    if adapters is None:
        return []
    if isinstance(adapters, str):
        return [adapters]
    return list(adapters)


def _refuse_adapter(path, proj):
    if "original_module" in proj._modules or "modules_to_save" in proj._modules:
        raise ValueError(
            f"Unsloth: tensor parallelism v1 cannot shard '{path}': it is a PEFT modules_to_save / trainable "
            "copy (a full trainable projection). Keep modules_to_save outside the attention / MLP projections."
        )
    if "lora_A" not in proj._modules:
        if "base_layer" in proj._modules:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 supports LoRA adapters only ({path}: {type(proj).__name__})."
            )
        return
    names = list(proj.lora_A.keys())
    if len(names) > 1:
        raise ValueError(
            f"Unsloth: tensor parallelism v1 supports one LoRA adapter per layer ({path} has {names})."
        )
    magnitude = getattr(proj, "lora_magnitude_vector", None)
    if magnitude is not None and len(magnitude):
        raise ValueError(f"Unsloth: tensor parallelism v1 does not support DoRA ({path}).")
    variants = proj.__dict__.get("lora_variant")
    if variants:
        raise ValueError(f"Unsloth: tensor parallelism v1 does not support LoRA variants ({path}).")
    for name in names:
        dropout = proj.lora_dropout[name] if "lora_dropout" in proj._modules else None
        if getattr(dropout, "p", 0) > 0:
            # Masks must match across ranks for replicated inputs and differ for sharded ones.
            raise ValueError(
                f"Unsloth: tensor parallelism v1 does not support lora_dropout > 0 ({path}); use lora_dropout = 0."
            )
        if proj.lora_B[name].bias is not None:
            raise ValueError(f"Unsloth: tensor parallelism v1 does not support lora_bias ({path}).")
        if type(proj.lora_A[name]) is not nn.Linear:
            raise ValueError(
                f"Unsloth: tensor parallelism v1 supports Linear LoRA factors only ({path})."
            )


def _as_param(tensor, like):
    param = nn.Parameter(tensor, requires_grad = like.requires_grad)
    return param


def _set_dtensor(
    module,
    name,
    full,
    mesh,
    placement,
    parts = None,
    shard_dim = None,
    rank = 0,
    tp = 1,
):
    """Replace module.<name> by a DTensor parameter (Shard with parts permuted per rank, or Replicate)."""
    DTensor, _, Replicate, Shard = _dtensor_api()
    if placement == "shard":
        local = shard_slice(full.detach(), shard_dim, parts, rank, tp)
        shape = list(full.shape)
        dt = DTensor.from_local(
            local,
            mesh,
            [Shard(shard_dim)],
            run_check = False,
            shape = torch.Size(shape),
            stride = torch.empty(shape, device = "meta").stride(),
        )
    else:
        dt = DTensor.from_local(full.detach(), mesh, [Replicate()], run_check = False)
    param = _as_param(dt, full)
    if placement == "shard" and len(parts) > 1:
        param._unsloth_tp_parts = (shard_dim, tuple(parts))
    module._parameters[name] = param
    if param.requires_grad and placement != "shard":
        param.register_hook(_replicate_grad)
    return param


def _check_params4bit(path, weight, style, tp):
    """The 4-bit blocksize refusals, run before anything is sharded."""
    state = weight.quant_state
    blocksize = state.blocksize
    in_features = state.shape[1]
    if in_features % blocksize or blocksize % 2:
        raise ValueError(
            f"Unsloth: 4-bit '{path}': in_features = {in_features} is not a multiple of the quantization "
            f"blocksize {blocksize}; its blocks span rows, which tensor parallelism v1 cannot split."
        )
    if style == ROWWISE and (in_features // tp) % blocksize:
        raise ValueError(
            f"Unsloth: 4-bit '{path}': in_features / tp = {in_features // tp} is not a multiple of the "
            f"quantization blocksize {blocksize}; row sharding would split blocks."
        )


def _shard_params4bit(path, base, weight, style, parts, rank, tp):
    """This rank's 4-bit shard (bitwise slice of the 1-GPU weight); double-quant absmax kept as float32."""
    import bitsandbytes as bnb

    _check_params4bit(path, weight, style, tp)
    state = weight.quant_state
    blocksize = state.blocksize
    out_features, in_features = state.shape
    codes = weight.data.reshape(-1).view(torch.uint8).view(out_features, in_features // 2)
    absmax = state.absmax
    if state.nested:
        absmax = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
    absmax = absmax.float().view(out_features, in_features // blocksize)
    if style == COLWISE:
        codes = shard_slice(codes, 0, parts, rank, tp)
        absmax = shard_slice(absmax, 0, parts, rank, tp)
        shape = (sum(parts) // tp, in_features)
    else:
        codes = shard_slice(codes, 1, (in_features // 2,), rank, tp)
        absmax = shard_slice(absmax, 1, (in_features // blocksize,), rank, tp)
        shape = (out_features, in_features // tp)
    storage = weight.data.dtype
    packed = codes.contiguous().view(storage).reshape(-1, 1)
    local_state = bnb.functional.QuantState(
        absmax = absmax.contiguous(),
        shape = torch.Size(shape),
        code = state.code,
        blocksize = blocksize,
        quant_type = state.quant_type,
        dtype = state.dtype,
    )
    new = bnb.nn.Params4bit(
        packed,
        requires_grad = False,
        quant_state = local_state,
        blocksize = blocksize,
        compress_statistics = False,
        quant_type = state.quant_type,
        quant_storage = storage,
        module = base,
        bnb_quantized = True,
    )
    new._unsloth_tp_layout = (style, tuple(parts), rank, tp)
    base._parameters["weight"] = new
    if "quant_state" in base.__dict__:
        base.quant_state = local_state


def _all_reduce(tensor, group):
    import torch.distributed._functional_collectives as funcol

    out = funcol.all_reduce(tensor.contiguous(), "sum", group)
    wait = getattr(out, "wait", None)
    return wait() if callable(wait) else out


class _CopyToTP(torch.autograd.Function):
    """Identity forward, all-reduce backward: the region input every rank reads in full."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad, ctx.group), None


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce forward, identity backward: the row projection's partial sums."""

    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce(x, group)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def copy_to_tp(x, group):
    return _CopyToTP.apply(x, group)


def reduce_from_tp(x, group):
    return _ReduceFromTP.apply(x, group)


def _input_name(module):
    """The name of the region forward's first argument (hidden_states, x, ...), for keyword calls."""
    try:
        params = list(inspect.signature(module.forward).parameters.values())
    except (TypeError, ValueError):
        return "hidden_states"
    if params and params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD):
        return params[0].name
    return "hidden_states"


def _region_pre_hook(module, args, kwargs):
    group = module._unsloth_tp_group
    for name in (module.__dict__.get("_unsloth_tp_input", "hidden_states"), "hidden_states"):
        if isinstance(kwargs.get(name), torch.Tensor):
            kwargs = dict(kwargs)
            kwargs[name] = copy_to_tp(kwargs[name], group)
            return args, kwargs
    if args and isinstance(args[0], torch.Tensor):
        return (copy_to_tp(args[0], group),) + tuple(args[1:]), kwargs
    raise RuntimeError(
        f"Unsloth: tensor parallel region {type(module).__name__} got no tensor input "
        f"({module.__dict__.get('_unsloth_tp_input', 'hidden_states')})."
    )


def _region_finish(module, y):
    y = reduce_from_tp(y, module._unsloth_tp_group)
    for proj in module._unsloth_tp_row_projs:
        bias = _weight(proj)[1].bias
        if bias is not None:
            y = y + _local_complete(bias).to(y.dtype)
    return y


def _region_post_hook(module, args, output):
    if isinstance(output, torch.Tensor):
        return _region_finish(module, output)
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return type(output)((_region_finish(module, output[0]), *output[1:]))
    raise RuntimeError(
        f"Unsloth: tensor parallel region {type(module).__name__} returned {type(output)}."
    )


def _tp_projection_forward(self, x, *args, **kwargs):
    weight, base = _weight(self)
    W = _local(weight)
    bias = (
        base.bias if self._unsloth_tp_style == COLWISE else None
    )  # row bias: once, after the reduce
    bias = None if bias is None else _local(bias)
    input_dtype = x.dtype
    if _is_params4bit(W):
        import bitsandbytes as bnb

        compute = getattr(base, "compute_dtype", None) or x.dtype
        out = bnb.matmul_4bit(x.to(compute), W.t(), bias = None, quant_state = W.quant_state)
        if bias is not None:
            out = out + bias.to(out.dtype)
        out = out.to(input_dtype)
    else:
        out = F.linear(x, W, bias)
    lora_A = self._modules.get("lora_A")
    if lora_A is None or getattr(self, "disable_adapters", True) or getattr(self, "merged", False):
        return out
    for adapter in _active_adapter(self) or ():
        if adapter not in lora_A:
            continue
        A = _local(lora_A[adapter].weight)
        B = _local(self.lora_B[adapter].weight)
        scaling = self.scaling[adapter]
        x_lora = x if torch.is_autocast_enabled() else x.to(A.dtype)
        x_lora = self.lora_dropout[adapter](x_lora)
        out = out + (F.linear(F.linear(x_lora, A), B) * scaling).to(out.dtype)
    return out


def _decoder_of(inner):
    get_decoder = getattr(inner, "get_decoder", None)
    if not callable(get_decoder):
        return None
    try:
        decoder = get_decoder()
    except Exception:
        return None
    return decoder if isinstance(decoder, nn.Module) else None


def _tp_chain(model):
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


def is_tensor_parallel(model) -> bool:
    """True when apply_tensor_parallel sharded `model`, a wrapper level below it, or any module inside it."""
    if not isinstance(model, nn.Module):
        return getattr(model, _TP_FLAG, None) is not None
    try:
        chain = _tp_chain(model)
    except Exception:
        chain = [model]
    if any(m.__dict__.get(_TP_FLAG) is not None for m in chain):
        return True
    return any(m.__dict__.get("_unsloth_tp_style") is not None for m in model.modules())


def _default_mesh(model):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        raise RuntimeError(
            "Unsloth: apply_tensor_parallel needs an initialized process group (torchrun) or a tp_mesh."
        )
    device = next(model.parameters()).device
    return init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names = ("tp",))


def _refuse_generate(*args, **kwargs):
    raise RuntimeError(
        "Unsloth: generation is not supported on a tensor parallel model in v1; save the adapter "
        "(save_tp_adapter) and generate on one GPU."
    )


def _cache_in_use(past):
    if past is None:
        return False
    length = getattr(past, "get_seq_length", None)
    if callable(length):
        try:
            return int(length()) > 0
        except Exception:
            return True
    try:
        return len(past) > 0
    except TypeError:
        return True


def _refuse_cached_forward(module):
    """Refuse a filled KV cache (instance attribute: LoraModel calls self.model.forward directly)."""
    forward = module.forward
    try:
        names = list(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        names = []
    index = names.index("past_key_values") if "past_key_values" in names else None

    def forward_without_cache(*args, **kwargs):
        past = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if past is None and index is not None and len(args) > index:
            past = args[index]
        if _cache_in_use(past):
            raise RuntimeError(
                "Unsloth: cached decoding (past_key_values) is not supported on a tensor parallel model in "
                "v1; save the adapter (save_tp_adapter) and generate on one GPU."
            )
        return forward(*args, **kwargs)

    forward_without_cache.__wrapped__ = forward
    module.__dict__["forward"] = forward_without_cache


def _sync_trainable(model, group, src):
    import torch.distributed as dist
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                dist.broadcast(param.data, src = src, group = group)


def _set_local_heads(module, region, tp):
    heads, kv_heads = region.heads // tp, region.kv_heads // tp
    module._unsloth_tp_heads = heads
    module._unsloth_tp_kv_heads = kv_heads
    for name, value, always in (
        ("num_heads", heads, True),
        ("num_key_value_heads", kv_heads, True),
        ("num_attention_heads", heads, False),
        ("num_kv_heads", kv_heads, False),
    ):
        if always or isinstance(module.__dict__.get(name), int):
            module.__dict__[name] = value
    config = module.__dict__.get("config")
    if config is None:
        return
    try:
        source = _forward_source(module)
    except (OSError, TypeError):
        source = ""
    if _GLOBAL_HEAD_READS.search(source) and "_unsloth_tp_heads" not in source:
        # Per-module config copy with local counts (Phi3 fused qkv); the shared config is left alone.
        local = copy.copy(config)
        local.num_attention_heads = heads
        if getattr(config, "num_key_value_heads", None) is not None:
            local.num_key_value_heads = kv_heads
        module.__dict__["config"] = local


def apply_tensor_parallel(
    model,
    tp_mesh = None,
    plan = None,
    config: Optional[TPConfig] = None,
) -> nn.Module:
    """Shard every decoder attention / MLP of `model` over `tp_mesh` (None = all ranks), in place."""
    if is_tensor_parallel(model):
        return model
    config = config or TPConfig()
    if plan is None:
        plan = config.plan
    if tp_mesh is None:
        tp_mesh = _default_mesh(model)
    if tp_mesh.ndim != 1:
        raise ValueError("Unsloth: apply_tensor_parallel takes a 1-D tp mesh (pass mesh['tp']).")
    tp = tp_mesh.size()
    rank = tp_mesh.get_local_rank()
    group = tp_mesh.get_group()
    if (
        any(_is_dtensor(p) for p in model.parameters())
        or (getattr(model, "tp_size", None) or 1) > 1
    ):
        raise ValueError(
            "Unsloth: this model is already distributed (DTensor parameters or a transformers tp_plan); "
            "load it without tp_plan before apply_tensor_parallel."
        )
    tp_plan = derive_tp_plan(model, tp, plan)
    for region in tp_plan.regions:
        for proj in region.projs:
            path = f"{region.path}.{proj.name}"
            weight, base = _weight(proj.module)
            _refuse_base(path, proj.module, weight, base)
            _refuse_adapter(path, proj.module)
            if _is_params4bit(weight):
                _check_params4bit(path, weight, proj.style, tp)
        for name, norm in region.norms:
            for p in norm.parameters():
                if type(p) not in (nn.Parameter, torch.Tensor):
                    raise ValueError(
                        f"Unsloth: cannot replicate {region.path}.{name} ({type(p).__name__})."
                    )
    chain = _tp_chain(model)
    if config.sync_trainable and tp > 1:
        import torch.distributed as dist
        _sync_trainable(model, group, dist.get_global_rank(group, 0))
    hooked_norms = (
        set()
    )  # a norm weight shared by several regions gets one tp sum, not one per region
    for region in tp_plan.regions:
        for proj in region.projs:
            _shard_projection(region, proj, tp_mesh, rank, tp)
        for _, norm in region.norms:
            for p in norm.parameters():
                if p.requires_grad and id(p) not in hooked_norms:
                    hooked_norms.add(id(p))
                    # Each rank normalizes its own heads with the shared weight: the grad is a partial sum.
                    p.register_hook(lambda g, group = group: _all_reduce(g, group))
        module = region.module
        module._unsloth_tp_group = group
        module._unsloth_tp_row_projs = [p.module for p in region.projs if p.style == ROWWISE]
        module._unsloth_tp_input = _input_name(module)
        if region.kind == "attention":
            _set_local_heads(module, region, tp)
        module.register_forward_pre_hook(_region_pre_hook, with_kwargs = True)
        module.register_forward_hook(_region_post_hook)
    # Unsloth decode paths bypass the regions: mark and guard every wrapper level.
    info = {"size": tp, "rank": rank, "regions": len(tp_plan.regions), "source": tp_plan.source}
    inner = _unwrap(model)
    decoder = _decoder_of(inner)
    for level in chain:
        level.__dict__[_TP_FLAG] = info
        level.__dict__["_unsloth_tp_mesh"] = tp_mesh
        level.__dict__["_unsloth_tp_plan"] = tp_plan
        if callable(getattr(level, "generate", None)):
            level.__dict__["generate"] = _refuse_generate
        if level is inner or level is decoder:
            _refuse_cached_forward(level)
    if rank == 0:
        print(
            f"Unsloth: tensor parallel over {tp} ranks: {len(tp_plan.regions)} attention / MLP regions "
            f"({tp_plan.source} plan), embeddings / lm_head / norms replicated"
            + (
                f", {tp_plan.replicated_layers} layers replicated"
                if tp_plan.replicated_layers
                else ""
            )
            + "."
        )
    return model


def _shard_projection(region, proj, mesh, rank, tp):
    module = proj.module
    weight, base = _weight(module)
    col = proj.style == COLWISE
    parts = proj.parts
    in_parts = (_logical_shape(weight, base)[1],)
    if _is_params4bit(weight):
        _shard_params4bit(f"{region.path}.{proj.name}", base, weight, proj.style, parts, rank, tp)
    else:
        _set_dtensor(
            base,
            "weight",
            weight,
            mesh,
            "shard",
            parts if col else in_parts,
            0 if col else 1,
            rank,
            tp,
        )
    if base.bias is not None:
        if col:
            _set_dtensor(base, "bias", base.bias, mesh, "shard", parts, 0, rank, tp)
        else:
            _set_dtensor(base, "bias", base.bias, mesh, "replicate")
    lora_A = module._modules.get("lora_A")
    if lora_A is not None:
        for adapter in lora_A.keys():
            A, B = lora_A[adapter], module.lora_B[adapter]
            if col:
                _set_dtensor(A, "weight", A.weight, mesh, "replicate")
                _set_dtensor(B, "weight", B.weight, mesh, "shard", parts, 0, rank, tp)
            else:
                _set_dtensor(A, "weight", A.weight, mesh, "shard", in_parts, 1, rank, tp)
                _set_dtensor(B, "weight", B.weight, mesh, "replicate")
    module._unsloth_tp_style = proj.style
    module._unsloth_tp_local = (
        _tp_lora_parameters  # read by kernels/utils.py get_lora_parameters(_bias)
    )
    module.forward = types.MethodType(_tp_projection_forward, module)


def tp_clip_grad_norm_(
    parameters,
    max_norm,
    mesh = None,
    eps = 1e-6,
):
    """Clip by the global 2-norm: sharded grads summed over tp, replicated ones counted once."""
    import torch.distributed as dist

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    sharded = replicated = None
    group = mesh.get_group() if mesh is not None else None
    for p in params:
        g = p.grad
        if _is_dtensor(g):
            g = _replicate_grad(g)
            if g is not p.grad:
                p.grad = g
            group = g.device_mesh.get_group() if group is None else group
            local = g.to_local().float()
            sq = local.square().sum()
            if any(pl.is_shard() for pl in g.placements):
                sharded = sq if sharded is None else sharded + sq
                continue
        else:
            sq = g.float().square().sum()
        replicated = sq if replicated is None else replicated + sq
    device = (
        params[0].grad.device
        if not _is_dtensor(params[0].grad)
        else params[0].grad.to_local().device
    )
    total = torch.zeros((), device = device) if replicated is None else replicated.to(device)
    if group is not None:
        shard_sq = torch.zeros((), device = device) if sharded is None else sharded.to(device)
        dist.all_reduce(shard_sq, group = group)
        total = total + shard_sq
    total = total.sqrt()
    coef = float(max_norm) / (float(total) + eps)
    if coef < 1:
        for p in params:
            p.grad.mul_(coef)
    return total


def _full(tensor, param = None):
    if not _is_dtensor(tensor):
        return tensor
    full = tensor.full_tensor()
    layout = getattr(param, "_unsloth_tp_parts", None) if param is not None else None
    if layout is not None:
        dim, parts = layout
        full = _unpermute(full, dim, parts, tensor.device_mesh.size())
    return full


def tp_full_state_dict(model, adapters_only = False):
    """State dict with DTensors gathered (collective); 4-bit shards stay local."""
    state = model.state_dict(keep_vars = True)
    # PEFT picks embedding weights out of the given state dict (save_embedding_layers, resized vocab).
    embeddings = set()
    for get in ("get_input_embeddings", "get_output_embeddings"):
        module = getattr(model, get, lambda: None)()
        weight = getattr(module, "weight", None)
        if weight is not None:
            embeddings.add(id(weight))
    out = {}
    with torch.no_grad():
        for key, value in state.items():
            if (
                adapters_only
                and "lora_" not in key
                and not getattr(value, "requires_grad", False)
                and id(value) not in embeddings
            ):
                continue
            out[key] = _full(value, value).detach()
    return out


def _peft_state_dict(model, **kwargs):
    """What PEFT selects for save_pretrained, DTensors gathered; local tensors by reference (PEFT reruns
    its selection inside save_pretrained and must see the same model)."""
    from peft import get_peft_model_state_dict

    raw = model.state_dict(keep_vars = True)
    adapters = kwargs.get("selected_adapters") or list(model.peft_config)
    known = {id(value) for value in raw.values()}
    wanted, unmapped = set(), 0
    for adapter in adapters:
        selected = get_peft_model_state_dict(
            model,
            state_dict = raw,
            adapter_name = adapter,
            save_embedding_layers = kwargs.get("save_embedding_layers", "auto"),
        )
        wanted.update(id(value) for value in selected.values())
        unmapped += sum(_is_dtensor(v) and id(v) not in known for v in selected.values())
    if unmapped:
        raise RuntimeError(
            "Unsloth: PEFT selected tensor parallel tensors that are not in the model state dict; "
            "save_tp_adapter cannot gather them."
        )
    state = {}
    with torch.no_grad():
        for key, value in raw.items():
            if _is_dtensor(value):
                if id(value) in wanted:
                    state[key] = _full(value, value).detach()
            else:
                state[key] = value.detach()
    return state


def save_tp_adapter(model, save_directory, **kwargs):
    """Write the standard 1-GPU PEFT adapter. Collective (tp mesh = whole world in v1); rank 0 writes."""
    import torch.distributed as dist

    mesh = None
    for level in _tp_chain(model):
        mesh = mesh or level.__dict__.get("_unsloth_tp_mesh")
    if dist.is_initialized() and mesh is not None and mesh.size() != dist.get_world_size():
        raise ValueError(
            f"Unsloth: save_tp_adapter writes on global rank 0 behind a world barrier, but the tensor parallel "
            f"mesh has {mesh.size()} of {dist.get_world_size()} ranks (data parallel > 1 is outside v1). "
            "Gather with tp_full_state_dict and write from one rank yourself."
        )
    state = _peft_state_dict(model, **kwargs)
    rank = dist.get_rank() if dist.is_initialized() else 0
    kwargs.setdefault("is_main_process", rank == 0)
    if kwargs["is_main_process"]:
        model.save_pretrained(save_directory, state_dict = state, **kwargs)
    if dist.is_initialized():
        dist.barrier()
