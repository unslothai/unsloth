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

Sharded loading (16-bit safetensors checkpoints): each rank reads only its slices of the attention / MLP
projections, so the load-time peak per rank is the replicated parts plus 1 / tp of the rest:

    model, tokenizer = load_tensor_parallel("unsloth/Qwen3-8B", mesh["tp"], dtype = torch.float16)
    # = with sharded_loading(mesh["tp"]): FastLanguageModel.from_pretrained(..., load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, ...)  # full-size LoRA, same init as 1 GPU
    apply_tensor_parallel(model, mesh["tp"])  # required before the first forward

Data parallel x tensor parallel (native FSDP2 over the dp dimension, torch 2.7+): every tp rank of a dp replica
reads the same batch, dp replicas read different ones:

    mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names = ("dp", "tp"))
    apply_tensor_parallel(model, mesh["tp"])
    shard_model(model, FSDP2Config(mesh = mesh["dp"]))  # frozen tp shards replicated over dp, LoRA sharded
    ... tp_clip_grad_norm_(params, 1.0) (or native_clip_grad_norm_); save_tp_adapter(model, "out")

The HF / TRL Trainer trains a tensor parallel model (tp only, or dp x tp after shard_model above): no DDP over
the tp ranks, batches sharded over dp only, tp-aware clipping and float16 scaler. Trainer checkpoints are per
rank (unsloth_tp/rank*.pt: local shards of the trainable weights and optimizer state, resumed on the same
layout) next to the 1-GPU adapter; bitsandbytes Adam / AdamW step the local shards; no evaluation;
trainer.save_model() writes the 1-GPU adapter.

Unmodified scripts: UNSLOTH_TP_SIZE=k torchrun ... applies all of the above from the environment
(models/tp_env.py), with UNSLOTH_FSDP2=1 (native FSDP2) or DDP-style gradient averaging for the dp dimension.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
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
    "check_tp_dp_composition",
    "derive_tp_plan",
    "is_tensor_parallel",
    "load_tensor_parallel",
    "sharded_loading",
    "shard_slice",
    "tp_clip_grad_norm_",
    "tp_full_state_dict",
    "tp_trainer_setup",
    "save_tp_adapter",
]

_TP_FLAG = "_unsloth_tp"
COLWISE = "colwise"
ROWWISE = "rowwise"
HEAD_NORM = "replicated_with_grad_allreduce"
ELEMENTWISE = "elementwise_replicated"  # scalar-parameter activations inside an MLP

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
_MLP_COL = ("gate_proj", "up_proj", "w1", "w3", "fc1", "c_fc", "fc_in", "dense_h_to_4h")
_MLP_PACKED = ("gate_up_proj",)  # [gate, up] along the output dim
_MLP_ROW = ("down_proj", "w2", "fc2", "fc_out", "dense_4h_to_h", "c_proj")
_HEAD_NORMS = (
    "q_norm",
    "k_norm",
    "query_norm",
    "key_norm",
    "q_layernorm",
    "k_layernorm",
    "query_layernorm",
    "key_layernorm",
)
_MLA_NAMES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm")
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# An attention forward reading these sizes off the config would reshape to the global head count.
_GLOBAL_HEAD_READS = re.compile(r"config\.(num_attention_heads|num_key_value_heads)\b")
_GLOBAL_WIDTH_READS = re.compile(
    r"(config|self)\.(hidden_size|embed_dim|all_head_size|n_embd|d_model|split_size)\b"
)
# An MLP forward reducing over its features (activation sparsity statistics, a norm, a softmax) would see
# only this rank's 1 / tp of the intermediate dimension; one reading the intermediate size splits wrongly.
_FEATURE_REDUCTIONS = re.compile(
    r"(?:\btorch\.|\bF\.|\bfunctional\.|\.)(mean|std|var|var_mean|std_mean|sum|nansum|prod|amax|amin|"
    r"argmax|argmin|topk|kthvalue|sort|argsort|softmax|log_softmax|logsumexp|cumsum|cumprod|norm|normalize|"
    r"layer_norm|rms_norm|group_norm)\s*\("
)
_GLOBAL_INTERMEDIATE_READS = re.compile(r"(config|self)\.(intermediate_size|ffn_dim|n_inner)\b")
# A fused qkv_proj is [q, k, v] blocks only when the forward slices it by the (local) head counts (Phi3).
_LOCAL_QKV_SLICE = re.compile(r"(num_key_value_heads|num_kv_heads)\s*\*\s*self\.head_dim")


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
    # A decoder-only use of an encoder-decoder config (BartForCausalLM): num_hidden_layers is the encoder's.
    counts = {n, getattr(text, "decoder_layers", None)} - {None}
    prefix = ""
    for name, module in inner.named_modules():
        if module is decoder:
            prefix = name
            break
    for name, module in decoder.named_modules():
        if (
            isinstance(module, nn.ModuleList)
            and len(module)
            and (n is None or len(module) in counts)
        ):
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
        if _elementwise_scalars(child):
            return ELEMENTWISE, None
    return None, None


def _elementwise_scalars(module):
    """A parameterised activation with scalar parameters only (Apertus xIELU): elementwise on any shard of
    the intermediate dimension, so it stays replicated (gradients summed over tp)."""
    if isinstance(module, nn.Linear) or "lora_A" in module._modules or list(module.children()):
        return False
    if any(p.numel() != 1 for p in module.parameters()):
        return False
    try:
        source = inspect.getsource(type(module))
    except (OSError, TypeError):
        return False
    return not _FEATURE_REDUCTIONS.search(_drop_init(source))


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
    regions, replicated, seen = [], 0, {}
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
            found += 1
            if id(child) in seen:
                continue  # a block shared by several layers (Zamba2, ALBERT style): sharded once
            seen[id(child)] = path
            regions.append(_region(path, child, kind, compiled, text, tp_size))
        # A pipeline parallel pass-through (a layer another stage owns) is not a replicated layer.
        replicated += found == 0 and "_pp_index" not in layer.__dict__
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
        if style in (HEAD_NORM, ELEMENTWISE):
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
            weight = norm._parameters.get("weight")
            numel = (
                weight.numel() if weight is not None else sum(p.numel() for p in norm.parameters())
            )
            if numel != head_dim:
                raise ValueError(
                    f"Unsloth: '{path}.{name}' normalizes {numel} features, not one head ({head_dim}): a norm "
                    "across all heads needs the full projection, which tensor parallelism v1 does not gather."
                )
        _check_attention_source(module, path)
    else:
        if any(not _elementwise_scalars(norm) for _, norm in norms):
            raise ValueError(f"Unsloth: unexpected per-head norm in MLP '{path}'.")
        _check_mlp_source(module, path)
    for proj in projs:
        weight, base = _weight(proj.module)
        out_features, in_features = _logical_shape(weight, base)
        if proj.style == ROWWISE:
            _check_divisible(f"{proj.name} in_features", in_features, tp, path)
            proj.parts = (out_features,)
            continue
        if proj.parts == "qkv":
            _check_fused_qkv_source(module, path, proj.name)
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
    full = getattr(weight, "_unsloth_tp_full_shape", None)  # a sharded_loading slice
    if full is not None:
        return tuple(full)
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


def _drop_init(source):
    """A class source without its __init__ (sizes read there are construction, not forward math)."""
    return re.sub(r"\n(\s+)def __init__\(.*?(?=\n\1def |\Z)", "\n", source, flags = re.S)


def _region_source(module, path, what):
    """The forward of a region module plus the methods of its own class that forward calls (self._helper)."""
    try:
        source = _forward_source(module)
    except (OSError, TypeError):
        raise ValueError(
            f"Unsloth: tensor parallelism cannot read the forward of '{path}' ({type(module).__name__}) to "
            f"check {what}."
        )
    seen = set()
    for name in re.findall(r"\bself\.(\w+)\s*\(", source):
        if name in seen or name in module._modules or name == "forward":
            continue
        seen.add(name)
        method = getattr(type(module), name, None)
        if not inspect.isfunction(method):
            continue
        try:
            source += "\n" + inspect.getsource(inspect.unwrap(method))
        except (OSError, TypeError):
            pass
    return source


def _input_width_names(source, module):
    """Variables a forward reads the full input width into (`bsz, len, dim = hidden_states.size()`)."""
    names = set()
    for x in {_input_name(module), "hidden_states"}:
        x = re.escape(x)
        for m in re.finditer(
            r"\b\w+\s*,\s*\w+\s*,\s*(\w+)\s*=\s*" + x + r"\.(?:size\(\)|shape\b)", source
        ):
            names.add(m.group(1))
        for m in re.finditer(
            r"\b(\w+)\s*=\s*" + x + r"\.(?:shape\[(?:-1|2)\]|size\((?:-1|2)\))", source
        ):
            names.add(m.group(1))
    names.discard("_")
    return sorted(n for n in names if len(re.findall(r"\b" + re.escape(n) + r"\b", source)) > 1)


def _check_attention_source(module, path):
    source = _region_source(module, path, "it uses this rank's head count")
    if "_unsloth_tp_heads" in source:
        return  # an Unsloth forward written for tensor parallelism
    read = _GLOBAL_WIDTH_READS.search(source)
    if read:
        raise ValueError(
            f"Unsloth: the attention forward of '{path}' ({type(module).__name__}) reshapes with "
            f"{read.group(0)}, the global width; tensor parallelism v1 needs it to follow the local heads."
        )
    width = _input_width_names(source, module)
    if width:
        raise ValueError(
            f"Unsloth: the attention forward of '{path}' ({type(module).__name__}) keeps the input width "
            f"({', '.join(width)}) for its output shape; tensor parallelism v1 needs it to follow the local heads."
        )


def _check_mlp_source(module, path):
    source = _region_source(module, path, "it is elementwise over the intermediate features")
    if "_unsloth_tp" in source:
        return
    reduction = _FEATURE_REDUCTIONS.search(source)
    if reduction:
        raise ValueError(
            f"Unsloth: the MLP forward of '{path}' ({type(module).__name__}) reduces over its features "
            f"({reduction.group(0).strip()}...); with tensor parallelism each rank holds 1 / tp of them, so v1 "
            "cannot shard it."
        )
    read = _GLOBAL_INTERMEDIATE_READS.search(source)
    if read:
        raise ValueError(
            f"Unsloth: the MLP forward of '{path}' ({type(module).__name__}) reads {read.group(0)}, the global "
            "size; tensor parallelism v1 needs it to follow the local shard."
        )


def _check_fused_qkv_source(module, path, name):
    try:
        source = _forward_source(module)
    except (OSError, TypeError):
        source = ""
    if "_unsloth_tp_heads" in source or _LOCAL_QKV_SLICE.search(source):
        return
    raise ValueError(
        f"Unsloth: the attention forward of '{path}' ({type(module).__name__}) does not slice the fused "
        f"'{name}' by the key / value head count ([q, k, v] blocks); its layout is unknown to tensor "
        "parallelism v1."
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
    if (
        not _is_params4bit(weight)
        and _preloaded(weight) is None
        and type(weight) not in (nn.Parameter, torch.Tensor)
    ):
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
        _tag_layout(module, name, shard_dim, parts, tp)
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


def quant_state_copy(state, device):
    """A bitsandbytes QuantState on `device`; `state` itself (QuantState.to moves in place) is left as is."""
    out = copy.copy(state)
    if getattr(state, "state2", None) is not None:
        out.state2 = copy.copy(state.state2)
    out.to(device)
    return out


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
    # The 1-GPU QuantState (absmax still double quantized), on the CPU: tp_replica rebuilds the 1-GPU weight.
    new._unsloth_tp_quant_state = quant_state_copy(state, "cpu")
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
    if module.__dict__.get("_unsloth_tp_gathered"):
        return None  # tp_replica: whole weights in place, the region runs as on one GPU
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
    if module.__dict__.get("_unsloth_tp_gathered"):
        return None
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


_MISSING = object()


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


def _replica_generate(level, prior):
    """generate on a tensor parallel level: every rank gathers a whole replica first (tp_replica)."""

    def generate(*args, **kwargs):
        from .tp_replica import tp_generate
        return tp_generate(level, prior, args, kwargs)

    generate._unsloth_tp_generate = True
    return generate


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


def _mark_tokenizers(chain):
    """The tokenizer Unsloth keeps on the model saves once per tensor parallel world (save.py)."""
    for level in chain:
        tokenizer = level.__dict__.get("_saved_temp_tokenizer")
        if tokenizer is not None and hasattr(tokenizer, "__dict__"):
            from .tp_replica import mark_world_save
            mark_world_save(tokenizer)


def _sync_trainable(model, group, src):
    import torch.distributed as dist
    with torch.no_grad():
        for param in model.parameters():
            # sharded_loading slices differ per rank (read from the same checkpoint)
            if param.requires_grad and _preloaded(param) is None:
                dist.broadcast(param.data, src = src, group = group)


def _set_local_heads(module, region, tp):
    heads, kv_heads = region.heads // tp, region.kv_heads // tp
    # What tp_replica puts back while a whole replica is gathered.
    module.__dict__["_unsloth_tp_saved_attrs"] = {
        name: module.__dict__.get(name, _MISSING)
        for name in (
            "num_heads",
            "num_key_value_heads",
            "num_attention_heads",
            "num_kv_heads",
            "config",
        )
    }
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
    if any(level.__dict__.get("_unsloth_pp") is not None for level in _tp_chain(model)):
        # pp x tp (torchtitan order): each pipeline stage is tensor parallel over the tp dimension of the
        # same mesh. Checked before any change.
        _check_pp_tp_composition(model, tp_mesh)
    tp_plan = derive_tp_plan(model, tp, plan)
    covered = set()
    for region in tp_plan.regions:
        for proj in region.projs:
            path = f"{region.path}.{proj.name}"
            weight, base = _weight(proj.module)
            _refuse_base(path, proj.module, weight, base)
            _refuse_adapter(path, proj.module)
            _check_preloaded(path, proj, weight, base, tp_mesh)
            if id(weight) in covered:
                raise ValueError(
                    f"Unsloth: '{path}' shares its weight with another sharded projection; tensor "
                    "parallelism v1 shards every projection once."
                )
            covered.update((id(weight), id(base.bias)))
            if _is_params4bit(weight):
                _check_params4bit(path, weight, proj.style, tp)
        for name, norm in region.norms:
            for p in norm.parameters():
                if type(p) not in (nn.Parameter, torch.Tensor):
                    raise ValueError(
                        f"Unsloth: cannot replicate {region.path}.{name} ({type(p).__name__})."
                    )
    stray = [
        n for n, p in model.named_parameters() if _preloaded(p) is not None and id(p) not in covered
    ]
    if stray:
        raise ValueError(
            f"Unsloth: {stray[:3]} were sharded by sharded_loading but are not in this tensor parallel plan; "
            "pass the same plan to sharded_loading and apply_tensor_parallel."
        )
    if tp > 1 and covered and any(_preloaded(p) is not None for p in model.parameters()):
        _refuse_weight_dependent_lora_init(model)
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
        level.__dict__["_unsloth_tp_root"] = model
        if callable(getattr(level, "generate", None)):
            level.__dict__["generate"] = _replica_generate(level, level.__dict__.get("generate"))
        if level is inner or level is decoder:
            _refuse_cached_forward(level)
    _mark_tokenizers(chain)
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
    if _preloaded(weight) is not None:
        _wrap_preloaded(base, "weight", weight, mesh, 0 if col else 1)
    elif _is_params4bit(weight):
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
            if _preloaded(base.bias) is not None:
                _wrap_preloaded(base, "bias", base.bias, mesh, 0)
            else:
                _set_dtensor(base, "bias", base.bias, mesh, "shard", parts, 0, rank, tp)
        else:
            # sharded_loading keeps row biases as plain full tensors: replicated here as in v1
            _set_dtensor(base, "bias", base.bias, mesh, "replicate")
    guard = base.__dict__.pop("_unsloth_tp_load_guard", None)
    if guard is not None:
        guard.remove()
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
    module.__dict__["_unsloth_tp_prior_forward"] = module.__dict__.get("forward", _MISSING)
    module._unsloth_tp_style = proj.style
    module._unsloth_tp_local = (
        _tp_lora_parameters  # read by kernels/utils.py get_lora_parameters(_bias)
    )
    module.forward = types.MethodType(_tp_projection_forward, module)


def _check_pp_tp_composition(model, tp_mesh):
    from .pp_native import check_mesh_composition, pipeline_stage_of

    stage = pipeline_stage_of(model)
    if stage is None or stage.model is not model:
        raise ValueError(
            "Unsloth: tensor parallelism does not compose with Unsloth pipeline parallelism "
            "(apply_pipeline_parallel) here: pass the model apply_pipeline_parallel was given."
        )
    if stage.dp_mesh is not None:
        raise ValueError(
            "Unsloth: apply_tensor_parallel must come before shard_model on a pipeline parallel model "
            "(pp, then tp inside each stage, then FSDP2 over dp)."
        )
    import torch.distributed as dist

    meshes = {"pp": stage.mesh, "tp": tp_mesh}
    world = dist.get_world_size() if dist.is_initialized() else 1
    if stage.mesh.size() * tp_mesh.size() == world:
        try:
            check_mesh_composition(meshes, "pipeline parallel x tensor parallel")
        except ValueError as exc:
            raise ValueError(
                "Unsloth: tensor parallelism does not compose with Unsloth pipeline parallelism on these "
                f"meshes. {exc}"
            ) from None
    else:
        # pp x dp x tp: the dp dimension comes with shard_model; here only orthogonality and divisibility.
        pp_ranks, tp_ranks = set(stage.mesh.mesh.tolist()), set(tp_mesh.mesh.tolist())
        if (
            tp_mesh.ndim != 1
            or pp_ranks & tp_ranks != {dist.get_rank()}
            or world % (stage.mesh.size() * tp_mesh.size())
        ):
            raise ValueError(
                "Unsloth: tensor parallelism does not compose with Unsloth pipeline parallelism on these "
                f"meshes: the tp mesh (ranks {sorted(tp_ranks)}) must be the tensor parallel dimension of the "
                f"mesh whose pp dimension (ranks {sorted(pp_ranks)}) apply_pipeline_parallel used: pass "
                "mesh['tp'] of init_device_mesh(..., (pp, tp)) or (pp, dp, tp)."
            )
    return stage


def _pp_size(model):
    for level in _tp_chain(model):
        info = level.__dict__.get("_unsloth_pp")
        if info is not None:
            return int(info.get("size", 1))
    return 1


def check_tp_dp_composition(model, dp_mesh):
    """shard_model(model, FSDP2Config(mesh = dp_mesh)) on a model sharded by apply_tensor_parallel: valid when
    dp_mesh and the tp mesh are the two dimensions of one (dp, tp) mesh covering the world. Marks the model."""
    import torch.distributed as dist

    chain = _tp_chain(model)
    tp_mesh = next(
        (m.__dict__["_unsloth_tp_mesh"] for m in chain if "_unsloth_tp_mesh" in m.__dict__), None
    )
    if tp_mesh is None:
        raise ValueError(
            "Unsloth: native FSDP2 over a tensor parallel model needs the outermost model (the one "
            "apply_tensor_parallel was given, or a wrapper of it)."
        )
    if dp_mesh.ndim != 1:
        raise ValueError(
            "Unsloth: native FSDP2 over a tensor parallel model takes a 1-D dp mesh (mesh['dp'])."
        )
    dp_ranks, tp_ranks = set(dp_mesh.mesh.tolist()), set(tp_mesh.mesh.tolist())
    me = dist.get_rank()
    # A pipeline parallel model (apply_pipeline_parallel first): every stage is one (dp, tp) slice of the world.
    pp = _pp_size(model)
    if dp_ranks & tp_ranks != {me} or dp_mesh.size() * tp_mesh.size() * pp != dist.get_world_size():
        raise ValueError(
            f"Unsloth: native FSDP2's mesh (ranks {sorted(dp_ranks)}) must be the data parallel dimension of the "
            f"(dp, tp) mesh whose tp dimension (ranks {sorted(tp_ranks)}) apply_tensor_parallel used, covering "
            f"all {dist.get_world_size()} ranks: pass FSDP2Config(mesh = mesh['dp'])."
        )
    try:
        from torch.distributed.device_mesh import _mesh_resources
        same_root = _mesh_resources.get_root_mesh(dp_mesh) is _mesh_resources.get_root_mesh(tp_mesh)
    except Exception:
        same_root = True
    if not same_root:
        raise ValueError(
            "Unsloth: the dp and tp meshes must come from one init_device_mesh(..., (dp, tp), mesh_dim_names = "
            "('dp', 'tp')) (FSDP2 shards the tp DTensors over its dp dimension)."
        )
    plan = next(
        (m.__dict__["_unsloth_tp_plan"] for m in chain if "_unsloth_tp_plan" in m.__dict__), None
    )
    for region in plan.regions if plan is not None else ():
        for name, norm in region.norms:
            if any(p.requires_grad for p in norm.parameters()):
                # Their tp sum is a hook on the Parameter, and FSDP2 replaces the Parameter objects.
                raise ValueError(
                    f"Unsloth: trainable per-head norms ({region.path}.{name}) are not supported with tensor "
                    "parallel x FSDP2 data parallel; keep them frozen."
                )
    for level in chain:
        level.__dict__["_unsloth_tp_dp_mesh"] = dp_mesh
    return tp_mesh


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
    if any(_is_dtensor(p.grad) and p.grad.device_mesh.ndim > 1 for p in params):
        return _clip_over_world(params, max_norm, eps)
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
    _scale_grads_(params, total, max_norm, eps)
    return total


def _scale_grads_(params, total, max_norm, eps):
    """torch's clip: a tensor coefficient clamped at 1 and applied unconditionally (no host sync, no
    data-dependent branch). DTensor grads are scaled through their local shard."""
    with torch.no_grad():
        coef = torch.clamp(max_norm / (total + eps), max = 1.0)
        for p in params:
            g = p.grad
            local = g.to_local() if _is_dtensor(g) else g
            c = coef.to(device = local.device, dtype = torch.float32)
            if local.dtype in (torch.float16, torch.bfloat16):
                # A small coefficient rounds to 0 in fp16 (a device tensor is cast to the grad dtype).
                local.copy_(local.float().mul_(c))
            else:
                local.mul_(c.to(local.dtype))


def _clip_over_world(params, max_norm, eps):
    """tp_clip_grad_norm_ for gradients on a 2-D (dp, tp) mesh (shard_model over the dp mesh): every
    element's square is counted once, divided by the number of ranks holding a replica of it."""
    import math

    import torch.distributed as dist

    total = None
    world = dist.get_world_size()
    for p in params:
        g = _replicate_grad(p.grad)
        if g is not p.grad:
            p.grad = g
        if _is_dtensor(g):
            mesh = g.device_mesh
            if world % mesh.size():
                raise ValueError(
                    f"Unsloth: tp_clip_grad_norm_ got a gradient on {mesh.size()} of {world} ranks."
                )
            # Replicate dims hold copies; Shard and FSDP2's _StridedShard (not is_shard() on newer torch) split.
            # A gradient on the dp mesh only (a parameter outside the tp regions, e.g. modules_to_save) is the
            # same on every tp rank: world / mesh.size() more copies.
            copies = math.prod(
                mesh.size(i) for i, pl in enumerate(g.placements) if pl.is_replicate()
            ) * (world // mesh.size())
            sq = g.to_local().float().square().sum() / copies
        else:
            sq = g.float().square().sum() / world  # the same on every rank
        total = sq if total is None else total + sq
    dist.all_reduce(total)
    total = total.sqrt()
    _scale_grads_(params, total, max_norm, eps)
    return total


def _full(
    tensor,
    param = None,
    layout = None,
):
    """The whole tensor (collective over its mesh), fused / packed parts back in their original order."""
    if not _is_dtensor(tensor):
        return tensor
    full = tensor.full_tensor()
    if layout is None and param is not None:
        layout = getattr(param, "_unsloth_tp_parts", None)
        layout = None if layout is None else (*layout, tensor.device_mesh.size())
    if layout is not None:
        dim, parts, tp = layout
        full = _unpermute(full, dim, parts, tp)
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
    layouts = _param_layouts(model)
    with torch.no_grad():
        for key, value in state.items():
            if (
                adapters_only
                and "lora_" not in key
                and not getattr(value, "requires_grad", False)
                and id(value) not in embeddings
            ):
                continue
            out[key] = _full(value, value, layouts.get(id(value))).detach()
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
    layouts = _param_layouts(model)
    with torch.no_grad():
        for key, value in raw.items():
            if _is_dtensor(value):
                if id(value) in wanted:
                    state[key] = _full(value, value, layouts.get(id(value))).detach()
            else:
                state[key] = value.detach()
    return state


def save_tp_adapter(model, save_directory, **kwargs):
    """Write the standard 1-GPU PEFT adapter. Collective (tp mesh = whole world in v1); rank 0 writes."""
    import torch.distributed as dist

    mesh = dp_mesh = None
    for level in _tp_chain(model):
        mesh = mesh or level.__dict__.get("_unsloth_tp_mesh")
        dp_mesh = (
            dp_mesh
            or level.__dict__.get("_unsloth_tp_dp_mesh")
            or level.__dict__.get("_unsloth_tp_ddp_mesh")
        )
    # Under shard_model over the dp mesh every trainable DTensor spans the whole (dp, tp) world; DDP-style dp
    # replicas hold the same adapter, gathered over each tp group.
    ranks = (
        mesh.size() * (dp_mesh.size() if dp_mesh is not None else 1) if mesh is not None else None
    )
    if dist.is_initialized() and mesh is not None and ranks != dist.get_world_size():
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


# Sharded loading (v2): each rank reads only its slices of the sharded projections from the checkpoint, so the
# load-time peak is the replicated parts plus 1 / tp of the attention / MLP weights, never the whole model.

_PRELOADED = "_unsloth_tp_preloaded"
_TP_QUANTIZED_HINT = "Load 4-bit / 8-bit / quantized models in full and call apply_tensor_parallel, or pass load_in_4bit = False."
_UNSUPPORTED_LOAD_KWARGS = (
    "load_in_4bit",
    "load_in_8bit",
    "quantization_config",
    "tp_plan",
    "tp_size",
    "device_mesh",
    "distributed_config",
    "gguf_file",
    "variant",
    "from_tf",
    "from_flax",
)


def _preloaded(t):
    return None if t is None else getattr(t, _PRELOADED, None)


def _refuse_weight_dependent_lora_init(model):
    """PiSSA / OLoRA / CorDA / EVA / LoftQ build the factors (and residual) from the weight; on a sharded_loading
    slice that is this rank's slice only, so the factors come out shard-sized and differ per rank."""
    configs = {}
    for level in _tp_chain(model):
        configs.update(getattr(level, "peft_config", None) or {})
    bad = {
        name: cfg.init_lora_weights
        for name, cfg in configs.items()
        if getattr(cfg, "init_lora_weights", True) not in (True, False, "gaussian")
    }
    if bad:
        raise ValueError(
            f"Unsloth: init_lora_weights {bad} depends on the base weight, which sharded_loading holds only as "
            "this rank's slice; use init_lora_weights = True / 'gaussian', or load the full model (no "
            "sharded_loading) before get_peft_model and apply_tensor_parallel."
        )


def _check_preloaded(path, proj, weight, base, mesh):
    """A pre-sharded projection must carry exactly the layout apply_tensor_parallel derives now."""
    want = (proj.style, tuple(proj.parts), mesh.size())
    for what, t in (("weight", weight), ("bias", base.bias)):
        layout = _preloaded(t)
        if layout is None:
            continue
        if tuple(layout) != want:
            raise ValueError(
                f"Unsloth: '{path}.{what}' was sharded by sharded_loading as {layout}, but the tensor parallel "
                f"plan now gives {want}; pass the same plan and tp mesh to both."
            )
        ranks = getattr(t, "_unsloth_tp_preload_ranks", None)
        if ranks != mesh.mesh.tolist():
            raise ValueError(
                f"Unsloth: '{path}.{what}' was sharded by sharded_loading over ranks {ranks}, not this tp "
                f"mesh {mesh.mesh.tolist()}."
            )


def _wrap_preloaded(module, name, local, mesh, dim):
    """A sharded_loading slice -> the DTensor parameter v1 would have built from the full tensor."""
    DTensor, _, _, Shard = _dtensor_api()
    shape = torch.Size(local._unsloth_tp_full_shape)
    dt = DTensor.from_local(
        local.detach(),
        mesh,
        [Shard(dim)],
        run_check = False,
        shape = shape,
        stride = torch.empty(shape, device = "meta").stride(),
    )
    param = _as_param(dt, local)
    parts = getattr(local, "_unsloth_tp_parts", None)
    if parts is not None:
        param._unsloth_tp_parts = parts
        _tag_layout(module, name, parts[0], parts[1], mesh.size())
    module._parameters[name] = param
    return param


def _tag_layout(module, name, dim, parts, tp):
    # On the module too: FSDP2 replaces the Parameter objects (and their attributes) when it shards them.
    module.__dict__.setdefault("_unsloth_tp_layouts", {})[name] = (dim, tuple(parts), tp)


def _param_layouts(model):
    """{id(parameter): (dim, parts, tp)} for the permuted (fused / packed) tp parameters of `model`."""
    out = {}
    for module in model.modules():
        for name, layout in (module.__dict__.get("_unsloth_tp_layouts") or {}).items():
            param = module._parameters.get(name)
            if param is not None:
                out[id(param)] = layout
    return out


def _refuse_unapplied_forward(module, args):
    raise RuntimeError(
        "Unsloth: this model was loaded with tensor parallel sharded loading (each rank holds slices of the "
        "attention / MLP weights); call apply_tensor_parallel(model, tp_mesh) after get_peft_model and "
        "before running it."
    )


def _checkpoint_index(name, kwargs):
    """{tensor name: local safetensors path} of a checkpoint directory or Hub repo."""
    import json
    import os

    sub = kwargs.get("subfolder") or ""

    def keys_of(path):
        from safetensors import safe_open
        with safe_open(path, framework = "pt", device = "cpu") as f:
            return {key: path for key in f.keys()}

    if os.path.isdir(name):
        root = os.path.join(name, sub)
        index = os.path.join(root, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding = "utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            return {k: os.path.join(root, v) for k, v in weight_map.items()}
        single = os.path.join(root, "model.safetensors")
        if os.path.isfile(single):
            return keys_of(single)
        raise ValueError(
            f"Unsloth: tensor parallel sharded loading needs a safetensors checkpoint; none in {root}."
        )
    from huggingface_hub import hf_hub_download

    hub = dict(
        subfolder = sub or None,
        revision = kwargs.get("revision"),
        token = kwargs.get("token"),
        cache_dir = kwargs.get("cache_dir"),
        local_files_only = bool(kwargs.get("local_files_only", False)),
    )
    try:
        index = hf_hub_download(name, "model.safetensors.index.json", **hub)
    except Exception:
        index = None
    if index is None:
        return keys_of(hf_hub_download(name, "model.safetensors", **hub))
    with open(index, encoding = "utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    paths = {v: hf_hub_download(name, v, **hub) for v in sorted(set(weight_map.values()))}
    return {k: paths[v] for k, v in weight_map.items()}


class _SliceReader:
    """safetensors reads of whole tensors or of this rank's slices (only the slice's bytes are read)."""

    def __init__(self, index):
        self.index, self.handles = index, {}
        self.full_bytes = self.slice_bytes = 0

    def _open(self, key):
        path = self.index[key]
        if path not in self.handles:
            from safetensors import safe_open
            self.handles[path] = safe_open(path, framework = "pt", device = "cpu")
        return self.handles[path]

    def shape(self, key):
        return tuple(self._open(key).get_slice(key).get_shape())

    def full(self, key):
        t = self._open(key).get_tensor(key)
        self.full_bytes += t.numel() * t.element_size()
        return t

    def slice(self, key, dim, parts, rank, tp):
        view = self._open(key).get_slice(key)
        pieces, offset = [], 0
        for size in parts:
            step = size // tp
            start = offset + rank * step
            pieces.append(view[start : start + step] if dim == 0 else view[:, start : start + step])
            offset += size
        t = torch.cat(pieces, dim = dim) if len(pieces) > 1 else pieces[0]
        self.slice_bytes += t.numel() * t.element_size()
        return t

    def rows(self, key, start, stop):
        """Rows [start, stop) of dim 0 (an FSDP2 chunk); only those bytes are read."""
        t = self._open(key).get_slice(key)[start:stop]
        self.slice_bytes += t.numel() * t.element_size()
        return t


def _checkpoint_key(names, model, index):
    prefix = getattr(_unwrap(model), "base_model_prefix", "") or ""
    for name in names:
        candidates = [name]
        if prefix and name.startswith(prefix + "."):
            candidates.append(name[len(prefix) + 1 :])
        elif prefix:
            candidates.append(f"{prefix}.{name}")
        for key in candidates:
            if key in index:
                return key
    return None


def _checkpoint_aliases(
    model,
    index,
    key_mapping = None,
):
    """{model key: checkpoint key} for a checkpoint saved under another naming: transformers 5's conversion
    mapping renames (VLMs save `model.layers.*` for `model.language_model.layers.*`) and 4.x
    `_checkpoint_conversion_mapping`. A model key a WeightConverter builds (merged experts, fused or permuted
    weights) maps to None: it is not one checkpoint tensor, so it cannot be read sliced. `key_mapping` is the
    load's own (from_pretrained(key_mapping = ...), e.g. Unsloth's text-only VLM loads)."""
    legacy = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    renamings = converters = None
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
            rename_source_key,
        )

        try:
            transforms = get_model_conversion_mapping(model, key_mapping = key_mapping)
        except TypeError:
            transforms = get_model_conversion_mapping(model)
        renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
        converters = [t for t in transforms if isinstance(t, WeightConverter)]
    except Exception:
        # transformers 4.x: from_pretrained's key_mapping replaces _checkpoint_conversion_mapping.
        legacy = key_mapping or legacy
    if not legacy and not renamings and not converters:
        return {}
    meta = dict.fromkeys(model.state_dict().keys(), True)
    prefix = getattr(model, "base_model_prefix", None) or None
    out = {}
    for key in index:
        new = key
        for pattern, replacement in legacy.items():
            new, n = re.subn(pattern, replacement, new)
            if n:
                break
        converted = None
        if renamings is not None:
            try:
                new, converted = rename_source_key(new, renamings, converters, prefix, meta)
            except Exception:
                converted = None
        if converted is not None:
            out[new] = None
        elif new != key:
            out.setdefault(new, key)
    return out


def _buffer_skeleton(model):
    """The model rebuilt with meta parameters and real buffers (RoPE tables), as from_pretrained builds them."""
    from accelerate import init_empty_weights

    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    previous = torch.get_default_dtype()
    with torch.random.fork_rng(devices = []), init_empty_weights(include_buffers = False):
        torch.set_default_dtype(dtype)
        try:
            return type(model)(copy.deepcopy(model.config))
        finally:
            torch.set_default_dtype(previous)


def _parameter_slots(model):
    """{id(parameter): (parameter, [(module, name)], [qualified names])}: tied parameters are one slot."""
    slots = {}
    for module_name, module in model.named_modules(remove_duplicate = False):
        for pname, p in module._parameters.items():
            if p is not None:
                entry = slots.setdefault(id(p), (p, [], []))
                entry[1].append((module, pname))
                entry[2].append(f"{module_name}.{pname}" if module_name else pname)
    return slots


def _fill_from_checkpoint(
    model,
    name,
    kwargs,
    device,
    read_local,
    label = "tensor parallel",
    fallback = "full load, then apply_tensor_parallel",
):
    """Fill a meta model from its safetensors checkpoint. `read_local(old, reader, key)` returns this rank's
    Parameter for a sharded tensor (reading only its bytes) or None, and then the tensor is read whole.
    Meta buffers come from the checkpoint or a meta-parameter skeleton of the same class (RoPE tables).
    Returns (reader, sharded numel, whole numel)."""
    index = _checkpoint_index(name, kwargs)
    reader = _SliceReader(index)
    aliases = None
    sharded_numel = full_numel = 0

    def find(names):
        nonlocal aliases
        key = _checkpoint_key(names, model, index)
        if key is not None:
            return key
        if aliases is None:
            aliases = _checkpoint_aliases(model, index, (kwargs or {}).get("key_mapping"))
        for n in names:
            if n in aliases:
                if aliases[n] is None:
                    raise ValueError(
                        f"Unsloth: {label} sharded loading cannot read '{n}': transformers builds it from several "
                        f"checkpoint tensors (a weight conversion); load this model without sharded loading "
                        f"({fallback})."
                    )
                return aliases[n]
        return None

    for old, where, names in _parameter_slots(model).values():
        key = find(names)
        if key is None:
            raise ValueError(
                f"Unsloth: {label} sharded loading found no checkpoint tensor for '{names[0]}'; load "
                f"this model without sharded loading ({fallback})."
            )
        if reader.shape(key) != tuple(old.shape):
            raise ValueError(
                f"Unsloth: checkpoint tensor '{key}' has shape {reader.shape(key)}, the model expects "
                f"{tuple(old.shape)}."
            )
        new = read_local(old, reader, key)
        if new is None:
            value = reader.full(key).to(device = device, dtype = old.dtype)
            new = nn.Parameter(value, requires_grad = old.requires_grad)
            full_numel += old.numel()
        else:
            sharded_numel += old.numel()
        for module, pname in where:
            module._parameters[pname] = new
    skeleton = None
    for module_name, module in model.named_modules():
        for bname, buf in list(module._buffers.items()):
            if buf is None or not buf.is_meta:
                continue
            key = f"{module_name}.{bname}" if module_name else bname
            source = find([key]) if bname not in module._non_persistent_buffers_set else None
            if source is not None:
                value = reader.full(source)
            else:
                skeleton = skeleton if skeleton is not None else _buffer_skeleton(model)
                value = skeleton.get_buffer(key)
                if value.is_meta:
                    raise ValueError(f"Unsloth: sharded loading cannot rebuild the buffer '{key}'.")
            module._buffers[bname] = value.to(device = device, dtype = buf.dtype)
    if isinstance(getattr(model, "hf_device_map", None), dict):
        model.hf_device_map = {"": device.index if device.type == "cuda" else device.type}
    return reader, sharded_numel, full_numel


def _materialize_sharded(model, name, kwargs, mesh, plan, device):
    """Fill a meta model from the checkpoint: replicated tensors whole, sharded projections as local slices."""
    tp, rank = mesh.size(), mesh.get_local_rank()
    tp_plan = derive_tp_plan(model, tp, plan)
    shards = {}
    for region in tp_plan.regions:
        for proj in region.projs:
            weight, base = _weight(proj.module)
            col = proj.style == COLWISE
            shards[id(weight)] = (proj, 0 if col else 1, proj.parts if col else (weight.shape[1],))
            if base.bias is not None and col:
                shards[id(base.bias)] = (proj, 0, proj.parts)

    def read_local(old, reader, key):
        shard = shards.get(id(old))
        if shard is None:
            return None
        proj, dim, parts = shard
        local = reader.slice(key, dim, parts, rank, tp).to(device = device, dtype = old.dtype)
        # A plain local tensor tagged with its full shape: nn.Linear keeps the full in / out features, so
        # get_peft_model builds full-size LoRA factors with the 1-GPU init (a DTensor would make PEFT size
        # them from the shard). apply_tensor_parallel wraps it into the DTensor.
        new = nn.Parameter(local.contiguous(), requires_grad = old.requires_grad)
        if len(parts) > 1:
            new._unsloth_tp_parts = (dim, tuple(parts))
        new._unsloth_tp_full_shape = tuple(old.shape)
        new._unsloth_tp_preload_ranks = mesh.mesh.tolist()
        setattr(new, _PRELOADED, (proj.style, tuple(proj.parts), tp))
        return new

    reader, sharded_numel, full_numel = _fill_from_checkpoint(
        model, name, kwargs, device, read_local
    )
    for region in tp_plan.regions:
        for proj in region.projs:
            base = _weight(proj.module)[1]
            base.__dict__["_unsloth_tp_load_guard"] = base.register_forward_pre_hook(
                _refuse_unapplied_forward
            )
    stats = {
        "tp": tp,
        "rank": rank,
        "sharded_numel": sharded_numel,
        "replicated_numel": full_numel,
        "read_bytes_full": reader.full_bytes,
        "read_bytes_sliced": reader.slice_bytes,
        "regions": len(tp_plan.regions),
        "source": tp_plan.source,
    }
    model.__dict__["_unsloth_tp_load_stats"] = stats
    return stats


def _load_device(device_map, mesh):
    if mesh.device_type != "cuda":
        return torch.device(mesh.device_type)
    if isinstance(device_map, dict) and len(device_map) == 1:
        (value,) = device_map.values()
        if isinstance(value, int) or (isinstance(value, str) and value.startswith("cuda")):
            return torch.device(value if isinstance(value, str) else f"cuda:{value}")
    if isinstance(device_map, (int, torch.device)) or (
        isinstance(device_map, str) and device_map.startswith("cuda")
    ):
        return torch.device(f"cuda:{device_map}" if isinstance(device_map, int) else device_map)
    return torch.device("cuda", torch.cuda.current_device())


def _from_pretrained_sharded(original, cls, name, args, kwargs, state):
    if state["active"]:
        return original(cls, name, *args, **kwargs)  # a nested load inside from_pretrained
    config = kwargs.get("config")
    label = state.get("label", "tensor parallel")
    for key in _UNSUPPORTED_LOAD_KWARGS:
        if kwargs.get(key) or (key == "quantization_config" and getattr(config, key, None)):
            raise ValueError(
                f"Unsloth: {label} sharded loading supports 16-bit safetensors checkpoints only "
                f"(got {key}). {state.get('quantized_hint', _TP_QUANTIZED_HINT)}"
            )
    mesh = state["mesh"] if state["mesh"] is not None else _world_mesh()
    if mesh.ndim != 1:
        raise ValueError(
            state.get(
                "mesh_hint", "Unsloth: sharded_loading takes a 1-D tp mesh (pass mesh['tp'])."
            )
        )
    device = _load_device(kwargs.get("device_map"), mesh)
    kwargs = dict(kwargs, device_map = "meta")
    state["active"] = True
    try:
        with torch.random.fork_rng(devices = []):
            model = original(cls, name, *args, **kwargs)
    finally:
        state["active"] = False
    text = _text_config(model)
    if getattr(text, "quantization_config", None) or getattr(
        model.config, "quantization_config", None
    ):
        raise ValueError(
            f"Unsloth: {label} sharded loading supports 16-bit checkpoints only; this checkpoint is "
            f"pre-quantized. {state.get('prequantized_hint', 'Load it in full and call apply_tensor_parallel.')}"
        )
    materialize = state.get("materialize", _materialize_sharded)
    materialize(model, name, kwargs, mesh, state["plan"], device)
    state["models"].append(model)
    return model


def _world_mesh():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        raise RuntimeError(
            "Unsloth: sharded loading needs an initialized process group (torchrun) or a mesh."
        )
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(device_type, (dist.get_world_size(),), mesh_dim_names = ("tp",))


@contextlib.contextmanager
def sharded_loading(tp_mesh = None, plan = None):
    """Inside this context every `from_pretrained` of a transformers model (FastLanguageModel's included)
    builds the model on meta, then each rank reads from the safetensors checkpoint the replicated tensors
    whole and only its own slices of the attention / MLP projections (DTensors with the full logical shape,
    so get_peft_model sizes and seeds LoRA exactly as on one GPU). 16-bit checkpoints only.

        with sharded_loading(mesh["tp"]):
            model, tokenizer = FastLanguageModel.from_pretrained(name, load_in_4bit = False, ...)
        model = FastLanguageModel.get_peft_model(model, ...)
        apply_tensor_parallel(model, mesh["tp"])  # required before the first forward
    """
    with _sharded_from_pretrained({"mesh": tp_mesh, "plan": plan}) as state:
        yield state


@contextlib.contextmanager
def _sharded_from_pretrained(state):
    """Swap transformers' PreTrainedModel.from_pretrained for the meta build + checkpoint fill of `state`
    (mesh, plan, materialize(model, name, kwargs, mesh, plan, device), message parts). Shared by tensor
    parallel and native FSDP2 sharded loading."""
    from transformers import PreTrainedModel

    descriptor = PreTrainedModel.__dict__["from_pretrained"]
    original = descriptor.__func__
    state = dict(state, models = [], active = False)

    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _from_pretrained_sharded(
            original, cls, pretrained_model_name_or_path, args, kwargs, state
        )

    PreTrainedModel.from_pretrained = classmethod(from_pretrained)
    try:
        yield state
    finally:
        PreTrainedModel.from_pretrained = descriptor


def load_tensor_parallel(
    model_name,
    tp_mesh = None,
    loader = None,
    plan = None,
    **kwargs,
):
    """`loader(model_name, **kwargs)` under sharded_loading (default FastLanguageModel.from_pretrained, 16-bit).
    Returns what the loader returns; raises if no model went through the sharded path."""
    if loader is None:
        from unsloth import FastLanguageModel
        loader = FastLanguageModel.from_pretrained
        kwargs.setdefault("load_in_4bit", False)
    with sharded_loading(tp_mesh, plan) as state:
        out = loader(model_name, **kwargs)
    if not state["models"]:
        raise RuntimeError(
            "Unsloth: load_tensor_parallel's loader did not call transformers from_pretrained; nothing was "
            "loaded sharded."
        )
    return out


# Trainer (v2): the HF / TRL Trainer trains a tensor parallel model. accelerate must not wrap the tp ranks in
# DDP, every tp rank of a dp replica must read the same batches (the dataloader is sharded over dp only), and
# gradients are clipped with tp_clip_grad_norm_. Data parallel comes from shard_model over mesh["dp"] (native
# FSDP2, which then wraps the model itself). The loss needs no change: the Trainer's token count gathered over
# the world counts each dp replica's tokens tp times and its loss scale multiplies by dp * tp, so per rank the
# loss is (sum over its tokens) * dp / (dp replicas' tokens), which FSDP2's mean over dp (or nothing, dp = 1)
# turns into the token mean over the dp replicas, as for DDP on dp GPUs.

_TP_DATA = "_unsloth_tp_data"
_DDP_MESH = "_unsloth_tp_ddp_mesh"
_DP_SYNC = "_unsloth_tp_dp_sync"
_UNSUPPORTED_OPTIMIZERS = (
    "8bit",
    "bnb",
    "paged",
    "lomo",
    "galore",
    "apollo",
    "torchao",
    "q_galore",
)


def _tp_meshes(model):
    chain = _tp_chain(model)
    tp = next(
        (m.__dict__["_unsloth_tp_mesh"] for m in chain if "_unsloth_tp_mesh" in m.__dict__), None
    )
    dp = next(
        (m.__dict__["_unsloth_tp_dp_mesh"] for m in chain if "_unsloth_tp_dp_mesh" in m.__dict__),
        None,
    )
    if (
        dp is None
    ):  # UNSLOTH_TP_SIZE with DDP-style data parallel (tp_env): gradients averaged over dp
        dp = next(
            (m.__dict__[_DDP_MESH] for m in chain if m.__dict__.get(_DDP_MESH) is not None), None
        )
    return tp, dp


def _ddp_mesh(model):
    return next(
        (m.__dict__[_DDP_MESH] for m in _tp_chain(model) if m.__dict__.get(_DDP_MESH) is not None),
        None,
    )


def _env_hint():
    from .tp_env import env_hint
    return env_hint()


def _strategy(value):
    value = getattr(value, "value", value)
    return str(value or "no").lower()


def _optim_name(args):
    optim = getattr(args, "optim", "")
    return str(getattr(optim, "value", optim) or "").strip().lower()


def _keeps_bnb(name, cpu_shards):
    from .fsdp2_native import _BNB_LOCAL_SHARD_OPTIMS, _bnb_local_shards_available
    return name in _BNB_LOCAL_SHARD_OPTIMS and _bnb_local_shards_available(cpu_shards)


def _tp_bnb_optim(args, cpu_shards):
    """bitsandbytes Adam / AdamW (the notebooks' adamw_8bit) step each DTensor's local shard
    (fsdp2_bnb_optim, as under native FSDP2); the other bitsandbytes optims with a torch equivalent switch to it."""
    from .fsdp2_native import _BNB_OPTIMS

    name = _optim_name(args)
    if _keeps_bnb(name, cpu_shards):
        return False
    target = _BNB_OPTIMS.get(name)
    if target is None:
        return False
    if target == "adamw_torch_fused" and (cpu_shards or not torch.cuda.is_available()):
        target = "adamw_torch"
    _print_rank0(
        f"Unsloth: tensor parallel parameters are DTensors, which this bitsandbytes optimizer cannot step "
        f"(UNSLOTH_FSDP2_BNB_LOCAL=0 or no local-shard support). Switching optim = '{name}' to '{target}'."
    )
    try:
        from transformers.training_args import OptimizerNames
        args.optim = OptimizerNames(target)
    except Exception:
        args.optim = target
    return True


def _print_rank0(message):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def _refuse_tp_trainer_args(args, cpu_shards = False):
    problems = []
    if getattr(args, "deepspeed", None):
        problems.append("DeepSpeed")
    if getattr(args, "fsdp", None) or getattr(args, "parallelism_config", None) is not None:
        problems.append(
            "accelerate FSDP / ParallelismConfig (use shard_model over mesh['dp'] for data parallel)"
        )
    if (
        _strategy(getattr(args, "eval_strategy", getattr(args, "evaluation_strategy", "no")))
        != "no"
    ):
        problems.append("evaluation during training (set eval_strategy = 'no')")
    for flag in ("load_best_model_at_end", "push_to_hub"):
        if getattr(args, flag, False):
            problems.append(flag)
    optim = _optim_name(args)
    if any(key in optim for key in _UNSUPPORTED_OPTIMIZERS) and not _keeps_bnb(optim, cpu_shards):
        problems.append(f"optim = '{optim}' (use 'adamw_torch' or 'adamw_8bit')")
    config = getattr(args, "accelerator_config", None)
    if getattr(config, "dispatch_batches", None):
        problems.append("accelerator_config.dispatch_batches")
    if problems:
        raise RuntimeError(
            "Unsloth: the Trainer on a tensor parallel model does not support "
            + "; ".join(problems)
            + "."
            + _env_hint()
        )


def _tp_fix_train_sampler(sampler, args, dp_size, dp_rank):
    """Make a Trainer train sampler identical on the tp ranks of one data parallel replica."""
    from transformers import trainer_pt_utils

    rebalance = getattr(trainer_pt_utils, "BatchRebalanceSampler", None)
    if rebalance is not None and type(sampler) is rebalance:
        # Built from the global world size / rank, so tp peers would draw different rows: rebuild over dp.
        grad_accum = sampler.grad_accum
        return rebalance(
            lengths = sampler.lengths,
            effective_batch_size = args.train_batch_size * grad_accum * dp_size,
            dp_size = dp_size,
            grad_accum = grad_accum,
            shuffle = sampler.shuffle,
            seed = sampler.seed,
            rank = dp_rank,
            drop_last = sampler.drop_last,
        )
    if (
        type(sampler) is trainer_pt_utils.LengthGroupedSampler
        and getattr(sampler, "generator", None) is None
    ):
        # Global torch RNG desyncs across tp peers; own generator seeded seed + epoch keeps resume exact.
        sampler.__class__ = _tp_length_grouped_sampler_class(type(sampler))
        sampler.seed = args.data_seed if getattr(args, "data_seed", None) is not None else args.seed
        sampler.epoch = 0
        sampler.generator = torch.Generator().manual_seed(sampler.seed)
    return sampler


_TP_LENGTH_GROUPED = {}


def _tp_length_grouped_sampler_class(base):
    cls = _TP_LENGTH_GROUPED.get(base)
    if cls is None:

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            self.generator = torch.Generator().manual_seed(self.seed + self.epoch)
            return base.__iter__(self)

        cls = type(
            "UnslothTensorParallelLengthGroupedSampler",
            (base,),
            {"set_epoch": set_epoch, "__iter__": __iter__, "_unsloth_tp_epoch_seeded": True},
        )
        _TP_LENGTH_GROUPED[base] = cls
    return cls


def _tp_forward_set_epoch(data_loader):
    """BatchSamplerShard has no set_epoch, so on resume the inner sampler would replay epoch 0: forward it."""
    shard = getattr(data_loader, "batch_sampler", None)
    inner = getattr(shard, "batch_sampler", None)
    if shard is None or inner is None or hasattr(shard, "set_epoch"):
        return
    from transformers import trainer_pt_utils

    rebalance = getattr(trainer_pt_utils, "BatchRebalanceSampler", None)
    targets = [
        x
        for x in (inner, getattr(inner, "sampler", None))
        if getattr(type(x), "_unsloth_tp_epoch_seeded", False)
        or (rebalance is not None and type(x) is rebalance)
    ]
    if not targets:
        return

    def set_epoch(epoch):
        for target in targets:
            target.set_epoch(epoch)

    shard.set_epoch = set_epoch


def _tp_train_sampler(trainer, dp_size, dp_rank):
    get_sampler = getattr(trainer, "_get_train_sampler", None)
    if get_sampler is None:
        return

    @functools.wraps(get_sampler)
    def _get_train_sampler(*args, **kwargs):
        return _tp_fix_train_sampler(get_sampler(*args, **kwargs), trainer.args, dp_size, dp_rank)

    trainer.__dict__["_get_train_sampler"] = _get_train_sampler


def tp_trainer_setup(
    model = None,
    phase = "pre",
    trainer = None,
):
    """Called around Trainer.__init__ (models/_utils.py) for a tensor parallel model. Returns True if it is one."""
    import torch.distributed as dist

    if phase == "pre":
        if model is None or not is_tensor_parallel(model):
            return False
        tp_mesh, dp_mesh = _tp_meshes(model)
        if tp_mesh is None:
            raise RuntimeError(
                "Unsloth: pass the Trainer the outermost model (the one apply_tensor_parallel was given, or a "
                "wrapper of it)."
            )
        ranks = tp_mesh.size() * (dp_mesh.size() if dp_mesh is not None else 1)
        if dist.is_initialized() and ranks != dist.get_world_size():
            raise RuntimeError(
                f"Unsloth: the Trainer on a tensor parallel model needs every rank in the tp mesh, or in the (dp, "
                f"tp) mesh after shard_model(model, FSDP2Config(mesh = mesh['dp'])); the tp mesh covers "
                f"{ranks} of {dist.get_world_size()} ranks."
            )
        _install_tp_accelerate_hooks()
        return True
    model = trainer.model
    tp_mesh, dp_mesh = _tp_meshes(model)
    cpu_shards = tp_mesh.device_type == "cpu"
    _tp_bnb_optim(trainer.args, cpu_shards)
    _refuse_tp_trainer_args(trainer.args, cpu_shards)
    ddp_mesh = _ddp_mesh(model)
    dp_size, dp_rank = (dp_mesh.size(), dp_mesh.get_local_rank()) if dp_mesh is not None else (1, 0)
    trainer.accelerator.__dict__[_TP_DATA] = (dp_size, dp_rank)
    _tp_train_sampler(trainer, dp_size, dp_rank)
    trainer.__dict__["_unsloth_tensor_parallel"] = (
        True  # the generated TRL trainers skip for_inference()
    )
    _tp_scaler(trainer, dp_size)
    trainer.__dict__["save_model"] = types.MethodType(_tp_trainer_save_model, trainer)
    for name in ("evaluate", "predict"):
        trainer.__dict__[name] = _refuse_tp_trainer_call(name)
    # Trainer checkpoints: per-rank local shards (instance level: before native FSDP2's class hooks, whose
    # full-state gathers do not restore the packed tp part order).
    for name, method in (
        ("_save_optimizer_and_scheduler", _tp_save_optimizer_and_scheduler),
        ("_load_optimizer_and_scheduler", _tp_load_optimizer_and_scheduler),
        ("_load_from_checkpoint", _tp_load_from_checkpoint),
    ):
        trainer.__dict__[name] = types.MethodType(method, trainer)
    sync = None
    if ddp_mesh is not None:
        # DDP-style dp (UNSLOTH_TP_SIZE without UNSLOTH_FSDP2): torch DDP cannot wrap DTensor parameters, so
        # the gradients are averaged over the dp mesh before clipping (Accelerator.clip_grad_norm_ below) or,
        # without clipping, at the optimizer step.
        sync = {"mesh": ddp_mesh, "synced": False}
        trainer.accelerator.__dict__[_DP_SYNC] = sync
    create_optimizer = trainer.create_optimizer

    def create_optimizer_tp(*args, **kwargs):
        # A trainable outside the tp regions (modules_to_save) is a plain tensor next to the LoRA DTensors:
        # foreach / fused kernels take one kind per group. Split before create_scheduler records the groups.
        optimizer = create_optimizer(*args, **kwargs)
        from .fsdp2_native import _inner_optimizer, _mixed_param_groups, _split_mixed_param_groups

        inner = _inner_optimizer(optimizer)
        if getattr(trainer, "lr_scheduler", None) is not None and _mixed_param_groups(inner):
            # A prebuilt scheduler keeps one base LR / lambda per original group: split groups would go unscheduled.
            raise ValueError(
                "Unsloth: this tensor parallel model mixes DTensor (tensor parallel) and plain trainable parameters "
                "in one optimizer group, which foreach / fused optimizers cannot step, and Trainer(optimizers = "
                "(optimizer, scheduler)) already built the scheduler over those groups. Build the optimizer over "
                "`unsloth.models.fsdp2_native.split_param_groups(...)` before the scheduler, or pass only the optimizer."
            )
        _split_mixed_param_groups(inner)
        _tp_localize_bnb(optimizer)
        if sync is not None and not getattr(inner, "_unsloth_tp_dp_hooked", False):
            inner.register_step_pre_hook(functools.partial(_dp_sync_before_step, sync))
            inner._unsloth_tp_dp_hooked = True
        return optimizer

    trainer.__dict__["create_optimizer"] = create_optimizer_tp
    # transformers 5.x clips mixed DTensor / plain gradients (modules_to_save next to the LoRA DTensors) in its
    # own _mixed_mesh_grad_norm, which sums every DTensor's local square over its mesh and skips the dp
    # average: keep them on Accelerator.clip_grad_norm_ (tp_clip_grad_norm_ counts each element once).
    trainer.__dict__["_has_mixed_mesh_grads"] = lambda model: False
    return True


def _tp_localize_bnb(optimizer):
    """A bitsandbytes Adam / AdamW steps the local shards of the tp DTensors (fsdp2_bnb_optim)."""
    from .fsdp2_native import _inner_optimizer, _is_bnb_optimizer_class, _localize_bnb_optimizer

    inner = _inner_optimizer(optimizer)
    if _localize_bnb_optimizer(optimizer) or not _is_bnb_optimizer_class(type(inner)):
        return
    raise RuntimeError(
        f"Unsloth: a tensor parallel model's parameters are DTensors, which this bitsandbytes optimizer "
        f"({type(inner).__name__}) cannot step; only bitsandbytes Adam / AdamW step local shards. Use optim = "
        "'adamw_8bit' or 'adamw_torch'." + _env_hint()
    )


def _dp_average(params, mesh):
    """Average the gradients over the dp mesh (DDP's all-reduce, one per dtype / device bucket). A trainable
    with a gradient on some dp replicas only (an unrouted expert, an unused tower) gets zeros on the others
    first, as DDP does, so every replica reduces the same buckets; one with no gradient anywhere stays None."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    params = [p for p in params if p.requires_grad]
    if not params:
        return
    group, n = mesh.get_group(), mesh.size()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if mesh.device_type == "cuda"
        else torch.device(mesh.device_type)
    )
    has = torch.tensor([p.grad is not None for p in params], dtype = torch.int32, device = device)
    dist.all_reduce(has, op = dist.ReduceOp.MAX, group = group)
    buckets = {}
    for p, any_grad in zip(params, has.tolist()):
        if not any_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        g = p.grad
        local = g._local_tensor if _is_dtensor(g) else g
        buckets.setdefault((local.device, local.dtype), []).append(local)
    with torch.no_grad():
        for tensors in buckets.values():
            flat = _flatten_dense_tensors(tensors)
            if flat.dtype == torch.float16:
                # fp16 sum can overflow past 65504 though the mean is finite.
                flat = flat.float()
            dist.all_reduce(flat, group = group)
            flat.div_(n)
            for t, value in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
                t.copy_(value)


def _dp_sync_before_step(sync, optimizer, args, kwargs):
    if sync["synced"]:
        sync["synced"] = False  # clip_grad_norm_ already averaged this step's gradients
        return None
    _dp_average([p for group in optimizer.param_groups for p in group["params"]], sync["mesh"])
    return None


# Trainer checkpoints under tensor parallelism: every rank writes its own local shards of the trainable
# weights and of the optimizer state (DTensor entries as local tensors), the scheduler / scaler / RNG / trainer
# state stay as the Trainer writes them, and save_model writes the 1-GPU adapter. A resume needs the same layout.
_CKPT_DIR = "unsloth_tp"


def _ckpt_path(directory, rank):
    import os
    return os.path.join(directory, _CKPT_DIR, f"rank{rank:05d}.pt")


def _tp_layout(trainer):
    import torch.distributed as dist

    tp_mesh, dp_mesh = _tp_meshes(trainer.model)
    params = [
        (name, tuple((p._local_tensor if _is_dtensor(p) else p).shape))
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    ]
    return {
        "world": dist.get_world_size() if dist.is_initialized() else 1,
        "tp": tp_mesh.size() if tp_mesh is not None else 1,
        "dp": dp_mesh.size() if dp_mesh is not None else 1,
        "dp_kind": None
        if dp_mesh is None
        else ("ddp" if _ddp_mesh(trainer.model) is not None else "fsdp2"),
        "params": params,
    }


def _optimizer_local_state(optimizer):
    from .fsdp2_native import _inner_optimizer

    state_dict = _inner_optimizer(optimizer).state_dict()
    dtensors, state = [], {}
    for index, entry in state_dict["state"].items():
        out = {}
        for key, value in entry.items():
            if _is_dtensor(value):
                out[key] = value.to_local().detach()
                dtensors.append((index, key))
            else:
                out[key] = value.detach() if torch.is_tensor(value) else value
        state[index] = out
    return {"state": state, "param_groups": state_dict["param_groups"], "dtensor": dtensors}


def _load_optimizer_local_state(optimizer, saved):
    from .fsdp2_native import _inner_optimizer

    DTensor = _dtensor_api()[0]
    inner = _inner_optimizer(optimizer)
    params = [p for group in inner.param_groups for p in group["params"]]
    saved_ids = [i for group in saved["param_groups"] for i in group["params"]]
    if len(params) != len(saved_ids):
        raise RuntimeError(
            f"Unsloth: tensor parallel resume: the checkpoint's optimizer has {len(saved_ids)} parameters, this "
            f"one {len(params)}."
        )
    param_of = dict(zip(saved_ids, params))
    wrap = {tuple(x) for x in saved["dtensor"]}
    state = {}
    for index, entry in saved["state"].items():
        p = param_of[index]
        out = {}
        for key, value in entry.items():
            if (index, key) in wrap:
                local = p._local_tensor
                value = DTensor.from_local(
                    value.to(local.device),
                    p.device_mesh,
                    p.placements,
                    run_check = False,
                    shape = p.shape,
                    stride = p.stride(),
                )
            out[key] = value
        state[index] = out
    inner.load_state_dict({"state": state, "param_groups": saved["param_groups"]})


def _tp_save_optimizer_and_scheduler(self, output_dir):
    import os

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    path = _ckpt_path(output_dir, rank)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    trainables = {
        name: (p._local_tensor if _is_dtensor(p) else p).detach()
        for name, p in self.model.named_parameters()
        if p.requires_grad
    }
    payload = {
        "layout": _tp_layout(self),
        "trainables": trainables,
        "optimizer": _optimizer_local_state(self.optimizer)
        if getattr(self, "optimizer", None) is not None
        else None,
    }
    torch.save(payload, path)
    scheduler = getattr(self, "lr_scheduler", None)
    if self.args.should_save and scheduler is not None:
        from transformers.trainer import SCHEDULER_NAME
        torch.save(scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
    if dist.is_initialized():
        dist.barrier()  # every rank's file is on disk before the Trainer state marks the checkpoint complete


def _tp_resume_payload(self, checkpoint):
    import os

    import torch.distributed as dist

    cached = self.__dict__.get("_unsloth_tp_resume")
    if cached is not None and cached[0] == checkpoint:
        return cached[1]
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = _ckpt_path(checkpoint, rank)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Unsloth: tensor parallel resume needs the per-rank state {path}, which this checkpoint does not "
            "have (it was not written by a tensor parallel Trainer, or with save_only_model = True, which saves "
            "no optimizer state to resume from)." + _env_hint()
        )
    payload = torch.load(path, map_location = "cpu", weights_only = True)
    want, got = _tp_layout(self), payload["layout"]
    if want != got:
        diff = {
            k: (got.get(k), want.get(k))
            for k in ("world", "tp", "dp", "dp_kind")
            if got.get(k) != want.get(k)
        }
        raise RuntimeError(
            f"Unsloth: tensor parallel resume from {checkpoint}: the checkpoint was written on another layout "
            f"({diff or 'trainable parameter names / shard shapes differ'}); resume with the same world size, "
            "UNSLOTH_TP_SIZE and data parallel mode."
        )
    self.__dict__["_unsloth_tp_resume"] = (checkpoint, payload)
    return payload


def _tp_load_from_checkpoint(
    self,
    resume_from_checkpoint,
    model = None,
):
    payload = _tp_resume_payload(self, resume_from_checkpoint)
    target = model if model is not None else self.model
    with torch.no_grad():
        for name, p in target.named_parameters():
            if p.requires_grad:
                local = p._local_tensor if _is_dtensor(p) else p
                local.copy_(payload["trainables"][name].to(local.device))
    _print_rank0(
        f"Unsloth: tensor parallel resume: every rank restored its local shards from {resume_from_checkpoint}."
    )


def _tp_load_optimizer_and_scheduler(self, checkpoint, *args, **kwargs):
    import os

    if checkpoint is None:
        return None
    payload = _tp_resume_payload(self, checkpoint)
    if payload.get("optimizer") is not None:
        _load_optimizer_local_state(self.optimizer, payload["optimizer"])
    from transformers.trainer import SCHEDULER_NAME

    path = os.path.join(checkpoint, SCHEDULER_NAME)
    if getattr(self, "lr_scheduler", None) is not None and os.path.isfile(path):
        self.lr_scheduler.load_state_dict(torch.load(path, weights_only = True))
    self.__dict__.pop("_unsloth_tp_resume", None)
    return None


def _refuse_tp_trainer_call(name):
    def refuse(*args, **kwargs):
        raise RuntimeError(
            f"Unsloth: Trainer.{name} is not supported on a tensor parallel model (its gather would count every "
            "tp rank); run the forward in your own loop, or save the adapter and evaluate on one GPU."
            + _env_hint()
        )

    return refuse


def _tp_trainer_save_model(
    self,
    output_dir = None,
    _internal_call = False,
):
    """Trainer.save_model for a tensor parallel model: the 1-GPU PEFT adapter (collective, rank 0 writes)."""
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    save_tp_adapter(self.model, output_dir)
    processing = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
    if processing is not None and self.args.should_save:
        processing.save_pretrained(output_dir)


def _tp_scaler(trainer, dp):
    """float16: one GradScaler inf decision for all ranks (each tp rank sees other gradient shards)."""
    from .fsdp2_native import _scaler_for_native

    accelerator = trainer.accelerator
    if not _scaler_for_native(trainer):
        return
    handler = getattr(accelerator, "scaler_handler", None)
    user = handler.to_kwargs() if handler is not None and hasattr(handler, "to_kwargs") else {}
    if "init_scale" not in user:
        # The loss scale grows with dp (the Trainer's token-count scaling), never with tp.
        accelerator.scaler._init_scale = 2.0**16 / dp


def _install_tp_accelerate_hooks():
    from accelerate import Accelerator

    if getattr(Accelerator, "_unsloth_tp_hooks", False):
        return
    original_prepare_model = Accelerator.prepare_model
    original_data_loader = Accelerator.prepare_data_loader
    original_clip = Accelerator.clip_grad_norm_

    def _tp_model(self, model):
        if self.__dict__.get(_TP_DATA) is None or not is_tensor_parallel(model):
            return False
        try:
            from .fsdp2_native import is_native_fsdp2
        except Exception:
            return True
        return not is_native_fsdp2(model)  # native FSDP2 (dp > 1) prepares it itself

    def prepare_model(
        self,
        model,
        device_placement = None,
        evaluation_mode = False,
    ):
        if not _tp_model(self, model):
            return original_prepare_model(
                self, model, device_placement = device_placement, evaluation_mode = evaluation_mode
            )
        # No DDP over the tp ranks and no device move; evaluation_mode keeps only the autocast wrap.
        if getattr(model, "_is_accelerate_prepared", False):
            if model not in self._models:
                self._models.append(model)
            return model
        return original_prepare_model(self, model, device_placement = False, evaluation_mode = True)

    def prepare_data_loader(self, data_loader, *args, **kwargs):
        layout = self.__dict__.get(_TP_DATA)
        if layout is None:
            return original_data_loader(self, data_loader, *args, **kwargs)
        if isinstance(getattr(data_loader, "dataset", None), torch.utils.data.IterableDataset):
            raise RuntimeError(
                "Unsloth: the Trainer on a tensor parallel model needs a map-style dataset (accelerate "
                "dispatches iterable datasets over every rank)." + _env_hint()
            )
        dp_size, dp_rank = layout
        # The Trainer seeds workers with partial(seed_worker, rank = process_index): tp peers must draw the
        # same random transforms / collation, so seed them by dp rank too.
        init_fn = getattr(data_loader, "worker_init_fn", None)
        if isinstance(init_fn, functools.partial) and "rank" in (init_fn.keywords or {}):
            data_loader.worker_init_fn = functools.partial(
                init_fn.func, *init_fn.args, **{**init_fn.keywords, "rank": dp_rank}
            )
        cls = type(self)
        # accelerate shards the batches by num_processes / process_index: give it the dp layout, so the tp
        # ranks of a replica read the same batches.
        self.__class__ = type(
            "UnslothTensorParallelDataAccelerator",
            (cls,),
            {
                "num_processes": property(lambda s: dp_size),
                "process_index": property(lambda s: dp_rank),
            },
        )
        try:
            prepared = original_data_loader(self, data_loader, *args, **kwargs)
        finally:
            self.__class__ = cls
        _tp_forward_set_epoch(prepared)
        return prepared

    def clip_grad_norm_(
        self,
        parameters,
        max_norm,
        norm_type = 2,
    ):
        if self.__dict__.get(_TP_DATA) is None:
            return original_clip(self, parameters, max_norm, norm_type = norm_type)
        if float(norm_type) != 2.0:
            raise RuntimeError(
                "Unsloth: tensor parallel gradient clipping supports the 2-norm only."
            )
        self.unscale_gradients()
        parameters = list(parameters)
        sync = self.__dict__.get(_DP_SYNC)
        if sync is not None:
            _dp_average(parameters, sync["mesh"])
            sync["synced"] = True
        return tp_clip_grad_norm_(parameters, max_norm)

    Accelerator.prepare_model = prepare_model
    Accelerator.prepare_data_loader = prepare_data_loader
    Accelerator.clip_grad_norm_ = clip_grad_norm_
    Accelerator._unsloth_tp_hooks = True
