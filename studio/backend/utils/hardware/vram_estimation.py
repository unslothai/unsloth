# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Training VRAM estimation.

Total VRAM = weights + LoRA adapters + optimizer states + gradients
           + activations + CUDA overhead.
Activation formula from unsloth_zoo/vllm_utils.py.
All constants empirically calibrated against Llama-3.2-1B on B200.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

QUANT_4BIT_FACTOR = 16 / 5
DOUBLE_QUANT_4BIT_FACTOR = (
    3.6  # bnb_4bit_use_double_quant; see VRAM_ESTIMATION.md section 1
)
CUDA_OVERHEAD_BYTES = int(1.4 * 1024**3)  # calibrated on RTX 5070 Ti
NON_FLASH_ATTENTION_FACTOR = (
    12.0  # eager attention score+workspace overhead; see VRAM_ESTIMATION.md section 5
)

LINEAR_ATTENTION_IMPLS = frozenset({"flash_attention_2", "sdpa", "flex_attention"})

_SKIP_MODULE_TEXT_PREFIXES = frozenset(
    {
        "model",
        "model.model",
        "language_model",
        "language_model.model",
        "model.language_model",
        "model.language_model.model",
    }
)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
ATTENTION_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_TARGET_MODULES = {"gate_proj", "up_proj", "down_proj"}

# Empirically calibrated bytes/param — see VRAM_ESTIMATION.md for rationale.
OPTIMIZER_BYTES_PER_PARAM: Dict[str, int] = {
    "adamw_8bit": 4,  # BNB upcasts to fp32 during step
    "paged_adamw_8bit": 4,
    "adamw_bnb_8bit": 4,
    "paged_adamw_32bit": 8,
    "adamw_torch": 6,  # fused, no master copy
    "adamw_torch_fused": 6,
    "sgd": 4,
}

# (full_ft_multiplier, lora_multiplier) — fraction of num_layers.
# LoRA: frozen base layers skip activation storage, but you always need
# at least ~1 layer in flight during backprop recomputation.
GC_LAYER_MULTIPLIERS = {
    "none": (None, None),
    "true": (2.0, 1.0),
    "unsloth": (1.5, 1.0),
}


@dataclass
class ModelArchConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = True
    num_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: int = 0
    shared_expert_intermediate_size: Optional[int] = None
    num_experts_per_tok: int = 1
    num_dense_layers: int = 0
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    head_dim: Optional[int] = None
    global_head_dim: Optional[int] = None
    num_global_key_value_heads: Optional[int] = None
    attention_k_eq_v: bool = False
    layer_types: Optional[list] = None
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    vocab_size_per_layer_input: int = 0
    hidden_size_per_layer_input: int = 0
    quantization_skip_modules: list = field(default_factory = list)
    quant_4bit_factor: float = QUANT_4BIT_FACTOR
    moe_has_dense_mlp: bool = False
    dense_layer_indices: tuple = ()
    dense_intermediate_size: Optional[int] = None


@dataclass
class TrainingVramConfig:
    training_method: str = "qlora"
    batch_size: int = 4
    max_seq_length: int = 2048
    lora_rank: int = 16
    target_modules: list = field(default_factory = lambda: list(DEFAULT_TARGET_MODULES))
    gradient_checkpointing: str = "unsloth"
    optimizer: str = "adamw_8bit"
    load_in_4bit: bool = True
    attention_implementation: str = "flash_attention_2"


@dataclass
class VramBreakdown:
    model_weights: int
    lora_adapters: int
    optimizer_states: int
    gradients: int
    activations: int
    cuda_overhead: int
    # Equals `activations`; retained for backward compatibility with
    # consumers that read this field.
    activations_computed: int = 0

    @property
    def total(self) -> int:
        return (
            self.model_weights
            + self.lora_adapters
            + self.optimizer_states
            + self.gradients
            + self.activations
            + self.cuda_overhead
        )

    def min_gpu_vram(self, n_gpus: int) -> int:
        """Minimum VRAM a single GPU needs: its shard + non-shardable costs.

        Weights/LoRA/optimizer/gradients shard across GPUs.
        Activations do NOT shard (the GPU running a layer holds them).
        """
        shardable = (
            self.model_weights
            + self.lora_adapters
            + self.optimizer_states
            + self.gradients
        )
        per_gpu_fixed = self.activations + self.cuda_overhead
        return shardable // max(n_gpus, 1) + per_gpu_fixed

    def to_gb_dict(self) -> Dict[str, float]:
        return {
            "model_weights_gb": round(self.model_weights / (1024**3), 3),
            "lora_adapters_gb": round(self.lora_adapters / (1024**3), 3),
            "optimizer_states_gb": round(self.optimizer_states / (1024**3), 3),
            "gradients_gb": round(self.gradients / (1024**3), 3),
            "activations_gb": round(self.activations / (1024**3), 3),
            "cuda_overhead_gb": round(self.cuda_overhead / (1024**3), 3),
            "total_gb": round(self.total / (1024**3), 3),
        }


def _first_scalar(value):
    # why: ERNIE MoE configs ship moe_intermediate_size / moe_num_experts as
    # [routed, shared] lists; downstream arithmetic needs the routed scalar.
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _max_scalar(value):
    # why: Hunyuan-V1-MoE moe_topk can be a per-layer list; activation
    # accounting uses the max top-k as a conservative upper bound.
    if isinstance(value, (list, tuple)):
        items = [v for v in value if v is not None]
        return max(items) if items else None
    return value


def _compute_dense_layer_indices(text_config, total_layers: int) -> tuple:
    """Layer indices that use dense MLP instead of MoE. Position matters."""
    # why: transformers Exaone-MoE / Laguna / Hy_v3 / GLM-MoE-DSA / GLM4-MoE-Lite /
    # Ernie4_5_VL_MoE prefer per-position `mlp_layer_types` over the prefix-style
    # `first_k_dense_replace` and may omit `decoder_sparse_step` entirely.
    layer_types = getattr(text_config, "mlp_layer_types", None)
    if layer_types:
        return tuple(
            i
            for i, t in enumerate(layer_types[:total_layers])
            if str(t).lower() == "dense"
        )

    # why: Llama4TextConfig.__init__ auto-populates self.moe_layers from
    # interleave_moe_layer_step; Llama4TextDecoderLayer dispatches via
    # `layer_idx in config.moe_layers` (modeling_llama4.py).
    llama4_moe_layers = getattr(text_config, "moe_layers", None)
    if llama4_moe_layers is not None:
        moe_indices = {int(i) for i in llama4_moe_layers}
        return tuple(i for i in range(total_layers) if i not in moe_indices)

    # why: transformers ERNIE 4.5 MoE / ERNIE 4.5 VL MoE declare MoE layers
    # via moe_layer_start_index / moe_layer_end_index / moe_layer_interval;
    # the model's per-layer guard is `(layer_idx + 1) % interval == 0` with
    # start <= layer_idx <= end (modeling_ernie4_5_moe.py).
    moe_start = getattr(text_config, "moe_layer_start_index", None)
    moe_interval = getattr(text_config, "moe_layer_interval", None)
    if moe_start is not None and moe_interval is not None and int(moe_interval) > 0:
        moe_end_raw = getattr(text_config, "moe_layer_end_index", None)
        end = (
            total_layers
            if moe_end_raw is None or int(moe_end_raw) == -1
            else min(int(moe_end_raw) + 1, total_layers)
        )
        start = max(0, int(moe_start))
        interval = int(moe_interval)
        moe_indices = {i for i in range(start, end) if (i + 1) % interval == 0}
        return tuple(i for i in range(total_layers) if i not in moe_indices)

    first_k = getattr(text_config, "first_k_dense_replace", None)
    if first_k is not None:
        return tuple(range(min(int(first_k), total_layers)))

    sparse_step = getattr(text_config, "decoder_sparse_step", None)
    mlp_only = getattr(text_config, "mlp_only_layers", None) or []
    if sparse_step is not None and sparse_step > 0:
        mlp_only_set = {int(i) for i in mlp_only}
        return tuple(
            i
            for i in range(total_layers)
            if i in mlp_only_set or (i + 1) % sparse_step != 0
        )
    return ()


def extract_arch_config(hf_config) -> Optional[ModelArchConfig]:
    text_config = getattr(hf_config, "text_config", None) or hf_config
    quantization_config = getattr(hf_config, "quantization_config", None) or {}
    if not isinstance(quantization_config, dict):
        quantization_config = getattr(quantization_config, "to_dict", lambda: {})()
    quant_4bit_factor = (
        DOUBLE_QUANT_4BIT_FACTOR
        if quantization_config.get("bnb_4bit_use_double_quant", False)
        else QUANT_4BIT_FACTOR
    )

    hidden_size = getattr(text_config, "hidden_size", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    num_heads = getattr(text_config, "num_attention_heads", None)
    intermediate_size = getattr(text_config, "intermediate_size", None)
    vocab_size = getattr(text_config, "vocab_size", None)

    if isinstance(intermediate_size, (list, tuple)):
        intermediate_size = intermediate_size[0] if intermediate_size else None
    if intermediate_size is None and hidden_size is not None:
        intermediate_size = hidden_size * 4

    if not all(
        v is not None
        for v in (hidden_size, num_layers, num_heads, intermediate_size, vocab_size)
    ):
        return None
    if num_heads <= 0:
        return None

    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)

    # why: DBRX places its MoE attrs on the DbrxFFNConfig sub-config; probe
    # ffn_config as a secondary source so DBRX is not misclassified as dense.
    ffn_config = getattr(text_config, "ffn_config", None)

    def _moe_attr(name):
        value = getattr(text_config, name, None)
        if value is None and ffn_config is not None:
            value = getattr(ffn_config, name, None)
        return value

    num_experts = None
    for attr in (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
    ):
        num_experts = _first_scalar(_moe_attr(attr))
        if num_experts is not None:
            break

    moe_intermediate_raw = _moe_attr("moe_intermediate_size")
    if moe_intermediate_raw is None:
        moe_intermediate_raw = _moe_attr("ffn_hidden_size")
    moe_intermediate = _first_scalar(moe_intermediate_raw)
    # why: Exaone-MoE / ERNIE families alias num_shared_experts /
    # moe_num_shared_experts to the canonical n_shared_experts.
    n_shared_experts = (
        _first_scalar(_moe_attr("n_shared_experts"))
        or _first_scalar(_moe_attr("num_shared_experts"))
        or _first_scalar(_moe_attr("moe_num_shared_experts"))
        or 0
    )
    shared_expert_intermediate_size = _moe_attr("shared_expert_intermediate_size")
    if shared_expert_intermediate_size and n_shared_experts == 0:
        n_shared_experts = 1
    # why: DBRX exposes moe_top_k, Hunyuan-V1-MoE exposes moe_topk (which can
    # be a per-layer list); _max_scalar normalizes list values to the worst
    # case so int(...) below cannot crash on the canonical attribute_map path.
    num_experts_per_tok = (
        _max_scalar(_moe_attr("num_experts_per_tok"))
        or _max_scalar(_moe_attr("top_k_experts"))
        or _max_scalar(_moe_attr("moe_top_k"))
        or _max_scalar(_moe_attr("moe_topk"))
        or 1
    )

    dense_layer_indices: tuple = ()
    if num_experts is not None and num_experts > 1:
        dense_layer_indices = _compute_dense_layer_indices(text_config, num_layers)
    num_dense_layers = len(dense_layer_indices)

    # why: Llama4 dense layers use intermediate_size_mlp; routed and shared
    # experts use intermediate_size. Llama4TextMoe builds one shared_expert
    # per MoE layer (modeling_llama4.py).
    intermediate_size_mlp_raw = _first_scalar(_moe_attr("intermediate_size_mlp"))
    dense_intermediate_size = (
        int(intermediate_size_mlp_raw)
        if intermediate_size_mlp_raw is not None
        else None
    )
    if (
        intermediate_size_mlp_raw is not None
        and num_experts is not None
        and num_experts > 1
        and shared_expert_intermediate_size is None
        and n_shared_experts == 0
    ):
        n_shared_experts = 1

    q_lora_rank = getattr(text_config, "q_lora_rank", None)
    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    qk_nope_head_dim = getattr(text_config, "qk_nope_head_dim", None)
    qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", None)
    v_head_dim = getattr(text_config, "v_head_dim", None)

    return ModelArchConfig(
        hidden_size = hidden_size,
        num_hidden_layers = num_layers,
        num_attention_heads = num_heads,
        num_key_value_heads = num_kv_heads,
        intermediate_size = intermediate_size,
        vocab_size = vocab_size,
        tie_word_embeddings = getattr(text_config, "tie_word_embeddings", True),
        num_experts = num_experts,
        moe_intermediate_size = moe_intermediate,
        n_shared_experts = n_shared_experts,
        shared_expert_intermediate_size = shared_expert_intermediate_size,
        num_experts_per_tok = int(num_experts_per_tok),
        num_dense_layers = num_dense_layers,
        q_lora_rank = q_lora_rank,
        kv_lora_rank = kv_lora_rank,
        qk_nope_head_dim = qk_nope_head_dim,
        qk_rope_head_dim = qk_rope_head_dim,
        v_head_dim = v_head_dim,
        head_dim = getattr(text_config, "head_dim", None),
        global_head_dim = getattr(text_config, "global_head_dim", None),
        num_global_key_value_heads = getattr(
            text_config,
            "num_global_key_value_heads",
            None,
        ),
        attention_k_eq_v = bool(getattr(text_config, "attention_k_eq_v", False)),
        layer_types = getattr(text_config, "layer_types", None),
        num_kv_shared_layers = getattr(text_config, "num_kv_shared_layers", None) or 0,
        use_double_wide_mlp = bool(getattr(text_config, "use_double_wide_mlp", False)),
        vocab_size_per_layer_input = getattr(
            text_config,
            "vocab_size_per_layer_input",
            None,
        )
        or 0,
        hidden_size_per_layer_input = getattr(
            text_config,
            "hidden_size_per_layer_input",
            None,
        )
        or 0,
        quantization_skip_modules = list(
            quantization_config.get("llm_int8_skip_modules", []) or []
        ),
        quant_4bit_factor = quant_4bit_factor,
        moe_has_dense_mlp = bool(getattr(text_config, "enable_moe_block", False)),
        dense_layer_indices = dense_layer_indices,
        dense_intermediate_size = dense_intermediate_size,
    )


def _targets_all_linear(target_modules) -> bool:
    # why: peft LoraConfig accepts target_modules="all-linear" as a bare
    # string; iterating a string yields chars and never matches the set.
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    normalized = {str(module).lower().replace("_", "-") for module in target_modules}
    return normalized == {"all-linear"}


def _head_dim(arch: ModelArchConfig) -> int:
    return arch.head_dim or arch.hidden_size // arch.num_attention_heads


def _layer_types(arch: ModelArchConfig) -> list:
    if arch.layer_types and len(arch.layer_types) == arch.num_hidden_layers:
        return arch.layer_types
    return ["full_attention"] * arch.num_hidden_layers


def _uses_structured_layer_shapes(arch: ModelArchConfig) -> bool:
    # MLA configs have their own q/kv low-rank projection shape formulas in
    # _compute_attn_elements / _lora_attn_elements; do not let head_dim or
    # other structured fields override that path.
    if arch.q_lora_rank is not None:
        return False
    return bool(
        arch.layer_types
        or arch.head_dim is not None
        or arch.global_head_dim is not None
        or arch.num_global_key_value_heads is not None
        or arch.attention_k_eq_v
        or arch.num_kv_shared_layers > 0
        or arch.use_double_wide_mlp
    )


def _is_kv_shared_layer(arch: ModelArchConfig, layer_idx: int) -> bool:
    if arch.num_kv_shared_layers <= 0:
        return False
    first_shared = arch.num_hidden_layers - arch.num_kv_shared_layers
    # why: transformers Gemma4 (modeling_gemma4.py:1031, modular_gemma4.py:863)
    # uses the same `> 0` guard so a fully-shared config raises during model
    # construction; matching upstream avoids producing a detailed estimate
    # for a shape the actual model code rejects.
    return layer_idx >= first_shared > 0


def _is_dense_mlp_layer(arch: ModelArchConfig, layer_idx: int) -> bool:
    if arch.dense_layer_indices:
        return layer_idx in arch.dense_layer_indices
    return layer_idx < arch.num_dense_layers


def _per_layer_input_quantizable(arch: ModelArchConfig) -> int:
    # why: Gemma4 PLE block adds per_layer_model_projection (single Linear),
    # per_layer_input_gate (per layer), and per_layer_projection (per layer);
    # see transformers gemma4/modular_gemma4.py:1077-1083 and :1247-1253.
    pli = arch.hidden_size_per_layer_input
    if pli <= 0:
        return 0
    n_layers = arch.num_hidden_layers
    hd = arch.hidden_size
    return hd * (n_layers * pli) + (hd * pli) * n_layers + (pli * hd) * n_layers


def _per_layer_input_norm_elements(arch: ModelArchConfig) -> int:
    pli = arch.hidden_size_per_layer_input
    if pli <= 0:
        return 0
    n_layers = arch.num_hidden_layers
    hd = arch.hidden_size
    return hd * n_layers + pli


def _per_layer_input_lora_params(
    arch: ModelArchConfig,
    r: int,
    target_modules,
) -> int:
    # why: Unsloth's get_peft_regex (unsloth_zoo/peft_utils.py) requires module
    # names to contain a component tag (mlp/attn/...); PLE module names lack
    # any tag, so all-linear training does NOT attach LoRA to them. Only count
    # PLE LoRA when the user explicitly names PLE modules.
    pli = arch.hidden_size_per_layer_input
    if pli <= 0:
        return 0
    targets = (
        {target_modules}
        if isinstance(target_modules, str)
        else set(target_modules or [])
    )
    n_layers = arch.num_hidden_layers
    hd = arch.hidden_size
    total = 0
    if "per_layer_model_projection" in targets:
        total += hd * r + r * (n_layers * pli)
    if "per_layer_input_gate" in targets:
        total += (hd * r + r * pli) * n_layers
    if "per_layer_projection" in targets:
        total += (pli * r + r * hd) * n_layers
    return total


def _layer_attention_dims(arch: ModelArchConfig, layer_idx: int) -> tuple:
    layer_types = _layer_types(arch)
    layer_type = layer_types[layer_idx]
    is_sliding = layer_type == "sliding_attention"
    head_dim = (
        arch.global_head_dim
        if not is_sliding and arch.global_head_dim
        else _head_dim(arch)
    )
    use_alt_attention = arch.attention_k_eq_v and not is_sliding
    num_kv_heads = (
        arch.num_global_key_value_heads
        if use_alt_attention and arch.num_global_key_value_heads
        else arch.num_key_value_heads
    )
    q_size = arch.num_attention_heads * head_dim
    kv_size = num_kv_heads * head_dim
    has_k = not _is_kv_shared_layer(arch, layer_idx)
    has_v = has_k and not use_alt_attention
    return q_size, kv_size, has_k, has_v


def _layer_mlp_size(arch: ModelArchConfig, layer_idx: int) -> int:
    if arch.use_double_wide_mlp and _is_kv_shared_layer(arch, layer_idx):
        return _dense_mlp_size(arch) * 2
    return _dense_mlp_size(arch)


def _text_linear_dims(
    arch: ModelArchConfig,
    layer_idx: int,
) -> Dict[str, tuple[int, int]]:
    hd = arch.hidden_size
    if _uses_structured_layer_shapes(arch):
        q_size, kv_size, has_k, has_v = _layer_attention_dims(arch, layer_idx)
        mlp_size = _layer_mlp_size(arch, layer_idx)
    else:
        q_size = hd
        kv_size = _get_kv_size(arch)
        has_k = True
        has_v = True
        mlp_size = _get_mlp_size(arch)

    dims = {
        "q_proj": (hd, q_size),
        "o_proj": (q_size, hd),
    }
    if has_k:
        dims["k_proj"] = (hd, kv_size)
    if has_v:
        dims["v_proj"] = (hd, kv_size)

    dims.update(
        {
            "gate_proj": (hd, mlp_size),
            "up_proj": (hd, mlp_size),
            "down_proj": (mlp_size, hd),
        }
    )
    return dims


def _module_path_matches(skip_module: str, alias: str) -> bool:
    skip_parts = [part for part in skip_module.split(".") if part]
    alias_parts = [part for part in alias.split(".") if part]
    if not skip_parts or not alias_parts:
        return False
    if alias_parts[0] == "layers":
        return skip_parts == alias_parts
    if len(skip_parts) <= len(alias_parts):
        # why: transformers BNB quantizer suffix-matches short skip entries
        # like ["q_proj"] / ["lm_head"] against full module paths, so a skip
        # shorter than the alias is a tail match.
        return alias_parts[-len(skip_parts) :] == skip_parts
    if skip_parts[-len(alias_parts) :] != alias_parts:
        return False
    prefix_parts = skip_parts[: len(skip_parts) - len(alias_parts)]
    if not prefix_parts:
        return True
    # why: bound the prefix to known text-tower roots so VLM skip names like
    # vision_tower.model.layers.<i>.self_attn.q_proj do not shadow the text
    # alias model.layers.<i>.self_attn.q_proj.
    return ".".join(prefix_parts) in _SKIP_MODULE_TEXT_PREFIXES


def _add_module_aliases(
    aliases: Dict[str, str],
    canonical: str,
    suffix: str,
) -> None:
    for prefix in (
        "",
        "model",
        "model.model",
        "language_model",
        "language_model.model",
        "model.language_model",
        "model.language_model.model",
    ):
        alias = f"{prefix}.{suffix}" if prefix else suffix
        aliases[alias] = canonical


def _build_text_module_elements(
    arch: ModelArchConfig,
) -> tuple[Dict[str, int], Dict[str, str]]:
    elements: Dict[str, int] = {}
    aliases: Dict[str, str] = {}

    is_mla = arch.q_lora_rank is not None and not _uses_structured_layer_shapes(arch)
    pli = arch.hidden_size_per_layer_input
    hd_global = arch.hidden_size

    for layer_idx in range(arch.num_hidden_layers):
        layer_modules: Dict[str, int] = {}
        dims = _text_linear_dims(arch, layer_idx)
        attn_dims = {
            name: dim for name, dim in dims.items() if name in ATTENTION_TARGET_MODULES
        }
        mlp_dims = {
            name: dim for name, dim in dims.items() if name in MLP_TARGET_MODULES
        }

        if is_mla:
            # why: _text_linear_dims uses (hd, hd) for q/o; MLA actually splits
            # into q_a/q_b/kv_a/kv_b, so emit a single self_attn aggregate at
            # the authoritative MLA per-layer total.
            layer_modules["self_attn"] = _compute_attn_elements(arch)
        else:
            for name, (in_dim, out_dim) in attn_dims.items():
                layer_modules[f"self_attn.{name}"] = in_dim * out_dim

        if arch.num_experts and arch.num_experts > 1:
            if _is_dense_mlp_layer(arch, layer_idx):
                layer_modules.update(
                    {
                        f"mlp.{name}": in_dim * out_dim
                        for name, (in_dim, out_dim) in mlp_dims.items()
                    }
                )
            else:
                layer_modules["mlp.experts"] = _compute_routed_moe_elements(arch)
                shared_moe = _compute_shared_moe_elements(arch)
                if shared_moe:
                    # why: Qwen3.5-MoE exposes shared expert as
                    # mlp.shared_expert; Exaone-MoE/Laguna/GLM-style configs use
                    # mlp.shared_experts. Register both names so child-path
                    # llm_int8_skip_modules entries match the right shared block.
                    layer_modules["mlp.shared_expert"] = shared_moe
                if arch.moe_has_dense_mlp:
                    # why: enable_moe_block runs the dense MLP and the MoE
                    # experts in parallel; register both for skip matching.
                    # Non-structured _text_linear_dims returns mlp_size from
                    # _get_mlp_size which prefers moe_intermediate_size, so
                    # rebuild dense dims from arch.intermediate_size directly.
                    if _uses_structured_layer_shapes(arch):
                        dense_dims = mlp_dims
                    else:
                        hd = arch.hidden_size
                        inter = arch.intermediate_size
                        dense_dims = {
                            "gate_proj": (hd, inter),
                            "up_proj": (hd, inter),
                            "down_proj": (inter, hd),
                        }
                    layer_modules.update(
                        {
                            f"mlp.{name}": in_dim * out_dim
                            for name, (in_dim, out_dim) in dense_dims.items()
                        }
                    )
        else:
            layer_modules.update(
                {
                    f"mlp.{name}": in_dim * out_dim
                    for name, (in_dim, out_dim) in mlp_dims.items()
                }
            )

        if pli > 0:
            # why: register PLE per-layer linears so llm_int8_skip_modules
            # entries like model.layers.0.per_layer_input_gate match.
            layer_modules["per_layer_input_gate"] = hd_global * pli
            layer_modules["per_layer_projection"] = pli * hd_global

        attn_total = sum(
            value
            for name, value in layer_modules.items()
            if name == "self_attn" or name.startswith("self_attn.")
        )
        # why: gemma4 enable_moe_block puts routed experts at the sibling
        # layers.<i>.experts attribute, not under self.mlp; the layer's "mlp"
        # aggregate must reflect only the dense MLP path so a skip module
        # `model.layers.0.mlp` does not over-skip into the experts block.
        is_sibling_experts = bool(arch.moe_has_dense_mlp)
        mlp_total = sum(
            value
            for name, value in layer_modules.items()
            if (
                name == "mlp"
                or (
                    name.startswith("mlp.")
                    and not (is_sibling_experts and name == "mlp.experts")
                )
            )
        )
        experts_total = layer_modules.get("mlp.experts", 0) if is_sibling_experts else 0
        layer_total = sum(layer_modules.values())

        aggregate_modules = {
            f"text.layers.{layer_idx}": layer_total,
            f"text.layers.{layer_idx}.self_attn": attn_total,
            f"text.layers.{layer_idx}.mlp": mlp_total,
        }
        if experts_total:
            aggregate_modules[f"text.layers.{layer_idx}.experts"] = experts_total
        elements.update(aggregate_modules)
        for canonical in aggregate_modules:
            suffix = canonical.removeprefix("text.")
            _add_module_aliases(aliases, canonical, suffix)

        for name, value in layer_modules.items():
            canonical = f"text.layers.{layer_idx}.{name}"
            elements[canonical] = value
            _add_module_aliases(aliases, canonical, canonical.removeprefix("text."))
            if name == "mlp.experts" and arch.moe_has_dense_mlp:
                # why: gemma4 enable_moe_block exposes routed experts at
                # layers.<i>.experts (sibling of self.mlp), not under mlp.
                _add_module_aliases(aliases, canonical, f"layers.{layer_idx}.experts")
            elif name == "mlp.shared_expert":
                # why: Exaone-MoE / Laguna / GLM-style configs use the plural
                # `shared_experts` attribute name; register both spellings.
                _add_module_aliases(
                    aliases,
                    canonical,
                    f"layers.{layer_idx}.mlp.shared_experts",
                )

    if pli > 0:
        canonical = "text.per_layer_model_projection"
        elements[canonical] = hd_global * (arch.num_hidden_layers * pli)
        _add_module_aliases(aliases, canonical, canonical.removeprefix("text."))

    return elements, aliases


def _compute_skipped_quantizable_elements(arch: ModelArchConfig) -> int:
    if not arch.quantization_skip_modules:
        return 0

    module_elements, aliases = _build_text_module_elements(arch)
    matched = set()
    for skip_module in arch.quantization_skip_modules:
        for alias, canonical in aliases.items():
            if _module_path_matches(skip_module, alias):
                matched.add(canonical)

    pruned = {
        canonical
        for canonical in matched
        if not any(
            canonical != parent and canonical.startswith(f"{parent}.")
            for parent in matched
        )
    }
    return sum(module_elements[canonical] for canonical in pruned)


def _get_kv_size(arch: ModelArchConfig) -> int:
    return (arch.hidden_size // arch.num_attention_heads) * arch.num_key_value_heads


def _get_mlp_size(arch: ModelArchConfig) -> int:
    if arch.moe_intermediate_size is not None:
        return arch.moe_intermediate_size
    return arch.intermediate_size


def _dense_mlp_size(arch: ModelArchConfig) -> int:
    # why: Llama4 dense layers use intermediate_size_mlp; routed/shared
    # experts use intermediate_size. Other configs leave the field None.
    return arch.dense_intermediate_size or arch.intermediate_size


def _get_num_experts(arch: ModelArchConfig) -> int:
    return arch.num_experts if arch.num_experts and arch.num_experts > 1 else 1


def _compute_attn_elements(arch: ModelArchConfig) -> int:
    """Attention weight elements per layer."""
    hd = arch.hidden_size
    if arch.q_lora_rank is not None:
        nh = arch.num_attention_heads
        qk_head = arch.qk_nope_head_dim + arch.qk_rope_head_dim
        q_a = hd * arch.q_lora_rank
        q_b = arch.q_lora_rank * (nh * qk_head)
        kv_a = hd * (arch.kv_lora_rank + arch.qk_rope_head_dim)
        kv_b = arch.kv_lora_rank * (nh * (arch.qk_nope_head_dim + arch.v_head_dim))
        o = (nh * arch.v_head_dim) * hd
        norms = arch.q_lora_rank + arch.kv_lora_rank
        return q_a + q_b + kv_a + kv_b + o + norms
    kv_size = _get_kv_size(arch)
    return (hd + kv_size + kv_size + hd) * hd


def _compute_dense_mlp_elements(arch: ModelArchConfig) -> int:
    return arch.hidden_size * _dense_mlp_size(arch) * 3


def _shared_expert_size(arch: ModelArchConfig) -> int:
    # why: Qwen3.5-MoE shared expert has its own intermediate_size (default 512)
    # distinct from moe_intermediate_size; fall back to routed mlp_size for
    # families that share it (deepseek-style configs).
    return arch.shared_expert_intermediate_size or _get_mlp_size(arch)


def _compute_routed_moe_elements(arch: ModelArchConfig) -> int:
    hd = arch.hidden_size
    n_experts = _get_num_experts(arch)
    return hd * _get_mlp_size(arch) * 3 * n_experts + n_experts * hd


def _compute_shared_moe_elements(arch: ModelArchConfig) -> int:
    if not arch.n_shared_experts:
        return 0
    hd = arch.hidden_size
    shared_size = _shared_expert_size(arch)
    total = hd * shared_size * 3 * arch.n_shared_experts
    # why: only Qwen2-MoE / Qwen3.5-MoE define a shared_expert_gate Linear
    # (hidden_size→1); other families (Exaone-MoE, HY-V3, GLM4-MoE-Lite, Laguna)
    # have shared_experts without a gate. shared_expert_intermediate_size is the
    # Qwen-style discriminator.
    if arch.shared_expert_intermediate_size:
        total += arch.n_shared_experts * hd
    return total


def _compute_moe_mlp_elements(arch: ModelArchConfig) -> int:
    return _compute_routed_moe_elements(arch) + _compute_shared_moe_elements(arch)


def _compute_layer_elements(arch: ModelArchConfig):
    """Return (total_quantizable, layernorms_per_layer, embed, lm_head) element counts.

    total_quantizable is summed across ALL layers (not per-layer).
    """
    hd = arch.hidden_size
    n_layers = arch.num_hidden_layers
    n_experts = _get_num_experts(arch)

    if _uses_structured_layer_shapes(arch):
        attn_total = 0
        per_layer_dense_mlp = []
        for layer_idx in range(n_layers):
            layer_dense_mlp = 0
            for name, (in_dim, out_dim) in _text_linear_dims(
                arch,
                layer_idx,
            ).items():
                elements = in_dim * out_dim
                if name in ATTENTION_TARGET_MODULES:
                    attn_total += elements
                elif name in MLP_TARGET_MODULES:
                    layer_dense_mlp += elements
            per_layer_dense_mlp.append(layer_dense_mlp)
        if n_experts > 1:
            n_dense = arch.num_dense_layers
            n_moe = n_layers - n_dense
            moe_mlp_total = _compute_moe_mlp_elements(arch) * n_moe
            if arch.moe_has_dense_mlp:
                # why: enable_moe_block runs dense MLP and MoE experts in
                # parallel; count dense for every layer alongside MoE.
                mlp_total = sum(per_layer_dense_mlp) + moe_mlp_total
            else:
                dense_only_total = sum(
                    value
                    for i, value in enumerate(per_layer_dense_mlp)
                    if _is_dense_mlp_layer(arch, i)
                )
                mlp_total = moe_mlp_total + dense_only_total
        else:
            mlp_total = sum(per_layer_dense_mlp)
    elif n_experts > 1:
        attn_total = _compute_attn_elements(arch) * n_layers
        n_dense = arch.num_dense_layers
        n_moe = n_layers - n_dense
        moe_mlp_total = _compute_moe_mlp_elements(arch) * n_moe
        if arch.moe_has_dense_mlp:
            mlp_total = _compute_dense_mlp_elements(arch) * n_layers + moe_mlp_total
        else:
            mlp_total = moe_mlp_total + _compute_dense_mlp_elements(arch) * n_dense
    else:
        attn_total = _compute_attn_elements(arch) * n_layers
        mlp_total = _compute_dense_mlp_elements(arch) * n_layers

    layernorms = 2 * hd
    per_layer_embed = (
        arch.vocab_size_per_layer_input * arch.hidden_size_per_layer_input * n_layers
    )
    ple_text_linear = _per_layer_input_quantizable(arch)
    ple_norms = _per_layer_input_norm_elements(arch)
    embed_tokens = arch.vocab_size * hd + per_layer_embed + ple_norms
    lm_head = 0 if arch.tie_word_embeddings else arch.vocab_size * hd
    return attn_total + mlp_total + ple_text_linear, layernorms, embed_tokens, lm_head


def compute_model_weights_bytes(
    arch: ModelArchConfig,
    training_method: str,
    load_in_4bit: bool,
) -> int:
    total_quantizable, layernorms, embed_tokens, lm_head = _compute_layer_elements(arch)
    n_layers = arch.num_hidden_layers
    non_quantizable = layernorms * n_layers + embed_tokens + lm_head

    if training_method == "qlora" and load_in_4bit:
        skipped_quantizable = min(
            _compute_skipped_quantizable_elements(arch),
            total_quantizable,
        )
        quantized = total_quantizable - skipped_quantizable
        return int(
            quantized * 2 / arch.quant_4bit_factor
            + skipped_quantizable * 2
            + non_quantizable * 2
        )

    return int((total_quantizable + non_quantizable) * 2)


def compute_total_params(arch: ModelArchConfig) -> int:
    total_quantizable, layernorms, embed_tokens, lm_head = _compute_layer_elements(arch)
    n_layers = arch.num_hidden_layers
    return total_quantizable + layernorms * n_layers + embed_tokens + lm_head


def _lora_attn_elements(
    arch: ModelArchConfig,
    r: int,
    target_modules: list,
) -> int:
    hd = arch.hidden_size
    if arch.q_lora_rank is not None:
        # MLA: q_proj->q_b, k_proj->kv_a, v_proj->kv_b, o_proj->o
        nh = arch.num_attention_heads
        qk_head = arch.qk_nope_head_dim + arch.qk_rope_head_dim
        kv_out = nh * (arch.qk_nope_head_dim + arch.v_head_dim)
        o_in = nh * arch.v_head_dim
        dims = {
            "q_proj": (arch.q_lora_rank, nh * qk_head),
            "k_proj": (hd, arch.kv_lora_rank + arch.qk_rope_head_dim),
            "v_proj": (arch.kv_lora_rank, kv_out),
            "o_proj": (o_in, hd),
        }
    else:
        kv_size = _get_kv_size(arch)
        dims = {
            "q_proj": (hd, hd),
            "k_proj": (hd, kv_size),
            "v_proj": (hd, kv_size),
            "o_proj": (hd, hd),
        }
    total = 0
    for name, (in_dim, out_dim) in dims.items():
        if name in target_modules:
            total += in_dim * r + r * out_dim
    return total


def _lora_mlp_elements(
    hd: int,
    mlp_size: int,
    r: int,
    target_modules: list,
    expert_mult: int,
) -> int:
    module_ab = {
        "gate_proj": (hd * r, r * mlp_size),
        "up_proj": (hd * r, r * mlp_size),
        "down_proj": (mlp_size * r, r * hd),
    }
    total = 0
    for name, (a, b) in module_ab.items():
        if name in target_modules:
            total += (a + b) * expert_mult
    return total


def compute_lora_params(
    arch: ModelArchConfig,
    lora_rank: int,
    target_modules: list,
) -> int:
    all_linear = _targets_all_linear(target_modules)
    selected_modules = list(DEFAULT_TARGET_MODULES) if all_linear else target_modules
    hd = arch.hidden_size
    r = lora_rank
    n_layers = arch.num_hidden_layers
    n_experts = _get_num_experts(arch)

    use_structured_shapes = _uses_structured_layer_shapes(arch)
    if use_structured_shapes:
        attn_total = 0
        structured_dense_mlp = 0
        per_layer_dense_mlp = []
        for layer_idx in range(n_layers):
            layer_dense = 0
            for name, (in_dim, out_dim) in _text_linear_dims(
                arch,
                layer_idx,
            ).items():
                if name not in selected_modules:
                    continue
                if name in ATTENTION_TARGET_MODULES:
                    attn_total += in_dim * r + r * out_dim
                elif name in MLP_TARGET_MODULES:
                    layer_dense += in_dim * r + r * out_dim
            per_layer_dense_mlp.append(layer_dense)
            structured_dense_mlp += layer_dense
        if n_experts > 1:
            n_dense = arch.num_dense_layers
            n_moe = n_layers - n_dense
            # why: peft "all-linear" attaches LoRA to nn.Linear only;
            # routed experts are nn.Parameter and need explicit
            # gate_proj/up_proj/down_proj naming via Unsloth's
            # get_moe_target_parameters. Shared experts are nn.Linear and
            # are picked up by get_peft_regex.
            routed_moe = (
                0
                if all_linear
                else _lora_mlp_elements(
                    hd,
                    _get_mlp_size(arch),
                    r,
                    selected_modules,
                    n_experts,
                )
            )
            shared_moe = _lora_mlp_elements(
                hd,
                _shared_expert_size(arch),
                r,
                selected_modules,
                arch.n_shared_experts,
            )
            moe_mlp = routed_moe + shared_moe
            if arch.moe_has_dense_mlp:
                # why: parallel dense MLP coexists with MoE on every layer.
                mlp_total = structured_dense_mlp + moe_mlp * n_moe
            else:
                dense_only = sum(
                    value
                    for i, value in enumerate(per_layer_dense_mlp)
                    if _is_dense_mlp_layer(arch, i)
                )
                mlp_total = moe_mlp * n_moe + dense_only
        else:
            mlp_total = structured_dense_mlp
        return (
            attn_total
            + mlp_total
            + _per_layer_input_lora_params(arch, r, target_modules)
        )
    elif n_experts > 1:
        attn_total = _lora_attn_elements(arch, r, selected_modules) * n_layers
        n_dense = arch.num_dense_layers
        n_moe = n_layers - n_dense
        # why: routed and shared experts may use different intermediate sizes
        # (Qwen3.5-MoE: routed mlp_size != shared_expert_intermediate_size).
        # See structured branch for the all-linear exclusion rationale; only
        # routed (nn.Parameter) experts are excluded under all-linear.
        routed_moe = (
            0
            if all_linear
            else _lora_mlp_elements(
                hd,
                _get_mlp_size(arch),
                r,
                selected_modules,
                n_experts,
            )
        )
        shared_moe = _lora_mlp_elements(
            hd,
            _shared_expert_size(arch),
            r,
            selected_modules,
            arch.n_shared_experts,
        )
        moe_mlp = routed_moe + shared_moe
        dense_mlp = _lora_mlp_elements(
            hd,
            _dense_mlp_size(arch),
            r,
            selected_modules,
            1,
        )
        if arch.moe_has_dense_mlp:
            mlp_total = moe_mlp * n_moe + dense_mlp * n_layers
        else:
            mlp_total = moe_mlp * n_moe + dense_mlp * n_dense
    else:
        attn_total = _lora_attn_elements(arch, r, selected_modules) * n_layers
        mlp_total = (
            _lora_mlp_elements(
                hd,
                _dense_mlp_size(arch),
                r,
                selected_modules,
                1,
            )
            * n_layers
        )

    return (
        attn_total + mlp_total + _per_layer_input_lora_params(arch, r, target_modules)
    )


def compute_lora_adapter_bytes(lora_params: int) -> int:
    return lora_params * 2


def compute_optimizer_bytes(trainable_params: int, optimizer: str) -> int:
    optimizer_key = optimizer.lower().replace("-", "_")
    bytes_per_param = OPTIMIZER_BYTES_PER_PARAM.get(optimizer_key, 4)
    return trainable_params * bytes_per_param


def compute_gradient_bytes(trainable_params: int) -> int:
    return trainable_params * 2


def _is_linear_attention(attention_implementation: Optional[str]) -> bool:
    # why: PyTorch SDPA dispatches to flash/memory-efficient O(n) backends; only
    # eager (and other non-flash impls) need the quadratic correction.
    return attention_implementation in LINEAR_ATTENTION_IMPLS


def _compute_non_flash_attention_bytes(
    arch: ModelArchConfig,
    batch_size: int,
    seq_len: int,
    effective_layers: float,
) -> int:
    score_elements = batch_size * arch.num_attention_heads * seq_len * seq_len
    return int(score_elements * 2 * NON_FLASH_ATTENTION_FACTOR * effective_layers)


def _layer_qkv_mlp_sizes(arch: ModelArchConfig, layer_idx: int) -> tuple:
    n_experts = _get_num_experts(arch)
    is_moe_layer = n_experts > 1 and not _is_dense_mlp_layer(arch, layer_idx)
    if _uses_structured_layer_shapes(arch):
        q_size, kv_size, _has_k, _has_v = _layer_attention_dims(arch, layer_idx)
        # why: KV-shared layers (Gemma4/Gemma3n) drop k_proj/v_proj WEIGHTS but
        # the donor layer's K/V tensors stay alive across the shared range, so
        # activation memory still pays for kv_size; only the weight path uses
        # has_k/has_v.
        layer_type = _layer_types(arch)[layer_idx]
        use_alt_attention = arch.attention_k_eq_v and layer_type != "sliding_attention"
        kv_count = 1 if use_alt_attention else 2
        qkv_size = q_size + kv_size * kv_count
        if is_moe_layer:
            # why: each token routes through `num_experts_per_tok` experts; their
            # gate/up/down intermediates are all live during MLP forward.
            mlp_size = _get_mlp_size(arch) * arch.num_experts_per_tok
            if arch.n_shared_experts:
                mlp_size += _shared_expert_size(arch) * arch.n_shared_experts
            if arch.moe_has_dense_mlp:
                mlp_size += _layer_mlp_size(arch, layer_idx)
        else:
            mlp_size = _layer_mlp_size(arch, layer_idx)
        return qkv_size, mlp_size
    kv_size = _get_kv_size(arch)
    if is_moe_layer:
        mlp_size = _get_mlp_size(arch) * arch.num_experts_per_tok
        if arch.n_shared_experts:
            mlp_size += _shared_expert_size(arch) * arch.n_shared_experts
        if arch.moe_has_dense_mlp:
            mlp_size += arch.intermediate_size
    else:
        mlp_size = _get_mlp_size(arch)
    return arch.hidden_size + kv_size + kv_size, mlp_size


def _per_layer_activation_bytes(
    arch: ModelArchConfig,
    layer_idx: int,
    batch_size: int,
    seq_len: int,
) -> int:
    qkv_size, mlp_size = _layer_qkv_mlp_sizes(arch, layer_idx)
    activation_qkv = seq_len * batch_size * qkv_size
    residual_memory = (seq_len * batch_size) * 2
    activation_mlp = seq_len * batch_size * (mlp_size + mlp_size)
    # why: per_layer_input_gate (hd-sized) and per_layer_projection (pli-sized)
    # outputs materialize once per decoder layer when hidden_size_per_layer_input
    # is set; see gemma4/modular_gemma4.py:1141-1145.
    pli = arch.hidden_size_per_layer_input
    activation_ple = seq_len * batch_size * (arch.hidden_size + pli) if pli > 0 else 0
    return int(
        (activation_qkv + residual_memory + activation_mlp + activation_ple) * 2 * 1.25
    )


def compute_activation_bytes(
    arch: ModelArchConfig,
    batch_size: int,
    seq_len: int,
    gradient_checkpointing: str,
    is_lora: bool = False,
    attention_implementation: Optional[str] = "flash_attention_2",
) -> int:
    n_layers = arch.num_hidden_layers

    gc_key = gradient_checkpointing.lower()
    gc_entry = GC_LAYER_MULTIPLIERS.get(gc_key, (None, None))
    full_ft_mult, lora_mult = gc_entry
    gc_multiplier = lora_mult if is_lora else full_ft_mult

    if gc_multiplier is None:
        effective_layers = n_layers
        linear_bytes = sum(
            _per_layer_activation_bytes(arch, i, batch_size, seq_len)
            for i in range(n_layers)
        )
    else:
        effective_layers = gc_multiplier
        max_layer_bytes = max(
            _per_layer_activation_bytes(arch, i, batch_size, seq_len)
            for i in range(n_layers)
        )
        linear_bytes = int(max_layer_bytes * effective_layers)

    # why: gemma4 per_layer_model_projection runs once outside the per-decoder
    # loop and materializes a [B, S, L, PLI] tensor; see modular_gemma4.py:1247.
    pli = arch.hidden_size_per_layer_input
    if pli > 0:
        linear_bytes += int(seq_len * batch_size * n_layers * pli * 2 * 1.25)

    if _is_linear_attention(attention_implementation):
        return linear_bytes
    return max(
        linear_bytes,
        _compute_non_flash_attention_bytes(
            arch,
            batch_size,
            seq_len,
            effective_layers,
        ),
    )


def estimate_training_vram(
    arch: ModelArchConfig,
    config: TrainingVramConfig,
) -> VramBreakdown:
    method = config.training_method.lower()
    is_lora = method in ("qlora", "lora")
    load_in_4bit = config.load_in_4bit or method == "qlora"

    model_weights = compute_model_weights_bytes(arch, method, load_in_4bit)

    lora_params = 0
    lora_adapter_bytes = 0
    if is_lora:
        lora_params = compute_lora_params(
            arch,
            config.lora_rank,
            config.target_modules,
        )
        lora_adapter_bytes = compute_lora_adapter_bytes(lora_params)

    trainable_params = lora_params if is_lora else compute_total_params(arch)
    optimizer_bytes = compute_optimizer_bytes(trainable_params, config.optimizer)
    activations_computed = compute_activation_bytes(
        arch,
        config.batch_size,
        config.max_seq_length,
        config.gradient_checkpointing,
        is_lora = is_lora,
        attention_implementation = config.attention_implementation,
    )
    raw_gradient_bytes = compute_gradient_bytes(trainable_params)
    gradient_floor = int(model_weights * 0.15)
    if is_lora:
        gradient_floor = min(
            gradient_floor,
            max(activations_computed, optimizer_bytes),
        )
    gradient_bytes = max(raw_gradient_bytes, gradient_floor)
    activation_bytes = activations_computed

    return VramBreakdown(
        model_weights = model_weights,
        lora_adapters = lora_adapter_bytes,
        optimizer_states = optimizer_bytes,
        gradients = gradient_bytes,
        activations = activation_bytes,
        cuda_overhead = CUDA_OVERHEAD_BYTES,
        activations_computed = activations_computed,
    )
