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
CUDA_OVERHEAD_BYTES = int(1.4 * 1024**3)  # calibrated on RTX 5070 Ti

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

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
    num_dense_layers: int = 0
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None


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


@dataclass
class VramBreakdown:
    model_weights: int
    lora_adapters: int
    optimizer_states: int
    gradients: int
    activations: int
    cuda_overhead: int
    # The computed (formula-based) activation cost before floors.
    # This is the true per-layer cost that doesn't shard across GPUs.
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
        The computed activation cost does NOT shard (one GPU runs the layer).
        The floor portion (activations - computed) is overhead that shards.
        """
        shardable = (
            self.model_weights
            + self.lora_adapters
            + self.optimizer_states
            + self.gradients
            + (self.activations - self.activations_computed)  # floor overhead shards
        )
        per_gpu_fixed = self.activations_computed + self.cuda_overhead
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


def _compute_num_dense_layers(text_config, total_layers: int) -> int:
    """Count how many layers use dense MLP instead of MoE."""
    first_k = getattr(text_config, "first_k_dense_replace", None)
    if first_k is not None:
        return min(int(first_k), total_layers)

    sparse_step = getattr(text_config, "decoder_sparse_step", None)
    mlp_only = getattr(text_config, "mlp_only_layers", None) or []
    if sparse_step is not None and sparse_step > 0:
        mlp_only_set = set(mlp_only)
        moe_count = sum(
            1
            for i in range(total_layers)
            if i not in mlp_only_set and (i + 1) % sparse_step == 0
        )
        return total_layers - moe_count

    return 0


def extract_arch_config(hf_config) -> Optional[ModelArchConfig]:
    text_config = getattr(hf_config, "text_config", None) or hf_config

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

    num_experts = None
    for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        num_experts = getattr(text_config, attr, None)
        if num_experts is not None:
            break

    moe_intermediate = getattr(text_config, "moe_intermediate_size", None)
    n_shared_experts = getattr(text_config, "n_shared_experts", None) or 0

    num_dense_layers = 0
    if num_experts is not None and num_experts > 1:
        num_dense_layers = _compute_num_dense_layers(text_config, num_layers)

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
        num_dense_layers = num_dense_layers,
        q_lora_rank = q_lora_rank,
        kv_lora_rank = kv_lora_rank,
        qk_nope_head_dim = qk_nope_head_dim,
        qk_rope_head_dim = qk_rope_head_dim,
        v_head_dim = v_head_dim,
    )


def _get_kv_size(arch: ModelArchConfig) -> int:
    return (arch.hidden_size // arch.num_attention_heads) * arch.num_key_value_heads


def _get_mlp_size(arch: ModelArchConfig) -> int:
    if arch.moe_intermediate_size is not None:
        return arch.moe_intermediate_size
    return arch.intermediate_size


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
    return arch.hidden_size * arch.intermediate_size * 3


def _compute_moe_mlp_elements(arch: ModelArchConfig) -> int:
    hd = arch.hidden_size
    mlp_size = _get_mlp_size(arch)
    n_experts = _get_num_experts(arch)
    return hd * mlp_size * 3 * (n_experts + arch.n_shared_experts) + n_experts * hd


def _compute_layer_elements(arch: ModelArchConfig):
    """Return (total_quantizable, layernorms_per_layer, embed, lm_head) element counts.

    total_quantizable is summed across ALL layers (not per-layer).
    """
    hd = arch.hidden_size
    n_layers = arch.num_hidden_layers
    n_experts = _get_num_experts(arch)

    attn_total = _compute_attn_elements(arch) * n_layers

    if n_experts > 1:
        n_dense = arch.num_dense_layers
        n_moe = n_layers - n_dense
        mlp_total = (
            _compute_moe_mlp_elements(arch) * n_moe
            + _compute_dense_mlp_elements(arch) * n_dense
        )
    else:
        mlp_total = _compute_dense_mlp_elements(arch) * n_layers

    layernorms = 2 * hd
    embed_tokens = arch.vocab_size * hd
    lm_head = 0 if arch.tie_word_embeddings else arch.vocab_size * hd
    return attn_total + mlp_total, layernorms, embed_tokens, lm_head


def compute_model_weights_bytes(
    arch: ModelArchConfig,
    training_method: str,
    load_in_4bit: bool,
) -> int:
    total_quantizable, layernorms, embed_tokens, lm_head = _compute_layer_elements(arch)
    n_layers = arch.num_hidden_layers
    non_quantizable = layernorms * n_layers + embed_tokens + lm_head

    if training_method == "qlora" and load_in_4bit:
        return int(total_quantizable * 2 / QUANT_4BIT_FACTOR + non_quantizable * 2)

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
    hd = arch.hidden_size
    r = lora_rank
    n_layers = arch.num_hidden_layers
    n_experts = _get_num_experts(arch)

    attn_total = _lora_attn_elements(arch, r, target_modules) * n_layers

    if n_experts > 1:
        n_dense = arch.num_dense_layers
        n_moe = n_layers - n_dense
        # Include shared experts alongside routed experts
        moe_expert_mult = n_experts + arch.n_shared_experts
        moe_mlp = _lora_mlp_elements(
            hd,
            _get_mlp_size(arch),
            r,
            target_modules,
            moe_expert_mult,
        )
        dense_mlp = _lora_mlp_elements(
            hd,
            arch.intermediate_size,
            r,
            target_modules,
            1,
        )
        mlp_total = moe_mlp * n_moe + dense_mlp * n_dense
    else:
        mlp_total = (
            _lora_mlp_elements(
                hd,
                arch.intermediate_size,
                r,
                target_modules,
                1,
            )
            * n_layers
        )

    return attn_total + mlp_total


def compute_lora_adapter_bytes(lora_params: int) -> int:
    return lora_params * 2


def compute_optimizer_bytes(trainable_params: int, optimizer: str) -> int:
    optimizer_key = optimizer.lower().replace("-", "_")
    bytes_per_param = OPTIMIZER_BYTES_PER_PARAM.get(optimizer_key, 4)
    return trainable_params * bytes_per_param


def compute_gradient_bytes(trainable_params: int) -> int:
    return trainable_params * 2


def compute_activation_bytes(
    arch: ModelArchConfig,
    batch_size: int,
    seq_len: int,
    gradient_checkpointing: str,
    is_lora: bool = False,
) -> int:
    hd = arch.hidden_size
    kv_size = _get_kv_size(arch)
    mlp_size = _get_mlp_size(arch)
    bsz = batch_size
    n_layers = arch.num_hidden_layers

    activation_qkv = seq_len * bsz * (hd + kv_size + kv_size)
    residual_memory = (seq_len * bsz) * 2
    activation_mlp = seq_len * bsz * (mlp_size + mlp_size)

    per_layer_bytes = (activation_qkv + residual_memory + activation_mlp) * 2
    per_layer_bytes = int(per_layer_bytes * 1.25)

    gc_key = gradient_checkpointing.lower()
    gc_entry = GC_LAYER_MULTIPLIERS.get(gc_key, (None, None))
    full_ft_mult, lora_mult = gc_entry
    gc_multiplier = lora_mult if is_lora else full_ft_mult

    if gc_multiplier is None:
        effective_layers = n_layers
    else:
        effective_layers = gc_multiplier

    return int(per_layer_bytes * effective_layers)


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
    gradient_bytes = max(
        compute_gradient_bytes(trainable_params),
        int(model_weights * 0.15),
    )
    activations_computed = compute_activation_bytes(
        arch,
        config.batch_size,
        config.max_seq_length,
        config.gradient_checkpointing,
        is_lora = is_lora,
    )
    activation_bytes = max(
        activations_computed,
        int(model_weights * 0.15 * (config.batch_size / 2)),
    )

    return VramBreakdown(
        model_weights = model_weights,
        lora_adapters = lora_adapter_bytes,
        optimizer_states = optimizer_bytes,
        gradients = gradient_bytes,
        activations = activation_bytes,
        cuda_overhead = CUDA_OVERHEAD_BYTES,
        activations_computed = activations_computed,
    )
