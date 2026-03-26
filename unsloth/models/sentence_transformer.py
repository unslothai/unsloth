# Copyright 2025 electroglyph. All rights reserved.
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

from .loader import FastModel, DISABLE_SDPA_MODEL_NAMES
from ._utils import SUPPORTS_BFLOAT16
import inspect
import json
import os
import types
from huggingface_hub import hf_hub_download
from typing import Optional, NamedTuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_outputs import BaseModelOutput
from collections import OrderedDict
from transformers.models.distilbert import modeling_distilbert
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
import transformers
from packaging.version import Version
import re
from transformers import AutoModel, AutoConfig
from transformers.models.auto.auto_factory import _get_model_class
import tempfile
from huggingface_hub import HfApi, get_token
from ..save import (
    unsloth_save_pretrained_torchao,
    unsloth_save_pretrained_gguf,
    unsloth_push_to_hub_gguf,
)
import contextlib
import shutil

try:
    from ..kernels.layernorm import fast_layernorm

    _HAS_FAST_LAYERNORM = True
except ImportError:
    _HAS_FAST_LAYERNORM = False

try:
    from ..kernels.fused_pooling import fused_layernorm_mean_pool

    _HAS_FUSED_POOLING = True
except ImportError:
    _HAS_FUSED_POOLING = False


class EncoderSeqInfo(NamedTuple):
    seq_lengths: torch.Tensor  # (B,)
    cu_seqlens: torch.Tensor  # (B+1,)
    max_seqlen: int
    indices: torch.Tensor  # (total_tokens,)


def get_encoder_seq_info(attention_mask):
    """Build packed-sequence metadata from a (B, S) attention mask."""
    device = attention_mask.device
    seq_lengths = attention_mask.sum(dim = 1).to(dtype = torch.int32, device = device)

    cu_seqlens = torch.empty(
        seq_lengths.numel() + 1,
        dtype = torch.int32,
        device = device,
    )
    cu_seqlens[0] = 0
    torch.cumsum(seq_lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])

    max_seqlen = int(seq_lengths.max().item())
    indices = torch.nonzero(attention_mask.flatten(), as_tuple = False).squeeze(-1)

    return EncoderSeqInfo(seq_lengths, cu_seqlens, max_seqlen, indices)


def unpad_input(input_ids, seq_info, token_type_ids = None):
    """Remove padding tokens from a (B, S) batch."""
    unpadded_ids = input_ids.flatten()[seq_info.indices]
    unpadded_token_type_ids = None
    if token_type_ids is not None:
        unpadded_token_type_ids = token_type_ids.flatten()[seq_info.indices]
    return unpadded_ids, unpadded_token_type_ids


def pad_output(unpadded_output, seq_info, batch_size, max_seq_len):
    """Re-pad (total_tokens, D) back to (B, S, D)."""
    hidden_dim = unpadded_output.shape[-1]
    output = torch.zeros(
        batch_size * max_seq_len,
        hidden_dim,
        dtype = unpadded_output.dtype,
        device = unpadded_output.device,
    )
    output[seq_info.indices] = unpadded_output
    return output.view(batch_size, max_seq_len, hidden_dim)


class GuidedProjection(nn.Module):
    """Trainable projection after pooling for embedding space transformation."""

    def __init__(
        self,
        dim: int,
        output_dim: Optional[int] = None,
        use_bias: bool = False,
        use_residual: bool = True,
        init: str = "identity",
    ):
        super().__init__()
        output_dim = output_dim or dim
        self.dim = dim
        self.output_dim = output_dim
        self.use_residual = use_residual and (dim == output_dim)
        self._init_method = init

        self.proj = nn.Linear(dim, output_dim, bias = use_bias)

        if init == "identity" and dim == output_dim:
            nn.init.eye_(self.proj.weight)
        elif init == "orthogonal":
            nn.init.orthogonal_(self.proj.weight)
        else:
            nn.init.xavier_uniform_(self.proj.weight)

        if use_bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        if self.use_residual:
            projected = projected + x
        return F.normalize(projected, p = 2, dim = -1)

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        bias = self.proj.bias is not None
        return (
            f"dim={self.dim}, output_dim={self.output_dim}, "
            f"use_bias={bias}, use_residual={self.use_residual}, "
            f"init='{self._init_method}'"
        )


class GuidedProjectionPooling(nn.Module):
    """Pooling + GuidedProjection as a single ST pipeline module."""

    PROJECTION_WEIGHTS_NAME = "guided_projection.pt"
    PROJECTION_CONFIG_NAME = "guided_projection_config.json"

    def __init__(self, pooling_module: nn.Module, projection: GuidedProjection):
        super().__init__()
        self.pooling = pooling_module
        self.projection = projection

    def forward(self, features: dict) -> dict:
        features = self.pooling(features)
        if "sentence_embedding" in features:
            features["sentence_embedding"] = self.projection(
                features["sentence_embedding"]
            )
        return features

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pooling, name)

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        os.makedirs(output_path, exist_ok = True)

        if hasattr(self.pooling, "save"):
            self.pooling.save(output_path)

        torch.save(
            self.projection.state_dict(),
            os.path.join(output_path, self.PROJECTION_WEIGHTS_NAME),
        )

        config = {
            "dim": self.projection.dim,
            "output_dim": self.projection.output_dim,
            "use_bias": self.projection.proj.bias is not None,
            "use_residual": self.projection.use_residual,
            "init": self.projection._init_method,
        }
        with open(os.path.join(output_path, self.PROJECTION_CONFIG_NAME), "w") as f:
            json.dump(config, f, indent = 2)

    @classmethod
    def load(
        cls, input_path: str, pooling_module: nn.Module
    ) -> "GuidedProjectionPooling":
        config_path = os.path.join(input_path, cls.PROJECTION_CONFIG_NAME)
        weights_path = os.path.join(input_path, cls.PROJECTION_WEIGHTS_NAME)

        with open(config_path, "r") as f:
            config = json.load(f)

        projection = GuidedProjection(
            dim = config["dim"],
            output_dim = config["output_dim"],
            use_bias = config["use_bias"],
            use_residual = config["use_residual"],
            init = config["init"],
        )
        projection.load_state_dict(
            torch.load(weights_path, map_location = "cpu", weights_only = True)
        )

        return cls(pooling_module, projection)


def attach_guided_projection(st_model, dim = None, **kwargs):
    """Freeze encoder and attach a GuidedProjection after pooling."""
    if dim is None:
        encoder = None
        for mod in st_model:
            if hasattr(mod, "auto_model"):
                encoder = mod
                break

        if encoder is None:
            raise ValueError(
                "Could not locate a Transformer module. Specify `dim` explicitly."
            )

        config = encoder.auto_model.config
        dim = getattr(config, "hidden_size", None)
        if dim is None:
            raise ValueError(
                f"Could not infer hidden_size from {type(config).__name__}. "
                f"Specify `dim` explicitly."
            )

    for param in st_model.parameters():
        param.requires_grad = False

    projection = GuidedProjection(dim = dim, **kwargs)

    pooling_idx = None
    pooling_mod = None
    if hasattr(st_model, "_modules"):
        for key, mod in st_model._modules.items():
            if mod.__class__.__name__ == "Pooling":
                pooling_idx = key
                pooling_mod = mod
                break

    if pooling_mod is not None and pooling_idx is not None:
        wrapper = GuidedProjectionPooling(pooling_mod, projection)
        st_model._modules[pooling_idx] = wrapper
    else:
        import warnings

        warnings.warn(
            "Unsloth: No Pooling module found. GuidedProjection appended to module list.",
            stacklevel = 2,
        )
        next_key = str(len(st_model._modules))
        st_model._modules[next_key] = projection

    return projection


_VARLEN_ATTN_AVAILABLE = False
try:
    from torch.nn.attention.varlen import varlen_attn as _torch_varlen_attn

    # Requires Ampere+ (SM80); guard to avoid T4 error
    if torch.cuda.is_available():
        _major, _ = torch.cuda.get_device_capability()
        _VARLEN_ATTN_AVAILABLE = _major >= 8
    else:
        _VARLEN_ATTN_AVAILABLE = True  # CPU-only env, won't actually be used
except ImportError:
    pass


_FLASH_ATTN_VARLEN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    # Requires Ampere+
    if torch.cuda.is_available():
        _major, _ = torch.cuda.get_device_capability()
        _FLASH_ATTN_VARLEN_AVAILABLE = _major >= 8
    else:
        _FLASH_ATTN_VARLEN_AVAILABLE = True
except ImportError:
    pass

_XFORMERS_ATTN_AVAILABLE = False
_XFORMERS_DROPOUT_SAFE = True
try:
    from xformers.ops import (
        memory_efficient_attention as _xformers_memory_efficient_attention,
    )
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalMask as _XFormersBlockDiagonalMask,
    )

    _XFORMERS_ATTN_AVAILABLE = True
    # xformers dropout unsafe on pre-Ampere; use F.dropout instead
    if torch.cuda.is_available():
        _major, _ = torch.cuda.get_device_capability()
        _XFORMERS_DROPOUT_SAFE = _major >= 8
except ImportError:
    pass

_SM_MAJOR = 0
if torch.cuda.is_available():
    _SM_MAJOR, _ = torch.cuda.get_device_capability()


def _patch_encoder_layernorms(model):
    """Replace nn.LayerNorm with Triton kernel."""
    if not _HAS_FAST_LAYERNORM:
        return 0

    import torch.nn as nn

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward

            def make_fast_forward(ln_module):
                def _fast_forward(X):
                    return fast_layernorm(ln_module, X)

                return _fast_forward

            module.forward = make_fast_forward(module)
            count += 1

    return count


def _patch_encoder_attention_lora(model):
    """Fuse Q/K/V LoRA into single LoRA_QKV backward. Call after PEFT, before compile."""
    try:
        from ..kernels.fast_lora import LoRA_QKV
        from ..kernels.utils import get_lora_parameters
    except ImportError:
        return 0

    QKV_ATTRS = [
        ("query", "key", "value"),  # BERT, RoBERTa, XLM-RoBERTa, ALBERT, ELECTRA
        ("q", "k", "v"),  # MPNet
        ("q_lin", "k_lin", "v_lin"),  # DistilBERT
    ]

    count = 0
    for _name, module in model.named_modules():
        detected = None
        for q_attr, k_attr, v_attr in QKV_ATTRS:
            q_mod = getattr(module, q_attr, None)
            k_mod = getattr(module, k_attr, None)
            v_mod = getattr(module, v_attr, None)
            if q_mod is None or k_mod is None or v_mod is None:
                continue
            if (
                hasattr(q_mod, "lora_A")
                and hasattr(k_mod, "lora_A")
                and hasattr(v_mod, "lora_A")
            ):
                detected = (q_attr, k_attr, v_attr)
                break

        if detected is None:
            continue

        q_attr, k_attr, v_attr = detected
        q_mod = getattr(module, q_attr)
        k_mod = getattr(module, k_attr)
        v_mod = getattr(module, v_attr)

        q_mod._original_forward = q_mod.forward
        k_mod._original_forward = k_mod.forward
        v_mod._original_forward = v_mod.forward

        def _make_fused_forwards(attn_mod, qm, km, vm):
            def q_fused(x, *args, **kwargs):
                QW, QW_quant, QA, QB, QS = get_lora_parameters(qm)
                KW, KW_quant, KA, KB, KS = get_lora_parameters(km)
                VW, VW_quant, VA, VB, VS = get_lora_parameters(vm)

                Q, K, V = LoRA_QKV.apply(
                    x,
                    QW,
                    QW_quant,
                    QA,
                    QB,
                    QS,
                    KW,
                    KW_quant,
                    KA,
                    KB,
                    KS,
                    VW,
                    VW_quant,
                    VA,
                    VB,
                    VS,
                    False,
                )

                # LoRA_QKV doesn't handle bias
                q_bias = getattr(qm.base_layer, "bias", None)
                k_bias = getattr(km.base_layer, "bias", None)
                v_bias = getattr(vm.base_layer, "bias", None)
                if q_bias is not None:
                    Q = Q + q_bias
                if k_bias is not None:
                    K = K + k_bias
                if v_bias is not None:
                    V = V + v_bias

                attn_mod._fused_k = K
                attn_mod._fused_v = V
                return Q

            def k_fused(x, *args, **kwargs):
                cached = getattr(attn_mod, "_fused_k", None)
                if cached is not None:
                    del attn_mod._fused_k
                    return cached
                return km._original_forward(x, *args, **kwargs)

            def v_fused(x, *args, **kwargs):
                cached = getattr(attn_mod, "_fused_v", None)
                if cached is not None:
                    del attn_mod._fused_v
                    return cached
                return vm._original_forward(x, *args, **kwargs)

            return q_fused, k_fused, v_fused

        q_fwd, k_fwd, v_fwd = _make_fused_forwards(module, q_mod, k_mod, v_mod)
        q_mod.forward = q_fwd
        k_mod.forward = k_fwd
        v_mod.forward = v_fwd
        count += 1

    return count


def _check_sparsity_support():
    """Check if 2:4 sparsity is supported on this GPU."""
    if not torch.cuda.is_available():
        return (False, "CUDA is not available.")

    try:
        from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    except ImportError:
        return (False, "torch.sparse.SparseSemiStructuredTensor not available.")

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    if major < 8:
        return (False, f"No sparse tensor cores on sm_{sm} (requires sm_80+).")

    if major >= 12:
        return (
            False,
            f"Not beneficial on sm_{sm} (cuSPARSELt overhead at encoder dims).",
        )

    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        test_w = torch.zeros(32, 32, device = "cuda", dtype = torch.float16)
        test_w[:, 0::4] = 1.0
        test_w[:, 1::4] = 1.0
        _ = to_sparse_semi_structured(test_w)
    except Exception as e:
        return (False, f"CUTLASS not available on sm_{sm}: {e}")

    return (True, f"sm_{sm}, CUTLASS backend")


def _apply_sparsity_to_base_weights(peft_model, target_modules = None):
    """Apply 2:4 magnitude pruning to frozen base weights (not LoRA adapters)."""
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

    SparseSemiStructuredTensor._FORCE_CUTLASS = True

    if target_modules is None:
        target_modules = {
            "query",
            "key",
            "value",
            "dense",
            "q",
            "k",
            "v",
            "q_lin",
            "k_lin",
            "v_lin",
        }

    count = 0
    for name, module in peft_model.named_modules():
        base = getattr(module, "base_layer", None)
        if base is None or not isinstance(base, torch.nn.Linear):
            continue

        leaf_name = name.split(".")[-1]
        if leaf_name not in target_modules:
            continue

        w = base.weight.data
        if w.shape[0] % 4 != 0 or w.shape[1] % 4 != 0:
            continue

        # Keep top-2 per group of 4
        w_abs = w.detach().abs().view(w.shape[0], -1, 4)
        _, topk = w_abs.topk(2, dim = -1)
        mask = torch.zeros_like(w_abs, dtype = torch.bool)
        mask.scatter_(-1, topk, True)
        mask = mask.view(w.shape)
        w.mul_(mask)

        base._dense_weight = w.clone()
        _rg = base.weight.requires_grad
        base.weight = torch.nn.Parameter(
            to_sparse_semi_structured(w), requires_grad = _rg
        )
        count += 1

    return count


def _remove_sparsity_from_base_weights(peft_model):
    """Restore dense weights for saving/merging."""
    count = 0
    for _name, module in peft_model.named_modules():
        base = getattr(module, "base_layer", None)
        if base is None or not isinstance(base, torch.nn.Linear):
            continue
        if hasattr(base, "_dense_weight"):
            base.weight = torch.nn.Parameter(base._dense_weight, requires_grad = False)
            del base._dense_weight
            count += 1
    return count


def _patch_fused_pooling(st_model):
    """Fuse final LayerNorm + mean Pooling into single Triton kernel."""
    if not _HAS_FUSED_POOLING:
        return False

    transformer_mod = None
    pooling_mod = None
    for mod in st_model:
        if hasattr(mod, "auto_model"):
            transformer_mod = mod
        if mod.__class__.__name__ == "Pooling":
            pooling_mod = mod

    if transformer_mod is None or pooling_mod is None:
        return False

    if not getattr(pooling_mod, "pooling_mode_mean_tokens", False):
        return False

    if (
        getattr(pooling_mod, "pooling_mode_cls_token", False)
        or getattr(pooling_mod, "pooling_mode_max_tokens", False)
        or getattr(pooling_mod, "pooling_mode_mean_sqrt_len_tokens", False)
        or getattr(pooling_mod, "pooling_mode_weightedmean_tokens", False)
        or getattr(pooling_mod, "pooling_mode_lasttoken", False)
        or not getattr(pooling_mod, "include_prompt", True)
    ):
        return False

    # unwrap PEFT if needed
    inner = transformer_mod.auto_model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    base = inner
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model

    last_ln_name = None
    last_ln = None
    for name, module in base.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            last_ln_name = name
            last_ln = module

    if last_ln is None:
        return False

    stored_ln = last_ln

    # replace last LayerNorm with Identity
    parts = last_ln_name.split(".")
    parent = base
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], torch.nn.Identity())

    _original_pooling_forward = pooling_mod.forward
    pooling_mod._fused_ln = stored_ln
    pooling_mod._fused_ln_parent = parent
    pooling_mod._fused_ln_attr = parts[-1]
    pooling_mod._original_pooling_forward = _original_pooling_forward

    def _fused_pooling_forward(features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features.get(
            "attention_mask",
            torch.ones(
                token_embeddings.shape[:-1],
                device = token_embeddings.device,
                dtype = torch.int64,
            ),
        )

        pooled = fused_layernorm_mean_pool(stored_ln, token_embeddings, attention_mask)
        features["sentence_embedding"] = pooled
        return features

    pooling_mod._fused_pooling_forward = _fused_pooling_forward
    pooling_mod.forward = _fused_pooling_forward
    return True


@contextlib.contextmanager
def _restore_fused_pooling_ln(st_model):
    """Temporarily restore original LayerNorm for save operations."""
    pooling_mod = None
    for mod in st_model:
        if hasattr(mod, "_fused_ln"):
            pooling_mod = mod
            break
    if pooling_mod is None:
        yield
        return
    parent = pooling_mod._fused_ln_parent
    attr = pooling_mod._fused_ln_attr
    ln = pooling_mod._fused_ln
    identity = getattr(parent, attr)
    setattr(parent, attr, ln)
    old_fwd = pooling_mod.forward
    pooling_mod.forward = pooling_mod._original_pooling_forward
    try:
        yield
    finally:
        setattr(parent, attr, identity)
        pooling_mod.forward = old_fwd


# modernbert excluded — has native unpadding (indices, cu_seqlens args)
_UNPAD_SUPPORTED_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "albert",
    "electra",
    "mpnet",
    "distilbert",
}
_UNPAD_MIN_PADDING_RATIO = 0.15


def _register_varlen_attention():
    """Register unsloth_varlen in ALL_ATTENTION_FUNCTIONS (transformers 5.x)."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except (ImportError, AttributeError):
        return False

    if "unsloth_varlen" in ALL_ATTENTION_FUNCTIONS:
        return True

    def _unsloth_varlen_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        # Varlen metadata stored on config (PEFT wrappers reject unknown kwargs)
        _config = getattr(module, "config", None)
        cu_seqlens = getattr(_config, "_unsloth_cu_seqlens", None)
        if cu_seqlens is None:
            is_causal = (
                is_causal
                if is_causal is not None
                else getattr(module, "is_causal", False)
            )
            is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask = attention_mask,
                dropout_p = dropout,
                scale = scaling,
                is_causal = is_causal,
            )
            return attn_output.transpose(1, 2).contiguous(), None

        max_seqlen = _config._unsloth_max_seqlen
        seq_lengths = getattr(_config, "_unsloth_seq_lengths", None)

        if _FLASH_ATTN_VARLEN_AVAILABLE:
            q = query.squeeze(0).transpose(0, 1).contiguous()
            k = key.squeeze(0).transpose(0, 1).contiguous()
            v = value.squeeze(0).transpose(0, 1).contiguous()
            out = _flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q = cu_seqlens,
                cu_seqlens_k = cu_seqlens,
                max_seqlen_q = max_seqlen,
                max_seqlen_k = max_seqlen,
                causal = False,
                dropout_p = dropout,
                softmax_scale = scaling,
            )
            attn_output = out.transpose(0, 1).unsqueeze(0)
        elif _XFORMERS_ATTN_AVAILABLE:
            q_x = query.transpose(1, 2)
            k_x = key.transpose(1, 2)
            v_x = value.transpose(1, 2)
            attn_bias = _XFormersBlockDiagonalMask.from_seqlens(seq_lengths.tolist())
            if _XFORMERS_DROPOUT_SAFE:
                out_x = _xformers_memory_efficient_attention(
                    q_x,
                    k_x,
                    v_x,
                    attn_bias = attn_bias,
                    p = dropout,
                    scale = scaling,
                )
            else:
                out_x = _xformers_memory_efficient_attention(
                    q_x,
                    k_x,
                    v_x,
                    attn_bias = attn_bias,
                    p = 0.0,
                    scale = scaling,
                )
                if dropout > 0.0:
                    out_x = F.dropout(out_x, p = dropout, training = True)
            attn_output = out_x.transpose(1, 2)
        elif _VARLEN_ATTN_AVAILABLE:
            q = query.squeeze(0).transpose(0, 1).contiguous()
            k = key.squeeze(0).transpose(0, 1).contiguous()
            v = value.squeeze(0).transpose(0, 1).contiguous()
            out = _torch_varlen_attn(
                q,
                k,
                v,
                cu_seq_q = cu_seqlens,
                cu_seq_k = cu_seqlens,
                max_q = max_seqlen,
                max_k = max_seqlen,
                is_causal = False,
            )
            attn_output = out.transpose(0, 1).unsqueeze(0)
        else:
            # Bool mask fallback
            segment_ids = torch.repeat_interleave(
                torch.arange(seq_lengths.shape[0], device = query.device),
                seq_lengths.long(),
            )
            bool_mask = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
            bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)
            # GQA: repeat KV heads
            if key.shape[1] != query.shape[1]:
                n_rep = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(n_rep, dim = 1)
                value = value.repeat_interleave(n_rep, dim = 1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask = bool_mask,
                dropout_p = dropout,
                scale = scaling,
                is_causal = False,
            )

        return attn_output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("unsloth_varlen", _unsloth_varlen_attention)
    return True


_VARLEN_ATTN_REGISTERED = _register_varlen_attention()


def _patch_unpadded_encoder(st_model, model_type):
    """Patch Transformer forward for variable-length batching (unpadding)."""
    if model_type not in _UNPAD_SUPPORTED_TYPES:
        return False

    transformer_mod = None
    for mod in st_model:
        if hasattr(mod, "auto_model"):
            transformer_mod = mod
            break

    if transformer_mod is None:
        return False

    inner = transformer_mod.auto_model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    config = inner.config

    _orig_attn_impl = getattr(config, "_attn_implementation", "sdpa")

    # XLM-RoBERTa position_ids start at padding_idx + 1
    _position_offset = 0
    for mod in inner.modules():
        if hasattr(mod, "position_embeddings") and hasattr(
            mod.position_embeddings, "padding_idx"
        ):
            _pad_idx = mod.position_embeddings.padding_idx
            if _pad_idx is not None:
                _position_offset = _pad_idx + 1
            break

    _original_forward = transformer_mod.forward

    # Only use the ALL_ATTENTION_FUNCTIONS registry on transformers 5.x+.
    # On 4.x, BERT/RoBERTa bake their attention class at __init__ time,
    # so changing config._attn_implementation after construction has no effect.
    _use_attn_interface = (
        _VARLEN_ATTN_REGISTERED and Version(transformers.__version__).major >= 5
    )

    if not _use_attn_interface:
        # transformers 4.x: F.sdpa monkey-patching
        _use_varlen = (
            _FLASH_ATTN_VARLEN_AVAILABLE
            or _VARLEN_ATTN_AVAILABLE
            or _XFORMERS_ATTN_AVAILABLE
        )
        _original_sdpa = torch.nn.functional.scaled_dot_product_attention

        # NOTE: Thread-safety limitation (transformers 4.x path only).
        # The below closure monkey-patches the global F.scaled_dot_product_attention for the
        # duration of a forward pass. Two concurrent forward passes will race on the global.
        # The transformers 5.x path (ALL_ATTENTION_FUNCTIONS registry) does not have this issue.
        # Resolution: upgrade to transformers >=5.0, or use single-threaded DataLoader.
        def _varlen_sdpa(
            query,
            key,
            value,
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False,
            scale = None,
            **extra_kwargs,
        ):
            cu_seqlens = getattr(config, "_unsloth_cu_seqlens", None)
            if cu_seqlens is None:
                return _original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask = attn_mask,
                    dropout_p = dropout_p,
                    is_causal = is_causal,
                    scale = scale,
                    **extra_kwargs,
                )
            max_seqlen = config._unsloth_max_seqlen
            q = query.squeeze(0).transpose(0, 1).contiguous()
            k = key.squeeze(0).transpose(0, 1).contiguous()
            v = value.squeeze(0).transpose(0, 1).contiguous()
            if _FLASH_ATTN_VARLEN_AVAILABLE:
                out = _flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q = cu_seqlens,
                    cu_seqlens_k = cu_seqlens,
                    max_seqlen_q = max_seqlen,
                    max_seqlen_k = max_seqlen,
                    causal = is_causal,
                    dropout_p = dropout_p,
                    softmax_scale = scale,
                )
                return out.transpose(0, 1).unsqueeze(0)
            elif _XFORMERS_ATTN_AVAILABLE:
                seq_lengths = config._unsloth_seq_lengths
                q_x = query.transpose(1, 2)
                k_x = key.transpose(1, 2)
                v_x = value.transpose(1, 2)
                attn_bias = _XFormersBlockDiagonalMask.from_seqlens(
                    seq_lengths.tolist()
                )
                if _XFORMERS_DROPOUT_SAFE:
                    out_x = _xformers_memory_efficient_attention(
                        q_x,
                        k_x,
                        v_x,
                        attn_bias = attn_bias,
                        p = dropout_p,
                        scale = scale,
                    )
                else:
                    out_x = _xformers_memory_efficient_attention(
                        q_x,
                        k_x,
                        v_x,
                        attn_bias = attn_bias,
                        p = 0.0,
                        scale = scale,
                    )
                    if dropout_p > 0.0:
                        out_x = F.dropout(out_x, p = dropout_p, training = True)
                return out_x.transpose(1, 2)
            elif _VARLEN_ATTN_AVAILABLE:
                out = _torch_varlen_attn(
                    q,
                    k,
                    v,
                    cu_seq_q = cu_seqlens,
                    cu_seq_k = cu_seqlens,
                    max_q = max_seqlen,
                    max_k = max_seqlen,
                    is_causal = is_causal,
                )
                return out.transpose(0, 1).unsqueeze(0)
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask = attn_mask,
                dropout_p = dropout_p,
                is_causal = is_causal,
                scale = scale,
                **extra_kwargs,
            )

        def _bool_mask_sdpa(
            query,
            key,
            value,
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False,
            scale = None,
            **extra_kwargs,
        ):
            bool_mask = getattr(config, "_unsloth_bool_mask", None)
            if bool_mask is None:
                return _original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask = attn_mask,
                    dropout_p = dropout_p,
                    is_causal = is_causal,
                    scale = scale,
                    **extra_kwargs,
                )
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask = bool_mask,
                dropout_p = dropout_p,
                is_causal = False,
                scale = scale,
            )

        config._unsloth_cu_seqlens = None
        config._unsloth_max_seqlen = None
        config._unsloth_seq_lengths = None

    def _unpadded_forward(features, **kwargs):
        attention_mask = features.get("attention_mask")
        if attention_mask is None:
            return _original_forward(features, **kwargs)

        auto_model = transformer_mod.auto_model
        actual_model = (
            auto_model._orig_mod if hasattr(auto_model, "_orig_mod") else auto_model
        )
        if not actual_model.training:
            return _original_forward(features, **kwargs)

        # Skip when compiled (recompiles on every shape) or when gradient
        # checkpointing is active (config resets before checkpoint recompute)
        if hasattr(auto_model, "_orig_mod"):
            return _original_forward(features, **kwargs)
        if getattr(actual_model, "gradient_checkpointing", False):
            return _original_forward(features, **kwargs)

        B, S = attention_mask.shape
        device = attention_mask.device

        seq_info = get_encoder_seq_info(attention_mask)
        total_tokens = int(seq_info.cu_seqlens[-1].item())

        if total_tokens >= B * S * (1.0 - _UNPAD_MIN_PADDING_RATIO):
            return _original_forward(features, **kwargs)

        input_ids = features["input_ids"]
        packed_ids = input_ids.flatten()[seq_info.indices].unsqueeze(0)

        _offsets = torch.repeat_interleave(
            seq_info.cu_seqlens[:-1], seq_info.seq_lengths.long()
        )
        position_ids = (
            torch.arange(total_tokens, device = device) - _offsets + _position_offset
        ).unsqueeze(0)

        packed_features = {
            "input_ids": packed_ids,
            "position_ids": position_ids,
        }

        if "token_type_ids" in features and features["token_type_ids"] is not None:
            packed_features["token_type_ids"] = (
                features["token_type_ids"].flatten()[seq_info.indices].unsqueeze(0)
            )

        trans_features = {
            k: v
            for k, v in packed_features.items()
            if k in transformer_mod.model_forward_params
        }

        if _use_attn_interface:
            # Transformers 5.x: varlen via ALL_ATTENTION_FUNCTIONS
            config._attn_implementation = "unsloth_varlen"
            config._unsloth_cu_seqlens = seq_info.cu_seqlens
            config._unsloth_max_seqlen = seq_info.max_seqlen
            config._unsloth_seq_lengths = seq_info.seq_lengths
            try:
                outputs = auto_model(
                    **trans_features,
                    return_dict = True,
                    **kwargs,
                )
            finally:
                config._attn_implementation = _orig_attn_impl
                config._unsloth_cu_seqlens = None
                config._unsloth_max_seqlen = None
                config._unsloth_seq_lengths = None
        elif _use_varlen:
            # Transformers 4.x Tier 1: monkey-patch F.sdpa → varlen kernels
            config._attn_implementation = "sdpa"
            config._unsloth_cu_seqlens = seq_info.cu_seqlens
            config._unsloth_max_seqlen = seq_info.max_seqlen
            config._unsloth_seq_lengths = seq_info.seq_lengths
            torch.nn.functional.scaled_dot_product_attention = _varlen_sdpa
            try:
                outputs = auto_model(**trans_features, return_dict = True, **kwargs)
            finally:
                torch.nn.functional.scaled_dot_product_attention = _original_sdpa
                config._attn_implementation = _orig_attn_impl
                config._unsloth_cu_seqlens = None
                config._unsloth_max_seqlen = None
                config._unsloth_seq_lengths = None
        else:
            # Transformers 4.x Tier 2: bool mask SDPA fallback
            config._attn_implementation = "sdpa"
            segment_ids = torch.repeat_interleave(
                torch.arange(seq_info.seq_lengths.shape[0], device = device),
                seq_info.seq_lengths.long(),
            )
            bool_mask = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
            config._unsloth_bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)
            torch.nn.functional.scaled_dot_product_attention = _bool_mask_sdpa
            try:
                outputs = auto_model(**trans_features, return_dict = True, **kwargs)
            finally:
                torch.nn.functional.scaled_dot_product_attention = _original_sdpa
                config._attn_implementation = _orig_attn_impl
                config._unsloth_bool_mask = None

        packed_embeddings = outputs[0].squeeze(0)  # (total_tokens, D)

        token_embeddings = pad_output(packed_embeddings, seq_info, B, S)
        features["token_embeddings"] = token_embeddings
        features["attention_mask"] = attention_mask

        return features

    transformer_mod.forward = _unpadded_forward
    transformer_mod._original_forward = _original_forward
    transformer_mod._unpadding_active = True

    if _use_attn_interface:
        backend_name = "attn_interface"
    elif _FLASH_ATTN_VARLEN_AVAILABLE:
        backend_name = "flash_attn_varlen"
    elif _XFORMERS_ATTN_AVAILABLE:
        backend_name = "xformers"
    elif _VARLEN_ATTN_AVAILABLE:
        backend_name = "torch_varlen"
    else:
        backend_name = "bool_mask_sdpa"
    transformer_mod._unpadding_backend = backend_name
    return True


def _patch_unpadded_decoder(st_model):
    """Patch Transformer forward for variable-length batching on causal decoders."""
    transformer_mod = None
    for mod in st_model:
        if hasattr(mod, "auto_model"):
            transformer_mod = mod
            break

    if transformer_mod is None:
        return False

    if hasattr(transformer_mod, "model_forward_params"):
        transformer_mod.model_forward_params.add("packed_seq_lengths")

    _original_forward = transformer_mod.forward

    def _unpadded_forward(features, **kwargs):
        attention_mask = features.get("attention_mask")
        if attention_mask is None:
            return _original_forward(features, **kwargs)

        auto_model = transformer_mod.auto_model
        actual_model = (
            auto_model._orig_mod if hasattr(auto_model, "_orig_mod") else auto_model
        )
        if not actual_model.training:
            return _original_forward(features, **kwargs)

        B, S = attention_mask.shape
        device = attention_mask.device

        seq_info = get_encoder_seq_info(attention_mask)
        total_tokens = int(seq_info.cu_seqlens[-1].item())

        if total_tokens >= B * S * (1.0 - _UNPAD_MIN_PADDING_RATIO):
            return _original_forward(features, **kwargs)

        input_ids = features["input_ids"]
        packed_ids = input_ids.flatten()[seq_info.indices].unsqueeze(0)

        _offsets = torch.repeat_interleave(
            seq_info.cu_seqlens[:-1], seq_info.seq_lengths.long()
        )
        position_ids = (torch.arange(total_tokens, device = device) - _offsets).unsqueeze(
            0
        )

        packed_features = {
            "input_ids": packed_ids,
            "position_ids": position_ids,
            "packed_seq_lengths": seq_info.seq_lengths,
        }

        if "token_type_ids" in features and features["token_type_ids"] is not None:
            packed_features["token_type_ids"] = (
                features["token_type_ids"].flatten()[seq_info.indices].unsqueeze(0)
            )

        trans_features = {
            k: v
            for k, v in packed_features.items()
            if k in transformer_mod.model_forward_params
        }

        outputs = auto_model(**trans_features, return_dict = True, **kwargs)
        packed_embeddings = outputs[0].squeeze(0)  # (total_tokens, D)

        token_embeddings = pad_output(packed_embeddings, seq_info, B, S)
        features["token_embeddings"] = token_embeddings
        features["attention_mask"] = attention_mask
        return features

    transformer_mod.forward = _unpadded_forward
    transformer_mod._original_forward = _original_forward
    transformer_mod._unpadding_active = True
    transformer_mod._unpadding_backend = "native_packing"
    return True


_POOLING_PATCHED = False


def _patch_efficient_pooling():
    """Monkey-patch Pooling to skip redundant expand()."""
    global _POOLING_PATCHED
    if _POOLING_PATCHED:
        return
    _POOLING_PATCHED = True

    try:
        from sentence_transformers.models import Pooling

        _original_forward = Pooling.forward

        def _efficient_forward(self, features):
            token_embeddings = features["token_embeddings"]
            attention_mask = features.get(
                "attention_mask",
                torch.ones(
                    token_embeddings.shape[:-1],
                    device = token_embeddings.device,
                    dtype = torch.int64,
                ),
            )

            if not self.include_prompt and "prompt_length" in features:
                prompt_length = features["prompt_length"]
                if isinstance(prompt_length, torch.Tensor):
                    prompt_length = int(prompt_length[0].item())
                attention_mask = attention_mask.clone()
                # Handle left-padded sequences
                pad_lengths = (attention_mask == 0).to(torch.int32).argmin(dim = 1)
                for i in range(attention_mask.shape[0]):
                    start = int(pad_lengths[i].item())
                    attention_mask[i, start : start + prompt_length] = 0

            output_vectors = []

            if self.pooling_mode_cls_token:
                cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])
                output_vectors.append(cls_token)

            if self.pooling_mode_max_tokens:
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .to(token_embeddings.dtype)
                )
                token_embeddings[input_mask_expanded == 0] = -1e9
                max_over_time = torch.max(token_embeddings, 1)[0]
                output_vectors.append(max_over_time)

            if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
                mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
                sum_embeddings = (token_embeddings * mask).sum(dim = 1)

                if "token_weights_sum" in features:
                    sum_mask = (
                        features["token_weights_sum"]
                        .unsqueeze(-1)
                        .expand(sum_embeddings.size())
                    )
                else:
                    sum_mask = mask.sum(dim = 1)

                sum_mask = torch.clamp(sum_mask, min = 1e-9)

                if self.pooling_mode_mean_tokens:
                    output_vectors.append(sum_embeddings / sum_mask)
                if self.pooling_mode_mean_sqrt_len_tokens:
                    output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

            if self.pooling_mode_weightedmean_tokens:
                return _original_forward(self, features)

            if self.pooling_mode_lasttoken:
                return _original_forward(self, features)

            output_vector = torch.cat(output_vectors, 1)
            features["sentence_embedding"] = output_vector
            return features

        Pooling.forward = _efficient_forward
    except Exception as e:
        import warnings

        warnings.warn(f"Unsloth: Failed to patch Pooling: {e}", stacklevel = 2)


_MNRL_PATCHED = False


def _patch_mnrl_loss():
    """Monkey-patch MNRL with fused chunked contrastive loss."""
    global _MNRL_PATCHED
    if _MNRL_PATCHED:
        return
    _MNRL_PATCHED = True

    try:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from ..kernels.contrastive_loss import FusedContrastiveLoss

        _original_forward = MultipleNegativesRankingLoss.forward

        def _fused_forward(self, sentence_features, labels = None):
            first_ids = sentence_features[0].get("input_ids")
            if first_ids is not None and first_ids.shape[0] < 8:
                return _original_forward(self, sentence_features, labels)

            # Fall back for non-default MNRL configurations
            if (
                getattr(self, "gather_across_devices", False)
                or getattr(self, "directions", None) not in (None, ("query_to_doc",))
                or getattr(self, "partition_mode", None)
                not in (None, "disabled", "joint")
                or getattr(self, "hardness_mode", None) is not None
            ):
                return _original_forward(self, sentence_features, labels)

            reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
            embeddings_a = reps[0]
            embeddings_b = torch.cat(reps[1:], dim = 0)

            try:
                from sentence_transformers.util import cos_sim

                is_cosine = self.similarity_fct is cos_sim
            except ImportError:
                is_cosine = True

            if is_cosine:
                embeddings_a = torch.nn.functional.normalize(embeddings_a, p = 2, dim = 1)
                embeddings_b = torch.nn.functional.normalize(embeddings_b, p = 2, dim = 1)

            return FusedContrastiveLoss.apply(embeddings_a, embeddings_b, self.scale)

        MultipleNegativesRankingLoss.forward = _fused_forward
        MultipleNegativesRankingLoss._original_forward = _original_forward
        print(
            "Unsloth: Patched MultipleNegativesRankingLoss with fused contrastive loss"
        )
    except Exception as e:
        import warnings

        warnings.warn(
            f"Unsloth: Failed to patch MultipleNegativesRankingLoss: {e}", stacklevel = 2
        )


def _save_pretrained_torchao(
    self,
    save_directory,
    tokenizer = None,
    torchao_config = None,
    push_to_hub = False,
    token = None,
):
    with _restore_fused_pooling_ln(self):
        self.save_pretrained(save_directory)

    # grab inner model
    inner_model = self[0].auto_model
    if hasattr(inner_model, "_orig_mod"):
        inner_model = inner_model._orig_mod

    # merge LoRA first
    if hasattr(inner_model, "merge_and_unload"):
        inner_model = inner_model.merge_and_unload()

    # confirm Transformer path
    transformer_path = "0_Transformer"
    modules_path = os.path.join(save_directory, "modules.json")
    if os.path.exists(modules_path):
        try:
            with open(modules_path, "r") as f:
                modules = json.load(f)
            for m in modules:
                if m.get("type", "").endswith("Transformer"):
                    transformer_path = m.get("path", "")
                    break
        except:
            pass

    transformer_dir = os.path.join(save_directory, transformer_path)
    transformer_dir = os.path.abspath(transformer_dir)

    if tokenizer is None:
        tokenizer = self.tokenizer

    @contextlib.contextmanager
    def patch_unsloth_save():
        original_causal = transformers.AutoModelForCausalLM
        original_rmtree = shutil.rmtree
        # unsloth_save_pretrained_torchao expects AutoModelForCausalLM
        transformers.AutoModelForCausalLM = transformers.AutoModel
        # prevent unsloth from deleting the unquantized model directory
        shutil.rmtree = lambda *args, **kwargs: None
        try:
            yield
        finally:
            # unpatch
            transformers.AutoModelForCausalLM = original_causal
            shutil.rmtree = original_rmtree

    with patch_unsloth_save():
        unsloth_save_pretrained_torchao(
            inner_model,
            transformer_dir,
            tokenizer = tokenizer,
            torchao_config = torchao_config,
            push_to_hub = push_to_hub,
            token = token,
        )

    torchao_dir = transformer_dir + "-torchao"
    if os.path.exists(torchao_dir):
        if not os.path.exists(transformer_dir):
            os.makedirs(transformer_dir, exist_ok = True)

        # move contents
        for item in os.listdir(torchao_dir):
            s = os.path.join(torchao_dir, item)
            d = os.path.join(transformer_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok = True)
            else:
                shutil.copy2(s, d)

        # remove torchao dir
        shutil.rmtree(torchao_dir)

        # remove conflicting safetensors if we brought in bin
        if os.path.exists(os.path.join(transformer_dir, "pytorch_model.bin")):
            safetensors_path = os.path.join(transformer_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                try:
                    os.remove(safetensors_path)
                except:
                    pass

    try:
        FastSentenceTransformer._add_unsloth_branding(save_directory)
    except:
        pass


class FastSentenceTransformer(FastModel):
    @staticmethod
    def _read_pooling_mode(model_name, token):
        """
        Read the pooling mode from the modules.json file if it exists, otherwise return "mean".
        """
        try:
            if os.path.exists(model_name) and os.path.exists(
                os.path.join(model_name, "modules.json")
            ):
                modules_json_path = os.path.join(model_name, "modules.json")
            else:
                modules_json_path = hf_hub_download(
                    model_name, "modules.json", token = token
                )

            with open(modules_json_path, "r") as f:
                modules_config = json.load(f)

            pooling_config_path = None
            for module in modules_config:
                if module.get("type", "") == "sentence_transformers.models.Pooling":
                    pooling_path = module.get("path", "")
                    if pooling_path:
                        # try to find config.json for pooling module
                        if os.path.exists(model_name) and os.path.exists(
                            os.path.join(model_name, pooling_path, "config.json")
                        ):
                            pooling_config_path = os.path.join(
                                model_name, pooling_path, "config.json"
                            )
                        else:
                            pooling_config_path = hf_hub_download(
                                model_name,
                                os.path.join(pooling_path, "config.json"),
                                token = token,
                            )
                        break

            if pooling_config_path:
                with open(pooling_config_path, "r") as f:
                    pooling_config = json.load(f)
                    # from here:
                    # https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/models/Pooling.py#L43
                    pooling_map = {
                        "pooling_mode_cls_token": "cls",
                        "pooling_mode_mean_tokens": "mean",
                        "pooling_mode_max_tokens": "max",
                        "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
                        "pooling_mode_weightedmean_tokens": "weightedmean",
                        "pooling_mode_lasttoken": "lasttoken",
                    }
                    for config_key, mode in pooling_map.items():
                        if pooling_config.get(config_key):
                            if mode != "mean":
                                print(f"Pooling mode detected as {mode}, updating...")
                            return mode

        except Exception as e:
            print(
                f"Failed to detect pooling mode, not a sentence-transformers model. Using default pooling mode 'mean', this may or may not work."
            )
            return "mean"

    @staticmethod
    def _patch_mpnet_v4():
        """
        Patch the MPNetModel to support gradient checkpointing.
        Supports transformers 4.
        """
        from transformers.models.mpnet import modeling_mpnet

        # add supports_gradient_checkpointing flag
        modeling_mpnet.MPNetModel.supports_gradient_checkpointing = True

        # add _set_gradient_checkpointing method
        def _set_gradient_checkpointing(self, module = None, value = True):
            if module is None:
                module = self.encoder
            if isinstance(module, modeling_mpnet.MPNetEncoder):
                module.gradient_checkpointing = value

        modeling_mpnet.MPNetModel._set_gradient_checkpointing = (
            _set_gradient_checkpointing
        )

        # patch MPNetEncoder.forward to support checkpointing
        # based on:
        # https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mpnet/modeling_mpnet.py#L321
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False,
            **kwargs,
        ):
            position_bias = self.compute_position_bias(hidden_states)
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # do gradient checkpointing if enabled and training
                if getattr(self, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions = output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        head_mask[i] if head_mask is not None else None,
                        position_bias,
                        use_reentrant = True,  # fix for torch 2.9
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        head_mask[i] if head_mask is not None else None,
                        position_bias,
                        output_attentions = output_attentions,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, all_hidden_states, all_attentions]
                    if v is not None
                )
            return BaseModelOutput(
                last_hidden_state = hidden_states,
                hidden_states = all_hidden_states,
                attentions = all_attentions,
            )

        # assign the patched forward
        modeling_mpnet.MPNetEncoder.forward = forward

    @staticmethod
    def _patch_mpnet_v5():
        """
        Patch the MPNetModel to support gradient checkpointing.
        Supports transformers 5.
        """
        from transformers.models.mpnet import modeling_mpnet

        # add supports_gradient_checkpointing flag
        modeling_mpnet.MPNetModel.supports_gradient_checkpointing = True

        # add _set_gradient_checkpointing method
        def _set_gradient_checkpointing(self, module = None, value = True):
            if module is None:
                module = self.encoder
            if isinstance(module, modeling_mpnet.MPNetEncoder):
                module.gradient_checkpointing = value

        modeling_mpnet.MPNetModel._set_gradient_checkpointing = (
            _set_gradient_checkpointing
        )

        # patch MPNetEncoder.forward to support checkpointing
        # based on:
        # https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/mpnet/modeling_mpnet.py#L284
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False,
            **kwargs,
        ):
            position_bias = self.compute_position_bias(hidden_states)
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # do gradient checkpointing if enabled and training
                if getattr(self, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions = output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        position_bias,
                        use_reentrant = True,  # required for torch >= 2.9
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        position_bias,
                        output_attentions,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, all_hidden_states, all_attentions]
                    if v is not None
                )
            return BaseModelOutput(
                last_hidden_state = hidden_states,
                hidden_states = all_hidden_states,
                attentions = all_attentions,
            )

        modeling_mpnet.MPNetEncoder.forward = forward

    @staticmethod
    def _patch_distilbert_v4():
        """
        Patch DistilBert forward to use positional args (for PEFT compatibility).
        Transformers 4 version.
        """

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time"
                )
            elif input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            head_mask_is_none = head_mask is None
            # Prepare head mask if needed
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embeddings = self.embeddings(
                input_ids, inputs_embeds
            )  # (bs, seq_length, dim)

            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = (
                    attention_mask
                    if (attention_mask is not None and 0 in attention_mask)
                    else None
                )
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(
                        input_shape, device = device
                    )  # (bs, seq_length)

                if (
                    self.config._attn_implementation == "sdpa"
                    and head_mask_is_none
                    and not output_attentions
                ):
                    attention_mask = _prepare_4d_attention_mask_for_sdpa(
                        attention_mask, embeddings.dtype, tgt_len = input_shape[1]
                    )
            return self.transformer(
                embeddings,
                attention_mask,
                head_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        modeling_distilbert.DistilBertModel.forward = forward

    @staticmethod
    def _has_add_pooling_layer(config, auto_model_class = None):
        """
        Checks if the model class supports the `add_pooling_layer` argument
        """
        try:
            if auto_model_class is None:
                auto_model_class = AutoModel
            # try to resolve the class
            model_class = _get_model_class(config, auto_model_class._model_mapping)

            if model_class:
                sig = inspect.signature(model_class.__init__)
                return "add_pooling_layer" in sig.parameters
        except:
            pass

        return False

    @staticmethod
    def _patch_distilbert_v5():
        """
        Patch DistilBert forward to use positional args (for PEFT compatibility).
        Transformers 5 version.
        """
        from transformers.masking_utils import create_bidirectional_mask

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You must specify exactly one of input_ids or inputs_embeds"
                )

            embeddings = self.embeddings(input_ids, inputs_embeds, position_ids)

            attention_mask = create_bidirectional_mask(
                config = self.config,
                input_embeds = embeddings,
                attention_mask = attention_mask,
            )

            return self.transformer(
                embeddings,
                attention_mask,
                **kwargs,
            )

        modeling_distilbert.DistilBertModel.forward = forward

    @staticmethod
    def _add_unsloth_tags(repo_id, token, tags = None):
        """
        Add Unsloth and sentence-transformers tags to the Hugging Face Hub repository.
        """
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        if tags is None:
            tags = []
        tags.extend(["unsloth", "sentence-transformers"])
        try:
            api.add_tags(
                repo_id = repo_id,
                tags = tags,
                repo_type = "model",
            )
        except:
            pass

    @staticmethod
    def _add_unsloth_branding(save_directory):
        """
        Add Unsloth branding to the README.md file generated by sentence-transformers.
        """
        readme_path = os.path.join(save_directory, "README.md")
        if not os.path.exists(readme_path):
            return

        with open(readme_path, "r", encoding = "utf-8") as f:
            content = f.read()

        # add unsloth tag to frontmatter
        if "---\ntags:\n" in content:
            content = content.replace("---\ntags:\n", "---\ntags:\n- unsloth\n")
        else:
            # if tags exist but not right at start, use regex to append
            pattern = r"(^tags:\s*\n)"
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(
                    pattern, r"\1- unsloth\n", content, count = 1, flags = re.MULTILINE
                )

        # add branding badge and text
        branding = (
            "\n\nThis model was finetuned with [Unsloth](https://github.com/unslothai/unsloth).\n\n"
            '[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)\n'
        )

        # add to description
        if "# SentenceTransformer" in content:
            parts = content.split("# SentenceTransformer", 1)
            content = parts[0] + "# SentenceTransformer" + branding + parts[1]
        else:
            content += branding

        with open(readme_path, "w", encoding = "utf-8") as f:
            f.write(content)

    @staticmethod
    def _module_path(model_name, token = None):
        """
        Returns the path to the modules.json file or None
        """
        try:
            if os.path.exists(model_name) and os.path.isdir(model_name):
                path = os.path.join(model_name, "modules.json")
                return path if os.path.exists(path) else None
            else:
                try:
                    return hf_hub_download(model_name, "modules.json", token = token)
                except:
                    return None
        except:
            return None

    @staticmethod
    def _create_transformer_module(
        model_name,
        model,
        tokenizer,
        max_seq_length,
        trust_remote_code,
    ):
        """Helper to create and configure a Transformer module."""
        from sentence_transformers.models import Transformer

        original_from_pretrained = AutoModel.from_pretrained

        def return_existing_model(*args, **kwargs):
            return model

        try:
            # Temporarily redirect AutoModel loading to return our pre-loaded model
            AutoModel.from_pretrained = return_existing_model

            # Initialize Transformer
            transformer_module = Transformer(
                model_name,
                max_seq_length = max_seq_length,
                model_args = {"trust_remote_code": trust_remote_code},
                config_args = {"trust_remote_code": trust_remote_code},
            )
        finally:
            # Restore original functionality immediately
            AutoModel.from_pretrained = original_from_pretrained

        transformer_module.tokenizer = tokenizer
        transformer_module.do_lower_case = getattr(tokenizer, "do_lower_case", False)

        # sentence-transformers only passes along known keys to model.forward
        model_forward_params = list(inspect.signature(model.forward).parameters)
        transformer_module.model_forward_params = set(model_forward_params) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
        }

        # determine max_seq_length if not provided
        if max_seq_length is None:
            if hasattr(model, "config") and hasattr(
                model.config, "max_position_embeddings"
            ):
                max_seq_length = model.config.max_position_embeddings
            elif hasattr(tokenizer, "model_max_length"):
                max_seq_length = tokenizer.model_max_length
            else:
                max_seq_length = 512

        transformer_module.max_seq_length = max_seq_length
        transformer_module.config_keys = ["max_seq_length", "do_lower_case"]
        transformer_module.save_in_root = True

        if hasattr(model, "config"):
            model.config.tokenizer_class = tokenizer.__class__.__name__

        return transformer_module

    @staticmethod
    def _load_modules(
        model_name,
        token,
        model,
        tokenizer,
        max_seq_length,
        pooling_mode,
        trust_remote_code = False,
    ) -> tuple[OrderedDict, bool]:
        """
        Load modules from modules.json if available, otherwise fallback to hard-coded modules.

        Returns:
            tuple[OrderedDict, bool]: (modules, no_modules_json)
        """
        from sentence_transformers.util import import_from_string, load_dir_path
        from sentence_transformers.models import Pooling, Normalize

        modules = OrderedDict()
        modules_json_path = FastSentenceTransformer._module_path(model_name, token)

        if modules_json_path:
            with open(modules_json_path, encoding = "utf8") as f:
                modules_config = json.load(f)

            for module_config in modules_config:
                class_ref = module_config["type"]
                name = module_config.get(
                    "name", str(module_config.get("idx", len(modules)))
                )

                if class_ref == "sentence_transformers.models.Transformer":
                    transformer_module = (
                        FastSentenceTransformer._create_transformer_module(
                            model_name,
                            model,
                            tokenizer,
                            max_seq_length,
                            trust_remote_code,
                        )
                    )
                    modules[name] = transformer_module
                else:
                    # load other modules (Pooling, Normalize, etc.)
                    module_path = module_config["path"]
                    if os.path.isdir(model_name):
                        load_path = os.path.join(model_name, module_path)
                    else:
                        try:
                            load_path = load_dir_path(
                                model_name, module_path, token = token
                            )
                        except Exception as e:
                            print(
                                f"Unsloth Warning: Could not download module {module_path}: {e}"
                            )
                            continue

                    module_class = import_from_string(class_ref)
                    try:
                        module = module_class.load(load_path)
                        modules[name] = module
                    except Exception as e:
                        print(
                            f"Unsloth Warning: Failed to load module {name} ({class_ref}): {e}"
                        )

            return modules, False

        # fallback if no modules.json (non sentence-transformers models)
        print(
            "Unsloth: No modules.json found, falling back to [Transformer, Pooling, Normalize]. This may or may not work."
        )

        transformer_module = FastSentenceTransformer._create_transformer_module(
            model_name, model, tokenizer, max_seq_length, trust_remote_code
        )
        modules["0"] = transformer_module

        hidden_size = getattr(model.config, "hidden_size", 768)

        if pooling_mode == "mean":
            pooling_mode = FastSentenceTransformer._read_pooling_mode(model_name, token)

        modules["1"] = Pooling(
            word_embedding_dimension = hidden_size, pooling_mode = pooling_mode
        )
        modules["2"] = Normalize()

        return modules, True

    # Encoder model types that benefit from native torch.compile instead of Unsloth patching
    ENCODER_MODEL_TYPES = {
        "mpnet",
        "bert",
        "distilbert",
        "modernbert",
        "roberta",
        "xlm-roberta",
        "albert",
        "electra",
    }

    @staticmethod
    def _estimate_compile_threshold(
        model,
        batch_size = None,
        grad_accum = None,
        max_seq_length = None,
    ):
        """Estimate min training steps for torch.compile to be beneficial."""
        if hasattr(model, "__getitem__"):
            try:
                inner = model[0].auto_model
                params = sum(p.numel() for p in inner.parameters())
            except:
                params = 100_000_000  # Default to 100M if can't determine
        else:
            params = sum(p.numel() for p in model.parameters())

        model_type = None
        try:
            if "inner" in locals():
                model_type = getattr(getattr(inner, "config", None), "model_type", None)
        except Exception:
            model_type = None
        if isinstance(model_type, str):
            model_type = model_type.lower()

        params_m = params / 1e6

        if params_m < 50:
            estimated_warmup = 35 + params_m * 0.3
            base_speedup = 1.35
        elif params_m < 200:
            estimated_warmup = 12 + params_m * 0.03
            base_speedup = 1.75
        else:
            estimated_warmup = 15 + params_m * 0.04
            base_speedup = 1.60

        naive_ms = 50 + params_m * 1.0
        compiled_ms = naive_ms / base_speedup
        time_saved_per_step_s = (naive_ms - compiled_ms) / 1000

        if time_saved_per_step_s > 0:
            breakeven = estimated_warmup / time_saved_per_step_s
        else:
            breakeven = float("inf")

        threshold = breakeven * 1.2

        generic_scale = 1.0
        fast_scale = 1.0
        if (
            batch_size is not None
            or grad_accum is not None
            or max_seq_length is not None
        ):
            try:
                bs = int(batch_size) if batch_size is not None else 2
                ga = int(grad_accum) if grad_accum is not None else 4
                seq = int(max_seq_length) if max_seq_length is not None else 512
            except Exception:
                bs, ga, seq = 2, 4, 512

            bs = max(1, bs)
            ga = max(1, ga)
            seq = max(64, min(seq, 8192))

            ref_bs, ref_ga, ref_seq = 2, 4, 512

            ga_scale = (ref_ga / ga) ** 1.0
            bs_seq_scale = ((ref_bs * ref_seq) / (bs * seq)) ** 0.15
            generic_scale = 0.35 * ga_scale * bs_seq_scale
            generic_scale = max(0.05, min(generic_scale, 5.0))

            fast_ga_scale = (ref_ga / ga) ** 1.5
            fast_bs_seq_scale = ((ref_bs * ref_seq) / (bs * seq)) ** 0.25
            fast_scale = 0.2 * fast_ga_scale * fast_bs_seq_scale
            fast_scale = max(0.05, min(fast_scale, 5.0))

        generic_threshold = threshold * generic_scale * 1.25

        is_fast_type = (
            isinstance(model_type, str)
            and model_type in FastSentenceTransformer.ENCODER_MODEL_TYPES
        )
        if is_fast_type:
            fast_threshold = threshold * fast_scale * 1.5
            final_threshold = min(generic_threshold, fast_threshold)
        else:
            final_threshold = generic_threshold

        if model_type == "mpnet":
            final_threshold *= 0.7

        return int(max(20, final_threshold))

    @staticmethod
    def _apply_torch_compile(model, mode = "default"):
        """Apply torch.compile to a SentenceTransformer model."""
        if hasattr(model, "__getitem__"):
            inner_model = model[0].auto_model
            compiled = torch.compile(inner_model, mode = mode)
            model[0].auto_model = compiled
            model.__dict__["_orig_mod"] = model
        else:
            model = torch.compile(model, mode = mode)
        return model

    @staticmethod
    def from_pretrained(
        model_name,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = False,  # Changed default: 4-bit is slow for encoders
        load_in_8bit = False,
        load_in_16bit = True,  # Changed default: 16-bit is optimal for encoders
        full_finetuning = False,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        trust_remote_code = False,
        use_gradient_checkpointing = False,  # Changed default: conflicts with torch.compile
        resize_model_vocab = None,
        revision = None,
        use_exact_model_name = False,
        offload_embedding = False,
        random_state = 3407,
        max_lora_rank = 64,
        disable_log_stats = True,
        qat_scheme = None,
        unsloth_tiled_mlp = False,
        pooling_mode = "mean",
        for_inference = False,
        use_guided_projection = False,
        **kwargs,
    ):
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Transformer, Pooling, Normalize
        except ImportError:
            raise ImportError(
                "Unsloth: To use `FastSentenceTransformer`, you must install `sentence-transformers`.\n"
                "Run `pip install sentence-transformers` to install it."
            )

        if for_inference:
            st_device = device_map
            if isinstance(st_device, dict) or (
                isinstance(st_device, str) and st_device in ["auto", "sequential"]
            ):
                st_device = None

            model_kwargs = kwargs.get("model_kwargs", {})
            model_kwargs["dtype"] = dtype if dtype is not None else "auto"

            st_kwargs = {
                "device": st_device,
                "trust_remote_code": trust_remote_code,
                "token": token,
                "revision": revision,
                "model_kwargs": model_kwargs,
            }

            known_keys = [
                "cache_folder",
                "truncate_dim",
                "tokenizer_kwargs",
                "config_kwargs",
            ]
            for k in known_keys:
                if k in kwargs:
                    st_kwargs[k] = kwargs[k]

            st_model = SentenceTransformer(model_name, **st_kwargs)
            return st_model

        if full_finetuning and (load_in_4bit or load_in_8bit):
            print(
                "Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA."
            )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_16bit = False

        if int(load_in_4bit) + int(load_in_8bit) + int(load_in_16bit) >= 2:
            raise RuntimeError(
                "Unsloth: Can only load in 4bit or 8bit or 16bit, not a combination!\n"
                "Also, we by default set `load_in_16bit = True`.\n"
                "If you want 4bit LoRA finetuning, set `load_in_16bit = False` and `load_in_4bit = True`\n"
                "If you want 8bit finetuning, set both `load_in_16bit = False` and `load_in_8bit = True`"
            )

        if "auto_model" not in kwargs:
            kwargs["auto_model"] = AutoModel

        _patch_mnrl_loss()
        _patch_efficient_pooling()

        transformers4 = Version(transformers.__version__).major < 5
        model_type = ""
        config = None
        try:
            config = AutoConfig.from_pretrained(
                model_name, token = token, trust_remote_code = trust_remote_code
            )
            model_type = getattr(config, "model_type", "")
        except:
            pass

        is_encoder_model = (
            model_type.lower() in FastSentenceTransformer.ENCODER_MODEL_TYPES
        )
        use_fast_encoder = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") != "1"
        if use_fast_encoder and is_encoder_model:
            if full_finetuning:
                compile_mode = "max-autotune"
            else:
                compile_mode = "default"

            if dtype is None:
                if load_in_16bit:
                    dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
                else:
                    dtype = torch.float32
            elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
                print(
                    "Unsloth: Device does not support bfloat16. Using float16 instead."
                )
                dtype = torch.float16

            st_device = device_map
            if isinstance(st_device, dict) or (
                isinstance(st_device, str) and st_device in ["auto", "sequential"]
            ):
                st_device = "cuda"

            supports_sdpa = False
            supports_flash_attn_2 = False
            if config is not None:
                try:
                    model_class = _get_model_class(
                        config, kwargs.get("auto_model", AutoModel)._model_mapping
                    )
                    supports_sdpa = getattr(model_class, "_supports_sdpa", False)
                    supports_flash_attn_2 = getattr(
                        model_class, "_supports_flash_attn_2", False
                    )
                except:
                    pass

            if supports_flash_attn_2:
                try:
                    import flash_attn  # noqa: F401
                except ImportError:
                    supports_flash_attn_2 = False

            # force-enable flash_attn_2 for known encoder types
            if not supports_flash_attn_2 and model_type in _UNPAD_SUPPORTED_TYPES:
                try:
                    import flash_attn  # noqa: F401

                    supports_flash_attn_2 = True
                except ImportError:
                    pass

            _use_new_dtype_kwarg = Version(transformers.__version__) >= Version(
                "4.48.0"
            )
            model_kwargs = (
                {"dtype": dtype} if _use_new_dtype_kwarg else {"torch_dtype": dtype}
            )

            _force_eager = False
            for _sdpa_model in DISABLE_SDPA_MODEL_NAMES:
                if _sdpa_model in model_type.lower():
                    supports_sdpa = False
                    supports_flash_attn_2 = False
                    _force_eager = True
                    break

            if supports_flash_attn_2:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            elif supports_sdpa:
                model_kwargs["attn_implementation"] = "sdpa"
            elif _force_eager:
                model_kwargs["attn_implementation"] = "eager"

            if supports_flash_attn_2:
                attn_str = " + FlashAttention 2"
            elif supports_sdpa:
                attn_str = " + SDPA"
            else:
                attn_str = ""

            if load_in_4bit:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} with 4-bit quantization{attn_str}"
                )
            else:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} (torch.compile{attn_str})"
                )

            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_compute_dtype = dtype,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_use_double_quant = True,
                )
                model_kwargs["quantization_config"] = bnb_config
                st_device = None

            _use_gc = use_gradient_checkpointing
            if _use_gc and _use_gc != False:
                print(
                    "Unsloth Warning: Gradient checkpointing is incompatible with torch.compile."
                )
                print("Disabling torch.compile to enable gradient checkpointing.")
                compile_mode = None  # Disable compilation

                is_mpnet = "mpnet" == model_type.lower()

                if is_mpnet and transformers4:
                    FastSentenceTransformer._patch_mpnet_v4()
                elif is_mpnet:
                    FastSentenceTransformer._patch_mpnet_v5()

            st_model = SentenceTransformer(
                model_name,
                device = st_device,
                trust_remote_code = trust_remote_code,
                token = token,
                revision = revision,
                model_kwargs = model_kwargs,
            )

            st_model._unsloth_fast_encoder = True
            st_model._compile_mode = compile_mode
            st_model._dtype = dtype
            st_model._load_in_4bit = load_in_4bit
            st_model._full_finetuning = full_finetuning
            st_model.no_modules = False

            if compile_mode is None and _HAS_FAST_LAYERNORM:
                inner_model = st_model[0].auto_model
                ln_count = _patch_encoder_layernorms(inner_model)
                if ln_count > 0:
                    print(
                        f"Unsloth: Patched {ln_count} LayerNorm modules with Triton kernel"
                    )

            if compile_mode is None and _HAS_FUSED_POOLING:
                if _patch_fused_pooling(st_model):
                    print(
                        "Unsloth: Fused final LayerNorm + Mean Pooling into single Triton kernel"
                    )

            _unpad_env = os.environ.get("UNSLOTH_UNPADDING", "1")
            if _unpad_env == "1":
                if _patch_unpadded_encoder(st_model, model_type):
                    backend = getattr(st_model[0], "_unpadding_backend", "unknown")
                    print(
                        f"Unsloth: Enabled variable-length batching (unpadding) via {backend}"
                    )

            if use_guided_projection:
                if _use_gc and _use_gc != False:
                    print(
                        "Unsloth Warning: use_guided_projection is incompatible with "
                        "gradient checkpointing (no encoder backward pass needed). "
                        "Disabling guided projection."
                    )
                elif not full_finetuning:
                    projection = attach_guided_projection(st_model)
                    st_model._guided_projection = projection
                    print(
                        f"Unsloth: Attached guided projection ({projection.num_trainable_parameters:,} "
                        f"trainable params). Encoder frozen — use projection.parameters() for optimizer."
                    )
                else:
                    print(
                        "Unsloth Warning: use_guided_projection is not compatible with "
                        "full_finetuning=True (encoder already trainable). Skipping."
                    )

            def _save_pretrained_merged(self, save_directory, **save_kwargs):
                with _restore_fused_pooling_ln(self):
                    self.save_pretrained(save_directory)
                    tokenizer = save_kwargs.pop("tokenizer", self.tokenizer)
                    if hasattr(self[0], "auto_model"):
                        inner = self[0].auto_model
                        if hasattr(inner, "_orig_mod"):
                            inner = inner._orig_mod
                        if getattr(self, "_sparsity_applied", False):
                            _remove_sparsity_from_base_weights(inner)
                        if hasattr(inner, "merge_and_unload"):
                            merged = inner.merge_and_unload()
                            merged.save_pretrained(save_directory)
                        elif hasattr(inner, "save_pretrained"):
                            inner.save_pretrained(save_directory)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(save_directory)
                    FastSentenceTransformer._add_unsloth_branding(save_directory)

            st_model.save_pretrained_merged = types.MethodType(
                _save_pretrained_merged, st_model
            )

            st_model.save_pretrained_torchao = types.MethodType(
                _save_pretrained_torchao, st_model
            )

            st_model.save_pretrained_gguf = types.MethodType(
                unsloth_save_pretrained_gguf, st_model
            )

            st_model.push_to_hub_gguf = types.MethodType(
                unsloth_push_to_hub_gguf, st_model
            )

            def _push_to_hub_merged(self, repo_id, **push_kwargs):
                hub_token = push_kwargs.get("token", None) or get_token()
                if hub_token is None:
                    raise ValueError("No HF token provided")
                api = HfApi(token = hub_token)
                try:
                    api.create_repo(
                        repo_id = repo_id,
                        private = push_kwargs.get("private"),
                        exist_ok = True,
                        repo_type = "model",
                    )
                except:
                    pass
                FastSentenceTransformer._add_unsloth_tags(repo_id, hub_token)
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.save_pretrained_merged(temp_dir, **push_kwargs)
                    api.upload_folder(
                        folder_path = temp_dir,
                        repo_id = repo_id,
                        commit_message = push_kwargs.get(
                            "commit_message", "Upload model"
                        ),
                    )
                print(f"Unsloth: Pushed to https://huggingface.co/{repo_id}")

            st_model.push_to_hub_merged = types.MethodType(
                _push_to_hub_merged, st_model
            )

            return st_model

        if is_encoder_model and load_in_4bit:
            print(
                "Unsloth Warning: 4-bit quantization adds ~2.3x overhead for encoder models."
            )
            print("Consider using load_in_16bit=True for better performance.")

        if "add_pooling_layer" not in kwargs:
            supported = FastSentenceTransformer._has_add_pooling_layer(
                config, kwargs.get("auto_model", AutoModel)
            )
            if supported:
                kwargs["add_pooling_layer"] = False

        fp8 = kwargs.pop("load_in_fp8", None)
        if fp8:
            logging.info("Unsloth: Disabling fp8 for model")
        load_in_fp8 = False

        old_environ = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

        is_distilbert = "distilbert" == model_type.lower()
        is_mpnet = "mpnet" == model_type.lower()

        if is_distilbert and transformers4:
            FastSentenceTransformer._patch_distilbert_v4()
        elif is_distilbert:
            FastSentenceTransformer._patch_distilbert_v5()
        elif is_mpnet and transformers4:
            FastSentenceTransformer._patch_mpnet_v4()
        elif is_mpnet:
            FastSentenceTransformer._patch_mpnet_v5()

        has_modules_json = (
            FastSentenceTransformer._module_path(model_name, token) is not None
        )

        if not has_modules_json and load_in_4bit:
            print(
                "Unsloth: No modules.json found. This is not a sentence-transformers model.\n"
                "Forcing 16-bit loading to simplify merged model saving."
            )
            load_in_4bit = False
            load_in_16bit = True

        try:
            model, tokenizer = FastModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                load_in_16bit = load_in_16bit,
                full_finetuning = full_finetuning,
                token = token,
                device_map = device_map,
                rope_scaling = rope_scaling,
                fix_tokenizer = fix_tokenizer,
                trust_remote_code = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab = resize_model_vocab,
                revision = revision,
                return_logits = False,
                use_exact_model_name = use_exact_model_name,
                offload_embedding = offload_embedding,
                random_state = random_state,
                max_lora_rank = max_lora_rank,
                disable_log_stats = disable_log_stats,
                qat_scheme = qat_scheme,
                load_in_fp8 = load_in_fp8,
                unsloth_tiled_mlp = unsloth_tiled_mlp,
                **kwargs,
            )
        finally:
            os.environ["UNSLOTH_WARN_UNINITIALIZED"] = old_environ

        # try to load modules, otherwise fallback to old hard-coded modules
        from sentence_transformers import SentenceTransformer

        modules, no_modules = FastSentenceTransformer._load_modules(
            model_name,
            token,
            model,
            tokenizer,
            max_seq_length,
            pooling_mode,
            trust_remote_code = trust_remote_code,
        )

        st_device = device_map
        if isinstance(st_device, dict) or (
            isinstance(st_device, str) and st_device in ["auto", "sequential"]
        ):
            st_device = None

        st_model = SentenceTransformer(modules = modules, device = st_device)
        st_model.no_modules = no_modules

        _inner = None
        _is_bidirectional = False
        for _mod in st_model:
            if hasattr(_mod, "auto_model"):
                _am = _mod.auto_model
                _am_unwrap = _am._orig_mod if hasattr(_am, "_orig_mod") else _am
                _cfg = getattr(_am_unwrap, "config", None)
                _is_bidirectional = getattr(_cfg, "use_bidirectional_attention", False)
                _is_decoder = bool(getattr(_cfg, "is_decoder", False))
                if (
                    _cfg is not None
                    and getattr(_cfg, "_attn_implementation", None) == "flex_attention"
                ):
                    if _is_bidirectional:
                        # Short seqs don't need sliding window BlockMask; SDPA is faster
                        _sw = getattr(_cfg, "sliding_window", None) or float("inf")
                        if max_seq_length < _sw:
                            _cfg._attn_implementation = "sdpa"
                            if hasattr(_cfg, "attn_implementation"):
                                _cfg.attn_implementation = "sdpa"
                            print(
                                f"Unsloth: Overriding flex_attention → sdpa for bidirectional model "
                                f"(max_seq_length={max_seq_length} < sliding_window={_sw})"
                            )
                        else:
                            print(
                                f"Unsloth: Keeping flex_attention for bidirectional model "
                                f"(max_seq_length={max_seq_length} >= sliding_window={_sw})"
                            )
                    else:
                        _has_fa2 = False
                        try:
                            import flash_attn  # noqa: F401

                            _has_fa2 = getattr(
                                _am_unwrap.__class__, "_supports_flash_attn_2", False
                            )
                        except ImportError:
                            pass
                        _best_attn = "flash_attention_2" if _has_fa2 else "sdpa"
                        _cfg._attn_implementation = _best_attn
                        if hasattr(_cfg, "attn_implementation"):
                            _cfg.attn_implementation = _best_attn
                        print(
                            f"Unsloth: Overriding flex_attention → {_best_attn} for sentence transformer training"
                        )
                _inner = _am_unwrap
                break

        if _inner is not None and _HAS_FAST_LAYERNORM:
            _ln_count = _patch_encoder_layernorms(_inner)
            if _ln_count > 0:
                print(
                    f"Unsloth: Patched {_ln_count} LayerNorm modules with Triton kernel"
                )

        if _HAS_FUSED_POOLING:
            if _patch_fused_pooling(st_model):
                print(
                    "Unsloth: Fused final LayerNorm + Mean Pooling into single Triton kernel"
                )

        # Skip bidirectional decoders — Unsloth patch closure prevents varlen injection
        _unpad_env = os.environ.get("UNSLOTH_UNPADDING", "1")
        if _unpad_env == "1" and _is_decoder and not _is_bidirectional:
            if _patch_unpadded_decoder(st_model):
                _backend = getattr(st_model[0], "_unpadding_backend", "unknown")
                print(
                    f"Unsloth: Enabled variable-length batching (unpadding) via {_backend}"
                )

        def _save_pretrained_merged(self, save_directory, **kwargs):
            with _restore_fused_pooling_ln(self):
                # check which adapter files exist before save_pretrained
                adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
                existing_before = {
                    f
                    for f in adapter_files
                    if os.path.exists(os.path.join(save_directory, f))
                }

                self.save_pretrained(save_directory)

                for file in adapter_files:
                    if file not in existing_before:
                        try:
                            os.remove(os.path.join(save_directory, file))
                        except:
                            pass

                tokenizer = kwargs.pop("tokenizer", self.tokenizer)
                if self.no_modules:
                    print(
                        "Unsloth: No modules detected. Using standard merge_and_unload for saving..."
                    )
                    safe_kwargs = kwargs.copy()
                    unsloth_args = [
                        "save_method",
                        "temporary_location",
                        "maximum_memory_usage",
                    ]
                    for k in unsloth_args:
                        safe_kwargs.pop(k, None)

                    inner_auto = self[0].auto_model
                    if getattr(self, "_sparsity_applied", False):
                        _remove_sparsity_from_base_weights(inner_auto)
                    merged_model = inner_auto.merge_and_unload()
                    merged_model.save_pretrained(save_directory, **safe_kwargs)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(save_directory)
                else:
                    if getattr(self, "_sparsity_applied", False):
                        _remove_sparsity_from_base_weights(self[0].auto_model)
                    self[0].auto_model.save_pretrained_merged(
                        save_directory, tokenizer = tokenizer, **kwargs
                    )

                try:
                    FastSentenceTransformer._add_unsloth_branding(save_directory)
                except Exception as e:
                    print(f"Unsloth Warning: Failed to add branding to README: {e}")

        st_model.save_pretrained_merged = types.MethodType(
            _save_pretrained_merged, st_model
        )

        st_model.save_pretrained_torchao = types.MethodType(
            _save_pretrained_torchao, st_model
        )

        st_model.save_pretrained_gguf = types.MethodType(
            unsloth_save_pretrained_gguf, st_model
        )

        st_model.push_to_hub_gguf = types.MethodType(unsloth_push_to_hub_gguf, st_model)

        def _push_to_hub_merged(self, repo_id, **kwargs):
            token = kwargs.get("token", None) or get_token()
            if token is None:
                raise ValueError(
                    "No HF token provided. Please provide a token or login with `hf auth login`"
                )
            private = kwargs.get("private", None)
            commit_message = kwargs.get("commit_message", "Upload model")

            from huggingface_hub import HfApi

            api = HfApi(token = token)
            try:
                api.create_repo(
                    repo_id = repo_id,
                    private = private,
                    exist_ok = True,
                    repo_type = "model",
                )
            except:
                pass

            FastSentenceTransformer._add_unsloth_tags(repo_id, token)

            with tempfile.TemporaryDirectory() as temp_dir:
                self.save_pretrained_merged(temp_dir, **kwargs)
                api.upload_folder(
                    folder_path = temp_dir,
                    repo_id = repo_id,
                    commit_message = commit_message,
                )
            print(
                f"Unsloth: Successfully pushed merged model to https://huggingface.co/{repo_id}"
            )

        st_model.push_to_hub_merged = types.MethodType(_push_to_hub_merged, st_model)
        return st_model

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = [
            "query",
            "key",
            "value",
            "dense",
        ],
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = False,  # Changed default: conflicts with torch.compile
        random_state = 3407,
        max_seq_length = 2048,
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        **kwargs,
    ):
        from sentence_transformers import SentenceTransformer
        from peft import LoraConfig, get_peft_model as peft_get_peft_model

        if "task_type" not in kwargs:
            kwargs["task_type"] = "FEATURE_EXTRACTION"
            print("Setting task_type to FEATURE_EXTRACTION")

        if isinstance(model, SentenceTransformer):
            # Check if this is a fast encoder model (uses torch.compile instead of Unsloth patching)
            is_fast_encoder = getattr(model, "_unsloth_fast_encoder", False)

            if is_fast_encoder:
                transformer_module = model[0]
                inner_model = transformer_module.auto_model

                is_quantized = (
                    getattr(inner_model, "is_quantized", False)
                    or getattr(inner_model.config, "quantization_config", None)
                    is not None
                )

                gc_enabled = False

                if use_gradient_checkpointing and use_gradient_checkpointing != False:
                    import transformers
                    from packaging.version import Version

                    transformers4 = Version(transformers.__version__).major < 5
                    model_type = getattr(inner_model.config, "model_type", "").lower()

                    if model_type == "mpnet" and transformers4:
                        FastSentenceTransformer._patch_mpnet_v4()
                    elif model_type == "mpnet":
                        FastSentenceTransformer._patch_mpnet_v5()

                if is_quantized:
                    from ._utils import prepare_model_for_kbit_training

                    _gc_for_kbit = (
                        use_gradient_checkpointing
                        if use_gradient_checkpointing
                        else False
                    )
                    try:
                        inner_model = prepare_model_for_kbit_training(
                            inner_model,
                            use_gradient_checkpointing = _gc_for_kbit,
                        )
                        print("Unsloth: Prepared quantized model for k-bit training")
                        gc_enabled = bool(_gc_for_kbit)
                    except ValueError as e:
                        if "does not support gradient checkpointing" in str(e):
                            print(
                                f"Unsloth Warning: {inner_model.__class__.__name__} does not support gradient checkpointing. Skipping."
                            )
                            inner_model = prepare_model_for_kbit_training(
                                inner_model,
                                use_gradient_checkpointing = False,
                            )
                            print(
                                "Unsloth: Prepared quantized model for k-bit training (without gradient checkpointing)"
                            )
                        else:
                            raise

                elif use_gradient_checkpointing and use_gradient_checkpointing != False:
                    if hasattr(inner_model, "gradient_checkpointing_enable"):
                        try:
                            inner_model.gradient_checkpointing_enable()
                            print("Unsloth: Enabled gradient checkpointing")
                            gc_enabled = True
                        except ValueError as e:
                            if "does not support gradient checkpointing" in str(e):
                                print(
                                    f"Unsloth Warning: {inner_model.__class__.__name__} does not support gradient checkpointing. Skipping."
                                )

                lora_config = LoraConfig(
                    r = r,
                    lora_alpha = lora_alpha,
                    target_modules = target_modules,
                    lora_dropout = lora_dropout,
                    bias = bias,
                    task_type = kwargs.get("task_type", "FEATURE_EXTRACTION"),
                )

                peft_model = peft_get_peft_model(inner_model, lora_config)

                qat_scheme = kwargs.get("qat_scheme", None)
                if qat_scheme is not None:
                    from ._utils import _prepare_model_for_qat

                    peft_model = _prepare_model_for_qat(peft_model, qat_scheme)

                compile_mode = getattr(model, "_compile_mode", "default")
                if compile_mode is None and not gc_enabled:
                    compile_mode = "default"
                    print(
                        "Unsloth: Re-enabling torch.compile since gradient checkpointing is not supported"
                    )

                transformer_module.auto_model = peft_model

                sparsity_env = os.environ.get("UNSLOTH_SPARSITY", "auto")
                if sparsity_env.lower() == "1":
                    do_sparsity = True
                elif sparsity_env.lower() == "0":
                    do_sparsity = False
                else:
                    do_sparsity = True

                if do_sparsity and not getattr(model, "_full_finetuning", False):
                    supported, sparsity_msg = _check_sparsity_support()
                    if supported:
                        sparse_count = _apply_sparsity_to_base_weights(peft_model)
                        if sparse_count > 0:
                            model._sparsity_applied = True
                            print(
                                f"Unsloth: Applied 2:4 sparsity to {sparse_count} base layer(s) ({sparsity_msg})"
                            )
                    elif sparsity_env.lower() == "1":
                        print(
                            f"Unsloth Warning: UNSLOTH_SPARSITY=1 but not supported: {sparsity_msg}"
                        )

                if compile_mode is not None:
                    model._compile_mode = compile_mode
                    # threshold re-estimated with batch/seq info in _patch_sentence_transformer_trainer
                    model._compile_pending = True
                    print(
                        "Unsloth: torch.compile deferred until training starts (threshold computed from TrainingArguments)"
                    )
                else:
                    model._compile_mode = None
                    model._compile_pending = False

                    fused_attn_count = _patch_encoder_attention_lora(peft_model)
                    if fused_attn_count > 0:
                        print(
                            f"Unsloth: Fused LoRA QKV backward for {fused_attn_count} attention layer(s)"
                        )

                    print(
                        "Unsloth: torch.compile disabled (gradient checkpointing enabled)"
                    )

                return model

            transformer_module = model[0]
            inner_model = transformer_module.auto_model

            peft_model = FastModel.get_peft_model(
                model = inner_model,
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                layers_to_transform = layers_to_transform,
                layers_pattern = layers_pattern,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                max_seq_length = max_seq_length,
                use_rslora = use_rslora,
                modules_to_save = modules_to_save,
                init_lora_weights = init_lora_weights,
                loftq_config = loftq_config,
                **kwargs,
            )

            transformer_module.auto_model = peft_model
            return model
        else:
            return FastModel.get_peft_model(
                model = model,
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                layers_to_transform = layers_to_transform,
                layers_pattern = layers_pattern,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                max_seq_length = max_seq_length,
                use_rslora = use_rslora,
                modules_to_save = modules_to_save,
                init_lora_weights = init_lora_weights,
                loftq_config = loftq_config,
                **kwargs,
            )


def _patch_sentence_transformer_trainer():
    """Auto-apply torch.compile when training steps exceed breakeven threshold."""
    try:
        from sentence_transformers import SentenceTransformerTrainer
    except ImportError:
        return  # sentence_transformers not installed

    if getattr(SentenceTransformerTrainer, "_unsloth_auto_compile_patched", False):
        return  # Already patched

    from functools import wraps

    _original_init = SentenceTransformerTrainer.__init__

    @wraps(_original_init)
    def _patched_init(self, *args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else None)
        training_args = kwargs.get("args") or (args[1] if len(args) > 1 else None)

        if (
            model is not None
            and training_args is not None
            and getattr(model, "_compile_pending", False)
        ):
            max_steps = getattr(training_args, "max_steps", -1)
            compile_mode = getattr(model, "_compile_mode", "default")

            batch_size = getattr(training_args, "per_device_train_batch_size", None)
            grad_accum = getattr(training_args, "gradient_accumulation_steps", None)
            max_seq_length = getattr(model, "max_seq_length", None)
            if max_seq_length is None and hasattr(model, "__getitem__"):
                try:
                    max_seq_length = getattr(model[0], "max_seq_length", None)
                except Exception:
                    max_seq_length = None
            if max_seq_length is None:
                tokenizer = getattr(model, "tokenizer", None)
                max_seq_length = (
                    getattr(tokenizer, "model_max_length", None)
                    if tokenizer is not None
                    else None
                )

            threshold = FastSentenceTransformer._estimate_compile_threshold(
                model,
                batch_size = batch_size,
                grad_accum = grad_accum,
                max_seq_length = max_seq_length,
            )
            model._compile_threshold = threshold

            if max_steps > 0 and max_steps >= threshold:
                is_full_ft = getattr(model, "_full_finetuning", False)
                if max_steps >= 500 or is_full_ft:
                    compile_mode = "max-autotune"
                else:
                    compile_mode = "default"

                print(
                    f"Unsloth: Auto-compiling model ({max_steps} steps >= {threshold} threshold, mode={compile_mode})"
                )
                FastSentenceTransformer._apply_torch_compile(model, mode = compile_mode)
                model._compile_pending = False
            elif max_steps > 0:
                print(
                    f"Unsloth: Skipping torch.compile ({max_steps} steps < {threshold} threshold)"
                )
                model._compile_pending = False

        _original_init(self, *args, **kwargs)

        if hasattr(self, "args") and self.args is not None:
            if not self.args.dataloader_pin_memory:
                self.args.dataloader_pin_memory = True
            if (
                self.args.dataloader_num_workers == 0
                and os.environ.get("UNSLOTH_NUM_WORKERS") != "0"
            ):
                print(
                    "Unsloth: Setting dataloader_num_workers=2. Set UNSLOTH_NUM_WORKERS=0 to disable."
                )
                self.args.dataloader_num_workers = 2

        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            if hasattr(self, "args") and self.args is not None:
                if self.args.fp16 or self.args.bf16:
                    print(
                        "Unsloth: Switching to float32 training since model cannot work with float16"
                    )
                    self.args.fp16 = False
                    self.args.bf16 = False
                    if hasattr(self.args, "bf16_full_eval"):
                        self.args.bf16_full_eval = False
                    if hasattr(self.args, "fp16_full_eval"):
                        self.args.fp16_full_eval = False

    SentenceTransformerTrainer.__init__ = _patched_init
    SentenceTransformerTrainer._unsloth_auto_compile_patched = True


_patch_sentence_transformer_trainer()
