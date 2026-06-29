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
from ._utils import (
    SUPPORTS_BFLOAT16,
    resolve_model_class,
    resolve_encoder_attention_implementation,
    HAS_FLASH_ATTENTION_VARLEN,
)
from unsloth_zoo.hf_utils import add_dtype_kwargs
import inspect
import json
import os
import threading
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


def unpad_input(
    input_ids,
    seq_info,
    token_type_ids = None,
):
    """Remove padding tokens from a (B, S) batch.

    This is a deliberate 3-line gather that reuses ``seq_info.indices`` (already
    computed once in ``get_encoder_seq_info``), so there is no per-call recompute
    of ``cu_seqlens``/``indices``. We intentionally do NOT use
    ``flash_attn.bert_padding.unpad_input`` (pulls a hard flash-attn dependency
    and recomputes the metadata) or transformers' private ``_upad_input``.
    """
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
        # The projection keeps fp32 master weights while the encoder may emit
        # bf16/fp16 pooled embeddings — cast like the Dense dtype patch does.
        if x.dtype != self.proj.weight.dtype:
            x = x.to(self.proj.weight.dtype)
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
    """Pooling + GuidedProjection as a single sentence transformers pipeline module."""

    PROJECTION_WEIGHTS_NAME = "guided_projection.pt"
    PROJECTION_SAFETENSORS_NAME = "guided_projection.safetensors"
    PROJECTION_CONFIG_NAME = "guided_projection_config.json"

    def __init__(self, pooling_module: nn.Module, projection: GuidedProjection):
        super().__init__()
        self.pooling = pooling_module
        self.projection = projection

    def forward(self, features: dict) -> dict:
        features = self.pooling(features)
        if "sentence_embedding" in features:
            features["sentence_embedding"] = self.projection(features["sentence_embedding"])
        return features

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pooling, name)

    def save(
        self,
        output_path: str,
        safe_serialization: bool = True,
    ) -> None:
        os.makedirs(output_path, exist_ok = True)

        if hasattr(self.pooling, "save"):
            self.pooling.save(output_path)

        saved_safetensors = False
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                save_file(
                    self.projection.state_dict(),
                    os.path.join(output_path, self.PROJECTION_SAFETENSORS_NAME),
                )
                saved_safetensors = True
            except ImportError:
                pass
        if not saved_safetensors:
            torch.save(
                self.projection.state_dict(),
                os.path.join(output_path, self.PROJECTION_WEIGHTS_NAME),
            )

        # modules.json will reference this unsloth class, so vanilla
        # sentence-transformers cannot reload the checkpoint. Say so at save
        # time instead of letting users discover it via ImportError later.
        print(
            "Unsloth: this checkpoint contains a GuidedProjectionPooling module — "
            "loading it requires unsloth to be installed."
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
        cls,
        input_path: str,
        pooling_module: Optional[nn.Module] = None,
        **kwargs,
    ) -> "GuidedProjectionPooling":
        # sentence transformers >= 5.4 / resume call this as load(path) or load(path, **kwargs)
        # (e.g. subfolder=...) with no pooling_module, so accept both and stay
        # self-contained.
        subfolder = kwargs.get("subfolder", "") or ""
        if subfolder:
            input_path = os.path.join(input_path, subfolder)

        config_path = os.path.join(input_path, cls.PROJECTION_CONFIG_NAME)
        weights_path = os.path.join(input_path, cls.PROJECTION_WEIGHTS_NAME)
        safetensors_path = os.path.join(input_path, cls.PROJECTION_SAFETENSORS_NAME)

        with open(config_path, "r") as f:
            config = json.load(f)

        projection = GuidedProjection(
            dim = config["dim"],
            output_dim = config["output_dim"],
            use_bias = config["use_bias"],
            use_residual = config["use_residual"],
            init = config["init"],
        )
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(weights_path, map_location = "cpu", weights_only = True)
        projection.load_state_dict(state_dict)

        if pooling_module is None:
            # Reconstruct the wrapped Pooling saved alongside the projection.
            from sentence_transformers.models import Pooling
            try:
                pooling_module = Pooling.load(input_path)
            except Exception:
                pooling_module = Pooling(config["dim"])

        return cls(pooling_module, projection)


def attach_guided_projection(
    st_model,
    dim = None,
    **kwargs,
):
    """Freeze encoder and attach a GuidedProjection after pooling."""
    pooling_idx = None
    pooling_mod = None
    if hasattr(st_model, "_modules"):
        for key, mod in st_model._modules.items():
            if mod.__class__.__name__ == "Pooling":
                pooling_idx = key
                pooling_mod = mod
                break

    if dim is None and pooling_mod is not None:
        # Prefer the pooled embedding width: Pooling can concatenate several
        # modes (e.g. cls + mean), making sentence_embedding wider than
        # config.hidden_size.
        for _dim_attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            _dim_fn = getattr(pooling_mod, _dim_attr, None)
            if callable(_dim_fn):
                try:
                    dim = int(_dim_fn())
                    break
                except Exception:
                    pass

    if dim is None:
        encoder = None
        for mod in st_model:
            if hasattr(mod, "auto_model"):
                encoder = mod
                break

        if encoder is None:
            raise ValueError("Could not locate a Transformer module. Specify `dim` explicitly.")

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
    # The SentenceTransformer is already on its target device; a fresh CPU
    # projection would crash a direct forward with a device mismatch
    # (encode()/Trainer re-place the model, a bare model(features) does not).
    try:
        projection.to(next(st_model.parameters()).device)
    except StopIteration:
        pass

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


# Flash-attention varlen availability is centralised in _utils
# (HAS_FLASH_ATTENTION_VARLEN is already SM80-gated and reuses unsloth's single
# flash-attn install check). We only import the kernel itself here for the
# call sites below; it stays None when unavailable.
_FLASH_ATTN_VARLEN_AVAILABLE = HAS_FLASH_ATTENTION_VARLEN
_flash_attn_varlen_func = None
if _FLASH_ATTN_VARLEN_AVAILABLE:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

_XFORMERS_ATTN_AVAILABLE = False
_XFORMERS_CAUSAL_AVAILABLE = False
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

    # BlockDiagonalCausalMask powers the decoder packed-varlen path. Unlike
    # flash-attn / torch varlen_attn (Ampere+ only) it runs on Turing (T4) via the
    # cutlass backend, and unlike FlexAttention it needs no torch.compile (so no
    # per-shape recompiles). Separate try so a missing name can't disable the
    # (non-causal) encoder path above.
    try:
        from xformers.ops.fmha.attn_bias import (
            BlockDiagonalCausalMask as _XFormersBlockDiagonalCausalMask,
        )
        _XFORMERS_CAUSAL_AVAILABLE = True
    except ImportError:
        pass
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
        # The Triton kernel reads the bias unconditionally, so biasless LayerNorms
        # (e.g. ModernBERT with norm_bias = False) must keep their native forward.
        if (
            isinstance(module, nn.LayerNorm)
            and module.elementwise_affine
            and module.bias is not None
        ):
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward

            def make_fast_forward(ln_module):
                def _fast_forward(X):
                    # Fast_Layernorm.backward returns no dW/db, so trainable
                    # LayerNorms (e.g. full finetuning) take the native forward.
                    if ln_module.weight.requires_grad or ln_module.bias.requires_grad:
                        return ln_module._original_forward(X)
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

    def _active_lora_dropout(m):
        # The fused QKV path does not apply LoRA dropout; detect active dropout
        # so we can skip fusing those modules and keep training math correct (K3).
        dd = getattr(m, "lora_dropout", None)
        if dd is None:
            return False
        mods = list(dd.values()) if hasattr(dd, "values") else [dd]
        return any(getattr(d, "p", 0.0) and float(d.p) > 0.0 for d in mods)

    count = 0
    dropout_skipped = 0
    for _name, module in model.named_modules():
        detected = None
        for q_attr, k_attr, v_attr in QKV_ATTRS:
            q_mod = getattr(module, q_attr, None)
            k_mod = getattr(module, k_attr, None)
            v_mod = getattr(module, v_attr, None)
            if q_mod is None or k_mod is None or v_mod is None:
                continue
            if hasattr(q_mod, "lora_A") and hasattr(k_mod, "lora_A") and hasattr(v_mod, "lora_A"):
                detected = (q_attr, k_attr, v_attr)
                break

        if detected is None:
            continue

        q_attr, k_attr, v_attr = detected
        q_mod = getattr(module, q_attr)
        k_mod = getattr(module, k_attr)
        v_mod = getattr(module, v_attr)

        if (
            _active_lora_dropout(q_mod)
            or _active_lora_dropout(k_mod)
            or _active_lora_dropout(v_mod)
        ):
            dropout_skipped += 1
            continue

        q_mod._original_forward = q_mod.forward
        k_mod._original_forward = k_mod.forward
        v_mod._original_forward = v_mod.forward

        def _make_fused_forwards(attn_mod, qm, km, vm):
            def q_fused(x, *args, **kwargs):
                # Self-healing: drop any stale K/V cache left by a prior forward
                # that raised or was recomputed (gradient checkpointing) before
                # k_fused/v_fused consumed it.
                attn_mod._fused_k = None
                attn_mod._fused_v = None
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

    if dropout_skipped:
        import warnings
        warnings.warn(
            f"Unsloth: Skipped fused QKV LoRA for {dropout_skipped} attention "
            "module(s) with lora_dropout>0 (fused path does not apply LoRA dropout).",
            stacklevel = 2,
        )
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
        # CUTLASS 2:4 requires dimensions that are a multiple of (32, 64); a 32x32
        # probe fails the column constraint and wrongly reports "unsupported" on
        # otherwise-capable Ampere GPUs, so use a 128x128 probe instead.
        test_w = torch.zeros(128, 128, device = "cuda", dtype = torch.float16)
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
        # CUTLASS 2:4 sparse GEMM is a fp16/bf16/int8 tensor-core path (fp32 gets
        # no speedup), and to_sparse_semi_structured requires the weight dims to be
        # multiples of (32, 64) - a plain "% 4" check lets through shapes (e.g.
        # 384x100) that then crash. Skip anything that does not qualify.
        if w.dtype not in (torch.float16, torch.bfloat16, torch.int8):
            continue
        if w.shape[0] % 32 != 0 or w.shape[1] % 64 != 0:
            continue

        _rg = base.weight.requires_grad
        base._dense_weight_rg = _rg

        # Keep top-2 per group of 4
        w_abs = w.detach().abs().view(w.shape[0], -1, 4)
        _, topk = w_abs.topk(2, dim = -1)
        mask = torch.zeros_like(w_abs, dtype = torch.bool)
        mask.scatter_(-1, topk, True)
        mask = mask.view(w.shape)
        w.mul_(mask)

        # Mark as sparsified. We do NOT keep a resident dense clone of the pruned
        # weights: pruning (w.mul_) already happened before to_sparse_semi_structured,
        # so the dense form is reconstructed losslessly via .to_dense() at save/merge
        # (see _remove_sparsity_from_base_weights). This avoids holding ~1x the base
        # weight in memory for the whole run.
        base._unsloth_sparsified = True

        base.weight = torch.nn.Parameter(to_sparse_semi_structured(w), requires_grad = _rg)
        count += 1

    return count


def _remove_sparsity_from_base_weights(peft_model):
    """Restore dense weights for saving/merging."""
    count = 0
    for _name, module in peft_model.named_modules():
        base = getattr(module, "base_layer", None)
        if base is None or not isinstance(base, torch.nn.Linear):
            continue

        w = base.weight
        is_sparse = "SparseSemiStructured" in type(w).__name__ or (
            "SparseSemiStructured" in type(getattr(w, "data", w)).__name__
        )

        if getattr(base, "_unsloth_sparsified", False):
            _rg = getattr(base, "_dense_weight_rg", False)
            # Reconstruct the dense (already-pruned) weight from the semi-structured
            # tensor; pruning happened before sparsifying, so this is lossless.
            dense = w.to_dense() if is_sparse else w.data
            base.weight = torch.nn.Parameter(dense.contiguous(), requires_grad = _rg)
            del base._unsloth_sparsified
            if hasattr(base, "_dense_weight_rg"):
                del base._dense_weight_rg
            count += 1
        elif is_sparse:
            # Backstop (currently unreachable): a base layer holds a 2:4 sparse
            # weight but was never tagged by _apply_sparsity_to_base_weights, so we
            # cannot safely restore it. Warn rather than silently save a non-standard
            # sparse artifact.
            logging.getLogger(__name__).warning(
                "Unsloth Warning: %s holds a 2:4 sparse weight but is not tagged "
                "_unsloth_sparsified; skipping dense restore for save/merge.",
                _name,
            )
    return count


# Families whose last-registered LayerNorm is a true terminal LN. ALBERT is
# excluded: its LayerNorms are shared across repeated layers, so swapping the
# last-registered one for Identity would corrupt every layer pass.
_FUSED_POOLING_SAFE_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "electra",
    "mpnet",
    "distilbert",
    "modernbert",
}


def _patch_fused_pooling(st_model):
    """Fuse final LayerNorm + mean Pooling into single Triton kernel."""
    if not _HAS_FUSED_POOLING:
        return False

    # Idempotency: if already fused, a 2nd call would fuse the wrong (Identity)
    # LayerNorm. Bail out early.
    for _mod in st_model:
        if hasattr(_mod, "_fused_ln"):
            return True

    transformer_mod = None
    pooling_mod = None
    for mod in st_model:
        if hasattr(mod, "auto_model"):
            transformer_mod = mod
        if mod.__class__.__name__ == "Pooling":
            pooling_mod = mod

    if transformer_mod is None or pooling_mod is None:
        return False

    # sentence-transformers >= 5.x has no pooling_mode_* booleans until
    # _ensure_pooling_flags reconstructs them from the pooling_mode string;
    # without this the guard below always fails and the fused kernel is
    # silently skipped on ST 5.x.
    _ensure_pooling_flags(pooling_mod)

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

    _ft_cfg = getattr(inner, "config", None)
    if str(getattr(_ft_cfg, "model_type", "")).lower() not in _FUSED_POOLING_SAFE_TYPES:
        return False

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

    # fused_layernorm_mean_pool requires a bias (e.g. ModernBERT defaults to
    # norm_bias = False, leaving every LayerNorm biasless).
    if last_ln.bias is None:
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

        # Native-varlen batches (sentence-transformers >= 5.x can pass
        # cu_seq_lens metadata with token_embeddings packed into one row) are
        # not the padded (B, S, H) layout the fused kernel assumes — fall back.
        # Same for token_weights_sum (WordWeights pipelines): ST's mean pooling
        # divides by the weighted denominator, the fused kernel by token count.
        if (
            "cu_seq_lens_q" in features
            or "cu_seqlens" in features
            or "token_weights_sum" in features
            or token_embeddings.shape[:-1] != attention_mask.shape
        ):
            features["token_embeddings"] = stored_ln(token_embeddings)
            return _original_pooling_forward(features)

        # The fused kernel assumes right-padding (real tokens occupy [0, seq_len)).
        # Fall back to the original pooling for left-/irregularly-padded batches
        # so embeddings stay correct.
        seq_lengths = attention_mask.sum(dim = 1)
        seq_dim = attention_mask.shape[1]
        right_padded = (
            torch.arange(seq_dim, device = attention_mask.device).unsqueeze(0)
            < seq_lengths.unsqueeze(1)
        ).to(attention_mask.dtype)
        if not torch.equal(attention_mask, right_padded):
            # token_embeddings are PRE-LayerNorm (encoder LN is now Identity);
            # re-apply stored LN since original pooling doesn't.
            features["token_embeddings"] = stored_ln(token_embeddings)
            return _original_pooling_forward(features)

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


def _gguf_save_with_restore(self, *args, **kwargs):
    # GGUF export must restore the real LayerNorm that fused pooling swapped for
    # Identity, exactly like the merged/torchao save paths do.
    with _restore_fused_pooling_ln(self):
        return unsloth_save_pretrained_gguf(self, *args, **kwargs)


def _gguf_push_with_restore(self, *args, **kwargs):
    with _restore_fused_pooling_ln(self):
        return unsloth_push_to_hub_gguf(self, *args, **kwargs)


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
# Every known bidirectional encoder, including ones we must never unpad-patch
# (modernbert: native unpadding). _patch_unpadded_decoder refuses these so an
# encoder that falls through when the fast-encoder path is disabled
# (UNSLOTH_COMPILE_DISABLE=1) never receives a causal block-diagonal mask.
_UNPAD_KNOWN_ENCODER_TYPES = _UNPAD_SUPPORTED_TYPES | {"modernbert"}
_UNPAD_MIN_PADDING_RATIO = 0.15
# Decoder packing under sdpa/eager enforces sequence boundaries with an explicit
# block-diagonal causal mask, which is O(total_tokens^2) in memory. Above this many
# real tokens we skip packing and fall back to the padded forward to avoid an OOM.
# (flash_attention_2 uses O(N) varlen via position_ids and is NOT subject to this cap.)
_UNPAD_DECODER_MAX_PACKED_TOKENS = 16384


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
            is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", False)
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
        elif _XFORMERS_ATTN_AVAILABLE and (dropout == 0.0 or _XFORMERS_DROPOUT_SAFE):
            # Pre-Ampere xformers can't apply attention dropout; with dropout > 0
            # fall through to bool-mask SDPA, which applies it on the attention probs.
            q_x = query.transpose(1, 2)
            k_x = key.transpose(1, 2)
            v_x = value.transpose(1, 2)
            attn_bias = _XFormersBlockDiagonalMask.from_seqlens(seq_lengths.tolist())
            out_x = _xformers_memory_efficient_attention(
                q_x,
                k_x,
                v_x,
                attn_bias = attn_bias,
                p = dropout,
                scale = scaling,
            )
            attn_output = out_x.transpose(1, 2)
        elif _VARLEN_ATTN_AVAILABLE and dropout == 0.0:
            # torch varlen_attn has no dropout arg; with dropout > 0 use bool-mask SDPA.
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


def _is_gradient_checkpointing(model):
    """GC active anywhere (handles PEFT / unsloth GC). Under GC the packed path
    must use the sdpa block-mask (a tensor kwarg, replayed on recompute), not the
    xformers tier (its config swap + stashed seqlens don't survive recompute)."""
    if getattr(model, "gradient_checkpointing", False):
        return True
    try:
        return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    except Exception:
        return False


def _resolve_sliding_window(config):
    """Effective sliding-window size, or None for full attention. A full causal
    block mask only matches windowed attention when seqlen <= window."""
    if config is None:
        return None
    sw = getattr(config, "sliding_window", None)
    if not sw:
        return None
    if getattr(config, "use_sliding_window", True) is False:  # Qwen3: present but off
        return None
    layer_types = getattr(config, "layer_types", None)  # Gemma3: only if a layer is sliding
    if layer_types is not None and not any("sliding" in str(t) for t in layer_types):
        return None
    return int(sw)


def _repeat_kv_heads(hidden, n_rep):
    """Expand GQA key/value heads to match query heads. Mirrors transformers'
    repeat_kv on a (batch, num_kv_heads, seq, head_dim) tensor."""
    if n_rep == 1:
        return hidden
    b, h, s, d = hidden.shape
    return hidden[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _xformers_blockdiag_causal_attention(
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
    """Packed causal attention via xformers BlockDiagonalCausalMask (runs on
    Turing/T4+; O(sum seqlen^2)). seqlens are stashed on the config by
    _patch_unpadded_decoder. The passed attention_mask is ignored in favour of the
    bias, so transformers' mask handling for an unknown impl can't corrupt it.
    Validated on transformers 4.56.2 / 5.5.0 / 5.10.2."""
    seqlens = getattr(getattr(module, "config", None), "_unsloth_blockdiag_seqlens", None)
    if seqlens is None:
        # Only valid inside the packed forward (which stashes seqlens). Firing
        # without them (e.g. a GC recompute after the impl was restored) would run
        # unmasked attention over the packed row and leak across boundaries -> fail
        # loud. GC training routes through the sdpa block-mask tier to avoid this.
        raise RuntimeError(
            "unsloth_blockdiag_causal attention invoked without packed seqlens on "
            "the config (bug; set UNSLOTH_UNPADDING=0 to disable packing and report)."
        )
    n_rep = getattr(module, "num_key_value_groups", 1) or 1  # (B,H,T,D); expand GQA kv heads
    if n_rep > 1:
        key = _repeat_kv_heads(key, n_rep)
        value = _repeat_kv_heads(value, n_rep)
    # xformers wants (B,T,H,D). Pre-Ampere can't apply attention dropout (p forced
    # to 0); _patch_unpadded_decoder routes dropout > 0 models to the sdpa tier there.
    # Reuse the bias built once per forward (identical across layers).
    bias = getattr(module.config, "_unsloth_blockdiag_bias", None)
    if bias is None:
        bias = _XFormersBlockDiagonalCausalMask.from_seqlens(seqlens)
    out = _xformers_memory_efficient_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias = bias,
        p = float(dropout) if _XFORMERS_DROPOUT_SAFE else 0.0,
        scale = scaling,
    )
    return out, None


def _register_xformers_blockdiag_causal():
    """Register the xformers causal packed-varlen dispatcher (transformers
    attention-interface). Returns whether it is usable."""
    if not _XFORMERS_CAUSAL_AVAILABLE:
        return False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        return False
    try:
        if "unsloth_blockdiag_causal" not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS.register(
                "unsloth_blockdiag_causal", _xformers_blockdiag_causal_attention
            )
        return True
    except Exception:
        return False


_XFORMERS_CAUSAL_REGISTERED = _register_xformers_blockdiag_causal()


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

    # Idempotency: a 2nd call (e.g. re-running from_pretrained on the same
    # instance) must not wrap the already-wrapped forward — double-packing
    # corrupts token embeddings silently.
    if getattr(transformer_mod, "_unpadding_active", False):
        return True

    inner = transformer_mod.auto_model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    config = inner.config

    # The packed forward must reset positions per sequence via position_ids; if the
    # model's forward can't accept them (e.g. DistilBERT on transformers 4.x), the
    # packed row would silently get absolute positions 0..total_tokens, so skip.
    _fwd_params = getattr(transformer_mod, "model_forward_params", None)
    if _fwd_params is None:
        _fwd_params = set(inspect.signature(inner.forward).parameters)
    if "position_ids" not in _fwd_params:
        return False

    _orig_attn_impl = getattr(config, "_attn_implementation", "sdpa")

    # XLM-RoBERTa position_ids start at padding_idx + 1
    _position_offset = 0
    for mod in inner.modules():
        if hasattr(mod, "position_embeddings") and hasattr(mod.position_embeddings, "padding_idx"):
            _pad_idx = mod.position_embeddings.padding_idx
            if _pad_idx is not None:
                _position_offset = _pad_idx + 1
            break

    _original_forward = transformer_mod.forward

    # Only use the ALL_ATTENTION_FUNCTIONS registry on transformers 5.x+.
    # On 4.x, BERT/RoBERTa bake their attention class at __init__ time,
    # so changing config._attn_implementation after construction has no effect.
    _use_attn_interface = _VARLEN_ATTN_REGISTERED and Version(transformers.__version__).major >= 5

    if not _use_attn_interface:
        # transformers 4.x: F.sdpa monkey-patching
        _use_varlen = (
            _FLASH_ATTN_VARLEN_AVAILABLE or _VARLEN_ATTN_AVAILABLE or _XFORMERS_ATTN_AVAILABLE
        )
        _original_sdpa = torch.nn.functional.scaled_dot_product_attention

        # NOTE: Thread-safety limitation (transformers 4.x path only).
        # The below closure monkey-patches the global F.scaled_dot_product_attention for the
        # duration of a forward pass. Two concurrent forward passes will race on the global.
        # The transformers 5.x path (ALL_ATTENTION_FUNCTIONS registry) does not have this issue.
        # DataLoader workers are separate processes and do NOT trigger this; the race needs
        # concurrent in-process forwards (e.g. encode() from multiple threads).
        # Resolution: upgrade to transformers >=5.0, or avoid multi-threaded forwards.
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
            elif _XFORMERS_ATTN_AVAILABLE and (dropout_p == 0.0 or _XFORMERS_DROPOUT_SAFE):
                # Pre-Ampere xformers can't apply attention dropout; with dropout > 0
                # fall through to the bool-mask SDPA fallback below, which applies it
                # on the attention probs.
                seq_lengths = config._unsloth_seq_lengths
                q_x = query.transpose(1, 2)
                k_x = key.transpose(1, 2)
                v_x = value.transpose(1, 2)
                attn_bias = _XFormersBlockDiagonalMask.from_seqlens(seq_lengths.tolist())
                out_x = _xformers_memory_efficient_attention(
                    q_x,
                    k_x,
                    v_x,
                    attn_bias = attn_bias,
                    p = dropout_p,
                    scale = scale,
                )
                return out_x.transpose(1, 2)
            elif _VARLEN_ATTN_AVAILABLE and dropout_p == 0.0:
                # torch varlen_attn has no dropout arg; with dropout > 0 use bool-mask SDPA.
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
            # Packed-call fallback: a block-diagonal bool mask keeps sequence
            # boundaries while letting SDPA apply true attention dropout.
            seq_lengths = config._unsloth_seq_lengths
            segment_ids = torch.repeat_interleave(
                torch.arange(seq_lengths.shape[0], device = query.device),
                seq_lengths.long(),
            )
            bool_mask = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask = bool_mask.unsqueeze(0).unsqueeze(0),
                dropout_p = dropout_p,
                is_causal = False,
                scale = scale,
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
        actual_model = auto_model._orig_mod if hasattr(auto_model, "_orig_mod") else auto_model
        if not actual_model.training:
            return _original_forward(features, **kwargs)

        # Skip when compiled (recompiles on every shape) or when gradient
        # checkpointing is active (config resets before checkpoint recompute)
        if hasattr(auto_model, "_orig_mod"):
            return _original_forward(features, **kwargs)
        if _is_gradient_checkpointing(actual_model):
            return _original_forward(features, **kwargs)

        B, S = attention_mask.shape
        device = attention_mask.device

        # Cheap precheck: only the token count decides the padding-ratio bailout,
        # so avoid building full packed metadata (max().item() sync + nonzero +
        # cumsum) on lightly-padded batches that will bail out.
        total_tokens = int(attention_mask.sum().item())
        if total_tokens >= B * S * (1.0 - _UNPAD_MIN_PADDING_RATIO):
            return _original_forward(features, **kwargs)

        # Packed position_ids restart at 0 per sequence, which matches the padded
        # forward of absolute-position encoders only for right-padded rows.
        _row_lengths = attention_mask.sum(dim = 1)
        _right_padded = (
            torch.arange(S, device = device).unsqueeze(0) < _row_lengths.unsqueeze(1)
        ).to(attention_mask.dtype)
        if not torch.equal(attention_mask, _right_padded):
            return _original_forward(features, **kwargs)

        seq_info = get_encoder_seq_info(attention_mask)

        input_ids = features["input_ids"]
        packed_ids = input_ids.flatten()[seq_info.indices].unsqueeze(0)

        _offsets = torch.repeat_interleave(seq_info.cu_seqlens[:-1], seq_info.seq_lengths.long())
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
            k: v for k, v in packed_features.items() if k in transformer_mod.model_forward_params
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

    # Idempotency: a 2nd call must not re-wrap the already-wrapped forward —
    # double-packing corrupts token embeddings silently.
    if getattr(transformer_mod, "_unpadding_active", False):
        return True

    _patch_am = transformer_mod.auto_model
    _patch_inner = _patch_am._orig_mod if hasattr(_patch_am, "_orig_mod") else _patch_am
    _patch_cfg = getattr(_patch_inner, "config", None)

    # Never apply a causal mask to a bidirectional encoder. Encoders usually divert
    # to the fast-encoder path, but UNSLOTH_COMPILE_DISABLE=1 lets a BERT/RoBERTa
    # model fall through to here, so exclude encoder types explicitly (incl.
    # modernbert, which is not in _UNPAD_SUPPORTED_TYPES but is still an encoder).
    _model_type = str(getattr(_patch_cfg, "model_type", "")).lower()
    if _model_type in _UNPAD_KNOWN_ENCODER_TYPES:
        return False

    _sliding_window = _resolve_sliding_window(_patch_cfg)  # None = full attention

    if hasattr(transformer_mod, "model_forward_params"):
        transformer_mod.model_forward_params.add("packed_seq_lengths")

    _original_forward = transformer_mod.forward

    def _unpadded_forward(features, **kwargs):
        attention_mask = features.get("attention_mask")
        if attention_mask is None:
            return _original_forward(features, **kwargs)

        auto_model = transformer_mod.auto_model
        actual_model = auto_model._orig_mod if hasattr(auto_model, "_orig_mod") else auto_model
        if not actual_model.training:
            return _original_forward(features, **kwargs)

        B, S = attention_mask.shape
        device = attention_mask.device

        # Cheap precheck: only the token count decides the padding-ratio bailout,
        # so avoid building full packed metadata (max().item() sync + nonzero +
        # cumsum) on lightly-padded batches that will bail out.
        total_tokens = int(attention_mask.sum().item())
        if total_tokens >= B * S * (1.0 - _UNPAD_MIN_PADDING_RATIO):
            return _original_forward(features, **kwargs)

        # Sliding-window models: the full causal block mask only equals windowed
        # attention when seqlen <= window. S upper-bounds real lengths, so above the
        # window fall back to padded rather than ignore the window.
        if _sliding_window is not None and S > _sliding_window:
            return _original_forward(features, **kwargs)

        # Boundaries must be enforced or a causal decoder attends across packed
        # sequences. Backend (same math; differ on cost/hardware/GC):
        #   FA2          -> position_ids varlen, O(N), Ampere+, GC-safe. Preferred.
        #   xformers     -> O(sum seqlen^2), Turing+; CUDA-only, NOT GC-safe -> off under GC/CPU.
        #   sdpa blockmask-> tensor kwarg, GC- & CPU-safe, O(total^2) so capped -> else padded.
        attn_impl = getattr(actual_model.config, "_attn_implementation", None)
        _native_varlen = attn_impl == "flash_attention_2"
        _gc_active = _is_gradient_checkpointing(actual_model)
        # Pre-Ampere xformers can't apply attention dropout (the dispatcher forces
        # p = 0); route models with live attention dropout to the sdpa block-mask
        # tier, which applies it on the attention probs. (This forward only runs
        # in training, so a nonzero config dropout is live.)
        _attn_dropout = float(
            getattr(actual_model.config, "attention_dropout", 0.0)
            or getattr(actual_model.config, "attn_pdrop", 0.0)
            or 0.0
        )
        _use_xformers = (
            _XFORMERS_CAUSAL_REGISTERED
            and not _native_varlen
            and not _gc_active
            and attention_mask.is_cuda
            and (_XFORMERS_DROPOUT_SAFE or _attn_dropout == 0.0)
        )
        _use_blockmask = not _native_varlen and not _use_xformers
        if _use_blockmask and total_tokens > _UNPAD_DECODER_MAX_PACKED_TOKENS:
            return _original_forward(features, **kwargs)

        seq_info = get_encoder_seq_info(attention_mask)

        input_ids = features["input_ids"]
        packed_ids = input_ids.flatten()[seq_info.indices].unsqueeze(0)

        _offsets = torch.repeat_interleave(seq_info.cu_seqlens[:-1], seq_info.seq_lengths.long())
        position_ids = (torch.arange(total_tokens, device = device) - _offsets).unsqueeze(0)

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
            k: v for k, v in packed_features.items() if k in transformer_mod.model_forward_params
        }
        # Drop any caller-supplied attention_mask so it cannot clobber the packing
        # semantics (each backend below enforces boundaries its own way).
        _extra = {k: v for k, v in kwargs.items() if k != "attention_mask"}

        if _use_xformers:
            # Stash seqlens + one prebuilt bias (reused by every layer) and flip the
            # attn impl only for this packed call (restored in finally).
            _cfg = actual_model.config
            _saved_impl = getattr(_cfg, "_attn_implementation", None)
            _seqlens = seq_info.seq_lengths.tolist()
            _cfg._unsloth_blockdiag_seqlens = _seqlens
            _cfg._unsloth_blockdiag_bias = _XFormersBlockDiagonalCausalMask.from_seqlens(_seqlens)
            _cfg._attn_implementation = "unsloth_blockdiag_causal"
            try:
                outputs = auto_model(**trans_features, return_dict = True, **_extra)
            finally:
                _cfg._attn_implementation = _saved_impl
                _cfg._unsloth_blockdiag_seqlens = None
                _cfg._unsloth_blockdiag_bias = None
        elif _native_varlen:
            # position_ids alone -> flash-attn-2 rebuilds cu_seqlens internally.
            outputs = auto_model(**trans_features, return_dict = True, **_extra)
        else:
            # sdpa/eager: block-diagonal causal mask (i attends j iff same seq and j<=i).
            _seg = torch.repeat_interleave(
                torch.arange(B, device = device), seq_info.seq_lengths.long()
            )
            _ar = torch.arange(total_tokens, device = device)
            _allowed = (_seg[None, :] == _seg[:, None]) & (_ar[None, :] <= _ar[:, None])
            _mask_dtype = next(actual_model.parameters()).dtype
            trans_features["attention_mask"] = torch.where(
                _allowed,
                torch.zeros((), dtype = _mask_dtype, device = device),
                torch.full((), torch.finfo(_mask_dtype).min, dtype = _mask_dtype, device = device),
            )[None, None]
            outputs = auto_model(**trans_features, return_dict = True, **_extra)

        packed_embeddings = outputs[0].squeeze(0)  # (total_tokens, D)

        token_embeddings = pad_output(packed_embeddings, seq_info, B, S)
        features["token_embeddings"] = token_embeddings
        features["attention_mask"] = attention_mask
        return features

    transformer_mod.forward = _unpadded_forward
    transformer_mod._original_forward = _original_forward
    transformer_mod._unpadding_active = True
    _am = transformer_mod.auto_model
    _actual = _am._orig_mod if hasattr(_am, "_orig_mod") else _am
    _impl = getattr(getattr(_actual, "config", None), "_attn_implementation", None)
    # Representative backend at patch time (the per-forward selection may fall back
    # to blockdiag_sdpa under gradient checkpointing / on CPU / above the cap).
    if _impl == "flash_attention_2":
        transformer_mod._unpadding_backend = "varlen_position_ids"
    elif _XFORMERS_CAUSAL_REGISTERED:
        transformer_mod._unpadding_backend = "xformers_blockdiag_causal"
    else:
        transformer_mod._unpadding_backend = "blockdiag_sdpa"
    return True


_POOLING_MODE_FLAGS = {
    "cls": "pooling_mode_cls_token",
    "max": "pooling_mode_max_tokens",
    "mean": "pooling_mode_mean_tokens",
    "mean_sqrt_len_tokens": "pooling_mode_mean_sqrt_len_tokens",
    "weightedmean": "pooling_mode_weightedmean_tokens",
    "lasttoken": "pooling_mode_lasttoken",
}


def _ensure_pooling_flags(pooling_mod):
    """sentence-transformers >= 5.x dropped the boolean ``pooling_mode_*``
    attributes in favour of a single ``pooling_mode`` string. The efficient /
    fused pooling patches read those booleans, so reconstruct them when absent
    (otherwise _efficient_forward raises AttributeError on sentence transformers 5.x)."""
    if hasattr(pooling_mod, "pooling_mode_cls_token"):
        return
    mode = getattr(pooling_mod, "pooling_mode", None)
    for attr in _POOLING_MODE_FLAGS.values():
        setattr(pooling_mod, attr, False)
    target = _POOLING_MODE_FLAGS.get(mode)
    if target is not None:
        setattr(pooling_mod, target, True)


_POOLING_PATCHED = False


def _patch_efficient_pooling():
    """Monkey-patch Pooling to skip redundant expand()."""
    global _POOLING_PATCHED
    if _POOLING_PATCHED:
        return

    try:
        from sentence_transformers.models import Pooling

        _original_forward = Pooling.forward

        def _efficient_forward(self, features):
            _ensure_pooling_flags(self)  # sentence transformers 5.x compat (booleans dropped)
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
                # Handle left-padded sequences (vectorized: removes B per-row
                # .item() device->host syncs)
                seq_len_dim = attention_mask.shape[1]
                pad_lengths = (attention_mask == 0).to(torch.int32).argmin(dim = 1)
                prompt_cols = (
                    pad_lengths.unsqueeze(1)
                    + torch.arange(prompt_length, device = attention_mask.device).unsqueeze(0)
                ).clamp(max = seq_len_dim - 1)
                attention_mask.scatter_(1, prompt_cols, 0)

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
                # out-of-place to avoid mutating a possibly grad-tracked / shared
                # token_embeddings tensor
                masked_embeddings = token_embeddings.masked_fill(input_mask_expanded == 0, -1e9)
                max_over_time = torch.max(masked_embeddings, 1)[0]
                output_vectors.append(max_over_time)

            if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
                mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
                sum_embeddings = (token_embeddings * mask).sum(dim = 1)

                if "token_weights_sum" in features:
                    sum_mask = (
                        features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
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

            # Global class patch means a Pooling with every standard mode False
            # would otherwise hit torch.cat([], 1) -> RuntimeError
            if not output_vectors:
                return _original_forward(self, features)
            output_vector = torch.cat(output_vectors, 1)
            features["sentence_embedding"] = output_vector
            return features

        Pooling.forward = _efficient_forward
        # Only mark patched after the assignment succeeds, so a failed import
        # doesn't permanently disable the patch for the process
        _POOLING_PATCHED = True
    except Exception as e:
        import warnings
        warnings.warn(f"Unsloth: Failed to patch Pooling: {e}", stacklevel = 2)


_DENSE_PATCHED = False


def _patch_dense_dtype():
    """Monkey-patch Dense.forward to cast input to match weight dtype.

    Models like Gemma3 use high-precision RMSNorm that outputs float32, but
    SentenceTransformer.__init__ casts Dense weights to the transformer's
    param dtype (e.g. bf16). This causes a dtype mismatch in F.linear.
    """
    global _DENSE_PATCHED
    if _DENSE_PATCHED:
        return

    try:
        from sentence_transformers.models import Dense

        _original_dense_forward = Dense.forward

        def _dtype_safe_forward(self, features):
            if "sentence_embedding" in features:
                target_dtype = self.linear.weight.dtype
                emb = features["sentence_embedding"]
                if emb.dtype != target_dtype:
                    features["sentence_embedding"] = emb.to(target_dtype)
            return _original_dense_forward(self, features)

        Dense.forward = _dtype_safe_forward
        # Set only on success so a failed import doesn't permanently disable
        # the patch (mirrors _POOLING_PATCHED / _MNRL_PATCHED).
        _DENSE_PATCHED = True
    except Exception:
        pass


# Below this score-matrix size (~4096^2) dense matmul + cross_entropy beats the
# chunked kernel. Override via UNSLOTH_CONTRASTIVE_MIN_ELEMENTS (0 = always chunk).
_FUSED_CONTRASTIVE_MIN_ELEMENTS = int(
    os.environ.get("UNSLOTH_CONTRASTIVE_MIN_ELEMENTS", 16_000_000)
)

_MNRL_PATCHED = False
# MNRL.forward is patched globally at model load, but most runs never use that
# loss. Announce the fused path lazily, the first time it actually runs.
_MNRL_FUSED_NOTICE_SHOWN = False


def _patch_mnrl_loss():
    """Monkey-patch MNRL with fused chunked contrastive loss."""
    global _MNRL_PATCHED
    if _MNRL_PATCHED:
        return

    try:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from ..kernels.contrastive_loss import FusedContrastiveLoss

        _original_forward = MultipleNegativesRankingLoss.forward

        def _fused_forward(
            self,
            sentence_features,
            labels = None,
        ):
            # Non-default MNRL setups: just use the original.
            if (
                getattr(self, "gather_across_devices", False)
                or getattr(self, "directions", None) not in (None, ("query_to_doc",))
                or getattr(self, "partition_mode", None) not in (None, "joint")
                or getattr(self, "hardness_mode", None) is not None
            ):
                return _original_forward(self, sentence_features, labels)

            # Check the similarity fn first, so a custom one bails out before we
            # pay for a forward pass.
            similarity_fct = getattr(self, "similarity_fct", None)
            try:
                from sentence_transformers.util import cos_sim, dot_score
                is_cosine = similarity_fct is cos_sim
                is_dot = similarity_fct is dot_score
            except ImportError:
                is_cosine = True
                is_dot = False
            # cos_sim: normalize then dot. dot_score: dot as-is. Anything else we
            # don't replicate, so fall back.
            if not (is_cosine or is_dot):
                return _original_forward(self, sentence_features, labels)

            reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
            embeddings_a = reps[0]
            embeddings_b = torch.cat(reps[1:], dim = 0)

            if is_cosine:
                embeddings_a = torch.nn.functional.normalize(embeddings_a, p = 2, dim = 1)
                embeddings_b = torch.nn.functional.normalize(embeddings_b, p = 2, dim = 1)

            scale = getattr(self, "scale", 20.0)

            global _MNRL_FUSED_NOTICE_SHOWN
            if not _MNRL_FUSED_NOTICE_SHOWN:
                _MNRL_FUSED_NOTICE_SHOWN = True
                print("Unsloth: Using optimized contrastive loss for MultipleNegativesRankingLoss")

            B_a = embeddings_a.shape[0]
            B_b = embeddings_b.shape[0]

            # Decide off the real matrix size, not the batch count. For typical
            # batches the chunked kernel is just overhead, so do the dense matmul
            # + cross_entropy and only chunk once the matrix is large.
            if B_a * B_b <= _FUSED_CONTRASTIVE_MIN_ELEMENTS:
                scores = (embeddings_a @ embeddings_b.t()) * scale
                labels = torch.arange(B_a, device = scores.device)
                return torch.nn.functional.cross_entropy(scores, labels)

            return FusedContrastiveLoss.apply(embeddings_a, embeddings_b, scale)

        MultipleNegativesRankingLoss.forward = _fused_forward
        MultipleNegativesRankingLoss._original_forward = _original_forward
        # Mark patched only after success so a failed import can be retried
        _MNRL_PATCHED = True
    except Exception as e:
        import warnings
        warnings.warn(f"Unsloth: Failed to patch MultipleNegativesRankingLoss: {e}", stacklevel = 2)


_CREATE_TRANSFORMER_MODULE_LOCK = threading.RLock()


def _save_pretrained_torchao(
    self,
    save_directory,
    tokenizer = None,
    torchao_config = None,
    push_to_hub = False,
    token = None,
    repo_id = None,
    private = False,
):
    # The restore guard must also cover the torchao conversion below: fused pooling
    # swapped the encoder's final LayerNorm for Identity, and the exported weights
    # must contain the real LayerNorm.
    with _restore_fused_pooling_ln(self):
        self.save_pretrained(save_directory)

        inner_model = self[0].auto_model
        if hasattr(inner_model, "_orig_mod"):
            inner_model = inner_model._orig_mod

        # merge LoRA first
        if hasattr(inner_model, "merge_and_unload"):
            inner_model = inner_model.merge_and_unload()

        transformer_path = "0_Transformer"
        modules_path = os.path.join(save_directory, "modules.json")
        if os.path.exists(modules_path):
            try:
                with open(modules_path, "r", encoding = "utf-8") as f:
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
                transformers.AutoModelForCausalLM = original_causal
                shutil.rmtree = original_rmtree

        with patch_unsloth_save():
            unsloth_save_pretrained_torchao(
                inner_model,
                transformer_dir,
                tokenizer = tokenizer,
                torchao_config = torchao_config,
                # The inner saver treats its save dir as a hub repo id when pushing;
                # convert locally and upload the assembled ST folder below instead.
                push_to_hub = False,
                token = token,
            )

    torchao_dir = transformer_dir + "-torchao"
    if os.path.exists(torchao_dir):
        if not os.path.exists(transformer_dir):
            os.makedirs(transformer_dir, exist_ok = True)

        for item in os.listdir(torchao_dir):
            s = os.path.join(torchao_dir, item)
            d = os.path.join(transformer_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok = True)
            else:
                shutil.copy2(s, d)

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

    if push_to_hub:
        if token is None:
            token = get_token()

        api = HfApi(token = token)
        if repo_id is None:
            # legacy behavior: save_directory doubles as the repo id when pushing
            repo_id = save_directory

        print(f"Unsloth: Uploading to {repo_id}...")
        try:
            api.create_repo(repo_id = repo_id, exist_ok = True, private = private)
            api.upload_folder(
                folder_path = save_directory,
                repo_id = repo_id,
                commit_message = "Upload torchao quantized SentenceTransformer model",
            )
            print(f"Unsloth: Uploaded to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Unsloth: Upload failed: {e}")


# Thanks Etherl:
def _save_pretrained_gguf(
    self,
    save_directory,
    tokenizer = None,
    quantization_method = "fast_quantized",
    first_conversion = None,
    push_to_hub = False,
    token = None,
    max_shard_size = "5GB",
    temporary_location = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage = 0.85,
    **kwargs,
):
    """
    Saves the SentenceTransformer model to GGUF format by saving the inner transformer model,
    converting it, and placing the resulting GGUF files in the save directory.
    """
    # 1. Save standard SentenceTransformer structure (configs, modules.json, etc.)
    self.save_pretrained(save_directory)

    # 2. Extract inner transformer model (PEFT merge handled by the gguf saver)
    inner_model = self[0].auto_model
    if hasattr(inner_model, "_orig_mod"):
        inner_model = inner_model._orig_mod

    # 3. Identify where the transformer weights are stored
    transformer_path = "0_Transformer"
    modules_path = os.path.join(save_directory, "modules.json")
    if os.path.exists(modules_path):
        try:
            with open(modules_path, "r", encoding = "utf-8") as f:
                modules = json.load(f)
            for m in modules:
                if m.get("type", "").endswith("Transformer"):
                    transformer_path = m.get("path", "")
                    break
        except:
            pass

    # Unsloth saves + converts here; absolute for later commonpath comparison
    transformer_dir = os.path.join(save_directory, transformer_path)
    transformer_dir = os.path.abspath(transformer_dir)

    if tokenizer is None:
        tokenizer = self.tokenizer

    # 4. Patch environment so Unsloth treats this embedding model correctly
    @contextlib.contextmanager
    def patch_unsloth_gguf_save():
        # Prevent deletion of the directory self.save_pretrained just created
        original_rmtree = shutil.rmtree
        try:
            yield
        finally:
            shutil.rmtree = original_rmtree

    # 5. Call Unsloth's GGUF saver on the inner model targeting the transformer subdirectory
    with patch_unsloth_gguf_save():
        result = unsloth_save_pretrained_gguf(
            inner_model,
            save_directory = transformer_dir,
            tokenizer = tokenizer,
            quantization_method = quantization_method,
            first_conversion = first_conversion,
            push_to_hub = False,  # Force local first to move files
            token = token,
            max_shard_size = max_shard_size,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
        )

    # 6. Move GGUF files from the subdirectory (0_Transformer) to the root save_directory
    gguf_files = result.get("gguf_files", [])

    new_gguf_locations = []

    for gguf_file in gguf_files:
        if os.path.exists(gguf_file):
            filename = os.path.basename(gguf_file)
            dest_path = os.path.join(save_directory, filename)

            abs_gguf_file = os.path.abspath(gguf_file)

            try:
                is_subpath = os.path.commonpath([abs_gguf_file, transformer_dir]) == transformer_dir
            except ValueError:
                # Windows different-drive commonpath raises
                is_subpath = False

            if is_subpath:
                # Move the GGUF out of transformer_dir to root
                shutil.move(gguf_file, dest_path)
                new_gguf_locations.append(dest_path)
            else:
                # Elsewhere: move to root if not already there
                if os.path.abspath(dest_path) != abs_gguf_file:
                    shutil.move(gguf_file, dest_path)
                new_gguf_locations.append(dest_path)

    result["gguf_files"] = new_gguf_locations

    # 7. Add branding
    try:
        FastSentenceTransformer._add_unsloth_branding(save_directory)

        # Add GGUF details to README
        readme_path = os.path.join(save_directory, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "a", encoding = "utf-8") as f:
                f.write("\n## GGUF Quantization\n")
                f.write(
                    f"This model contains GGUF quantized versions in: {', '.join([os.path.basename(f) for f in new_gguf_locations])}\n"
                )
    except:
        pass

    # 8. Handle Push to Hub if requested
    if push_to_hub:
        if token is None:
            token = get_token()

        api = HfApi(token = token)
        # legacy behavior: save_directory doubles as the repo id when pushing
        repo_id = kwargs.get("repo_id") or save_directory

        print(f"Unsloth: Uploading to {repo_id}...")
        try:
            api.create_repo(repo_id = repo_id, exist_ok = True, private = kwargs.get("private", False))
            api.upload_folder(
                folder_path = save_directory,
                repo_id = repo_id,
                commit_message = "Upload GGUF and SentenceTransformer model",
            )
            print(f"Unsloth: Uploaded to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Unsloth: Upload failed: {e}")

    return result


def _push_to_hub_gguf(
    self,
    repo_id,
    tokenizer = None,
    quantization_method = "fast_quantized",
    first_conversion = None,
    token = None,
    private = None,
    commit_message = "Upload GGUF SentenceTransformer model trained with Unsloth",
    commit_description = "Upload GGUF model trained with Unsloth 2x faster",
    max_shard_size = "5GB",
    temporary_location = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage = 0.85,
    create_pr = False,
    revision = None,
    tags = None,
    **kwargs,
):
    """
    Converts the SentenceTransformer model to GGUF format and pushes to the Hugging Face Hub.

    This method:
    1. Saves the model locally to a temporary directory in GGUF format.
    2. Uploads the GGUF files, config, Ollama Modelfile, and README to the Hub.
    3. Cleans up the temporary directory.

    Args:
        repo_id (str): The Hugging Face Hub repo ID (e.g., "username/model-name").
        tokenizer: The tokenizer to save. Defaults to `self.tokenizer`.
        quantization_method (str or list): GGUF quantization method(s). Can be a string or list of strings.
            Choose from the following options:
            * "not_quantized"  : Recommended. Fast conversion. Slow inference, big files.
            * "fast_quantized" : Recommended. Fast conversion. OK inference, OK file size.
            * "quantized"      : Recommended. Slow conversion. Fast inference, small files.
            * "f32"     : Not recommended. Retains 100% accuracy, but super slow and memory hungry.
            * "f16"     : Fastest conversion + retains 100% accuracy. Slow and memory hungry.
            * "q8_0"    : Fast conversion. High resource use, but generally acceptable.
            * "q4_k_m"  : Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K
            * "q5_k_m"  : Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K
            * "q2_k"    : Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.
            * "q3_k_l"  : Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K
            * "q3_k_m"  : Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K
            * "q3_k_s"  : Uses Q3_K for all tensors
            * "q4_0"    : Original quant method, 4-bit.
            * "q4_1"    : Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.
            * "q4_k_s"  : Uses Q4_K for all tensors
            * "q5_0"    : Higher accuracy, higher resource usage and slower inference.
            * "q5_1"    : Even higher accuracy, resource usage and slower inference.
            * "q5_k_s"  : Uses Q5_K for all tensors
            * "q6_k"    : Uses Q8_K for all tensors
        first_conversion (str, optional): The initial conversion format before quantization.
        token (str, optional): Hugging Face token. Uses cached token if not provided.
        private (bool, optional): Whether the repo should be private.
        commit_message (str): Commit message for the upload.
        commit_description (str): Commit description for the upload.
        max_shard_size (str): Maximum shard size for saving.
        temporary_location (str): Temp directory for intermediate files.
        maximum_memory_usage (float): Max fraction of memory to use.
        create_pr (bool): Whether to create a pull request instead of pushing directly.
        revision (str, optional): Branch/revision to push to.
        tags (list, optional): Additional tags for the repo.

    Returns:
        str: The full repo ID on Hugging Face Hub.
    """
    if token is None:
        token = get_token()
    if token is None:
        raise ValueError(
            "No HF token provided. Please provide a token or login with `huggingface-cli login`"
        )

    api = HfApi(token = token)

    if "/" not in repo_id:
        username = api.whoami()["name"]
        full_repo_id = f"{username}/{repo_id}"
    else:
        full_repo_id = repo_id

    model_name = full_repo_id.split("/")[-1]

    try:
        api.create_repo(
            repo_id = full_repo_id,
            private = private,
            exist_ok = True,
            repo_type = "model",
        )
    except Exception as e:
        print(f"Unsloth Warning: Could not create repo: {e}")

    # Convert locally in a temp dir, then upload
    with tempfile.TemporaryDirectory(prefix = "unsloth_st_gguf_") as temp_dir:
        print(f"Unsloth: Converting SentenceTransformer to GGUF format...")

        result = _save_pretrained_gguf(
            self,
            save_directory = temp_dir,
            tokenizer = tokenizer,
            quantization_method = quantization_method,
            first_conversion = first_conversion,
            push_to_hub = False,  # We handle upload ourselves
            token = token,
            max_shard_size = max_shard_size,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
        )

        gguf_files = result.get("gguf_files", [])
        modelfile_location = result.get("modelfile_location", None)
        is_vlm = result.get("is_vlm", False)
        fix_bos_token = result.get("fix_bos_token", False)

        print(f"Unsloth: Uploading GGUF to https://huggingface.co/{full_repo_id}...")

        # Upload GGUF files
        for file_location in gguf_files:
            if os.path.exists(file_location):
                filename = os.path.basename(file_location)
                print(f"  Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj = file_location,
                    path_in_repo = filename,
                    repo_id = full_repo_id,
                    repo_type = "model",
                    commit_message = commit_message,
                    commit_description = commit_description,
                    create_pr = create_pr,
                    revision = revision,
                )

        # Upload Modelfile if exists
        if modelfile_location and os.path.exists(modelfile_location):
            print("  Uploading Ollama Modelfile...")
            api.upload_file(
                path_or_fileobj = modelfile_location,
                path_in_repo = "Modelfile",
                repo_id = full_repo_id,
                repo_type = "model",
                commit_message = f"{commit_message} - Ollama Modelfile",
                create_pr = create_pr,
                revision = revision,
            )

        # Upload config.json if exists
        config_path = os.path.join(temp_dir, "config.json")
        if os.path.exists(config_path):
            print("  Uploading config.json...")
            api.upload_file(
                path_or_fileobj = config_path,
                path_in_repo = "config.json",
                repo_id = full_repo_id,
                repo_type = "model",
                commit_message = f"{commit_message} - config",
                create_pr = create_pr,
                revision = revision,
            )

        # Create and upload README
        gguf_basenames = [os.path.basename(f) for f in gguf_files if os.path.exists(f)]
        readme_content = f"""---
tags:
- gguf
- llama.cpp
- unsloth
- sentence-transformers
{"- vision-language-model" if is_vlm else ""}
---

# {model_name} - GGUF

This sentence-transformers model was finetuned and converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth).

## Available Model files:
"""
        for fname in gguf_basenames:
            readme_content += f"- `{fname}`\n"

        if modelfile_location and os.path.exists(modelfile_location):
            readme_content += "\n## Ollama\n"
            readme_content += "An Ollama Modelfile is included for easy deployment.\n"

        if fix_bos_token:
            readme_content += "\n## Note\n"
            readme_content += (
                "The model's BOS token behavior was adjusted for GGUF compatibility.\n"
            )

        readme_content += (
            "\nThis was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)\n"
            '[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)\n'
        )

        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, "w", encoding = "utf-8") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj = readme_path,
            path_in_repo = "README.md",
            repo_id = full_repo_id,
            repo_type = "model",
            commit_message = "Add README",
            create_pr = create_pr,
            revision = revision,
        )

    # Add tags
    all_tags = ["gguf", "llama-cpp", "unsloth", "sentence-transformers"]
    if is_vlm:
        all_tags.append("vision-language-model")
    if tags is not None:
        if isinstance(tags, (list, tuple)):
            all_tags.extend(tags)
        else:
            all_tags.append(tags)
    try:
        api.add_tags(repo_id = full_repo_id, tags = all_tags, repo_type = "model")
    except:
        pass

    print(f"Unsloth: Successfully uploaded GGUF to https://huggingface.co/{full_repo_id}")
    return full_repo_id


class FastSentenceTransformer(FastModel):
    @staticmethod
    def _save_base_config_for_processor_resume(config, output_path):
        """sentence-transformers >= 5.4 reloads Transformer modules via
        AutoProcessor, which falls back to AutoConfig for tokenizer-only
        roots -- so PEFT adapter checkpoints still need base config.json
        next to adapter_config.json."""
        if config is None or not getattr(config, "model_type", None):
            return
        if hasattr(config, "save_pretrained"):
            config.save_pretrained(output_path)
        elif hasattr(config, "to_json_file"):
            config_path = os.path.join(output_path, "config.json")
            config.to_json_file(config_path)

    @staticmethod
    def _patch_transformer_module_save_config(transformer_module, base_config = None):
        transformer_module._unsloth_st_managed = True
        if base_config is not None and getattr(base_config, "model_type", None):
            transformer_module._unsloth_base_config = base_config

        if getattr(transformer_module, "_unsloth_save_config_patched", False):
            return transformer_module

        original_save = transformer_module.save

        def _save_with_base_config(self, output_path, *args, **kwargs):
            original_save(output_path, *args, **kwargs)
            FastSentenceTransformer._save_base_config_for_processor_resume(
                getattr(self, "_unsloth_base_config", None), output_path
            )

        transformer_module.save = types.MethodType(_save_with_base_config, transformer_module)
        transformer_module._unsloth_save_config_patched = True
        return transformer_module

    @staticmethod
    def _read_pooling_mode(model_name, token):
        """Read the pooling mode from modules.json, else return "mean"."""
        try:
            if os.path.exists(model_name) and os.path.exists(
                os.path.join(model_name, "modules.json")
            ):
                modules_json_path = os.path.join(model_name, "modules.json")
            else:
                modules_json_path = hf_hub_download(model_name, "modules.json", token = token)

            with open(modules_json_path, "r", encoding = "utf-8") as f:
                modules_config = json.load(f)

            pooling_config_path = None
            for module in modules_config:
                if module.get("type", "") == "sentence_transformers.models.Pooling":
                    pooling_path = module.get("path", "")
                    if pooling_path:
                        # find config.json for the pooling module
                        local_candidate = os.path.join(model_name, pooling_path, "config.json")
                        # containment: pooling_path comes from modules.json;
                        # never follow it outside the model directory
                        if (
                            os.path.exists(model_name)
                            and os.path.exists(local_candidate)
                            and os.path.commonpath(
                                [os.path.abspath(model_name), os.path.abspath(local_candidate)]
                            )
                            == os.path.abspath(model_name)
                        ):
                            pooling_config_path = local_candidate
                        else:
                            pooling_config_path = hf_hub_download(
                                model_name,
                                os.path.join(pooling_path, "config.json"),
                                token = token,
                            )
                        break

            if pooling_config_path:
                with open(pooling_config_path, "r", encoding = "utf-8") as f:
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

            # All pooling_mode_* flags false, or no Pooling module entry: fall
            # back instead of implicitly returning None (the caller passes the
            # result straight to Pooling(pooling_mode=...)).
            return "mean"

        except Exception as e:
            print(
                f"Failed to detect pooling mode, not a sentence-transformers model. Using default pooling mode 'mean', this may or may not work."
            )
            return "mean"

    @staticmethod
    def _patch_mpnet_v4():
        """Patch MPNetModel for gradient checkpointing (transformers 4)."""
        from transformers.models.mpnet import modeling_mpnet

        modeling_mpnet.MPNetModel.supports_gradient_checkpointing = True

        def _set_gradient_checkpointing(
            self,
            module = None,
            value = True,
        ):
            if module is None:
                module = self.encoder
            if isinstance(module, modeling_mpnet.MPNetEncoder):
                module.gradient_checkpointing = value

        modeling_mpnet.MPNetModel._set_gradient_checkpointing = _set_gradient_checkpointing

        # patch MPNetEncoder.forward for checkpointing; based on:
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
                    v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
                )
            return BaseModelOutput(
                last_hidden_state = hidden_states,
                hidden_states = all_hidden_states,
                attentions = all_attentions,
            )

        modeling_mpnet.MPNetEncoder.forward = forward

    @staticmethod
    def _patch_mpnet_v5():
        """Patch MPNetModel for gradient checkpointing (transformers 5)."""
        from transformers.models.mpnet import modeling_mpnet

        modeling_mpnet.MPNetModel.supports_gradient_checkpointing = True

        def _set_gradient_checkpointing(
            self,
            module = None,
            value = True,
        ):
            if module is None:
                module = self.encoder
            if isinstance(module, modeling_mpnet.MPNetEncoder):
                module.gradient_checkpointing = value

        modeling_mpnet.MPNetModel._set_gradient_checkpointing = _set_gradient_checkpointing

        # patch MPNetEncoder.forward for checkpointing; based on:
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
                    v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
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
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            head_mask_is_none = head_mask is None
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)

            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = (
                    attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                )
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(input_shape, device = device)  # (bs, seq_length)

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
        """Check if the model class accepts the `add_pooling_layer` argument."""
        try:
            if auto_model_class is None:
                auto_model_class = AutoModel
            model_class = resolve_model_class(auto_model_class, config)

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
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
    def _add_unsloth_tags(
        repo_id,
        token,
        tags = None,
    ):
        """Add Unsloth + sentence-transformers tags to the HF Hub repo."""
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
        """Add Unsloth branding to the sentence-transformers-generated README.md."""
        readme_path = os.path.join(save_directory, "README.md")
        if not os.path.exists(readme_path):
            return

        with open(readme_path, "r", encoding = "utf-8") as f:
            content = f.read()

        # add unsloth tag to frontmatter
        if "---\ntags:\n" in content:
            content = content.replace("---\ntags:\n", "---\ntags:\n- unsloth\n")
        else:
            # tags exist but not at the start: append via regex
            pattern = r"(^tags:\s*\n)"
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, r"\1- unsloth\n", content, count = 1, flags = re.MULTILINE)

        branding = (
            "\n\nThis model was finetuned with [Unsloth](https://github.com/unslothai/unsloth).\n\n"
            '[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)\n'
        )

        if "# SentenceTransformer" in content:
            parts = content.split("# SentenceTransformer", 1)
            content = parts[0] + "# SentenceTransformer" + branding + parts[1]
        else:
            content += branding

        with open(readme_path, "w", encoding = "utf-8") as f:
            f.write(content)

    @staticmethod
    def _module_path(model_name, token = None):
        """Return the path to the modules.json file, or None."""
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
    def _create_transformer_module(model_name, model, tokenizer, max_seq_length, trust_remote_code):
        """Helper to create and configure a Transformer module."""
        from sentence_transformers.models import Transformer

        # Prevents loading the model a second time and redirects AutoProcessor/
        # AutoTokenizer so Transformer.__init__ picks up our pre-fixed tokenizer.
        # On sentence-transformers >=5.4 `tokenizer` is a read-only @property
        # backed by `self.processor`, so a post-init assignment raises; the
        # constructor redirect sets self.processor correctly instead.
        from transformers import AutoProcessor, AutoTokenizer

        def is_requested_model_name(args, kwargs):
            requested = None
            if args:
                requested = args[0]
            else:
                requested = kwargs.get("pretrained_model_name_or_path")
                if requested is None:
                    requested = kwargs.get("model_name_or_path")
            if requested is None:
                return False

            try:
                requested = os.fspath(requested)
                expected = os.fspath(model_name)
            except (TypeError, ValueError) as exception:
                logging.debug(
                    "Unsloth: Could not normalize SentenceTransformer model path: %s",
                    exception,
                )
                return False
            if requested == expected:
                return True

            try:
                if os.path.exists(requested) or os.path.exists(expected):
                    return os.path.abspath(requested) == os.path.abspath(expected)
            except (OSError, TypeError, ValueError) as exception:
                logging.debug(
                    "Unsloth: Could not compare SentenceTransformer model paths: %s",
                    exception,
                )
            return False

        with _CREATE_TRANSFORMER_MODULE_LOCK:
            original_model_from_pretrained = AutoModel.from_pretrained
            original_processor_from_pretrained = AutoProcessor.from_pretrained
            original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

            def return_existing_model(*args, **kwargs):
                if is_requested_model_name(args, kwargs):
                    return model
                return original_model_from_pretrained(*args, **kwargs)

            def return_existing_tokenizer(*args, **kwargs):
                if is_requested_model_name(args, kwargs):
                    return tokenizer
                return original_tokenizer_from_pretrained(*args, **kwargs)

            def return_existing_processor(*args, **kwargs):
                if is_requested_model_name(args, kwargs):
                    return tokenizer
                return original_processor_from_pretrained(*args, **kwargs)

            try:
                # Temporarily redirect Auto* loading to return our pre-loaded objects
                AutoModel.from_pretrained = return_existing_model
                AutoProcessor.from_pretrained = return_existing_processor
                AutoTokenizer.from_pretrained = return_existing_tokenizer

                transformer_init_params = inspect.signature(Transformer.__init__).parameters
                trust_remote_code_kwargs = {"trust_remote_code": trust_remote_code}
                do_lower_case = getattr(tokenizer, "do_lower_case", False)
                transformer_kwargs = {"max_seq_length": max_seq_length}
                if "do_lower_case" in transformer_init_params:
                    transformer_kwargs["do_lower_case"] = do_lower_case
                if "model_kwargs" in transformer_init_params:
                    transformer_kwargs["model_kwargs"] = trust_remote_code_kwargs.copy()
                    transformer_kwargs["config_kwargs"] = trust_remote_code_kwargs.copy()
                else:
                    transformer_kwargs["model_args"] = trust_remote_code_kwargs.copy()
                    transformer_kwargs["config_args"] = trust_remote_code_kwargs.copy()
                if "processor_kwargs" in transformer_init_params:
                    transformer_kwargs["processor_kwargs"] = trust_remote_code_kwargs.copy()
                elif "tokenizer_args" in transformer_init_params:
                    transformer_kwargs["tokenizer_args"] = trust_remote_code_kwargs.copy()

                transformer_module = Transformer(model_name, **transformer_kwargs)
            finally:
                # Restore original Auto* loading immediately
                AutoModel.from_pretrained = original_model_from_pretrained
                AutoProcessor.from_pretrained = original_processor_from_pretrained
                AutoTokenizer.from_pretrained = original_tokenizer_from_pretrained

        # On sentence-transformers >=5.4 `tokenizer` is a read-only property backed
        # by `self.processor` (already wired via the redirect above). On older
        # versions it's a regular attribute and the explicit assignment is required.
        if not isinstance(getattr(type(transformer_module), "tokenizer", None), property):
            transformer_module.tokenizer = tokenizer
        transformer_module.do_lower_case = getattr(tokenizer, "do_lower_case", False)

        # sentence-transformers only passes along known keys to model.forward
        preinit_model_forward_params = getattr(transformer_module, "model_forward_params", set())
        model_forward_params = list(inspect.signature(model.forward).parameters)
        transformer_module.model_forward_params = set(model_forward_params) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
            "return_dict",
        }
        transformer_module.model_forward_params |= preinit_model_forward_params

        # determine max_seq_length if not provided
        if max_seq_length is None:
            if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
                max_seq_length = model.config.max_position_embeddings
            elif hasattr(tokenizer, "model_max_length"):
                max_seq_length = tokenizer.model_max_length
            else:
                max_seq_length = 512

        transformer_module.max_seq_length = max_seq_length
        config_keys = list(getattr(transformer_module, "config_keys", []) or [])
        for config_key in ("max_seq_length", "do_lower_case"):
            if config_key not in config_keys:
                config_keys.append(config_key)
        transformer_module.config_keys = config_keys
        transformer_module.save_in_root = True
        FastSentenceTransformer._patch_transformer_module_save_config(
            transformer_module, getattr(model, "config", None)
        )

        if hasattr(model, "config"):
            model.config.tokenizer_class = tokenizer.__class__.__name__

        return transformer_module

    @staticmethod
    def _is_transformer_module_ref(class_ref):
        if class_ref in {
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.transformer.Transformer",
            "sentence_transformers.base.modules.transformer.Transformer",
        }:
            return True

        try:
            from sentence_transformers.models import Transformer
            from sentence_transformers.util import import_from_string

            module_class = import_from_string(class_ref)
            return module_class is Transformer
        except (ImportError, AttributeError, TypeError, ValueError) as exception:
            logging.debug(
                "Unsloth: Could not resolve SentenceTransformer module ref %r: %s",
                class_ref,
                exception,
            )
            return False

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
        """Load modules from modules.json, else fall back to hard-coded modules.

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
                name = module_config.get("name", str(module_config.get("idx", len(modules))))

                if FastSentenceTransformer._is_transformer_module_ref(class_ref):
                    transformer_module = FastSentenceTransformer._create_transformer_module(
                        model_name,
                        model,
                        tokenizer,
                        max_seq_length,
                        trust_remote_code,
                    )
                    modules[name] = transformer_module
                else:
                    # load other modules (Pooling, Normalize, etc.)
                    module_path = module_config["path"]
                    if os.path.isdir(model_name):
                        load_path = os.path.join(model_name, module_path)
                    else:
                        try:
                            load_path = load_dir_path(model_name, module_path, token = token)
                        except Exception as e:
                            print(f"Unsloth Warning: Could not download module {module_path}: {e}")
                            continue

                    module_class = import_from_string(class_ref)
                    try:
                        module = module_class.load(load_path)
                        modules[name] = module
                    except Exception as e:
                        print(f"Unsloth Warning: Failed to load module {name} ({class_ref}): {e}")

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

        modules["1"] = Pooling(word_embedding_dimension = hidden_size, pooling_mode = pooling_mode)
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

    # get_peft_model LoRA target defaults for encoder families whose projection
    # names differ from the BERT-style query/key/value/dense fallback.
    LORA_TARGET_MODULE_DEFAULTS = {
        "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
        "mpnet": ["q", "k", "v", "o", "dense"],
        "modernbert": ["Wqkv", "Wo", "Wi"],
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
        if batch_size is not None or grad_accum is not None or max_seq_length is not None:
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
            if isinstance(getattr(type(model[0]), "auto_model", None), property):
                model[0].model = compiled
            else:
                model[0].auto_model = compiled
            # Fix for accelerate unwrap_model bug:
            # When SentenceTransformer contains a compiled inner model,
            # accelerate checks has_compiled_regions() which returns True,
            # then tries to access model.__dict__["_orig_mod"] which fails.
            # This workaround sets _orig_mod to satisfy accelerate.
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
        _patch_dense_dtype()

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

        is_encoder_model = model_type.lower() in FastSentenceTransformer.ENCODER_MODEL_TYPES
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
                print("Unsloth: Device does not support bfloat16. Using float16 instead.")
                dtype = torch.float16

            st_device = device_map
            if isinstance(st_device, dict) or (
                isinstance(st_device, str) and st_device in ["auto", "sequential"]
            ):
                st_device = "cuda"

            # dtype kwarg: delegate the torch_dtype/dtype version gating to the
            # shared helper (keys off the actual config doc, not a hardcoded
            # transformers version).
            model_kwargs = add_dtype_kwargs(dtype, {})

            auto_model = kwargs.get("auto_model", AutoModel)
            _disable_sdpa = any(_m in model_type.lower() for _m in DISABLE_SDPA_MODEL_NAMES)

            # flash_attn_2 selection stays layered on top of the shared resolver:
            # only request it at load time when the model class itself supports it
            # (transformers 4.x: _supports_flash_attn_2, 5.x: _supports_flash_attn).
            # Forcing it for other types (e.g. MPNet) makes from_pretrained reject
            # the requested implementation before the Unsloth varlen patch could
            # ever take over.
            supports_flash_attn_2 = False
            if not _disable_sdpa:
                if config is not None:
                    try:
                        model_class = resolve_model_class(auto_model, config)
                        supports_flash_attn_2 = bool(
                            getattr(model_class, "_supports_flash_attn_2", False)
                            or getattr(model_class, "_supports_flash_attn", False)
                        )
                    except:
                        supports_flash_attn_2 = False
                if supports_flash_attn_2:
                    try:
                        import flash_attn  # noqa: F401
                    except ImportError:
                        supports_flash_attn_2 = False

            if supports_flash_attn_2:
                attn_impl = "flash_attention_2"
            else:
                # sdpa / eager / None (incl. DISABLE_SDPA_MODEL_NAMES handling) via
                # the shared encoder resolver in _utils.
                attn_impl = resolve_encoder_attention_implementation(
                    auto_model, config, model_type, DISABLE_SDPA_MODEL_NAMES
                )

            if attn_impl is not None:
                model_kwargs["attn_implementation"] = attn_impl

            if attn_impl == "flash_attention_2":
                attn_str = " + FlashAttention 2"
            elif attn_impl == "sdpa":
                attn_str = " + SDPA"
            else:
                attn_str = ""

            if load_in_4bit:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} with 4-bit quantization{attn_str}"
                )
            elif load_in_8bit:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} with 8-bit quantization{attn_str}"
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
            elif load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(load_in_8bit = True)
                model_kwargs["quantization_config"] = bnb_config
                st_device = None

            _use_gc = use_gradient_checkpointing
            if _use_gc and _use_gc != False:
                print("Unsloth Warning: Gradient checkpointing is incompatible with torch.compile.")
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
            FastSentenceTransformer._patch_transformer_module_save_config(
                st_model[0], getattr(st_model[0].auto_model, "config", None)
            )

            if compile_mode is None and _HAS_FAST_LAYERNORM:
                inner_model = st_model[0].auto_model
                ln_count = _patch_encoder_layernorms(inner_model)
                if ln_count > 0:
                    print(f"Unsloth: Patched {ln_count} LayerNorm modules with Triton kernel")

            if compile_mode is None and _HAS_FUSED_POOLING:
                if _patch_fused_pooling(st_model):
                    print("Unsloth: Fused final LayerNorm + Mean Pooling into single Triton kernel")

            _unpad_env = os.environ.get("UNSLOTH_UNPADDING", "1")
            if _unpad_env == "1":
                if _patch_unpadded_encoder(st_model, model_type):
                    backend = getattr(st_model[0], "_unpadding_backend", "unknown")
                    print(f"Unsloth: Enabled variable-length batching (unpadding) via {backend}")

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
                    # modules.json points the Transformer module at "" (root) on
                    # modern sentence-transformers, "0_Transformer" on legacy layouts;
                    # the merged weights must land where reload looks for them.
                    transformer_path = ""
                    modules_path = os.path.join(save_directory, "modules.json")
                    if os.path.exists(modules_path):
                        try:
                            with open(modules_path, "r", encoding = "utf-8") as f:
                                modules = json.load(f)
                            for m in modules:
                                if m.get("type", "").endswith("Transformer"):
                                    transformer_path = m.get("path", "")
                                    break
                        except:
                            pass
                    transformer_dir = os.path.join(save_directory, transformer_path)
                    if hasattr(self[0], "auto_model"):
                        inner = self[0].auto_model
                        if hasattr(inner, "_orig_mod"):
                            inner = inner._orig_mod
                        if getattr(self, "_sparsity_applied", False):
                            _remove_sparsity_from_base_weights(inner)
                        if hasattr(inner, "merge_and_unload"):
                            merged = inner.merge_and_unload()
                            merged.save_pretrained(transformer_dir)
                            # self.save_pretrained above wrote the PEFT adapter into
                            # the module dir; drop it so reload uses the merged
                            # weights instead of applying the adapter on top.
                            for _adapter_file in (
                                "adapter_model.safetensors",
                                "adapter_model.bin",
                                "adapter_config.json",
                            ):
                                _adapter_path = os.path.join(transformer_dir, _adapter_file)
                                if os.path.exists(_adapter_path):
                                    try:
                                        os.remove(_adapter_path)
                                    except OSError:
                                        pass
                        elif hasattr(inner, "save_pretrained"):
                            inner.save_pretrained(transformer_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(save_directory)
                    FastSentenceTransformer._add_unsloth_branding(save_directory)

            st_model.save_pretrained_merged = types.MethodType(_save_pretrained_merged, st_model)

            st_model.save_pretrained_torchao = types.MethodType(_save_pretrained_torchao, st_model)

            st_model.save_pretrained_gguf = types.MethodType(_gguf_save_with_restore, st_model)

            st_model.push_to_hub_gguf = types.MethodType(_gguf_push_with_restore, st_model)

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
                        commit_message = push_kwargs.get("commit_message", "Upload model"),
                    )
                print(f"Unsloth: Pushed to https://huggingface.co/{repo_id}")

            st_model.push_to_hub_merged = types.MethodType(_push_to_hub_merged, st_model)

            return st_model

        if is_encoder_model and load_in_4bit:
            print("Unsloth Warning: 4-bit quantization adds ~2.3x overhead for encoder models.")
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

        has_modules_json = FastSentenceTransformer._module_path(model_name, token) is not None

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
        _is_encoder_decoder = False
        for _mod in st_model:
            if hasattr(_mod, "auto_model"):
                _am = _mod.auto_model
                _am_unwrap = _am._orig_mod if hasattr(_am, "_orig_mod") else _am
                _cfg = getattr(_am_unwrap, "config", None)
                _is_bidirectional = getattr(_cfg, "use_bidirectional_attention", False)
                _is_encoder_decoder = bool(getattr(_cfg, "is_encoder_decoder", False))
                if (
                    _cfg is not None
                    and getattr(_cfg, "_attn_implementation", None) == "flex_attention"
                ):
                    if _is_bidirectional:
                        # flex_attention's create_block_mask Triton kernel can crash
                        # with illegal memory access on variable-length batches during
                        # training. SDPA handles sliding window via explicit masks and
                        # is stable for sentence transformer workloads.
                        _cfg._attn_implementation = "sdpa"
                        if hasattr(_cfg, "attn_implementation"):
                            _cfg.attn_implementation = "sdpa"
                        print(
                            "Unsloth: Overriding flex_attention -> sdpa for "
                            "bidirectional sentence transformer training"
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
                print(f"Unsloth: Patched {_ln_count} LayerNorm modules with Triton kernel")

        if _HAS_FUSED_POOLING:
            if _patch_fused_pooling(st_model):
                print("Unsloth: Fused final LayerNorm + Mean Pooling into single Triton kernel")

        # Enable variable-length batching (unpadding) for causal decoders. Encoders
        # take the fast-encoder path, so anything reaching here that is neither
        # bidirectional nor an encoder-decoder is a causal decoder. We do NOT gate on
        # config.is_decoder: that flag is the encoder-decoder cross-attention marker
        # and is always False on decoder-only LLMs (Qwen3 / Llama / Mistral), which
        # silently disabled this path for every model it was built for. Boundary
        # correctness is now enforced inside _patch_unpadded_decoder (varlen under
        # flash-attn-2, block-diagonal mask under sdpa), so the broad gate is safe.
        _unpad_env = os.environ.get("UNSLOTH_UNPADDING", "1")
        if _unpad_env == "1" and not _is_bidirectional and not _is_encoder_decoder:
            if _patch_unpadded_decoder(st_model):
                _backend = getattr(st_model[0], "_unpadding_backend", "unknown")
                print(f"Unsloth: Enabled variable-length batching (unpadding) via {_backend}")

        def _save_pretrained_merged(self, save_directory, **kwargs):
            with _restore_fused_pooling_ln(self):
                # check which adapter files exist before save_pretrained
                adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
                existing_before = {
                    f for f in adapter_files if os.path.exists(os.path.join(save_directory, f))
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
                    # A plain (non-PEFT) encoder has no merge_and_unload; fall
                    # back to a straight save instead of crashing.
                    if hasattr(inner_auto, "merge_and_unload"):
                        merged_model = inner_auto.merge_and_unload()
                    else:
                        merged_model = inner_auto
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

        st_model.save_pretrained_merged = types.MethodType(_save_pretrained_merged, st_model)

        st_model.save_pretrained_torchao = types.MethodType(_save_pretrained_torchao, st_model)

        st_model.save_pretrained_gguf = types.MethodType(_gguf_save_with_restore, st_model)

        st_model.push_to_hub_gguf = types.MethodType(_gguf_push_with_restore, st_model)

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
            print(f"Unsloth: Successfully pushed merged model to https://huggingface.co/{repo_id}")

        st_model.push_to_hub_merged = types.MethodType(_push_to_hub_merged, st_model)
        return st_model

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = None,  # resolved per model_type; BERT-style names by default
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

        if target_modules is None:
            # DistilBERT/MPNet/ModernBERT use different projection names than the
            # BERT-style default; with the wrong names PEFT finds no targets at all
            # (DistilBERT) or silently adapts only the FFN (MPNet).
            _tm_cfg = None
            if isinstance(model, SentenceTransformer):
                for _tm_mod in model:
                    if hasattr(_tm_mod, "auto_model"):
                        _tm_cfg = getattr(_tm_mod.auto_model, "config", None)
                        break
            else:
                _tm_cfg = getattr(model, "config", None)
            _tm_type = str(getattr(_tm_cfg, "model_type", "")).lower()
            target_modules = FastSentenceTransformer.LORA_TARGET_MODULE_DEFAULTS.get(
                _tm_type, ["query", "key", "value", "dense"]
            )

        if isinstance(model, SentenceTransformer):
            # Check if this is a fast encoder model (uses torch.compile instead of Unsloth patching)
            is_fast_encoder = getattr(model, "_unsloth_fast_encoder", False)

            if is_fast_encoder:
                transformer_module = model[0]
                inner_model = transformer_module.auto_model

                is_quantized = (
                    getattr(inner_model, "is_quantized", False)
                    or getattr(inner_model.config, "quantization_config", None) is not None
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
                        use_gradient_checkpointing if use_gradient_checkpointing else False
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

                # Re-assign the peft model back to the transformer module.
                # On sentence-transformers >=5.4 `auto_model` is a read-only property
                # backed by `self.model`, so write to the backing attribute there.
                if isinstance(getattr(type(transformer_module), "auto_model", None), property):
                    transformer_module.model = peft_model
                else:
                    transformer_module.auto_model = peft_model
                FastSentenceTransformer._patch_transformer_module_save_config(
                    transformer_module, getattr(inner_model, "config", None)
                )

                sparsity_env = os.environ.get("UNSLOTH_SPARSITY", "auto").lower()
                # Opt-in: 2:4 magnitude pruning alters the frozen base weights, so
                # it is only applied when explicitly enabled with UNSLOTH_SPARSITY=1.
                # The default ("auto") and "0" leave the base weights untouched.
                do_sparsity = sparsity_env == "1"
                _is_full_ft = getattr(model, "_full_finetuning", False)

                if do_sparsity and not _is_full_ft:
                    supported, sparsity_msg = _check_sparsity_support()
                    if supported:
                        sparse_count = _apply_sparsity_to_base_weights(peft_model)
                        if sparse_count > 0:
                            model._sparsity_applied = True
                            print(
                                f"Unsloth: Applied 2:4 sparsity to {sparse_count} base layer(s) ({sparsity_msg})"
                            )
                    else:
                        print(
                            f"Unsloth Warning: UNSLOTH_SPARSITY=1 but not supported: {sparsity_msg}"
                        )
                elif sparsity_env == "auto" and not _is_full_ft:
                    # One-time discoverability hint (sparsity is opt-in).
                    supported, _sparsity_msg = _check_sparsity_support()
                    if supported:
                        print(
                            "Unsloth: 2:4 sparsity is available (disabled by default). It can speed up training but may lower accuracy. "
                            "Set UNSLOTH_SPARSITY=1 to enable (alters frozen base weights)."
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

                    print("Unsloth: torch.compile disabled (gradient checkpointing enabled)")

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

            # re-assign the peft model back to the transformer module.
            # On sentence-transformers >=5.4 `auto_model` is a read-only property
            # backed by `self.model`, so write to the backing attribute there.
            if isinstance(getattr(type(transformer_module), "auto_model", None), property):
                transformer_module.model = peft_model
            else:
                transformer_module.auto_model = peft_model
            FastSentenceTransformer._patch_transformer_module_save_config(
                transformer_module, getattr(inner_model, "config", None)
            )
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
                    getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
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

        # Only mutate dataloader settings for Unsloth's own models; a plain
        # SentenceTransformerTrainer must keep the user's args
        if (
            isinstance(model, FastSentenceTransformer)
            and hasattr(self, "args")
            and self.args is not None
        ):
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


def _patch_st_trainer_load_from_checkpoint():
    try:
        from sentence_transformers import SentenceTransformerTrainer
    except ImportError:
        return
    if getattr(SentenceTransformerTrainer, "_unsloth_load_from_checkpoint_patched", False):
        return
    if not hasattr(SentenceTransformerTrainer, "_load_from_checkpoint"):
        return

    _original = SentenceTransformerTrainer._load_from_checkpoint

    def _unsloth_load_from_checkpoint(self, checkpoint_path):
        try:
            from peft import PeftModel, load_peft_weights, set_peft_model_state_dict
        except ImportError:
            return _original(self, checkpoint_path)

        try:
            mod0 = self.model[0]
        except (IndexError, TypeError):
            return _original(self, checkpoint_path)

        if isinstance(getattr(type(mod0), "auto_model", None), property):
            inner = getattr(mod0, "model", None)
        else:
            inner = getattr(mod0, "auto_model", None)
        inner = getattr(inner, "_orig_mod", inner)

        if not isinstance(inner, PeftModel):
            return _original(self, checkpoint_path)
        if not getattr(mod0, "_unsloth_st_managed", False):
            return _original(self, checkpoint_path)

        if not any(
            os.path.isfile(os.path.join(checkpoint_path, fn))
            for fn in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            return _original(self, checkpoint_path)

        adapter_name = getattr(inner, "active_adapter", None)
        if adapter_name is None and callable(getattr(inner, "active_adapters", None)):
            adapter_name = inner.active_adapters()
        if isinstance(adapter_name, (list, tuple, set)):
            if len(adapter_name) != 1:
                raise RuntimeError("Unsloth: Cannot resume multiple active PEFT adapters.")
            adapter_name = next(iter(adapter_name))
        adapter_name = adapter_name or "default"
        if adapter_name not in getattr(inner, "peft_config", {}):
            raise RuntimeError(f"Unsloth: PEFT adapter {adapter_name!r} is not loaded.")

        load_result = set_peft_model_state_dict(
            inner, load_peft_weights(checkpoint_path), adapter_name = adapter_name
        )
        unexpected = getattr(load_result, "unexpected_keys", []) or []
        missing = [
            x
            for x in (getattr(load_result, "missing_keys", []) or [])
            if f".{adapter_name}." in x or x.endswith(f".{adapter_name}")
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Unsloth: PEFT checkpoint does not match the active adapter "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]})."
            )

        modules_json = os.path.join(checkpoint_path, "modules.json")
        if not os.path.isfile(modules_json):
            raise RuntimeError("Unsloth: PEFT checkpoint is missing modules.json.")
        try:
            with open(modules_json, "r") as f:
                module_configs = json.load(f)
        except Exception as e:
            raise RuntimeError("Unsloth: Cannot parse checkpoint modules.json.") from e

        root = os.path.abspath(os.fspath(checkpoint_path))
        restored = set()
        for entry in module_configs:
            idx = int(entry.get("idx", -1))
            if idx == 0:
                continue
            if idx < 0 or idx >= len(self.model):
                raise RuntimeError(f"Unsloth: Bad module index in modules.json: {idx}.")
            module = self.model[idx]
            module_cls = type(module)
            saved_type = entry.get("type", "")
            if saved_type and not saved_type.endswith(f".{module_cls.__name__}"):
                raise RuntimeError(f"Unsloth: Checkpoint module {idx} type mismatch.")
            module_path = entry.get("path")
            module_dir = os.path.abspath(os.path.join(root, os.fspath(module_path or "")))
            try:
                inside_root = os.path.commonpath([root, module_dir]) == root
            except ValueError:
                inside_root = False
            if not module_path or not inside_root or not os.path.isdir(module_dir):
                raise RuntimeError(f"Unsloth: Bad checkpoint module path for index {idx}.")
            if not hasattr(module_cls, "load"):
                raise RuntimeError(f"Unsloth: Module {idx} cannot be reloaded.")
            fresh = module_cls.load(module_dir)
            if not isinstance(fresh, module_cls):
                raise RuntimeError(f"Unsloth: Module {idx} reload returned wrong type.")
            # Parameterless modules (Pooling, Normalize) make
            # next(module.parameters()) raise StopIteration; route through
            # the SentenceTransformer's device property instead.
            try:
                fresh.to(self.model.device)
            except AttributeError:
                pass
            self.model[idx] = fresh
            restored.add(idx)
        missing_idx = sorted(set(range(1, len(self.model))) - restored)
        if missing_idx:
            raise RuntimeError(
                f"Unsloth: Checkpoint modules.json is incomplete (missing idx={missing_idx[:8]})."
            )

    SentenceTransformerTrainer._load_from_checkpoint = _unsloth_load_from_checkpoint
    SentenceTransformerTrainer._unsloth_load_from_checkpoint_patched = True


_patch_sentence_transformer_trainer()
_patch_st_trainer_load_from_checkpoint()
