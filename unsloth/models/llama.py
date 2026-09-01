# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import gc
import math
import functools
from typing import Optional, Tuple, List, Union

from ._utils import *
from ._utils import apply_unsloth_gradient_checkpointing
from ._utils import __version__, importlib_version
from ._utils import move_to_device
from ._utils import (
    _get_inference_mode_context_manager,
    _prepare_model_for_qat,
    is_bfloat16_supported,
    get_quant_type,
    resolve_model_class,
)
from .loader_utils import (
    DEFAULT_DEVICE_MAP,
    _exclude_rope_inv_freq_from_ddp,
    _get_fp8_mode_and_check_settings,
    _restore_dropped_fp8_scales,
    planner_class_mismatch_reason,
    planner_model_class,
    planner_config_overrides,
    planner_hub_kwargs,
    planner_kwargs_with_max_memory,
    planner_quantization_kwargs,
    requested_device_map,
    resolve_unsloth_device_map,
)
from ..utils.packing import (
    get_packed_info_from_kwargs,
    mask_packed_boundary_labels,
    mask_packed_sequence_boundaries,
)
from ..utils.attention_dispatch import (
    AttentionConfig,
    AttentionContext,
    HAS_XFORMERS,
    run_attention,
    SDPA,
    select_attention_backend,
    resolve_prefix_seg_info,
)
from torch.nn.functional import scaled_dot_product_attention
from transformers import __version__ as transformers_version
from unsloth_zoo.utils import Version, _get_dtype
from unsloth_zoo.hf_utils import (
    dtype_from_config,
    add_dtype_kwargs,
    fix_lora_auto_mapping,
)
from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

try:
    from unsloth_zoo.device_map_planner import detect_logit_transforms
except ImportError:
    # Older unsloth_zoo: fall back to reading the config fields directly.
    detect_logit_transforms = None
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
    get_device_stats,
    clean_gpu_cache,
    get_current_device,
)

transformers_version = Version(transformers_version)
# Transformers moved rotary embeddings out of all attention layers.
IS_ATTENTION_REFACTOR = transformers_version > Version("4.47.1")
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except:
    GradientCheckpointingLayer = type(None)

from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from unsloth.models._attn_mask_compat import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ..kernels import *
from ..tokenizer_utils import *
from .vision import FastBaseModel

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoConfig,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import set_seed as transformers_set_seed
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from peft import PeftModelForCausalLM, PeftModelForSequenceClassification
from ..save import patch_saving_functions
import re, os, inspect, math, sys
import types

try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25.
    from huggingface_hub.utils._token import get_token
from triton import __version__ as triton_version

# Not `xformers is not None`: attention_dispatch turns HAS_XFORMERS off when the library imports
# but has no kernel that runs here. Recomputing from the import alone left the model on the
# xFormers path, and Mistral then skipped the 4D sliding-window mask, so every sequence longer
# than config.sliding_window attended to the whole causal history on SDPA.
BlockDiagonalCausalMask = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None


def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V


def original_apply_o(self, X):
    O = self.o_proj(X)
    return O


from math import sqrt as math_sqrt

KV_CACHE_INCREMENT = 512
torch_nn_functional_softmax = torch.nn.functional.softmax
# SDPA has GQA internally.
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__

from peft.utils.other import ModulesToSaveWrapper


def _offload_frozen_module_for_training(
    module: ModulesToSaveWrapper,
    device_type: str,
    offload_device: Optional[str] = "cpu",
    original_device = None,
) -> None:
    """Move the trainable copy to ``device_type`` and offload the frozen original.

    float16 is promoted to float32 for GPU compatibility (e.g. Tesla T4).
    ``offload_device`` currently only supports "cpu"; None leaves the frozen
    module in place. ``original_device`` is the placement recorded before any
    disk offload rebuilt the module, used when the copy no longer carries one.
    Modifies ``module`` in-place.
    See https://github.com/unslothai/unsloth/pull/1200 (Tesla T4 float32).
    """
    if not hasattr(module, "modules_to_save"):
        return None

    new_dtype = module.modules_to_save.default.weight.dtype
    if new_dtype == torch.float16:
        # Tesla T4 must use float32 and not float16; see #1200.
        new_dtype = torch.float32

    # `device_type` is bare ("cuda") and .to("cuda") resolves to the CURRENT device, so on a split
    # model it drags the trainable copy off its card. Prefer the index the copy already has; under
    # use_gradient_checkpointing="unsloth" it has none (rebuilt on CPU), so fall back to the
    # placement recorded before that, then the bare type.
    wanted_type = torch.device(device_type).type
    target_device = device_type
    for candidate in (module.modules_to_save.default.weight.device, original_device):
        if candidate is not None and torch.device(candidate).type == wanted_type:
            target_device = candidate
            break

    module.modules_to_save.default.to(device = target_device, dtype = new_dtype, non_blocking = True)
    module.modules_to_save.default.requires_grad_(True)

    if offload_device is not None:
        module.original_module.to(device = offload_device, non_blocking = True)
    module.original_module.requires_grad_(False)


def _fast_prepare_inputs_for_generation(
    self,
    input_ids,
    attention_mask = None,
    inputs_embeds = None,
    **kwargs,
):
    past_key_values = kwargs.get("past_key_values", None)
    original_attention_mask = attention_mask

    # Only use inputs_embeds on the first step (no cache); fixes #3798.
    use_inputs_embeds = inputs_embeds is not None and past_key_values is None

    if input_ids is not None and input_ids.numel() > 0:
        bs, seq_length = input_ids.shape
        device = input_ids.device
    elif inputs_embeds is not None:
        bs, seq_length, _ = inputs_embeds.shape
        device = inputs_embeds.device
    else:
        bs, seq_length = 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if past_key_values is not None:
        if len(past_key_values) == 0:
            past_key_values = None
            kwargs["past_key_values"] = None
            use_inputs_embeds = inputs_embeds is not None
        elif hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
            past_key_values = None
            kwargs["past_key_values"] = None
            use_inputs_embeds = inputs_embeds is not None
        else:
            if input_ids is not None and input_ids.numel() > 0:
                bs = input_ids.shape[0]
                input_ids = input_ids[:, [-1]]
                device = input_ids.device
                seq_length = 1
            elif inputs_embeds is not None:
                bs, seq_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                bs, seq_length = 1, 0
                device = "cuda" if torch.cuda.is_available() else "cpu"

            if hasattr(past_key_values, "get_seq_length"):
                past_len = int(past_key_values.get_seq_length())
            else:
                # Legacy tuple cache: (layer, (K, V)).
                past_len = int(past_key_values[0][0].shape[-2])

            max_cache_len = None
            if hasattr(past_key_values, "get_max_cache_shape"):
                m = past_key_values.get_max_cache_shape()
                max_cache_len = int(m) if m is not None and m > 0 else None
            elif hasattr(past_key_values, "get_max_length"):
                m = past_key_values.get_max_length()
                max_cache_len = int(m) if m is not None else None

            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                kwargs["cache_position"] = torch.arange(
                    past_len,
                    past_len + seq_length,
                    device = device,
                    dtype = torch.long,
                )
            else:
                if hasattr(cache_position, "device") and cache_position.device != device:
                    kwargs["cache_position"] = cache_position.to(device)

            base_model = self
            if hasattr(base_model, "base_model_prefix"):
                base_model = getattr(base_model, base_model.base_model_prefix)

            if hasattr(base_model, "_prepare_4d_causal_attention_mask_with_cache_position"):
                if not hasattr(base_model, "_unsloth_mask_needs_device"):

                    def _check_needs_device(fn) -> bool:
                        try:
                            sig = inspect.signature(inspect.unwrap(fn))
                            return "device" in sig.parameters
                        except:
                            # transformers <= 4.51.3 includes the device arg; later versions do not.
                            return transformers_version < Version("4.52.0")

                    base_model._unsloth_mask_needs_device = _check_needs_device(
                        base_model._prepare_4d_causal_attention_mask_with_cache_position
                    )

                if max_cache_len is not None:
                    target_length = max_cache_len
                elif original_attention_mask is not None and original_attention_mask.dim() == 2:
                    target_length = original_attention_mask.shape[-1]
                else:
                    target_length = past_len + seq_length

                mask_kwargs = {
                    "sequence_length": seq_length,
                    "target_length": target_length,
                    "dtype": self.dtype,
                    "cache_position": kwargs["cache_position"],
                    "batch_size": bs,
                    "config": self.config,
                    "past_key_values": past_key_values,
                }
                if base_model._unsloth_mask_needs_device:
                    mask_kwargs["device"] = device

                attention_mask = base_model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    **mask_kwargs,
                )
            else:
                if transformers_version <= Version("4.52.4"):
                    logger.warning_once(
                        f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                        "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                        "writing code, see Llama for an example implementation. If you're a user, please report this "
                        "issue on GitHub."
                    )

    if kwargs.get("position_ids", None) is None:
        if original_attention_mask is not None and original_attention_mask.dim() == 2:
            position_ids = original_attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(original_attention_mask == 0, 1)
            position_ids = position_ids[:, -seq_length:]
            kwargs["position_ids"] = position_ids
        elif kwargs.get("cache_position", None) is not None:
            cp = kwargs["cache_position"]
            if cp.dim() == 1:
                cp = cp.unsqueeze(0).expand(bs, -1)
            kwargs["position_ids"] = cp

    result = {
        "attention_mask": attention_mask,
        **kwargs,
    }
    if use_inputs_embeds:
        result["inputs_embeds"] = inputs_embeds
        result["input_ids"] = None
    else:
        result["input_ids"] = input_ids
    return result


def fix_prepare_inputs_for_generation(module):
    if hasattr(module, "prepare_inputs_for_generation"):
        module.prepare_inputs_for_generation = _fast_prepare_inputs_for_generation


torch_matmul = torch.matmul


def LlamaAttention_fast_forward_inference(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
    rotary_seq_len = None,
):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L406
    Fast inference using KV cache.
    QK^T can be computed in 4 chunks

    [Q, q] @ [K, k].T where q, k are the new tokens.
    [QK^T, Qk^T]
    [qK^T, qk^T]

    Since the attention mask wipes Qk^T, we just get
    [QK^T,    0]
    [qK^T, qk^T]

    Since softmax is row-wise, we get
    softmax([QK^T,    0])
    softmax([qK^T, qk^T])

    We then multiply by   [V]
                          [v]
    softmax([QK^T,    0]) [softmax(QK^T)V] *
    softmax([qK^T, qk^T]) [softmax([qK^T, qk^T]) @ [V, v]]

    But notice * [softmax(QK^T)V] is just the last attention.
    We just need to compute the last final row.

    This means we can pass in a row of Q, but we need to
    remember K and V, which are called the KV cache.
    """
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    K1, V1 = past_key_value
    dtype = Xn.dtype

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim

    hidden_size = self.config.hidden_size
    attention_size = n_heads * head_dim
    seq_len = K1.shape[-2]
    kv_seq_len = seq_len + 1

    device = hidden_states.device
    if do_prefill:
        self.paged_attention = torch.empty(
            (KV_CACHE_INCREMENT + seq_len + 1, 2, bsz, n_kv_heads, head_dim),
            dtype = dtype,
            device = device,
        )
        self.paged_attention_K = self.paged_attention[:, 0]
        self.paged_attention_V = self.paged_attention[:, 1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads * head_dim), dtype = dtype, device = device)
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)

        # Mistral Nemo 12b has weird dimensions.
        if attention_size != hidden_size:
            self.temp_O = torch.empty((bsz, 1, hidden_size), dtype = dtype, device = device)
        else:
            self.temp_O = self.temp_QA[1][:, :, :hidden_size]

        self.attention = torch.empty(
            (bsz, n_heads, 1, KV_CACHE_INCREMENT + seq_len), dtype = dtype, device = device
        )
        self.scalar = 1.0 / math_sqrt(self.head_dim)
        self.half_head_dim = head_dim // 2
    elif kv_seq_len >= self.paged_attention.shape[0]:
        self.paged_attention.resize_(
            (
                self.paged_attention.shape[0] + KV_CACHE_INCREMENT,
                2,
                bsz,
                n_kv_heads,
                head_dim,
            )
        )
        self.paged_attention_K = self.paged_attention[:, 0]
        self.paged_attention_V = self.paged_attention[:, 1]
        self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1] + KV_CACHE_INCREMENT))

    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])
    Kn = fast_linear_forward(self.k_proj, Xn, out = self.temp_KV[0])
    Vn = fast_linear_forward(self.v_proj, Xn, out = self.temp_KV[1])
    Qn = Qn.view(bsz, 1, n_heads, head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)


    # Must be done 2 steps before hitting full on a short KV cache, or it errors.
    if position_ids.dim() == 1:
        position_ids = position_ids[:, None]
    # Transformers 5.x accumulates position_ids as [batch, full_seq_len] across decode steps;
    # single-token inference only needs the last position.
    if position_ids.shape[-1] > 1:
        position_ids = position_ids[:, -1:]
    position_ids = position_ids.to(Qn.device)

    if rotary_seq_len is None:
        rotary_seq_len = max(kv_seq_len, int(position_ids.max().item()) + 1)
    self.rotary_emb.extend_rope_embedding(Vn, rotary_seq_len + 1)  # +1 slack
    cos, sin = self.rotary_emb.get_cached(rotary_seq_len, Qn.device.index or 0)

    cos = cos[position_ids].unsqueeze(1).to(device = Qn.device, dtype = Qn.dtype)
    sin = sin[position_ids].unsqueeze(1).to(device = Qn.device, dtype = Qn.dtype)

    h = self.half_head_dim

    RH_Q = self.RH_Q
    RH_Q[:, :, :, :h] = Qn[:, :, :, h:]
    RH_Q[:, :, :, h:] = Qn[:, :, :, :h]
    RH_Q[:, :, :, :h].neg_()
    Qn *= cos
    Qn.addcmul_(RH_Q, sin)

    RH_K = RH_Q[
        :, :n_kv_heads, :, :
    ]
    RH_K[:, :, :, :h] = Kn[:, :, :, h:]
    RH_K[:, :, :, h:] = Kn[:, :, :, :h]
    RH_K[:, :, :, :h].neg_()
    Kn *= cos
    Kn.addcmul_(RH_K, sin)

    self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)
    self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)
    Kn = self.paged_attention_K[:kv_seq_len].permute(1, 2, 0, 3)
    Vn = self.paged_attention_V[:kv_seq_len].permute(1, 2, 0, 3)

    # Handle sliding windows, from transformers models/mistral/modeling_mistral.py#L193
    sliding_window = getattr(self.config, "sliding_window", None)
    if sliding_window is not None and kv_seq_len > sliding_window:
        start = kv_seq_len - sliding_window
        Knn = Kn[:, :, start:, :]
        Vnn = Vn[:, :, start:, :]
        if attention_mask is not None:
            attention_mask = attention_mask[..., start:]
    else:
        Knn, Vnn = Kn, Vn

    # Grouped query attention.
    _, _, cached_len, _ = Knn.shape
    if bsz == 1 or ((not SDPA_HAS_GQA) and n_groups != 1):
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Knn = Knn.reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, head_dim)

    # When qlen == vlen and attn_mask is None, causal attention is the right choice.
    Q_len = Qn.shape[-2]
    K_len = Knn.shape[-2]
    if attention_mask is None and Q_len == K_len:
        is_causal = True
    else:
        is_causal = False
    if bsz == 1:
        Qn *= (
            self.scalar
        )
        # (Q * scalar) @ K beats (Q @ K) * scalar for stopping overflows; see ggerganov/llama.cpp#7805.
        A = torch_matmul(Qn, Knn.transpose(2, 3), out = self.attention[:, :, :, :cached_len])
        A[:] = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32)
        A = torch_matmul(A, Vnn, out = Qn)
    # attention_mask fixup for SDPA when the user passes a 2D padding mask.
    else:
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].to(torch.bool)
        elif (
            attention_mask is not None
            and attention_mask.dim() == 4
            and attention_mask.dtype != torch.bool
        ):
            # Decode is more stable with boolean keep masks than additive bf16 masks.
            attention_mask = attention_mask.eq(0)

        if SDPA_HAS_GQA:
            A = scaled_dot_product_attention(
                Qn,
                Knn,
                Vnn,
                attn_mask = attention_mask,
                is_causal = is_causal,
                enable_gqa = True,
            )
        else:
            A = scaled_dot_product_attention(
                Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = is_causal
            )
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)


torch_nn_functional_silu = torch.nn.functional.silu


def fast_swiglu_inference(
    self,
    X,
    temp_gate = None,
    temp_up = None,
    gate_multiplier = None,
    down_multiplier = None,
):
    bsz, _, hd = X.shape

    gate = fast_linear_forward(self.gate_proj, X, out = temp_gate)

    if gate_multiplier is not None:
        gate *= gate_multiplier

    up = fast_linear_forward(self.up_proj, X, out = temp_up)

    gate = torch_nn_functional_silu(gate, inplace = True)
    gate *= up

    down = fast_linear_forward(self.down_proj, gate, out = up[:, :, :hd])

    if down_multiplier is not None:
        down *= down_multiplier

    return down


torch_square = torch.square
torch_mean = torch.mean


def fast_rms_layernorm_inference(
    self,
    X,
    XX = None,
    XX2 = None,
    variance = None,
):
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim = True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out = XX2), -1, keepdim = True, out = variance)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None:
        X = XX.to(old_dtype)
    else:
        X.copy_(XX)

    X *= self.weight
    return X


def fast_rms_layernorm_inference_gemma(
    self,
    X,
    out_weight = None,
):
    XX = X.to(torch.float32)
    variance = XX.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if out_weight is None:
        out_weight = self.weight + 1.0
    else:
        out_weight[:] = self.weight
        out_weight += 1.0

    XX *= out_weight
    return XX.to(X.dtype)


# Normal layernorm with mean removal.
@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def fast_layernorm_compiled(layernorm, X):
    old_dtype = X.dtype
    X = X.float()
    mean = X.mean(-1, keepdim = True)
    Xbar = X - mean
    X = (
        Xbar
        * torch.rsqrt(Xbar.square().mean(-1, keepdim = True) + layernorm.variance_epsilon)
        * layernorm.weight.float()
    )
    return X.to(old_dtype)


# Ported from transformers models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    seq_info = get_packed_info_from_kwargs(kwargs, Q.device)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings and kv_seq_len <= position_embeddings[0].shape[0]:
        cos, sin = position_embeddings
    else:
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)
        cos, sin = rotary_emb.get_cached(kv_seq_len, Q.device.index)
        cos = cos.to(device = Q.device, dtype = Q.dtype)
        sin = sin.to(device = Q.device, dtype = Q.dtype)

    rope_position_ids = position_ids
    if rope_position_ids is None and seq_info is not None:
        rope_position_ids = kwargs.get("position_ids")

    Q, K = fast_rope_embedding(Q, K, cos, sin, rope_position_ids)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    past_key_value = (K, V) if use_cache else None

    use_varlen = seq_info is not None and past_key_value is None
    backend = SDPA if attention_mask is not None else select_attention_backend(use_varlen)

    # Should dropout be hardcoded to 0.0?
    config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_groups,
        flash_dense_kwargs = {"causal": True},
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True},
    )
    # PrefixGrouper seg table rides in **kwargs from the GRPO logprob forward (same route as
    # packed_seq_lengths); misuse (KV cache / padding mask) raises. None means the byte-identical
    # default. Reuse of this forward carries the branch to qwen2 and gemma.
    _pg_seg = resolve_prefix_seg_info(kwargs, past_key_value, attention_mask)
    context = AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = kv_seq_len,
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = hidden_states.requires_grad,
        seq_info = seq_info,
        attention_mask = attention_mask,
        causal_mask = causal_mask,
        prefix_seg_info = _pg_seg,
    )

    A = run_attention(config = config, context = context, Q = Q, K = K, V = V)
    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


# Ported from transformers models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if use_cache and hasattr(self, "_flag_for_generation"):
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            padding_mask = padding_mask,
            position_embeddings = position_embeddings,
            **kwargs,
        )
        hidden_states += residual

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states = fast_swiglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            padding_mask = padding_mask,
            position_embeddings = position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


# See unslothai/unsloth#404 (comment 2323473452).
__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}


# Ported from transformers models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    assert output_attentions is False
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization unless we were given packed metadata
    allow_overlength = (
        getattr(self, "_unsloth_allow_packed_overlength", False)
        or ("packed_seq_lengths" in kwargs)
        or ("prefix_seg_info" in kwargs and kwargs["prefix_seg_info"] is not None)
    )
    if hasattr(self, "max_seq_length") and not allow_overlength:
        if seq_length > self.max_seq_length:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
            logger.warning_once(
                f"Unsloth: Input IDs of shape {shape} with length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:, : self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, : self.max_seq_length, :]
        if attention_mask is not None and attention_mask.shape[-1] > self.max_seq_length:
            attention_mask = attention_mask[:, : self.max_seq_length]

    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    # KV cache position_ids are handled here already.
    if False:
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype = torch.int32,
            device = f"{DEVICE_TYPE_TORCH}:0",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)
    else:
        position_ids = None

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = inputs_embeds.to(_get_dtype(dtype_from_config(self.config)))

    # Normalized from Gemma: cast to bfloat16/float16 to match it exactly, since 3072**0.5 is
    # 55.5000 in bfloat16 against 55.4256 in float32.
    IS_GEMMA = self.config.model_type.startswith("gemma")
    IS_GEMMA2 = self.config.model_type.startswith("gemma2")
    IS_COHERE = self.config.model_type.startswith("cohere")
    IS_GRANITE = self.config.model_type.startswith("granite")
    IS_FALCON_H1 = self.config.model_type.startswith("falcon_h1")

    train_embed_tokens = self.embed_tokens.weight.requires_grad

    if IS_GEMMA:
        normalizer = torch.tensor(math_sqrt(self.config.hidden_size), dtype = inputs_embeds.dtype)

        if train_embed_tokens:
            # Careful: this must not be an inplace op.
            inputs_embeds = inputs_embeds * normalizer
        else:
            inputs_requires_grad = inputs_embeds.requires_grad
            if not inputs_embeds.is_leaf:
                inputs_embeds = inputs_embeds.detach()
                inputs_requires_grad = True
            elif inputs_requires_grad:
                inputs_embeds.requires_grad_(False)
            inputs_embeds *= normalizer
            if inputs_requires_grad:
                inputs_embeds.requires_grad_(True)

    # Fix up the attention mask by setting elements to 0, specifically for DPO.
    if (
        getattr(self, "_has_no_labels", False) is True
        and (attention_mask is not None)
        and attention_mask.ndim == 2
        and (past_key_values is None)
        and (not train_embed_tokens)
        and self.training
    ):
        # Careful: for inference the attention_mask is size (1, kv_seq_len) while input_embeds is (1, 1, 4096).
        inputs_requires_grad = inputs_embeds.requires_grad
        if not inputs_embeds.is_leaf:
            inputs_embeds = inputs_embeds.detach()
            inputs_requires_grad = True
        elif inputs_requires_grad:
            inputs_embeds.requires_grad_(False)
        attention_mask = attention_mask[:, : self.max_seq_length]  # Must resize!
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad:
            inputs_embeds.requires_grad_(True)

    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
        # Must NOT convert to bool; that weirdly causes errors.

    hidden_states = inputs_embeds
    if IS_GRANITE or IS_FALCON_H1:  # granite has embedding multiplier
        hidden_states = self.config.embedding_multiplier * hidden_states

    if past_key_values is None and self.training:
        use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Gradient checkpointing methods (ie sqrt).
    if hasattr(self, "_gradient_checkpointing_boundaries"):
        boundaries = self._gradient_checkpointing_boundaries
    else:
        boundaries = None

    gradient_checkpointing = False

    if self.gradient_checkpointing and self.training and not use_cache:
        gradient_checkpointing = True

    # Gemma2 has alternating SWA and global attn.
    use_static_mask = True
    dynamic_SWA_mask = None
    dynamic_GA_mask = None
    if IS_GEMMA2:
        if HAS_FLASH_ATTENTION_SOFTCAPPING and attention_mask is None:
            self.SWA_mask = True
            self.GA_mask = False
        elif attention_mask is not None:
            # Unsloth needs a 2D mask, not [2, 1, n, n] (#853), converted to float not bool
            # (pytorch/pytorch#103749).

            dynamic_SWA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = self.config.sliding_window,
            )
            dynamic_GA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = None,
            )
            use_static_mask = False

        elif not hasattr(self, "SWA_mask"):
            if HAS_FLEX_ATTENTION:
                self.SWA_mask = create_flex_attention_sliding_window_mask(
                    self.max_seq_length, self.config.sliding_window
                )
                self.GA_mask = create_flex_attention_causal_mask(self.max_seq_length)
            else:
                n = self.max_seq_length
                # masked_fill makes things slower.
                self.SWA_mask = (
                    AttentionMaskConverter(
                        is_causal = True,
                        sliding_window = self.config.sliding_window,
                    )
                    .to_causal_4d(
                        1,
                        n,
                        n,
                        dtype = inputs_embeds.dtype,
                        device = DEVICE_TYPE_TORCH,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                self.GA_mask = (
                    AttentionMaskConverter(
                        is_causal = True,
                    )
                    .to_causal_4d(
                        1,
                        n,
                        n,
                        dtype = inputs_embeds.dtype,
                        device = DEVICE_TYPE_TORCH,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            pass

    if (
        IS_ATTENTION_REFACTOR
        and (hasattr(self, "rotary_emb") or not hasattr(self.layers[0].self_attn, "rotary_emb"))
    ) or IS_GRANITE:
        # position_embeddings is mandatory on main (huggingface/transformers#34858), and granite always
        # had the attention refactor, so it always takes this path.
        self.rotary_emb.extend_rope_embedding(hidden_states, self.config.max_position_embeddings)
        position_embeddings = self.rotary_emb.get_cached(
            self.config.max_position_embeddings, hidden_states.device.index
        )
    else:
        position_embeddings = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        mask = causal_mask
        if IS_GEMMA2:
            use_sliding_window = idx % 2 == 0
            if use_sliding_window:
                mask = self.SWA_mask if use_static_mask else dynamic_SWA_mask
            else:
                mask = self.GA_mask if use_static_mask else dynamic_GA_mask
            kwargs["use_sliding_window"] = use_sliding_window

        if gradient_checkpointing and not isinstance(decoder_layer, GradientCheckpointingLayer):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(
                        *inputs,
                        past_key_value,
                        output_attentions,
                        padding_mask = padding_mask,
                        position_embeddings = position_embeddings,
                        **kwargs,
                    )

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                mask,
                attention_mask,
                position_ids,
                use_reentrant = True,
                preserve_rng_state = False,
            )
            hidden_states = layer_outputs[0]

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask = mask,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_value = past_key_value,
                output_attentions = output_attentions,
                use_cache = use_cache,
                padding_mask = padding_mask,
                position_embeddings = position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if use_cache:
        if IS_FALCON_H1:
            hidden_states = fast_rms_layernorm_inference(self.final_layernorm, hidden_states)
        else:
            hidden_states = (
                fast_rms_layernorm_inference_gemma if IS_GEMMA else fast_rms_layernorm_inference
            )(self.norm, hidden_states)
    elif IS_COHERE:
        hidden_states = self.norm(hidden_states)
    elif IS_FALCON_H1:
        hidden_states = fast_rms_layernorm(self.final_layernorm, hidden_states, gemma = IS_GEMMA)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states, gemma = IS_GEMMA)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_cache,
        hidden_states = all_hidden_states,
        attentions = all_self_attns,
    )


# Ported from transformers models/llama/modeling_llama.py#L825
def _LlamaModel_fast_forward_inference(
    attention_fast_forward_inference = LlamaAttention_fast_forward_inference,
    mlp_fast_forward_inference = fast_swiglu_inference,
):
    # Makes attention and MLP customisable for models like qwen3/cohere.
    def LlamaModel_fast_forward_inference_custom(
        self,
        input_ids,
        past_key_values,
        position_ids,
        attention_mask = None,
        **kwargs,
    ):
        input_ids = input_ids[:, : self.max_seq_length]
        bsz, q_len = input_ids.shape
        hd = self.config.hidden_size
        mlp_size = self.config.intermediate_size

        X = self.model.embed_tokens(input_ids)
        X = X.to(_get_dtype(dtype_from_config(self.config)))
        bsz, q_len, hd = X.shape
        assert q_len == 1
        residual = torch.empty(
            (bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE_TORCH}:0"
        )
        _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE_TORCH}:0")
        XX, XX2 = _XX[0], _XX[1]
        variance = torch.empty(
            (bsz, q_len, 1), dtype = torch.float32, device = f"{DEVICE_TYPE_TORCH}:0"
        )
        temp_mlp = torch.empty(
            (2, bsz, 1, mlp_size), dtype = X.dtype, device = f"{DEVICE_TYPE_TORCH}:0"
        )
        temp_gates, temp_ups = (
            tuple(temp_mlp[0].to(torch.device(x)) for x in range(DEVICE_COUNT)),
            tuple(temp_mlp[1].to(torch.device(x)) for x in range(DEVICE_COUNT)),
        )

        seq_len = past_key_values[0][0].shape[-2]
        kv_seq_len = seq_len + 1
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                X,
                seq_len,
                sliding_window = getattr(self.config, "sliding_window", None),
            )
            # Pre-convert to bool once for all layers, avoiding a per-layer .eq(0).
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.eq(0)
        else:
            attention_mask = None

        # Compute rotary_seq_len once to avoid a per-layer GPU-CPU sync from .item().
        rotary_seq_len = max(kv_seq_len, int(position_ids.max().item()) + 1)

        next_decoder_cache = []

        for idx, decoder_layer in enumerate(self.model.layers):
            device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
            X, residual, position_ids = move_to_device(device_index, X, residual, position_ids)
            residual.copy_(X)
            X = fast_rms_layernorm_inference(
                decoder_layer.input_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X, present_key_value = attention_fast_forward_inference(
                decoder_layer.self_attn,
                hidden_states = X,
                past_key_value = past_key_values[idx],
                position_ids = position_ids,
                attention_mask = attention_mask,
                do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
                rotary_seq_len = rotary_seq_len,
            )
            X += residual

            residual.copy_(X)
            X = fast_rms_layernorm_inference(
                decoder_layer.post_attention_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X = mlp_fast_forward_inference(
                decoder_layer.mlp,
                X,
                temp_gate = temp_gates[device_index],
                temp_up = temp_ups[device_index],
            )
            X += residual

            next_decoder_cache.append(present_key_value)
        X = fast_rms_layernorm_inference(
            self.model.norm,
            X,
            XX = XX,
            XX2 = XX2,
            variance = variance,
        )

        return BaseModelOutputWithPast(
            last_hidden_state = X,
            past_key_values = next_decoder_cache,
            hidden_states = [],
            attentions = [],
        )

    return LlamaModel_fast_forward_inference_custom


# LlamaModel_fast_forward_inference exists for backwards compatibility and is consumed by other models.
LlamaModel_fast_forward_inference = _LlamaModel_fast_forward_inference()


def CausalLM_fast_forward(fast_forward_inference):
    def _CausalLM_fast_forward(
        self,
        input_ids: torch.LongTensor = None,
        causal_mask: Optional[BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = 0,
        logits_to_keep: Optional[int] = 0,
        *args,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None:
            outputs = fast_forward_inference(
                self,
                input_ids,
                past_key_values,
                position_ids = position_ids,
                attention_mask = attention_mask,
                **kwargs,
            )
        else:
            causal_mask = xformers.attn_bias.LowerTriangularMask() if HAS_XFORMERS else None

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
            self.model._has_no_labels = labels is None
            outputs = self.model(
                input_ids = input_ids,
                causal_mask = causal_mask,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
                **kwargs,
            )
        hidden_states = outputs[0]

        bsz, q_len, hd = hidden_states.shape
        lm_head = self.lm_head.weight
        lm_head_device = lm_head.device

        logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
        logit_scaling = getattr(self.config, "logit_scale", 0)
        dtype = lm_head.dtype
        # Skip int max() if either is a tensor (HF selective-decode form).
        if isinstance(num_logits_to_keep, torch.Tensor) or isinstance(logits_to_keep, torch.Tensor):
            num_logits_to_keep = 0
        else:
            num_logits_to_keep = max(num_logits_to_keep, logits_to_keep)

        hidden_states = hidden_states.to(lm_head_device)
        if labels is not None:
            labels = labels.to(lm_head_device)

        # Output last hidden states without logits if asked.
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            if num_logits_to_keep != 0:
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]
            return CausalLMOutputWithPast(
                loss = None,
                logits = hidden_states,
                past_key_values = outputs.past_key_values,
                hidden_states = outputs.hidden_states,
                attentions = outputs.attentions,
            )

        if bsz == 1 and q_len == 1:
            logits = torch.mv(lm_head, hidden_states.ravel().to(dtype))
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif num_logits_to_keep != 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :].to(dtype))
        else:
            RETURN_LOGITS = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            # Below 1024 normal Unsloth uses less VRAM.
            if bsz * q_len <= 1024 and not RETURN_LOGITS:
                # unsloth_fused_ce_loss calculates the best chunk size to reduce VRAM usage.
                RETURN_LOGITS = False

            if not RETURN_LOGITS and labels is not None:
                n_items = kwargs.get("num_items_in_batch", None)
                if n_items is None:
                    n_items = kwargs.get("n_items", None)

                if self.config.model_type == "falcon_h1":
                    hidden_states = hidden_states * self.config.lm_head_multiplier

                # Packed-boundary guard on raw labels (the fused kernel shifts internally). This branch RETURNS,
                # so mask_packed_sequence_boundaries() below is dead on packed paths: it needs
                # UNSLOTH_RETURN_LOGITS=1, which forces packing off. Out-of-place, no-op without
                # packed_seq_lengths.
                labels = mask_packed_boundary_labels(
                    labels,
                    kwargs.get("packed_seq_lengths"),
                )

                # fused_linear_cross_entropy is disabled: T4 raises OutOfResources (needs 98304 shared memory
                # against a 65536 limit).
                loss = unsloth_fused_ce_loss(
                    trainer = None,
                    hidden_states = hidden_states,
                    lm_head_weight = lm_head,
                    lm_head_bias = None,
                    labels = labels,
                    mask = None,
                    n_items = n_items,
                    scaling = getattr(self, "accelerator_scaler", None),
                    target_gb = None,
                    torch_compile = True,
                    logit_softcapping = logit_softcapping,
                )
                if not return_dict:
                    # Fused CE never materializes logits; use EMPTY_LOGITS like the return_dict branch below (#2068).
                    output = (EMPTY_LOGITS,) + outputs[1:]
                    return (loss,) + output if loss is not None else output

                output = CausalLMOutputWithPast(
                    loss = loss,
                    logits = EMPTY_LOGITS,
                    past_key_values = outputs.past_key_values,
                    hidden_states = outputs.hidden_states,
                    attentions = outputs.attentions,
                )
                return output
            pass
            logits = self.lm_head(hidden_states.to(dtype))

        logits = logits.to(_get_dtype(dtype_from_config(self.config)))
        loss = None
        # Which field carries the scale is per family (cohere logit_scale, granite logits_scaling,
        # falcon_h1 lm_head_multiplier). The planner sizes the head's card from the same answer, so a
        # new family is taught once, not twice.
        if detect_logit_transforms is not None:
            _transforms = detect_logit_transforms(self.config)
            logit_softcapping = _transforms["logit_softcapping"]
            logit_scaling = _transforms["logit_scale_multiply"]
            if not logit_scaling and _transforms["logit_scale_divide"]:
                logit_scaling = 1 / _transforms["logit_scale_divide"]
        else:
            logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)
            logit_scaling = getattr(self.config, "logit_scale", 0)
            if self.config.model_type == "granite":
                logit_scaling = 1 / getattr(self.config, "logits_scaling", 1)
            elif self.config.model_type == "falcon_h1":
                logit_scaling = self.config.lm_head_multiplier

        if labels is not None:
            shift_logits = logits
            shift_labels = torch.empty_like(labels)
            shift_labels[..., :-1] = labels[..., 1:]
            shift_labels[..., -1] = -100
            mask_packed_sequence_boundaries(
                shift_labels,
                kwargs.get("packed_seq_lengths"),
            )
            n_items = kwargs.get("num_items_in_batch", None)
            if n_items is None:
                n_items = kwargs.get("n_items", None)
            loss = fast_cross_entropy_loss(
                logits = shift_logits,
                labels = shift_labels,
                logit_softcapping = logit_softcapping,
                logit_scaling = logit_scaling,
                n_items = n_items,
            )
        else:
            if logit_scaling != 0:
                if logits.requires_grad:
                    logits = logit_scaling * logits
                else:
                    logits *= logit_scaling
            if logit_softcapping != 0:
                if logits.requires_grad:
                    logits = (1.0 / logit_softcapping) * logits
                    logits = torch.tanh(logits)
                    logits = logit_softcapping * logits
                else:
                    logits *= 1.0 / logit_softcapping
                    logits.tanh_()
                    logits *= logit_softcapping

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )

    return _CausalLM_fast_forward


@torch._disable_dynamo
def PeftModel_fast_forward(
    self,
    input_ids = None,
    causal_mask = None,
    attention_mask = None,
    inputs_embeds = None,
    labels = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    task_ids = None,
    num_logits_to_keep = 0,
    logits_to_keep = 0,
    **kwargs,
):
    is_classification = "Classification" in str(type(self.base_model.model))
    if is_classification:
        return self.base_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            inputs_embeds = inputs_embeds,
            labels = labels,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            **kwargs,
        )
    else:
        return self.base_model(
            input_ids = input_ids,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            inputs_embeds = inputs_embeds,
            labels = labels,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            num_logits_to_keep = num_logits_to_keep,
            logits_to_keep = logits_to_keep,
            **kwargs,
        )


def _get_rope_theta(config, default = 10000.0):
    """Get rope_theta from config, handling both transformers 4.x and 5.x."""
    try:
        return config.rope_theta
    except (AttributeError, KeyError):
        pass
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        return rp.get("rope_theta", default)
    return default


def _rope_scaling_as_dict(rope_scaling):
    """Normalize config.rope_scaling (dict or config object) to a dict; {} on failure."""
    if isinstance(rope_scaling, dict):
        return rope_scaling
    for converter in ("to_dict", "dict"):
        fn = getattr(rope_scaling, converter, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        return {k: v for k, v in vars(rope_scaling).items() if not k.startswith("_")}
    except TypeError:
        return {}


def _extended_rope_scaling(config, factor):
    """RoPE scaling to extend a model past its native window. Keeps native llama3 as-is
    (linear extension is far worse for long context); everything else gets linear. Returns
    (scaling_or_None, type): None keeps llama3. The linear dict carries rope_theta so
    transformers v5 (which stores it under rope_parameters) keeps the real base, not 10000.
    Only llama3 is preserved because patch_llama_rope_scaling can only rebuild linear/llama3/
    longrope and its longrope branch needs a top-level original_max_position_embeddings."""
    existing = _rope_scaling_as_dict(
        getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
    )
    existing_type = existing.get("rope_type") or existing.get("type")
    if existing_type == "llama3":
        return None, existing_type
    return {
        "type": "linear",
        "factor": factor,
        "rope_theta": _get_rope_theta(config),
    }, existing_type


def _llama3_inv_freq_from_config(
    config,
    rope_scaling,
    device = "cpu",
):
    """llama3 inv_freq with factors from config; fallback when modeling_rope_utils is missing."""
    base = _get_rope_theta(config, default = 10000.0)
    dim = getattr(config, "head_dim", None)
    if dim is None:
        dim = int(config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype = torch.int64, device = device).float() / dim)
    )

    scale_factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    assert low_freq_wavelen != high_freq_wavelen

    # Vectorized meta-llama bands: high freqs kept, low divided by factor, medium blended.
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / scale_factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq / scale_factor + smooth * inv_freq
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def _vanilla_inv_freq_from_config(config, device = "cpu"):
    """Unscaled RoPE inv_freq (rope_type 'default'/None), matching the constructor's fallback."""
    base = _get_rope_theta(config, default = 10000.0)
    dim = getattr(config, "head_dim", None)
    if dim is None:
        dim = int(config.hidden_size // config.num_attention_heads)
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.int64, device = device).float() / dim))


def _compute_config_rope_inv_freq(config, rope_scaling):
    """(inv_freq, attention_scaling) per config.rope_scaling via transformers'
    ROPE_INIT_FUNCTIONS, with an inline llama3 fallback; (None, 1.0) on failure."""
    original_rope_scaling = rope_scaling
    rope_scaling = _rope_scaling_as_dict(rope_scaling)
    rope_type = rope_scaling.get("rope_type", None) or rope_scaling.get("type", None)
    # "default"/unset means unscaled RoPE: transformers >= 5 reports rope_type="default" for every
    # plain config and dropped "default" from ROPE_INIT_FUNCTIONS, so compute it directly instead
    # of warning per load.
    if rope_type in (None, "default"):
        return _vanilla_inv_freq_from_config(config).to(dtype = torch.float32, device = "cpu"), 1.0
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        try:
            inv_freq, attention_scaling = rope_init_fn(config, torch.device("cpu"))
        except Exception:
            # Object-style rope_scaling: retry with a config copy carrying the plain dict.
            if isinstance(original_rope_scaling, dict):
                raise
            import copy as _copy

            config_copy = _copy.copy(config)
            config_copy.rope_scaling = rope_scaling
            inv_freq, attention_scaling = rope_init_fn(config_copy, torch.device("cpu"))
        return inv_freq.to(dtype = torch.float32, device = "cpu"), float(attention_scaling)
    except Exception as exception:
        if rope_type == "llama3":
            try:
                return _llama3_inv_freq_from_config(config, rope_scaling), 1.0
            except Exception:
                pass
        logger.warning_once(
            f"Unsloth: Could not apply RoPE scaling '{rope_type}' from config "
            f"({type(exception).__name__}: {exception}); falling back to unscaled RoPE. "
            "Long-context generation may degrade."
        )
        return None, 1.0


# Static KV Cache landed in 4.38.0 and made training much slower (#168,
# huggingface/transformers#27931), so the old rotary embeddings are retained.
class LlamaRotaryEmbedding(torch.nn.Module):
    # RoPE buffer precision is wrong, so cast to int64; see huggingface/transformers#28837 and microsoft/DeepSpeed#4932.
    def __init__(
        self,
        dim = None,
        max_position_embeddings = 2048,
        base = 10000,
        device = None,
        config = None,
    ):
        super().__init__()
        # cos/sin multiplier (1.0 except yarn / longrope); set before any cache build.
        self.attention_scaling = 1.0
        # Base-class-from-config path (modern transformers): derive inv_freq like transformers so
        # config.rope_scaling is not dropped (#2405). Scaled subclasses excluded to avoid
        # double-scaling.
        if config is not None:
            base = _get_rope_theta(config, default = base)
            partial_rotary_factor = (
                config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            )
            dim = getattr(config, "head_dim", None)
            if dim is None:
                dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE_TORCH
            max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Kept so the v5 rope repair can rebuild the scaled inv_freq (#2405).
        self._unsloth_rope_config = config
        # Dynamic RoPE starts at a max of 4 * 8192 tokens and grows iteratively in increments of 8192.
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_sin_cached = [None] * DEVICE_COUNT

        inv_freq = self._unsloth_recompute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                seq_len = self.current_rope_size,
                device = torch.device(device_idx),
                dtype = torch.get_default_dtype(),
            )

        self.cos_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )
        self.sin_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )

    def _apply_inv_freq_scaling(self, inv_freq):
        """Override to apply custom inv_freq scaling (e.g., extended RoPE)."""
        return inv_freq

    def _unsloth_recompute_inv_freq(self):
        # Config scaling (llama3/yarn) first, else vanilla plus subclass scaling. Shared by __init__ and
        # the v5 rope repair so they cannot diverge.
        config = getattr(self, "_unsloth_rope_config", None)
        config_inv_freq = None
        rope_scaling = getattr(config, "rope_scaling", None) if config is not None else None
        if rope_scaling is not None and type(self) is LlamaRotaryEmbedding:
            config_inv_freq, self.attention_scaling = _compute_config_rope_inv_freq(
                config,
                rope_scaling,
            )
        if config_inv_freq is not None:
            return config_inv_freq
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype = torch.int64, device = "cpu").float() / self.dim)
        )
        return self._apply_inv_freq_scaling(inv_freq)

    def _apply_time_scaling(self, t):
        """Override to apply custom time scaling (e.g., linear scaling)."""
        return t

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # The original Llama codebase creates these on the target device in FP32 and multiplies in FP32.
        self.current_rope_size = seq_len
        t = torch.arange(
            self.current_rope_size, device = self.inv_freq.device, dtype = torch.int64
        ).float()
        t = self._apply_time_scaling(t)

        freqs = torch.outer(t, self.inv_freq)
        # Different from the paper, but it uses a different permutation to obtain the same calculation.
        emb = torch.cat((freqs, freqs), dim = -1)
        # Applied here so attention_scaling survives extend_rope_embedding rebuilds; the 1.0 default
        # keeps unscaled paths bit-identical.
        cos = (emb.cos() * self.attention_scaling).to(dtype = dtype, device = device, non_blocking = True)
        sin = (emb.sin() * self.attention_scaling).to(dtype = dtype, device = device, non_blocking = True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin

    def forward(
        self,
        x,
        position_ids = None,
        seq_len = None,
    ):
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len = seq_len, device = x.device, dtype = x.dtype)

        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )

    def get_cached(
        self,
        seq_len = None,
        device_index = None,
    ):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[device_index]

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size:
            return
        # Iteratively grow by increments of 8192.
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype
            )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # RoPE buffer precision is wrong, so cast to int64; see huggingface/transformers#28837 and microsoft/DeepSpeed#4932.
    def __init__(
        self,
        dim = None,
        max_position_embeddings = 2048,
        base = 10000,
        device = None,
        scaling_factor = 1.0,
        config = None,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(
            dim = dim,
            max_position_embeddings = max_position_embeddings,
            base = base,
            device = device,
            config = config,
        )

    def _apply_time_scaling(self, t):
        """Apply linear scaling to time indices."""
        return t / self.scaling_factor


# For Llama 3.1; see vllm model_executor/layers/rotary_embedding.py#L736
class LlamaExtendedRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim = None,
        max_position_embeddings = 2048,
        base = 10000,
        device = None,
        config = None,
    ):
        super().__init__(
            dim = dim,
            max_position_embeddings = max_position_embeddings,
            base = base,
            device = device,
            config = config,
        )

    # From meta-llama/llama-models llama3_1/api/model.py#L41
    def _apply_inv_freq_scaling(self, freqs: torch.Tensor):
        # llama3 factors come from the config, with Llama-3.1 defaults when built without one: hardcoding
        # 8 is wrong for Llama-3.2 (32). v5 renames rope_scaling to rope_parameters, so read either.
        config = getattr(self, "_unsloth_rope_config", None)
        rope_scaling = _rope_scaling_as_dict(
            getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
        )
        scale_factor = rope_scaling.get("factor", 8)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype = freqs.dtype, device = freqs.device)


class LongRopeRotaryEmbedding(torch.nn.Module):
    # For Phi 3.5 128K; see microsoft/Phi-3.5-mini-instruct modeling_phi3.py
    def __init__(
        self,
        dim = None,
        max_position_embeddings = 131072,
        original_max_position_embeddings = 4096,
        base = 10000,
        short_factor = None,
        long_factor = None,
        device = None,
        config = None,
    ):
        super().__init__()
        assert short_factor is not None
        assert long_factor is not None
        assert type(original_max_position_embeddings) is int

        if config is not None:
            base = _get_rope_theta(config, default = base)
            partial_rotary_factor = (
                config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            )
            dim = int((config.hidden_size // config.num_attention_heads))
            device = DEVICE_TYPE_TORCH
            max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        # Dynamic RoPE starts at a max of 4 * 8192 tokens and grows iteratively.
        self.current_rope_size = min(original_max_position_embeddings, self.max_position_embeddings)
        self.multi_gpu_short_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_short_sin_cached = [None] * DEVICE_COUNT
        self.multi_gpu_long_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_long_sin_cached = [None] * DEVICE_COUNT

        # Long RoPE is like RoPE except short sequences have one cos/sin and long sequences another.
        inv_freq_shape = (
            torch.arange(0, self.dim, 2, dtype = torch.int64, device = "cpu").float() / self.dim
        )
        short_factor = torch.tensor(short_factor, device = "cpu", dtype = torch.float32)
        long_factor = torch.tensor(long_factor, device = "cpu", dtype = torch.float32)
        short_inv_freq = 1.0 / (short_factor * self.base**inv_freq_shape)
        long_inv_freq = 1.0 / (long_factor * self.base**inv_freq_shape)

        # Phi-3 scale factor.
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
        self.scaling_factor = scaling_factor

        self.register_buffer("short_inv_freq", short_inv_freq, persistent = False)
        self.register_buffer("long_inv_freq", long_inv_freq, persistent = False)

        # Build here to make torch.jit.trace work, initializing the short-sequence cache for all devices.
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        t = torch.arange(
            original_max_position_embeddings,
            device = self.short_inv_freq.device,
            dtype = torch.int64,
        ).float()
        freqs = torch.outer(t, self.short_inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)

        for device_idx in range(DEVICE_COUNT):
            device_obj = torch.device(device_idx)
            cos_cached = (emb.cos() * self.scaling_factor).to(
                dtype = dtype, device = device_obj, non_blocking = True
            )
            sin_cached = (emb.sin() * self.scaling_factor).to(
                dtype = dtype, device = device_obj, non_blocking = True
            )
            self.multi_gpu_short_cos_cached[device_idx] = cos_cached
            self.multi_gpu_short_sin_cached[device_idx] = sin_cached

        self.short_cos_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )
        self.short_sin_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )
        self.long_cos_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )
        self.long_sin_cached = torch.empty(
            1, device = get_current_device(), dtype = torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # The original Llama codebase creates these on the target device in FP32 and multiplies in FP32.
        self.current_rope_size = seq_len

        t = torch.arange(
            self.current_rope_size, device = self.long_inv_freq.device, dtype = torch.int64
        ).float()
        freqs = torch.outer(t, self.long_inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        cos_cached = (emb.cos() * self.scaling_factor).to(
            dtype = dtype, device = device, non_blocking = True
        )
        sin_cached = (emb.sin() * self.scaling_factor).to(
            dtype = dtype, device = device, non_blocking = True
        )
        self.multi_gpu_long_cos_cached[device.index] = cos_cached
        self.multi_gpu_long_sin_cached[device.index] = sin_cached
        return cos_cached, sin_cached

    def forward(
        self,
        x,
        position_ids = None,
        seq_len = None,
    ):
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len = seq_len, device = x.device, dtype = x.dtype)

        device_index = x.device.index

        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return (
                self.multi_gpu_short_cos_cached[device_index][:seq_len],
                self.multi_gpu_short_sin_cached[device_index][:seq_len],
            )
        else:
            return (
                self.multi_gpu_long_cos_cached[device_index][:seq_len],
                self.multi_gpu_long_sin_cached[device_index][:seq_len],
            )

    def get_cached(
        self,
        seq_len = None,
        device_index = None,
    ):
        if device_index is None:
            device_index = get_current_device()
        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return self.multi_gpu_short_cos_cached[device_index], self.multi_gpu_short_sin_cached[
                device_index
            ]
        return self.multi_gpu_long_cos_cached[device_index], self.multi_gpu_long_sin_cached[
            device_index
        ]

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size:
            return
        # Iteratively grow by increments of 8192.
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                self.current_rope_size, device = torch.device(device_idx), dtype = x.dtype
            )


def unsloth_fast_generate(self, *args, **kwargs):
    restore_training_mode = self.training
    # Snapshot the real GC mode (e.g. "unsloth") before for_inference clears it, so the restore
    # preserves it rather than collapsing to a plain bool.
    use_gradient_checkpointing = next(
        (v for v in (getattr(m, "gradient_checkpointing", False) for m in self.modules()) if v),
        False,
    )

    FastLlamaModel.for_inference(self)

    # Unpack a BatchEncoding passed as input_ids (old notebooks do generate(input_ids=tokenizer(...))):
    # v5 generate() calls .shape on it and crashes.
    _maybe_encoding = kwargs.get("input_ids", None)
    if (
        _maybe_encoding is not None
        and not isinstance(_maybe_encoding, torch.Tensor)
        and hasattr(_maybe_encoding, "items")
    ):
        batch_data = kwargs.pop("input_ids")
        for key, val in batch_data.items():
            kwargs.setdefault(key, val)

    dtype = _get_dtype(dtype_from_config(self.config))

    if hasattr(self, "config") and hasattr(self.config, "max_position_embeddings"):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None and "max_new_tokens" in kwargs:
            _ids = kwargs["input_ids"]
            if hasattr(_ids, "shape") and (
                _ids.shape[-1] + kwargs["max_new_tokens"] > self.config.max_position_embeddings
            ):
                raise ValueError(
                    f"Unsloth: input length {_ids.shape[-1]} + max_new_tokens {kwargs['max_new_tokens']} exceeds the maximum sequence length of {self.config.max_position_embeddings}!\n"
                    "You will need to do long context extension by increasing the `max_seq_length` in `FastLanguageModel.from_pretrained`."
                )

    # Must patch accelerate for Xformers.

    kwargs["cache_implementation"] = "dynamic"
    # transformers 4.50 renamed num_logits_to_keep to logits_to_keep; pop both and re-emit under the
    # spelling forward() accepts.
    _provided_num = kwargs.pop("num_logits_to_keep", None)
    _provided_logits = kwargs.pop("logits_to_keep", None)
    _provided = _provided_logits if _provided_logits is not None else _provided_num
    try:
        _fwd_params = inspect.signature(self.forward).parameters
        _has_new = "logits_to_keep" in _fwd_params
        _has_old = "num_logits_to_keep" in _fwd_params
    except (TypeError, ValueError):
        # Opaque forward: keep the caller's spelling, defaulting to the new one.
        _has_old = _provided_num is not None and _provided_logits is None
        _has_new = not _has_old
    if _has_new:
        kwargs["logits_to_keep"] = _provided if _provided is not None else 1
    elif _has_old:
        kwargs["num_logits_to_keep"] = _provided if _provided is not None else 1

    kwargs.pop("token_type_ids", None)

    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Mixed precision autocast.
    with (
        _get_inference_mode_context_manager(self),
        torch.autocast(device_type = DEVICE_TYPE_TORCH, dtype = dtype),
    ):
        output = self._old_generate(*args, **kwargs)


    if restore_training_mode:
        FastLlamaModel.for_training(
            self,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )

    return output


def _vllm_will_load_weights(fast_inference, num_labels = None):
    """Whether vLLM, which takes no revision, ends up owning the weight load.

    The loader has to answer this before it probes the config, since that probe's ref
    decides which architecture class the load is dispatched to. Mirrors the checks at the
    top of from_pretrained below, which calls this too so the two cannot drift.
    """
    if not fast_inference or num_labels is not None:
        return False
    # from_pretrained clears fast_inference when vLLM is missing but re-enables it on hip.
    if DEVICE_TYPE == "hip":
        return True
    if not is_vLLM_available():
        return False
    if DEVICE_TYPE == "cuda" and torch.cuda.get_device_capability()[0] < 7:
        return False
    return True


class FastLlamaModel:
    @staticmethod
    def _prepare_for_qat(model, qat_scheme):
        model = _prepare_model_for_qat(model, qat_scheme)
        return model

    @staticmethod
    def pre_patch():
        init_name, function = patch_llama_rope_scaling(
            model_name = "llama",
            rope_module = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            extended_rope_module = LlamaExtendedRotaryEmbedding,
            attention_module = LlamaAttention,
            longrope_module = LongRopeRotaryEmbedding,
        )
        if init_name is not None:
            exec(function, globals())
            LlamaAttention.__init__ = eval(init_name)
        LlamaAttention.forward = LlamaAttention_fast_forward
        LlamaSdpaAttention.forward = LlamaAttention_fast_forward
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        LlamaModel.forward = LlamaModel_fast_forward
        LlamaForCausalLM.forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(LlamaForCausalLM)

        # Static KV Cache landed in 4.38.0 and made training much slower (#168,
        # huggingface/transformers#27931), so the old rotary embeddings are retained.
        import transformers.models.llama.modeling_llama

        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = (
            LlamaLinearScalingRotaryEmbedding
        )
        return

    @staticmethod
    def from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = DEFAULT_DEVICE_MAP,
        # Planner hints for device_map = "unsloth"; see resolve_unsloth_device_map.
        device_map_planner_kwargs = None,
        rope_scaling = None,
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        revision = None,
        tokenizer_revision = None,
        fast_inference = False,  # uses vLLM
        gpu_memory_utilization = 0.5,
        float8_kv_cache = False,
        random_state = 3407,
        max_lora_rank = 16,
        disable_log_stats = False,
        unsloth_vllm_standby = False,
        num_labels = None,
        qat_scheme = None,
        load_in_fp8 = False,  # fp8 LoRA (True, False, 'block')
        **kwargs,
    ):
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "0"
        if trust_remote_code:
            if fast_inference:
                raise NotImplementedError(
                    "Unsloth: Fast inference does not support `trust_remote_code` yet."
                )
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"
                "Are you certain you want to do remote code execution?"
            )
        if fast_inference:
            if not is_vLLM_available():
                print("Unsloth: vLLM is not installed! Will use Unsloth inference!")
                fast_inference = False
            if DEVICE_TYPE == "cuda":
                major_version, minor_version = torch.cuda.get_device_capability()
                if major_version < 7:
                    print(
                        "Unsloth: vLLM does not work on older GPUs - will switch to Unsloth inference!"
                    )
                    fast_inference = False
            elif DEVICE_TYPE == "hip":
                fast_inference = True
            if unsloth_vllm_standby and os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0":
                raise RuntimeError(
                    "Unsloth: `unsloth_vllm_standby` is True, but  environment variable `UNSLOTH_VLLM_STANDBY` is not set to 1!"
                )
            # Only vLLM cannot take a revision, and fast_inference may have just been cleared above while a
            # num_labels load stays in-process, so both can honour the pin. vLLM fetches the default branch,
            # so pinning only the config and tokenizer would mix two refs in one model.
            if _vllm_will_load_weights(fast_inference, num_labels) and revision is not None:
                logger.warning_once(
                    f"Unsloth: Ignoring revision = `{revision}` since vLLM loads weights from "
                    "the default branch. Use `fast_inference = False` to load a pinned revision."
                )
                revision = None
                tokenizer_revision = None

        if tokenizer_revision is None and tokenizer_name in (None, model_name):
            # A direct call, or a wrapper forwarding only `revision`, leaves this unset.
            tokenizer_revision = revision

        token = hf_login(token)
        if model_patcher is None:
            model_patcher = FastLlamaModel
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()

        gpu_stats_name, gpu_stats_snippet, max_memory = get_device_stats()

        try:
            vllm_version = f" vLLM: {importlib_version('vllm')}."
        except:
            vllm_version = ""

        statistics = (
            f"==((====))==  Unsloth {__version__}: Fast {model_patcher.__name__[4:-5]} patching. Transformers: {transformers_version}.{vllm_version}\n"
            f"   {chr(92)}{chr(92)}   /|    {gpu_stats_name}Num GPUs = {DEVICE_COUNT}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"
            f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"
            f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"
            f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        )

        print(statistics)

        if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
            old_hf_transfer = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]
            if old_hf_transfer in ("False", "false"):
                old_hf_transfer = "0"
            if old_hf_transfer in ("True", "true"):
                old_hf_transfer = "1"
        else:
            old_hf_transfer = "0"
        if old_hf_transfer == "1":
            print(
                "Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!"
            )
        if old_hf_transfer != "0":
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        model_patcher.pre_patch()
        # A download counter, to see whether environments are breaking or HF is down.
        get_statistics(kwargs.get("local_files_only", False))

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32

        # Respect a user-provided config so it is the single config object used everywhere below; else
        # HF gets it again through **kwargs alongside our config= and fails with a duplicate kwarg.
        user_config = kwargs.pop("config", None)
        if user_config is not None:
            model_config = user_config
            # model_name may have been remapped to a prequantized repo whose checkpoint needs its
            # quantization_config; graft it on, or the 4bit weights load without their quant state.
            if getattr(model_config, "quantization_config", None) is None:
                _checkpoint_config = AutoConfig.from_pretrained(
                    model_name,
                    token = token,
                    attn_implementation = "sdpa",
                    revision = revision,
                )
                _checkpoint_quant = getattr(_checkpoint_config, "quantization_config", None)
                if _checkpoint_quant is not None:
                    model_config.quantization_config = _checkpoint_quant
        else:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                attn_implementation = "sdpa",
                revision = revision,
            )
        model_config.model_name = model_name
        model_max_seq_length = model_config.max_position_embeddings

        verify_fp8_support_if_applicable(model_config)

        # Check if RoPE Scaling is even allowed
        model_function = MODEL_FOR_CAUSAL_LM_MAPPING[model_config.__class__]
        IS_FALCON_H1 = model_config.model_type.startswith("falcon_h1")

        preferred_attn_impl = resolve_attention_implementation(
            model_function, model_config, dtype = dtype
        )

        # Prefetch the repo (killable child) so the weight load is a cache hit. Runs after the
        # AutoConfig/model-class check so an unsupported repo fails on its small config fetch, and warms
        # the revision the load uses or the repo downloads twice.
        _prefetched = maybe_prefetch_hf_snapshot(
            model_name,
            token = token,
            revision = revision,
            cache_dir = kwargs.get("cache_dir"),
            local_files_only = kwargs.get("local_files_only", False),
            # Skip the warm only for a real vLLM load; a num_labels classification load still goes
            # in-process below, so it must be warmed even under fast_inference.
            fast_inference = fast_inference and num_labels is None,
            subfolder = kwargs.get("subfolder"),
            force_download = kwargs.get("force_download", False),
            use_safetensors = kwargs.get("use_safetensors"),
            from_tf = kwargs.get("from_tf", False),
            from_flax = kwargs.get("from_flax", False),
            # A bare load reads only ROOT weights, so skip subdir weights; ignored when a subfolder is set.
            weights_at_root = True,
            variant = kwargs.get("variant"),  # forward so the warm keeps the variant .bin
            gguf_file = kwargs.get(
                "gguf_file"
            ),  # forward so the warm fetches the GGUF (else ignored)
        )
        # The child did the forced download; clear the flag so the load reuses the warm cache.
        if _prefetched and kwargs.get("force_download", False):
            kwargs["force_download"] = False

        # The tokenizer always loads in-process, and without an explicit cache_dir Colab/Kaggle route it
        # to a special tokenizer cache rather than the one the base snapshot warmed, so resolve the
        # cache_dir the tokenizer load will actually use.
        from ..tokenizer_utils import (
            IS_COLAB_ENVIRONMENT,
            IS_KAGGLE_ENVIRONMENT,
            KAGGLE_TMP,
        )

        _tokenizer_repo = (
            tokenizer_name if (isinstance(tokenizer_name, str) and tokenizer_name) else model_name
        )
        _tokenizer_cache_dir = kwargs.get("cache_dir")
        if _tokenizer_cache_dir is None:
            if IS_COLAB_ENVIRONMENT:
                _tokenizer_cache_dir = "huggingface_tokenizers_cache"
            elif IS_KAGGLE_ENVIRONMENT:
                _tokenizer_cache_dir = os.path.join(KAGGLE_TMP, "huggingface_tokenizers_cache")
        # Warm the tokenizer repo into the cache the load will use whenever the base warm did not cover
        # it: a distinct tokenizer repo, fast_inference (base warm skipped), or a differing cache_dir.
        _warm_tokenizer_repo = (
            isinstance(_tokenizer_repo, str)
            and bool(_tokenizer_repo)
            and (
                _tokenizer_repo != model_name
                or fast_inference
                or _tokenizer_cache_dir != kwargs.get("cache_dir")
            )
        )
        if _warm_tokenizer_repo:
            maybe_prefetch_hf_snapshot(
                _tokenizer_repo,
                token = token,
                cache_dir = _tokenizer_cache_dir,
                local_files_only = kwargs.get("local_files_only", False),
                tokenizer_only = True,
                revision = tokenizer_revision,
            )

        has_rope_scaling = False
        try:
            with open(inspect.getfile(model_function), "r", encoding = "utf-8") as file:
                has_rope_scaling = "self.config.rope_scaling" in file.read()
        except:
            pass
        has_rope_scaling = True

        # If max_seq_length is not specified, use the maximum from the config.
        if max_seq_length is None:
            max_seq_length = model_max_seq_length

        if (rope_scaling is None) and (max_seq_length > model_max_seq_length):
            factor = max_seq_length / model_max_seq_length

            if fast_inference:
                raise NotImplementedError(
                    "Unsloth: Fast inference does not yet work with RoPE Scaling."
                )

            linear_scaling, native_type = _extended_rope_scaling(model_config, factor)
            if linear_scaling is not None:
                # Native llama3 scaling already handles long context; just widen the window.
                logger.warning_once(
                    f"Unsloth: {model_name} can only handle sequence lengths of at most "
                    f"{model_max_seq_length}.\nBut with kaiokendev's RoPE scaling of "
                    f"{round(factor, 3)}, it can be magically be extended to "
                    f"{max_seq_length}!"
                )
                if not has_rope_scaling:
                    raise RuntimeError(
                        f"However, {model_name} doesn't support RoPE Scaling!\n"
                        "Please file a feature request at https://github.com/unslothai/unsloth."
                    )
                kwargs["rope_scaling"] = linear_scaling
            else:
                # Native llama3 scaling already handles long context; just widen the window.
                logger.warning_once(
                    f"Unsloth: extending {model_name} to {max_seq_length} using its native "
                    f"{native_type} RoPE scaling."
                )

        from .loader_utils import (
            check_and_disable_bitsandbytes_loading,
            sync_unsloth_model_name_bnb_flags,
        )
        from unsloth_zoo.utils import get_quant_type

        load_in_8bit = kwargs.get("load_in_8bit", False)

        # Disable bitsandbytes loading if the model has non-bitsandbytes quantization.
        load_in_4bit, load_in_8bit, _ckpt_quant_method = check_and_disable_bitsandbytes_loading(
            model_config, load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit
        )
        # Correct UNSLOTH_MODEL_NAME's bnb tokens now the effective bnb state is known (the per-load env
        # was built before remap/disable). gpt-oss only.
        sync_unsloth_model_name_bnb_flags(load_in_4bit, load_in_8bit)

        # num_labels sends the load to AutoModelForSequenceClassification, whose `score` replaces the
        # lm_head the planner named off the repo's config, so dispatch_model refuses the map. The weights
        # load against user_config while the planner rebuilds the repo's from model_name.
        _planner_skip_reason = None
        if user_config is not None:
            _planner_skip_reason = (
                "a caller-supplied config may not describe the repo the planner rebuilds"
            )
        if _planner_skip_reason is None and num_labels is not None:
            _planner_skip_reason = (
                planner_class_mismatch_reason(
                    resolve_model_class(AutoModelForSequenceClassification, model_config),
                    planner_model_class(model_config, trust_remote_code = trust_remote_code),
                )
                or "num_labels loads a task head the repo config does not describe"
            )

        # A prequantized checkpoint is always sized by its own config.json: merge_quantization_configs
        # overlays loading attributes for GPTQ / AWQ / AutoRound / FbgemmFp8 / CompressedTensors / Mxfp4
        # and never for bitsandbytes, so neither the merge below (#5027) nor any argument here reaches the
        # plan. Worth refusing over only for mamba, whose out_proj stacks are GiBs of error; the other
        # reachable gap is MoE routers, ~70 MiB even on Qwen3-235B-A22B, inside the planner's 256 MiB
        # margin.
        if _planner_skip_reason is None and IS_FALCON_H1 and _ckpt_quant_method == "bitsandbytes":
            _bundled = getattr(model_config, "quantization_config", None)
            _bundled_skip = (
                _bundled.get("llm_int8_skip_modules")
                if isinstance(_bundled, dict)
                else getattr(_bundled, "llm_int8_skip_modules", None)
            ) or []
            if not all(_m in _bundled_skip for _m in ("mamba", "out_proj")):
                _planner_skip_reason = (
                    "this prequantized checkpoint does not bundle the mamba exclusions the "
                    "load adds, so the plan would size dense weights at 4bit"
                )

        # Here, not in loader.py: the mapper there can still substitute the repo (a -bnb-4bit name
        # resolving to its 16-bit twin), so a plan sized for the caller's name is the wrong plan.
        device_map = resolve_unsloth_device_map(
            requested_device_map(device_map),
            model_name,
            fast_inference = fast_inference,
            planner_kwargs = planner_kwargs_with_max_memory(device_map_planner_kwargs, kwargs),
            skip_reason = _planner_skip_reason,
            **planner_config_overrides(kwargs),
            token = token,
            trust_remote_code = trust_remote_code,
            **planner_hub_kwargs(kwargs),
            revision = revision,
            # The dtype the load gets, not the checkpoint's: from_pretrained overrides config.json, and
            # planning the wrong one mis-sizes weights 2x either way.
            **add_dtype_kwargs(dtype),
            # The caller's own config, still untouched in kwargs here, overrides the flags: loader.py clears
            # them whenever it forwards one.
            **planner_quantization_kwargs(
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                quantization_config = kwargs.get("quantization_config", None),
                # The same extra the bnb config below adds.
                extra_skip_modules = ["out_proj"] if IS_FALCON_H1 else None,
            ),
        )

        bnb_config = None
        _ckpt_qcfg = getattr(model_config, "quantization_config", None)

        if load_in_4bit:
            llm_int8_skip_modules = SKIP_QUANTIZATION_MODULES.copy()
            if IS_FALCON_H1:
                # out_proj cannot be quantized due to mamba kernels; see tiiuae/Falcon-H1#13 (comment 2918671274).
                llm_int8_skip_modules.append("out_proj")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = dtype,
                llm_int8_skip_modules = llm_int8_skip_modules,
            )
            # Pre-quantized checkpoints use the quantization_config baked into config.json, ignoring our
            # runtime BitsAndBytesConfig, so merge our skip list into it to keep task heads like `score` in
            # compute dtype (#5027).
            if _ckpt_quant_method == "bitsandbytes" and _ckpt_qcfg is not None:
                if isinstance(_ckpt_qcfg, dict):
                    _ckpt_skip = list(_ckpt_qcfg.get("llm_int8_skip_modules") or [])
                    for _m in llm_int8_skip_modules:
                        if _m not in _ckpt_skip:
                            _ckpt_skip.append(_m)
                    _ckpt_qcfg["llm_int8_skip_modules"] = _ckpt_skip
                else:
                    _ckpt_skip = list(getattr(_ckpt_qcfg, "llm_int8_skip_modules", None) or [])
                    for _m in llm_int8_skip_modules:
                        if _m not in _ckpt_skip:
                            _ckpt_skip.append(_m)
                    try:
                        _ckpt_qcfg.llm_int8_skip_modules = _ckpt_skip
                    except Exception:
                        pass

        # RoPE Scaling's max_position_embeddings must be updated; see togethercomputer/LLaMA-2-7B-32K discussion 12.
        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        kwargs.pop("attn_implementation", None)  # No need since we auto call it

        # Cannot be None, since HF now checks for the config.
        if load_in_4bit:
            kwargs["quantization_config"] = bnb_config

        kwargs = add_dtype_kwargs(dtype, kwargs)

        raise_handler = RaiseUninitialized()
        try:
            if num_labels is not None:
                # Transformers 5.x @strict config classes reject unexpected kwargs like num_labels and
                # max_position_embeddings, so set them on the config object and pass config= instead.
                set_task_config_attr(model_config, "num_labels", num_labels)
                if max_position_embeddings is not None:
                    model_config.max_position_embeddings = max_position_embeddings
                # Pop config-level attrs that would be rejected by @strict model init.
                for _cfg_key in ("id2label", "label2id", "rope_scaling"):
                    _cfg_val = kwargs.pop(_cfg_key, None)
                    if _cfg_val is not None:
                        if _cfg_key in ("id2label", "label2id"):
                            set_task_config_attr(model_config, _cfg_key, _cfg_val)
                        else:
                            setattr(model_config, _cfg_key, _cfg_val)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    config = model_config,
                    device_map = device_map,
                    token = token,
                    trust_remote_code = trust_remote_code,
                    attn_implementation = preferred_attn_impl,
                    revision = revision,
                    **kwargs,
                )
                # Defensive: ensure the task head is in a floating dtype, guarding against any path leaving it
                # as integer storage (#5027).
                for _head_name in ("score", "classifier", "qa_outputs"):
                    _head = getattr(model, _head_name, None)
                    if (
                        _head is not None
                        and hasattr(_head, "weight")
                        and not _head.weight.is_floating_point()
                    ):
                        _head.to(dtype)
                # Attach dispatch hooks for bnb multi-device loads.
                from unsloth.models.vision import _attach_bnb_multidevice_hooks

                _attach_bnb_multidevice_hooks(
                    model,
                    load_in_4bit = load_in_4bit,
                    load_in_8bit = kwargs.get("load_in_8bit", False),
                    offload_embedding = False,
                    fast_inference = fast_inference,
                )
                # Re-apply block-fp8 weight_scale_inv tensors transformers dropped on load (#6200), reading
                # scales from the same revision as the weights.
                _restore_dropped_fp8_scales(
                    model,
                    model_name,
                    local_files_only = kwargs.get("local_files_only", False),
                    token = token,
                    revision = revision,
                    subfolder = kwargs.get("subfolder"),
                    cache_dir = kwargs.get("cache_dir"),
                    variant = kwargs.get("variant"),
                )
            elif not fast_inference:
                if user_config is not None:
                    # Transformers 5.x @strict model init rejects extra kwargs next to config=, so set the override
                    # on the config and pass the single config object through.
                    if max_position_embeddings is not None:
                        model_config.max_position_embeddings = max_position_embeddings
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config = model_config,
                        device_map = device_map,
                        token = token,
                        trust_remote_code = trust_remote_code,
                        attn_implementation = preferred_attn_impl,
                        revision = revision,
                        **kwargs,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map = device_map,
                        token = token,
                        max_position_embeddings = max_position_embeddings,
                        trust_remote_code = trust_remote_code,
                        attn_implementation = preferred_attn_impl,
                        revision = revision,
                        **kwargs,
                    )
                from unsloth.models.vision import _attach_bnb_multidevice_hooks

                _attach_bnb_multidevice_hooks(
                    model,
                    load_in_4bit = load_in_4bit,
                    load_in_8bit = kwargs.get("load_in_8bit", False),
                    offload_embedding = False,
                    fast_inference = False,
                )
                # Re-apply block-fp8 weight_scale_inv tensors transformers dropped on load (#6200), reading
                # scales from the same revision as the weights.
                _restore_dropped_fp8_scales(
                    model,
                    model_name,
                    local_files_only = kwargs.get("local_files_only", False),
                    token = token,
                    revision = revision,
                    subfolder = kwargs.get("subfolder"),
                    cache_dir = kwargs.get("cache_dir"),
                    variant = kwargs.get("variant"),
                )
                model.fast_generate = make_fast_generate_wrapper(model.generate)
                model.fast_generate_batches = None
            else:
                from unsloth_zoo.vllm_utils import (
                    load_vllm,
                    get_vllm_state_dict,
                    convert_vllm_to_huggingface,
                    generate_batches,
                )

                fp8_mode = None
                if load_in_fp8 != False:
                    fp8_mode = _get_fp8_mode_and_check_settings(
                        load_in_fp8,
                        fast_inference,
                    )

                allowed_args = inspect.getfullargspec(load_vllm).args
                load_vllm_kwargs = dict(
                    model_name = model_name,
                    config = model_config,
                    gpu_memory_utilization = gpu_memory_utilization,
                    max_seq_length = max_seq_length,
                    dtype = dtype,
                    float8_kv_cache = float8_kv_cache,
                    enable_lora = True,
                    max_lora_rank = max_lora_rank,
                    disable_log_stats = disable_log_stats,
                    use_bitsandbytes = load_in_4bit,
                    unsloth_vllm_standby = unsloth_vllm_standby,
                    fp8_mode = fp8_mode,
                )
                for allowed_arg in allowed_args:
                    if allowed_arg not in load_vllm_kwargs and allowed_arg in kwargs:
                        load_vllm_kwargs[allowed_arg] = kwargs[allowed_arg]
                pass

                llm = load_vllm(**load_vllm_kwargs)

                _, quant_state_dict = get_vllm_state_dict(
                    llm,
                    config = model_config,
                    load_in_fp8 = load_in_fp8,
                )
                model = convert_vllm_to_huggingface(
                    quant_state_dict, model_config, dtype, bnb_config
                )
                model.vllm_engine = llm
                llm.shared_weights = True
                model.fast_generate = model.vllm_engine.generate
                model.fast_generate_batches = functools.partial(generate_batches, model.vllm_engine)
        finally:
            raise_handler.remove()
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Counteract saved tokenizers.
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        # Route the tokenizer load to the custom cache_dir the prefetch warmed.
        _tokenizer_cache_kwargs = {}
        if kwargs.get("cache_dir") is not None:
            _tokenizer_cache_kwargs["cache_dir"] = kwargs["cache_dir"]
        tokenizer = load_correct_tokenizer(
            tokenizer_name = tokenizer_name,
            model_max_length = max_position_embeddings,
            padding_side = "right",
            token = token,
            trust_remote_code = trust_remote_code,
            fix_tokenizer = fix_tokenizer,
            revision = tokenizer_revision,
            **_tokenizer_cache_kwargs,
        )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model, tokenizer = model_patcher.post_patch(model, tokenizer, correct_dtype = dtype)

        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o = original_apply_o

        from transformers.trainer import Trainer

        try:
            if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
                inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
                Trainer._original_training_loop = inner_training_loop
            else:
                inner_training_loop = Trainer._original_training_loop
        except:
            raise RuntimeError("Unsloth: Unsuccessfully patched inner_training_loop")

        import transformers.trainer

        items_in_trainer = dir(transformers.trainer)
        good_items = []
        for item in items_in_trainer:
            if item in inner_training_loop:
                good_items.append(item)
        exec(
            "from transformers.trainer import (" + ", ".join(x for x in good_items) + ")",
            globals(),
        )

        start = re.search(r"logger\.info\([\"\'].+?Running training", inner_training_loop).span(0)[
            0
        ]
        end = inner_training_loop.find("\n\n", start)
        original_debug = inner_training_loop[start:end]
        spaces = re.search(r"\n([\s\t]{1,})", original_debug).group(0)[1:]
        front_spaces = re.match(r"([\s\t]{1,})", inner_training_loop).group(0)

        # Cannot use a backslash escape (SyntaxWarning in Python 3.12), so use chr(92).
        debug_info = """debug_info = \\
        f"==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = {len(set(p.device for p in model.parameters()))}\\n"\\
        f"   {chr(92)}{chr(92)}   /|    Num examples = {num_examples:,} | Num Epochs = {num_train_epochs:,} | Total steps = {max_steps:,}\\n"\\
        f"O^O/ {chr(92)}_/ {chr(92)}    Batch size per device = {self._train_batch_size:,} | Gradient accumulation steps = {args.gradient_accumulation_steps}\\n"\\
        f"{chr(92)}        /    Data Parallel GPUs = {args.world_size} | Total batch size ({self._train_batch_size} x {args.gradient_accumulation_steps} x {args.world_size}) = {total_train_batch_size:,}\\n"\\
        f' "-____-"     Trainable parameters = {get_model_param_count(model, trainable_only=True):,} of {get_model_param_count(model):,} ({get_model_param_count(model, trainable_only=True)/get_model_param_count(model)*100:.2f}% trained)'
        logger.warning(debug_info)
        import gc
        for _ in range(3):
            gc.collect()
            clean_gpu_cache()"""

        debug_info = debug_info.split("\n")
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

        debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once('Unsloth is running with multi GPUs - the effective batch size is multiplied by ' + str(n_total_devices))
        debug_info ="""
        debug_info = debug_info.split("\n")
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace("debug_info =", debug_info, 1)

        front_spaces = re.match(r"[\t\s]{1,}", inner_training_loop).group(0)
        inner_training_loop = re.sub(
            r"^" + front_spaces, "", inner_training_loop, flags = re.MULTILINE
        )
        inner_training_loop = inner_training_loop.replace(
            "train_dataloader = tpu_spmd_dataloader(train_dataloader)",
            "raise RuntimeError('Unsloth: TPUs are not yet supported!')",
        )
        inner_training_loop = inner_training_loop.replace(
            "_inner_training_loop",
            "_fast_inner_training_loop",
            1,
        )
        inner_training_loop = inner_training_loop.replace(
            "is_torch_tpu_available()",
            "False",
        )
        # Wire the stray-forward compile-cache reset into the plain Trainer path: get_peft_model arms
        # the detector for every LoRA model, but only the TRL SFT/RL wrappers run the reset, so a
        # grad-enabled probe before a bare Trainer.train() would keep the poisoned Dynamo cache.
        # Anchored on the first body statement; a harmless no-op if upstream drops that line.
        inner_training_loop = inner_training_loop.replace(
            "self.accelerator.free_memory()",
            "self.accelerator.free_memory()\n"
            "    try:\n"
            "        from unsloth.models._utils import _unsloth_reset_stray_compile_cache as _unsloth_reset_cc\n"
            "        _unsloth_reset_cc(self)\n"
            "    except Exception: pass",
            1,
        )
        exec(inner_training_loop, globals())
        Trainer._inner_training_loop = _fast_inner_training_loop

        model.max_seq_length = max_seq_length
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m = m.model
        m.max_seq_length = max_seq_length
        for module in model.modules():
            module.max_seq_length = max_seq_length

        if fix_tokenizer:
            tokenizer = check_tokenizer(
                model = model,
                tokenizer = tokenizer,
                model_name = model_name,
                model_max_length = max_position_embeddings,
                padding_side = "right",
                token = token,
                cache_dir = kwargs.get("cache_dir"),
            )
        patch_saving_functions(tokenizer)

        # Not necessary since we requires transformers >= 4.37
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[: len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path": name})

        # Log the Unsloth version for future inference fastpaths.
        model.config.update({"unsloth_version": __version__})

        patch_saving_functions(model)
        Trainer._inner_training_loop = _fast_inner_training_loop

        # Fix gradient accumulation; see #4982.
        apply_accepts_loss_kwargs_fix(model)
        patch_gradient_accumulation_fix(Trainer)

        tokenizer.padding_side = "left"
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model._saved_temp_tokenizer = tokenizer

            internal_model = internal_model.model
        internal_model._saved_temp_tokenizer = tokenizer
        # Prevent Transformers Trainer from auto-wrapping Unsloth LoRA models in DP.
        _mark_unsloth_disable_data_parallel(model)

        # For transformers > 4.47.1, rotary_emb must be added to all attention layers.
        if IS_ATTENTION_REFACTOR or hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = rotary_emb

        model.for_training = functools.partial(FastLlamaModel.for_training, model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model

        is_classification = "Classification" in str(type(model))
        if not is_classification and model.generate.__name__ != "unsloth_fast_generate":
            model._old_generate = model.generate
            unsloth_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_fast_generate, model)
        # Zero weight[padding_idx] only for embeddings NOT tied to lm_head: when tied, zeroing the row
        # forces pad logit = 0, which beats the negative logits of real tokens (Gemma) and makes the
        # decoder emit <pad>. Skipped when eos_token == pad_token.
        eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if tokenizer is not None and eos_token_id != pad_token_id:
            lm_head = getattr(model, "lm_head", None)
            lm_head_weight = getattr(lm_head, "weight", None) if lm_head is not None else None
            with torch.no_grad():
                for name, module in model.named_modules():
                    if type(module) is torch.nn.Embedding:
                        if (
                            getattr(module, "weight", None) is not None
                            and getattr(module, "padding_idx", None) is not None
                        ):
                            if module.padding_idx < module.weight.shape[0]:
                                if (
                                    lm_head_weight is not None
                                    and module.weight.data_ptr() == lm_head_weight.data_ptr()
                                ):
                                    continue
                                module.weight[module.padding_idx] = 0

        # LAST: post_patch replaces the embedding modules and the QKV/MLP patching below replaces the
        # forwards a hook wraps, so an earlier attach is lost. Skipped under vLLM, which owns the
        # weights.
        if not fast_inference:
            try:
                from unsloth.models.vision import _repair_dispatch_hooks
                _repaired = _repair_dispatch_hooks(model)
                if _repaired:
                    logger.info(
                        f"Unsloth: re-attached dispatch hooks to {_repaired} module(s) "
                        "left unhooked by the split, so the model trains."
                    )
            except Exception as _exc:
                logger.warning(
                    f"Unsloth: could not check the dispatch hooks "
                    f"({type(_exc).__name__}: {_exc})."
                )
        return model, tokenizer

    @staticmethod
    def post_patch(
        model,
        tokenizer,
        correct_dtype = None,
    ):
        model, tokenizer = patch_model_and_tokenizer(
            model, tokenizer, downcast_rope = True, correct_dtype = correct_dtype
        )
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        layers_to_transform = None,
        layers_pattern = None,
        finetune_last_n_layers = None,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,  # not used anymore
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        temporary_location = "_unsloth_temporary_saved_buffers",
        qat_scheme = None,
        target_parameters = None,  # For MoE expert layers (nn.Parameter)
        ensure_weight_tying = None,  # None = auto (tie when we redirect a tied pair)
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            for peft_arg, flag in (
                ("finetune_vision_layers", False),
                ("finetune_language_layers", True),
                ("finetune_attention_modules", True),
                ("finetune_mlp_modules", True),
                ("finetune_audio_layers", False),
            ):
                if peft_arg not in kwargs:
                    kwargs[peft_arg] = flag
            return FastBaseModel.get_peft_model(
                model = model,
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                layers_to_transform = layers_to_transform,
                layers_pattern = layers_pattern,
                finetune_last_n_layers = finetune_last_n_layers,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                max_seq_length = max_seq_length,
                use_rslora = use_rslora,
                modules_to_save = modules_to_save,
                init_lora_weights = init_lora_weights,
                loftq_config = loftq_config,
                temporary_location = temporary_location,
                target_parameters = target_parameters,
                ensure_weight_tying = ensure_weight_tying,
                **kwargs,
            )
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            # Full finetuning still compiles, so a stray pre-train forward can poison the cache; install the
            # detector here too (idempotent).
            _unsloth_install_pretrain_detector(model)
            return model
        transformers_set_seed(random_state)

        max_seq = getattr(model, "max_seq_length", 512)
        dtype = model.get_input_embeddings().weight.dtype
        use_gradient_checkpointing = apply_unsloth_gradient_checkpointing(
            use_gradient_checkpointing, max_seq, dtype
        )

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM) or isinstance(
            model, PeftModelForSequenceClassification
        ):
            assert hasattr(model, "peft_config")

            peft_config = model.peft_config["default"].to_dict()
            check_parameters = [
                "r",
                "lora_alpha",
                "lora_dropout",
                "bias",
                "layers_to_transform",
                "layers_pattern",
                "use_rslora",
                "init_lora_weights",
            ]
            check_all = True
            for param in check_parameters:
                check_all = check_all and (peft_config[param] == eval(param))

            # use_dora arrives via **kwargs, not as a named parameter, so it cannot go in check_parameters
            # (which evals each entry as a bound local); compare it explicitly, or a DoRA request against an
            # existing plain-LoRA adapter silently passes through unchanged.
            check_all = check_all and (
                bool(peft_config.get("use_dora", False)) == bool(kwargs.get("use_dora", False))
            )

            # Everything the stored adapter trains, wherever PEFT filed it: tying moves the counterpart to
            # modules_to_tie, which to_dict() drops, so read the config object.
            requested_modules_to_save = modules_to_save
            modules_to_save = list(peft_config["modules_to_save"] or [])
            old_target_modules = list(peft_config["target_modules"]) + modules_to_save
            old_target_modules += list(
                getattr(model.peft_config["default"], "modules_to_tie", None) or []
            )

            # The new side is what THIS call asks for, through the same redirect; built from the stored
            # lists instead, a narrowed request would compare equal.
            requested_targets, requested_saved, _ = _redirect_embedding_targets(
                target_modules,
                requested_modules_to_save,
                skip = _vllm_unmovable_embedding_modules(model, target_modules),
            )
            new_target_modules = list(requested_targets) + list(requested_saved or [])
            if isinstance(model, PeftModelForSequenceClassification):
                # PeftModelForSequenceClassification.__init__ extends modules_to_save with these
                # unconditionally, so the stored config names modules the caller never asked for. Mirror it, or
                # an unchanged request looks different.
                new_target_modules += ["classifier", "score"]
            # Tying is not in check_parameters and leaves both lists looking the same, so compare it
            # separately, on what it does rather than what was asked: on an untied model the request changes
            # nothing either way.
            _requested_ties = _effective_weight_tying(
                model,
                requested_saved,
                ensure_weight_tying,
            )
            _stored_ties = bool(getattr(model.peft_config["default"], "modules_to_tie", None))
            check_all = check_all and (_requested_ties == _stored_ties)
            if _requested_ties:
                # PEFT files the counterpart under modules_to_tie, which the old side counts, so name it here
                # even when the caller listed only one side.
                new_target_modules += list(EMBEDDING_MODULES)
            # Per-expert Linear MoE experts (gpt-oss bnb-4bit) were auto-added to the saved target_modules
            # at creation, so recompute them and keep a repeat get_peft_model call idempotent.
            new_target_modules += get_moe_target_modules(model, target_modules)

            # Fold away PEFT's model.embed_tokens alias, and only that one: layers.0.q_proj is a real target
            # and is not layers.1.q_proj.
            def _leaf_name(module):
                return _embedding_leaf(module) or module

            new_target_modules = {_leaf_name(x) for x in new_target_modules}
            old_target_modules = {_leaf_name(x) for x in old_target_modules}
            check_all = check_all and (len(old_target_modules ^ new_target_modules) == 0)

            check_all = check_all and (
                (loftq_config == {} or loftq_config is None)
                and (peft_config["loftq_config"] == {} or peft_config["loftq_config"] is None)
            )

            if check_all:
                logger.warning("Unsloth: Already have LoRA adapters! We shall skip this step.")

                if "embed_tokens" in new_target_modules:
                    print("Unsloth: Training embed_tokens in mixed precision to save VRAM")

                    _offload_frozen_module_for_training(
                        model.get_input_embeddings(), DEVICE_TYPE_TORCH
                    )

                if "lm_head" in new_target_modules:
                    print("Unsloth: Training lm_head in mixed precision to save VRAM")

                    _offload_frozen_module_for_training(
                        model.get_output_embeddings(), DEVICE_TYPE_TORCH
                    )

                # A pre-wrapped PEFT model passes through here; still arm the detector so an RL trainer can
                # reset a compile cache poisoned by a pre-train forward.
                _unsloth_install_pretrain_detector(model)
                # This branch returns before patch_peft_model, so record here too;
                # apply_unsloth_gradient_checkpointing above already re-patched global state to match (#4735).
                model._unsloth_gradient_checkpointing = use_gradient_checkpointing
                model = _exclude_rope_inv_freq_from_ddp(model)
                return model
            else:
                raise TypeError(
                    "Unsloth: Your model already has LoRA adapters. Your new parameters are different."
                )

        if loftq_config is None:
            loftq_config = {}

        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora" in signature

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )

        if not (
            type(init_lora_weights) is bool
            or init_lora_weights == "gaussian"
            or init_lora_weights == "loftq"
            or init_lora_weights == "corda"
        ):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq", "corda"].'
            )

        if init_lora_weights == "loftq":
            if not SUPPORTS_LOFTQ:
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )

            if loftq_config == {}:
                from peft import LoftQConfig
                logger.warning_once(
                    "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"
                    "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)

            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )

        assert type(use_rslora) is bool
        if use_rslora:
            if not SUPPORTS_RSLORA:
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )

        accepted_modules = frozenset(
            (
                "lm_head",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        model.config.update({"unsloth_version": __version__})

        if type(modules_to_save) is tuple:
            modules_to_save = list(modules_to_save)

        train_lm_head = False
        train_embed_tokens = False
        final_modules = []
        # LoRA on these never trains (see _redirect_embedding_targets); modules_to_save does, and is
        # what embedding_learning_rate matches.
        target_modules, modules_to_save, _moved_embedding_modules = _redirect_embedding_targets(
            target_modules,
            modules_to_save,
            skip = _vllm_unmovable_embedding_modules(model, target_modules),
        )
        _raise_if_no_lora_targets_left(
            target_modules,
            _moved_embedding_modules,
            target_parameters,
        )
        ensure_weight_tying = _effective_weight_tying(
            model,
            modules_to_save,
            ensure_weight_tying,
        )
        modules_to_save = _drop_tied_output_module(
            model,
            modules_to_save,
            ensure_weight_tying,
        )
        for module in _moved_embedding_modules:
            if _embedding_leaf(module) == "embed_tokens":
                train_embed_tokens = True
            else:
                train_lm_head = True
        if _moved_embedding_modules:
            logger.warning_once(
                f"Unsloth: Moved {', '.join(_moved_embedding_modules)} from `target_modules` "
                f"to `modules_to_save`, so they are trained as full weight matrices.\n"
                f"This uses more VRAM than LoRA. Please list them in `modules_to_save` directly."
            )
        for module in target_modules:
            try:
                assert module in accepted_modules
                final_modules.append(module)
            except AssertionError as e:
                final_modules.append(module)
                print(
                    "Unsloth: You added custom modules, but Unsloth hasn't optimized for this.\n"
                    "Beware - your finetuning might be noticeably slower!"
                )

        if hasattr(model, "_need_to_train_embeddings"):
            # embed_tokens/lm_head may already be trained, either as LoRA targets in final_modules or via
            # modules_to_save.
            _embed_already_trained = train_embed_tokens or "embed_tokens" in final_modules
            _lm_head_already_trained = train_lm_head or "lm_head" in final_modules
            if not _lm_head_already_trained or not _embed_already_trained:
                print(
                    "Unsloth: You added new tokens but did not specify if you wanted to "
                    "train the lm_head and embed_tokens.\nWe must turn it on for you."
                )

                # Only add to modules_to_save if not already a LoRA target.
                if not _embed_already_trained:
                    train_embed_tokens = True
                    if modules_to_save is None:
                        modules_to_save = ["embed_tokens"]
                    elif "embed_tokens" not in modules_to_save:
                        modules_to_save.append("embed_tokens")

                if not _lm_head_already_trained:
                    train_lm_head = True
                    if modules_to_save is None:
                        modules_to_save = ["lm_head"]
                    elif "lm_head" not in modules_to_save:
                        modules_to_save.append("lm_head")


        # Fixing untrained tokens here is wrong: it can cause reserved tokens to pop out.

        if modules_to_save is not None:
            for module in modules_to_save:
                # By leaf: PEFT resolves model.embed_tokens to the same module.
                leaf = _embedding_leaf(module)
                if leaf == "lm_head":
                    train_lm_head = True
                elif leaf == "embed_tokens":
                    train_embed_tokens = True
                else:
                    raise TypeError(
                        f"Unsloth: Module = {module} is not allowed. Only 'lm_head' and 'embed_tokens' is allowed."
                    )
        if isinstance(modules_to_save, (tuple, list)):
            modules_to_save = list(set(modules_to_save))

        vllm_engine = None
        if hasattr(model, "vllm_engine"):
            vllm_engine = model.vllm_engine
            vllm_fast_generate = model.fast_generate
            vllm_fast_generate_batches = model.fast_generate_batches

            if modules_to_save is not None:
                raise NotImplementedError(
                    "Unsloth: Currently fast inference does not work with training embeddings or lm_head."
                )

            if bias != "none":
                raise NotImplementedError(
                    "Unsloth: Currently fast inference does not work with using biases for LoRA."
                )

        # Does not get lora yet, so take the name from model, not base model.
        is_classification = "Classification" in str(type(model))

        # Auto-detect MoE models and populate target_parameters for expert layers.
        if target_parameters is None:
            target_parameters = get_moe_target_parameters(model, target_modules)

        # Per-expert Linear expert layouts (gpt-oss bnb-4bit) are Linear modules, not fused Parameters,
        # so target them via target_modules.
        _moe_module_targets = get_moe_target_modules(model, target_modules)
        if _moe_module_targets:
            _added = [t for t in _moe_module_targets if t not in final_modules]
            final_modules.extend(_added)
            if _added:
                print(
                    f"Unsloth: Detected MoE model with per-expert Linear experts. "
                    f"Enabling LoRA on {len(_added)} expert projection modules."
                )
                warn_if_zoo_cannot_merge_moe_experts()

        if finetune_last_n_layers is not None and layers_to_transform is None:
            from .vision import _get_total_transformer_layers
            _total_layers = _get_total_transformer_layers(model)
            if _total_layers is not None and _total_layers > 0:
                _n = max(1, min(int(finetune_last_n_layers), _total_layers))
                layers_to_transform = list(range(_total_layers - _n, _total_layers))

        arguments = dict(
            r = r,
            lora_alpha = lora_alpha,
            target_modules = final_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = TaskType.CAUSAL_LM if not is_classification else TaskType.SEQ_CLS,
            layers_to_transform = layers_to_transform,
            init_lora_weights = init_lora_weights,
            loftq_config = loftq_config,
            use_rslora = use_rslora,
            modules_to_save = modules_to_save,
            target_parameters = target_parameters,
            ensure_weight_tying = ensure_weight_tying,
            **kwargs,
        )
        # Older PEFT has no ensure_weight_tying; passing it would TypeError.
        if "ensure_weight_tying" not in inspect.signature(LoraConfig).parameters:
            del arguments["ensure_weight_tying"]
        if not SUPPORTS_LOFTQ:
            del arguments["loftq_config"]
        if not SUPPORTS_RSLORA:
            del arguments["use_rslora"]

        _saved_temp_tokenizer = model._saved_temp_tokenizer

        lora_config = LoraConfig(**arguments)
        input_embeddings_device = model.get_input_embeddings().weight.device
        if is_classification:
            output_embeddings_device = model.score.weight.device
        else:
            output_embeddings_device = model.get_output_embeddings().weight.device

        if use_gradient_checkpointing == "unsloth":
            if train_embed_tokens:
                print("Unsloth: Offloading input_embeddings to disk to save VRAM")
                offload_input_embeddings(model, temporary_location)

            for _ in range(3):
                gc.collect()
                clean_gpu_cache()

            if train_lm_head:
                print("Unsloth: Offloading output_embeddings to disk to save VRAM")
                offload_output_embeddings(model, temporary_location)

            for _ in range(3):
                gc.collect()
                clean_gpu_cache()

        model = _get_peft_model(model, lora_config)

        try:
            from .vision import _lift_endpoint_hooks_onto_adapters
            _lifted = _lift_endpoint_hooks_onto_adapters(model)
            if _lifted:
                logger.info(
                    f"Unsloth: lifted dispatch hooks onto {_lifted} adapter-wrapped "
                    "embedding module(s), so a split model trains them."
                )
        except Exception as _exc:
            logger.warning_once(f"Unsloth: could not lift adapter hooks: {_exc}")
        fix_lora_auto_mapping(model)

        if qat_scheme is not None:
            print("Unsloth: Applying QAT to mitigate quantization degradation")
            model = FastLlamaModel._prepare_for_qat(model, qat_scheme)

        model._saved_temp_tokenizer = _saved_temp_tokenizer

        model = FastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)

        if ensure_weight_tying:
            try:
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()

                if input_embeddings is not None and output_embeddings is not None:

                    def _retie_parameter(target_module, source_module):
                        if not hasattr(source_module, "weight"):
                            return
                        weight = source_module.weight
                        if "weight" in getattr(target_module, "_parameters", {}):
                            target_module._parameters.pop("weight")
                        if hasattr(target_module, "weight"):
                            try:
                                delattr(target_module, "weight")
                            except Exception as exc:
                                logger.warning_once(
                                    f"Unsloth: Could not delete existing weight attr during retie on "
                                    f"{type(target_module).__name__}: {exc}"
                                )
                        target_module.register_parameter("weight", weight)

                    # Tie the trainable copies created by ModulesToSaveWrapper first, since those are used in
                    # forward, then the original_module references if present.
                    if hasattr(input_embeddings, "modules_to_save") and hasattr(
                        output_embeddings, "modules_to_save"
                    ):
                        if hasattr(input_embeddings.modules_to_save, "default") and hasattr(
                            output_embeddings.modules_to_save, "default"
                        ):
                            _retie_parameter(
                                output_embeddings.modules_to_save.default,
                                input_embeddings.modules_to_save.default,
                            )

                    if hasattr(input_embeddings, "original_module") and hasattr(
                        output_embeddings, "original_module"
                    ):
                        _retie_parameter(
                            output_embeddings.original_module,
                            input_embeddings.original_module,
                        )
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Failed to ensure weight tying between embeddings and lm_head: {e}"
                )

        if train_embed_tokens:
            print("Unsloth: Training embed_tokens in mixed precision to save VRAM")
            assert hasattr(model.get_input_embeddings(), "modules_to_save")

            _offload_frozen_module_for_training(
                model.get_input_embeddings(),
                DEVICE_TYPE_TORCH,
                offload_device = None,
                original_device = input_embeddings_device,
            )

        if train_lm_head:
            print("Unsloth: Training lm_head in mixed precision to save VRAM")
            assert hasattr(model.get_output_embeddings(), "modules_to_save")

            _offload_frozen_module_for_training(
                model.get_output_embeddings(),
                DEVICE_TYPE_TORCH,
                offload_device = None,
                original_device = output_embeddings_device,
            )

        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            internal_model = internal_model.model
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        # Prevent Transformers Trainer from auto-wrapping Unsloth LoRA models in DP.
        _mark_unsloth_disable_data_parallel(model)

        for _ in range(3):
            gc.collect()
            clean_gpu_cache()

        patch_peft_fast_inference(model)

        model.for_training = functools.partial(FastLlamaModel.for_training, model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        # Detect a stray pre-train forward so train() can drop the torch.compile graph cache it would
        # otherwise poison (see prepare_for_training_mode).
        _unsloth_install_pretrain_detector(model)
        model = _exclude_rope_inv_freq_from_ddp(model)
        return model

    @staticmethod
    def patch_peft_model(model, use_gradient_checkpointing = "unsloth"):
        # Persist the effective GC mode so the trainer restores it verbatim: for_inference() clears the
        # module flags every GRPO step, and TrainingArguments defaults it to False, which would silently
        # disable it at train time (#4735). Recorded here so loader.py's from_pretrained path is covered.
        model._unsloth_gradient_checkpointing = use_gradient_checkpointing
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            return FastBaseModel.patch_peft_model(
                model = model,
                use_gradient_checkpointing = use_gradient_checkpointing,
            )
        if not isinstance(model, PeftModelForCausalLM) and not isinstance(
            model, PeftModelForSequenceClassification
        ):
            raise TypeError("Unsloth: Your model needs to call `.get_peft_model` first!")

        model_type = model.config.model_type

        if model_type == "llama":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "mistral":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen2":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "gemma":
            apply_lora_mlp = apply_lora_mlp_geglu_approx
        elif model_type == "gemma2":
            apply_lora_mlp = apply_lora_mlp_geglu_approx
        elif model_type == "cohere":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "granite":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen3":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "falcon_h1":
            apply_lora_mlp = apply_lora_mlp_swiglu
        elif model_type == "qwen3moe":
            apply_lora_mlp = apply_lora_mlp_swiglu
        else:
            raise NotImplementedError(f"Unsloth: {model_type} is not yet implemented!")

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )

        for active_adapter in model.peft_config.keys():
            if False:
                name = model.peft_config[active_adapter].base_model_name_or_path
                if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                    name = name[: len(name) - len("-bnb-4bit")]
                    model.peft_config[active_adapter].base_model_name_or_path = name
                pass
            # Adding a revision to enable future fast inference paths bugs out; see #492.

        from transformers.trainer import Trainer

        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            raise RuntimeError("Unsloth: Unsuccessfully patched Trainer! Please file a bug report!")

        # loftq_config must not be None, but rather {}.
        all_configs = model.peft_config
        for key, current_config in all_configs.items():
            if hasattr(current_config, "loftq_config") and current_config.loftq_config is None:
                new_args = current_config.__dict__
                new_args["loftq_config"] = {}
                current_config = current_config.__class__(**new_args)
                all_configs[key] = current_config

        n_mlp = 0
        n_qkv = 0
        n_o = 0

        active_adapter = (
            model.active_adapters[0] if hasattr(model, "active_adapters") else model.active_adapter
        )

        lora_dropout = model.peft_config[active_adapter].lora_dropout
        bias = model.peft_config[active_adapter].bias

        # QKV is not inplace-edited for Cohere.
        _apply_lora_mlp = (
            functools.partial(apply_lora_mlp, inplace = False)
            if model_type == "cohere"
            else apply_lora_mlp
        )

        if lora_dropout == 0 and bias == "none":
            for idx, layer in enumerate(model.model.model.layers):
                if model_type != "falcon_h1":
                    # LoRAMLP.apply has no gate/down multiplier support yet, so falcon h1 is not patched for now.

                    mlp_module = layer.mlp
                    gate_proj = mlp_module.gate_proj
                    up_proj = mlp_module.up_proj
                    down_proj = mlp_module.down_proj

                    if (
                        hasattr(gate_proj, "lora_A")
                        and hasattr(up_proj, "lora_A")
                        and hasattr(down_proj, "lora_A")
                        and (getattr(gate_proj, "base_layer", gate_proj).bias is None)
                        and (getattr(up_proj, "base_layer", up_proj).bias is None)
                        and (getattr(down_proj, "base_layer", down_proj).bias is None)
                        and (len(getattr(gate_proj, "lora_magnitude_vector", []) or []) == 0)
                        and (len(getattr(up_proj, "lora_magnitude_vector", []) or []) == 0)
                        and (len(getattr(down_proj, "lora_magnitude_vector", []) or []) == 0)
                    ):
                        # See stackoverflow.com/questions/50599045 on replacing a function within a class of a module.
                        if hasattr(mlp_module, "_unsloth_forward"):
                            # Then the mlp has been patched to use TiledMLP.
                            mlp_module._unsloth_forward = types.MethodType(
                                _apply_lora_mlp, mlp_module
                            )
                        else:
                            mlp_module.forward = types.MethodType(_apply_lora_mlp, mlp_module)
                        n_mlp += 1
                    else:
                        logger.warning_once(
                            "Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters\n"
                            "are not enabled or a bias term (like in Qwen) is used."
                        )

                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                if (
                    hasattr(q_proj, "lora_A")
                    and hasattr(k_proj, "lora_A")
                    and hasattr(v_proj, "lora_A")
                    and (getattr(q_proj, "base_layer", q_proj).bias is None)
                    and (getattr(k_proj, "base_layer", k_proj).bias is None)
                    and (getattr(v_proj, "base_layer", v_proj).bias is None)
                    and (len(getattr(q_proj, "lora_magnitude_vector", []) or []) == 0)
                    and (len(getattr(k_proj, "lora_magnitude_vector", []) or []) == 0)
                    and (len(getattr(v_proj, "lora_magnitude_vector", []) or []) == 0)
                ):
                    layer.self_attn.apply_qkv = apply_lora_qkv
                    n_qkv += 1
                else:
                    if model_type == "qwen2":
                        n_qkv += 1
                    else:
                        logger.warning_once(
                            "Not an error, but Unsloth cannot patch Attention layers with our manual autograd engine since either LoRA adapters\n"
                            "are not enabled or a bias term (like in Qwen) is used."
                        )

                o_proj = layer.self_attn.o_proj
                if (
                    hasattr(o_proj, "lora_A")
                    and (getattr(o_proj, "base_layer", o_proj).bias is None)
                    and (len(getattr(o_proj, "lora_magnitude_vector", []) or []) == 0)
                ):
                    layer.self_attn.apply_o = apply_lora_o
                    n_o += 1
                else:
                    logger.warning_once(
                        "Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n"
                        "are not enabled or a bias term (like in Qwen) is used."
                    )

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )
        patch_saving_functions(model)

        # Patch cross entropy loss labels; fixes #10.
        max_seq_length = model.max_seq_length
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        internal_model.max_seq_length = max_seq_length
        # Since transformers 4.53, must turn on explicitly
        # Since transformers 4.53, must turn off explicitly
        for module in model.modules():
            module.max_seq_length = max_seq_length

        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            internal_model = internal_model.model
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"

        for _ in range(3):
            gc.collect()
            clean_gpu_cache()

        patch_peft_fast_inference(model)

        model.for_training = functools.partial(FastLlamaModel.for_training, model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        _unsloth_install_pretrain_detector(model)
        return model

    @staticmethod
    def for_inference(model):
        if not hasattr(model, "parameters"):
            raise TypeError(
                "Unsloth: I think you're passing a tokenizer, not the model to for_inference!"
            )

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
            if hasattr(m, "training"):
                m.training = False
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "left"
            m._flag_for_generation = True

        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)
        model.eval()  # to turn off training on modules deeper in

        # Since transformers 4.53, this must be turned off explicitly.
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = False
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = False

        # Restore use_cache values that prepare_model_for_training disabled for gradient checkpointing
        # (older unsloth_zoo has no restore helper).
        try:
            from unsloth_zoo.training_utils import restore_use_cache
            restore_use_cache(model)
        except ImportError:
            pass
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError(
                "Unsloth: I think you're passing a tokenizer, not the model to for_training!"
            )

        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"):
                m.training = True
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "right"
            if hasattr(m, "_flag_for_generation"):
                try:
                    # A PEFT wrapper delegates the read but owns nothing to delete.
                    del m._flag_for_generation
                except AttributeError:
                    pass

        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)
        model.train()  # to turn on training on modules deeper in

        # Since transformers 4.53, this must be turned on explicitly.
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = use_gradient_checkpointing

        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = True
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = True

        # Re-disable use_cache if prepare_model_for_training had disabled it and for_inference restored
        # it; the record only exists after a disable.
        if (
            use_gradient_checkpointing
            and getattr(model, "_unsloth_use_cache_originals", None) is not None
        ):
            try:
                from unsloth_zoo.training_utils import disable_use_cache
                disable_use_cache(model)
            except ImportError:
                pass
        return model


from .rl import PatchFastRL

# Auto-enable grouped-GEMM MoE (tf<5 ModuleList experts) on built / PEFT'd models. Wrap the
# loader leaves before PatchFastRL so downstream patchers see the wrapped versions.
try:
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import wrap_loader_for_grouped_moe
    FastLlamaModel.from_pretrained = staticmethod(
        wrap_loader_for_grouped_moe(FastLlamaModel.from_pretrained)
    )
    FastLlamaModel.get_peft_model = staticmethod(
        wrap_loader_for_grouped_moe(FastLlamaModel.get_peft_model)
    )
except Exception:
    pass

PatchFastRL(FastLanguageModel = FastLlamaModel)
