# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-TTS fast fine-tuning support.

Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-*) is a TTS system whose trainable core is a
Qwen3-style autoregressive Talker (``Qwen3TTSTalkerForConditionalGeneration``)
that predicts codec tokens, plus a small Qwen3-style code predictor
(``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``) that fills in the
remaining RVQ codebooks. Both use QK-normalized attention exactly like Qwen3,
so Unsloth's fused QKV/O projections, fused RMS norms and Triton/SDPA
attention backends apply directly.

The Talker uses multimodal RoPE (mRoPE); there is no fused mRoPE kernel, so
the original ``apply_multimodal_rotary_pos_emb`` is kept for exact numerics.
The code predictor uses standard RoPE.

Usage mirrors the official Qwen3-TTS fine-tuning script, only faster::

    from unsloth.models.qwen3_tts import FastQwen3TTSModel
    FastQwen3TTSModel.pre_patch()  # swaps in the fast forwards below
    # ... then run finetuning/sft_12hz.py from QwenLM/Qwen3-TTS unchanged ...

or via the regular Unsloth entry point::

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", ...)
"""

from .loader_utils import DEFAULT_DEVICE_MAP
from .llama import *
from ..utils.packing import get_packed_info_from_kwargs
from ..utils.attention_dispatch import (
    AttentionConfig,
    AttentionContext,
    run_attention,
    SDPA,
    select_attention_backend,
    resolve_prefix_seg_info,
)
from .llama import (
    original_apply_qkv,
    original_apply_o,
    fast_rms_layernorm,
)

# Qwen3-TTS is not (yet) part of transformers itself; its modeling code ships
# with the `qwen-tts` PyPI package and as `trust_remote_code` on the Hub.
# Import it lazily-tolerantly: importing unsloth must never break when the
# package is absent -- the error is raised only when Qwen3-TTS is requested.
_QWEN3_TTS_IMPORT_ERROR = None
try:
    from transformers.models.qwen3_tts.modeling_qwen3_tts import (
        Qwen3TTSTalkerAttention,
        Qwen3TTSTalkerDecoderLayer,
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSAttention,
        Qwen3TTSDecoderLayer,
        Qwen3TTSTalkerCodePredictorModel,
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSRMSNorm,
        Qwen3TTSForConditionalGeneration,
        apply_multimodal_rotary_pos_emb,
        apply_rotary_pos_emb,
    )
    _QWEN3_TTS_SOURCE = "transformers"
except ImportError:
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerAttention,
            Qwen3TTSTalkerDecoderLayer,
            Qwen3TTSTalkerModel,
            Qwen3TTSTalkerForConditionalGeneration,
            Qwen3TTSAttention,
            Qwen3TTSDecoderLayer,
            Qwen3TTSTalkerCodePredictorModel,
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
            Qwen3TTSRMSNorm,
            Qwen3TTSForConditionalGeneration,
            apply_multimodal_rotary_pos_emb,
            apply_rotary_pos_emb,
        )
        _QWEN3_TTS_SOURCE = "qwen-tts"
    except ImportError as e:
        _QWEN3_TTS_IMPORT_ERROR = e
        _QWEN3_TTS_SOURCE = None
        Qwen3TTSTalkerAttention = None
        Qwen3TTSTalkerDecoderLayer = None
        Qwen3TTSTalkerModel = None
        Qwen3TTSTalkerForConditionalGeneration = None
        Qwen3TTSAttention = None
        Qwen3TTSDecoderLayer = None
        Qwen3TTSTalkerCodePredictorModel = None
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration = None
        Qwen3TTSRMSNorm = None
        Qwen3TTSForConditionalGeneration = None
        apply_multimodal_rotary_pos_emb = None
        apply_rotary_pos_emb = None


def _require_qwen3_tts():
    if _QWEN3_TTS_SOURCE is None:
        raise ImportError(
            "Unsloth: Qwen3-TTS support needs the Qwen3-TTS modeling code, which is not "
            "part of your installed transformers.\n"
            "Install it with `pip install qwen-tts`, or load the model with "
            "`trust_remote_code=True` so the modeling code comes from the Hub.\n"
            f"(import error was: {_QWEN3_TTS_IMPORT_ERROR})"
        )
    return


def _run_unsloth_attention(
    self,
    Q,
    K,
    V,
    bsz,
    q_len,
    n_heads,
    n_kv_heads,
    n_groups,
    head_dim,
    attention_mask,
    hidden_states,
    kwargs,
):
    """Shared tail of the fast forwards: backend selection + attention + o_proj."""
    seq_info = get_packed_info_from_kwargs(kwargs, hidden_states.device)
    use_varlen = seq_info is not None
    backend = SDPA if attention_mask is not None else select_attention_backend(use_varlen)
    attention_config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_groups,
        flash_dense_kwargs = {"causal": True},
        flash_varlen_kwargs = {
            "dropout_p": 0.0,
            "causal": True,
            "softmax_scale": getattr(self, "softmax_scale", None),
        },
    )
    _pg_seg = resolve_prefix_seg_info(kwargs, None, attention_mask)
    context = AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = K.shape[-2],
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = hidden_states.requires_grad,
        seq_info = seq_info,
        attention_mask = attention_mask,
        causal_mask = None,
        prefix_seg_info = _pg_seg,
    )
    A = run_attention(config = attention_config, context = context, Q = Q, K = K, V = V)
    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = getattr(self, "apply_o", original_apply_o)(self, attn_output)
    return attn_output


def Qwen3TTSTalkerAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fast Talker attention for training.

    Mirrors the original ``Qwen3TTSTalkerAttention.forward`` numerics, but uses
    Unsloth's fused QKV/O projections (LoRA-aware via ``apply_qkv``/``apply_o``),
    fused RMS QK-norms and the Triton/SDPA attention backends. The Talker's
    multimodal RoPE is applied with the original ``apply_multimodal_rotary_pos_emb``
    for exact numerics -- there is no fused mRoPE kernel.

    When a KV cache is passed (inference), falls back to the original forward so
    generation behaviour is bit-identical.
    """
    if past_key_values is not None:
        return self._unsloth_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            cache_position = cache_position,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = getattr(self, "apply_qkv", original_apply_qkv)(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim)
    K = K.view(bsz, q_len, n_kv_heads, head_dim)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    # Qwen3-style QK-Norm; a compiled norm mismatches Transformers' numbers,
    # so use the fused RMS norm (same reasoning as FastQwen3Model).
    Q = fast_rms_layernorm(self.q_norm, Q)
    K = fast_rms_layernorm(self.k_norm, K)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)

    cos, sin = position_embeddings
    Q, K = apply_multimodal_rotary_pos_emb(
        Q,
        K,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
        self.rope_scaling["interleaved"],
    )

    attn_output = _run_unsloth_attention(
        self,
        Q,
        K,
        V,
        bsz,
        q_len,
        n_heads,
        n_kv_heads,
        n_groups,
        head_dim,
        attention_mask,
        hidden_states,
        kwargs,
    )
    return attn_output, None


def Qwen3TTSAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fast code-predictor attention for training.

    Same treatment as the Talker, but the code predictor uses standard RoPE via
    the original ``apply_rotary_pos_emb``. Falls back to the original forward
    when a KV cache is passed (inference).
    """
    if past_key_values is not None:
        return self._unsloth_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            cache_position = cache_position,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = getattr(self, "apply_qkv", original_apply_qkv)(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim)
    K = K.view(bsz, q_len, n_kv_heads, head_dim)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    Q = fast_rms_layernorm(self.q_norm, Q)
    K = fast_rms_layernorm(self.k_norm, K)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)

    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    attn_output = _run_unsloth_attention(
        self,
        Q,
        K,
        V,
        bsz,
        q_len,
        n_heads,
        n_kv_heads,
        n_groups,
        head_dim,
        attention_mask,
        hidden_states,
        kwargs,
    )
    return attn_output, None


class FastQwen3TTSModel(FastLlamaModel):
    """Unsloth entry point for Qwen3-TTS fine-tuning.

    ``pre_patch()`` swaps the Talker and code-predictor attention forwards (and
    RMS norms) for Unsloth's fused versions. Everything else -- the Talker's
    causal-LM forward, the mRoPE rotary module, the speaker encoder -- is left
    untouched, so the official ``finetuning/sft_12hz.py`` training script from
    QwenLM/Qwen3-TTS runs unchanged, only faster and with less VRAM.
    """

    @staticmethod
    def pre_patch():
        _require_qwen3_tts()
        # Keep originals so inference (KV cache path) stays bit-identical.
        Qwen3TTSTalkerAttention._unsloth_original_forward = Qwen3TTSTalkerAttention.forward
        Qwen3TTSAttention._unsloth_original_forward = Qwen3TTSAttention.forward

        Qwen3TTSTalkerAttention.forward = Qwen3TTSTalkerAttention_fast_forward
        Qwen3TTSAttention.forward = Qwen3TTSAttention_fast_forward
        Qwen3TTSRMSNorm.forward = fast_rms_layernorm
        return

    @staticmethod
    def from_pretrained(
        model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        device_map = DEFAULT_DEVICE_MAP,
        rope_scaling = None,
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        **kwargs,
    ):
        _require_qwen3_tts()
        return FastLlamaModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            load_in_8bit = load_in_8bit,
            device_map = device_map,
            rope_scaling = rope_scaling,
            fix_tokenizer = fix_tokenizer,
            model_patcher = FastQwen3TTSModel,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
