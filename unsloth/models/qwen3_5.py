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

# Fixes https://github.com/unslothai/unsloth/issues/4188
# Qwen3.5 has a 248,320-token vocabulary (1.64x larger than Qwen3).
# At 8K context the full logits tensor is 8192 x 248320 x 4 bytes = 7.68 GB,
# which exceeds free VRAM on T4/P100 after model load.
#
# Root cause: loader.py listed "qwen3_5" in FORCE_FLOAT32 but never dispatched
# it to an optimised class, so the model fell through to a bare HF load with no
# fast-forward patching and full logits were materialised every training step.
#
# Fix: patch Qwen3_5ForConditionalGeneration.forward (the class HF uses for all
# Qwen3.5 text models, including base variants) to call unsloth_fused_ce_loss
# directly from hidden_states, bypassing logits materialisation entirely.
#
# Gated DeltaNet (GDN) linear-attention layers are intentionally NOT patched --
# they already have Triton kernels via flash-linear-attention and are
# architecturally incompatible with Unsloth's standard attention optimisations.

from .llama import *
import os
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.hf_utils import dtype_from_config
from .llama import FastLlamaModel

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5CausalLMOutputWithPast,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    raise ImportError(
        "Unsloth: Your transformers version does not support Qwen3.5.\n"
        'Try `pip install --upgrade "transformers>=5.0.0"`\n'
        "then restart your session."
    )


def _qwen3_5_compute_loss_or_logits(
    self, hidden_states, labels, logits_to_keep, vocab_size, **kwargs
):
    """
    Shared helper: given hidden_states from the backbone, return (loss, logits).

    Exactly one of loss/logits will be the primary result:
    - Single-token decode  -> logits via fast torch.mv
    - Partial-logits path  -> logits for the last logits_to_keep tokens
    - Training with labels -> loss via unsloth_fused_ce_loss (no logits materialised)
    - Eval / inference     -> full logits, then optional loss via self.loss_function

    Returns:
        loss   (Tensor or None)
        logits (Tensor or EMPTY_LOGITS)
    """
    lm_head_weight = self.lm_head.weight
    hidden_states = hidden_states.to(lm_head_weight.device)
    bsz, q_len, _ = hidden_states.shape
    out_dtype = _get_dtype(dtype_from_config(self.config))

    # Fast single-token decode (inference / generation)
    if bsz == 1 and q_len == 1 and labels is None:
        logits = torch.mv(
            lm_head_weight, hidden_states.ravel().to(lm_head_weight.dtype)
        )
        logits = logits.unsqueeze(0).unsqueeze(0).to(out_dtype)
        return None, logits

    # Partial-logits path (e.g. logits_to_keep for speculative decoding)
    if logits_to_keep != 0:
        slice_idx = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :].to(lm_head_weight.dtype))
        return None, logits.to(out_dtype)

    # Training path: fused CE avoids materialising the 7.68 GB logits tensor.
    #
    # Note: llama.py skips fused CE for bsz * q_len <= 1024, since for short
    # sequences the savings are marginal. We unconditionally use fused CE for
    # Qwen3.5 -- even a 32-token sequence produces a 32 x 248320 x 4 = 31 MB
    # logit tensor, and the chunked CE overhead is negligible vs the OOM risk.
    if labels is not None and os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1":
        labels = labels.to(lm_head_weight.device)
        n_items = kwargs.get("num_items_in_batch")
        if n_items is None:
            n_items = kwargs.get("n_items")
        loss = unsloth_fused_ce_loss(
            trainer = None,
            hidden_states = hidden_states,
            lm_head_weight = lm_head_weight,
            lm_head_bias = None,
            labels = labels,
            mask = None,
            n_items = n_items,
            scaling = getattr(self, "accelerator_scaler", None),
            target_gb = None,
            torch_compile = True,
            logit_softcapping = 0,  # Qwen3.5 has no logit softcapping
        )
        return loss, EMPTY_LOGITS

    # Eval / inference path
    logits = self.lm_head(hidden_states.to(lm_head_weight.dtype)).to(out_dtype)
    loss = None
    if labels is not None:
        labels = labels.to(lm_head_weight.device)
        loss = self.loss_function(
            logits = logits, labels = labels, vocab_size = vocab_size, **kwargs
        )
    return loss, logits


def Qwen3_5ForConditionalGeneration_fast_forward(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    pixel_values = None,
    pixel_values_videos = None,
    image_grid_thw = None,
    video_grid_thw = None,
    mm_token_type_ids = None,
    cache_position = None,
    logits_to_keep = 0,
    num_logits_to_keep = 0,
    return_dict = None,
    **kwargs,
):
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    # Normalise both generation knobs
    logits_to_keep = max(logits_to_keep, num_logits_to_keep)

    outputs = self.model(
        input_ids = input_ids,
        pixel_values = pixel_values,
        pixel_values_videos = pixel_values_videos,
        image_grid_thw = image_grid_thw,
        video_grid_thw = video_grid_thw,
        position_ids = position_ids,
        attention_mask = attention_mask,
        past_key_values = past_key_values,
        inputs_embeds = inputs_embeds,
        cache_position = cache_position,
        mm_token_type_ids = mm_token_type_ids,
        return_dict = return_dict,
        **kwargs,
    )

    # Return hidden states as logits when requested
    if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
        hidden_states = outputs[0]
        if logits_to_keep != 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        if not return_dict:
            return (hidden_states,) + outputs[1:]
        return Qwen3_5CausalLMOutputWithPast(
            loss = None,
            logits = hidden_states,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
            rope_deltas = getattr(outputs, "rope_deltas", None),
        )

    loss, logits = _qwen3_5_compute_loss_or_logits(
        self,
        outputs[0],
        labels,
        logits_to_keep,
        vocab_size = self.config.text_config.vocab_size,
        **kwargs,
    )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Qwen3_5CausalLMOutputWithPast(
        loss = loss,
        logits = logits,
        past_key_values = outputs.past_key_values,
        hidden_states = outputs.hidden_states,
        attentions = outputs.attentions,
        rope_deltas = getattr(outputs, "rope_deltas", None),
    )


def Qwen3_5ForCausalLM_fast_forward(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    cache_position = None,
    logits_to_keep = 0,
    num_logits_to_keep = 0,
    return_dict = None,
    **kwargs,
):
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    # Normalise both generation knobs
    logits_to_keep = max(logits_to_keep, num_logits_to_keep)

    outputs = self.model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        position_ids = position_ids,
        past_key_values = past_key_values,
        inputs_embeds = inputs_embeds,
        use_cache = use_cache,
        cache_position = cache_position,
        return_dict = return_dict,
        **kwargs,
    )

    # Return hidden states as logits when requested
    if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
        hidden_states = outputs[0]
        if logits_to_keep != 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        if not return_dict:
            return (hidden_states,) + outputs[1:]
        return CausalLMOutputWithPast(
            loss = None,
            logits = hidden_states,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )

    loss, logits = _qwen3_5_compute_loss_or_logits(
        self,
        outputs[0],
        labels,
        logits_to_keep,
        vocab_size = self.config.vocab_size,
        **kwargs,
    )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss = loss,
        logits = logits,
        past_key_values = outputs.past_key_values,
        hidden_states = outputs.hidden_states,
        attentions = outputs.attentions,
    )


class FastQwen3_5Model(FastLlamaModel):
    """
    Unsloth optimisation for Qwen3.5 hybrid GDN (Gated DeltaNet) models.

    Qwen3.5 interleaves standard transformer attention layers with Gated
    DeltaNet linear-attention layers.  GDN layers use native Triton kernels
    from flash-linear-attention and are architecturally incompatible with
    Unsloth's standard attention patches (gated query projections, different
    forward signatures).  This class therefore only patches the top-level
    CausalLM forward to call unsloth_fused_ce_loss directly from
    hidden_states, which eliminates the 7.68 GB logits tensor that causes
    OOM on T4/P100 at 8K context.

    Memory saving at batch=1, seq=8192:
        Standard:          8192 x 248320 x 4 = 7.68 GB  (OOM on T4)
        unsloth_fused_ce:  chunked, ~0.24-0.95 GB peak   (fits)

    Fixes: https://github.com/unslothai/unsloth/issues/4188
    """

    @staticmethod
    def pre_patch():
        Qwen3_5ForConditionalGeneration.forward = (
            Qwen3_5ForConditionalGeneration_fast_forward
        )
        Qwen3_5ForCausalLM.forward = Qwen3_5ForCausalLM_fast_forward
        return

    @staticmethod
    def from_pretrained(
        model_name = "Qwen/Qwen3.5-9B",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = token,
            device_map = device_map,
            rope_scaling = rope_scaling,
            fix_tokenizer = fix_tokenizer,
            model_patcher = FastQwen3_5Model,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
