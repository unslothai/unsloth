# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re
from typing import Union, List, Any, Tuple, Dict, Callable, Optional
import inspect
import torch
import os

UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : UNSLOTH_COMPILE_MAXIMUM,
    "shape_padding"     : True,
    "trace.enabled"     : UNSLOTH_COMPILE_DEBUG,
    "triton.cudagraphs" : False,
}

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []

def patch_Gemma3Processor():
    try:
        import transformers.models.gemma3.processing_gemma3
    except:
        return
    from transformers.models.gemma3.processing_gemma3 import (
        ImageInput,
        PreTokenizedInput,
        Unpack,
        Gemma3ProcessorKwargs,
        make_nested_list_of_images,
        TextInput,
        BatchFeature,
        to_py_obj,
    )
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        batched_images = None
        if images is not None:
            try:
                batched_images = make_nested_list_of_images(images)
            except ValueError as e:
                # Maybe it's texts and not images? Gemma3 defaults to images
                if text is None:
                    text = images
                    images = None
                else:
                    raise ValueError(e)
        pass
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            # batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            batch_num_crops = to_py_obj(image_inputs.pop("num_crops"))
            text_with_crops = text
            for batch_idx, (prompt, images, num_crops) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text_with_crops[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        # text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"], return_tensors="np")

        # Fix double BOS tokens
        bos = self.tokenizer.bos_token
        n = len(bos)
        text = [x[i + n:] if (i := x.find(bos)) != -1 else x for x in text]

        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        # [TODO] FAILS for batched tokens since text_inputs["input_ids"] is a list of lists, so np.array creates an object!
        input_ids = text_inputs["input_ids"]
        image_token_id = self.image_token_id
        mm_token_type_ids = [[1 if y == image_token_id else 0 for y in x] for x in input_ids]
        # array_ids = np.array(text_inputs["input_ids"])
        # mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
        # mm_token_type_ids[array_ids == self.image_token_id] = 1
        # text_inputs = {k: v.tolist() for k, v in text_inputs.items()}  # in case user requested list inputs
        text_inputs["token_type_ids"] = mm_token_type_ids#.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
    pass
    old_keys = inspect.signature(transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__).parameters
    new_keys = inspect.signature(__call__).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3Processor.")
    else:
        transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__ = __call__
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3Processor)


def patch_Gemma3ForConditionalGeneration():
    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    from transformers.models.gemma3.modeling_gemma3 import (
        HybridCache,
        Gemma3CausalLMOutputWithPast,
        logger,
        is_torchdynamo_compiling,
        Cache,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
        )
        if labels is not None and attention_mask is not None:
            attention_mask = attention_mask.to(device = labels.device)
            labels[attention_mask == 0] = -100
        pass
        outputs = self.language_model(
            labels=labels,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        labels = None


        logits = outputs.logits
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        loss = outputs.loss

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3ForConditionalGeneration.")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration)


def patch_Gemma3ForConditionalGeneration_causal_mask():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try: import transformers.models.gemma3.modeling_gemma3
    except: return
    from transformers.models.gemma3.modeling_gemma3 import (
        StaticCache,
        HybridCache,
    )
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            return attention_mask

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            return attention_mask

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(torch.float16).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=torch.float16, device=cache_position.device
        )

        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Apply bidirectional mask on images if token type ids are provided
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            token_type_mask[token_type_ids == 0] = False  # if text token do not change anything
            token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
                token_type_mask, 0.0
            )

        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration._update_causal_mask).parameters
    new_keys = inspect.signature(_update_causal_mask).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3ForConditionalGeneration.")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration._update_causal_mask = _update_causal_mask
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration_causal_mask)


def patch_Gemma3TextScaledWordEmbedding():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try: import transformers.models.gemma3.modeling_gemma3
    except: return
    def forward(self, input_ids: torch.Tensor):
        input_embeds = torch.nn.functional.embedding(
            input_ids,
            weight = self.weight,
            padding_idx = self.padding_idx,
        )
        return input_embeds.to(torch.float32) * self.embed_scale
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3TextScaledWordEmbedding.")
    else:
        forward = torch.compile(forward, fullgraph = True, dynamic = True, options = torch_compile_options)
        transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding.forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3TextScaledWordEmbedding)


def patch_Gemma3RMSNorm():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try: import transformers.models.gemma3.modeling_gemma3
    except: return
    def forward(self, x):
        x = x.to(torch.float32)
        output = x * torch.rsqrt(x.square().mean(-1, keepdim = True) + self.eps)
        return output * (1.0 + self.weight.float())
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3RMSNorm.")
    else:
        forward = torch.compile(forward, fullgraph = True, dynamic = True, options = torch_compile_options)
        transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm.forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3RMSNorm)


def patch_Gemma3MLP():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try: import transformers.models.gemma3.modeling_gemma3
    except: return
    def forward(self, x):
        x = x.to(torch.float16)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj.to(torch.float32)
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3MLP.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3MLP.")
    else:
        forward = torch.compile(forward, fullgraph = False, dynamic = True, options = torch_compile_options)
        transformers.models.gemma3.modeling_gemma3.Gemma3MLP.forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3MLP)


def patch_Gemma3Attention():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        downcast_dtype = torch.float16
    else:
        downcast_dtype = torch.bfloat16
    try: import transformers.models.gemma3.modeling_gemma3
    except: return
    from transformers.models.gemma3.modeling_gemma3 import (
        Cache,
        Unpack,
        FlashAttentionKwargs,
        apply_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
        logger,
        eager_attention_forward,
    )
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = hidden_states.to(downcast_dtype)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
        #             "Falling back to eager attention. This warning can be removed using the argument "
        #             '`attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states.to(downcast_dtype),
        #     key_states.to(downcast_dtype),
        #     value_states.to(downcast_dtype),
        #     attention_mask.to(downcast_dtype),
        #     dropout=self.attention_dropout if self.training else 0.0,
        #     scaling=self.scaling,
        #     sliding_window=self.sliding_window,
        #     **kwargs,
        # )
        attn_output = scaled_dot_product_attention(
            query_states.to(downcast_dtype),
            key_states.to(downcast_dtype),
            value_states.to(downcast_dtype),
            attn_mask=attention_mask.to(downcast_dtype),
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            enable_gqa=getattr(self, "num_key_value_groups", 1) != 1,
        ).transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1)#.contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    pass
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3Attention.")
    else:
        forward = torch.compiler.disable(forward, recursive = False)
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention)

def patch_SmolVLMForConditionalGeneration_forward():
    try:
        import transformers.models.smolvlm.modeling_smolvlm
    except:
        return

    from typing import List, Optional, Tuple, Union

    from transformers.models.smolvlm.modeling_smolvlm import (
        CrossEntropyLoss,
        SmolVLMCausalLMOutputWithPast,
        SmolVLMForConditionalGeneration,
    )

    # Check if the fix is already present (either from Transformers library or previous Unsloth patch)
    # We look for the specific device handling code that fixes torch.compile indentation errors,
    # rather than an Unsloth marker, to handle cases where the fix comes from upstream
    current_forward_source = inspect.getsource(
        transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration.forward
    )
    if "shift_labels.view(-1).to(shift_logits.device)" in current_forward_source:
        return  # Already patched

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, SmolVLMCausalLMOutputWithPast]:
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask != 0
                ].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(
                    shift_logits.device
                ),  # The fix is here - explicit device conversion
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SmolVLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    # Check if we can patch the model by comparing signatures
    old_keys = inspect.signature(
        transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration.forward
    ).parameters
    new_keys = inspect.signature(forward).parameters

    if old_keys != new_keys:
        print(
            "Unsloth: Failed to patch SmolVLMForConditionalGeneration forward function."
        )
    else:
        transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration.forward = (
            forward
        )
        print(
            "Unsloth: Successfully patched SmolVLMForConditionalGeneration for better torch.compile compatibility."
        )

    return


# Add the patch to the TEMPORARY_PATCHES list
TEMPORARY_PATCHES.append(patch_SmolVLMForConditionalGeneration_forward)
