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
from typing import Union, List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
import os
import logging

from .common import TEMPORARY_PATCHES, torch_compile_options, UNSLOTH_ENABLE_LOGGING

logger = logging.getLogger(__name__)


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
            for batch_idx, (prompt, images_for_item, num_crops_for_item) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images_for_item) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images_for_item)} images."
                    )

                iterable_num_crops = num_crops_for_item

                if isinstance(num_crops_for_item, int):
                        if len(image_indexes) > 0:
                            iterable_num_crops = [num_crops_for_item] + [0] * (len(image_indexes) - 1)
                        else:
                            iterable_num_crops = []

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(iterable_num_crops, image_indexes))):
                    if isinstance(num, int) and num > 0:
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
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch Gemma3Processor.")
    else:
        transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__ = __call__
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3Processor)

def patch_Gemma3ForConditionalGeneration_forward_router():
    try:
            import transformers.models.gemma3.modeling_gemma3
    except:
            return
    from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3CausalLMOutputWithPast,
            Cache,
        )

    def forward_router(
         self,
         input_ids: torch.LongTensor = None,
         pixel_values: torch.FloatTensor = None,
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
         return_dict: Optional[bool] = None,
         logits_to_keep: Union[int, torch.Tensor] = 0,
         **lm_kwargs,
                       ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        # Routing logic
        is_text_only = (
            pixel_values is None and
            token_type_ids is None and
            (input_ids is not None or inputs_embeds is not None)
        )

        if is_text_only:
            return self.forward_llm(
                input_ids,
                pixel_values,
                attention_mask,
                position_ids,
                past_key_values,
                token_type_ids,
                cache_position,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                logits_to_keep,
                **lm_kwargs)
        else:
            return self.forward_multimodal(
                input_ids,
                pixel_values,
                attention_mask,
                position_ids,
                past_key_values,
                token_type_ids,
                cache_position,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                logits_to_keep,
                **lm_kwargs)

    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward).parameters
    new_keys = inspect.signature(forward_router).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed patching Gemma3ForConditionalGeneration")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward = forward_router
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration_forward_router)


def patch_Gemma3ForConditionalGeneration_forward_multimodal():
    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    from transformers.models.gemma3.modeling_gemma3 import (
            HybridCache,
            Gemma3CausalLMOutputWithPast,
            BaseModelOutputWithPast,
            CausalLMOutputWithPast,
            logger,
            is_torchdynamo_compiling,
            Cache,
        )

    def forward_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
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
        return_dict: Optional[bool] = None,
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
          #if input_ids is not None and self.config.image_token_index >= self.vocab_size:
          if input_ids is not None and self.config.image_token_index >= self.config.text_config.vocab_size:
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
              #image_features = self.get_image_features(pixel_values)
              image_features = self.model.get_image_features(pixel_values)

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
          if labels is not None and self.model.pad_token_id in labels:
              logger.warning_once(
                  "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                  "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
              )
              labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

          causal_mask = self.model._update_causal_mask(
              attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
          )
          if labels is not None and attention_mask is not None:
              attention_mask = attention_mask.to(device = labels.device)
              labels[attention_mask == 0] = -100
          pass
          outputs = self.model(
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
          hidden_states = outputs[0]
          # Ensure dtype compatibility with lm_head
          # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
          slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep


          # Check if we're on a float16 machine with forced float32
          if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
              # Compute in float32 to avoid overflow, then convert back
              hidden_states_slice = hidden_states[:, slice_indices, :].to(torch.float32)
              lm_head_weight = self.lm_head.weight.to(torch.float32)
              logits = torch.nn.functional.linear(hidden_states_slice, lm_head_weight, self.lm_head.bias)
              if labels is None:
                  # If no loss computation, convert back to original dtype
                  logits = logits.to(self.lm_head.weight.dtype)
          else:
              # Normal path - ensure dtype compatibility
              hidden_states = hidden_states.to(self.lm_head.weight.dtype)
              logits = self.lm_head(hidden_states[:, slice_indices, :])

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
    new_keys = inspect.signature(forward_multimodal).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed patching Gemma3ForConditionalGeneration Multimodal Forward")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward_multimodal = forward_multimodal
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration_forward_multimodal)


def patch_Gemma3ForConditionalGeneration_forward_llm():
     try:
        import transformers.models.gemma3.modeling_gemma3
     except:
        return
     from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3CausalLMOutputWithPast,
        Cache,
    )

     def forward_llm(
         self,
         input_ids: torch.LongTensor = None,
         pixel_values: torch.FloatTensor = None,
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
         return_dict: Optional[bool] = None,
         logits_to_keep: Union[int, torch.Tensor] = 0,
         **lm_kwargs,
     ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # Direct route through language_model
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
              # Compute in float32 to avoid overflow, then convert back
              hidden_states_slice = hidden_states[:, slice_indices, :].to(torch.float32)
              lm_head_weight = self.lm_head.weight.to(torch.float32)
              logits = torch.nn.functional.linear(hidden_states_slice, lm_head_weight, self.lm_head.bias)
              if labels is None:
                  # If no loss computation, convert back to original dtype
                  logits = logits.to(self.lm_head.weight.dtype)
        else:
              # Normal path - ensure dtype compatibility
              hidden_states = hidden_states.to(self.lm_head.weight.dtype)
              logits = self.lm_head(hidden_states[:, slice_indices, :])


        # Apply softcapping if configured
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **lm_kwargs)

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=None,  # No images in text-only mode
        )
     pass

     old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward).parameters
     new_keys = inspect.signature(forward_llm).parameters
     if old_keys != new_keys:
         if UNSLOTH_ENABLE_LOGGING:
             print("Unsloth: Failed patching Gemma3ForConditionalGeneration language forward")
     else:
         transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward_llm = forward_llm
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration_forward_llm)


def patch_Gemma3ForCausalLM_forward_router():
    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    from transformers.models.gemma3.modeling_gemma3 import (
        CausalLMOutputWithPast,
        HybridCache
    )

    def forward_router(
         self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:

        # Routing logic
        is_grpo_training = (
            os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1" and
            self.training and
            labels is None
        )


        if is_grpo_training:
            return self.grpo_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                cache_position,
                logits_to_keep,
                **loss_kwargs
                )
        else:
            return self.original_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                cache_position,
                logits_to_keep,
                **loss_kwargs
                )

    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.forward).parameters
    new_keys = inspect.signature(forward_router).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed patching Gemma3ForCausalLM")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.forward = forward_router
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForCausalLM_forward_router)


def patch_Gemma3ForCausalLM():
    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    from transformers.models.gemma3.modeling_gemma3 import (
        CausalLMOutputWithPast,
        logger,
        HybridCache
    )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            # Compute in float32 to avoid overflow, then convert back
            hidden_states_slice = hidden_states[:, slice_indices, :].to(torch.float32)
            lm_head_weight = self.lm_head.weight.to(torch.float32)
            lm_head_bias = self.lm_head.bias.to(torch.float32) if self.lm_head.bias is not None else None
            logits = torch.nn.functional.linear(hidden_states_slice, lm_head_weight, lm_head_bias)
            if labels is None:
                # If no loss computation, convert back to original dtype
                logits = logits.to(self.lm_head.weight.dtype)
        else:
            # ensure dtype compatibility
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Apply final logit softcapping if configured
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3ForCausalLM  compatibility.")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.original_forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForCausalLM)


def patch_Gemma3ForCausalLMGRPO():
    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    from transformers.models.gemma3.modeling_gemma3 import (
        CausalLMOutputWithPast,
        logger,
        HybridCache
    )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Handle logits_to_keep parameter first
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        # Convert hidden states to logits using lm_head
        # Handle UNSLOTH_FORCE_FLOAT32 for the projection
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            hidden_states_fp32 = hidden_states.to(torch.float32)
            lm_head_weight = self.lm_head.weight.to(torch.float32)
            lm_head_bias = self.lm_head.bias.to(torch.float32) if self.lm_head.bias is not None else None
            logits = torch.nn.functional.linear(hidden_states_fp32, lm_head_weight, lm_head_bias)
            # Keep in float32 for GRPO
        else:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)
            logits = self.lm_head(hidden_states)
            # Convert to float32 for GRPO compatibility
            logits = logits.to(torch.float32)

        # Apply final logit softcapping if configured
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # GRPO expects logits without the last position (next token prediction)
        # Only remove last position if we have more than 1 position
        if logits.dim() == 3 and logits.shape[1] > 1:  # (batch, seq_len, vocab_size)
            # Remove last position for GRPO compatibility
            logits = logits[:, :-1, :]

            # Ensure contiguous memory layout for torch.compile compatibility
            logits = logits.contiguous()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,  # Return actual logits for GRPO
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Apply the patch
    old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        print("Unsloth: Failed to patch Gemma3ForCausalLM for GRPO compatibility.")
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM.grpo_forward = forward
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForCausalLMGRPO)

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

    if hasattr(transformers.models.gemma3.modeling_gemma3, "Gemma3Model"):
        old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3Model._update_causal_mask).parameters
        new_keys = inspect.signature(_update_causal_mask).parameters
        if old_keys != new_keys:
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Failed to patch Gemma3Model.")
        else:
            transformers.models.gemma3.modeling_gemma3.Gemma3Model._update_causal_mask = _update_causal_mask
    else:
        old_keys = inspect.signature(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration._update_causal_mask).parameters
        new_keys = inspect.signature(_update_causal_mask).parameters
        if old_keys != new_keys:
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Failed to patch Gemma3ForConditionalGeneration._update_causal_mask.")
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
        if UNSLOTH_ENABLE_LOGGING:
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

    original_rmsnorm_forward = transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm.forward

    def forward(self, x): # x can be fp32 (from embeddings) or fp16 (from MLP/Attn)
        # Internals in fp32
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)

        # self.weight is bf16 (from vision.py loading if UNSLOTH_FORCE_FLOAT32="1")
        # So, cast self.weight to fp32 for the (1.0 + weight) operation
        output_fp32 = hidden_states_fp32 * (1.0 + self.weight.to(torch.float32))

        # Clamp to fp16 range before casting back to fp16
        fp16_max = torch.finfo(torch.float16).max
        fp16_min = torch.finfo(torch.float16).min
        clamped_output_fp32 = torch.clamp(output_fp32, min=fp16_min, max=fp16_max)

        return clamped_output_fp32.to(torch.float16) # Output fp16
    pass
    old_keys = inspect.signature(original_rmsnorm_forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch Gemma3RMSNorm (adjusted). Signature mismatch.")
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

    original_mlp_forward = transformers.models.gemma3.modeling_gemma3.Gemma3MLP.forward

    def forward(self, x): # x is fp16 from RMSNorm
        gate_proj_out = self.gate_proj(x)
        up_proj_out = self.up_proj(x)

        #  Upcast to fp32
        gate_proj_fp32 = gate_proj_out.to(torch.float32)
        up_proj_fp32 = up_proj_out.to(torch.float32)
        activated_fp32 = self.act_fn(gate_proj_fp32) # Activation in fp32
        intermediate_fp32 = activated_fp32 * up_proj_fp32 # Product in fp32

        # Downcast and down_proj
        intermediate_fp16 = intermediate_fp32.to(torch.float16)
        down_proj_out = self.down_proj(intermediate_fp16)
        return down_proj_out
    pass

    old_keys = inspect.signature(original_mlp_forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING: print("Unsloth: Failed to patch Gemma3MLP")
    else:
        forward = torch.compile(forward, fullgraph = False, dynamic = True, options = torch_compile_options)
        transformers.models.gemma3.modeling_gemma3.Gemma3MLP.forward = forward

    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3MLP)


def patch_Gemma3Attention():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return

    try:
        import transformers.models.gemma3.modeling_gemma3
    except:
        return
    import typing
    from transformers.models.gemma3.modeling_gemma3 import (
        Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,
    )

    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    original_hf_attention_forward = transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1] # For reshaping o_proj output later

        # Determine head shapes
        # Assuming these attributes are standard for Gemma3Attention
        # If not, they might come from self.config
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # For projections view: (bsz, q_len, num_specific_heads, head_dim)
        query_hidden_shape = (bsz, q_len, num_heads, head_dim)
        kv_hidden_shape    = (bsz, q_len, num_key_value_heads, head_dim)

        # 1. Projections (q, k, v) in fp16
        # hidden_states is already fp16. Weights of q_proj, k_proj, v_proj are fp16.
        query_states_fp16 = self.q_proj(hidden_states) # output fp16
        key_states_fp16 = self.k_proj(hidden_states)     # output fp16
        value_states_fp16 = self.v_proj(hidden_states)   # output fp16

        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
        query_states_fp32 = query_states_fp16.view(query_hidden_shape).to(torch.float32).transpose(1, 2)
        key_states_fp32   = key_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2)
        value_states_fp32 = value_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2) # V for attention also fp32

        # 3. Normalization (q_norm, k_norm are RMSNorms)
        query_norm_out_fp16 = self.q_norm(query_states_fp32)
        key_norm_out_fp16   = self.k_norm(key_states_fp32)

        query_states_fp32 = query_norm_out_fp16.to(torch.float32)
        key_states_fp32   = key_norm_out_fp16.to(torch.float32)

        # 4. Rotary Positional Embeddings in fp32
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            raise ValueError("Position embeddings not provided as (cos, sin) tuple to Gemma3Attention")

        cos, sin = position_embeddings
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)
        query_states_fp32, key_states_fp32 = apply_rotary_pos_emb(query_states_fp32, key_states_fp32, cos_fp32, sin_fp32)

        # 5. KV Cache update (using fp32 K, V)
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_fp32, "cos": cos_fp32, "cache_position": cache_position
            }
            # Add sliding_window if the attribute exists (common in newer models)
            if hasattr(self, "sliding_window") and self.sliding_window is not None:
                 cache_kwargs["sliding_window"] = self.sliding_window
            key_states_fp32, value_states_fp32 = past_key_value.update(
                key_states_fp32, value_states_fp32, self.layer_idx, cache_kwargs
            )

        # 6. Core Attention mechanism (SDPA) in fp32
        attn_mask_for_sdpa = attention_mask
        if attn_mask_for_sdpa is not None:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(torch.float32)

        output_attentions = kwargs.get("output_attentions", False)


        attn_output_fp32 = scaled_dot_product_attention(
            query_states_fp32,
            key_states_fp32,
            value_states_fp32,
            attn_mask=attn_mask_for_sdpa,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # is_causal=False, # Mask handles causality. If mask is None and q_len > 1, this might be true.
                               # Gemma3's _update_causal_mask provides the explicit mask.
            scale=getattr(self, "scaling", None), # Use self.scaling if defined, else SDPA default
            enable_gqa=getattr(self, "num_key_value_groups", 1) != 1,
        )
        attn_weights = None # Defaulting to None

        # 7. Reshape and Downcast for Output Projection
        # attn_output_fp32 from SDPA is (bsz, num_heads, q_len, head_dim)
        attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        # Reshape to (bsz, q_len, num_query_heads * head_dim) which is (bsz, q_len, model_hidden_size)
        # Using -1 for the last dimension is robust and aligns with your original example.
        attn_output_fp32 = attn_output_fp32.reshape(bsz, q_len, -1) # REVISED FIX

        attn_output_fp16 = attn_output_fp32.to(torch.float16)

        # 8. Output Projection (o_proj) in fp16
        attn_output_projected = self.o_proj(attn_output_fp16) # fp16 output

        return attn_output_projected, attn_weights # 3-tuple return
    pass

    old_keys_sig = inspect.signature(original_hf_attention_forward)
    new_keys_sig = inspect.signature(forward)

    if old_keys_sig.parameters != new_keys_sig.parameters or old_keys_sig.return_annotation != new_keys_sig.return_annotation:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch Gemma3Attention (adjusted for signature matching). Signature mismatch with original HF method.")
    else:
        forward = torch.compiler.disable(forward, recursive = False)
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward = forward
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Patched Gemma3Attention.forward (adjusted for signature matching, output fp16).")
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention)
