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

import torch
import torch.nn as nn
import inspect
from typing import List, Optional, Tuple, Union
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING


def patch_SmolVLMForConditionalGeneration_forward():
    try:
        import transformers.models.smolvlm.modeling_smolvlm
    except:
        return

    from typing import List, Optional, Tuple, Union

    from transformers.models.smolvlm.modeling_smolvlm import (
        SmolVLMCausalLMOutputWithPast,
    )

    # helps normalize text sensitive to spaces, tabs and newlines to allow proper comparison
    def normalize_text(text: str) -> str:
        """Ultra-compact code text normalizer."""
        import re
        return re.sub(r'\s*([=+\-*/%&|^<>!(),{}\[\]:])\s*', r'\1', re.sub(r'\s+', ' ', re.sub(r'#.*?$|/\*.*?\*/', '', re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL), flags=re.MULTILINE)).strip())

    # Newest transformers update now fixes this issue by assigning loss to self.loss_function, which defaults to a LossForCausalLM that implements a fixed
    # CrossEntropyLoss in transformers.loss.loss_utils.py. Once transformers pypi releases the main repo, we can completely remove this patch.
    current_forward_source = inspect.getsource(
        transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration.forward
    )
    if normalize_text("loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)") in normalize_text(current_forward_source):
        return  # Already patched

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
            loss_fct = transformers.models.smolvlm.modeling_smolvlm.CrossEntropyLoss()
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
        if UNSLOTH_ENABLE_LOGGING:
            print(
                "Unsloth: Failed to patch SmolVLMForConditionalGeneration forward function."
            )
        pass
    else:
        transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration.forward = (
            forward
        )
        pass
    return
pass
TEMPORARY_PATCHES.append(patch_SmolVLMForConditionalGeneration_forward)


def patch_CsmBackboneModelEmbeddings_forward():
    try:
        import transformers.models.csm.modeling_csm
    except:
        return


    def forward(self, input_ids):
        input_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        # fix for dtype cast
        dtype = input_embeds.dtype
        input_embeds = input_embeds.sum(dim=2).to(dtype)
        return input_embeds

    old_keys = inspect.signature(
        transformers.models.csm.modeling_csm.CsmBackboneModelEmbeddings.forward
    ).parameters
    new_keys = inspect.signature(forward).parameters

    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch CsmBackboneModelEmbeddings forward.")
    else:
        transformers.models.csm.modeling_csm.CsmBackboneModelEmbeddings.forward = forward
pass
TEMPORARY_PATCHES.append(patch_CsmBackboneModelEmbeddings_forward)


def patch_CsmDepthDecoderForCausalLM_forward():
    try:
        import transformers.models.csm.modeling_csm
    except:
        return

    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.csm.modeling_csm import Cache, Unpack, KwargsForCausalLM
    from transformers.loss.loss_utils import ForCausalLMLoss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=backbone_last_hidden_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.codebooks_head(
            hidden_states[:, slice_indices, :], cache_position[slice_indices] if cache_position is not None else None
        )
        logits = logits.contiguous()

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = ForCausalLMLoss
            loss = loss_fct(
                logits=logits, labels=None, vocab_size=self.config.vocab_size, shift_labels=shift_labels, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    pass

    old_keys = inspect.signature(
        transformers.models.csm.modeling_csm.CsmDepthDecoderForCausalLM.forward
    ).parameters
    new_keys = inspect.signature(forward).parameters

    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch CsmDepthDecoderForCausalLM forward.")
    else:
        transformers.models.csm.modeling_csm.CsmDepthDecoderForCausalLM.forward = forward
pass
TEMPORARY_PATCHES.append(patch_CsmDepthDecoderForCausalLM_forward)


def patch_CsmForConditionalGeneration_forward():
    try:
        import transformers.models.csm.modeling_csm
    except:
        return

    from transformers.models.csm.modeling_csm import Cache, Unpack, KwargsForCausalLM, CsmOutputWithPast
    from transformers.loss.loss_utils import ForCausalLMLoss
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CsmOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and input_ids.ndim == 2:
            merged_inputs = self._merge_input_ids_with_input_values(
                input_ids, input_values, input_values_cutoffs, labels
            )
            inputs_embeds = merged_inputs["inputs_embeds"]
            labels = merged_inputs["labels"]
            input_ids = None

        backbone_outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        backbone_hidden_states = backbone_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            # select first codebook as labels for the backbone model
            backbone_labels = labels[:, :, 0]
            backbone_loss = self.loss_function(
                logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs
            )

            # for the depth decoder, we need to select the frames to train on
            # those are frames where the label is not uniformly `ignore_index` along the codebook dimension
            train_mask = ~(labels[:, :, 1:] == -100).all(dim=-1)
            depth_decoder_input_ids = labels[train_mask][..., : self.config.num_codebooks - 1]
            # add place holder in position 0 that will be replaced by the backbone_last_hidden_state
            depth_decoder_input_ids = torch.nn.functional.pad(depth_decoder_input_ids, (1, 0), value=0)

            train_idxs = train_mask.nonzero(as_tuple=True)
            backbone_last_hidden_states = backbone_hidden_states[train_idxs[0], train_idxs[1] - 1, :]
            depth_decoder_labels = labels[train_mask]

            # Fix: explicitly pass kwargs to depth decoder to get access to num_items_in_batch
            depth_decoder_kwargs = kwargs.copy()
            # backbone loss num_items is based on the 0th codebooks index
            # while depth loss num_items is based on the the remaining 31 codebooks
            # therefore num_items_in_batch should be multiplied by 31
            if 'num_items_in_batch' in depth_decoder_kwargs:
                depth_decoder_kwargs['num_items_in_batch'] = depth_decoder_kwargs['num_items_in_batch'] * 31

            # make sure return_dict is set to True
            depth_decoder_kwargs.pop('return_dict', None)

            depth_decoder_outputs = self.depth_decoder(
                input_ids=depth_decoder_input_ids,
                backbone_last_hidden_state=backbone_last_hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                labels=depth_decoder_labels,
                # Fix: explicitly pass kwargs to depth decoder to get access to num_items_in_batch
                **depth_decoder_kwargs,
            )

            depth_decoder_loss = depth_decoder_outputs.loss
            loss = backbone_loss + depth_decoder_loss

        return CsmOutputWithPast(
            loss=loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_decoder_loss,
            logits=backbone_logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            depth_decoder_logits=depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            depth_decoder_past_key_values=depth_decoder_outputs.past_key_values
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_hidden_states=depth_decoder_outputs.hidden_states
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_attentions=depth_decoder_outputs.attentions if depth_decoder_outputs is not None else None,
        )
    pass

    old_keys = inspect.signature(
        transformers.models.csm.modeling_csm.CsmForConditionalGeneration.forward
    ).parameters
    new_keys = inspect.signature(forward).parameters

    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch CsmForConditionalGeneration forward.")
    else:
        transformers.models.csm.modeling_csm.CsmForConditionalGeneration.forward = forward
pass
TEMPORARY_PATCHES.append(patch_CsmForConditionalGeneration_forward)


def patch_CsmForConditionalGeneration_merge():

    try:
        import transformers.models.csm.modeling_csm
    except:
        return


    def _merge_input_ids_with_input_values(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Merges the input_ids and input_values to produce a single inputs_embeds tensor:
        1 - Infers the codec model on the input_values to retreive codebook token.
        2 - Embeds codebook tokens and places them at the correct positions in the inputs_embeds tensor.
        3 - If labels are provided, expands them to match codebook dimensions and position the target codebook tokens in the inputs_embeds tensor.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input ids to embed.
            input_values (`torch.Tensor` of shape `(batch_size, channels, audio_sequence_length)`):
                The audio input values to embed.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`):
                The cutoffs of the audio input values relative to its batch index, padded with -1 when no audio.
        """
        inputs_embeds = self.embed_text_tokens(input_ids)

        if input_values is not None:
            # infer input_values_mask
            input_values_cutoffs = torch.nn.functional.pad(input_values_cutoffs, (1, 0))
            audio_lengths = input_values_cutoffs[input_values_cutoffs >= 0].diff()
            audio_lengths = audio_lengths[audio_lengths > 0]
            input_values_mask = torch.arange(input_values_cutoffs.max(), device=input_values.device).expand(
                len(audio_lengths), -1
            )
            input_values_mask = input_values_mask < audio_lengths.unsqueeze(1)

            # =======================================
            # TODO: @eustlb, this should be batched !!!
            # but requires making sure batched inference of the codec model works as intended
            audio_tokens_list = []
            for batch_input_values, batch_input_values_cutoffs in zip(input_values, input_values_cutoffs):
                batch_input_values_cutoffs = batch_input_values_cutoffs[batch_input_values_cutoffs >= 0]
                for i in range(batch_input_values_cutoffs.shape[0] - 1):
                    start_idx = batch_input_values_cutoffs[i]
                    end_idx = batch_input_values_cutoffs[i + 1]
                    audio_batch = batch_input_values[..., start_idx:end_idx]
                    codec_outputs = self.codec_model.encode(audio_batch.unsqueeze(0))
                    codebook_ids = codec_outputs.audio_codes.transpose(1, -1)
                    audio_tokens_list.append(codebook_ids[0])

            max_audio_frames = max(el.shape[0] for el in audio_tokens_list)
            batched_audio_token_ids = torch.stack(
                [torch.nn.functional.pad(el, (0, 0, 0, max_audio_frames - el.shape[0])) for el in audio_tokens_list]
            )
            audio_codes_mask = self.codec_model.get_audio_codes_mask(input_values_mask)
            # =======================================
            audio_token_id = self.config.audio_token_id
            audio_token_mask = input_ids == audio_token_id

            audio_embeds = self.backbone_model.embed_tokens(batched_audio_token_ids)
            inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask]

            # same for the audio eos token
            audio_eos_frame_ids = (
                torch.ones((1, 1, self.config.num_codebooks), device=input_ids.device, dtype=torch.long)
                * self.config.codebook_eos_token_id
            )
            audio_eos_embeds = self.backbone_model.embed_tokens(audio_eos_frame_ids).squeeze(1)

            audio_eos_token_mask = input_ids == self.config.audio_eos_token_id
            inputs_embeds[audio_eos_token_mask] = audio_eos_embeds.repeat(audio_eos_token_mask.sum(), 1)

            # if the labels are provided, we need to expand the labels to (batch_size, seq_length, num_codebooks)
            if labels is not None:
                labels_expanded = labels.unsqueeze(-1).repeat(1, 1, self.config.num_codebooks)
                labels_expanded[audio_token_mask] = batched_audio_token_ids[audio_codes_mask]
                # fix make sure to set eos_token_id as a valid label to predict
                labels_expanded[audio_eos_token_mask] = self.config.codebook_eos_token_id
                # mask depth decoder
                depth_decoder_ignore_frames_idxs = (labels == -101).nonzero(as_tuple=True)
                labels_expanded[depth_decoder_ignore_frames_idxs[0], depth_decoder_ignore_frames_idxs[1], 1:] = -100
                labels = labels_expanded

        return {"inputs_embeds": inputs_embeds, "labels": labels}
    pass
    old_keys = inspect.signature(
        transformers.models.csm.modeling_csm.CsmForConditionalGeneration._merge_input_ids_with_input_values
    ).parameters
    new_keys = inspect.signature(_merge_input_ids_with_input_values).parameters

    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch CsmForConditionalGeneration _merge_input_ids_with_input_values.")
    else:
        transformers.models.csm.modeling_csm.CsmForConditionalGeneration._merge_input_ids_with_input_values = _merge_input_ids_with_input_values
pass
TEMPORARY_PATCHES.append(patch_CsmForConditionalGeneration_merge)
