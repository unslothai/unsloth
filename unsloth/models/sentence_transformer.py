# Copyright 2025 electroglyph. All rights reserved.
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

import logging

from .loader import FastModel
from ._utils import SUPPORTS_BFLOAT16
import inspect
import json
import os
import types
from huggingface_hub import hf_hub_download
from typing import Optional
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
from ..save import unsloth_save_pretrained_torchao, unsloth_save_pretrained_gguf
import contextlib
import shutil


def _save_pretrained_torchao(
    self,
    save_directory,
    tokenizer = None,
    torchao_config = None,
    push_to_hub = False,
    token = None,
):
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

    # avoid `0_Transformer-torchao`, it was either this or fix modules.json
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

    # 2. Extract inner transformer model
    inner_model = self[0].auto_model
    if hasattr(inner_model, "_orig_mod"):
        inner_model = inner_model._orig_mod

    # If it's a PEFT model, unsloth_save_pretrained_gguf handles merging,
    # but we pass the inner model wrapper.

    # 3. Identify where the transformer weights are stored
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

    # This is where Unsloth will perform the save + conversion operations
    transformer_dir = os.path.join(save_directory, transformer_path)
    # Ensure this path is absolute for consistent comparison later
    transformer_dir = os.path.abspath(transformer_dir)

    if tokenizer is None:
        tokenizer = self.tokenizer

    # 4. Patch environment to ensure Unsloth treats this embedding model correctly
    @contextlib.contextmanager
    def patch_unsloth_gguf_save():
        # Prevent deletion of the directory we just created via self.save_pretrained
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
            push_to_hub = False, # Force local first to move files
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
            
            # Convert to absolute path to avoid mixing relative/absolute in commonpath
            abs_gguf_file = os.path.abspath(gguf_file)
            
            # Check if file is inside transformer_dir (subpath)
            try:
                is_subpath = os.path.commonpath([abs_gguf_file, transformer_dir]) == transformer_dir
            except ValueError:
                # Can happen on Windows with different drives, or mix of absolute/relative (handled by abspath above)
                is_subpath = False

            if is_subpath:
                # If the GGUF file is inside the transformer_dir, move it out to root
                shutil.move(gguf_file, dest_path)
                new_gguf_locations.append(dest_path)
            else:
                # If it's elsewhere, move it to root if not already there
                if os.path.abspath(dest_path) != abs_gguf_file:
                    shutil.move(gguf_file, dest_path)
                new_gguf_locations.append(dest_path)

    # Update result with new locations
    result["gguf_files"] = new_gguf_locations

    # 7. Add branding
    try:
        FastSentenceTransformer._add_unsloth_branding(save_directory)
        
        # Add GGUF details to README
        readme_path = os.path.join(save_directory, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "a", encoding="utf-8") as f:
                f.write("\n## GGUF Quantization\n")
                f.write(f"This model contains GGUF quantized versions in: {', '.join([os.path.basename(f) for f in new_gguf_locations])}\n")
    except:
        pass

    # 8. Handle Push to Hub if requested
    if push_to_hub:
        if token is None: 
            token = get_token()
        
        api = HfApi(token = token)
        repo_id = save_directory # Assuming save_directory is the repo name if pushing
        
        print(f"Unsloth: Uploading to {repo_id}...")
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=kwargs.get("private", False))
            api.upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                commit_message="Upload GGUF and SentenceTransformer model",
            )
            print(f"Unsloth: Uploaded to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Unsloth: Upload failed: {e}")

    return result


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

    # should prolly be done upstream instead of this hackfest here
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
                        # bog standard checkpoint
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
                    # original code from here on
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
                        # checkpoint
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
                    # original code from here on
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
        # change kwargs to positional args to be compatible with peft_utils
        """
        Patch the forward method of the DistilBertModel to use positional arguments instead of keyword arguments.
        Transformers 4 version.
        """

        # based on:
        # https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/distilbert/modeling_distilbert.py#L666
        # original code from here on:
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
            # patch here, change kwargs to positional args:
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
        Patch the forward method of the DistilBertModel to use positional arguments instead of keyword arguments.
        Transformers 5 version.
        """
        # based on:
        # https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/models/distilbert/modeling_distilbert.py#L386
        # original code from here on:
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

            # patch here: unsloth gradient checkpointing hook needs positional arguments
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

        # prevents sentence-transformers from loading the model a second time, thanks Etherl
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
        "roberta",
        "xlm-roberta",
        "albert",
        "electra",
    }

    @staticmethod
    def _estimate_compile_threshold(model):
        """
        Estimate the minimum training steps needed for torch.compile to be beneficial.
        Returns the threshold with a 1.2x safety margin built in.

        Based on empirical benchmarks:
        - Larger models have lower breakeven (more time saved per step)
        - Warmup time scales with model size but speedup also increases
        """
        # Get parameter count from inner model
        if hasattr(model, "__getitem__"):
            try:
                inner = model[0].auto_model
                params = sum(p.numel() for p in inner.parameters())
            except:
                params = 100_000_000  # Default to 100M if can't determine
        else:
            params = sum(p.numel() for p in model.parameters())

        params_m = params / 1e6

        # Empirical formula based on benchmarks with batch_size=2, grad_accum=4
        # Small models: high fixed overhead, lower speedup
        # Large models: warmup scales but speedup is significant
        if params_m < 50:
            estimated_warmup = 35 + params_m * 0.3
            base_speedup = 1.35
        elif params_m < 200:
            estimated_warmup = 12 + params_m * 0.03
            base_speedup = 1.75
        else:
            estimated_warmup = 15 + params_m * 0.04
            base_speedup = 1.60

        # Estimate time per step (ms) and time saved
        naive_ms = 50 + params_m * 1.0
        compiled_ms = naive_ms / base_speedup
        time_saved_per_step_s = (naive_ms - compiled_ms) / 1000

        if time_saved_per_step_s > 0:
            breakeven = estimated_warmup / time_saved_per_step_s
        else:
            breakeven = float("inf")

        # Return threshold with 1.2x safety margin
        return int(breakeven * 1.2)

    @staticmethod
    def _apply_torch_compile(model, mode = "default"):
        """
        Apply torch.compile to a SentenceTransformer model.
        Includes workaround for accelerate's unwrap_model bug.
        """
        if hasattr(model, "__getitem__"):
            inner_model = model[0].auto_model
            compiled = torch.compile(inner_model, mode = mode)
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

        # if for_inference == True, skip Unsloth optimizations to avoid torch compile issues
        if for_inference:
            st_device = device_map
            if isinstance(st_device, dict) or (
                isinstance(st_device, str) and st_device in ["auto", "sequential"]
            ):
                st_device = None

            # this was added because when loading for inference it was defaulting to float32
            # propagate dtype to model_kwargs, default to "auto"
            model_kwargs = kwargs.get("model_kwargs", {})
            model_kwargs["dtype"] = dtype if dtype is not None else "auto"

            # filter kwargs for SentenceTransformer
            st_kwargs = {
                "device": st_device,
                "trust_remote_code": trust_remote_code,
                "token": token,
                "revision": revision,
                "model_kwargs": model_kwargs,
            }

            # add other known kwargs if present
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

        # sanity check, thanks Etherl:
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

        # Fast encoder path: Use native torch.compile for encoder models (6x speedup)
        # This bypasses Unsloth's auto-compiler which adds @torch.compiler.disable decorators
        # that interfere with torch.compile and cause runtime errors for encoder models.
        # NOTE: The old Unsloth path is BROKEN for encoder models with torch 2.9+ due to
        # conflicting @torch.compile and @torch.compiler.disable decorators.
        # Set UNSLOTH_COMPILE_DISABLE=1 to disable torch.compile and use the old path.
        is_encoder_model = (
            model_type.lower() in FastSentenceTransformer.ENCODER_MODEL_TYPES
        )
        use_fast_encoder = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") != "1"
        if use_fast_encoder and is_encoder_model:
            # torch.compile mode: "default" is safest for PEFT/LoRA training
            # Note: "reduce-overhead" uses CUDA Graphs which is incompatible with PEFT
            compile_mode = "default"

            # Determine dtype - handle float16 machines that don't support bfloat16
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

            # Determine device
            st_device = device_map
            if isinstance(st_device, dict) or (
                isinstance(st_device, str) and st_device in ["auto", "sequential"]
            ):
                st_device = "cuda"

            # Check if model supports SDPA (Scaled Dot Product Attention) for extra speedup
            supports_sdpa = False
            if config is not None:
                try:
                    model_class = _get_model_class(
                        config, kwargs.get("auto_model", AutoModel)._model_mapping
                    )
                    supports_sdpa = getattr(model_class, "_supports_sdpa", False)
                except:
                    pass

            # Build model_kwargs for SentenceTransformer
            model_kwargs = {"torch_dtype": dtype}

            # Enable SDPA if supported (1.2x extra speedup on top of torch.compile)
            if supports_sdpa:
                model_kwargs["attn_implementation"] = "sdpa"

            # Print optimization status
            sdpa_str = " + SDPA" if supports_sdpa else ""
            if load_in_4bit:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} with 4-bit quantization{sdpa_str}"
                )
            else:
                print(
                    f"Unsloth: Using fast encoder path for {model_type} (torch.compile{sdpa_str})"
                )

            # Handle 4-bit quantization via BitsAndBytesConfig
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_compute_dtype = dtype,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_use_double_quant = True,
                )
                model_kwargs["quantization_config"] = bnb_config
                # When using quantization, device must be handled by accelerate
                st_device = None

            # Handle gradient checkpointing - warn user it conflicts with torch.compile
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

            # Load via native SentenceTransformer (bypasses Unsloth patching)
            st_model = SentenceTransformer(
                model_name,
                device = st_device,
                trust_remote_code = trust_remote_code,
                token = token,
                revision = revision,
                model_kwargs = model_kwargs,
            )

            # Store metadata for get_peft_model
            st_model._unsloth_fast_encoder = True
            st_model._compile_mode = compile_mode
            st_model._dtype = dtype
            st_model._load_in_4bit = load_in_4bit
            st_model.no_modules = False

            # Add save methods
            def _save_pretrained_merged(self, save_directory, **save_kwargs):
                self.save_pretrained(save_directory)
                tokenizer = save_kwargs.pop("tokenizer", self.tokenizer)
                if hasattr(self[0], "auto_model"):
                    inner = self[0].auto_model
                    # Handle compiled model
                    if hasattr(inner, "_orig_mod"):
                        inner = inner._orig_mod
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
                _save_pretrained_gguf, st_model
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

        # Warn if using 4-bit with encoder (slow due to dequantization overhead)
        if is_encoder_model and load_in_4bit:
            print(
                "Unsloth Warning: 4-bit quantization adds ~2.3x overhead for encoder models."
            )
            print("Consider using load_in_16bit=True for better performance.")

        # check if the model supports add_pooling_layer
        if "add_pooling_layer" not in kwargs:
            supported = FastSentenceTransformer._has_add_pooling_layer(
                config, kwargs.get("auto_model", AutoModel)
            )
            if supported:
                kwargs["add_pooling_layer"] = False

        # forces fp8 to be False since it's not supported
        fp8 = kwargs.pop("load_in_fp8", None)
        if fp8:
            logging.info("Unsloth: Disabling fp8 for model")
        load_in_fp8 = False

        # this is a fix for Snowflake/snowflake-arctic-embed-l-v2.0
        # it has pooler weights which we don't care about for training,
        # however unsloth throws an exception if "UNSLOTH_WARN_UNINITIALIZED" == 1 and it sees unused weights
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

        # check if modules.json exists - if not, force 16-bit training
        # why? because i have to implement saving myself for these models, and i don't feel like adding dequantization
        # to the save_pretrained_merged for a model that really should be trained in 16-bit anyway
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

        def _save_pretrained_merged(self, save_directory, **kwargs):
            # check which adapter files exist before save_pretrained
            adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
            existing_before = {
                f
                for f in adapter_files
                if os.path.exists(os.path.join(save_directory, f))
            }

            # sentence-transformers config and modules only get saved if we call save_pretrained
            self.save_pretrained(save_directory)

            # remove LoRA adapters only if they were created by save_pretrained (not pre-existing)
            for file in adapter_files:
                if file not in existing_before:
                    try:
                        os.remove(os.path.join(save_directory, file))
                    except:
                        pass

            tokenizer = kwargs.pop("tokenizer", self.tokenizer)
            if self.no_modules:
                # fallback for non-sentence-transformers models
                print(
                    "Unsloth: No modules detected. Using standard merge_and_unload for saving..."
                )
                safe_kwargs = kwargs.copy()
                # filter out Unsloth-specific args that are not in huggingface's save_pretrained
                unsloth_args = [
                    "save_method",
                    "temporary_location",
                    "maximum_memory_usage",
                ]
                for k in unsloth_args:
                    safe_kwargs.pop(k, None)

                merged_model = self[0].auto_model.merge_and_unload()
                merged_model.save_pretrained(save_directory, **safe_kwargs)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_directory)
            else:
                self[0].auto_model.save_pretrained_merged(
                    save_directory, tokenizer = tokenizer, **kwargs
                )

            # add Unsloth branding to the generated README
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
            _save_pretrained_gguf, st_model
        )

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

            # order doesn't seem to matter for this after repo creation...
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
                # Fast encoder path: Use native PEFT + torch.compile (6x speedup)
                transformer_module = model[0]
                inner_model = transformer_module.auto_model

                # Check if model is quantized (4-bit/8-bit)
                is_quantized = (
                    getattr(inner_model, "is_quantized", False)
                    or getattr(inner_model.config, "quantization_config", None)
                    is not None
                )

                # Track if gradient checkpointing was actually enabled
                gc_enabled = False

                # this is needed when from_pretrained was called without gradient
                # checkpointing but get_peft_model requests it
                if use_gradient_checkpointing and use_gradient_checkpointing != False:
                    import transformers
                    from packaging.version import Version

                    transformers4 = Version(transformers.__version__).major < 5
                    model_type = getattr(inner_model.config, "model_type", "").lower()

                    if model_type == "mpnet" and transformers4:
                        FastSentenceTransformer._patch_mpnet_v4()
                    elif model_type == "mpnet":
                        FastSentenceTransformer._patch_mpnet_v5()

                # Prepare for k-bit training if quantized
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
                            # Model doesn't support gradient checkpointing, disable it
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

                # Enable gradient checkpointing if requested (only for non-quantized, since prepare_model handles it)
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

                # Create LoRA config
                lora_config = LoraConfig(
                    r = r,
                    lora_alpha = lora_alpha,
                    target_modules = target_modules,
                    lora_dropout = lora_dropout,
                    bias = bias,
                    task_type = kwargs.get("task_type", "FEATURE_EXTRACTION"),
                )

                # Apply PEFT directly (not through FastModel)
                peft_model = peft_get_peft_model(inner_model, lora_config)

                # Apply QAT if specified
                qat_scheme = kwargs.get("qat_scheme", None)
                if qat_scheme is not None:
                    from ._utils import _prepare_model_for_qat

                    peft_model = _prepare_model_for_qat(peft_model, qat_scheme)

                # Determine compile mode (only if not using gradient checkpointing)
                compile_mode = getattr(model, "_compile_mode", "default")
                # Re-enable torch.compile if gradient checkpointing was requested but couldn't be enabled
                if compile_mode is None and not gc_enabled:
                    compile_mode = "default"
                    print(
                        "Unsloth: Re-enabling torch.compile since gradient checkpointing is not supported"
                    )

                # Re-assign the peft model back to the transformer module
                transformer_module.auto_model = peft_model

                # Store compile info for auto-compile at trainer time
                # torch.compile is deferred until training starts so we can check max_steps
                if compile_mode is not None:
                    model._compile_mode = compile_mode
                    model._compile_threshold = (
                        FastSentenceTransformer._estimate_compile_threshold(model)
                    )
                    # Flag to indicate compile has not been applied yet
                    model._compile_pending = True
                    print(
                        f"Unsloth: torch.compile will be applied automatically if max_steps > {model._compile_threshold}"
                    )
                else:
                    model._compile_mode = None
                    model._compile_pending = False
                    print(
                        "Unsloth: torch.compile disabled (gradient checkpointing enabled)"
                    )

                return model

            # Original path for non-fast-encoder models
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

            # re-assign the peft model back to the transformer module
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
    """
    Patch SentenceTransformerTrainer to automatically apply torch.compile
    when training steps exceed the breakeven threshold.

    This is called automatically when this module is imported.
    """
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
        # Extract model and training_args
        model = kwargs.get("model") or (args[0] if args else None)
        training_args = kwargs.get("args") or (args[1] if len(args) > 1 else None)

        # Check if model has pending compile
        if (
            model is not None
            and training_args is not None
            and getattr(model, "_compile_pending", False)
        ):
            max_steps = getattr(training_args, "max_steps", -1)
            threshold = getattr(model, "_compile_threshold", 0)
            compile_mode = getattr(model, "_compile_mode", "default")

            if max_steps > 0 and max_steps >= threshold:
                print(
                    f"Unsloth: Auto-compiling model ({max_steps} steps >= {threshold} threshold)"
                )
                FastSentenceTransformer._apply_torch_compile(model, mode = compile_mode)
                model._compile_pending = False
            elif max_steps > 0:
                print(
                    f"Unsloth: Skipping torch.compile ({max_steps} steps < {threshold} threshold)"
                )
                model._compile_pending = False

        # Call original __init__
        _original_init(self, *args, **kwargs)

    SentenceTransformerTrainer.__init__ = _patched_init
    SentenceTransformerTrainer._unsloth_auto_compile_patched = True


# Auto-patch trainer on module import
_patch_sentence_transformer_trainer()
