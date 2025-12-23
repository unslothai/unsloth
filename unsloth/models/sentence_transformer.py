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

from .loader import FastModel
import inspect
import json
import os
import types
from huggingface_hub import hf_hub_download
from transformers.models.mpnet import modeling_mpnet
from typing import Optional
import torch
from transformers.modeling_outputs import BaseModelOutput
from collections import OrderedDict
from transformers.models.distilbert import modeling_distilbert
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
import transformers
from packaging.version import Version
from transformers import AutoModel, AutoConfig
from transformers.models.auto.auto_factory import _get_model_class
import tempfile
from huggingface_hub import HfApi, get_token

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
                f"\033[1;33mFailed to detect pooling mode, not a sentence-transformers model. You will have to handle pooling/normalization yourself for inference, but training should be fine.\033[0m"
            )
            return "mean"

    # should prolly be done upstream instead of this hackfest here
    @staticmethod
    def _patch_mpnet_v4():
        """
        Patch the MPNetModel to support gradient checkpointing.
        Supports transformers 4.
        """
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
                        use_reentrant = False,
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
                        use_reentrant = False,
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

        transformer_module = Transformer(
            model_name,
            max_seq_length = max_seq_length,
            model_args = {"trust_remote_code": trust_remote_code},
            config_args = {"trust_remote_code": trust_remote_code},
        )
        transformer_module.auto_model = model
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
            "Unsloth: No modules.json found, falling back to [Transformer, Pooling, Normalize]"
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

    @staticmethod
    def from_pretrained(
        model_name,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        trust_remote_code = False,
        use_gradient_checkpointing = "unsloth",
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
            st_model = SentenceTransformer(
                model_name, device = device_map, trust_remote_code = trust_remote_code
            )
            return st_model

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

        # check if the model supports add_pooling_layer
        if "add_pooling_layer" not in kwargs:
            supported = FastSentenceTransformer._has_add_pooling_layer(
                config, kwargs.get("auto_model", AutoModel)
            )
            if supported:
                kwargs["add_pooling_layer"] = False

        # forces fp8 to be False since it's not supported
        kwargs.pop("load_in_fp8", None)
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
                "\033[1;33mUnsloth: No modules.json found. This is not a sentence-transformers model.\n"
                "Forcing 16-bit loading to simplify merged model saving.\033[0m"
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

        st_model = SentenceTransformer(modules = modules, device = device_map)
        st_model.no_modules = no_modules

        def _save_pretrained_merged(self, save_directory, **kwargs):
            # sentence-transformers config and modules only get saved if we call save_pretrained
            self.save_pretrained(save_directory)

            # remove LoRA adapters since we are saving the merged model
            for file in ["adapter_model.safetensors", "adapter_config.json"]:
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

        st_model.save_pretrained_merged = types.MethodType(
            _save_pretrained_merged, st_model
        )

        def _push_to_hub_merged(self, repo_id, **kwargs):
            token = kwargs.get("token", None) or get_token()
            private = kwargs.get("private", None)
            commit_message = kwargs.get("commit_message", "Upload merged model")

            # save merged model to a temp directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                self.save_pretrained_merged(temp_dir, **kwargs)
                api = HfApi(token=token)
                api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    commit_message=commit_message,
                )
            print(f"Unsloth: Successfully pushed merged model to https://huggingface.co/{repo_id}")

        st_model.push_to_hub_merged = types.MethodType(
            _push_to_hub_merged, st_model
        )
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
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        **kwargs,
    ):
        from sentence_transformers import SentenceTransformer

        if "task_type" not in kwargs:
            kwargs["task_type"] = "FEATURE_EXTRACTION"
            print("Setting task_type to FEATURE_EXTRACTION")

        if isinstance(model, SentenceTransformer):
            # extract inner model from the transformer module
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
