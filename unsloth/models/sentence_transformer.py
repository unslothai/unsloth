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
from sentence_transformers.models import Transformer, Pooling, Normalize
from transformers.models.mpnet import modeling_mpnet
from sentence_transformers.util import import_from_string, load_dir_path
from typing import Optional
import torch
from transformers.modeling_outputs import BaseModelOutput
from collections import OrderedDict


class FastSentenceTransformer(FastModel):
    @staticmethod
    def _read_pooling_mode(model_name, token):
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
            print(f"Failed to detect pooling mode: {e}, defaulting to mean pooling.")
            return "mean"

    # should prolly be done upstream instead of this hackfest here
    @staticmethod
    def _patch_mpnet():
        try:
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
            # but head_mask is no longer used in transformers 5.0.0:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpnet/modeling_mpnet.py#L284
            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_dict: bool = False,
                **kwargs,
            ):
                # backwards compatibility for older transformers versions (4.57.3 and below)
                head_mask = kwargs.pop("head_mask", None)

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
                                return module(
                                    *inputs, output_attentions = output_attentions
                                )

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

        except Exception as e:
            print(f"Unsloth: Failed to patch MPNet for gradient checkpointing: {e}")

    @staticmethod
    def _load_modules(
        model_name,
        token,
        model,
        tokenizer,
        max_seq_length,
        pooling_mode,
        trust_remote_code = False,
    ):
        modules = OrderedDict()

        # grope around for modules.json
        modules_json_path = None
        if os.path.exists(model_name) and os.path.exists(
            os.path.join(model_name, "modules.json")
        ):
            modules_json_path = os.path.join(model_name, "modules.json")
        else:
            try:
                modules_json_path = hf_hub_download(
                    model_name, "modules.json", token = token
                )
            except:
                pass

        if modules_json_path and os.path.exists(modules_json_path):
            with open(modules_json_path, encoding = "utf8") as f:
                modules_config = json.load(f)

            for module_config in modules_config:
                class_ref = module_config["type"]
                name = (
                    module_config["name"]
                    if "name" in module_config
                    else str(module_config.get("idx", len(modules)))
                )

                # main module
                if class_ref == "sentence_transformers.models.Transformer":
                    transformer_module = Transformer(
                        model_name,
                        max_seq_length = max_seq_length,
                        model_args = {"trust_remote_code": trust_remote_code},
                        config_args = {"trust_remote_code": trust_remote_code},
                    )
                    transformer_module.auto_model = model
                    transformer_module.tokenizer = tokenizer

                    # move tokenizer do_lower_case to transformer module
                    transformer_module.do_lower_case = getattr(
                        tokenizer, "do_lower_case", False
                    )
                    model_forward_params = list(
                        inspect.signature(model.forward).parameters
                    )
                    transformer_module.model_forward_params = set(
                        model_forward_params
                    ) | {
                        "input_ids",
                        "attention_mask",
                        "token_type_ids",
                        "inputs_embeds",
                    }
                    if max_seq_length is None:
                        pass

                    # is this overkill? should we just force user to set it?
                    current_max_seq = max_seq_length
                    if current_max_seq is None:
                        if hasattr(model, "config") and hasattr(
                            model.config, "max_position_embeddings"
                        ):
                            current_max_seq = model.config.max_position_embeddings
                        elif hasattr(tokenizer, "model_max_length"):
                            current_max_seq = tokenizer.model_max_length
                        else:
                            current_max_seq = 512

                    transformer_module.max_seq_length = current_max_seq
                    transformer_module.config_keys = ["max_seq_length", "do_lower_case"]
                    transformer_module.save_in_root = True
                    if hasattr(model, "config"):
                        model.config.tokenizer_class = tokenizer.__class__.__name__

                    modules[name] = transformer_module

                # load other modules
                else:
                    module_path = module_config["path"]
                    if os.path.isdir(model_name):
                        load_path = os.path.join(model_name, module_path)
                    else:
                        # still looking
                        try:
                            load_path = load_dir_path(
                                model_name, module_path, token = token
                            )
                        except:
                            print(
                                f"Unsloth Warning: Could not download module {module_path} for {class_ref}. Skipping."
                            )
                            continue

                    module_class = import_from_string(class_ref)
                    # load module
                    try:
                        module = module_class.load(load_path)
                        modules[name] = module
                    except Exception as e:
                        print(
                            f"Unsloth Warning: Failed to load module {name} ({class_ref}) from {load_path}: {e}"
                        )

        else:
            # fallback if no modules.json, is this necessary?
            print(
                "Unsloth: No modules.json found, falling back to [Transformer, Pooling, Normalize]"
            )
            transformer_module = Transformer(
                model_name,
                max_seq_length = max_seq_length,
                model_args = {"trust_remote_code": trust_remote_code},
                config_args = {"trust_remote_code": trust_remote_code},
            )
            transformer_module.auto_model = model
            transformer_module.tokenizer = tokenizer

            # move tokenizer do_lower_case to transformer module
            transformer_module.do_lower_case = getattr(
                tokenizer, "do_lower_case", False
            )
            model_forward_params = list(inspect.signature(model.forward).parameters)
            transformer_module.model_forward_params = set(model_forward_params) | {
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "inputs_embeds",
            }

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
            # add tokenizer class to config for sentence-transformers
            if hasattr(model, "config"):
                model.config.tokenizer_class = tokenizer.__class__.__name__

            modules["0"] = transformer_module

            hidden_size = (
                model.config.hidden_size
                if hasattr(model.config, "hidden_size")
                else 768
            )

            if pooling_mode == "mean":
                pooling_mode = FastSentenceTransformer._read_pooling_mode(
                    model_name, token
                )

            pooling_module = Pooling(
                word_embedding_dimension = hidden_size,
                pooling_mode = pooling_mode,
            )
            # end of fallback
            modules["1"] = pooling_module
            modules["2"] = Normalize()
        return modules

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
        **kwargs,
    ):
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Transformer, Pooling, Normalize
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Unsloth: To use `FastSentenceTransformer`, you must install `sentence-transformers`.\n"
                "Run `pip install sentence-transformers` to install it."
            )

        if "auto_model" not in kwargs:
            kwargs["auto_model"] = AutoModel

        if "add_pooling_layer" not in kwargs:
            kwargs["add_pooling_layer"] = False

        # forces fp8 to be False since it's not supported
        kwargs.pop("load_in_fp8", None)
        load_in_fp8 = False

        # this is a fix for Snowflake/snowflake-arctic-embed-l-v2.0
        # it has pooler weights which we don't care about for training,
        # however unsloth throws an exception if "UNSLOTH_WARN_UNINITIALIZED" == 1 and it sees unused weights
        old_environ = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

        if "mpnet" in model_name.lower():
            FastSentenceTransformer._patch_mpnet()

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

        modules = FastSentenceTransformer._load_modules(
            model_name,
            token,
            model,
            tokenizer,
            max_seq_length,
            pooling_mode,
            trust_remote_code = trust_remote_code,
        )

        st_model = SentenceTransformer(modules = modules, device = device_map)

        def _save_pretrained_merged(self, save_directory, **kwargs):
            # sentence-transformers config and modules only get saved if we call save_pretrained
            self.save_pretrained(save_directory)

            # remove LoRA adapters since we are saving the merged model
            for file in ["adapter_model.safetensors", "adapter_config.json"]:
                try:
                    os.remove(os.path.join(save_directory, file))
                except:
                    pass

            # save merged weights
            tokenizer = kwargs.pop("tokenizer", self.tokenizer)
            self[0].auto_model.save_pretrained_merged(
                save_directory, tokenizer = tokenizer, **kwargs
            )

        st_model.save_pretrained_merged = types.MethodType(
            _save_pretrained_merged, st_model
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
