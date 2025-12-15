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
import torch
import inspect
import json
import os
from huggingface_hub import hf_hub_download


class FastSentenceTransformer(FastModel):
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
        load_in_fp8 = False,
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

        old_environ = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

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

        transformer_module = Transformer.__new__(Transformer)
        torch.nn.Module.__init__(transformer_module)
        transformer_module.auto_model = model
        transformer_module.tokenizer = tokenizer
        transformer_module.do_lower_case = False
        if hasattr(tokenizer, "do_lower_case"):
            transformer_module.do_lower_case = tokenizer.do_lower_case

        model_forward_params = list(inspect.signature(model.forward).parameters)
        transformer_module.model_forward_params = set(model_forward_params) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
        }

        if max_seq_length is None:
            if (
                hasattr(model, "config")
                and hasattr(model.config, "max_position_embeddings")
                and hasattr(tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    model.config.max_position_embeddings, tokenizer.model_max_length
                )
            elif hasattr(model.config, "max_position_embeddings"):
                max_seq_length = model.config.max_position_embeddings
            elif hasattr(tokenizer, "model_max_length"):
                max_seq_length = tokenizer.model_max_length
            else:
                max_seq_length = 512  # default

        transformer_module.max_seq_length = max_seq_length
        transformer_module.config_keys = ["max_seq_length", "do_lower_case"]
        transformer_module.save_in_root = True
        if hasattr(model, "config"):
            model.config.tokenizer_class = tokenizer.__class__.__name__

        hidden_size = model.config.hidden_size

        # detect pooling mode if not specified/default
        if pooling_mode == "mean":
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
                        pooling_map = {
                            "pooling_mode_cls_token": "cls",
                            "pooling_mode_mean_tokens": "mean",
                            "pooling_mode_max_tokens": "max",
                            "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
                        }
                        for config_key, mode in pooling_map.items():
                            if pooling_config.get(config_key):
                                print(f"Pooling mode detected as {mode}, updating...")
                                pooling_mode = mode
                                break

            except Exception as e:
                print(
                    f"Failed to detect pooling mode: {e}, defaulting to mean pooling."
                )

        pooling_module = Pooling(
            word_embedding_dimension = hidden_size,
            pooling_mode = pooling_mode,
        )
        normalize_module = Normalize()
        modules = [transformer_module, pooling_module, normalize_module]
        st_model = SentenceTransformer(modules = modules)
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
