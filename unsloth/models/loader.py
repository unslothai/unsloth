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

from .llama import FastLlamaModel, logger
from .mistral import FastMistralModel
from transformers import AutoConfig


class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name = "mistralai/Mistral-7B-v0.1",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        *args, **kwargs,
    ):
        model_config = AutoConfig.from_pretrained(model_name)
        model_type = model_config.model_type

        if model_type == "llama":
            return FastLlamaModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                token = token,
                device_map = device_map,
                rope_scaling = rope_scaling,
                *args, **kwargs,
            )
        elif model_type == "mistral":
            if rope_scaling is not None:
                logger.warning_once("Mistral models do not support RoPE scaling.")
            return FastMistralModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                token = token,
                device_map = device_map,
                *args, **kwargs,
            )
        else:
            raise NotImplementedError(
                f"{model_name} not supported yet! Make an issue to https://github.com/unslothai/unsloth!",
            )
    pass
pass
