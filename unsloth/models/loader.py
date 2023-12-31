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
from transformers import __version__ as transformers_version

FOURBIT_MAPPER = \
{
    "unsloth/mistral-7b-bnb-4bit"    : "unsloth/mistral-7b",
    "unsloth/llama-2-7b-bnb-4bit"    : "unsloth/llama-2-7b",
    "unsloth/llama-2-13b-bnb-4bit"   : "unsloth/llama-13-7b",
    "unsloth/codellama-34b-bnb-4bit" : "codellama/CodeLlama-34b-hf",
    "unsloth/zephyr-sft-bnb-4bit"    : "unsloth/zephyr-sft",
}

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
major, minor = transformers_version.split(".")[:2]
major, minor = int(major), int(minor)
SUPPORTS_FOURBIT = (major > 4) or (major == 4 and minor >= 37)
del major, minor


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        *args, **kwargs,
    ):
        if not SUPPORTS_FOURBIT and model_name in FOURBIT_MAPPER:
            model_name = FOURBIT_MAPPER[model_name]
            logger.warning_once(
                f"Unsloth: Your transformers version of {transformers_version} does not support native "\
                f"4bit loading.\nThe minimum required version is 4.37.\n"\
                f'Try `pip install "git+https://github.com/huggingface/transformers.git"`\n'\
                f"to obtain the latest transformers build, then restart this session.\n"\
                f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
            )
        pass

        model_config = AutoConfig.from_pretrained(model_name)
        model_type = model_config.model_type

        if   model_type == "llama":   dispatch_model = FastLlamaModel
        elif model_type == "mistral": dispatch_model = FastMistralModel
        else:
            raise NotImplementedError(
                f"Unsloth: {model_name} not supported yet!\n"\
                "Make an issue to https://github.com/unslothai/unsloth!",
            )

        return dispatch_model.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = token,
            device_map = device_map,
            rope_scaling = rope_scaling,
            *args, **kwargs,
        )
    pass
pass
