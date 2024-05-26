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
from .qwen2 import FastQwen2Model
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from .mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER
import os

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
major, minor = transformers_version.split(".")[:2]
major, minor = int(major), int(minor)
SUPPORTS_FOURBIT = (major > 4) or (major == 4 and minor >= 37)
SUPPORTS_GEMMA   = (major > 4) or (major == 4 and minor >= 38)
if SUPPORTS_GEMMA:
    from .gemma import FastGemmaModel
del major, minor


def _get_model_name(model_name, load_in_4bit = True):

    if not SUPPORTS_FOURBIT and model_name in INT_TO_FLOAT_MAPPER:
        model_name = INT_TO_FLOAT_MAPPER[model_name]
        logger.warning_once(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "\
            f"4bit loading.\nThe minimum required version is 4.37.\n"\
            f'Try `pip install --upgrade "transformers>=4.37"`\n'\
            f"to obtain the latest transformers build, then restart this session.\n"\
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
    
    elif not load_in_4bit and model_name in INT_TO_FLOAT_MAPPER:
        new_model_name = INT_TO_FLOAT_MAPPER[model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
        #     f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        # )
        model_name = new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and model_name in FLOAT_TO_INT_MAPPER:
        new_model_name = FLOAT_TO_INT_MAPPER[model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
        #     f"We shall load `{new_model_name}` for 4x faster loading."
        # )
        model_name = new_model_name
    pass

    return model_name
pass


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name     = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = None,
        dtype          = None,
        load_in_4bit   = True,
        token          = None,
        device_map     = "sequential",
        rope_scaling   = None,
        fix_tokenizer  = True,
        trust_remote_code = False,
        use_gradient_checkpointing = True,
        resize_model_vocab = None,
        *args, **kwargs,
    ):
        if token is None and "HF_TOKEN" in os.environ:
            token = os.environ["HF_TOKEN"]

        if token is None and "HUGGINGFACE_TOKEN" in os.environ:
            token = os.environ["HUGGINGFACE_TOKEN"]

        old_model_name = model_name
        model_name = _get_model_name(model_name, load_in_4bit)

        # First check if it's a normal model via AutoConfig
        is_peft = False
        try:
            model_config = AutoConfig.from_pretrained(model_name, token = token)
            is_peft = False
        except:
            try:
                # Most likely a PEFT model
                peft_config = PeftConfig.from_pretrained(model_name, token = token)
            except:
                raise RuntimeError(f"Unsloth: `{model_name}` is not a full model or a PEFT model.")
            
            # Check base model again for PEFT
            model_name = _get_model_name(peft_config.base_model_name_or_path, load_in_4bit)
            model_config = AutoConfig.from_pretrained(model_name, token = token)
            is_peft = True
        pass

        model_type = model_config.model_type

        if   model_type == "llama":   dispatch_model = FastLlamaModel
        elif model_type == "mistral": dispatch_model = FastMistralModel
        elif model_type == "gemma":
            if not SUPPORTS_GEMMA:
                raise RuntimeError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"\
                    f"The minimum required version is 4.38.\n"\
                    f'Try `pip install --upgrade "transformers>=4.38"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            dispatch_model = FastGemmaModel
        elif model_type == "qwen2":
            dispatch_model = FastQwen2Model
        else:
            raise NotImplementedError(
                f"Unsloth: {model_name} not supported yet!\n"\
                "Make an issue to https://github.com/unslothai/unsloth!",
            )
        pass

        # Check if this is local model since the tokenizer gets overwritten
        if  os.path.exists(os.path.join(old_model_name, "tokenizer_config.json")) and \
            os.path.exists(os.path.join(old_model_name, "tokenizer.json")) and \
            os.path.exists(os.path.join(old_model_name, "special_tokens_map.json")):

            tokenizer_name = old_model_name
        else:
            tokenizer_name = None
        pass

        model, tokenizer = dispatch_model.from_pretrained(
            model_name     = model_name,
            max_seq_length = max_seq_length,
            dtype          = dtype,
            load_in_4bit   = load_in_4bit,
            token          = token,
            device_map     = device_map,
            rope_scaling   = rope_scaling,
            fix_tokenizer  = fix_tokenizer,
            model_patcher  = dispatch_model,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            *args, **kwargs,
        )
        
        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["unsloth",])
        pass
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(["unsloth",])
        pass

        if load_in_4bit:
            # Fix up bitsandbytes config
            quantization_config = \
            {
                # Sometimes torch_dtype is not a string!!
                "bnb_4bit_compute_dtype"           : model.config.to_dict()["torch_dtype"],
                "bnb_4bit_quant_type"              : "nf4",
                "bnb_4bit_use_double_quant"        : True,
                "llm_int8_enable_fp32_cpu_offload" : False,
                "llm_int8_has_fp16_weight"         : False,
                "llm_int8_skip_modules"            : None,
                "llm_int8_threshold"               : 6.0,
                "load_in_4bit"                     : True,
                "load_in_8bit"                     : False,
                "quant_method"                     : "bitsandbytes",
            }
            model.config.update({"quantization_config" : quantization_config})
        pass

        if is_peft:
            # Now add PEFT adapters
            model = PeftModel.from_pretrained(model, old_model_name, token = token)
            # Patch it as well!
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer
    pass
pass
