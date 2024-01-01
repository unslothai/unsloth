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

import torch
from typing import Union, Optional, List, Any, Callable
import numpy as np
import warnings
import gc
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "torch")
import bitsandbytes as bnb
from transformers.models.llama.modeling_llama import logger
from platform import system as platform_system
platform_system = platform_system()

__version__ = "2024.1"

# Get Flash Attention v2 if Ampere (RTX 30xx, A100)
major_version, minor_version = torch.cuda.get_device_capability()
if major_version >= 8:
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH_ATTENTION = True
    except:
        HAS_FLASH_ATTENTION = False
else:
    # Tri Dao's benchmark shows xformers is faster for now.
    HAS_FLASH_ATTENTION = False
pass
import xformers.ops.fmha as xformers
xformers_attention = xformers.memory_efficient_attention
from xformers import __version__ as xformers_version

__all__ = [
    "prepare_model_for_kbit_training",
    "patch_tokenizer",
    "check_tokenizer",
    "xformers",
    "xformers_attention",
    "xformers_version",
    "__version__",
    "HAS_FLASH_ATTENTION",
    "platform_system",
]


def prepare_model_for_kbit_training(
    model                      : Any,
    use_gradient_checkpointing : bool = True,
    use_reentrant              : Optional[bool] = True,
) -> Any:
    """
    Calculates where to place the gradient checkpoints given n_layers.
    We also freeze all other layers's gradients

    Args:
        model: Any LlamaModel with layers.
        use_gradient_checkpointing (`bool`, *optional*):
            Default enabled. Provides memory savings by not saving all activations,
            but only some.
        use_reentrant (`bool`, *optional*):
            https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L354
            Optimal gradient checkpointing algorithm which will be the default in
            future Pytorch versions.
    """

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model
pass


def patch_tokenizer(model, tokenizer):
    model.config.update({"unsloth_version" : __version__})
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        # Fixes https://github.com/unslothai/unsloth/issues/5
        if hasattr(tokenizer, "unk_token"):
            tokenizer.add_special_tokens({"pad_token" : tokenizer.unk_token})
            tokenizer.pad_token = tokenizer.unk_token
        else:
            logger.warning_one(
                f"{model.config._name_or_path} does not have a padding or unknown token!\n"\
                f"Will use the EOS token of id {tokenizer.eos_token_id} as padding."
            )
            assert(hasattr(tokenizer, "eos_token"))
            tokenizer.add_special_tokens({"pad_token" : tokenizer.eos_token})
            tokenizer.pad_token = tokenizer.eos_token
        config = model.config.update({"pad_token_id" : tokenizer.eos_token_id})
    pass
    return model, tokenizer
pass


def check_tokenizer(model, tokenizer):
    # Checks tokenizer for out of bounds ids.
    # Mainly a fix for https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
    # where <sep> had token id=32002.
    # See https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
    special_tokens_map = tokenizer.special_tokens_map
    max_embedding_size = model.model.embed_tokens.weight.shape[0]

    for token_name, token_content in special_tokens_map.items():
        if type(token_content) is not str: continue
        token_ids = tokenizer([token_content], add_special_tokens = False, return_attention_mask = False)
        token_ids = token_ids.input_ids[0][0]
        if token_ids < 0 or token_ids >= max_embedding_size:
            raise RuntimeError(
                f"Unsloth: Extra special token `{token_content}` with id={token_ids} exceeds "\
                f"the maximum vocabulary size of {max_embedding_size}. You must fix the tokenizer "\
                "or else out of bounds memory accesses will occur."
            )
        pass
    pass
pass
