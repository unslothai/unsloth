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
import warnings
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "torch")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")
import bitsandbytes as bnb
from transformers.models.llama.modeling_llama import logger
from transformers import AutoTokenizer
from platform import system as platform_system
platform_system = platform_system()
import math

__version__ = "2024.2"

# Get Flash Attention v2 if Ampere (RTX 30xx, A100)
major_version, minor_version = torch.cuda.get_device_capability()
if major_version >= 8:
    try:
        from flash_attn import flash_attn_func
        # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
        try:
            from flash_attn.flash_attn_interface import flash_attn_cuda
            HAS_FLASH_ATTENTION = True
        except:
            logger.warning_once(
                "Unsloth: Your Flash Attention 2 installation seems to be broken?\n"\
                "A possible explanation is you have a new CUDA version which isn't\n"\
                "yet compatible with FA2? Please file a ticket to Unsloth or FA2.\n"\
                "We shall now use Xformers instead, which gets a 0.01% performance hit.\n"\
                "We found this negligible impact by benchmarking on 1x A100."
            )
            HAS_FLASH_ATTENTION = False
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


IGNORED_TOKENIZER_CHECKING = frozenset((
    "CodeLlamaTokenizerFast",
    "CodeLlamaTokenizer",
))

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

    # Freeze all parameters except LoRA
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    pass

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
    if model is not None:
        model.config.update({"unsloth_version" : __version__})
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        # Fixes https://github.com/unslothai/unsloth/issues/5
        if hasattr(tokenizer, "unk_token"):
            tokenizer.add_special_tokens({"pad_token" : tokenizer.unk_token})
            tokenizer.pad_token = tokenizer.unk_token
        else:
            name = model.config._name_or_path if model is not None else "Model"
            logger.warning_one(
                f"{name} does not have a padding or unknown token!\n"\
                f"Will use the EOS token of id {tokenizer.eos_token_id} as padding."
            )
            assert(hasattr(tokenizer, "eos_token"))
            tokenizer.add_special_tokens({"pad_token" : tokenizer.eos_token})
            tokenizer.pad_token = tokenizer.eos_token
        if model is not None:
            config = model.config.update({"pad_token_id" : tokenizer.eos_token_id})
    pass
    return model, tokenizer
pass


def check_tokenizer(
    model,
    tokenizer,
    model_name = "unsloth/llama-2-7b-bnb-4bit",
    model_max_length = 4096,
    padding_side = "right",
    token = None,
    _reload = True,
):
    # Checks tokenizer for out of bounds ids.
    # Mainly a fix for https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
    # where <sep> had token id=32002.
    # See https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
    # Seems like the Fast tokenizer in Rust breaks things!

    # We ignore some of them!
    if tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
        return tokenizer
    pass

    max_embedding_size = model.model.embed_tokens.weight.shape[0]
    added_tokens_fast = tokenizer.added_tokens_decoder
    added_tokens_fast = {index : str(value) for index, value in added_tokens_fast.items()}
    sorted_keys = sorted(added_tokens_fast)
    added_tokens_fast = {key : added_tokens_fast[key] for key in sorted_keys}

    for j, index in enumerate(added_tokens_fast.keys()):
        if index >= max_embedding_size:
            bad_indices = list(added_tokens_fast.keys  ())[j:]
            bad_tokens  = list(added_tokens_fast.values())[j:]

            if not _reload:
                # Try removing the token
                added_tokens = [str(x) for x in tokenizer.added_tokens_decoder.values()]
                special_tokens = tokenizer.special_tokens_map
                import itertools
                special_tokens = frozenset(
                    itertools.chain.from_iterable(
                        [x] if type(x) is str else x for x in special_tokens.values()
                    )
                )
                can_be_removed1 = [x for x in bad_tokens if x not in special_tokens]
                can_be_removed2 = [x for x in can_be_removed1 if x in tokenizer._added_tokens_encoder.keys()]

                # Check of extra tokens can in fact we removed!

                if  (len(can_be_removed1) == len(bad_tokens)) and \
                    (len(can_be_removed2) == len(bad_tokens)):
                    # Yes it can be fixed!
                    for bad_token in can_be_removed1:
                        remove_id = tokenizer._added_tokens_encoder[bad_token]
                        del tokenizer._added_tokens_decoder[remove_id]
                        del tokenizer._added_tokens_encoder[bad_token]
                    pass
                    # Confirm 1 more time!
                    if max(tokenizer.added_tokens_decoder.keys()) < max_embedding_size:
                        logger.warning_once(
                            f"Unsloth loaded a broken tokenizer `{model_name}`, but managed to repair it!\n"\
                            f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"\
                            "We removed these bad tokens. If you think this is incorrect, fix your tokenizer first."
                        )
                        return tokenizer
                    pass
                pass

                # :( Failure
                raise RuntimeError(
                    f"Unsloth tried to load `{model_name}`, but cannot succeed.\n"\
                    f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"\
                    f"Fix your tokenizer since it'll perform out of bounds memory accesses."
                )
            pass
            
            # Try slow tokenizer which can fix things!
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length = model_max_length,
                padding_side = padding_side,
                token = token,
                use_fast = False,
            )
            return check_tokenizer(
                model = model,
                tokenizer = tokenizer,
                model_name = model_name,
                model_max_length = model_max_length,
                padding_side = padding_side,
                token = token,
                _reload = False,
            )
            break
        pass
    pass
    return tokenizer
pass


# Weirdly LoraLayer.update_layer downcasts PEFT layers to float16??
# For mixed precision, we need it to be in float32 not float16.
def LoraLayer_update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights,
    use_rslora = False):
    # This code works for linear layers, override for other layer types
    if r <= 0:
        raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = torch.nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = torch.nn.Identity()

    self.lora_dropout.update(torch.nn.ModuleDict({adapter_name: lora_dropout_layer}))
    # Actual trainable parameters
    self.lora_A[adapter_name] = torch.nn.Linear(self.in_features, r, bias=False)
    self.lora_B[adapter_name] = torch.nn.Linear(r, self.out_features, bias=False)
    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
        self.scaling[adapter_name] = lora_alpha / r

    if init_lora_weights == "loftq":
        # We manually check for PEFT
        if not hasattr(self, "loftq_init"):
            import peft
            raise RuntimeError(
                f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"\
                "Please install PEFT 0.7.2 or higher.\n"\
                "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
            )
        pass
        self.loftq_init(adapter_name)

    elif init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)

    # check weight and qweight (for GPTQ)
    for weight_name in ("weight", "qweight"):
        weight = getattr(self.get_base_layer(), weight_name, None)
        if weight is not None:
            # [INCORRECT code]
            # 
            # the layer is already completely initialized, this is an update
            # if weight.dtype.is_floating_point or weight.dtype.is_complex:
            #     self.to(weight.device, dtype=weight.dtype)
            # else:
            #     self.to(weight.device)
            self.to(weight.device, non_blocking = True)
            break
    self.set_adapter(self.active_adapters)
pass

# Fix up incorrect downcasting of LoRA weights
from peft.tuners.lora.layer import LoraLayer
LoraLayer.update_layer = LoraLayer_update_layer
from peft.tuners.lora import LoraLayer
LoraLayer.update_layer = LoraLayer_update_layer
