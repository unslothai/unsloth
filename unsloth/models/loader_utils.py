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

from ..device_type import DEVICE_TYPE_TORCH
import importlib
import os
import torch
import re
import tempfile
from typing import Union
from .mapper import (
    INT_TO_FLOAT_MAPPER,
    FLOAT_TO_INT_MAPPER,
    MAP_TO_UNSLOTH_16bit,
    FLOAT_TO_FP8_BLOCK_MAPPER,
    FLOAT_TO_FP8_ROW_MAPPER,
)

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from packaging.version import Version
from transformers import __version__ as transformers_version
from unsloth.models._utils import TorchAOConfig
from unsloth_zoo.utils import Version
import gc

transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")

LOCAL_RANK_KEYS = ("LOCAL_RANK", "RANK")
WORLD_SIZE_KEYS = ("WORLD_SIZE",)

BAD_MAPPINGS = {
    "unsloth/Qwen3-32B-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-32B-bnb-4bit".lower(),  # 32B dynamic quant is way too big
    "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B".lower(),  # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B".lower(),  # We rather do it on the fly
    "unsloth/Qwen3-30B-A3B-Base-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B-Base".lower(),  # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-Base-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B-Base".lower(),  # We rather do it on the fly
}


def _get_env_int(keys):
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _infer_distributed_ranks():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
    return _get_env_int(LOCAL_RANK_KEYS), _get_env_int(WORLD_SIZE_KEYS)


def is_distributed():
    rank, world_size = _infer_distributed_ranks()
    return (world_size or 1) > 1 or (rank is not None and rank > 0)


def prepare_device_map():
    rank, world_size = _infer_distributed_ranks()
    distributed = (world_size or 1) > 1 or (rank is not None and rank > 0)
    if not distributed:
        return None, False

    local_rank = 0 if rank is None else rank
    device_map = {"": f"{DEVICE_TYPE_TORCH}:{local_rank}"}
    try:
        if DEVICE_TYPE_TORCH == "cuda":
            torch.cuda.set_device(local_rank)
        elif DEVICE_TYPE_TORCH == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.set_device(local_rank)
    except Exception:
        pass
    return device_map, True


def __get_model_name(
    model_name,
    load_in_4bit = True,
    INT_TO_FLOAT_MAPPER = None,
    FLOAT_TO_INT_MAPPER = None,
    MAP_TO_UNSLOTH_16bit = None,
    load_in_fp8 = False,
    FLOAT_TO_FP8_BLOCK_MAPPER = None,
    FLOAT_TO_FP8_ROW_MAPPER = None,
):
    model_name = str(model_name)
    lower_model_name = model_name.lower()

    assert load_in_fp8 in (True, False, "block")
    if load_in_fp8 != False:
        if load_in_fp8 == True and (os.environ.get("UNSLOTH_HAS_FBGEMM", "0") == "1"):
            if lower_model_name in FLOAT_TO_FP8_ROW_MAPPER:
                # Faster row scaling only works if FBGEMM works!
                return FLOAT_TO_FP8_ROW_MAPPER[lower_model_name]
            elif lower_model_name in FLOAT_TO_FP8_BLOCK_MAPPER:
                # Otherwise we use the slower blockwise type
                return FLOAT_TO_FP8_BLOCK_MAPPER[lower_model_name]
        else:
            if lower_model_name in FLOAT_TO_FP8_BLOCK_MAPPER:
                return FLOAT_TO_FP8_BLOCK_MAPPER[lower_model_name]
        return None

    elif not SUPPORTS_FOURBIT and lower_model_name in INT_TO_FLOAT_MAPPER:
        model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        print(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "
            f"4bit loading.\nThe minimum required version is 4.37.\n"
            f'Try `pip install --upgrade "transformers>=4.37"`\n'
            f"to obtain the latest transformers build, then restart this session.\n"
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
        return model_name

    elif not load_in_4bit and lower_model_name in INT_TO_FLOAT_MAPPER:
        new_model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
        #     f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        # )
        return new_model_name

    elif not load_in_4bit and lower_model_name in MAP_TO_UNSLOTH_16bit:
        new_model_name = MAP_TO_UNSLOTH_16bit[lower_model_name]
        return new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and lower_model_name in FLOAT_TO_INT_MAPPER:
        # Support returning original full -bnb-4bit name if specified specifically
        # since we'll map it to the dynamic version instead
        if lower_model_name.endswith("-bnb-4bit"):
            return lower_model_name

        new_model_name = FLOAT_TO_INT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
        #     f"We shall load `{new_model_name}` for 4x faster loading."
        # )
        return new_model_name

    return None


def _get_new_mapper():
    try:
        import requests

        new_mapper = "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
        with requests.get(new_mapper, timeout = 3) as new_mapper:
            new_mapper = new_mapper.text
        new_mapper = new_mapper[new_mapper.find("__INT_TO_FLOAT_MAPPER") :]
        new_mapper = (
            new_mapper.replace("INT_TO_FLOAT_MAPPER", "NEW_INT_TO_FLOAT_MAPPER")
            .replace("FLOAT_TO_INT_MAPPER", "NEW_FLOAT_TO_INT_MAPPER")
            .replace("MAP_TO_UNSLOTH_16bit", "NEW_MAP_TO_UNSLOTH_16bit")
        )

        exec(new_mapper, globals())
        return (
            NEW_INT_TO_FLOAT_MAPPER,
            NEW_FLOAT_TO_INT_MAPPER,
            NEW_MAP_TO_UNSLOTH_16bit,
        )
    except:
        return {}, {}, {}


def get_model_name(model_name, load_in_4bit = True, load_in_fp8 = False):
    assert load_in_fp8 in (True, False, "block")
    new_model_name = __get_model_name(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        INT_TO_FLOAT_MAPPER = INT_TO_FLOAT_MAPPER,
        FLOAT_TO_INT_MAPPER = FLOAT_TO_INT_MAPPER,
        MAP_TO_UNSLOTH_16bit = MAP_TO_UNSLOTH_16bit,
        load_in_fp8 = load_in_fp8,
        FLOAT_TO_FP8_BLOCK_MAPPER = FLOAT_TO_FP8_BLOCK_MAPPER,
        FLOAT_TO_FP8_ROW_MAPPER = FLOAT_TO_FP8_ROW_MAPPER,
    )
    # In the rare case, we convert bad model names to other names
    # For eg too large dynamic quants or MoEs
    if (
        new_model_name is not None
        and type(new_model_name) is str
        and new_model_name.lower() in BAD_MAPPINGS
    ):
        new_model_name = BAD_MAPPINGS[new_model_name.lower()]

    if (
        new_model_name is None
        and model_name.count("/") == 1
        and model_name[0].isalnum()
    ):
        # Try checking if a new Unsloth version allows it!
        NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit = (
            _get_new_mapper()
        )
        upgraded_model_name = __get_model_name(
            model_name = model_name,
            load_in_4bit = load_in_4bit,
            INT_TO_FLOAT_MAPPER = NEW_INT_TO_FLOAT_MAPPER,
            FLOAT_TO_INT_MAPPER = NEW_FLOAT_TO_INT_MAPPER,
            MAP_TO_UNSLOTH_16bit = NEW_MAP_TO_UNSLOTH_16bit,
            load_in_fp8 = load_in_fp8,
            FLOAT_TO_FP8_BLOCK_MAPPER = FLOAT_TO_FP8_BLOCK_MAPPER,
            FLOAT_TO_FP8_ROW_MAPPER = FLOAT_TO_FP8_ROW_MAPPER,
        )
        if upgraded_model_name is not None:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version! Please update Unsloth via:\n\n"
                "pip uninstall unsloth unsloth_zoo -y\n"
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
                'pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"\n'
            )
    if load_in_fp8 != False:
        # Handle on the fly TorchAO FP8 quantization
        return new_model_name
    return new_model_name if new_model_name is not None else model_name


def _get_torchao_fp8_config(fp8_mode: str):
    """
    Return a `torchao.quantization.Float8DynamicActivationFloat8WeightConfig`
    to be used for `load_in_fp8=True`.
    """
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerBlock,
        PerRow,
    )

    if fp8_mode == "row":
        granularity = PerRow()
    elif fp8_mode == "block":
        granularity = (PerBlock([1, 128]), PerBlock([128, 128]))
    else:
        raise ValueError("Unsloth: `load_in_fp8` supports only 'row' or 'block'")

    return Float8DynamicActivationFloat8WeightConfig(
        granularity = granularity,
        activation_value_lb = 1e-12,
    )


def _offline_quantize_to_fp8(model_name: str, fp8_mode: str) -> str:
    """
    Quantizes the model to fp8 using torchao and saving the quantized model to a
    temporary location. Return the path to the quantized model.

    Note: Once on-the-fly quantization is added in vllm in
    https://github.com/vllm-project/vllm/pull/26327, we should
    dynamically quantize the model there instead:

      llm = LLM(
        ...
        hf_overrides={"quantization_config_file": "torchao_config.json"},
      )
    """
    temp_dir = tempfile.gettempdir()
    new_model_name = model_name.split("/")[-1] + "-fp8-" + fp8_mode
    new_model_name = os.path.join(temp_dir, new_model_name)
    print(
        f"Unsloth: Quantizing '{model_name}' to fp8, using model_name='{new_model_name}' instead"
    )

    if not os.path.isdir(new_model_name):
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            AutoProcessor,
            TorchAoConfig,
            AutoConfig,
        )

        qconfig = _get_torchao_fp8_config(fp8_mode)
        qconfig = TorchAoConfig(qconfig)
        config = AutoConfig.from_pretrained(model_name)
        is_vlm = any(
            x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
            for x in config.architectures
        )
        is_vlm = is_vlm or hasattr(config, "vision_config")
        auto_model = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        auto_processor = AutoProcessor if is_vlm else AutoTokenizer
        model = auto_model.from_pretrained(
            model_name,
            torch_dtype = "auto",
            device_map = "auto",
            quantization_config = qconfig,
        )
        tokenizer = auto_processor.from_pretrained(model_name)
        model.save_pretrained(new_model_name, safe_serialization = False)
        del model
        for _ in range(2):
            torch.cuda.empty_cache()
            gc.collect()
        tokenizer.save_pretrained(new_model_name)
    return new_model_name


def _tag_model_with_fp8_torchao_config(model: torch.nn.Module, fp8_mode: str):
    """
    Tag a model with a `TorchAOConfig` so downstream callers will know what to do with it.
    """
    try:
        base_config = _get_torchao_fp8_config(fp8_mode)
        model.torchao_config = TorchAOConfig(
            qat_scheme = None,
            base_config_and_filter_fns = [(base_config, None)],
        )
    except:
        pass


def _get_fp8_mode_and_check_settings(
    load_in_fp8: Union[bool, str],
    fast_inference: bool,
    full_finetuning: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    load_in_16bit: bool,
    use_exact_model_name: bool,
) -> str:
    """
    Assuming `load_in_fp8` is enabled, raise appropriate errors on incompatible settings
    and environment. Currently this feature requires:

    1. H100 GPUs or after
    2. torchao 0.15.0+ (or nightly)
    3. torch 2.9.0+
    4. If fbgemm_gpu_genai is installed, require 1.4.1+

    Returns the fp8 mode, one of "row" or "block".
    """
    assert load_in_fp8 is not False
    if load_in_fp8 is True:
        fp8_mode = "row"  # default
    else:
        fp8_mode = load_in_fp8

    # Check user settings
    if fp8_mode not in ["row", "block"]:
        raise ValueError(
            f"Unsloth: `load_in_fp8` can only be 'row' or 'block', got '{fp8_mode}'"
        )
    if not fast_inference:
        raise ValueError(
            "Unsloth: `load_in_fp8` is only supported for `fast_inference` for now"
        )
    if full_finetuning:
        raise ValueError(
            "Unsloth: `load_in_fp8` is not compatible with full finetuning"
        )
    if load_in_4bit or load_in_8bit or load_in_16bit:
        raise ValueError(
            "Unsloth: `load_in_fp8` is not compatible with `load_in_4bit`, `load_in_8bit` or `load_in_16bit`",
        )
    if use_exact_model_name:
        raise ValueError("Unsloth: `load_in_fp8` requires `use_exact_model_name=False`")

    # Check if this is Hopper or above
    if not (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (9, 0)
    ):
        raise ValueError(
            "Unsloth: On the fly `load_in_fp8` requires H100 GPUs or after. Try `unsloth/Qwen3-8B` instead."
        )

    # Check if torch >= 2.9.0
    if Version(torch.__version__) < Version("2.9.0"):
        raise ValueError(
            "Unsloth: On the fly `load_in_fp8` requires torch 2.9.0+. Try `unsloth/Qwen3-8B` instead."
        )

    # Check if torchao has this PR: https://github.com/pytorch/ao/pull/3158,
    # which will be released in 0.15.0.
    if importlib.util.find_spec("torchao") is None:
        raise ValueError(
            "Unsloth: Please install torchao for on the fly float8 to work! Try `unsloth/Qwen3-8B` instead."
        )
    import torchao

    error_message = (
        "Unsloth: `load_in_fp8` requires torchao 0.15.0+ (or nightly).\n"
        f"You have torchao version={torchao.__version__}\n"
        "Use `pip install --upgrade --force-reinstall torchao`"
    )
    if Version(torchao.__version__) < Version("0.15.0"):
        raise ValueError(error_message)

    # If fbgemm_gpu_genai is installed, check if it's >= 1.4.1
    if (
        importlib.util.find_spec("fbgemm_gpu") is not None
        and importlib.util.find_spec("fbgemm_gpu.experimental") is not None
    ):
        import fbgemm_gpu.experimental.gen_ai

        if Version(fbgemm_gpu.__version__) < Version("1.4.1"):
            raise ValueError(
                "Unsloth: On the fly `load_in_fp8` is only compatible with fbgemm_gpu_genai 1.4.1+. Try `unsloth/Qwen3-8B` instead."
            )
    return fp8_mode
