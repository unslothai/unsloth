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
import contextlib
import threading as _threading
import functools
from typing import Union
from .mapper import (
    INT_TO_FLOAT_MAPPER,
    FLOAT_TO_INT_MAPPER,
    MAP_TO_UNSLOTH_16bit,
    FLOAT_TO_FP8_BLOCK_MAPPER,
    FLOAT_TO_FP8_ROW_MAPPER,
)

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from transformers import __version__ as transformers_version
from unsloth.models._utils import TorchAOConfig
from unsloth_zoo.utils import Version, get_quant_type
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


def _get_torchao_fp8_config(fp8_mode):
    # Lazy import so a broken optional vLLM install doesn't break `import unsloth`.
    from unsloth_zoo.vllm_utils import _get_torchao_fp8_config as _impl
    return _impl(fp8_mode)


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
    if (
        torch.distributed.is_available()
        and getattr(torch.distributed, "is_initialized", lambda: False)()
    ):
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
        # No pre-quantized model found. vllm >= 0.12.0 quantizes to FP8 on the
        # fly (return original name); older vllm falls through to offline quant.
        if importlib.util.find_spec("vllm") is not None:
            import vllm
            if Version(vllm.__version__) >= Version("0.12.0"):
                return model_name
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
        # Keep an explicit -bnb-4bit name; otherwise map to the dynamic version.
        if lower_model_name.endswith("-bnb-4bit"):
            return model_name

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

        new_mapper = (
            "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
        )
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


def _resolve_with_mappers(
    model_name, load_in_4bit, load_in_fp8, int_to_float, float_to_int, map_to_unsloth_16bit
):
    return __get_model_name(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        INT_TO_FLOAT_MAPPER = int_to_float,
        FLOAT_TO_INT_MAPPER = float_to_int,
        MAP_TO_UNSLOTH_16bit = map_to_unsloth_16bit,
        load_in_fp8 = load_in_fp8,
        FLOAT_TO_FP8_BLOCK_MAPPER = FLOAT_TO_FP8_BLOCK_MAPPER,
        FLOAT_TO_FP8_ROW_MAPPER = FLOAT_TO_FP8_ROW_MAPPER,
    )


def get_model_name(
    model_name,
    load_in_4bit = True,
    load_in_fp8 = False,
    token = None,
    trust_remote_code = False,
):
    assert load_in_fp8 in (True, False, "block")
    new_model_name = _resolve_with_mappers(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        load_in_fp8 = load_in_fp8,
        int_to_float = INT_TO_FLOAT_MAPPER,
        float_to_int = FLOAT_TO_INT_MAPPER,
        map_to_unsloth_16bit = MAP_TO_UNSLOTH_16bit,
    )
    # Remap "bad" names (e.g. oversized dynamic quants or MoEs)
    if (
        new_model_name is not None
        and type(new_model_name) is str
        and new_model_name.lower() in BAD_MAPPINGS
    ):
        new_model_name = BAD_MAPPINGS[new_model_name.lower()]
    elif new_model_name is None and model_name.lower() in BAD_MAPPINGS:
        # Some bad names (e.g. the `-unsloth-bnb-4bit` dynamic quants) are keys
        # of the mappers, not values, so the resolver returns None for them and
        # the remap above is skipped; remap the input name directly instead.
        new_model_name = BAD_MAPPINGS[model_name.lower()]

    if (
        new_model_name is None
        and model_name.count("/") == 1
        and model_name[0].isalnum()
        and not _env_says_offline()  # offline: skip the remote (raw GitHub) mapper refresh
    ):
        # Try checking if a new Unsloth version allows it!
        NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit = (
            _get_new_mapper()
        )
        upgraded_model_name = _resolve_with_mappers(
            model_name = model_name,
            load_in_4bit = load_in_4bit,
            load_in_fp8 = load_in_fp8,
            int_to_float = NEW_INT_TO_FLOAT_MAPPER,
            float_to_int = NEW_FLOAT_TO_INT_MAPPER,
            map_to_unsloth_16bit = NEW_MAP_TO_UNSLOTH_16bit,
        )
        if upgraded_model_name is not None:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version! Please update Unsloth via:\n\n"
                "pip uninstall unsloth unsloth_zoo -y\n"
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
                'pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"\n'
            )

    if new_model_name is None:
        new_model_name = model_name

    return new_model_name


def _offline_quantize_to_fp8(
    model_name: str,
    fp8_mode: str,
    *,
    text_only: bool = False,
) -> str:
    """Quantize the model to fp8 via torchao, save to a temp dir, return its path.

    For vllm >= 0.12.0, prefer dynamic quantization in vllm instead (via
    hf_overrides={"quantization_config_file": "torchao_config.json"}).
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        AutoProcessor,
        TorchAoConfig,
        AutoConfig,
    )

    config = AutoConfig.from_pretrained(model_name)
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in (getattr(config, "architectures", None) or [])
    )
    is_vlm = is_vlm or hasattr(config, "vision_config")
    # Decide text-only before the cache name so the fp8 artifact and its path stay in sync. #5816
    text_config = None
    if text_only and hasattr(config, "vision_config"):
        from ._utils import (
            _get_text_only_config,
            resolve_model_class,
            _is_family_text_decoder,
        )

        candidate = _get_text_only_config(config, model_name)
        text_class = resolve_model_class(AutoModelForCausalLM, candidate)
        if text_class is not None and _is_family_text_decoder(
            getattr(config, "model_type", ""),
            getattr(candidate, "model_type", ""),
        ):
            text_config = candidate
            is_vlm = False

    temp_dir = tempfile.gettempdir()
    # Cache text-only and full-VLM artifacts separately so neither reuses the other. #5816
    cache_name = model_name.split("/")[-1] + "-fp8-" + fp8_mode
    if text_config is not None:
        cache_name += "-text-only"
    new_model_name = os.path.join(temp_dir, cache_name)
    print(f"Unsloth: Quantizing '{model_name}' to fp8, using model_name='{new_model_name}' instead")

    if not os.path.isdir(new_model_name):
        from ._utils import _apply_text_only_key_mapping

        qconfig = _get_torchao_fp8_config(fp8_mode)
        qconfig = TorchAoConfig(qconfig)
        load_kwargs = dict(torch_dtype = "auto", device_map = "auto", quantization_config = qconfig)
        if text_config is not None:
            _apply_text_only_key_mapping(load_kwargs, config, text_config)
            config = text_config
        auto_model = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        auto_processor = AutoProcessor if is_vlm else AutoTokenizer
        model = auto_model.from_pretrained(
            model_name,
            config = config,
            **load_kwargs,
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
    """Tag a model with a `TorchAOConfig` so downstream callers know how to handle it."""
    try:
        base_config = _get_torchao_fp8_config(fp8_mode)
        model.torchao_config = TorchAOConfig(
            qat_scheme = None,
            base_config_and_filter_fns = [(base_config, None)],
        )
    except:
        pass


_FP8_DTYPES = tuple(
    dtype
    for dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if dtype is not None
)


def _fp8_block_size_from_config(model):
    """Return the [block_out, block_in] block size of an fp8 checkpoint, or None if not block-fp8."""
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return None
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    if not isinstance(quant, dict):
        return None
    if quant.get("quant_method") != "fp8":
        return None
    block = quant.get("weight_block_size")
    if not block:
        return None
    if isinstance(block, (int, float)):
        block = [block, block]
    elif isinstance(block, (list, tuple)):
        if len(block) == 1:
            block = [block[0], block[0]]
        elif len(block) < 2:
            return None
    else:
        return None
    return [int(block[0]), int(block[1])]


def _load_fp8_weight_map(
    model_name,
    local_files_only,
    token,
    revision = None,
    subfolder = None,
    cache_dir = None,
):
    """Return the checkpoint's tensor->file map, using the same snapshot the load used.

    Prefers the sharded `model.safetensors.index.json`; falls back to a single `model.safetensors`
    (every tensor maps to that one file) so unsharded checkpoints are covered too.
    """

    def _local_path(filename):
        return (
            os.path.join(model_name, subfolder, filename)
            if subfolder
            else os.path.join(model_name, filename)
        )

    def _remote_path(filename):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            model_name,
            filename,
            revision = revision,
            subfolder = subfolder,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            token = token,
        )

    index_file = "model.safetensors.index.json"
    single_file = "model.safetensors"
    is_local = os.path.isdir(model_name)

    # Sharded checkpoint.
    if is_local and os.path.exists(_local_path(index_file)):
        index_path = _local_path(index_file)
    elif not is_local:
        try:
            index_path = _remote_path(index_file)
        except Exception:
            index_path = None
    else:
        index_path = None
    if index_path is not None:
        import json
        with open(index_path, "r") as f:
            return json.load(f).get("weight_map", None)

    # Unsharded single file: map every tensor to it.
    try:
        if is_local and os.path.exists(_local_path(single_file)):
            single_path = _local_path(single_file)
        elif not is_local:
            single_path = _remote_path(single_file)
        else:
            return None
        from safetensors import safe_open
        with safe_open(single_path, framework = "pt") as f:
            return {key: single_file for key in f.keys()}
    except Exception:
        return None


def _resolve_fp8_shard(
    model_name,
    shard,
    local_files_only,
    token,
    revision = None,
    subfolder = None,
    cache_dir = None,
):
    """Resolve a checkpoint shard filename to a local path (repo id or local dir)."""
    if os.path.isdir(model_name):
        return (
            os.path.join(model_name, subfolder, shard)
            if subfolder
            else os.path.join(model_name, shard)
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model_name,
        shard,
        revision = revision,
        subfolder = subfolder,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        token = token,
    )


def _match_fp8_module(module_by_name, base):
    """Resolve a checkpoint module name to a live module, allowing for VLM key remappings.

    VLM loads can name the text tower differently from the checkpoint keys: `text_only=True`
    strips the `language_model.` wrapper (so `model.language_model.layers.*` -> `model.layers.*`),
    and full VLM loads may expose `model.language_model.*` while the checkpoint stores
    `language_model.model.*`. Try the raw key first, then a few safe remappings.
    """
    if base in module_by_name:
        return module_by_name[base]
    candidates = []
    if "language_model." in base:
        candidates.append(base.replace("language_model.", "", 1))  # text-only: drop wrapper
    if "language_model.model." in base:
        candidates.append(base.replace("language_model.model.", "model.language_model.", 1))
    if base.startswith("language_model."):
        candidates.append("model." + base)  # add model. prefix
    for candidate in candidates:
        if candidate in module_by_name:
            return module_by_name[candidate]
    return None


def _restore_dropped_fp8_scales(
    model,
    model_name,
    *,
    local_files_only = False,
    token = None,
    revision = None,
    subfolder = None,
    cache_dir = None,
    variant = None,
):
    """Re-apply block-fp8 `weight_scale_inv` tensors that transformers dropped on load.

    On some block-scale fp8 checkpoints (e.g. Qwen3.6-27B-FP8, issue #6200) transformers fails to
    convert a Linear (such as `mlp.gate_proj`) to an fp8 module, loading the raw quantized values
    into a plain bf16 weight and discarding its `weight_scale_inv` as an unexpected key. The weight
    is then used un-scaled, producing a garbage model. For every checkpoint scale whose live weight
    is not fp8, dequantize the orphaned weight in place. Modules that were converted correctly keep
    an fp8 weight and are skipped, so a healthy checkpoint is a no-op. Returns (restored, skipped).
    """
    try:
        block = _fp8_block_size_from_config(model)
        if block is None or not _FP8_DTYPES:
            return (0, 0)
        # A variant load reads variant-named files; skip to avoid applying default scales to them.
        if variant:
            return (0, 0)
        # No fp8 params means the checkpoint was dequantized on purpose (e.g. load_in_16bit);
        # re-applying a scale would corrupt those already-correct 16bit weights, so do nothing.
        if not any(p.dtype in _FP8_DTYPES for p in model.parameters()):
            return (0, 0)
        weight_map = _load_fp8_weight_map(
            model_name, local_files_only, token, revision, subfolder, cache_dir
        )
        if not weight_map:
            return (0, 0)

        scale_keys = {k: v for k, v in weight_map.items() if k.endswith(".weight_scale_inv")}
        if not scale_keys:
            return (0, 0)

        module_by_name = dict(model.named_modules())
        bs0, bs1 = block
        restored = 0
        skipped = 0
        failed = 0
        offloaded = 0
        shard_cache = {}
        for scale_key, shard in scale_keys.items():
            base = scale_key[: -len(".weight_scale_inv")]
            module = _match_fp8_module(module_by_name, base)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            if weight.device.type == "meta":
                # Disk-offloaded layer: weight lives on meta until forward, so it cannot be
                # scaled in place here. Count and warn rather than silently leave it unscaled.
                offloaded += 1
                continue
            if weight.dtype in _FP8_DTYPES:
                # Correctly converted fp8 module: the fp8 path already handles the scale.
                skipped += 1
                continue

            # Errors after this point are per-tensor: warn and continue, never abort or hide them.
            try:
                if shard not in shard_cache:
                    from safetensors import safe_open
                    shard_path = _resolve_fp8_shard(
                        model_name,
                        shard,
                        local_files_only,
                        token,
                        revision,
                        subfolder,
                        cache_dir,
                    )
                    shard_cache[shard] = safe_open(shard_path, framework = "pt")
                scale = shard_cache[shard].get_tensor(scale_key).to(torch.float32)

                out_features, in_features = weight.shape
                out_blocks = (out_features + bs0 - 1) // bs0
                in_blocks = (in_features + bs1 - 1) // bs1
                if tuple(scale.shape) == (out_blocks, in_blocks):
                    pass
                elif tuple(scale.shape) == (in_blocks, out_blocks) and out_blocks != in_blocks:
                    # Transposed block layout: same handling as the fp8 forward path.
                    scale = scale.t().contiguous()
                else:
                    # Shape does not match the block grid: skip rather than apply a wrong scale.
                    continue
                scale = scale.to(weight.device)
                with torch.no_grad():
                    if out_features % bs0 == 0 and in_features % bs1 == 0:
                        # Memory-frugal path: multiply block views in place against the broadcast
                        # fp32 scale, avoiding a full expanded scale and fp32 copy that could OOM.
                        # The in-place multiply promotes to fp32, matching the fallback exactly.
                        module.weight.data.view(out_blocks, bs0, in_blocks, bs1).mul_(
                            scale[:, None, :, None]
                        )
                    else:
                        scale_expanded = scale.repeat_interleave(bs0, dim = 0).repeat_interleave(
                            bs1, dim = 1
                        )[:out_features, :in_features]
                        module.weight.data = (weight.to(torch.float32) * scale_expanded).to(
                            weight.dtype
                        )
                restored += 1
            except Exception:
                failed += 1
                continue

        if restored > 0:
            print(f"Unsloth: Restored {restored} dropped FP8 weight_scale_inv tensor(s) on load")
        if failed > 0:
            print(f"Unsloth: {failed} dropped FP8 weight_scale_inv tensor(s) could not be restored")
        if offloaded > 0:
            print(
                f"Unsloth: {offloaded} dropped FP8 weight_scale_inv tensor(s) skipped because the "
                "layer is disk-offloaded; load without disk offload so the scales can be restored"
            )
        return (restored, skipped)
    except Exception:
        return (0, 0)


def check_and_disable_bitsandbytes_loading(
    model_config,
    load_in_4bit = True,
    load_in_8bit = False,
    verbose = True,
):
    """
    Check if we should disable bitsandbytes loading (load_in_4bit/load_in_8bit)
    because the model already has a non-bitsandbytes quantization config.
    If so, disable BOTH 4bit and 8bit loading and print a warning message.

    Args:
        model_config: The AutoConfig object from the model
        load_in_4bit: Whether load_in_4bit is currently enabled
        load_in_8bit: Whether load_in_8bit is currently enabled
        verbose: Whether to print warning messages

    Returns:
        tuple: (load_in_4bit, load_in_8bit, quant_method)
            load_in_4bit/load_in_8bit will be False if they were disabled
            quant_method is the detected quantization method or None
    """
    quant_method = get_quant_type(model_config)

    if quant_method is None or quant_method == "bitsandbytes":
        return load_in_4bit, load_in_8bit, quant_method

    # Model has a non-bitsandbytes quantization config (e.g., compressed-tensors, gptq, awq)
    # We should disable BOTH bitsandbytes loading to avoid config conflicts
    if load_in_4bit or load_in_8bit:
        if verbose:
            print(
                f"Unsloth: Model already quantized with {quant_method}. "
                f"Disabling `load_in_4bit` and `load_in_8bit` to avoid quantization config conflict."
            )
        load_in_4bit = False
        load_in_8bit = False

    return load_in_4bit, load_in_8bit, quant_method


def sync_unsloth_model_name_bnb_flags(load_in_4bit, load_in_8bit):
    """Make UNSLOTH_MODEL_NAME's `_load_in_4bit_`/`_load_in_8bit_` tokens match the EFFECTIVE bnb
    state (after get_model_name remap + check_and_disable). The per-load env is built from the
    pre-remap config (None for adapter-only PEFT repos), so its tokens can be wrong once the base
    resolves. Only the gpt-oss patch reads them, so this is gated to gpt-oss; no-op otherwise."""
    name = os.environ.get("UNSLOTH_MODEL_NAME", "")
    if "gpt_oss" not in name.replace("-", "_"):
        return
    for flag, present in (
        ("_load_in_4bit_", bool(load_in_4bit)),
        ("_load_in_8bit_", bool(load_in_8bit)),
    ):
        if present and flag not in name:
            name += flag
        elif not present and flag in name:
            name = name.replace(flag, "")
    os.environ["UNSLOTH_MODEL_NAME"] = name


def _get_fp8_mode_and_check_settings(
    load_in_fp8: Union[bool, str],
    fast_inference: bool,
    full_finetuning: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    load_in_16bit: bool = False,
) -> str:
    """Validate `load_in_fp8` settings/environment and return the fp8 mode
    ("row" or "block"). Requires H100+, torchao 0.15.0+, torch 2.9.0+, and
    fbgemm_gpu_genai 1.4.1+ if installed.
    """
    assert load_in_fp8 is not False
    if load_in_fp8 is True:
        fp8_mode = "row"  # default
    else:
        fp8_mode = load_in_fp8

    # Check user settings
    if fp8_mode not in ["row", "block"]:
        raise ValueError(f"Unsloth: `load_in_fp8` can only be 'row' or 'block', got '{fp8_mode}'")
    if full_finetuning:
        raise ValueError("Unsloth: `load_in_fp8` is not compatible with full finetuning")
    if load_in_4bit or load_in_8bit or load_in_16bit:
        raise ValueError(
            "Unsloth: `load_in_fp8` is not compatible with `load_in_4bit`, `load_in_8bit` or `load_in_16bit`",
        )

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

    # If fbgemm_gpu_genai is installed and old, disable FBGEMM and use Triton instead
    if (
        importlib.util.find_spec("fbgemm_gpu") is not None
        and importlib.util.find_spec("fbgemm_gpu.experimental") is not None
    ):
        import fbgemm_gpu.experimental.gen_ai
        if Version(fbgemm_gpu.__version__) < Version("1.4.1"):
            # Old FBGEMM version - disable and use Triton kernels instead
            os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
            from unsloth_zoo.log import logger
            logger.info(
                f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu.__version__} is old for FP8 loading. "
                f"Using Triton kernels instead."
            )
    return fp8_mode


# Rotary inv_freq buffers are deliberately kept on CPU - Unsloth pre-builds a
# cos/sin cache per GPU instead (see LlamaRotaryEmbedding.multi_gpu_cos_cached)
# so the GPU-resident lookup never needs to move the tiny inv_freq tensor itself.
# torch.nn.parallel.DistributedDataParallel ignores device entirely when it
# broadcasts buffers across ranks, so a CPU buffer crashes NCCL's
# _broadcast_coalesced with "No backend type associated with device type cpu".
# Telling DDP to skip these specific buffers avoids that crash without moving
# inv_freq to GPU (which would break the per-GPU cache design) and without
# disabling buffer broadcast for every other module (the user's workaround).
# Re-run this after wrapping with PEFT too - the buffers' fully qualified
# names change once they sit under a PeftModel (eg "base_model.model...").
# https://github.com/unslothai/unsloth/issues/6656
_ROTARY_INV_FREQ_BUFFER_NAMES = ("inv_freq", "short_inv_freq", "long_inv_freq")


def _exclude_rope_inv_freq_from_ddp(model):
    ignored = list(getattr(model, "_ddp_params_and_buffers_to_ignore", None) or [])
    for module_name, module in model.named_modules():
        for buffer_name, _ in module.named_buffers(recurse = False):
            if buffer_name in _ROTARY_INV_FREQ_BUFFER_NAMES:
                fqn = f"{module_name}.{buffer_name}" if module_name else buffer_name
                if fqn not in ignored:
                    ignored.append(fqn)
    if ignored:
        try:
            from torch.nn.parallel import DistributedDataParallel
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignored)
        except Exception:
            # Private PyTorch API - fall back to setting the attribute DDP reads
            # directly if it ever moves or changes signature.
            model._ddp_params_and_buffers_to_ignore = ignored
    return model


# =============================================================================
# Offline loading - single source of truth (shared by vision.py, loader.py and
# the Unsloth exporter). Decide offline ONCE at the load boundary and force it
# ONCE around the whole load, so every nested HF call inherits it.
# =============================================================================

_OFFLINE_ENV_VALUES = {"1", "true", "yes", "on"}
_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _env_says_offline():
    """True if an HF offline env var is set to a truthy value."""
    return any(
        os.environ.get(_k, "").strip().lower() in _OFFLINE_ENV_VALUES for _k in _OFFLINE_ENV_KEYS
    )


def _get_effective_local_files_only(kwargs):
    """Offline if local_files_only is truthy or an HF offline env var is set. Read-only."""
    if kwargs.get("local_files_only", None):
        return True
    return _env_says_offline()


def _is_offline_related_error(exc):
    """True if exc (or its cause/context chain) is a lost-connection error, not a
    missing file. Plain FileNotFoundError propagates; LocalEntryNotFoundError is offline."""
    import socket
    import ssl
    import urllib.error

    # Match network failures by type (locale independent), not just message wording.
    _net_types = [ConnectionError, TimeoutError, socket.gaierror, urllib.error.URLError]
    _offline_fnf_types = ()  # FileNotFoundError subclasses that count as offline
    # urllib HTTPError is a URLError subclass: judge by status (5xx offline, 4xx propagates).
    _http_types = (urllib.error.HTTPError,)
    # TLS/cert failures are security-sensitive (MITM, expired CA): never offline-retry them.
    _ssl_types = [ssl.SSLError]
    try:
        import requests

        _net_types += [requests.exceptions.ConnectionError, requests.exceptions.Timeout]
        _http_types += (requests.exceptions.HTTPError,)
        _ssl_types.append(requests.exceptions.SSLError)
    except Exception:
        pass
    try:
        from huggingface_hub.errors import (
            OfflineModeIsEnabled,
            HfHubHTTPError,
            LocalEntryNotFoundError,
        )

        _net_types += [OfflineModeIsEnabled, LocalEntryNotFoundError]
        _offline_fnf_types = (LocalEntryNotFoundError,)
        _http_types += (HfHubHTTPError,)
    except Exception:
        pass
    _net_types = tuple(_net_types)
    _ssl_types = tuple(_ssl_types)

    def _http_status(e):
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
        if code is None:
            code = getattr(e, "status_code", None)
        if code is None:
            code = getattr(e, "code", None)  # urllib.error.HTTPError uses .code
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    _wording = (
        "couldn't connect",
        "could not connect",
        "connection error",
        "connectionerror",
        "max retries",
        "offline",
        "timed out",
        "timeout",
        "couldn't reach",
        "could not reach",
        "failed to resolve",
        "getaddrinfo",
        "name resolution",
        "no address associated",
        "network is unreachable",
        "connection refused",
        "we couldn't connect to",
        "proxyerror",
        # Raw socket.gaierror DNS wording (Linux / macOS)
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
    )
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # TLS/cert failure (corporate MITM, expired CA): security-sensitive, never retry from
        # cache. Skip this node; a deeper cause in the chain may still be a genuine outage.
        if isinstance(cur, _ssl_types) or isinstance(getattr(cur, "reason", None), _ssl_types):
            cur = cur.__cause__ or cur.__context__
            continue
        is_fnf = isinstance(cur, FileNotFoundError) and not isinstance(cur, _offline_fnf_types)
        # urllib HTTPError is a URLError (net type) but must be judged by status code below,
        # unlike LocalEntryNotFoundError (an HfHubHTTPError that is always offline).
        if (
            isinstance(cur, _net_types)
            and not is_fnf
            and not isinstance(cur, urllib.error.HTTPError)
        ):
            return True
        if isinstance(cur, _http_types):
            code = _http_status(cur)
            if code is not None and 500 <= code < 600:
                return True
            # No status -> wording fallback (coded 4xx already decided above).
            if code is None and not is_fnf and any(w in str(cur).lower() for w in _wording):
                return True
        # OSError wording fallback (HTTP status already decided above).
        elif isinstance(cur, OSError) and not is_fnf:
            if any(w in str(cur).lower() for w in _wording):
                return True
        cur = cur.__cause__ or cur.__context__
    return False


# Process-wide HF offline state; the depth counter lets nested windows share one
# flip (first entrant saves originals, last exit restores). Lock guards flip/restore.
_force_offline_lock = _threading.RLock()
_force_offline_depth = 0
_force_offline_saved = []  # in-process module attributes
_force_offline_saved_env = {}  # HF offline env-var originals


def _reset_hf_sessions():
    """Clear hub's per-thread cached Sessions so the next rebuilds against the current
    offline flag. On hub 0.x the offline adapter is baked in at Session creation. Best-effort."""
    try:
        from huggingface_hub.utils._http import reset_sessions
    except Exception:
        try:
            from huggingface_hub.utils import reset_sessions
        except Exception:
            return
    try:
        reset_sessions()
    except Exception:
        pass


@contextlib.contextmanager
def _force_hf_offline():
    """Force HF offline for the window. local_files_only alone is not enough
    (transformers < 5 still pings /api/models), so set BOTH the env vars (cover
    subprocesses + raw urllib/requests) AND the in-process hub/transformers constants.
    Process-global; the refcount keeps restore correct under nesting / overlap."""
    global _force_offline_depth, _force_offline_saved, _force_offline_saved_env
    with _force_offline_lock:
        if _force_offline_depth == 0:
            saved = []
            saved_env = {}
            # Snapshot in-process constants BEFORE forcing the env: a module first imported
            # here would otherwise initialize its constant from the just-set "1" and we would
            # save (then restore) True, pinning the process offline after the window.
            try:
                import huggingface_hub.constants as _hfc
                if hasattr(_hfc, "HF_HUB_OFFLINE"):
                    saved.append((_hfc, "HF_HUB_OFFLINE", _hfc.HF_HUB_OFFLINE))
            except Exception:
                pass
            try:
                import transformers.utils.hub as _tuh
                for _attr in ("_is_offline_mode", "OFFLINE"):
                    if hasattr(_tuh, _attr):
                        saved.append((_tuh, _attr, getattr(_tuh, _attr)))
            except Exception:
                pass
            # Now force the env vars and flip the snapshotted constants to offline.
            for _k in _OFFLINE_ENV_KEYS:
                saved_env[_k] = os.environ.get(_k)
                os.environ[_k] = "1"
            for _obj, _attr, _ in saved:
                try:
                    setattr(_obj, _attr, True)
                except Exception:
                    pass
            _force_offline_saved = saved
            _force_offline_saved_env = saved_env
            # Rebuild cached sessions so they pick up the offline adapter.
            _reset_hf_sessions()
        _force_offline_depth += 1
    try:
        yield
    finally:
        with _force_offline_lock:
            _force_offline_depth -= 1
            if _force_offline_depth == 0:
                for obj, attr, val in _force_offline_saved:
                    try:
                        setattr(obj, attr, val)
                    except Exception:
                        pass
                _force_offline_saved = []
                for _k, _v in _force_offline_saved_env.items():
                    if _v is None:
                        os.environ.pop(_k, None)
                    else:
                        os.environ[_k] = _v
                _force_offline_saved_env = {}
                # Drop offline-mounted sessions so later online calls rebuild for the network.
                _reset_hf_sessions()


def _progress_bars_were_disabled():
    """Snapshot HF progress-bar state (None if unknown); pairs with _restore_progress_bars."""
    try:
        from huggingface_hub.utils import are_progress_bars_disabled
        return are_progress_bars_disabled()
    except Exception:
        return None


def _restore_progress_bars(were_disabled):
    """Re-enable HF progress bars only if a failed attempt left them disabled after they
    were enabled (a loader disables them around config probes and skips re-enabling on
    error). No-op if the user had them disabled or the state is unknown."""
    if were_disabled is False:
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except Exception:
            pass


def _offline_aware_load(fn):
    """Decide offline ONCE (local_files_only kwarg or env) and force it around the
    whole load. If we started online and hit a network error, retry once forced-offline.
    Network-up online path is unchanged: no window, no retry."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if _get_effective_local_files_only(kwargs):
            kwargs["local_files_only"] = True
            with _force_hf_offline():
                return fn(*args, **kwargs)
        _pb_were_disabled = _progress_bars_were_disabled()  # restore before any retry
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Skip if not network-related, or already retried by a nested decorator
            # (else outer layers reload the whole model again).
            if not _is_offline_related_error(e) or getattr(e, "_unsloth_offline_retried", False):
                raise
        # Retry OUTSIDE the except so the failed attempt's traceback (a partial model)
        # is freed before reallocating, else a large VLM can OOM on the second load.
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass
        # A failed attempt may have left HF progress bars disabled; restore before retry.
        _restore_progress_bars(_pb_were_disabled)
        kwargs["local_files_only"] = True
        try:
            with _force_hf_offline():
                return fn(*args, **kwargs)
        except Exception as e:
            # Tag so an enclosing _offline_aware_load skips its own redundant retry.
            try:
                e._unsloth_offline_retried = True
            except Exception:
                pass
            raise

    return _wrapper


def _has_local_tokenizer_files(path):
    """True if a local dir has a loadable tokenizer (BPE vocab.json needs merges.txt;
    special_tokens_map.json is not required)."""
    return (
        os.path.exists(os.path.join(path, "tokenizer.json"))
        or os.path.exists(os.path.join(path, "tokenizer.model"))
        or (
            os.path.exists(os.path.join(path, "vocab.json"))
            and os.path.exists(os.path.join(path, "merges.txt"))
        )
        or os.path.exists(os.path.join(path, "vocab.txt"))
        or os.path.exists(os.path.join(path, "spiece.model"))
    )


def _has_local_processor_files(path):
    """True if a local dir ships a processor/image-processor config (a VLM needs this to
    build AutoProcessor; tokenizer files alone are not enough)."""
    return os.path.exists(os.path.join(path, "processor_config.json")) or os.path.exists(
        os.path.join(path, "preprocessor_config.json")
    )


def _resolve_checkpoint_tokenizer_name(
    old_model_name,
    kwargs,
    require_processor = False,
):
    """tokenizer_name for a PEFT/checkpoint load: caller override, else the local checkpoint
    dir if self-sufficient, else None (base repo). Always popped from kwargs (also passed
    explicitly downstream). For a VLM (require_processor), the dir must also ship processor
    files; otherwise fall back to the base repo whose cached processor still loads."""
    explicit = kwargs.pop("tokenizer_name", None)
    if explicit is not None:
        return explicit
    has_config = os.path.exists(os.path.join(old_model_name, "tokenizer_config.json"))
    if not (has_config and _has_local_tokenizer_files(old_model_name)):
        return None
    if require_processor and not _has_local_processor_files(old_model_name):
        return None
    return old_model_name
