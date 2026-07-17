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

from unsloth_zoo.utils import Version
from importlib.metadata import version as importlib_version
from unsloth_zoo.hf_utils import dtype_from_config, HAS_TORCH_DTYPE
from unsloth_zoo.llama_cpp import (
    convert_to_gguf,
    quantize_gguf,
    use_local_gguf,
    install_llama_cpp,
    check_llama_cpp,
    _download_convert_hf_to_gguf,
)

# H4: Defensive imports -- these were added in unsloth-zoo PR #526
# and may not exist on older versions
try:
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR, IS_WINDOWS
except ImportError:
    import sys
    IS_WINDOWS = sys.platform == "win32"
    LLAMA_CPP_DEFAULT_DIR = "llama.cpp"
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from peft.tuners.lora import Linear as Peft_Linear
from typing import Optional, Callable, Union, List
import sys
import requests
import torch
import os
import json
import shutil
import pickle
import gc
import functools
from transformers.models.llama.modeling_llama import logger
from .kernels import fast_dequantize, QUANT_STATE, get_lora_parameters_bias
import subprocess
import psutil
import re
from transformers.models.llama.modeling_llama import logger
from .models.loader_utils import get_model_name
from .models._utils import _convert_torchao_model
from .ollama_template_mappers import OLLAMA_TEMPLATES, MODEL_TO_OLLAMA_TEMPLATE_MAPPER
from transformers import ProcessorMixin, PreTrainedTokenizerBase
from huggingface_hub import HfApi

try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        # For older versions of huggingface_hub
        from huggingface_hub.utils._token import get_token
from pathlib import Path
from peft import PeftModelForCausalLM, PeftModel

__all__ = [
    "print_quantization_methods",
    "unsloth_save_model",
    "save_to_gguf",
    "patch_saving_functions",
    "create_huggingface_repo",
]

# llama.cpp specific targets - all takes 90s. Below takes 60s
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-cli",
    "llama-server",
]

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
del keynames

# Weights
LLAMA_WEIGHTS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LLAMA_LAYERNORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)

# https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
# From https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html
ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "f16": "Float16  - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q2_k_l": "Q2_K_L with q8_0 output/token embeddings for higher quality than plain Q2_K.",
    "q3_k_l": "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m": "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "q3_k_xs": "3-bit extra small quantization",
}

# IQ (importance-matrix) quants. llama.cpp refuses these without an imatrix, so they are only
# accepted when imatrix_file=... is supplied to save_pretrained_gguf / push_to_hub_gguf.
IMATRIX_QUANTS = {
    "iq1_s": "1.56 bpw. Smallest, lowest quality. Needs an imatrix.",
    "iq1_m": "1.75 bpw. Very small. Needs an imatrix.",
    "iq2_xxs": "2.06 bpw. Needs an imatrix.",
    "iq2_xs": "2.31 bpw. Needs an imatrix.",
    "iq2_s": "2.5 bpw. Needs an imatrix.",
    "iq2_m": "2.7 bpw. Needs an imatrix.",
    "iq3_xxs": "3.06 bpw. Needs an imatrix.",
    "iq3_s": "3.44 bpw. Needs an imatrix.",
    "iq3_m": "3.66 bpw. Needs an imatrix.",
    "iq4_nl": "4.5 bpw non-linear. Benefits from an imatrix.",
    "iq4_xs": "4.25 bpw. Benefits from an imatrix.",
}


def has_curl():
    return shutil.which("curl") is not None


CURL_FLAG = "-DLLAMA_CURL=ON" if has_curl() else "-DLLAMA_CURL=OFF"


# FP8/FP4 compressed export via llm-compressor (for vLLM).
# save_method alias -> (llm-compressor scheme, needs_calibration, output dir suffix).
# alias -> (llm-compressor scheme, needs_calibration, output-dir suffix). needs_calibration is
# True only for schemes with static activation scales (FP8 static, NVFP4); everything else is
# weight-only or dynamic-activation and runs data-free. Unsupported schemes in the installed
# compressed-tensors (e.g. MXFP8 on older stacks) are gated by _scheme_is_available at runtime.
COMPRESSED_EXPORT_SCHEMES = {
    # FP8
    "fp8": ("FP8_DYNAMIC", False, "fp8"),
    "fp8_dynamic": ("FP8_DYNAMIC", False, "fp8"),
    "dynamic_fp8": ("FP8_DYNAMIC", False, "fp8"),
    "w8a8_fp8": ("FP8_DYNAMIC", False, "fp8"),
    "fp8_static": ("FP8", True, "fp8-static"),
    "static_fp8": ("FP8", True, "fp8-static"),
    "fp8_block": ("FP8_BLOCK", False, "fp8-block"),
    "block_fp8": ("FP8_BLOCK", False, "fp8-block"),
    # INT8 / INT-weight
    "int8": ("INT8", False, "int8"),
    "w8a8": ("W8A8", False, "w8a8"),
    "w8a8_int8": ("W8A8", False, "w8a8"),
    "w8a16": ("W8A16", False, "w8a16"),
    "int8_weight": ("W8A16", False, "w8a16"),
    "w4a16": ("W4A16", False, "w4a16"),
    "int4": ("W4A16", False, "w4a16"),
    "int4_weight": ("W4A16", False, "w4a16"),
    "w4a16_asym": ("W4A16_ASYM", False, "w4a16-asym"),
    "w4a8": ("W4A8", False, "w4a8"),
    "w4afp8": ("W4AFP8", False, "w4afp8"),
    # MXFP (microscaling)
    "mxfp8": ("MXFP8", False, "mxfp8"),
    "w8a8_mxfp8": ("MXFP8", False, "mxfp8"),
    "mxfp4": ("MXFP4", False, "mxfp4"),
    "w4a4_mxfp4": ("MXFP4", False, "mxfp4"),
    "mxfp4a16": ("MXFP4A16", False, "mxfp4a16"),
    "w4a16_mxfp4": ("MXFP4A16", False, "mxfp4a16"),
    # NVFP4
    "nvfp4": ("NVFP4", True, "nvfp4"),
    "w4a4_nvfp4": ("NVFP4", True, "nvfp4"),
    "nvfp4a16": ("NVFP4A16", False, "nvfp4a16"),
    "w4a16_nvfp4": ("NVFP4A16", False, "nvfp4a16"),
}


def _normalize_compressed_method(save_method):
    """Return (scheme, needs_calibration, suffix) if `save_method` is an FP8/FP4 compressed
    export, else None (so normal lora / merged_16bit / merged_4bit handling proceeds).

    Near-miss FP8/FP4 names that are not supported raise a precise error instead of silently
    falling through to the generic "unknown save_method" message.
    """
    if not isinstance(save_method, str):
        return None
    key = save_method.lower().strip().replace("-", "_").replace(" ", "_")
    if key in COMPRESSED_EXPORT_SCHEMES:
        return COMPRESSED_EXPORT_SCHEMES[key]
    if any(tag in key for tag in ("fp8", "fp4", "mxfp", "nvfp", "w4a", "w8a", "int4", "int8")):
        supported = ", ".join(sorted(COMPRESSED_EXPORT_SCHEMES.keys()))
        raise RuntimeError(
            f"Unsloth: save_method='{save_method}' is not a supported compressed export.\n"
            f"Supported compressed-tensors export methods: {supported}"
        )
    return None


def _is_cmake_only_llama_cpp(llama_cpp_dir: str = "llama.cpp") -> bool:
    """
    True if llama.cpp's Makefile is the post-CMake-migration deprecation stub,
    so `make` cannot build it. A genuinely missing/empty checkout returns False
    so it isn't treated as CMake-only: the caller then probes make and fails
    loudly on a real error rather than silently assuming a CMake build.
    """
    makefile_path = os.path.join(llama_cpp_dir, "Makefile")
    if not os.path.exists(makefile_path):
        # No Makefile: only CMake-only if a real CMake project is present
        return os.path.exists(os.path.join(llama_cpp_dir, "CMakeLists.txt"))
    try:
        with open(makefile_path, "r", encoding = "utf-8", errors = "ignore") as f:
            content = f.read(4096).lower()
            if "cmake" in content and "deprecated" in content:
                return True
            if "build system changed" in content:
                return True
    except (IOError, OSError):
        pass
    return False


def print_quantization_methods():
    for key, value in ALLOWED_QUANTS.items():
        print(f'"{key}"  ==> {value}')
    print("\nIQ low-bit quants (save_pretrained_gguf(..., imatrix_file=True or '...path')):")
    for key, value in IMATRIX_QUANTS.items():
        print(f'"{key}"  ==> {value}')
    print("\nCompressed-tensors export (save_pretrained_merged(..., save_method=...), for vLLM):")
    seen = set()
    for key, (scheme, needs_calib, _suffix) in COMPRESSED_EXPORT_SCHEMES.items():
        if scheme in seen:
            continue
        seen.add(scheme)
        note = "needs calibration data" if needs_calib else "data-free"
        print(f'"{key}"  ==> llm-compressor {scheme} ({note})')


def _quantize_q2_k_l(
    input_gguf: Union[str, os.PathLike],
    output_gguf: Union[str, os.PathLike],
    quantizer_location: Union[str, os.PathLike],
    n_threads: int,
    print_output: bool = True,
    imatrix = None,
):
    # "Q2_K_L" is an Unsloth preset, not a native llama.cpp ftype: q2_k with
    # output/token-embedding tensors kept at q8_0 for higher precision.
    command = [
        str(quantizer_location),
        *(["--imatrix", str(imatrix)] if imatrix else []),
        "--output-tensor-type",
        "q8_0",
        "--token-embedding-type",
        "q8_0",
        str(input_gguf),
        str(output_gguf),
        "q2_k",
        str(n_threads),
    ]

    if print_output:
        print(
            "Unsloth: Quantizing as Q2_K_L preset "
            "(q2_k + --output-tensor-type q8_0 --token-embedding-type q8_0)..."
        )

    try:
        if print_output:
            with subprocess.Popen(
                command,
                shell = False,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                bufsize = 1,
            ) as sp:
                assert sp.stdout is not None
                for line in sp.stdout:
                    print(line, end = "", flush = True)

                returncode = sp.wait()
                if returncode != 0:
                    raise RuntimeError(
                        f"Failed to quantize {input_gguf} to q2_k_l: process exited with code {returncode}"
                    )
        else:
            subprocess.run(
                command,
                shell = False,
                check = True,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
            )
    except subprocess.CalledProcessError as e:
        if print_output and hasattr(e, "stdout") and e.stdout:
            print(e.stdout)
        error_details = ""
        if hasattr(e, "stdout") and e.stdout:
            error_details += f"\nSubprocess stdout:\n{e.stdout}"
        if hasattr(e, "stderr") and e.stderr:
            error_details += f"\nSubprocess stderr:\n{e.stderr}"
        raise RuntimeError(f"Failed to quantize {input_gguf} to q2_k_l: {e}{error_details}")

    output_path = Path(output_gguf)
    if not output_path.exists():
        raise RuntimeError(f"Quantization failed - output file {output_gguf} not created")

    if print_output:
        file_size_bytes = output_path.stat().st_size
        file_size_gb = file_size_bytes / (1024**3)
        print(f"Unsloth: Successfully quantized to {output_gguf} (size: {file_size_gb:.2f}GB)")
    return str(output_gguf)


def check_if_sentencepiece_model(model, temporary_location = "_unsloth_sentencepiece_temp"):
    if not hasattr(model, "_saved_temp_tokenizer"):
        return False

    temp_tokenizer = model._saved_temp_tokenizer
    sentencepiece_model = False
    file_location = os.path.join(temporary_location, temp_tokenizer.name_or_path)
    created_folder = False
    if not os.path.exists(file_location):
        created_folder = True
        os.makedirs(file_location)
    temp_tokenizer.save_pretrained(file_location)
    if os.path.isfile(f"{file_location}/tokenizer.model"):
        sentencepiece_model = True
    if created_folder:
        shutil.rmtree(file_location, ignore_errors = True)
    return sentencepiece_model


_TOKENIZER_MODEL_CACHE = {}


def _has_tokenizer_model(tokenizer, token = None):
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    if tokenizer is None:
        return False

    source = getattr(tokenizer, "name_or_path", None)
    if not isinstance(source, str) or not source:
        return False
    if os.path.isdir(source):
        return os.path.isfile(os.path.join(source, "tokenizer.model"))
    if source in _TOKENIZER_MODEL_CACHE:
        return _TOKENIZER_MODEL_CACHE[source]

    try:
        repo_info = HfApi(token = token).model_info(source, files_metadata = False)
    except Exception:
        return False

    has_tokenizer_model = any(
        sibling.rfilename == "tokenizer.model" for sibling in (repo_info.siblings or [])
    )
    _TOKENIZER_MODEL_CACHE[source] = has_tokenizer_model
    return has_tokenizer_model


def _preserve_sentencepiece_tokenizer_assets(
    tokenizer,
    save_directory,
    token = None,
):
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    if tokenizer is None or not os.path.isdir(save_directory):
        return

    tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
    if os.path.isfile(tokenizer_config_path):
        desired_added_tokens_decoder = {}
        for token_id, added_token in getattr(tokenizer, "added_tokens_decoder", {}).items():
            desired_added_tokens_decoder[str(token_id)] = {
                "content": getattr(added_token, "content", str(added_token)),
                "single_word": getattr(added_token, "single_word", False),
                "lstrip": getattr(added_token, "lstrip", False),
                "rstrip": getattr(added_token, "rstrip", False),
                "normalized": getattr(added_token, "normalized", True),
                "special": getattr(added_token, "special", False),
            }
        if desired_added_tokens_decoder:
            with open(tokenizer_config_path, "r", encoding = "utf-8") as file:
                tokenizer_config = json.load(file)
            if tokenizer_config.get("added_tokens_decoder") != desired_added_tokens_decoder:
                tokenizer_config["added_tokens_decoder"] = desired_added_tokens_decoder
                with open(tokenizer_config_path, "w", encoding = "utf-8") as file:
                    json.dump(tokenizer_config, file, indent = 2, ensure_ascii = False)
                    file.write("\n")
                logger.warning_once(
                    f"Unsloth: Restored added_tokens_decoder metadata in "
                    f"{tokenizer_config_path}."
                )

    tokenizer_model = os.path.join(save_directory, "tokenizer.model")
    downloaded_path = None
    if not os.path.isfile(tokenizer_model) and _has_tokenizer_model(
        tokenizer,
        token = token,
    ):
        source = getattr(tokenizer, "name_or_path", None)
        if isinstance(source, str) and source:
            if os.path.isdir(source):
                local_path = os.path.join(source, "tokenizer.model")
                if os.path.isfile(local_path):
                    downloaded_path = local_path
            else:
                from huggingface_hub import hf_hub_download
                try:
                    downloaded_path = hf_hub_download(
                        repo_id = source,
                        filename = "tokenizer.model",
                        token = token,
                    )
                except Exception:
                    downloaded_path = None

    if not os.path.isfile(tokenizer_model) and downloaded_path is not None:
        shutil.copy2(downloaded_path, tokenizer_model)
        logger.warning_once(
            f"Unsloth: Preserved sentencepiece asset `tokenizer.model` in " f"{save_directory}."
        )


def _free_cached_model(model):
    from huggingface_hub import scan_cache_dir
    cached_repos = list(scan_cache_dir().repos)

    # Go through every cached repo, and delete the one that matches the model we want to save.
    # Can save 4GB of disk space - useful for Kaggle systems.
    for cached_repo in cached_repos:
        if cached_repo.repo_id == model.config._name_or_path:
            remove_cache_commit = list(cached_repo.revisions)[0].commit_hash
            delete_strategy = scan_cache_dir().delete_revisions(
                remove_cache_commit,
            )

            logger.warning_once(
                "Unsloth: Will remove a cached repo with size "
                + delete_strategy.expected_freed_size_str,
            )

            delete_strategy.execute()


def _merge_lora(layer, name):
    bias = getattr(layer, "bias", None)
    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
        # Is LoRA so we need to merge!
        W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)
        if quant_state is not None:
            dtype = quant_state.dtype if type(quant_state) is not list else quant_state[2]
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype
        W = W.to(torch.float32).t()
        # W = W.t()

        if A is not None:
            # sAB = (A.t().to(torch.float32) @ (s * B.t().to(torch.float32)))
            # W += sAB
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha = s)
            # W.addmm_(A.t().to(W.dtype), B.t().to(W.dtype), alpha = s)
            # if not torch.isfinite(W).all():
            maximum_element = torch.max(W.min().abs(), W.max())
            if not torch.isfinite(maximum_element).item():
                raise ValueError(f"Unsloth: Merge failed.\n{name} has some elements = infinity.")
        W = W.t().to(dtype)
    else:
        W = layer.weight
    return W, bias


def fast_save_pickle(shard, name):
    # Use this if # CPUs is <= 2
    print(f"Unsloth: Saving {name}...")
    torch.save(
        shard,
        name,
        # HIGHEST_PROTOCOL seems to not work with Pytorch!
        # pickle_module   = pickle,
        # pickle_protocol = pickle.HIGHEST_PROTOCOL,
    )
    return


def _preserve_tokenizer_eos_token(
    tokenizer,
    save_directory,
    filename_prefix = None,
):
    """Restore tokenizer_config.json eos_token from the tokenizer passed to save.

    Some merge paths may re-save or mutate tokenizer metadata after the tokenizer
    is written. Gemma 4 instruct models use `<turn|>` as their chat EOS token;
    if tokenizer_config.json is reset to the raw base `<eos>` token, runtimes such
    as vLLM will not stop generation correctly. Keep the serialized metadata in
    sync with the source tokenizer without failing the save if the config is not
    present or cannot be edited.

    `filename_prefix` mirrors the same argument on Transformers'
    `PreTrainedTokenizerBase.save_pretrained`: when provided, the tokenizer
    config is written as `{filename_prefix}-tokenizer_config.json` instead of
    `tokenizer_config.json`.
    """
    if tokenizer is None or save_directory is None:
        return

    source_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    eos_token = getattr(source_tokenizer, "eos_token", None)
    if eos_token is None and source_tokenizer is not tokenizer:
        eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        return
    eos_token = str(eos_token)

    tokenizer_config_name = (
        f"{filename_prefix}-tokenizer_config.json" if filename_prefix else "tokenizer_config.json"
    )
    tokenizer_config = os.path.join(str(save_directory), tokenizer_config_name)
    if not os.path.isfile(tokenizer_config):
        return

    try:
        with open(tokenizer_config, "r", encoding = "utf-8") as file:
            config = json.load(file)

        if config.get("eos_token") == eos_token:
            return

        config["eos_token"] = eos_token
        with open(tokenizer_config, "w", encoding = "utf-8") as file:
            json.dump(config, file, indent = 2, ensure_ascii = False)
            file.write("\n")
    except Exception as error:
        logger.warning_once(
            f"Unsloth: Could not preserve tokenizer eos_token in {tokenizer_config}: {error}"
        )


def _is_qwen3_5_vlm(model):
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "vision_config"):
        return False
    architectures = getattr(config, "architectures", None) or ()
    return any(
        architecture
        in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        )
        for architecture in architectures
    ) or getattr(config, "model_type", None) in ("qwen3_5", "qwen3_5_moe")


def _is_gpt_oss(model):
    config = getattr(model, "config", None)
    if config is None:
        return False
    architectures = getattr(config, "architectures", None) or ()
    return "GptOssForCausalLM" in architectures or getattr(config, "model_type", None) in (
        "gpt-oss",
        "gpt_oss",
    )


def _qwen3_5_vlm_state_dict_for_save(state_dict):
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("language_model.model."):
            new_key = "model.language_model." + key[len("language_model.model.") :]
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model.lm_head."):
            new_key = "lm_head." + key[len("language_model.lm_head.") :]
        else:
            new_key = key
        remapped_state_dict[new_key] = value
    return remapped_state_dict


def _coerce_tied_weights_keys_to_dict(model):
    """Coerce each module's legacy list/tuple/set ``_tied_weights_keys`` to dict form,
    returning ``[(module, original), ...]`` for the caller to restore.

    transformers >= 5 ``save_pretrained`` reads ``_tied_weights_keys.keys()``, so a model
    still declaring it as a list (e.g. NemotronH) crashes mid-save.
    """
    originals = []
    try:
        modules = list(model.modules())
    except Exception:
        return originals
    for module in modules:
        keys = getattr(module, "_tied_weights_keys", None)
        if isinstance(keys, (list, tuple, set)):
            try:
                module._tied_weights_keys = {k: k for k in keys}
                originals.append((module, keys))
            except Exception:
                pass
    return originals


def _restore_tied_weights_keys(originals):
    """Undo _coerce_tied_weights_keys_to_dict."""
    for module, keys in originals:
        try:
            module._tied_weights_keys = keys
        except Exception:
            pass


def _normalize_tied_weights_keys_for_save(save_fn):
    """Coerce legacy list-form ``_tied_weights_keys`` to dict for the duration of a save,
    then restore: transformers >= 5 re-ties from the dict's *values*, so a persisted
    ``{k: k}`` self-map would no-op a later resize/re-tie. ``model`` is the first positional
    arg (bound-method ``self``) or the ``model=`` keyword.
    """

    @functools.wraps(save_fn)
    def wrapper(*args, **kwargs):
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if model is None:
            model = kwargs.get("self")
        originals = _coerce_tied_weights_keys_to_dict(model) if model is not None else []
        try:
            return save_fn(*args, **kwargs)
        finally:
            _restore_tied_weights_keys(originals)

    return wrapper


@_normalize_tied_weights_keys_for_save
@torch.inference_mode
def unsloth_save_model(
    model,
    tokenizer,
    save_directory: Union[str, os.PathLike],
    save_method: str = "lora",  # ["lora", "merged_16bit", "merged_4bit"]
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    # Push to hub
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
    # Our functions
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
    datasets: Optional[List[str]] = None,
):
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    if token is None:
        token = get_token()

    if commit_message is None:
        commit_message = ""
    if "Unsloth" not in commit_message:
        commit_message += " (Trained with Unsloth)"
    commit_message = commit_message.lstrip()

    if commit_description is None:
        commit_description = "Upload model trained with Unsloth 2x faster"
    elif "Unsloth 2x faster" not in commit_description:
        commit_description += " (Trained with Unsloth 2x faster)"

    if save_method == "merged_4bit":
        raise RuntimeError(
            "Unsloth: Merging into 4bit will cause your model to lose accuracy if you plan\n"
            "to merge to GGUF or others later on. I suggest you to do this as a final step\n"
            "if you're planning to do multiple saves.\n"
            "If you are certain, change `save_method` to `merged_4bit_forced`."
        )
    elif save_method == "merged_4bit_forced":
        save_method = "merged_4bit"

    save_pretrained_settings = dict(locals())
    for deletion in (
        "model",
        "tokenizer",
        "save_method",
        "temporary_location",
        "maximum_memory_usage",
        "datasets",
    ):
        del save_pretrained_settings[deletion]

    # First check for a token!
    if push_to_hub:
        from huggingface_hub import whoami
        try:
            username = whoami(token = token)["name"]
        except:
            raise RuntimeError(
                "Unsloth: Please supply a token!\nGo to https://huggingface.co/settings/tokens"
            )

    assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95

    # Clean memory up first
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()

    save_method = save_method.lower().replace(" ", "_")
    if save_method != "lora" and save_method != "merged_16bit" and save_method != "merged_4bit":
        raise RuntimeError(
            "Unsloth: You must select one of 3 options when saving models:\n"
            '"lora"         ==> This is the fastest and easiet. Just saves LoRA modules.\n'
            '"merged_16bit" ==> This merges LoRA weights and saves to float16. Needed for llama.cpp / GGUF.\n'
            '"merged_4bit"  ==> This merges LoRA weights and saves to 4bit. Useful for DPO / inference.'
        )

    if save_method == "merged_4bit":
        print("Unsloth: Merging 4bit and LoRA weights to 4bit...")
        print("This might take 5 minutes...")

        # Counteract no LoRA adapters!
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        print("Done.")

    if tags is not None:
        assert isinstance(tags, (list, tuple))
        tags = list(tags) + [
            "unsloth",
        ]
    else:
        tags = [
            "unsloth",
        ]
    save_pretrained_settings["tags"] = tags

    if ((save_method == "lora") or (save_method == "merged_4bit")) and push_to_hub:
        if token is None:
            raise RuntimeError(
                "Unsloth: Pushing to HF requires a token. Pass `token = 'hf_....'`\n"
                "Go to https://huggingface.co/settings/tokens."
            )

        if save_method == "lora":
            print("Unsloth: Saving LoRA adapters. Please wait...")
        elif save_method == "merged_4bit":
            print("Unsloth: Saving 4bit Bitsandbytes model. Please wait...")

        # Update model tag
        _ = upload_to_huggingface(
            model,
            save_directory,
            token,
            "finetuned",
            "trl",
            file_location = None,
            old_username = None,
            private = private,
            datasets = datasets,
        )

        getattr(model, "original_push_to_hub", model.push_to_hub)(
            repo_id = save_directory,
            use_temp_dir = use_temp_dir,
            commit_message = commit_message,
            private = private,
            token = token,
            max_shard_size = max_shard_size,
            create_pr = create_pr,
            safe_serialization = safe_serialization,
            revision = revision,
            commit_description = commit_description,
            tags = tags,
        )
        if tokenizer is not None:
            # Set padding side to left for inference
            _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            old_padding_side = _tokenizer.padding_side
            _tokenizer.padding_side = "left"

            getattr(tokenizer, "original_push_to_hub", tokenizer.push_to_hub)(
                repo_id = save_directory,
                use_temp_dir = use_temp_dir,
                commit_message = commit_message,
                private = private,
                token = token,
                max_shard_size = max_shard_size,
                create_pr = create_pr,
                safe_serialization = safe_serialization,
                revision = revision,
                commit_description = commit_description,
                tags = tags,
            )

            # Revert back padding side
            _tokenizer.padding_side = old_padding_side

        if hasattr(model, "config"):
            print(f"Saved {save_method} model to https://huggingface.co/" + save_directory)
        return save_directory, None

    # Tokenizer has different saving arguments
    tokenizer_save_settings = {
        "save_directory": save_pretrained_settings["save_directory"],
        "legacy_format": None,
        "filename_prefix": None,
        "push_to_hub": save_pretrained_settings["push_to_hub"],
        "private": save_pretrained_settings["private"],
        "token": save_pretrained_settings["token"],
    }

    # Check if PEFT Model or not - if yes, 3 levels. If not 2 levels.
    from peft import PeftModelForCausalLM

    if isinstance(model, PeftModelForCausalLM):
        internal_model = model.model
    else:
        internal_model = model

    # Cannot be converted properly!
    if (
        (save_method == "merged_4bit")
        or (save_method == "lora")
        or (not hasattr(model, "model") or not hasattr(internal_model.model, "layers"))
    ):
        # Do general saving
        # Edit save_pretrained_settings
        # [TODO] _create_repo has errors due to **kwargs getting accepted
        # commit_description does not seem to work?
        what_to_delete = (
            (
                "use_temp_dir",
                "commit_message",
                "create_pr",
                "revision",
                "commit_description",
                "tags",
            )
            if save_pretrained_settings["push_to_hub"] is False
            else (
                "use_temp_dir",
                "create_pr",
                "revision",
                "tags",
                "commit_description",
            )
        )
        for deletion in what_to_delete:
            del save_pretrained_settings[deletion]
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(
                [
                    "unsloth",
                ]
            )

        # Update model tag
        if push_to_hub:
            _ = upload_to_huggingface(
                model,
                save_pretrained_settings["save_directory"],
                token,
                "finetuned",
                "trl",
                file_location = None,
                old_username = None,
                private = private,
                datasets = datasets,
            )

        if tokenizer is not None:
            print("Unsloth: Saving tokenizer...", end = "")

            # Set padding side to left for inference
            _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            old_padding_side = _tokenizer.padding_side
            _tokenizer.padding_side = "left"

            tokenizer.save_pretrained(**tokenizer_save_settings)

            # Revert back padding side
            _tokenizer.padding_side = old_padding_side

            print(" Done.")
        else:
            print()

        print("Unsloth: Saving model...", end = "")
        if save_method != "lora":
            print(" This might take 10 minutes for Llama-7b...", end = "")

        # [TODO] Is this correct?
        if save_method == "lora":
            save_pretrained_settings["selected_adapters"] = None

        model.save_pretrained(**save_pretrained_settings)

        if push_to_hub and hasattr(model, "config"):
            print("Saved to https://huggingface.co/" + save_pretrained_settings["save_directory"])

        print(" Done.")
        return save_directory, None

    # If push_to_hub, we must remove the .../ part of a repo
    username = None
    if push_to_hub and "/" in save_directory:
        # +1 solves absolute path issues
        new_save_directory = save_directory
        username = new_save_directory[: new_save_directory.find("/")]
        new_save_directory = new_save_directory[new_save_directory.find("/") + 1 :]
        if IS_KAGGLE_ENVIRONMENT:
            new_save_directory = os.path.join(
                KAGGLE_TMP, new_save_directory[new_save_directory.find("/") + 1 :]
            )
            logger.warning_once(
                "Unsloth: You are pushing to hub in Kaggle environment.\n"
                f"To save memory, we shall move {save_directory} to {new_save_directory}"
            )
        else:
            logger.warning_once(
                f"Unsloth: You are pushing to hub, but you passed your HF username = {username}.\n"
                f"We shall truncate {save_directory} to {new_save_directory}"
            )

        save_pretrained_settings["save_directory"] = new_save_directory
        tokenizer_save_settings["save_directory"] = new_save_directory
        save_directory = new_save_directory

    print("Unsloth: Merging 4bit and LoRA weights to 16bit...")

    # Determine max RAM usage minus sharding
    max_ram = psutil.virtual_memory().available
    sharded_ram_usage = 5 * 1024 * 1024 * 1024
    if type(max_shard_size) is str:
        gb_found = re.match(r"([0-9]{1,})[\s]{0,}GB", max_shard_size, flags = re.IGNORECASE)
        mb_found = re.match(r"([0-9]{1,})[\s]{0,}MB", max_shard_size, flags = re.IGNORECASE)
        if gb_found:
            sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024
        elif mb_found:
            sharded_ram_usage = int(mb_found.group(1)) * 1024 * 1024
    elif type(max_shard_size) is int:
        sharded_ram_usage = max_shard_size

    # Switch to our fast saving modules if it's a slow PC!
    n_cpus = psutil.cpu_count(logical = False)
    if n_cpus is None:
        n_cpus = psutil.cpu_count()
    if n_cpus is None:
        n_cpus = 1

    if safe_serialization is None:
        safe_serialization = True
        save_pretrained_settings["safe_serialization"] = safe_serialization

    elif safe_serialization and (n_cpus <= 2):
        logger.warning_once(
            f"Unsloth: You have {n_cpus} CPUs. Using `safe_serialization` is 10x slower.\n"
            f"We shall switch to Pytorch saving, which might take 3 minutes and not 30 minutes.\n"
            f"To force `safe_serialization`, set it to `None` instead.",
        )
        safe_serialization = False
        save_function = fast_save_pickle
        save_pretrained_settings["safe_serialization"] = safe_serialization
        save_pretrained_settings["save_function"] = save_function

    # Only safe_serialization uses more RAM
    if safe_serialization:
        max_ram -= sharded_ram_usage
    else:
        max_ram -= sharded_ram_usage * 0.25  # Uses much less

    max_ram = int(max(0, max_ram) * maximum_memory_usage)
    print(
        f"Unsloth: Will use up to "
        f"{round(max_ram/1024/1024/1024, 2)} out of "
        f"{round(psutil.virtual_memory().total/1024/1024/1024, 2)} RAM for saving."
    )

    # Move temporary_location to /tmp in Kaggle
    if IS_KAGGLE_ENVIRONMENT:
        temporary_location = os.path.join(KAGGLE_TMP, temporary_location)

    # Max directory for disk saving
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    # Check if Kaggle or Colab, since only 20GB of Disk space allowed.
    if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
        # We free up 4GB of space
        logger.warning_once(
            "Unsloth: Kaggle/Colab has limited disk space. We need to delete the downloaded\n"
            "model which will save 4-16GB of disk space, allowing you to save on Kaggle/Colab."
        )
        _free_cached_model(internal_model)

    # HF also uses a OrderedDict
    from collections import OrderedDict

    state_dict = OrderedDict()

    torch_dtype = dtype_from_config(internal_model.config)
    if type(torch_dtype) is str:
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16

    # Check modules to save float32 dtype
    state_dict["model.embed_tokens.weight"] = internal_model.model.embed_tokens.weight.data.to(
        torch_dtype
    )

    max_vram = int(torch.cuda.get_device_properties(0).total_memory * maximum_memory_usage)

    print("Unsloth: Saving model... This might take 5 minutes ...")

    from tqdm import tqdm as ProgressBar

    for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W, bias = _merge_lora(proj, name)

            # Bias term
            if bias is not None:
                state_dict[f"model.layers.{j}.{item}.bias"] = bias

            if (torch.cuda.memory_allocated() + W.nbytes) < max_vram:
                # Save to GPU memory
                state_dict[name] = W
            # [TODO] Saving to RAM seems to leak memory???
            # elif (max_ram - W.nbytes) > 0:
            #     # Save to CPU memory
            #     logger.warning_once(f"We will save to RAM and not VRAM now.")
            #     state_dict[name] = W.to("cpu", non_blocking = True, copy = True)
            #     max_ram = max(max_ram - W.nbytes, 0)
            else:
                # Save to Disk
                logger.warning_once("\nWe will save to Disk and not RAM now.")
                filename = os.path.join(temporary_location, f"{name}.pt")
                torch.save(
                    W,
                    filename,
                    pickle_module = pickle,
                    pickle_protocol = pickle.HIGHEST_PROTOCOL,
                )
                # weights_only = True weirdly fails?
                state_dict[name] = torch.load(
                    filename, map_location = "cpu", mmap = True, weights_only = False
                )
        for item in LLAMA_LAYERNORMS:
            try:
                # Skip for Gemma 2
                state_dict[f"model.layers.{j}.{item}.weight"] = eval(f"layer.{item}.weight.data")
            except:
                continue

    state_dict["model.norm.weight"] = internal_model.model.norm.weight.data
    # Check for modules_to_save float32 dtype

    # Check for tied weights
    if (
        internal_model.model.embed_tokens.weight.data_ptr()
        != internal_model.lm_head.weight.data_ptr()
    ):
        state_dict["lm_head.weight"] = internal_model.lm_head.weight.data.to(torch_dtype)

    # All tensors MUST be type torch.Tensor and not torch.nn.parameter.Parameter
    for key, value in state_dict.items():
        if hasattr(value, "data"):
            state_dict[key] = value = value.data
        if type(value) is not torch.Tensor:
            logger.warning_once(f"Unsloth: {key} is not a Tensor but a {type(value)}.")

    # Edit save_pretrained_settings
    # [TODO] _create_repo has errors due to **kwargs getting accepted
    save_pretrained_settings["state_dict"] = state_dict

    # commit_description does not seem to work?
    what_to_delete = (
        (
            "use_temp_dir",
            "commit_message",
            "create_pr",
            "revision",
            "commit_description",
            "tags",
        )
        if not push_to_hub
        else (
            "use_temp_dir",
            "create_pr",
            "revision",
            "tags",
            "commit_description",
        )
    )
    for deletion in what_to_delete:
        del save_pretrained_settings[deletion]
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(
            [
                "unsloth",
            ]
        )

    # Update model tag
    if push_to_hub:
        _ = upload_to_huggingface(
            model,
            save_pretrained_settings["save_directory"],
            token,
            "finetuned",
            "trl",
            file_location = None,
            old_username = username,
            private = private,
            datasets = datasets,
        )

    # First check if we're pushing to an organization!
    save_directory = save_pretrained_settings["save_directory"]

    if save_pretrained_settings["push_to_hub"]:
        new_save_directory, new_username = _determine_username(save_directory, username, token)

        if token is not None:
            from huggingface_hub import whoami
            actual_username = whoami(token = token)["name"]
        else:
            actual_username = username

    # Check if pushing to an organization
    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        # We upload everything at the end!
        tokenizer_save_settings["push_to_hub"] = False
        tokenizer_save_settings["save_directory"] = new_save_directory

    # Save tokenizer
    if tokenizer is not None:
        print("Unsloth: Saving tokenizer...", end = "")

        # Set padding side to left for inference
        _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        old_padding_side = _tokenizer.padding_side
        _tokenizer.padding_side = "left"

        tokenizer.save_pretrained(**tokenizer_save_settings)
        _preserve_tokenizer_eos_token(
            tokenizer,
            tokenizer_save_settings["save_directory"],
            filename_prefix = tokenizer_save_settings.get("filename_prefix"),
        )

        # Revert back padding side
        _tokenizer.padding_side = old_padding_side

        print(" Done.")
    else:
        print()

    # Since merged, edit quantization_config
    old_config = model.config
    new_config = model.config.to_dict()
    if "quantization_config" in new_config:
        del new_config["quantization_config"]
    original_model = model
    new_config = type(model.config).from_dict(new_config)
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = new_config
    model.config = new_config

    # Save!
    # [TODO] --> is this correct?
    # save_pretrained_settings["selected_adapters"] = None

    # Check if pushing to an organization
    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        # Pushing to organization: .save_pretrained doesn't work, so save
        # locally first then upload manually.
        save_pretrained_settings["save_directory"] = new_save_directory
        save_pretrained_settings["push_to_hub"] = False
        internal_model.save_pretrained(**save_pretrained_settings)

        # Now manually go through each file and upload them manually!
        filenames = os.listdir(new_save_directory)

        hf_api = HfApi(token = save_pretrained_settings["token"])

        print("Unsloth: Uploading all files... Please wait...")
        hf_api.upload_folder(
            folder_path = new_save_directory,
            path_in_repo = ".",
            repo_id = new_save_directory,
            repo_type = "model",
            commit_message = "(Trained with Unsloth)",
            ignore_patterns = "*.md",
        )
    else:
        internal_model.save_pretrained(**save_pretrained_settings)

    # Revert config back
    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = old_config
    model.config = old_config
    print("Done.")

    if push_to_hub and hasattr(model, "config"):
        print(
            f"Saved merged model to https://huggingface.co/{username}/{save_directory.lstrip('/').split('/')[-1]}"
        )

    save_pretrained_settings["state_dict"] = None

    for j, (key, value) in enumerate(state_dict.items()):
        state_dict[key] = None
        if j % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    state_dict = None
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

    # Remove temporary location

    shutil.rmtree(temporary_location, ignore_errors = True)

    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    return save_directory, username


def install_llama_cpp_clone_non_blocking():
    full_command = [
        "git",
        "clone",
        "--recursive",
        "https://github.com/ggerganov/llama.cpp",
    ]
    run_installer = subprocess.Popen(
        full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT
    )
    return run_installer


def install_llama_cpp_make_non_blocking():
    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # env = { **os.environ, "LLAMA_CUDA": "1", }

    # Skip the make-clean probe on CMake-only checkouts (its error output is misleading)
    IS_CMAKE = _is_cmake_only_llama_cpp("llama.cpp")

    if not IS_CMAKE:
        # Confirm make still works, silently
        try:
            result = subprocess.run(
                ["make", "clean", "-C", "llama.cpp"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
            IS_CMAKE = result.returncode != 0
        except FileNotFoundError:
            # No make executable; use CMake
            IS_CMAKE = True

    if not IS_CMAKE:
        # Uses old MAKE
        n_jobs = max(int((psutil.cpu_count() or 1) * 1.5), 1)
        full_command = ["make", "all", "-j" + str(n_jobs), "-C", "llama.cpp"]
    else:
        # Uses new CMAKE
        n_jobs = max(int(psutil.cpu_count() or 1), 1)  # Use less CPUs since 1.5x faster
        check = os.system(
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}"
        )

        if check != 0:
            raise RuntimeError(
                f"*** Unsloth: Failed compiling llama.cpp using os.system(...) with error {check}. Please report this ASAP!"
            )
        # f"cmake --build llama.cpp/build --config Release -j{psutil.cpu_count()*2} --clean-first --target {' '.join(LLAMA_CPP_TARGETS)}",
        full_command = [
            "cmake",
            "--build",
            "llama.cpp/build",
            "--config",
            "Release",
            "-j" + str(n_jobs),
            "--clean-first",
            "--target",
        ] + LLAMA_CPP_TARGETS
    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # run_installer = subprocess.Popen(full_command, env = env, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    run_installer = subprocess.Popen(
        full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT
    )
    return run_installer, IS_CMAKE


def install_python_non_blocking(packages = []):
    full_command = ["pip", "install"] + packages
    run_installer = subprocess.Popen(
        full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT
    )
    return run_installer


# Bound the first-use auto-install so no unvetted release is pulled: not an inflated "0.999.0", nor
# a crafted higher in-range patch like "0.12.999" from a mirror. Cap to the exact vetted patch and
# bump deliberately. Floor 0.6.0 keeps torch>=2.4 resolvable (0.7+ need torch>=2.7; torch pinned below).
_LLM_COMPRESSOR_SPEC = "llmcompressor>=0.6.0,<=0.12.0"


def install_llm_compressor():
    """Import llm-compressor, installing it on first use for FP8/FP4 export.

    Installs a version-pinned llm-compressor, pinning the current torch + transformers so pip does
    not upgrade them. Set UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL=1 to forbid the auto-install.
    Returns (oneshot, QuantizationModifier).
    """
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        return oneshot, QuantizationModifier
    except Exception:
        pass

    # Opt-out for locked-down / air-gapped setups: forbid the auto-install, require a manual one.
    if os.environ.get("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "0").lower() not in (
        "0",
        "",
        "false",
        "no",
    ):
        raise RuntimeError(
            "Unsloth: llm-compressor is required for FP8/FP4 compressed export but is not "
            "installed, and automatic installation is disabled via "
            "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL. Install it manually with:\n"
            f"    uv pip install --python {sys.executable} '{_LLM_COMPRESSOR_SPEC}'\n"
            "(pin torch and transformers to your current versions to avoid upgrading them)."
        )

    print(
        "Unsloth: Installing llm-compressor for FP8/FP4 export "
        f"({_LLM_COMPRESSOR_SPEC}; pinning your torch + transformers so they are not upgraded). "
        "This can take a few minutes..."
    )
    import importlib
    import tempfile

    constraints = ""
    try:
        import torch as _torch
        constraints += f"torch=={_torch.__version__.split('+')[0]}\n"
    except Exception:
        pass
    try:
        import transformers as _tf
        constraints += f"transformers=={_tf.__version__}\n"
    except Exception:
        pass

    # Prefer pip, but fall back to uv when this interpreter has no pip seeded (common in
    # uv-created / relocatable venvs), so the export does not hard-fail with "No module named pip".
    import importlib.util

    if importlib.util.find_spec("pip") is not None:
        cmd = [sys.executable, "-m", "pip", "install", _LLM_COMPRESSOR_SPEC]
    elif shutil.which("uv") is not None:
        cmd = ["uv", "pip", "install", "--python", sys.executable, _LLM_COMPRESSOR_SPEC]
    else:
        raise RuntimeError(
            "Unsloth: cannot install llm-compressor because this environment has neither pip nor "
            f"uv. Install it manually with:\n    uv pip install --python {sys.executable} '{_LLM_COMPRESSOR_SPEC}'\n"
            "(pin torch and transformers to your current versions to avoid upgrading them)."
        )
    cpath = None
    if constraints:
        with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as f:
            f.write(constraints)
            cpath = f.name
        cmd += ["-c", cpath]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Unsloth: Failed to install llm-compressor. Install it manually with:\n"
            f"    uv pip install --python {sys.executable} '{_LLM_COMPRESSOR_SPEC}'\n"
            f"or, if pip is available:\n    {sys.executable} -m pip install '{_LLM_COMPRESSOR_SPEC}'\n"
            "(pin torch and transformers to your current versions to avoid upgrading them).\n"
            f"Underlying error: {e}"
        )
    finally:
        if cpath is not None:
            try:
                os.remove(cpath)
            except Exception:
                pass

    importlib.invalidate_caches()
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception as e:
        raise RuntimeError(
            "Unsloth: llm-compressor was installed but could not be imported. "
            "Please restart your Python session and try again.\n"
            f"Underlying error: {repr(e)}"
        )
    return oneshot, QuantizationModifier


def try_execute(commands, force_complete = False):
    for command in commands:
        with subprocess.Popen(
            command,
            shell = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            bufsize = 1,
        ) as sp:
            for line in sp.stdout:
                line = line.decode("utf-8", errors = "replace")
                if "undefined reference" in line:
                    raise RuntimeError(
                        f"*** Unsloth: Failed compiling llama.cpp with {line}. Please report this ASAP!"
                    )
                elif "deprecated" in line:
                    return "CMAKE"
                elif "Unknown argument" in line:
                    raise RuntimeError(
                        f"*** Unsloth: Failed compiling llama.cpp with {line}. Please report this ASAP!"
                    )
                elif "***" in line:
                    raise RuntimeError(
                        f"*** Unsloth: Failed compiling llama.cpp with {line}. Please report this ASAP!"
                    )
                print(line, flush = True, end = "")
            if force_complete and sp.returncode is not None and sp.returncode != 0:
                raise subprocess.CalledProcessError(sp.returncode, sp.args)
    return None


def install_llama_cpp_old(version = -10):
    # Download the 10th latest release since the latest might be broken!
    # FALLBACK mechanism
    releases = subprocess.check_output(
        ["git", "ls-remote", "--tags", "https://github.com/ggerganov/llama.cpp.git"]
    )
    releases = releases.decode("utf-8").replace("\t", " ").split("\n")
    for i, x in enumerate(releases):
        if "refs/tags/b" not in x:
            break
    releases = releases[:i]
    latest = releases[-1]
    version = releases[version].split(" ")[0]

    # Check if the llama.cpp exists
    if os.path.exists("llama.cpp"):
        print(
            "**[WARNING]** You have a llama.cpp directory which is broken.\n"
            "Unsloth will DELETE the broken directory and install a new one.\n"
            "Press CTRL + C / cancel this if this is wrong. We shall wait 30 seconds.\n"
        )
        import time

        for i in range(30):
            print(f"**[WARNING]** Deleting llama.cpp directory... {30-i} seconds left.")
            time.sleep(1)

        shutil.rmtree("llama.cpp", ignore_errors = True)

    # Clone a specific commit
    # Also don't use the GPU!
    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        f"cd llama.cpp && git reset --hard {version} && git clean -df",
    ]
    try_execute(commands)

    # Detect CMake-only build system before trying make
    use_cmake = _is_cmake_only_llama_cpp("llama.cpp")

    if not use_cmake:
        # Try using MAKE
        commands = [
            "make clean -C llama.cpp",
            f"make all -j{(psutil.cpu_count() or 1)*2} -C llama.cpp",
        ]
        use_cmake = try_execute(commands) == "CMAKE"

    if use_cmake:
        # Use CMAKE
        commands = [
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}",
            f"cmake --build llama.cpp/build --config Release -j{(psutil.cpu_count() or 1)*2} --clean-first --target {' '.join(LLAMA_CPP_TARGETS)}",
            "cp llama.cpp/build/bin/llama-* llama.cpp",
            "rm -rf llama.cpp/build",
        ]
        try_execute(commands)

    # Check if successful
    if not (
        os.path.exists("llama.cpp/llama-quantize.exe")
        or os.path.exists("llama.cpp/llama-quantize")
        or os.path.exists("llama.cpp/quantize.exe")
        or os.path.exists("llama.cpp/quantize")
        or os.path.exists("llama.cpp/build/bin/llama-quantize")
        or os.path.exists("llama.cpp/build/bin/quantize")
    ):
        raise RuntimeError(
            "Unsloth: The file 'llama.cpp/llama-quantize' or `llama.cpp/quantize` does not exist.\n"
            "We've also double checked the building directory under 'llama.cpp/build/bin/'.\n"
            "But we expect this file to exist! Check if the file exists under llama.cpp and investigate the building process of llama.cpp (make/cmake)!"
        )


def install_llama_cpp_blocking(use_cuda = False):
    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # use_cuda = "LLAMA_CUDA=1" if use_cuda else ""

    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        "pip install gguf protobuf",
    ]
    if os.path.exists("llama.cpp"):
        return
    try_execute(commands)

    # Detect CMake-only build system before trying make
    use_cmake = _is_cmake_only_llama_cpp("llama.cpp")

    if not use_cmake:
        commands = [
            "make clean -C llama.cpp",
            # https://github.com/ggerganov/llama.cpp/issues/7062
            # Weirdly GPU conversion for GGUF breaks??
            # f"{use_cuda} make all -j{(psutil.cpu_count() or 1)*2} -C llama.cpp",
            f"make all -j{(psutil.cpu_count() or 1)*2} -C llama.cpp",
        ]
        use_cmake = try_execute(commands) == "CMAKE"

    if use_cmake:
        # Use CMAKE
        commands = [
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}",
            f"cmake --build llama.cpp/build --config Release -j{(psutil.cpu_count() or 1)*2} --clean-first --target {' '.join(LLAMA_CPP_TARGETS)}",
            "cp llama.cpp/build/bin/llama-* llama.cpp",
            "rm -rf llama.cpp/build",
        ]
        try_execute(commands)


def get_executable(executables):
    # Get system locations (System Path).split(system separator)
    system_directories = os.environ.get("PATH").split(os.pathsep)

    for directory in system_directories:
        for executable in executables:
            path = os.path.join(directory, executable)
            # Check if the executable exists and is executable
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
    return None


def save_to_gguf(
    model_name: str,
    model_type: str,
    model_dtype: str,
    is_sentencepiece: bool = False,
    model_directory: str = "unsloth_finetuned_model",
    quantization_method = "fast_quantized",  # Can be a list of options! ["q4_k_m", "q8_0", "q5_k_m"]
    first_conversion: str = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
    imatrix = None,
):
    """
    Orchestrates the complete GGUF conversion process.
    Handles installation, conversion, and quantization.
    `imatrix` is a local importance-matrix path (already resolved); it is forwarded to
    llama-quantize and is required for the IQ low-bit quant types.
    """
    # print_output True only if UNSLOTH_ENABLE_LOGGING=1
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print_output = True
    else:
        print_output = False

    # Validate model dtype
    assert model_dtype == "float16" or model_dtype == "bfloat16"
    model_dtype = "f16" if model_dtype == "float16" else "bf16"

    # Convert quantization_method to list
    if isinstance(quantization_method, list):
        pass
    elif isinstance(quantization_method, str):
        quantization_method = [
            quantization_method,
        ]
    elif isinstance(quantization_method, tuple):
        quantization_method = list(quantization_method)
    else:
        raise TypeError("Unsloth: quantization_method can only be a string or a list of strings")

    # Check if bfloat16 is supported
    if model_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "Unsloth: Cannot convert to bf16 GGUF since your computer doesn't support it.\n"
            "We shall switch instead to f16."
        )
        model_dtype = "f16"

    # Check first_conversion as well
    if first_conversion is None:
        first_conversion = model_dtype

    has_imatrix = imatrix is not None and str(imatrix) != ""
    if has_imatrix:
        # quantize_gguf gained the imatrix kwarg in a recent unsloth_zoo; fail fast (before the
        # expensive conversion) if the installed version cannot apply it, rather than dropping it.
        import inspect
        if "imatrix" not in inspect.signature(quantize_gguf).parameters:
            raise RuntimeError(
                "Unsloth: your installed unsloth_zoo's quantize_gguf does not support imatrix.\n"
                "Please upgrade it:  uv pip install --upgrade unsloth_zoo"
            )

    # Map quant methods
    new_quantization_methods = []
    for quant_method in quantization_method:
        if quant_method == "not_quantized":
            quant_method = model_dtype
        elif quant_method == "fast_quantized":
            quant_method = "q8_0"
        elif quant_method == "quantized":
            quant_method = "q4_k_m"
        elif quant_method is None:
            quant_method = "q8_0"

        # IQ low-bit quants are only valid with an imatrix; other methods use the normal allow-list.
        if quant_method in IMATRIX_QUANTS:
            if not has_imatrix:
                raise RuntimeError(
                    f"Unsloth: quant method '{quant_method}' is an IQ low-bit quant that requires an "
                    "importance matrix. Pass imatrix_file=True (to fetch the upstream Unsloth imatrix) "
                    "or imatrix_file='/path/to/imatrix' to save_pretrained_gguf / push_to_hub_gguf."
                )
        elif quant_method not in ALLOWED_QUANTS.keys():
            error = f"Unsloth: Quant method = [{quant_method}] not supported. Choose from below:\n"
            for key, value in ALLOWED_QUANTS.items():
                error += f"[{key}] => {value}\n"
            for key, value in IMATRIX_QUANTS.items():
                error += f"[{key}] => {value} (needs imatrix_file)\n"
            raise RuntimeError(error)

        new_quantization_methods.append(quant_method)
    quantization_method = new_quantization_methods

    # Determine optimal first_conversion
    if is_gpt_oss:
        print("Unsloth: GPT-OSS model detected - using special conversion settings")
        first_conversion = "None"  # No quantization for GPT-OSS
        # Only keep one conversion method since GPT-OSS doesn't quantize
        quantization_method = ["None"]
    else:
        if first_conversion is None:
            # Check if q8_0 is the ONLY quantization method requested
            if len(quantization_method) == 1 and quantization_method[0] == "q8_0":
                first_conversion = "None"  # Let llama-quantize do the direct conversion
            else:
                # For all other cases, choose the highest precision format
                # that can be requantized to all requested formats
                strength = 0
                for quant_method in quantization_method:
                    if quant_method == "f32":
                        strength = max(strength, 3)
                    elif quant_method == "f16":
                        strength = max(strength, 2)
                    elif quant_method == "bf16":
                        strength = max(strength, 1)
                    # Note: we don't set strength for q8_0 here since we handle it above

                if strength >= 3:
                    first_conversion = "f32"
                elif strength >= 2:
                    first_conversion = "f16"
                elif strength >= 1:
                    first_conversion = "bf16"
                else:
                    first_conversion = "bf16"  # requantizing from q8_0 disallowed in new llama.cpp default to bf16.

    # Check bfloat16 support again for first_conversion
    if first_conversion == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning("Unsloth: Switching bf16 to f16 due to hardware limitations")
        first_conversion = "f16"

    first_conversion_dtype = "" if first_conversion == "None" else first_conversion
    # Print conversion info
    print_info = (
        f"==((====))==  Unsloth: Conversion from HF to GGUF information\n"
        f"   {chr(92)}{chr(92)}   /|    [0] Installing llama.cpp might take 3 minutes.\n"
        f"O^O/ {chr(92)}_/ {chr(92)}    [1] Converting HF to GGUF {first_conversion_dtype} might take 3 minutes.\n"
        f"{chr(92)}        /    [2] Converting GGUF {first_conversion_dtype} to {quantization_method} might take 10 minutes each.\n"
        f' "-____-"     In total, you will have to wait at least 16 minutes.\n'
    )
    print(print_info)

    # Step 1: Ensure llama.cpp is installed
    try:
        quantizer_location, converter_location = check_llama_cpp()
        print("Unsloth: llama.cpp found in the system. Skipping installation.")
    except:
        print("Unsloth: Installing llama.cpp. This might take 3 minutes...")
        if IS_KAGGLE_ENVIRONMENT:
            # Kaggle: no CUDA support due to environment limitations
            quantizer_location, converter_location = install_llama_cpp(
                gpu_support = False, print_output = print_output
            )
        else:
            quantizer_location, converter_location = install_llama_cpp(
                gpu_support = False,  # GGUF conversion doesn't need CUDA
                print_output = print_output,
            )

    # Step 2: Download and patch converter script
    print("Unsloth: Preparing converter script...")
    with use_local_gguf():
        converter_path, supported_text_archs, supported_vision_archs = (
            _download_convert_hf_to_gguf()
        )

        # Step 3: Initial GGUF conversion
        print(f"Unsloth: [1] Converting model into {first_conversion_dtype} GGUF format.")
        print(f"This might take 3 minutes...")

        initial_files, is_vlm_update = convert_to_gguf(
            model_name = model_name,
            input_folder = model_directory,
            model_dtype = model_dtype,
            quantization_type = first_conversion,
            converter_location = converter_path,
            supported_text_archs = supported_text_archs,
            supported_vision_archs = supported_vision_archs,
            is_vlm = is_vlm,
            is_gpt_oss = is_gpt_oss,
            max_shard_size = "50GB",
            print_output = print_output,
        )
    # update is_vlm switch
    is_vlm = is_vlm_update
    # Check conversion success
    for file in initial_files:
        if not os.path.exists(file):
            if IS_KAGGLE_ENVIRONMENT:
                raise RuntimeError(
                    f"Unsloth: Conversion failed for {file}\n"
                    "You are in a Kaggle environment with limited disk space (20GB).\n"
                    "Try saving to /tmp for more space or use a smaller model.\n"
                    "Alternatively, save the 16bit model first, then convert manually."
                )
            else:
                raise RuntimeError(
                    f"Unsloth: Conversion failed for {file}\n"
                    "Please check disk space and try again."
                )

    # Move initial GGUF files into a dedicated _gguf directory
    gguf_directory = f"{model_directory}_gguf"
    os.makedirs(gguf_directory, exist_ok = True)
    moved_files = []
    for fpath in initial_files:
        dst = os.path.join(gguf_directory, os.path.basename(fpath))
        shutil.move(fpath, dst)
        moved_files.append(dst)
    initial_files = moved_files

    print(f"Unsloth: Initial conversion completed! Files: {initial_files}")

    # Step 4: Additional quantizations using llama-quantize
    all_saved_locations = initial_files.copy()

    # Get CPU count for quantization
    n_cpus = psutil.cpu_count()
    if n_cpus is None:
        n_cpus = 1
    n_cpus *= 2

    if not is_gpt_oss:
        base_gguf = initial_files[0]
        quants_created = False
        for quant_method in quantization_method:
            if quant_method != first_conversion:
                print(
                    f"Unsloth: [2] Converting GGUF {first_conversion_dtype} into {quant_method}. This might take 10 minutes..."
                )
                output_location = os.path.join(
                    gguf_directory, f"{model_name}.{quant_method.upper()}.gguf"
                )
                try:
                    if quant_method == "q2_k_l":
                        quantized_file = _quantize_q2_k_l(
                            input_gguf = base_gguf,
                            output_gguf = output_location,
                            quantizer_location = quantizer_location,
                            n_threads = n_cpus,
                            print_output = print_output,
                            imatrix = imatrix,
                        )
                    else:
                        # Use unsloth-zoo's standard quantization for all other methods. Only pass
                        # imatrix when set so older unsloth_zoo (no imatrix kwarg) still works for
                        # plain quants; an imatrix that cannot be applied was rejected above.
                        quant_kwargs = dict(
                            input_gguf = base_gguf,
                            output_gguf = output_location,
                            quant_type = quant_method,
                            quantizer_location = quantizer_location,
                            print_output = print_output,
                        )
                        if has_imatrix:
                            quant_kwargs["imatrix"] = imatrix
                        quantized_file = quantize_gguf(**quant_kwargs)
                    all_saved_locations.append(quantized_file)
                    quants_created = True
                except Exception as e:
                    if IS_KAGGLE_ENVIRONMENT:
                        raise RuntimeError(
                            f"Unsloth: Quantization failed for {output_location}\n"
                            "You are in a Kaggle environment, which might be the reason this is failing.\n"
                            "Kaggle only provides 20GB of disk space in the working directory.\n"
                            "Merging to 16bit for 7b models use 16GB of space.\n"
                            "This means using `model.{save_pretrained/push_to_hub}_merged` works, but\n"
                            "`model.{save_pretrained/push_to_hub}_gguf will use too much disk space.\n"
                            "You can try saving it to the `/tmp` directory for larger disk space.\n"
                            "I suggest you to save the 16bit model first, then use manual llama.cpp conversion.\n"
                            f"Error: {e}"
                        )
                    else:
                        if IS_WINDOWS:
                            build_instructions = (
                                f'cd "{LLAMA_CPP_DEFAULT_DIR}"\n'
                                f"cmake -S . -B build -DBUILD_SHARED_LIBS=OFF\n"
                                f"cmake --build build --config Release"
                            )
                        else:
                            build_instructions = (
                                f'cd "{LLAMA_CPP_DEFAULT_DIR}" && make clean && make all -j'
                            )

                        raise RuntimeError(
                            f"Unsloth: Quantization failed for {output_location}\n"
                            "You might have to compile llama.cpp yourself, then run this again.\n"
                            "You do not need to close this Python program. Run the following commands in a new terminal:\n"
                            f'git clone --recursive https://github.com/ggerganov/llama.cpp "{LLAMA_CPP_DEFAULT_DIR}"\n'
                            f"{build_instructions}\n"
                            "Once that's done, redo the quantization.\n"
                            f"Error: {e}"
                        )
        print("Unsloth: Model files cleanup...")
        want_full_precision = first_conversion in quantization_method
        if quants_created:
            # convert_to_gguf may return multiple base shards plus an mmproj entry,
            # so treat every initial file that is not an mmproj as part of the base set.
            base_files = [f for f in initial_files if "-mmproj" not in os.path.basename(f).lower()]
            if not want_full_precision:
                for f in base_files:
                    if f in all_saved_locations:
                        all_saved_locations.remove(f)
                    Path(f).unlink(missing_ok = True)

            # flip the list to get [text_model, mmproj] order. for text models stays the same.
            all_saved_locations.reverse()

            # When the base format is preserved, move base files (incl. shards) away from
            # list boundaries so example commands ([0]=model, [-1]=mmproj) stay correct.
            if want_full_precision and len(all_saved_locations) > len(base_files) + 1:
                for f in base_files:
                    if f in all_saved_locations:
                        all_saved_locations.remove(f)
                for i, f in enumerate(base_files):
                    all_saved_locations.insert(1 + i, f)
    else:
        print("Unsloth: GPT-OSS model - skipping additional quantizations")
        want_full_precision = True

    print(f"Unsloth: All GGUF conversions completed successfully!")
    print(f"Generated files: {all_saved_locations}")

    return all_saved_locations, want_full_precision, is_vlm


def unsloth_save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    save_method: str = "merged_16bit",  # ["lora", "merged_16bit", "merged_4bit"]
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
    datasets: Optional[List[str]] = None,
    calibration_dataset = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
):
    """
    Same as .save_pretrained(...) except 4bit weights are auto
    converted to float16 with as few overhead as possible.

    Choose for `save_method` to be either:
    1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
    2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
    3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    4.  FP8 / FP4 compressed export for vLLM (`fp8`, `mxfp4`, `nvfp4`, `mxfp8`): keeps the
        16bit merge at `save_directory` and writes the quantized checkpoint to
        `save_directory + "-<fmt>"`.
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"
            "You can do it separately via `tokenizer.save_pretrained(...)`"
        )

    # FP8 / FP4 compressed-tensors export (llm-compressor) -> handled separately.
    _compressed = _normalize_compressed_method(save_method)
    if _compressed is not None:
        scheme, needs_calibration, suffix = _compressed
        _unsloth_save_compressed_tensors(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            scheme = scheme,
            needs_calibration = needs_calibration,
            suffix = suffix,
            push_to_hub = push_to_hub,
            token = token,
            is_main_process = is_main_process,
            calibration_dataset = calibration_dataset,
            num_calibration_samples = num_calibration_samples,
            max_seq_length = max_seq_length,
            # Forward standard save kwargs to the 16bit merge.
            state_dict = state_dict,
            save_function = save_function,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            variant = variant,
            save_peft_format = save_peft_format,
            tags = tags,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            datasets = datasets,
        )
        for _ in range(3):
            gc.collect()
        return

    arguments = dict(locals())
    arguments["model"] = self
    del arguments["self"]
    del arguments["_compressed"]
    del arguments["calibration_dataset"]
    del arguments["num_calibration_samples"]
    del arguments["max_seq_length"]
    unsloth_save_model(**arguments)
    for _ in range(3):
        gc.collect()


def unsloth_push_to_hub_merged(
    self,
    repo_id: str,
    tokenizer = None,
    save_method: str = "merged_16bit",  # ["lora", "merged_16bit", "merged_4bit", "fp8", "mxfp4", "nvfp4", "mxfp8"]
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str, None] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
    datasets: Optional[List[str]] = None,
    calibration_dataset = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
):
    """
    Same as .push_to_hub(...) except 4bit weights are auto
    converted to float16 with as few overhead as possible.

    Choose for `save_method` to be either:
    1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
    2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
    3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    4.  FP8 / FP4 compressed export for vLLM: `fp8`, `mxfp4`, `nvfp4`, `mxfp8`.
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"
            "You can do it separately via `tokenizer.push_to_hub(...)`"
        )

    # FP8 / FP4 compressed-tensors export (llm-compressor) -> handled separately.
    _compressed = _normalize_compressed_method(save_method)
    if _compressed is not None:
        scheme, needs_calibration, suffix = _compressed
        _unsloth_save_compressed_tensors(
            model = self,
            save_directory = repo_id,
            tokenizer = tokenizer,
            scheme = scheme,
            needs_calibration = needs_calibration,
            suffix = suffix,
            push_to_hub = True,
            token = token,
            private = private,
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
            calibration_dataset = calibration_dataset,
            num_calibration_samples = num_calibration_samples,
            max_seq_length = max_seq_length,
            # Forward standard save kwargs to the 16bit merge.
            use_temp_dir = use_temp_dir,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            tags = tags,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            datasets = datasets,
        )
        for _ in range(3):
            gc.collect()
        return

    arguments = dict(locals())
    arguments["model"] = self
    arguments["save_directory"] = repo_id
    arguments["push_to_hub"] = True
    del arguments["self"]
    del arguments["repo_id"]
    del arguments["_compressed"]
    del arguments["calibration_dataset"]
    del arguments["num_calibration_samples"]
    del arguments["max_seq_length"]
    unsloth_save_model(**arguments)
    for _ in range(3):
        gc.collect()


MODEL_CARD = """---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""


def _determine_username(save_directory, old_username, token):
    username = ""
    save_directory = save_directory.lstrip("./")
    if "/" not in save_directory:
        from huggingface_hub import whoami
        try:
            username = whoami(token = token)["name"]
            if type(old_username) is str and username != old_username:
                username = old_username
            save_directory = f"{username}/{save_directory}"
        except:
            raise RuntimeError(f"Unsloth: {save_directory} is not a Huggingface directory.")
    else:
        username = save_directory.split("/")[0]
    return save_directory, username


def create_huggingface_repo(
    model,
    save_directory,
    token = None,
    private = False,
    datasets = None,
):
    if token is None:
        token = get_token()
    save_directory, username = _determine_username(save_directory, None, token)

    from huggingface_hub import create_repo

    try:
        create_repo(
            repo_id = save_directory,
            token = token,
            repo_type = "model",
            exist_ok = False,
            private = private,
        )

        # Create model card
        from huggingface_hub import ModelCard

        content = MODEL_CARD.format(
            username = username,
            base_model = model.config._name_or_path,
            model_type = model.config.model_type,
            method = "",
            extra = "unsloth",
        )
        card = ModelCard(content)
        if datasets:
            card.data.datasets = datasets
        card.push_to_hub(save_directory, token = token)
    except:
        # Repo already exists — update datasets metadata separately
        if datasets:
            try:
                from huggingface_hub import metadata_update
                metadata_update(save_directory, {"datasets": datasets}, overwrite = True, token = token)
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Could not update datasets metadata for {save_directory}: {e}"
                )
    hf_api = HfApi(token = token)
    return save_directory, hf_api


def upload_to_huggingface(
    model,
    save_directory,
    token,
    method,
    extra = "",
    file_location = None,
    old_username = None,
    private = None,
    create_config = True,
    datasets = None,
):
    save_directory, username = _determine_username(save_directory, old_username, token)

    from huggingface_hub import create_repo

    try:
        create_repo(
            repo_id = save_directory,
            token = token,
            repo_type = "model",
            exist_ok = False,
            private = private,
        )

        # Create model card
        from huggingface_hub import ModelCard

        content = MODEL_CARD.format(
            username = username,
            base_model = model.config._name_or_path,
            model_type = model.config.model_type,
            method = "",
            extra = extra,
        )
        card = ModelCard(content)
        if datasets:
            card.data.datasets = datasets
        card.push_to_hub(save_directory, token = token)
    except:
        # Repo already exists — update datasets metadata separately
        if datasets:
            try:
                from huggingface_hub import metadata_update
                metadata_update(save_directory, {"datasets": datasets}, overwrite = True, token = token)
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Could not update datasets metadata for {save_directory}: {e}"
                )

    if file_location is not None:
        # Now upload file
        hf_api = HfApi(token = token)

        if "/" in file_location:
            uploaded_location = file_location[file_location.rfind("/") + 1 :]
        else:
            uploaded_location = file_location

        # find ftevent file from tensorboard and upload it
        import glob

        ftevent_files = glob.glob("*out.tfevents*", recursive = True)
        if len(ftevent_files) > 0:
            print(
                "Unsloth: Uploading tensorboard files... Please wait...",
                file_location + "*out.tfevents*",
            )
            for ftevent_file in ftevent_files:
                hf_api.upload_file(
                    path_or_fileobj = ftevent_file,
                    path_in_repo = ftevent_file.replace(file_location, ""),
                    repo_id = save_directory,
                    repo_type = "model",
                    commit_message = "(Trained with Unsloth)",
                )

        hf_api.upload_file(
            path_or_fileobj = file_location,
            path_in_repo = uploaded_location,
            repo_id = save_directory,
            repo_type = "model",
            commit_message = "(Trained with Unsloth)",
        )

        # We also upload a config.json file
        if create_config:
            import json

            with open("_temporary_unsloth_config.json", "w", encoding = "utf-8") as file:
                json.dump({"model_type": model.config.model_type}, file, indent = 4)
            hf_api.upload_file(
                path_or_fileobj = "_temporary_unsloth_config.json",
                path_in_repo = "config.json",
                repo_id = save_directory,
                repo_type = "model",
                commit_message = "(Trained with Unsloth)",
            )
            os.remove("_temporary_unsloth_config.json")
    return username


def fix_tokenizer_bos_token(tokenizer):
    # Check if BOS added already, then warn
    fix_bos_token = False
    chat_template = getattr(tokenizer, "chat_template", None)

    if tokenizer("A").input_ids[0] == getattr(tokenizer, "bos_token_id", None):
        if chat_template is not None and (
            tokenizer.bos_token in chat_template
            or "{bos_token}" in chat_template.replace(" ", "")
            or "{bos_token+" in chat_template.replace(" ", "")
        ):
            fix_bos_token = True
            logger.warning(
                "Unsloth: ##### The current model auto adds a BOS token.\n"
                "Unsloth: ##### Your chat template has a BOS token. We shall remove it temporarily."
            )

            # Remove {{bos_token}}
            new_chat_template = re.sub(
                r"\{[\s]{0,}\{[\s]{0,}bos\_token[\s]{0,}\}[\s]{0,}\}", "", chat_template
            )
            # Remove {{bos_token +
            new_chat_template = re.sub(
                r"\{[\s]{0,}\{[\s]{0,}bos\_token[\s]{0,}\+[\s]{0,}",
                "",
                new_chat_template,
            )

            tokenizer.chat_template = new_chat_template

    return fix_bos_token, chat_template


def create_ollama_modelfile(tokenizer, base_model_name, model_location):
    """
    Creates an Ollama Modelfile.
    Use ollama.create(model = "new_ollama_model", modelfile = modelfile)
    """
    ollama_template_name = MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(base_model_name)
    if not ollama_template_name:
        print(
            f"Unsloth: No Ollama template mapping found for model '{base_model_name}'. Skipping Ollama Modelfile"
        )
        return None
    ollama_modelfile = OLLAMA_TEMPLATES.get(ollama_template_name)
    if not ollama_modelfile:
        print(
            f"Unsloth: No Ollama template mapping found for model '{base_model_name}'. Skipping Ollama Modelfile"
        )
        return None
    tokenizer._ollama_modelfile = ollama_modelfile  # This comes from the unpacking above
    modelfile = ollama_modelfile

    FILE_LOCATION_REPLACER = "⚫@✅#🦥__FILE_LOCATION__⚡@🦥#⛵"
    EOS_TOKEN_REPLACER = "⚫@✅#🦥__EOS_TOKEN__⚡@🦥#⛵"
    LEFT_BRACKET_REPLACER = "⚫@✅#🦥"
    RIGHT_BRACKET_REPLACER = "⚡@🦥#⛵"

    # Fixes https://github.com/unslothai/unsloth/issues/1087
    # We must convert all {'s and }'s but keep {__FILE_LOCATION__} intact
    modelfile = (
        modelfile.replace("{__FILE_LOCATION__}", FILE_LOCATION_REPLACER)
        .replace("{__EOS_TOKEN__}", EOS_TOKEN_REPLACER)
        .replace("{", LEFT_BRACKET_REPLACER)
        .replace("}", RIGHT_BRACKET_REPLACER)
    )

    # Revert {__FILE_LOCATION__} back
    modelfile = modelfile.replace(FILE_LOCATION_REPLACER, "{__FILE_LOCATION__}").replace(
        EOS_TOKEN_REPLACER, "{__EOS_TOKEN__}"
    )

    if "__EOS_TOKEN__" in modelfile:
        modelfile = modelfile.format(
            __FILE_LOCATION__ = model_location,
            __EOS_TOKEN__ = tokenizer.eos_token,
        )
    else:
        modelfile = modelfile.format(
            __FILE_LOCATION__ = model_location,
        )

    modelfile = modelfile.replace("⚫@✅#🦥", "{").replace("⚡@🦥#⛵", "}").rstrip()

    return modelfile


def create_ollama_model(username: str, model_name: str, tag: str, modelfile_path: str):
    try:
        init_check = subprocess.run(
            ["curl", "http://localhost:11434"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 3,
        )
        if init_check.returncode == 0:
            print(init_check.stdout.strip())
        else:
            print("Ollama Server is not Running")
    except subprocess.TimeoutExpired:
        return "Ollama Request Timeout"

    process = subprocess.Popen(
        [
            "ollama",
            "create",
            f"{username}/{model_name}:{tag}",
            "-f",
            f"{modelfile_path}",
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        bufsize = 1,
        universal_newlines = True,
        encoding = "utf-8",
        errors = "replace",
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end = "")
        sys.stdout.flush()

    return_code = process.wait()

    if return_code != 0:
        print(f"\nMODEL CREATED FAILED WITH RETURN CODE {return_code}")
    else:
        print("\nMODEL CREATED SUCCESSFULLY")


def push_to_ollama_hub(username: str, model_name: str, tag: str):
    try:
        init_check = subprocess.run(
            ["curl", "http://localhost:11434"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 3,
        )
        if init_check.returncode == 0:
            print(init_check.stdout.strip())
        else:
            print("Ollama Server is not Running")
    except subprocess.TimeoutExpired:
        return "Ollama Request Timeout"

    process = subprocess.Popen(
        ["ollama", "push", f"{username}/{model_name}:{tag}"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        bufsize = 1,
        universal_newlines = True,
        encoding = "utf-8",
        errors = "replace",
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end = "")
        sys.stdout.flush()

    return_code = process.wait()

    if return_code != 0:
        print(f"\nMODEL PUBLISHED FAILED WITH RETURN CODE {return_code}")
    else:
        print("\nMODEL PUBLISHED SUCCESSFULLY")


def push_to_ollama(tokenizer, gguf_location, username: str, model_name: str, tag: str):
    model_file = create_ollama_modelfile(tokenizer = tokenizer, gguf_location = gguf_location)

    with open(f"Modelfile_{model_name}", "w", encoding = "utf-8") as f:
        f.write(model_file)
        f.close()

    create_ollama_model(
        username = username,
        model_name = model_name,
        tag = tag,
        modelfile_path = f"Modelfile_{model_name}",
    )

    push_to_ollama_hub(username = username, model_name = model_name, tag = tag)

    print("Successfully pushed to ollama")


@_normalize_tied_weights_keys_for_save
def unsloth_save_pretrained_gguf(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    quantization_method = "fast_quantized",
    first_conversion: str = None,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    private: Optional[bool] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
    save_method: str = None,
    imatrix_file = None,
):
    """
    Same as .save_pretrained(...) except 4bit weights are auto
    converted to float16 then converted to GGUF / llama.cpp format.

    imatrix_file: importance matrix for llama-quantize. None = off; a path = use that file
    (a *.gguf_file is renamed to *.gguf); True = download the upstream unsloth/<base>-GGUF
    imatrix. Required for the IQ low-bit quants (iq2_xxs, iq4_xs, ...).

    Choose for `quantization_method` to be:
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q2_k_l"  : "Q2_K_L with --output-tensor-type q8_0 --token-embedding-type q8_0.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
    """
    if tokenizer is None:
        raise ValueError("Unsloth: Saving to GGUF must have a tokenizer.")
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    # save_method="lora" exports the adapter itself as a GGUF LoRA (not a merged model).
    if save_method is not None and str(save_method).lower() == "lora":
        if not is_main_process:
            return None
        if push_to_hub:
            raise ValueError(
                "Unsloth: Please use .push_to_hub_gguf(save_method='lora') instead of "
                ".save_pretrained_gguf(save_method='lora', push_to_hub=True)."
            )
        _qm = quantization_method
        if isinstance(_qm, (list, tuple)) and len(_qm) == 1:
            _qm = _qm[0]  # the gguf API allows a list; unwrap a single outtype
        if _qm in _LORA_GGUF_OUTTYPES:
            _outtype = _qm
        else:
            if _qm not in (None, "fast_quantized"):
                logger.warning_once(
                    f"Unsloth: LoRA GGUF export does not support "
                    f"quantization_method={quantization_method!r}; using outtype 'f16'. "
                    f"Valid LoRA outtypes: {_LORA_GGUF_OUTTYPES}."
                )
            _outtype = "f16"
        return _unsloth_save_lora_gguf(self, tokenizer, save_directory, outtype = _outtype)

    try:
        base_model_name = get_model_name(self.config._name_or_path, load_in_4bit = False)
        model_name = base_model_name.split("/")[-1]
    except:
        base_model_name = self.config._name_or_path
        model_name = base_model_name.split("/")[-1]

    # Check if push_to_hub is requested
    if push_to_hub:
        raise ValueError(
            "Unsloth: Please use .push_to_hub_gguf() instead of .save_pretrained_gguf() with push_to_hub=True"
        )

    # Step 1: Check if this is a VLM (Vision-Language Model) and check if gpt-oss
    is_vlm = False
    if hasattr(self, "config") and hasattr(self.config, "architectures"):
        is_vlm = any(
            x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
            for x in self.config.architectures
        )
        is_vlm = is_vlm or hasattr(self.config, "vision_config")

    is_processor = is_vlm and isinstance(tokenizer, ProcessorMixin)

    is_gpt_oss = _is_gpt_oss(self)
    # Step 2: Prepare arguments for model saving
    arguments = dict(locals())
    arguments["model"] = self
    arguments["tokenizer"] = tokenizer
    arguments["push_to_hub"] = False  # We handle upload ourselves
    # GPT-OSS needs mxfp4 save method
    if is_gpt_oss:
        if quantization_method is not None:
            _qm = (
                quantization_method
                if isinstance(quantization_method, (list, tuple))
                else [quantization_method]
            )
            _ignored = [q for q in _qm if str(q).lower() != "mxfp4"]
            if _ignored:
                logger.warning_once(
                    f"Unsloth: GPT-OSS does not support GGUF quantization "
                    f"(requested: {', '.join(str(q) for q in _ignored)}). "
                    f"Overriding to MXFP4 format. "
                    f"Pass quantization_method=None to suppress this warning."
                )
        arguments["save_method"] = "mxfp4"
    else:
        arguments["save_method"] = "merged_16bit"
    del arguments["self"]
    del arguments["quantization_method"]
    del arguments["first_conversion"]
    del arguments["is_vlm"]
    del arguments["is_gpt_oss"]
    del arguments["model_name"]
    del arguments["base_model_name"]
    del arguments["is_processor"]
    del arguments["imatrix_file"]  # only used by the gguf quantize step, not the 16bit merge

    # Step 3: Fix tokenizer BOS token if needed
    if is_processor:
        fix_bos_token, old_chat_template = fix_tokenizer_bos_token(tokenizer.tokenizer)
    else:
        fix_bos_token, old_chat_template = fix_tokenizer_bos_token(tokenizer)

    # Resolve the importance matrix (download upstream / validate path / rename *.gguf_file) up
    # front, so a bad path or an unavailable upstream imatrix fails before the expensive 16-bit
    # merge, and a failed auto-resolution never reaches the IQ-quant gate.
    imatrix_path = _resolve_imatrix_file(self, imatrix_file, token, save_directory)

    # Step 4: Save/merge model to 16-bit format
    is_peft_model = isinstance(self, PeftModelForCausalLM) or isinstance(self, PeftModel)

    if is_peft_model:
        print(f'Unsloth: Merging model weights to {"mxfp4" if is_gpt_oss else "16-bit"} format...')
        try:
            # Call unsloth_generic_save directly (it's in the same file)
            unsloth_generic_save(**arguments)

        except Exception as e:
            raise RuntimeError(f"Failed to save/merge model: {e}")
    else:
        # Non-PEFT model: checkpoint files already exist; point save_to_gguf
        # at the original path instead of re-saving to a temp subdir.
        original_path = getattr(self.config, "_name_or_path", None)
        if original_path and os.path.isdir(original_path):
            print(
                f"Unsloth: Model is not a PEFT model. Using existing checkpoint at {original_path}"
            )
            save_directory = original_path
            # Persist tokenizer fixes (e.g. BOS token stripping) to disk
            # so the GGUF converter picks up the corrected chat template.
            if tokenizer is not None:
                tokenizer.save_pretrained(save_directory)
        else:
            # Fallback: save the in-memory model to save_directory
            print("Unsloth: Model is not a PEFT model. Saving directly without LoRA merge...")
            os.makedirs(save_directory, exist_ok = True)
            try:
                self.save_pretrained(save_directory)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_directory)
            except Exception as e:
                raise RuntimeError(f"Failed to save model: {e}")

    if is_processor:
        tokenizer = tokenizer.tokenizer

    # Use old chat template if the bos is removed
    if fix_bos_token:
        tokenizer.chat_template = old_chat_template

    # Step 6: Clean up memory
    for _ in range(3):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 7: Get model dtype and type
    try:
        model_dtype = dtype_from_config(self.config)
        model_type = self.config.model_type
        if type(model_dtype) is str:
            assert model_dtype == "float16" or model_dtype == "bfloat16"
        elif model_dtype == torch.float16:
            model_dtype = "float16"
        elif model_dtype == torch.bfloat16:
            model_dtype = "bfloat16"
        else:
            raise TypeError("Unsloth: Model dtype can only be float16 or bfloat16")
    except Exception as e:
        # Fallback if dtype_from_config fails
        print(f"Unsloth: Could not determine dtype ({e}), defaulting to float16")
        model_dtype = "float16"

    # Step 8: Convert to GGUF format
    print("Unsloth: Converting to GGUF format...")

    # Convert quantization_method to list if string
    # Use old style quantization_method
    quantization_methods = []
    if quantization_method is not None:
        # Convert quantization_method to list
        if isinstance(quantization_method, list):
            pass
        elif isinstance(quantization_method, str):
            quantization_method = [
                quantization_method,
            ]
        elif isinstance(quantization_method, tuple):
            quantization_method = list(quantization_method)
        else:
            raise TypeError(
                "Unsloth: quantization_method can only be a string or a list of strings"
            )
        for i, quant_method in enumerate(quantization_method):
            quant_method = quant_method.lower()
            if quant_method == "not_quantized":
                quant_method = "f16"
            elif quant_method == "fast_quantized":
                quant_method = "q8_0"
            elif quant_method == "quantized":
                quant_method = "q4_k_m"
            elif quant_method is None:
                quant_method = "q8_0"
            quantization_methods.append(quant_method.lower())

    try:
        from .tokenizer_utils import fix_sentencepiece_gguf
        fix_sentencepiece_gguf(save_directory)
    except Exception as e:
        logger.warning(f"Unsloth: fix_sentencepiece_gguf skipped ({type(e).__name__}): {e}")

    try:
        all_file_locations, want_full_precision, is_vlm_update = save_to_gguf(
            model_name = model_name,
            model_type = model_type,
            model_dtype = model_dtype,
            is_sentencepiece = False,
            model_directory = save_directory,
            quantization_method = quantization_methods,
            first_conversion = first_conversion,
            is_vlm = is_vlm,  # Pass VLM flag
            is_gpt_oss = is_gpt_oss,  # Pass gpt_oss Flag
            imatrix = imatrix_path,
        )
    except Exception as e:
        if IS_KAGGLE_ENVIRONMENT:
            raise RuntimeError(
                f"Unsloth: GGUF conversion failed in Kaggle environment.\n"
                f"This is likely due to the 20GB disk space limit.\n"
                f"Try saving to /tmp directory or use a smaller model.\n"
                f"Error: {e}"
            )
        else:
            raise RuntimeError(f"Unsloth: GGUF conversion failed: {e}")

    # Step 9: Create Ollama modelfile
    gguf_directory = f"{save_directory}_gguf"
    modelfile_location = None
    ollama_success = False
    if all_file_locations:
        try:
            if is_vlm_update:
                modelfile = create_ollama_modelfile(tokenizer, base_model_name, ".")
            else:
                modelfile = create_ollama_modelfile(
                    tokenizer,
                    base_model_name,
                    os.path.basename(all_file_locations[0]),
                )
            if modelfile is not None:
                modelfile_location = os.path.join(gguf_directory, "Modelfile")
                with open(modelfile_location, "w", encoding = "utf-8") as file:
                    file.write(modelfile)
                ollama_success = True
        except Exception as e:
            print(f"Warning: Could not create Ollama modelfile: {e}")

    # Step 10: Show BOS token warning if applicable
    if fix_bos_token:
        logger.warning(
            "Unsloth: ##### The current model auto adds a BOS token.\n"
            "Unsloth: ##### We removed it in GGUF's chat template for you."
        )

    _exe = ".exe" if IS_WINDOWS else ""
    if IS_WINDOWS:
        _bin_dir = os.path.join(LLAMA_CPP_DEFAULT_DIR, "build", "bin", "Release")
    else:
        _bin_dir = LLAMA_CPP_DEFAULT_DIR

    if is_vlm_update:
        print("\n")
        print(
            f"Unsloth: example usage for Multimodal LLMs: {os.path.join(_bin_dir, 'llama-mtmd-cli' + _exe)} -m {all_file_locations[0]} --mmproj {all_file_locations[-1]}"
        )
        print("Unsloth: load image inside llama.cpp runner: /image test_image.jpg")
        print("Unsloth: Prompt model to describe the image")
    else:
        print(
            f'Unsloth: example usage for text only LLMs: {os.path.join(_bin_dir, "llama-cli" + _exe)} --model {all_file_locations[0]} -p "why is the sky blue?"'
        )

    if ollama_success:
        print(f"Unsloth: Saved Ollama Modelfile to {modelfile_location}")
        print(
            f"Unsloth: convert model to ollama format by running - ollama create model_name -f {modelfile_location}"
        )

    # Return a dict with all needed info for push_to_hub
    return {
        "save_directory": save_directory,
        "gguf_directory": gguf_directory,
        "gguf_files": all_file_locations,
        "modelfile_location": modelfile_location,
        "want_full_precision": want_full_precision,
        "is_vlm": is_vlm_update,
        "fix_bos_token": fix_bos_token,
    }


def unsloth_push_to_hub_gguf(
    self,
    repo_id: str,
    tokenizer = None,
    quantization_method = "fast_quantized",
    first_conversion: str = None,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str, None] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
    datasets: Optional[List[str]] = None,
    save_method: str = None,
    imatrix_file = None,
):
    """
    Same as .push_to_hub(...) except 4bit weights are auto
    converted to float16 then converted to GGUF / llama.cpp format.

    imatrix_file: importance matrix for llama-quantize (None = off; a path; or True to download
    the upstream unsloth/<base>-GGUF imatrix). Required for the IQ low-bit quants.

    Choose for `quantization_method` to be:
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q2_k_l"  : "Q2_K_L with --output-tensor-type q8_0 --token-embedding-type q8_0.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    """
    if tokenizer is None:
        raise ValueError("Unsloth: Saving to GGUF must have a tokenizer.")

    # save_method="lora" exports the adapter itself as a GGUF LoRA (not a merged model).
    if save_method is not None and str(save_method).lower() == "lora":
        if not is_main_process:
            return None  # only the main rank converts and uploads, like the local lora branch
        _qm = quantization_method
        if isinstance(_qm, (list, tuple)) and len(_qm) == 1:
            _qm = _qm[0]  # the gguf API allows a list; unwrap a single outtype
        if _qm in _LORA_GGUF_OUTTYPES:
            _outtype = _qm
        else:
            if _qm not in (None, "fast_quantized"):
                logger.warning_once(
                    f"Unsloth: LoRA GGUF export does not support "
                    f"quantization_method={quantization_method!r}; using outtype 'f16'. "
                    f"Valid LoRA outtypes: {_LORA_GGUF_OUTTYPES}."
                )
            _outtype = "f16"
        return _unsloth_save_lora_gguf(
            self,
            tokenizer,
            repo_id,
            outtype = _outtype,
            push_to_hub = True,
            token = token,
            private = private,
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
        )

    # Step 1: Determine save directory
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    if use_temp_dir or use_temp_dir is None:
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix = "unsloth_gguf_")
        save_directory = temp_dir
        cleanup_temp = True
    else:
        save_directory = model_name  # Use model name, not repo_id
        cleanup_temp = False

    # Step 2: Call save_pretrained_gguf to do the conversion
    print(f"Unsloth: Converting model to GGUF format...")

    try:
        # Call save_pretrained_gguf - it returns all the info we need
        result = unsloth_save_pretrained_gguf(
            self = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            quantization_method = quantization_method,
            first_conversion = first_conversion,
            push_to_hub = False,  # Never push from here
            token = token,  # forwarded so imatrix_file=True can read a gated/private upstream
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            imatrix_file = imatrix_file,
        )

        # Extract results
        all_file_locations = result["gguf_files"]
        modelfile_location = result["modelfile_location"]
        want_full_precision = result["want_full_precision"]
        is_vlm = result["is_vlm"]
        fix_bos_token = result["fix_bos_token"]
        actual_save_directory = result["save_directory"]

    except Exception as e:
        if cleanup_temp:
            for d in [save_directory, f"{save_directory}_gguf"]:
                try:
                    shutil.rmtree(d)
                except:
                    pass
        raise RuntimeError(f"Failed to convert model to GGUF: {e}")

    # Step 3: Upload to HuggingFace Hub
    print("Unsloth: Uploading GGUF to Huggingface Hub...")

    try:
        from huggingface_hub import HfApi

        api = HfApi(token = token)

        # Get full repo id
        if "/" not in repo_id:
            username = api.whoami()["name"]
            full_repo_id = f"{username}/{repo_id}"
        else:
            full_repo_id = repo_id

        # Create repo
        api.create_repo(
            repo_id = full_repo_id,
            repo_type = "model",
            private = private,
            exist_ok = True,
        )

        # Upload GGUF files
        for file_location in all_file_locations:
            original_name = os.path.basename(file_location)
            # Replace temp directory name with proper model name
            if cleanup_temp and "unsloth_gguf_" in original_name:
                # Extract the quantization part (e.g., ".Q8_0.gguf" or ".Q8_0-mmproj.gguf")
                quant_suffix = (
                    original_name.split(".", 1)[1] if "." in original_name else original_name
                )
                proper_name = f"{model_name}.{quant_suffix}"
            else:
                proper_name = original_name.replace(os.path.basename(save_directory), model_name)

            print(f"Uploading {proper_name}...")

            api.upload_file(
                path_or_fileobj = file_location,
                path_in_repo = proper_name,
                repo_id = full_repo_id,
                repo_type = "model",
                commit_message = commit_message,
                commit_description = commit_description,
                create_pr = create_pr,
                revision = revision,
            )

        # Upload config.json if exists
        config_path = os.path.join(actual_save_directory, "config.json")
        if os.path.exists(config_path):
            print("Uploading config.json...")
            api.upload_file(
                path_or_fileobj = config_path,
                path_in_repo = "config.json",
                repo_id = full_repo_id,
                repo_type = "model",
                commit_message = f"{commit_message} - config",
                create_pr = create_pr,
                revision = revision,
            )

        # Upload Modelfile if exists
        if modelfile_location and os.path.exists(modelfile_location):
            print("Uploading Ollama Modelfile...")
            api.upload_file(
                path_or_fileobj = modelfile_location,
                path_in_repo = "Modelfile",
                repo_id = full_repo_id,
                repo_type = "model",
                commit_message = f"{commit_message} - Ollama Modelfile",
                create_pr = create_pr,
                revision = revision,
            )

        # Create and upload README
        readme_content = f"""---
tags:
- gguf
- llama.cpp
- unsloth
{"- vision-language-model" if is_vlm else ""}
---

# {repo_id.split("/")[-1]} : GGUF

This model was finetuned and converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth).

**Example usage**:
- For text only LLMs:    `llama-cli -hf {repo_id} --jinja`
- For multimodal models: `llama-mtmd-cli -hf {repo_id} --jinja`

## Available Model files:
"""
        for file in all_file_locations:
            # Fix filename in README too
            original_name = os.path.basename(file)
            if cleanup_temp and "unsloth_gguf_" in original_name:
                quant_suffix = (
                    original_name.split(".", 1)[1] if "." in original_name else original_name
                )
                proper_name = f"{model_name}.{quant_suffix}"
            else:
                proper_name = original_name.replace(os.path.basename(save_directory), model_name)
            readme_content += f"- `{proper_name}`\n"

        # Special note for VLM with Modelfile
        if is_vlm and modelfile_location:
            readme_content += "\n## ⚠️ Ollama Note for Vision Models\n"
            readme_content += "**Important:** Ollama currently does not support separate mmproj files for vision models.\n\n"
            readme_content += "To create an Ollama model from this vision model:\n"
            readme_content += "1. Place the `Modelfile` in the same directory as the finetuned bf16 merged model\n"
            readme_content += "3. Run: `ollama create model_name -f ./Modelfile`\n"
            readme_content += "   (Replace `model_name` with your desired name)\n\n"
            readme_content += "This will create a unified bf16 model that Ollama can use.\n"
        elif modelfile_location:
            readme_content += "\n## Ollama\n"
            readme_content += "An Ollama Modelfile is included for easy deployment.\n"

        if fix_bos_token:
            readme_content += "\n## Note\n"
            readme_content += (
                "The model's BOS token behavior was adjusted for GGUF compatibility.\n"
            )

        readme_content += (
            "This was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)\n"
            '[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)\n'
        )

        readme_path = os.path.join(actual_save_directory, "README.md")
        with open(readme_path, "w", encoding = "utf-8") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj = readme_path,
            path_in_repo = "README.md",
            repo_id = full_repo_id,
            repo_type = "model",
            commit_message = "Add README",
            create_pr = create_pr,
            revision = revision,
        )

        print(f"Unsloth: Successfully uploaded GGUF to https://huggingface.co/{full_repo_id}")

        # Add tags
        if tags is None:
            tags = []
        tags.extend(["gguf", "llama-cpp", "unsloth"])
        if is_vlm:
            tags.append("vision-language-model")

        try:
            api.add_tags(
                repo_id = full_repo_id,
                tags = tags,
                repo_type = "model",
            )
        except:
            pass

        if datasets:
            try:
                from huggingface_hub import metadata_update
                metadata_update(full_repo_id, {"datasets": datasets}, overwrite = True, token = token)
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Could not update datasets metadata for {full_repo_id}: {e}"
                )

    except Exception as e:
        raise RuntimeError(f"Failed to upload to Hugging Face Hub: {e}")

    finally:
        # Clean up temporary directory
        if cleanup_temp:
            print("Unsloth: Cleaning up temporary files...")
            for d in [save_directory, f"{save_directory}_gguf"]:
                if os.path.exists(d):
                    try:
                        shutil.rmtree(d)
                    except:
                        pass

    return full_repo_id


# Corrected function to save LoRA to a custom directory
def save_lora_to_custom_dir(model, tokenizer, save_directory):
    # Create the custom directory if it doesn't exist
    os.makedirs(save_directory, exist_ok = True)

    # Call the unsloth_save_model function with the custom directory
    unsloth_save_model(
        model,
        tokenizer,
        save_directory = save_directory,
        save_method = "lora",
        push_to_hub = False,
    )


# Valid output float types for llama.cpp's convert_lora_to_gguf.py.
_LORA_GGUF_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "auto")


def _lora_base_model_id(model):
    """Base model id for a PEFT model: prefer the active adapter's recorded base, else the
    model config (the adapter's `base_model_name_or_path` is the authoritative source)."""
    base = None
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        adapter = getattr(model, "active_adapter", None)
        if callable(adapter):
            try:
                adapter = adapter()
            except Exception:
                adapter = None
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0] if adapter else None
        cfg = (
            peft_config.get(adapter) if adapter in peft_config else next(iter(peft_config.values()))
        )
        base = getattr(cfg, "base_model_name_or_path", None)
    if not base:
        base = getattr(getattr(model, "config", None), "_name_or_path", None)
    return os.fspath(base) if base else ""


# Upstream Unsloth GGUF repos ship a calibration imatrix under one of these names; the GGUF-format
# one is suffixed .gguf_file so the Hub does not list it as a model GGUF (renamed to .gguf locally).
_IMATRIX_UPSTREAM_NAMES = ("imatrix_unsloth.dat", "imatrix_unsloth.gguf_file")


def _gguf_repo_candidates(model):
    """Ordered, de-duplicated unsloth/<base>-GGUF repo ids to search for an upstream imatrix."""
    candidates = []
    raw_names = [
        _lora_base_model_id(model),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ]
    for raw in raw_names:
        if not raw:
            continue
        name = os.fspath(raw)
        if os.path.isdir(name):
            continue  # a local checkpoint has no upstream GGUF repo
        try:
            name = get_model_name(name, load_in_4bit = False)
        except Exception:
            pass
        if not name:
            continue
        # The upstream imatrix lives in unsloth/<base>-GGUF, so map any org (e.g. meta-llama/...)
        # onto the unsloth org; keep an already-formed -GGUF id as-is.
        repo = name if name.endswith("-GGUF") else f"unsloth/{name.split('/')[-1]}-GGUF"
        if repo not in candidates:
            candidates.append(repo)
    return candidates


def _materialize_imatrix(path, dest_dir):
    """Copy an imatrix into dest_dir (never mutate the HF cache) and rename *.gguf_file -> *.gguf."""
    os.makedirs(dest_dir, exist_ok = True)
    base = os.path.basename(path)
    if base.endswith(".gguf_file"):
        base = base[: -len(".gguf_file")] + ".gguf"
    local = os.path.join(dest_dir, base)
    shutil.copyfile(path, local)
    return local


def _resolve_imatrix_file(model, imatrix_file, token, dest_dir):
    """Turn the public imatrix_file value into a local imatrix path (or None).

    None/False -> None. A path -> that file (a *.gguf_file is renamed to *.gguf). True -> find and
    download the upstream unsloth/<base>-GGUF imatrix, raising a clear error if none exists.
    """
    if imatrix_file is None or imatrix_file is False:
        return None

    if imatrix_file is not True and isinstance(imatrix_file, (str, os.PathLike)):
        path = os.path.expanduser(os.fspath(imatrix_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Unsloth: imatrix_file '{path}' does not exist.")
        return _materialize_imatrix(path, dest_dir) if path.endswith(".gguf_file") else path

    if imatrix_file is not True:
        raise TypeError(
            "Unsloth: imatrix_file must be None, a path string, or True "
            f"(got {type(imatrix_file).__name__})."
        )

    # imatrix_file=True: auto-resolve from the upstream Unsloth GGUF repo. HfApi is the module-level
    # import (save.py top); hf_hub_download is imported here as it is not needed elsewhere.
    from huggingface_hub import hf_hub_download

    if token is None:
        token = get_token()
    api = HfApi(token = token)
    repos = _gguf_repo_candidates(model)
    for repo in repos:
        try:
            files = set(api.list_repo_files(repo))
        except Exception:
            continue
        for name in _IMATRIX_UPSTREAM_NAMES:
            if name in files:
                downloaded = hf_hub_download(repo_id = repo, filename = name, token = token)
                local = _materialize_imatrix(downloaded, dest_dir)
                print(f"Unsloth: Using imatrix '{name}' from '{repo}' -> '{local}'")
                return local
    raise RuntimeError(
        "Unsloth: imatrix_file=True but no upstream Unsloth imatrix was found.\n"
        f"  Searched repos: {repos or '(none derived from the base model)'}\n"
        f"  Searched files: {list(_IMATRIX_UPSTREAM_NAMES)}\n"
        "Pass imatrix_file='/path/to/imatrix.(dat|gguf)' to use your own."
    )


def _unsloth_save_lora_gguf(
    model,
    tokenizer,
    save_directory,
    outtype = "f16",
    push_to_hub = False,
    token = None,
    private = None,
    commit_message = "Converted LoRA to GGUF with Unsloth",
    commit_description = "Convert LoRA to GGUF format using Unsloth",
    create_pr = False,
    revision = None,
):
    """Export a PEFT/LoRA adapter straight to a GGUF LoRA file via llama.cpp's
    convert_lora_to_gguf.py (loadable with `llama-cli --lora ...`). For a full / merged model
    use save_pretrained_gguf instead. `save_directory` is a local dir, or a Hub repo id when
    push_to_hub=True. Returns the local .gguf path, or the repo id when pushing."""
    import tempfile

    if not isinstance(model, (PeftModelForCausalLM, PeftModel)):
        raise RuntimeError(
            "Unsloth: LoRA GGUF export needs a PEFT/LoRA model. "
            "For a full or merged model use save_pretrained_gguf(...) instead."
        )
    if outtype not in _LORA_GGUF_OUTTYPES:
        raise ValueError(
            f"Unsloth: LoRA GGUF outtype must be one of {_LORA_GGUF_OUTTYPES} (got '{outtype}')."
        )
    # Resolve a token even for local saves: the converter may fetch a gated/private base config.
    if token is None:
        token = get_token()

    # Resolve the dequantized base id (the adapter usually references a 4bit repo).
    base_model_id = _lora_base_model_id(model)
    if not base_model_id:
        raise RuntimeError(
            "Unsloth: could not determine the base model for LoRA GGUF export "
            "(no adapter base_model_name_or_path or model config _name_or_path)."
        )
    try:
        base_model_id = get_model_name(base_model_id, load_in_4bit = False)
    except Exception:
        pass
    # Windows-safe basename (handles both C:\... and / separators).
    if os.path.isdir(base_model_id):
        model_name = os.path.basename(os.path.normpath(base_model_id))
    else:
        model_name = base_model_id.replace("\\", "/").rstrip("/").split("/")[-1]
    if not model_name:
        model_name = "model"

    # Save the adapter; for a hub push use an isolated temp dir, else save_directory itself.
    if push_to_hub:
        lora_dir = tempfile.mkdtemp(prefix = "unsloth-lora-gguf-")
    else:
        os.makedirs(save_directory, exist_ok = True)
        lora_dir = save_directory

    # Wrap so the isolated temp dir used for hub pushes is always cleaned up, even on failure.
    try:
        save_lora_to_custom_dir(model, tokenizer, lora_dir)

        # Ensure a full llama.cpp checkout (ships convert_lora_to_gguf.py) and locate the converter.
        install_llama_cpp(just_clone_repo = True)
        converter = os.path.join(LLAMA_CPP_DEFAULT_DIR, "convert_lora_to_gguf.py")
        if not os.path.exists(converter):
            # A prebuilt llama.cpp install (or a reused CWD copy) carries binaries but not the
            # converter script, so force a dedicated source checkout that ships it.
            source_dir = os.path.join(
                os.path.dirname(os.path.normpath(LLAMA_CPP_DEFAULT_DIR)), "llama.cpp-source"
            )
            install_llama_cpp(llama_cpp_folder = source_dir, just_clone_repo = True)
            converter = os.path.join(source_dir, "convert_lora_to_gguf.py")
        if not os.path.exists(converter):
            raise RuntimeError(
                "Unsloth: convert_lora_to_gguf.py not found after installing a llama.cpp source "
                "checkout. A full llama.cpp source checkout is required for LoRA GGUF export."
            )

        out_gguf = os.path.join(lora_dir, f"{model_name}-lora-{outtype}.gguf")
        cmd = [sys.executable, converter, lora_dir, "--outfile", out_gguf, "--outtype", outtype]
        # A local base dir provides config directly; otherwise the id is resolved from the Hub.
        if os.path.isdir(base_model_id):
            cmd += ["--base", base_model_id]
        else:
            cmd += ["--base-model-id", base_model_id]
        if bool(getattr(model.config, "auto_map", None)):
            cmd.append("--trust-remote-code")

        # Expose the token to the converter so it can fetch a gated/private base config from the Hub.
        env = os.environ.copy()
        if isinstance(token, str) and token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

        print(f"Unsloth: Converting LoRA adapter at '{lora_dir}' to GGUF -> '{out_gguf}'")
        try:
            with subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                bufsize = 1,
                universal_newlines = True,
                encoding = "utf-8",
                errors = "replace",
                env = env,
            ) as sp:
                for line in sp.stdout:
                    print(line, end = "", flush = True)
                sp.wait()
                if sp.returncode != 0:
                    raise subprocess.CalledProcessError(sp.returncode, sp.args)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Unsloth: LoRA -> GGUF conversion failed (exit {e.returncode}). "
                "See the output above for details."
            )

        if not push_to_hub:
            print(f"Unsloth: Done. Saved LoRA GGUF to '{out_gguf}'")
            return out_gguf

        print(f"Unsloth: Uploading LoRA GGUF to '{save_directory}' ...")
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        api.create_repo(
            repo_id = save_directory,
            repo_type = "model",
            private = private,
            exist_ok = True,
        )
        api.upload_folder(
            folder_path = lora_dir,
            repo_id = save_directory,
            repo_type = "model",
            allow_patterns = ["*.gguf"],
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
        )
        print(f"Unsloth: Done. Uploaded to https://huggingface.co/{save_directory.lstrip('/')}")
        return save_directory
    finally:
        if push_to_hub:
            shutil.rmtree(lora_dir, ignore_errors = True)


def unsloth_convert_lora_to_ggml_and_push_to_hub(
    self,
    tokenizer,
    repo_id: str,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Converted LoRA to GGUF with Unsloth",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Convert LoRA to GGUF format using Unsloth",
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
    outtype: str = "f16",
):
    return _unsloth_save_lora_gguf(
        self,
        tokenizer,
        repo_id,
        outtype = outtype,
        push_to_hub = True,
        token = token,
        private = private,
        commit_message = commit_message,
        commit_description = commit_description,
        create_pr = create_pr,
        revision = revision,
    )


def unsloth_convert_lora_to_ggml_and_save_locally(
    self,
    save_directory: str,  # Added parameter for the folder name
    tokenizer,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
    outtype: str = "f16",
):
    return _unsloth_save_lora_gguf(self, tokenizer, save_directory, outtype = outtype)


from .models.loader_utils import get_model_name
from unsloth_zoo.saving_utils import (
    merge_and_overwrite_lora,
    prepare_saving,
)
from unsloth_zoo.llama_cpp import (
    install_llama_cpp,
    convert_to_gguf as _convert_to_gguf,
)


@torch.inference_mode
def save_to_gguf_generic(
    model,
    save_directory,
    tokenizer,
    quantization_method = None,
    quantization_type = "Q8_0",
    repo_id = None,
    token = None,
):
    if token is None and repo_id is not None:
        token = get_token()
    if repo_id is not None and token is None:
        raise RuntimeError("Unsloth: Please specify a token for uploading!")

    if not os.path.exists(os.path.join("llama.cpp", "unsloth_convert_hf_to_gguf.py")):
        install_llama_cpp(just_clone_repo = True)

    # Use old style quantization_method
    new_quantization_methods = []
    if quantization_method is not None:
        # Convert quantization_method to list
        if isinstance(quantization_method, list):
            pass
        elif isinstance(quantization_method, str):
            quantization_method = [
                quantization_method,
            ]
        elif isinstance(quantization_method, tuple):
            quantization_method = list(quantization_method)
        else:
            raise TypeError(
                "Unsloth: quantization_method can only be a string or a list of strings"
            )
        for i, quant_method in enumerate(quantization_method):
            quant_method = quant_method.lower()
            if quant_method == "not_quantized":
                quant_method = "f16"
            elif quant_method == "fast_quantized":
                quant_method = "q8_0"
            elif quant_method == "quantized":
                quant_method = "q4_k_m"
            elif quant_method is None:
                quant_method = "q8_0"
            new_quantization_methods.append(quant_method.lower())
    else:
        new_quantization_methods.append(quantization_type.lower())
    # Check if wrong method
    for quant_method in new_quantization_methods:
        if quant_method not in ALLOWED_QUANTS.keys():
            error = f"Unsloth: Quant method = [{quant_method}] not supported. Choose from below:\n"
            for key, value in ALLOWED_QUANTS.items():
                error += f"[{key}] => {value}\n"
            raise RuntimeError(error)

    # Go through all types and save individually - somewhat inefficient
    # since we save F16 / BF16 multiple times
    for quantization_type in new_quantization_methods:
        metadata = _convert_to_gguf(
            save_directory,
            print_output = True,
            quantization_type = quantization_type,
        )
        if repo_id is not None:
            prepare_saving(
                model,
                repo_id,
                push_to_hub = True,
                max_shard_size = "50GB",
                private = True,
                token = token,
            )

            from huggingface_hub import HfApi

            api = HfApi(token = token)
            api.upload_folder(
                folder_path = save_directory,
                repo_id = repo_id,
                repo_type = "model",
                allow_patterns = ["*.gguf"],
            )
    return metadata


@_normalize_tied_weights_keys_for_save
@torch.inference_mode
def unsloth_generic_save(
    model,
    tokenizer,
    save_directory: Union[str, os.PathLike] = "unsloth_finetuned_merge",
    save_method: str = "lora",  # ["lora", "merged_16bit", "merged_4bit"]
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    # Push to hub
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
    # Our functions
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
    datasets: Optional[List[str]] = None,
):
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    if token is None and push_to_hub:
        token = get_token()

    if save_method == "merged_4bit":
        raise RuntimeError(
            "Unsloth: Merging into 4bit will cause your model to lose accuracy if you plan\n"
            "to merge to GGUF or others later on. I suggest you to do this as a final step\n"
            "if you're planning to do multiple saves.\n"
            "If you are certain, change `save_method` to `merged_4bit_forced`."
        )
    elif save_method == "merged_4bit_forced":
        save_method = "merged_4bit"

    # Full-finetuned models (no LoRA) have no adapters to merge, so fall back
    # to save_pretrained, mirroring the torchao and GGUF save paths.
    _is_peft = isinstance(model, PeftModel)
    if not _is_peft:
        if not is_main_process:
            return

        _save_kwargs = dict(
            safe_serialization = safe_serialization,
            max_shard_size = max_shard_size,
            variant = variant,
        )
        is_qwen3_5_vlm = _is_qwen3_5_vlm(model)
        if ("16bit" in save_method or is_qwen3_5_vlm) and state_dict is None:
            state_dict = model.state_dict()
        if "16bit" in save_method:
            _target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            state_dict = {
                k: v.to(dtype = _target_dtype) if v.is_floating_point() else v
                for k, v in state_dict.items()
            }
        if is_qwen3_5_vlm:
            state_dict = _qwen3_5_vlm_state_dict_for_save(state_dict)
        if state_dict is not None:
            _save_kwargs["state_dict"] = state_dict

        if push_to_hub:
            print(f"Unsloth: Pushing full fine-tuned model to '{save_directory}' ...")
            model.push_to_hub(
                repo_id = save_directory,
                token = token,
                private = private,
                commit_message = commit_message,
                create_pr = create_pr,
                revision = revision,
                commit_description = commit_description,
                tags = tags,
                **_save_kwargs,
            )
            if tokenizer is not None:
                _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
                old_padding_side = _tokenizer.padding_side
                _tokenizer.padding_side = "left"
                tokenizer.push_to_hub(
                    save_directory,
                    token = token,
                    private = private,
                    commit_message = commit_message,
                    create_pr = create_pr,
                    revision = revision,
                )
                _tokenizer.padding_side = old_padding_side
        else:
            print(f"Unsloth: Saving full fine-tuned model to '{save_directory}' ...")
            model.save_pretrained(save_directory, **_save_kwargs)
            if tokenizer is not None:
                _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
                old_padding_side = _tokenizer.padding_side
                _tokenizer.padding_side = "left"
                tokenizer.save_pretrained(save_directory)
                _tokenizer.padding_side = old_padding_side

        print(f"Unsloth: Model saved successfully to '{save_directory}'")
    else:
        merge_and_overwrite_lora(
            get_model_name,
            model = model,
            tokenizer = tokenizer,
            save_directory = save_directory,
            push_to_hub = push_to_hub,
            private = private,
            token = token,
            save_method = save_method,
            output_dtype = None,
            low_disk_space_usage = True,
            use_temp_file = False,
        )

    if push_to_hub and datasets:
        try:
            from huggingface_hub import metadata_update
            save_dir, _ = _determine_username(save_directory, None, token)
            metadata_update(save_dir, {"datasets": datasets}, overwrite = True, token = token)
        except Exception as e:
            logger.warning_once(
                f"Unsloth: Could not update datasets metadata for {save_directory}: {e}"
            )

    return


def unsloth_generic_save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    save_method: str = "merged_16bit",  # ["lora", "merged_16bit", "merged_4bit", "fp8", "mxfp4", "nvfp4", "mxfp8"]
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
    datasets: Optional[List[str]] = None,
    calibration_dataset = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
):
    """
    Same as .push_to_hub(...) except 4bit weights are auto
    converted to float16 with as few overhead as possible.

    Choose for `save_method` to be either:
    1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
    2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
    3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    4.  FP8 / FP4 compressed export for vLLM via llm-compressor:
        `fp8` (dynamic W8A8), `mxfp4`, `nvfp4` (W4A4), `mxfp8`. The LoRA is merged to 16bit at
        `save_directory`, then a quantized checkpoint is written to `save_directory + "-<fmt>"`.
        `nvfp4` needs calibration data (defaults to ultrachat; override with `calibration_dataset`).
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"
            "You can do it separately via `tokenizer.save_pretrained(...)`"
        )

    # FP8 / FP4 compressed-tensors export (llm-compressor) -> handled separately.
    _compressed = _normalize_compressed_method(save_method)
    if _compressed is not None:
        scheme, needs_calibration, suffix = _compressed
        _unsloth_save_compressed_tensors(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            scheme = scheme,
            needs_calibration = needs_calibration,
            suffix = suffix,
            push_to_hub = push_to_hub,
            token = token,
            is_main_process = is_main_process,
            calibration_dataset = calibration_dataset,
            num_calibration_samples = num_calibration_samples,
            max_seq_length = max_seq_length,
            # Forward standard save kwargs to the 16bit merge.
            state_dict = state_dict,
            save_function = save_function,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            variant = variant,
            save_peft_format = save_peft_format,
            tags = tags,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            datasets = datasets,
        )
        for _ in range(3):
            gc.collect()
        return

    arguments = dict(locals())
    arguments["model"] = self
    del arguments["self"]
    del arguments["_compressed"]
    del arguments["calibration_dataset"]
    del arguments["num_calibration_samples"]
    del arguments["max_seq_length"]
    unsloth_generic_save(**arguments)
    for _ in range(3):
        gc.collect()


def unsloth_generic_push_to_hub_merged(
    self,
    repo_id: str,
    tokenizer = None,
    save_method: str = "merged_16bit",  # ["lora", "merged_16bit", "merged_4bit"]
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str, None] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
    datasets: Optional[List[str]] = None,
    calibration_dataset = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
):
    """
    Same as .push_to_hub(...) except 4bit weights are auto
    converted to float16 with as few overhead as possible.

    Choose for `save_method` to be either:
    1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
    2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
    3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    4.  FP8 / FP4 compressed export for vLLM: `fp8`, `mxfp4`, `nvfp4`, `mxfp8`.
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"
            "You can do it separately via `tokenizer.push_to_hub(...)`"
        )

    # FP8 / FP4 compressed-tensors export (llm-compressor) -> handled separately.
    _compressed = _normalize_compressed_method(save_method)
    if _compressed is not None:
        scheme, needs_calibration, suffix = _compressed
        _unsloth_save_compressed_tensors(
            model = self,
            save_directory = repo_id,
            tokenizer = tokenizer,
            scheme = scheme,
            needs_calibration = needs_calibration,
            suffix = suffix,
            push_to_hub = True,
            token = token,
            private = private,
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
            calibration_dataset = calibration_dataset,
            num_calibration_samples = num_calibration_samples,
            max_seq_length = max_seq_length,
            # Forward standard save kwargs to the 16bit merge.
            use_temp_dir = use_temp_dir,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            tags = tags,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            datasets = datasets,
        )
        for _ in range(3):
            gc.collect()
        return

    arguments = dict(locals())
    arguments["model"] = self
    arguments["save_directory"] = repo_id
    arguments["push_to_hub"] = True
    del arguments["self"]
    del arguments["repo_id"]
    del arguments["_compressed"]
    del arguments["calibration_dataset"]
    del arguments["num_calibration_samples"]
    del arguments["max_seq_length"]
    unsloth_generic_save(**arguments)
    for _ in range(3):
        gc.collect()


def _unsloth_save_torchao_with_attached_config(
    model,
    save_directory: Union[str, os.PathLike],
    tokenizer,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
):
    """Save a QAT-trained model by converting fake-quantized weights to real quantized weights."""
    # Convert QAT fake-quantized weights to real quantized weights
    _convert_torchao_model(model)
    # PEFT models also might come here, so parse it
    if isinstance(model, PeftModelForCausalLM):
        _unsloth_save_torchao_with_given_config(
            model = model,
            save_directory = save_directory,
            tokenizer = tokenizer,
            torchao_config = model.config.quantization_config,
            push_to_hub = push_to_hub,
            token = token,
        )
        return

    # TorchAO does not support safe_serialization reliably
    safe_serialization = False

    if push_to_hub:
        model.push_to_hub(save_directory, safe_serialization = safe_serialization, token = token)
        tokenizer.push_to_hub(save_directory, token = token)
    else:
        model.save_pretrained(save_directory, safe_serialization = safe_serialization)
        tokenizer.save_pretrained(save_directory)


def _unsloth_save_torchao_with_given_config(
    model,
    save_directory: Union[str, os.PathLike],
    tokenizer,
    torchao_config,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
):
    """Quantizes the model with torchao and saves a torchao quantized checkpoint

    Args
      `save_directory`: local folder path or huggingface hub ID when `push_to_hub` is set to True, e.g. `my_model`
      `torchao_config` (TorchAOBaseConfig): configuration for torchao quantization, full list: https://docs.pytorch.org/ao/main/api_ref_quantization.html#inference-apis-for-quantize
      `push_to_hub` (bool): whether to push the checkpoint to huggingface hub or save locally
    """

    if push_to_hub:
        assert token is not None, "Unsloth: Please specify a token for uploading!"

    assert (
        torchao_config is not None
    ), "Unsloth: Please specify a torchao_config for post-training quantization!"

    # first merge the lora weights
    arguments = dict(locals())
    arguments["push_to_hub"] = False  # We save ourselves
    arguments["save_method"] = "merged_16bit"  # Must be 16bit
    del arguments["torchao_config"]

    if not isinstance(model, PeftModelForCausalLM) and not isinstance(model, PeftModel):
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
    else:
        unsloth_generic_save(**arguments)

    for _ in range(3):
        gc.collect()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TorchAoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )
    from torchao import quantize_

    if isinstance(torchao_config, TorchAoConfig):
        quantization_config = torchao_config
    else:
        quantization_config = TorchAoConfig(quant_type = torchao_config)

    # Determine if this is a VLM
    is_vlm = False
    if hasattr(model, "config") and hasattr(model.config, "architectures"):
        is_vlm = any(
            x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
            for x in model.config.architectures
        )
        is_vlm = is_vlm or hasattr(model.config, "vision_config")
    auto_model = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    auto_processor = AutoProcessor if is_vlm else AutoTokenizer

    tokenizer = auto_processor.from_pretrained(save_directory)
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    # TorchAO must only use bfloat16 for loading (float16 fails)
    if HAS_TORCH_DTYPE:
        kwargs = {"torch_dtype": torch.bfloat16}
    else:
        kwargs = {"dtype": torch.bfloat16}

    # Reload with quantization applied
    quantized_model = auto_model.from_pretrained(
        save_directory,
        device_map = "auto",
        quantization_config = quantization_config,
        **kwargs,
    )

    torchao_save_directory = save_directory + "-torchao"

    # TorchAO does not support safe_serialization right now 0.14.0 seems broken!
    safe_serialization = Version(importlib_version("torchao")) > Version("0.14.0")
    safe_serialization = False

    if push_to_hub:
        quantized_model.push_to_hub(
            torchao_save_directory, safe_serialization = safe_serialization, token = token
        )
        tokenizer.push_to_hub(torchao_save_directory, token = token)
    else:
        quantized_model.save_pretrained(
            torchao_save_directory, safe_serialization = safe_serialization
        )
        tokenizer.save_pretrained(torchao_save_directory, token = token)

    # Clean up the intermediate unquantized model
    if os.path.exists(save_directory):
        try:
            shutil.rmtree(save_directory)
        except:
            pass


def _scheme_is_available(scheme):
    """True if `scheme` is a known preset in the installed compressed_tensors."""
    try:
        from compressed_tensors.quantization import quant_scheme as _qs

        presets = getattr(_qs, "PRESET_SCHEMES", None)
        if presets is None:
            return True
        return scheme in presets
    except Exception:
        # If we cannot introspect, let llm-compressor validate the scheme itself.
        return True


def _print_compressed_hw_note(scheme, out_dir):
    if scheme in ("FP8_DYNAMIC", "MXFP8"):
        hw = "NVIDIA GPUs with compute capability >= 8.9 (Ada / Hopper) or newer"
    else:
        hw = (
            "NVIDIA Blackwell (SM100+) for full activation quantization "
            "(older GPUs fall back to weight-only in vLLM)"
        )
    print(
        f"Unsloth: Saved {scheme} compressed checkpoint to '{out_dir}'.\n"
        f"Unsloth: Load it with vLLM for accelerated inference. Hardware for full speed: {hw}."
    )


def _offload_model_for_quantize_subprocess(model):
    """Best-effort: move the merged model's weights off the GPU before the
    llm-compressor subprocess loads its own copy from disk, so the GPUs need not
    hold both at once. Returns an opaque restore token for
    ``_restore_model_after_quantize_subprocess`` (None when nothing was moved).

    Two shapes are handled:
      * single-device CUDA model -> plain ``.to("cpu")``, restored with
        ``.to(device)``;
      * accelerate-dispatched model (a multi-GPU ``device_map`` shard, e.g. the
        Studio multi-GPU export load) -> accelerate's hooks are removed and the
        model moved to CPU, restored by re-dispatching over the recorded
        ``hf_device_map``. A plain ``.to("cpu")`` is invalid on a dispatched
        model, which is why the old single-device-only move skipped these and
        left every GPU holding a full copy while the subprocess loaded another.
        A map with non-GPU targets (cpu/disk offload) is left alone: accelerate
        is already offloading those weights.

    Quantized (bnb) models are never moved.
    """
    try:
        if not (torch.cuda.is_available() and hasattr(model, "parameters")):
            return None
        if (
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_quantized", False)
        ):
            return None
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            targets = {str(v) for v in device_map.values()}
            if not all(t.isdigit() or t.startswith("cuda") for t in targets):
                return None
            from accelerate.hooks import remove_hook_from_submodules

            remove_hook_from_submodules(model)
            model.to("cpu")
            return ("dispatch", dict(device_map))
        devices = {str(p.device) for p in model.parameters()}
        if len(devices) == 1 and next(iter(devices)).startswith("cuda"):
            device = next(model.parameters()).device
            model.to("cpu")
            return ("device", device)
    except Exception:
        return None
    return None


def _restore_model_after_quantize_subprocess(model, restore_token) -> None:
    """Undo ``_offload_model_for_quantize_subprocess``; warns instead of raising."""
    if restore_token is None:
        return
    kind, value = restore_token
    try:
        if kind == "dispatch":
            from accelerate import dispatch_model
            dispatch_model(model, device_map = value)
        else:
            model.to(value)  # restore the model to its original device
    except Exception:
        logger.warning_once(
            "Unsloth: could not restore the model to its original device(s) after compressed "
            "export; it may remain on CPU."
        )


def _unsloth_save_compressed_tensors(
    model,
    save_directory: Union[str, os.PathLike],
    tokenizer,
    scheme: str,
    needs_calibration: bool,
    suffix: str,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    calibration_dataset = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
    **merge_kwargs,
):
    """Export an FP8/FP4 compressed-tensors checkpoint via llm-compressor.

    Mirrors the torchao PTQ path: LoRA is first merged into the base model at 16bit and
    written to `save_directory` (which is kept). The merged checkpoint is then quantized with
    llm-compressor's `QuantizationModifier(scheme)` in a separate process (so Unsloth's
    transformers monkey-patches do not interfere), and written to the sibling directory
    `save_directory + "-" + suffix`. The result is intended for vLLM inference.
    """
    import tempfile

    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)
    # Resolve a token for the hub push and/or loading a gated calibration dataset in the subprocess.
    if token is None:
        token = get_token()

    # Only the main process installs deps, merges, quantizes, and uploads (mirrors the non-PEFT
    # save path); other ranks return at once so they neither race on dirs nor run pip installs.
    if not is_main_process:
        return None

    # 1) Install llm-compressor and gate on scheme availability BEFORE merging, so an unsupported
    #    scheme (e.g. mxfp8) fails fast instead of writing a full 16bit checkpoint first.
    install_llm_compressor()
    if not _scheme_is_available(scheme):
        try:
            import transformers as _tf
            tf_ver = _tf.__version__
        except Exception:
            tf_ver = "unknown"
        raise RuntimeError(
            f"Unsloth: scheme '{scheme}' is not available in your installed "
            f"compressed-tensors / llm-compressor.\n"
            f"It requires a newer llm-compressor that needs transformers>=5.9 "
            f"(you have transformers {tf_ver}).\n"
            "Use save_method in {fp8, mxfp4, nvfp4}, or upgrade transformers + llm-compressor."
        )

    # 2) Pick the local working dir. For a hub push, save_directory is a repo id, so merge and
    #    quantize inside an isolated temp dir instead of writing ./<repo_id> into the cwd.
    repo_id, work_tmp, calib_tmp, model_restore = None, None, None, None
    if push_to_hub:
        repo_id = os.fspath(save_directory)
        work_tmp = tempfile.mkdtemp(prefix = "unsloth-compressed-")
        local_dir = os.path.join(work_tmp, os.path.basename(repo_id.rstrip("/")) or "model")
    else:
        # Drop trailing separators so the sibling "<dir>-<fmt>" output is not nested inside <dir>.
        local_dir = os.fspath(save_directory)
        local_dir = local_dir.rstrip("/\\") or local_dir

    # Wrap the body so the isolated temp dirs are always cleaned up, even when the merge,
    # quantization, validation, or hub upload raises.
    api = None
    try:
        # Validate Hub access up front (a bad token / denied repo should fail before the expensive
        # merge and quantization, matching the normal push path). create_repo is idempotent.
        if push_to_hub:
            from huggingface_hub import HfApi
            api = HfApi(token = token)
            api.create_repo(
                repo_id = repo_id,
                repo_type = "model",
                private = merge_kwargs.get("private", None),
                exist_ok = True,
            )

        # 3) Merge to 16bit at local_dir (kept for local saves) via unsloth_generic_save, so LoRA
        #    adapters are merged and full-finetuned models written in 16bit consistently. Extra
        #    save kwargs (state_dict, max_shard_size, ...) flow through merge_kwargs.
        # The intermediate 16bit checkpoint is internal staging that the converter subprocess
        # reloads with default weight filenames, so never write variant-named shards here; the
        # user's variant (if any) is applied to the final compressed checkpoint in the subprocess.
        variant = merge_kwargs.pop("variant", None)
        print(f"Unsloth: Merging to 16bit before {scheme} quantization...")
        merge_args = dict(merge_kwargs)
        merge_args.update(
            dict(
                model = model,
                tokenizer = tokenizer,
                save_directory = local_dir,
                save_method = "merged_16bit",
                push_to_hub = False,
                token = token,
                is_main_process = is_main_process,
            )
        )
        unsloth_generic_save(**merge_args)

        # 4) Detect VLM + trust_remote_code from the in-memory model config. A vision/multimodal
        #    model exposes a vision_config or an explicitly vision-named architecture; a bare
        #    *ForConditionalGeneration also matches text seq2seq models (T5/BART/Whisper), so it
        #    is not treated as a VLM on its own.
        is_vlm = False
        if hasattr(model, "config"):
            archs = getattr(model.config, "architectures", None) or []
            is_vlm = hasattr(model.config, "vision_config") or any(
                x.endswith("ForVisionText2Text") for x in archs
            )
        if is_vlm:
            logger.warning(
                "Unsloth: FP8/FP4 compressed export for vision / multimodal models is "
                "experimental; vision-tower layers may be affected."
            )
        trust_remote_code = (
            bool(getattr(model.config, "auto_map", None)) if hasattr(model, "config") else False
        )

        # 5) Marshal the calibration dataset for the subprocess: None -> ultrachat default; a
        #    str/PathLike is a local save_to_disk dir if it exists else a Hub id; Dataset -> temp.
        calib_kind, calib_value = "none", ""
        if needs_calibration and calibration_dataset is not None:
            if isinstance(calibration_dataset, (str, os.PathLike)):
                calib_value = os.fspath(calibration_dataset)
                calib_kind = "disk" if os.path.isdir(calib_value) else "hfid"
            elif hasattr(calibration_dataset, "save_to_disk"):
                # Only persist the samples we need, so multi-GB training sets are not fully copied.
                ds_to_save = calibration_dataset
                # A DatasetDict's len() is the split count, not rows; pick one split first so the
                # row subsample below applies and we do not save every split to the temp dir.
                try:
                    from datasets import DatasetDict
                    if isinstance(ds_to_save, DatasetDict):
                        ds_to_save = ds_to_save.get("train", None) or next(
                            iter(ds_to_save.values())
                        )
                except Exception:
                    pass
                try:
                    if (
                        num_calibration_samples
                        and hasattr(ds_to_save, "select")
                        and len(ds_to_save) > num_calibration_samples
                    ):
                        ds_to_save = ds_to_save.shuffle(seed = 42).select(
                            range(num_calibration_samples)
                        )
                except Exception:
                    ds_to_save = calibration_dataset
                calib_tmp = tempfile.mkdtemp(prefix = "unsloth-calib-")
                shutil.rmtree(calib_tmp, ignore_errors = True)  # save_to_disk wants a fresh path
                ds_to_save.save_to_disk(calib_tmp)
                calib_kind, calib_value = "disk", calib_tmp
            else:
                raise TypeError(
                    "Unsloth: calibration_dataset must be None, a Hugging Face dataset id, a "
                    "local path saved with Dataset.save_to_disk(...), or a Dataset with "
                    "save_to_disk()."
                )
        elif not needs_calibration and calibration_dataset is not None:
            logger.warning_once(
                f"Unsloth: scheme '{scheme}' is data-free; ignoring calibration_dataset."
            )

        # 6) Quantize in a separate process: importing Unsloth patches transformers attention,
        #    which breaks the forward llm-compressor runs for calibration. Run the converter by
        #    file path (not `-m`) so the subprocess stays unpatched, like GGUF -> llama.cpp.
        out_dir = local_dir + "-" + suffix
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compressed_quantize.py")
        cmd = [
            sys.executable,
            runner,
            "--model",
            local_dir,
            "--scheme",
            scheme,
            "--out",
            out_dir,
            "--calibration-dataset-kind",
            calib_kind,
            "--num-calibration-samples",
            str(num_calibration_samples),
            "--max-seq-length",
            str(max_seq_length),
        ]
        if needs_calibration:
            cmd.append("--needs-calibration")
        if calib_value:
            cmd += ["--calibration-dataset", calib_value]
        if is_vlm:
            cmd.append("--is-vlm")
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if variant:
            cmd += ["--variant", variant]

        # Free the in-memory model's CUDA memory before the subprocess loads its own copy from
        # disk, so the GPUs need not hold both at once. Best-effort and restored in finally;
        # covers single-device CUDA models and accelerate-dispatched multi-GPU shards.
        model_restore = _offload_model_for_quantize_subprocess(model)
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Expose the token so the subprocess can load a gated/private calibration dataset.
        env = os.environ.copy()
        if isinstance(token, str) and token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

        print(
            f"Unsloth: Quantizing the merged model to {scheme} with llm-compressor "
            "(in a separate process)..."
        )
        try:
            subprocess.check_call(cmd, env = env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Unsloth: {scheme} quantization failed (llm-compressor subprocess exit "
                f"{e.returncode}). See the output above for details."
            )

        # 7) Validate the artifact.
        cfg_path = os.path.join(out_dir, "config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding = "utf-8") as f:
                cfg = json.load(f)
        if "quantization_config" not in cfg:
            raise RuntimeError(
                f"Unsloth: {scheme} export failed - no quantization_config written to {cfg_path}"
            )

        # 8) Optional hub upload of the compressed artifact (not the intermediate 16bit one).
        #    The repo was already created/validated up front, so just upload here.
        if push_to_hub:
            print(f"Unsloth: Uploading {scheme} checkpoint to '{repo_id}' ...")
            api.upload_folder(
                folder_path = out_dir,
                repo_id = repo_id,
                repo_type = "model",
                commit_message = merge_kwargs.get("commit_message", None),
                commit_description = merge_kwargs.get("commit_description", None),
                create_pr = merge_kwargs.get("create_pr", False),
                revision = merge_kwargs.get("revision", None),
            )
            # Attach datasets metadata to the pushed repo, like the normal merged push path.
            datasets = merge_kwargs.get("datasets", None)
            if datasets:
                try:
                    from huggingface_hub import metadata_update
                    metadata_update(repo_id, {"datasets": datasets}, overwrite = True, token = token)
                except Exception as meta_err:
                    logger.warning_once(
                        f"Unsloth: could not update datasets metadata for {repo_id}: {meta_err}"
                    )

        # 9) Inference hardware note.
        result = repo_id if push_to_hub else out_dir
        _print_compressed_hw_note(scheme, result)
        return result
    finally:
        _restore_model_after_quantize_subprocess(model, model_restore)
        if calib_tmp is not None and os.path.isdir(calib_tmp):
            shutil.rmtree(calib_tmp, ignore_errors = True)
        if work_tmp is not None:
            shutil.rmtree(work_tmp, ignore_errors = True)
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def unsloth_save_pretrained_torchao(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    torchao_config = None,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
):
    """Saves a torchao quantized model checkpoint.

    This function handles two mutually exclusive workflows:

    1. **QAT (Quantization-Aware Training)**: If the model was trained with `qat_scheme`
       parameter, do NOT pass `torchao_config`. The function will convert the QAT
       fake-quantized weights to real quantized weights and save directly.

    2. **PTQ (Post-Training Quantization)**: If you want to apply quantization to a
       regular model, pass a `torchao_config`. The model must NOT have been trained
       with `qat_scheme`.

    Args:
      `save_directory`: local folder path or huggingface hub ID when `push_to_hub` is True
      `tokenizer`: the tokenizer to save alongside the model
      `torchao_config` (TorchAOBaseConfig): configuration for torchao quantization.
          Required for PTQ, must be None for QAT models.
          Options: https://docs.pytorch.org/ao/main/api_ref_quantization.html#inference-apis-for-quantize
      `push_to_hub` (bool): whether to push to huggingface hub or save locally
      `token`: HuggingFace token for pushing to hub
    """
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    if token is None and push_to_hub:
        token = get_token()

    has_qat_config = hasattr(self, "_torchao_config") and self._torchao_config is not None

    if torchao_config is not None:
        # PTQ path: user provided a config, model must NOT have QAT config unless PEFT
        assert not has_qat_config, (
            "Unsloth: You passed `torchao_config` but this model was trained with `qat_scheme`. "
            "For QAT models, do not pass `torchao_config` - the quantization config is already "
            "attached to the model from training."
        )
        _unsloth_save_torchao_with_given_config(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            torchao_config = torchao_config,
            push_to_hub = push_to_hub,
            token = token,
        )
    else:
        # QAT path: no config provided, model must have QAT config
        assert has_qat_config, (
            "Unsloth: No `torchao_config` provided and model was not trained with `qat_scheme`. "
            "Either train with `qat_scheme` parameter, or provide a `torchao_config` for "
            "post-training quantization."
        )
        _unsloth_save_torchao_with_attached_config(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            push_to_hub = push_to_hub,
            token = token,
        )

    for _ in range(3):
        gc.collect()


def not_implemented_save(*args, **kwargs):
    raise NotImplementedError("Unsloth: Sorry GGUF is currently not supported for vision models!")


def patch_saving_functions(model, vision = False):
    import inspect
    import types
    from typing import Callable, Optional, Union, List

    # And now re add our saving methods!
    if model.push_to_hub.__name__ == "unsloth_push_to_hub":
        original_push_to_hub = model.original_push_to_hub
    else:
        original_push_to_hub = model.push_to_hub

    signature = str(inspect.signature(original_push_to_hub)).replace("NoneType", "None")
    signature = signature[1:]
    signature = re.sub("<function save at .+?>", "torch.save", signature)
    docs = original_push_to_hub.__doc__.encode("utf-8").decode("utf-8")

    push_to_hub_text = f'''def unsloth_push_to_hub(self, {signature}:
    """
    {docs}
    """
    arguments = dict(locals())
    del arguments["self"]
    if "tags" in arguments and arguments["tags"] is not None:
        assert(isinstance(arguments["tags"], (list, tuple)))
        arguments["tags"] = list(arguments["tags"]) + ["unsloth",]
    elif "tags" in arguments:
        arguments["tags"] = ["unsloth",]
    elif hasattr(self, "add_model_tags"):
        self.add_model_tags(["unsloth",])

    if "commit_message" in arguments:
        commit_message = arguments["commit_message"]
        if commit_message is not None:
            if not commit_message.endswith(" "): commit_message += " "
            if "Unsloth" not in commit_message:
                commit_message += "(Trained with Unsloth)"
        else:
            commit_message = "Upload model trained with Unsloth"
        arguments["commit_message"] = commit_message

    if "commit_description" in arguments:
        commit_description = arguments["commit_description"]
        if commit_description is not None:
            if not commit_description.endswith(" "): commit_description += " "
            if "Unsloth" not in commit_description:
                commit_description += "(Trained with Unsloth 2x faster)"
        else:
            commit_description = "Upload model trained with Unsloth 2x faster"
        arguments["commit_description"] = commit_description

    # Update model tag
    if hasattr(self, "config"):
        _ = upload_to_huggingface(
            self, arguments["repo_id"], arguments["token"],
            "finetuned", "trl", file_location = None,
            old_username = None, private = arguments["private"],
        )
    pass

    try:
        self.original_push_to_hub(**arguments)
    except:
        del arguments["tags"]
        self.original_push_to_hub(**arguments)
    pass

    if hasattr(self, "config"):
        print("Saved model to https://huggingface.co/" + arguments["repo_id"])
    pass
    '''
    exec(push_to_hub_text, globals())

    def unsloth_tokenizer_save_pretrained(
        self,
        save_directory,
        legacy_format = None,
        filename_prefix = None,
        push_to_hub = False,
        **kwargs,
    ):
        result = self.original_save_pretrained(
            save_directory,
            legacy_format = legacy_format,
            filename_prefix = filename_prefix,
            push_to_hub = False,
            **kwargs,
        )
        _preserve_sentencepiece_tokenizer_assets(
            self,
            save_directory,
            token = kwargs.get("token", None),
        )
        _preserve_tokenizer_eos_token(
            self,
            save_directory,
            filename_prefix = filename_prefix,
        )
        if push_to_hub:
            push_kwargs = dict(kwargs)
            repo_id = push_kwargs.pop("repo_id", save_directory)
            self.push_to_hub(repo_id, **push_kwargs)
        return result

    if (
        isinstance(model, PreTrainedTokenizerBase)
        and model.save_pretrained.__name__ != "unsloth_tokenizer_save_pretrained"
    ):
        model.original_save_pretrained = model.save_pretrained
        model.save_pretrained = types.MethodType(unsloth_tokenizer_save_pretrained, model)
    elif getattr(model, "tokenizer", None) is not None:
        patch_saving_functions(model.tokenizer)

    original_model = model
    while True:
        # Check if push_to_hub exists before accessing its __name__
        if (
            hasattr(original_model, "push_to_hub")
            and original_model.push_to_hub.__name__ != "unsloth_push_to_hub"
        ):
            original_model.original_push_to_hub = original_model.push_to_hub
            original_model.push_to_hub = types.MethodType(unsloth_push_to_hub, original_model)
            if hasattr(original_model, "add_model_tags"):
                original_model.add_model_tags(
                    [
                        "unsloth",
                    ]
                )

        if hasattr(original_model, "model"):
            original_model = original_model.model
        else:
            break

    # Add saving methods to top level model
    if not vision:
        if hasattr(model, "config"):
            # Counteract tokenizers
            model.push_to_hub_merged = types.MethodType(unsloth_generic_push_to_hub_merged, model)
            model.save_pretrained_merged = types.MethodType(
                unsloth_generic_save_pretrained_merged, model
            )
            model.push_to_hub_gguf = types.MethodType(unsloth_push_to_hub_gguf, model)
            model.save_pretrained_gguf = types.MethodType(unsloth_save_pretrained_gguf, model)
            model.save_pretrained_torchao = types.MethodType(unsloth_save_pretrained_torchao, model)
            model.push_to_hub_ggml = types.MethodType(
                unsloth_convert_lora_to_ggml_and_push_to_hub, model
            )
            model.save_pretrained_ggml = types.MethodType(
                unsloth_convert_lora_to_ggml_and_save_locally, model
            )
    else:
        # Vision only 1 option
        model.push_to_hub_merged = types.MethodType(unsloth_generic_push_to_hub_merged, model)
        model.save_pretrained_merged = types.MethodType(
            unsloth_generic_save_pretrained_merged, model
        )
        model.push_to_hub_gguf = types.MethodType(unsloth_push_to_hub_gguf, model)
        model.save_pretrained_gguf = types.MethodType(unsloth_save_pretrained_gguf, model)
        model.save_pretrained_torchao = types.MethodType(unsloth_save_pretrained_torchao, model)
    return model
