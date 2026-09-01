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

# Defensive imports: added in unsloth-zoo PR #526 and may not exist on older versions.
from unsloth_zoo.utils import Version
from importlib.metadata import version as importlib_version
from unsloth_zoo.hf_utils import dtype_from_config, HAS_TORCH_DTYPE
from contextlib import contextmanager
from unsloth_zoo.llama_cpp import (
    convert_to_gguf,
    quantize_gguf,
    use_local_gguf,
    install_llama_cpp,
    check_llama_cpp,
    _download_convert_hf_to_gguf,
)

try:
    from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR, IS_WINDOWS
except ImportError:
    import sys
    IS_WINDOWS = sys.platform == "win32"
    LLAMA_CPP_DEFAULT_DIR = "llama.cpp"
# Without bnb, peft stops exporting its 4bit LoRA layer too. Both names only feed isinstance checks, so
# placeholders nothing can match are exact stand-ins.
try:
    from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
    from peft.tuners.lora import Linear4bit as Peft_Linear4bit
except Exception:

    class Bnb_Linear4bit:
        pass

    class Peft_Linear4bit:
        pass


from peft.tuners.lora import Linear as Peft_Linear
from typing import Optional, Callable, Union, List
import sys
import math
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
import traceback
import psutil
import re
from transformers.models.llama.modeling_llama import logger
from .models.loader_utils import (
    get_model_name,
    _resolve_hub_repo_cached_file,
    _tokenizer_cache_dir,
    _tokenizer_revision,
    _tokenizer_wants_local_only,
)
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
        # For older versions of huggingface_hub.
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

# llama.cpp specific targets: all takes 90s, the below 60s.
LLAMA_CPP_TARGETS = [
    "llama-quantize",
    "llama-cli",
    "llama-server",
]

# is_kaggle_environment needs a real kernel, not a KAGGLE_* variable: the Kaggle CLI reads those on an
# ordinary laptop, and the branches below (/tmp save paths, deleting the cached base model) are all wrong
# there.
from .disk_utils import (
    KAGGLE_TMP,
    is_colab_environment,
    is_kaggle_environment,
    estimate_gguf_export_bytes,
    free_bytes,
    kaggle_tmp_redirect,
    logical_numel,
    model_16bit_bytes,
)

IS_COLAB_ENVIRONMENT = is_colab_environment()
IS_KAGGLE_ENVIRONMENT = is_kaggle_environment()

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

# See llama.cpp examples/quantize/quantize.cpp#L19 and mlabonne.github.io Quantize_Llama_2_models_using_ggml.
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

# IQ quants need an importance matrix; llama.cpp refuses them without one, so they are only accepted when
# imatrix_file=... is supplied.
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


def _describe_exception(exc) -> str:
    """A description that survives an exception with no message.

    `f"{exc}"` is the EMPTY STRING when the args are empty, so
    `f"Failed to save model: {exc}"` named nothing at all on a real Kaggle
    Q8_0 export. Leading with the type keeps the class of failure visible.
    """
    text = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def has_curl():
    return shutil.which("curl") is not None


CURL_FLAG = "-DLLAMA_CURL=ON" if has_curl() else "-DLLAMA_CURL=OFF"


# llm-compressor FP8/FP4 export for vLLM: alias -> (scheme, needs_calibration, dir suffix). needs_calibration
# only for static activation scales (FP8 static, NVFP4); the rest run data-free. Schemes absent from the
# installed compressed-tensors are gated at runtime.
COMPRESSED_EXPORT_SCHEMES = {
    "fp8": ("FP8_DYNAMIC", False, "fp8"),
    "fp8_dynamic": ("FP8_DYNAMIC", False, "fp8"),
    "dynamic_fp8": ("FP8_DYNAMIC", False, "fp8"),
    "w8a8_fp8": ("FP8_DYNAMIC", False, "fp8"),
    "fp8_static": ("FP8", True, "fp8-static"),
    "static_fp8": ("FP8", True, "fp8-static"),
    "fp8_block": ("FP8_BLOCK", False, "fp8-block"),
    "block_fp8": ("FP8_BLOCK", False, "fp8-block"),
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
    "mxfp8": ("MXFP8", False, "mxfp8"),
    "w8a8_mxfp8": ("MXFP8", False, "mxfp8"),
    "mxfp4": ("MXFP4", False, "mxfp4"),
    "w4a4_mxfp4": ("MXFP4", False, "mxfp4"),
    "mxfp4a16": ("MXFP4A16", False, "mxfp4a16"),
    "w4a16_mxfp4": ("MXFP4A16", False, "mxfp4a16"),
    "nvfp4": ("NVFP4", True, "nvfp4"),
    "w4a4_nvfp4": ("NVFP4", True, "nvfp4"),
    "nvfp4a16": ("NVFP4A16", False, "nvfp4a16"),
    "w4a16_nvfp4": ("NVFP4A16", False, "nvfp4a16"),
}


# torchao portable export: device-agnostic FP8 / INT8, no NVIDIA GPU needed. alias -> (kind, sibling suffix);
# FP8 writes safetensors, INT8 .bin, both load in vLLM.
TORCHAO_EXPORT_SCHEMES = {
    "torchao_fp8": ("fp8", "torchao-fp8"),
    "torchao_int8": ("int8", "torchao-int8"),
    "portable_fp8": ("fp8", "torchao-fp8"),
    "portable_int8": ("int8", "torchao-int8"),
}


def _normalize_torchao_method(save_method):
    """Return (kind, suffix) if `save_method` is a torchao portable FP8/INT8 export, else None."""
    if not isinstance(save_method, str):
        return None
    key = save_method.lower().strip().replace("-", "_").replace(" ", "_")
    return TORCHAO_EXPORT_SCHEMES.get(key)


def _loaded_via_remote_code(obj):
    """True if `obj`'s class comes from downloaded custom code (an auto_map module).

    Transformers loads auto_map code into the ``transformers_modules`` package, so a
    ``transformers_modules`` class proves the original load actually ran that remote code
    (which the caller's / Unsloth's consent gate scans at load time). Export paths derive their
    reload trust_remote_code from this - the already approved load decision - instead of from a
    checkpoint's static ``auto_map``: a model that loads with built-in classes must not have its
    unvetted remote code run when it is re-read during quantization export. Walks PEFT / wrapper
    layers so a LoRA over a custom-code base is still detected, and processor components so a
    custom tokenizer held inside a built-in processor keeps its approved trust.
    """
    seen = set()
    queue = [obj]
    while queue and len(seen) < 16:
        node = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        # __module__ can be None on dynamically created or C-extension classes; treat a non-string as "not
        # remote code" rather than crashing the export.
        module = getattr(type(node), "__module__", None)
        if isinstance(module, str) and module.startswith("transformers_modules"):
            return True
        if hasattr(node, "get_base_model"):
            try:
                queue.append(node.get_base_model())
            except Exception:
                pass
        # PEFT / trainer wrappers hold the real model in base_model / model; a ProcessorMixin holds its
        # (possibly custom-code) components as attributes.
        for attr in (
            "base_model",
            "model",
            "tokenizer",
            "image_processor",
            "feature_extractor",
            "video_processor",
        ):
            queue.append(getattr(node, attr, None))
    return False


def _normalize_compressed_method(save_method):
    """Return (scheme, needs_calibration, suffix) if `save_method` is an FP8/FP4 compressed
    export, else None (so normal lora / merged_16bit / merged_4bit handling proceeds).

    Near-miss FP8/FP4 names that are not supported raise a precise error instead of silently
    falling through to the generic "unknown save_method" message.
    """
    if not isinstance(save_method, str):
        return None
    key = save_method.lower().strip().replace("-", "_").replace(" ", "_")
    # torchao aliases route to the torchao path, so skip them before the "fp8" near-miss check.
    if key in TORCHAO_EXPORT_SCHEMES:
        return None
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
        # No Makefile: only CMake-only if a real CMake project is present.
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
    # "Q2_K_L" is an Unsloth preset, not a native llama.cpp ftype: q2_k with output and token embeddings kept
    # at q8_0.
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
        raise RuntimeError(
            f"Failed to quantize {input_gguf} to q2_k_l: "
            f"{_describe_exception(e)}{error_details}"
        ) from e

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
    # Refs of one repo can differ in whether they ship the asset, so memoize per ref.
    revision = _tokenizer_revision(tokenizer)
    cache_key = (source, revision)
    if cache_key in _TOKENIZER_MODEL_CACHE:
        return _TOKENIZER_MODEL_CACHE[cache_key]

    # Hub repo id: probe the local cache before model_info (#7481).
    cache_dir = _tokenizer_cache_dir(tokenizer) or os.environ.get("HF_HUB_CACHE")
    if not cache_dir:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(hf_home, "hub")

    cached_path = _resolve_hub_repo_cached_file(
        source,
        "tokenizer.model",
        token = token,
        local_files_only = True,
        cache_dir = cache_dir,
        revision = revision,
    )
    if cached_path is not None:
        _TOKENIZER_MODEL_CACHE[cache_key] = True
        return True

    if _tokenizer_wants_local_only(tokenizer):
        return False

    try:
        repo_info = HfApi(token = token).model_info(source, revision = revision, files_metadata = False)
    except Exception:
        return False

    has_tokenizer_model = any(
        sibling.rfilename == "tokenizer.model" for sibling in (repo_info.siblings or [])
    )
    _TOKENIZER_MODEL_CACHE[cache_key] = has_tokenizer_model
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
                    f"Unsloth: Restored added_tokens_decoder metadata in {tokenizer_config_path}."
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
                cache_dir = _tokenizer_cache_dir(tokenizer) or os.environ.get("HF_HUB_CACHE")
                if not cache_dir:
                    hf_home = os.environ.get("HF_HOME")
                    if hf_home:
                        cache_dir = os.path.join(hf_home, "hub")

                cached_path = _resolve_hub_repo_cached_file(
                    source,
                    "tokenizer.model",
                    token = token,
                    local_files_only = True,
                    cache_dir = cache_dir,
                    revision = _tokenizer_revision(tokenizer),
                )
                if cached_path is not None:
                    downloaded_path = cached_path
                else:
                    from huggingface_hub import hf_hub_download
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id = source,
                            filename = "tokenizer.model",
                            token = token,
                            local_files_only = _tokenizer_wants_local_only(tokenizer),
                            cache_dir = cache_dir,
                            revision = _tokenizer_revision(tokenizer),
                        )
                    except Exception:
                        downloaded_path = None

    if not os.path.isfile(tokenizer_model) and downloaded_path is not None:
        shutil.copy2(downloaded_path, tokenizer_model)
        logger.warning_once(
            f"Unsloth: Preserved sentencepiece asset `tokenizer.model` in {save_directory}."
        )


def _free_cached_model(model):
    from huggingface_hub import scan_cache_dir
    cached_repos = list(scan_cache_dir().repos)

    # Delete the cached repo matching the model being saved: saves 4GB of disk, useful on Kaggle.
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
        W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)
        if quant_state is not None:
            dtype = quant_state.dtype if type(quant_state) is not list else quant_state[2]
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype
        W = W.to(torch.float32).t()

        if A is not None:
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha = s)
            maximum_element = torch.max(W.min().abs(), W.max())
            if not torch.isfinite(maximum_element).item():
                raise ValueError(f"Unsloth: Merge failed.\n{name} has some elements = infinity.")
        W = W.t().to(dtype)
    else:
        W = layer.weight
    return W, bias


def fast_save_pickle(shard, name):
    print(f"Unsloth: Saving {name}...")
    torch.save(
        shard,
        name,
        # HIGHEST_PROTOCOL seems not to work with Pytorch.
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


def _is_vlm(model):
    config = getattr(model, "config", None)
    if config is None:
        return False
    architectures = getattr(config, "architectures", None) or ()
    return hasattr(config, "vision_config") or any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text")) for x in architectures
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
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
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

    if push_to_hub:
        from huggingface_hub import whoami
        try:
            username = whoami(token = token)["name"]
        except:
            raise RuntimeError(
                "Unsloth: Please supply a token!\nGo to https://huggingface.co/settings/tokens"
            )

    assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95

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

            _tokenizer.padding_side = old_padding_side

        if hasattr(model, "config"):
            print(f"Saved {save_method} model to https://huggingface.co/" + save_directory)
        return save_directory, None

    tokenizer_save_settings = {
        "save_directory": save_pretrained_settings["save_directory"],
        "legacy_format": None,
        "filename_prefix": None,
        "push_to_hub": save_pretrained_settings["push_to_hub"],
        "private": save_pretrained_settings["private"],
        "token": save_pretrained_settings["token"],
    }

    from peft import PeftModelForCausalLM

    if isinstance(model, PeftModelForCausalLM):
        internal_model = model.model
    else:
        internal_model = model

    if (
        (save_method == "merged_4bit")
        or (save_method == "lora")
        or (not hasattr(model, "model") or not hasattr(internal_model.model, "layers"))
    ):
        # _create_repo has errors due to **kwargs getting accepted, and commit_description does not seem to
        # work.
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

            _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            old_padding_side = _tokenizer.padding_side
            _tokenizer.padding_side = "left"

            tokenizer.save_pretrained(**tokenizer_save_settings)

            _tokenizer.padding_side = old_padding_side

            print(" Done.")
        else:
            print()

        print("Unsloth: Saving model...", end = "")
        if save_method != "lora":
            print(" This might take 10 minutes for Llama-7b...", end = "")

        # Correctness unverified.
        if save_method == "lora":
            save_pretrained_settings["selected_adapters"] = None

        model.save_pretrained(**save_pretrained_settings)

        if push_to_hub and hasattr(model, "config"):
            print("Saved to https://huggingface.co/" + save_pretrained_settings["save_directory"])

        print(" Done.")
        return save_directory, None

    # With push_to_hub the ".../" part of a repo must be removed; the +1 solves absolute path issues.
    username = None
    if push_to_hub and "/" in save_directory:
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

    if safe_serialization:
        max_ram -= sharded_ram_usage
    else:
        max_ram -= sharded_ram_usage * 0.25  # Uses much less

    max_ram = int(max(0, max_ram) * maximum_memory_usage)
    print(
        f"Unsloth: Will use up to "
        f"{round(max_ram / 1024 / 1024 / 1024, 2)} out of "
        f"{round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)} RAM for saving."
    )

    if IS_KAGGLE_ENVIRONMENT:
        temporary_location = os.path.join(KAGGLE_TMP, temporary_location)

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    # Kaggle and Colab allow only 20GB of disk, so free up 4GB of space.
    if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
        logger.warning_once(
            "Unsloth: Kaggle/Colab has limited disk space. We need to delete the downloaded\n"
            "model which will save 4-16GB of disk space, allowing you to save on Kaggle/Colab."
        )
        _free_cached_model(internal_model)

    from collections import OrderedDict

    state_dict = OrderedDict()

    torch_dtype = dtype_from_config(internal_model.config)
    if type(torch_dtype) is str:
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16

    state_dict["model.embed_tokens.weight"] = internal_model.model.embed_tokens.weight.data.to(
        torch_dtype
    )

    # A merged tensor lives on the GPU of its source layer, so budget against W's own device: a GPU0-only
    # check OOMs GPU1+ on a sharded model.
    _max_vram_by_device = {}

    def _device_vram_budget(dev):
        if dev.type != "cuda":
            return None
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if idx not in _max_vram_by_device:
            _max_vram_by_device[idx] = int(
                torch.cuda.get_device_properties(idx).total_memory * maximum_memory_usage
            )
        return _max_vram_by_device[idx]

    print("Unsloth: Saving model... This might take 5 minutes ...")

    from tqdm import tqdm as ProgressBar

    for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W, bias = _merge_lora(proj, name)

            if bias is not None:
                state_dict[f"model.layers.{j}.{item}.bias"] = bias

            _dev_budget = _device_vram_budget(W.device)
            if (
                _dev_budget is not None
                and (torch.cuda.memory_allocated(W.device) + W.nbytes) < _dev_budget
            ):
                # Already off-GPU: keeping it costs no VRAM
                state_dict[name] = W
            elif W.device.type != "cuda":
                # Fits on W's own GPU; already off-GPU means keeping it costs no VRAM.
                state_dict[name] = W
            # Saving to RAM seems to leak memory.
            else:
                logger.warning_once("\nWe will save to Disk and not RAM now.")
                filename = os.path.join(temporary_location, f"{name}.pt")
                torch.save(
                    W,
                    filename,
                    pickle_module = pickle,
                    pickle_protocol = pickle.HIGHEST_PROTOCOL,
                )
                # weights_only = True weirdly fails.
                state_dict[name] = torch.load(
                    filename, map_location = "cpu", mmap = True, weights_only = False
                )
        for item in LLAMA_LAYERNORMS:
            try:
                state_dict[f"model.layers.{j}.{item}.weight"] = eval(f"layer.{item}.weight.data")
            except:
                continue

    state_dict["model.norm.weight"] = internal_model.model.norm.weight.data

    if (
        internal_model.model.embed_tokens.weight.data_ptr()
        != internal_model.lm_head.weight.data_ptr()
    ):
        state_dict["lm_head.weight"] = internal_model.lm_head.weight.data.to(torch_dtype)

    # All tensors MUST be torch.Tensor and not torch.nn.parameter.Parameter.
    for key, value in state_dict.items():
        if hasattr(value, "data"):
            state_dict[key] = value = value.data
        if type(value) is not torch.Tensor:
            logger.warning_once(f"Unsloth: {key} is not a Tensor but a {type(value)}.")

    save_pretrained_settings["state_dict"] = state_dict

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

    save_directory = save_pretrained_settings["save_directory"]

    if save_pretrained_settings["push_to_hub"]:
        new_save_directory, new_username = _determine_username(save_directory, username, token)

        if token is not None:
            from huggingface_hub import whoami
            actual_username = whoami(token = token)["name"]
        else:
            actual_username = username

    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        tokenizer_save_settings["push_to_hub"] = False
        tokenizer_save_settings["save_directory"] = new_save_directory

    if tokenizer is not None:
        print("Unsloth: Saving tokenizer...", end = "")

        _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        old_padding_side = _tokenizer.padding_side
        _tokenizer.padding_side = "left"

        tokenizer.save_pretrained(**tokenizer_save_settings)
        _preserve_tokenizer_eos_token(
            tokenizer,
            tokenizer_save_settings["save_directory"],
            filename_prefix = tokenizer_save_settings.get("filename_prefix"),
        )

        _tokenizer.padding_side = old_padding_side

        print(" Done.")
    else:
        print()

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

    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        # Pushing to an organization: .save_pretrained does not work, so save locally first and upload
        # manually.
        save_pretrained_settings["save_directory"] = new_save_directory
        save_pretrained_settings["push_to_hub"] = False
        internal_model.save_pretrained(**save_pretrained_settings)

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
    # GPU conversion for GGUF weirdly breaks; see ggerganov/llama.cpp#7062.

    # Skip the make-clean probe on CMake-only checkouts, whose error output is misleading.
    IS_CMAKE = _is_cmake_only_llama_cpp("llama.cpp")

    if not IS_CMAKE:
        try:
            result = subprocess.run(
                ["make", "clean", "-C", "llama.cpp"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
            IS_CMAKE = result.returncode != 0
        except FileNotFoundError:
            IS_CMAKE = True

    if not IS_CMAKE:
        n_jobs = max(int((psutil.cpu_count() or 1) * 1.5), 1)
        full_command = ["make", "all", "-j" + str(n_jobs), "-C", "llama.cpp"]
    else:
        n_jobs = max(int(psutil.cpu_count() or 1), 1)
        check = os.system(
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}"
        )

        if check != 0:
            raise RuntimeError(
                f"*** Unsloth: Failed compiling llama.cpp using os.system(...) with error {check}. Please report this ASAP!"
            )
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
    run_installer = subprocess.Popen(
        full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT
    )
    return run_installer, IS_CMAKE


def install_python_non_blocking(packages = []):
    full_command = ["pip", "install"] + packages
    # GPU conversion for GGUF weirdly breaks; see ggerganov/llama.cpp#7062.
    run_installer = subprocess.Popen(
        full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT
    )
    return run_installer


# Cap the first-use auto-install at the exact vetted patch, so no inflated "0.999.0" or in-range "0.12.999"
# from a mirror is pulled. Floor 0.6.0 keeps torch>=2.4 resolvable (0.7+ need torch>=2.7).
_LLM_COMPRESSOR_SPEC = "llmcompressor>=0.6.0,<=0.12.0"

# Highest transformers llm-compressor 0.10.x/0.12.x runs against (it pins <=4.57.6). It imports
# TORCH_INIT_FUNCTIONS, removed in transformers 5.x, so a newer-transformers model dies with a cryptic
# ImportError AFTER the expensive merge. Bump with llm-compressor.
_LLM_COMPRESSOR_MAX_TRANSFORMERS = "4.57.6"


def _transformers_exceeds_llm_compressor_ceiling(transformers_version = None):
    """Return (exceeds, active_version) comparing the active transformers to the llm-compressor ceiling.

    `exceeds` is True only when we can parse both versions and the active transformers is strictly
    newer than `_LLM_COMPRESSOR_MAX_TRANSFORMERS`. Any parse failure returns False (fail open) so a
    real quantization attempt still surfaces the underlying error rather than a false positive.
    """
    if transformers_version is None:
        try:
            import transformers as _tf
            transformers_version = _tf.__version__
        except Exception:
            return False, "unknown"
    try:
        from packaging.version import parse as _parse

        # Drop any local build suffix ("4.57.6+abc") so it does not skew the comparison.
        active = _parse(str(transformers_version).split("+", 1)[0])
        ceiling = _parse(_LLM_COMPRESSOR_MAX_TRANSFORMERS)
        return active > ceiling, str(transformers_version)
    except Exception:
        return False, str(transformers_version)


# A caller (Unsloth Studio) can point this at an llm-compressor-main shadow (transformers>=5.9 over the
# existing torch); the subprocess then uses it and the ceiling check is bypassed.
_COMPRESSED_QUANTIZE_PYTHONPATH_ENV = "UNSLOTH_COMPRESSED_QUANTIZE_PYTHONPATH"


def _compressed_quantize_pythonpath():
    """Return the llm-compressor-main shadow PYTHONPATH, or None if not set."""
    pp = os.environ.get(_COMPRESSED_QUANTIZE_PYTHONPATH_ENV, "").strip()
    return pp or None


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

    # Prefer pip, falling back to uv when the interpreter has no pip seeded (uv-created or relocatable venvs),
    # so the export does not die on "No module named pip".
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
    # Download the 10th latest release since the latest might be broken; this is the fallback mechanism.
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

    if os.path.exists("llama.cpp"):
        print(
            "**[WARNING]** You have a llama.cpp directory which is broken.\n"
            "Unsloth will DELETE the broken directory and install a new one.\n"
            "Press CTRL + C / cancel this if this is wrong. We shall wait 30 seconds.\n"
        )
        import time

        for i in range(30):
            print(f"**[WARNING]** Deleting llama.cpp directory... {30 - i} seconds left.")
            time.sleep(1)

        shutil.rmtree("llama.cpp", ignore_errors = True)

    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        f"cd llama.cpp && git reset --hard {version} && git clean -df",
    ]
    try_execute(commands)

    use_cmake = _is_cmake_only_llama_cpp("llama.cpp")

    if not use_cmake:
        commands = [
            "make clean -C llama.cpp",
            f"make all -j{(psutil.cpu_count() or 1) * 2} -C llama.cpp",
        ]
        use_cmake = try_execute(commands) == "CMAKE"

    if use_cmake:
        commands = [
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}",
            f"cmake --build llama.cpp/build --config Release -j{(psutil.cpu_count() or 1) * 2} --clean-first --target {' '.join(LLAMA_CPP_TARGETS)}",
            "cp llama.cpp/build/bin/llama-* llama.cpp",
            "rm -rf llama.cpp/build",
        ]
        try_execute(commands)

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
    # GPU conversion for GGUF weirdly breaks; see ggerganov/llama.cpp#7062.

    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        "pip install gguf protobuf",
    ]
    if os.path.exists("llama.cpp"):
        return
    try_execute(commands)

    use_cmake = _is_cmake_only_llama_cpp("llama.cpp")

    if not use_cmake:
        commands = [
            "make clean -C llama.cpp",
            f"make all -j{(psutil.cpu_count() or 1) * 2} -C llama.cpp",
        ]
        use_cmake = try_execute(commands) == "CMAKE"

    if use_cmake:
        commands = [
            f"cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF {CURL_FLAG}",
            f"cmake --build llama.cpp/build --config Release -j{(psutil.cpu_count() or 1) * 2} --clean-first --target {' '.join(LLAMA_CPP_TARGETS)}",
            "cp llama.cpp/build/bin/llama-* llama.cpp",
            "rm -rf llama.cpp/build",
        ]
        try_execute(commands)


def get_executable(executables):
    system_directories = os.environ.get("PATH").split(os.pathsep)

    for directory in system_directories:
        for executable in executables:
            path = os.path.join(directory, executable)
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
    return None


# Output types convert_hf_to_gguf.py can emit directly via --outtype.
_DIRECT_CONVERT_OUTTYPES = ("f32", "f16", "bf16", "q8_0")
_FULL_PRECISION_GGUF_TYPES = ("f32", "f16", "bf16")
_GGUF_DEFAULT_SHARD_SIZE = "50GB"
_GGUF_NO_SHARDING = "0"
_GGUF_SHARD_SIZE_RE = re.compile(r"^(\d+)\s*([MG])B?$", re.IGNORECASE)


def _resolve_gguf_shard_size(gguf_shard_size: Optional[str]) -> str:
    """Validate and normalize the final GGUF shard size."""
    if gguf_shard_size is None:
        return _GGUF_DEFAULT_SHARD_SIZE
    if not isinstance(gguf_shard_size, str):
        raise TypeError("Unsloth: gguf_shard_size must be a string or None.")

    value = gguf_shard_size.strip()
    if value.casefold() in ("", "0", "none"):
        return _GGUF_NO_SHARDING

    match = _GGUF_SHARD_SIZE_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Unsloth: gguf_shard_size={gguf_shard_size!r} is invalid. "
            "Use a positive whole number in MB or GB, such as '500MB' or '4GB', "
            "or pass '0' for one file."
        )

    magnitude = int(match.group(1))
    unit = match.group(2).upper()
    if magnitude == 0:
        raise ValueError(
            "Unsloth: gguf_shard_size must be positive. Pass '0' without a unit "
            "to request one file."
        )
    multiplier = 1_000_000 if unit == "M" else 1_000_000_000
    if magnitude > sys.maxsize // multiplier:
        raise ValueError("Unsloth: gguf_shard_size is too large for this platform.")
    return f"{magnitude}{unit}B"


def _converter_gguf_shard_size(
    gguf_shard_size: str, first_conversion: str, quantization_methods, is_vlm: bool
) -> str:
    """Choose the converter limit without splitting final quantized files or VLM companions."""
    keeps_converter_output = first_conversion in quantization_methods
    if not is_vlm and keeps_converter_output and first_conversion in _FULL_PRECISION_GGUF_TYPES:
        return gguf_shard_size
    return _GGUF_NO_SHARDING


def _gguf_shard_size_bytes(gguf_shard_size: str) -> int:
    """Convert a normalized GGUF shard size to decimal bytes."""
    if gguf_shard_size == _GGUF_NO_SHARDING:
        return 0
    match = _GGUF_SHARD_SIZE_RE.fullmatch(gguf_shard_size)
    if match is None:
        raise ValueError(f"Unsloth: invalid normalized GGUF shard size {gguf_shard_size!r}.")
    multiplier = 1_000_000 if match.group(2).upper() == "M" else 1_000_000_000
    return int(match.group(1)) * multiplier


def _is_gguf_companion(path: Union[str, os.PathLike]) -> bool:
    """Return whether a GGUF is a vision or speculative-decoding companion."""
    file_path = Path(path)
    name = file_path.name.casefold()
    stem = name[:-5] if name.endswith(".gguf") else name
    return (
        stem.endswith("-mmproj")
        or stem.startswith("mmproj-")
        or stem.startswith("mtp-")
        or stem.endswith("-mtp")
        or file_path.parent.name.casefold() == "mtp"
    )


def _find_llama_gguf_split(quantizer_location: str) -> str:
    """Find the llama.cpp split utility beside supported install layouts."""
    executable = "llama-gguf-split.exe" if IS_WINDOWS else "llama-gguf-split"
    candidates = [shutil.which(executable)]
    quantizer_dir = os.path.dirname(os.path.abspath(quantizer_location))
    candidates.extend(
        [
            os.path.join(quantizer_dir, executable),
            os.path.join(LLAMA_CPP_DEFAULT_DIR, executable),
            os.path.join(LLAMA_CPP_DEFAULT_DIR, "build", "bin", executable),
            os.path.join(
                LLAMA_CPP_DEFAULT_DIR,
                "build",
                "bin",
                "Release",
                executable,
            ),
        ]
    )
    for candidate in dict.fromkeys(path for path in candidates if path):
        if os.path.isfile(candidate) and (IS_WINDOWS or os.access(candidate, os.X_OK)):
            return candidate
    raise RuntimeError(
        "Unsloth: post-conversion GGUF sharding requires llama-gguf-split. "
        "Upgrade unsloth_zoo and reinstall llama.cpp, then retry."
    )


def _split_main_gguf(initial_files, gguf_shard_size: str, quantizer_location: str):
    """Split one main GGUF while leaving mmproj and MTP companions untouched."""
    max_bytes = _gguf_shard_size_bytes(gguf_shard_size)
    if max_bytes == 0:
        return initial_files

    main_files = [os.fspath(path) for path in initial_files if not _is_gguf_companion(path)]
    if len(main_files) != 1:
        raise RuntimeError(
            "Unsloth: expected one unsharded main GGUF before companion-safe "
            f"splitting, found {len(main_files)}."
        )
    main_file = main_files[0]
    if os.path.getsize(main_file) <= max_bytes:
        return initial_files

    splitter = _find_llama_gguf_split(quantizer_location)
    parent = os.path.dirname(os.path.abspath(main_file))
    import tempfile

    with tempfile.TemporaryDirectory(prefix = ".unsloth_gguf_split_", dir = parent) as temp_dir:
        output_prefix = os.path.join(temp_dir, Path(main_file).stem)
        split_size = gguf_shard_size[:-1]
        try:
            result = subprocess.run(
                [
                    splitter,
                    "--split",
                    "--split-max-size",
                    split_size,
                    main_file,
                    output_prefix,
                ],
                check = True,
                capture_output = True,
                text = True,
            )
        except subprocess.CalledProcessError as exception:
            details = (exception.stderr or exception.stdout or "").strip()
            suffix = f"\n{details}" if details else ""
            raise RuntimeError(f"Unsloth: llama-gguf-split failed.{suffix}") from exception

        pattern = re.compile(
            rf"^{re.escape(Path(output_prefix).name)}-(\d{{5}})-of-(\d{{5}})\.gguf$"
        )
        shards = []
        for candidate in Path(temp_dir).iterdir():
            match = pattern.fullmatch(candidate.name)
            if match is not None:
                shards.append((int(match.group(1)), int(match.group(2)), candidate))
        shards.sort(key = lambda item: item[0])
        if not shards:
            details = (result.stderr or result.stdout or "").strip()
            suffix = f"\n{details}" if details else ""
            raise RuntimeError(f"Unsloth: llama-gguf-split produced no shards.{suffix}")

        total = shards[0][1]
        indices = [index for index, declared_total, _ in shards if declared_total == total]
        if len(indices) != len(shards) or indices != list(range(1, total + 1)):
            raise RuntimeError("Unsloth: llama-gguf-split produced an incomplete shard set.")
        if total == 1:
            return initial_files

        destinations = [os.path.join(parent, shard.name) for _, _, shard in shards]
        existing = [path for path in destinations if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                "Unsloth: refusing to overwrite an existing GGUF shard set: " + ", ".join(existing)
            )

        moved = []
        try:
            for (_, _, source), destination in zip(shards, destinations):
                os.replace(source, destination)
                moved.append(destination)
            os.unlink(main_file)
        except Exception:
            for destination in moved:
                try:
                    os.unlink(destination)
                except OSError:
                    pass
            raise

    output_files = []
    for path in initial_files:
        if os.path.abspath(os.fspath(path)) == os.path.abspath(main_file):
            output_files.extend(destinations)
        else:
            output_files.append(path)
    return output_files


def _choose_first_conversion(
    quantization_methods,
    model_dtype,
    has_imatrix = False,
):
    """Pick the dtype of the initial HF -> GGUF conversion.

    Single-pass fast path: when exactly one output type is requested and
    convert_hf_to_gguf.py can emit it directly (f32/f16/bf16/q8_0), convert straight to
    it - the llama-quantize pass and the 16-bit intermediate file are skipped entirely.
    An imatrix forces the two-pass route since only llama-quantize can apply one.

    Every other case converts to the source dtype first, so each requested method is
    quantized from weights identical to the checkpoint's.
    """
    unique_methods = set(quantization_methods)
    if len(unique_methods) == 1 and not has_imatrix:
        only_method = next(iter(unique_methods))
        if only_method in _DIRECT_CONVERT_OUTTYPES:
            return only_method
    return model_dtype


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
    gguf_directory: Optional[Union[str, os.PathLike]] = None,
    merge_is_disposable: bool = False,
    preexisting_weights = None,
    gguf_shard_size: Optional[str] = None,
):
    """
    Orchestrates the complete GGUF conversion process.
    Handles installation, conversion, and quantization.
    `imatrix` is a local importance-matrix path (already resolved); it is forwarded to
    llama-quantize and is required for the IQ low-bit quant types.
    `gguf_directory` can place outputs separately from the model input directory.
    `merge_is_disposable` says `model_directory` was written by this export purely to
    feed the converter, so its weights may be reclaimed if the quants would not
    otherwise fit. Off by default: a caller pointing this at a real checkpoint keeps it.
    `preexisting_weights` is what `model_directory` held before the merge wrote it,
    so reclamation can take only what this export produced. `None` means the caller
    cannot say, and nothing is reclaimed.
    `gguf_shard_size` applies to final f32, f16 and bf16 outputs. Quantized outputs
    remain single-file. None preserves the historical 50GB converter limit.
    """
    # print_output is True only if UNSLOTH_ENABLE_LOGGING=1.
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print_output = True
    else:
        print_output = False

    assert model_dtype == "float16" or model_dtype == "bfloat16"
    model_dtype = "f16" if model_dtype == "float16" else "bf16"

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

    if model_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "Unsloth: Cannot convert to bf16 GGUF since your computer doesn't support it.\n"
            "We shall switch instead to f16."
        )
        model_dtype = "f16"

    has_imatrix = imatrix is not None and str(imatrix) != ""
    if has_imatrix:
        # quantize_gguf gained the imatrix kwarg in a recent unsloth_zoo; fail fast before the expensive
        # conversion rather than silently dropping it.
        import inspect
        if "imatrix" not in inspect.signature(quantize_gguf).parameters:
            raise RuntimeError(
                "Unsloth: your installed unsloth_zoo's quantize_gguf does not support imatrix.\n"
                "Please upgrade it:  uv pip install --upgrade unsloth_zoo"
            )

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

    if is_gpt_oss:
        print("Unsloth: GPT-OSS model detected - using special conversion settings")
        first_conversion = "None"
        # GPT-OSS does not quantize, so keep only one conversion method.
        quantization_method = ["None"]
    elif first_conversion is None:
        first_conversion = _choose_first_conversion(
            quantization_method,
            model_dtype,
            has_imatrix = has_imatrix,
        )

    if first_conversion == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning("Unsloth: Switching bf16 to f16 due to hardware limitations")
        first_conversion = "f16"

    first_conversion_dtype = "" if first_conversion == "None" else first_conversion
    gguf_shard_size = _resolve_gguf_shard_size(gguf_shard_size)
    converter_shard_size = _converter_gguf_shard_size(
        gguf_shard_size,
        first_conversion,
        quantization_method,
        is_vlm,
    )
    needs_quantize_pass = any(m != first_conversion for m in quantization_method)
    if needs_quantize_pass:
        second_step = f"[2] Converting GGUF {first_conversion_dtype} to {quantization_method} might take 10 minutes each."
        total_line = "In total, you will have to wait at least 16 minutes."
    else:
        second_step = f"[2] Single-pass export: converting straight to {quantization_method} - no separate quantize step."
        total_line = "In total, you will have to wait at least 6 minutes."
    print_info = (
        f"==((====))==  Unsloth: Conversion from HF to GGUF information\n"
        f"   {chr(92)}{chr(92)}   /|    [0] Installing llama.cpp might take 3 minutes.\n"
        f"O^O/ {chr(92)}_/ {chr(92)}    [1] Converting HF to GGUF {first_conversion_dtype} might take 3 minutes.\n"
        f"{chr(92)}        /    {second_step}\n"
        f' "-____-"     {total_line}\n'
    )
    print(print_info)

    try:
        quantizer_location, converter_location = check_llama_cpp()
        print("Unsloth: llama.cpp found in the system. Skipping installation.")
    except:
        print("Unsloth: Installing llama.cpp. This might take 3 minutes...")
        if IS_KAGGLE_ENVIRONMENT:
            quantizer_location, converter_location = install_llama_cpp(
                gpu_support = False, print_output = print_output
            )
        else:
            # Kaggle: no CUDA support due to environment limitations.
            quantizer_location, converter_location = install_llama_cpp(
                gpu_support = False,
                print_output = print_output,
            )

    print("Unsloth: Preparing converter script...")
    with use_local_gguf():
        converter_path, supported_text_archs, supported_vision_archs = (
            _download_convert_hf_to_gguf()
        )

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
            max_shard_size = converter_shard_size,
            print_output = print_output,
        )
    is_vlm = is_vlm_update
    for file in initial_files:
        if not os.path.exists(file):
            # Gated like the outer handler: disk advice is only right when disk is the problem, and a
            # converter with no llama.cpp support fails here with space to spare.
            if IS_KAGGLE_ENVIRONMENT and _gguf_failure_looks_like_disk(
                RuntimeError(f"Conversion produced no output at {file}"),
                os.path.dirname(file) or None,
            ):
                raise RuntimeError(
                    f"Unsloth: Conversion failed for {file}\n"
                    "You are in a Kaggle environment with limited disk space (20GB).\n"
                    "Try saving to /tmp for more space or use a smaller model.\n"
                    "Alternatively, save the 16bit model first, then convert manually."
                )
            else:
                raise RuntimeError(
                    f"Unsloth: Conversion failed for {file}\nPlease check disk space and try again."
                )

    if gguf_directory is None:
        gguf_directory = f"{model_directory}_gguf"
    else:
        gguf_directory = os.fspath(gguf_directory)
    os.makedirs(gguf_directory, exist_ok = True)
    moved_files = []
    for fpath in initial_files:
        dst = os.path.join(gguf_directory, os.path.basename(fpath))
        shutil.move(fpath, dst)
        moved_files.append(dst)
    initial_files = moved_files

    if (
        is_vlm
        and first_conversion in _FULL_PRECISION_GGUF_TYPES
        and first_conversion in quantization_method
    ):
        initial_files = _split_main_gguf(
            initial_files,
            gguf_shard_size,
            quantizer_location,
        )

    print(f"Unsloth: Initial conversion completed! Files: {initial_files}")

    all_saved_locations = initial_files.copy()

    n_cpus = psutil.cpu_count()
    if n_cpus is None:
        n_cpus = 1
    n_cpus *= 2

    if not is_gpt_oss:
        base_gguf = initial_files[0]

        # Deduplicate while keeping order; a method equal to the base conversion is already on disk.
        methods_to_quantize = [
            m for m in dict.fromkeys(quantization_method) if m != first_conversion
        ]

        # The merge is not read again, and on a tight disk its bytes are what run the export out.
        # Nemotron-3-Nano-30B-A3B on 132GB: 63GB merge + 60GB GGUF + 18GB Q4_K_M = 141GB, and llama-quantize
        # died mid-write. Only fires when the room is not there.
        _free_merge_if_disk_is_tight(
            model_directory,
            gguf_directory,
            initial_files,
            quant_methods = methods_to_quantize,
            first_conversion = first_conversion,
            merge_is_disposable = merge_is_disposable,
            preexisting_weights = preexisting_weights,
        )

        def _quantize_one(quant_method, n_threads = None):
            output_location = os.path.join(
                gguf_directory, f"{model_name}.{quant_method.upper()}.gguf"
            )
            try:
                if quant_method == "q2_k_l":
                    return _quantize_q2_k_l(
                        input_gguf = base_gguf,
                        output_gguf = output_location,
                        quantizer_location = quantizer_location,
                        n_threads = n_threads if n_threads is not None else n_cpus,
                        print_output = print_output,
                        imatrix = imatrix,
                    )
                else:
                    # Standard unsloth-zoo quantization for everything else. Pass imatrix only when set, so an
                    # older zoo still handles plain quants; an inapplicable imatrix was rejected above.
                    quant_kwargs = dict(
                        input_gguf = base_gguf,
                        output_gguf = output_location,
                        quant_type = quant_method,
                        quantizer_location = quantizer_location,
                        print_output = print_output,
                    )
                    if has_imatrix:
                        quant_kwargs["imatrix"] = imatrix
                    if n_threads is not None:
                        quant_kwargs["n_threads"] = n_threads
                    return quantize_gguf(**quant_kwargs)
            except Exception as e:
                # Judge "no room" against what this pass writes, not a constant. Priced as a lower bound, not
                # the generous reclamation estimate: a high guess calls a disk full that had room.
                try:
                    _ratio = _gguf_output_size_ratio(
                        quant_method,
                        first_conversion,
                        upper_bound = False,
                    )
                    _needed = (
                        None
                        if _ratio is None
                        else int(
                            sum(
                                os.path.getsize(f)
                                for f in initial_files
                                if os.path.isfile(f) and not _is_gguf_companion(f)
                            )
                            * _ratio
                        )
                    )
                except OSError:
                    _needed = None
                # Same gate as above: a broken quantizer with 19GB free is not a disk problem, and the outer
                # handler cannot undo an explanation already baked into this message.
                if IS_KAGGLE_ENVIRONMENT and _gguf_failure_looks_like_disk(
                    e,
                    gguf_directory,
                    needed_bytes = _needed,
                    partial_output = output_location,
                ):
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
                    ) from e
                elif _gguf_failure_looks_like_disk(
                    e,
                    gguf_directory,
                    needed_bytes = _needed,
                    partial_output = output_location,
                ):
                    # Kaggle is not the only place a disk fills, and the rebuild advice below only fits a broken
                    # quantizer; on a full disk it is a long compile that fixes nothing.
                    try:
                        _free_gb = shutil.disk_usage(gguf_directory).free / 1024**3
                        _where = f" ({_free_gb:.1f}GB free at {gguf_directory})"
                    except OSError:
                        _where = ""
                    raise RuntimeError(
                        f"Unsloth: Quantization failed for {output_location}\n"
                        f"This looks like the disk running out, not a problem "
                        f"with llama.cpp{_where}.\n"
                        "The GGUF export needs room for the 16-bit merge, the "
                        "base GGUF and the quantized output at the same time.\n"
                        "Free some space, or save to a larger filesystem, then "
                        "run the quantization again.\n"
                        f"Error: {e}"
                    ) from e
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
                    ) from e  # keep the cause: the OOM check walks it for the returncode

        # Outputs already on disk pre-date this run; never delete them on a failure.
        preexisting_outputs = {
            m
            for m in methods_to_quantize
            if os.path.exists(os.path.join(gguf_directory, f"{model_name}.{m.upper()}.gguf"))
        }
        # Each llama-quantize pass loads the whole base GGUF into RAM, so run two at once only with headroom
        # for two copies, else a multi-quant export that fit sequentially OOMs.
        try:
            base_bytes = sum(os.path.getsize(f) for f in initial_files if not _is_gguf_companion(f))
            mem_ok = psutil.virtual_memory().available >= int(2.5 * base_bytes)
        except Exception:
            mem_ok = False
        # Independent llama-quantize runs can overlap, capped at 2 workers. Sequential when streaming logs, on
        # Kaggle/Colab, when RAM is tight, or when the kill switch is set.
        _parallel_flag = os.environ.get("UNSLOTH_PARALLEL_GGUF_QUANTS", "1").strip().lower()
        parallel_quants = (
            len(methods_to_quantize) > 1
            and not print_output
            and not IS_KAGGLE_ENVIRONMENT
            and not IS_COLAB_ENVIRONMENT
            and mem_ok
            and _parallel_flag not in ("0", "false", "no", "off", "")
        )
        if parallel_quants:
            max_workers = min(2, len(methods_to_quantize))
            # Split the thread budget so total threads match the sequential run.
            per_worker_threads = max(1, n_cpus // max_workers)
            print(
                f"Unsloth: [2] Converting GGUF {first_conversion_dtype} into "
                f"{methods_to_quantize}, {max_workers} at a time. This might take 10 minutes each..."
            )
            from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION

            quantized_files = [None] * len(methods_to_quantize)
            with ThreadPoolExecutor(max_workers = max_workers) as pool:
                future_to_idx = {
                    pool.submit(_quantize_one, method, per_worker_threads): i
                    for i, method in enumerate(methods_to_quantize)
                }
                done, pending = wait(future_to_idx, return_when = FIRST_EXCEPTION)
                # Do not start queued passes after a failure, to avoid filling the disk.
                for fut in pending:
                    fut.cancel()
                first_exc = next((f.exception() for f in done if f.exception() is not None), None)
                if first_exc is not None:
                    # Remove only outputs this run created: a pre-existing file, or a canceled pass that never
                    # wrote, stays, so a rerun never deletes a prior artifact. The base is kept for retry.
                    wait(future_to_idx)
                    for method in methods_to_quantize:
                        if method in preexisting_outputs:
                            continue
                        Path(
                            os.path.join(gguf_directory, f"{model_name}.{method.upper()}.gguf")
                        ).unlink(missing_ok = True)
                    raise first_exc
                for fut, i in future_to_idx.items():
                    quantized_files[i] = fut.result()
        else:
            quantized_files = []
            for quant_method in methods_to_quantize:
                print(
                    f"Unsloth: [2] Converting GGUF {first_conversion_dtype} into {quant_method}. This might take 10 minutes..."
                )
                quantized_files.append(_quantize_one(quant_method))

        all_saved_locations.extend(quantized_files)
        quants_created = len(quantized_files) > 0
        print("Unsloth: Model files cleanup...")
        want_full_precision = first_conversion in quantization_method
        if quants_created:
            # convert_to_gguf can return main shards plus companion files.
            base_files = [f for f in initial_files if not _is_gguf_companion(f)]
            if not want_full_precision:
                for f in base_files:
                    if f in all_saved_locations:
                        all_saved_locations.remove(f)
                    Path(f).unlink(missing_ok = True)

            # Flip the list to get [text_model, mmproj] order; text models stay the same.
            all_saved_locations.reverse()

            # When the base format is preserved, move base files (and shards) off the list boundaries so
            # example commands ([0]=model, [-1]=mmproj) stay correct.
            if want_full_precision and len(all_saved_locations) > len(base_files) + 1:
                for f in base_files:
                    if f in all_saved_locations:
                        all_saved_locations.remove(f)
                for i, f in enumerate(base_files):
                    all_saved_locations.insert(1 + i, f)

        for quant_method, quantized_file in zip(methods_to_quantize, quantized_files):
            if quant_method not in _FULL_PRECISION_GGUF_TYPES:
                continue
            split_files = _split_main_gguf(
                [quantized_file],
                gguf_shard_size,
                quantizer_location,
            )
            index = all_saved_locations.index(quantized_file)
            all_saved_locations[index : index + 1] = split_files
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

    # Kaggle's working directory is ~20GB while /tmp on the same kernel is terabytes, so relative paths under
    # /kaggle/working that do not fit move there. Absolute paths, hub pushes and non-Kaggle machines are
    # untouched.
    _forwards_state_dict, _writes_model_verbatim = _merge_writer_disposition(self, save_method)
    save_directory = _preflight_merge_disk(
        self,
        save_directory,
        save_method,
        push_to_hub = push_to_hub,
        state_dict = state_dict,
        forwards_state_dict = _forwards_state_dict,
        writes_model_verbatim = _writes_model_verbatim,
        # No writer_runs_merge_guard: a plain merge here is written by unsloth_save_model, which never reaches
        # merge_and_overwrite_lora. A compressed export does go through unsloth_generic_save, which the
        # preflight recognises from the method alone.
    )

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

    _torchao = _normalize_torchao_method(save_method)
    if _torchao is not None:
        kind, suffix = _torchao
        _unsloth_save_torchao(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            kind = kind,
            suffix = suffix,
            push_to_hub = push_to_hub,
            token = token,
            is_main_process = is_main_process,
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
    del arguments["_torchao"]
    del arguments["_forwards_state_dict"]
    del arguments["_writes_model_verbatim"]
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

    _torchao = _normalize_torchao_method(save_method)
    if _torchao is not None:
        kind, suffix = _torchao
        _unsloth_save_torchao(
            model = self,
            save_directory = repo_id,
            tokenizer = tokenizer,
            kind = kind,
            suffix = suffix,
            push_to_hub = True,
            token = token,
            is_main_process = True,
            private = private,
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
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
    del arguments["_torchao"]
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
        # Repo already exists, so update datasets metadata separately.
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
        if datasets:
            try:
                from huggingface_hub import metadata_update
                metadata_update(save_directory, {"datasets": datasets}, overwrite = True, token = token)
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Could not update datasets metadata for {save_directory}: {e}"
                )

    if file_location is not None:
        hf_api = HfApi(token = token)

        if "/" in file_location:
            uploaded_location = file_location[file_location.rfind("/") + 1 :]
        else:
            uploaded_location = file_location

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

            new_chat_template = re.sub(
                r"\{[\s]{0,}\{[\s]{0,}bos\_token[\s]{0,}\}[\s]{0,}\}", "", chat_template
            )
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
    tokenizer._ollama_modelfile = ollama_modelfile
    modelfile = ollama_modelfile

    FILE_LOCATION_REPLACER = "⚫@✅#🦥__FILE_LOCATION__⚡@🦥#⛵"
    EOS_TOKEN_REPLACER = "⚫@✅#🦥__EOS_TOKEN__⚡@🦥#⛵"
    LEFT_BRACKET_REPLACER = "⚫@✅#🦥"
    RIGHT_BRACKET_REPLACER = "⚡@🦥#⛵"

    # Convert all {'s and }'s but keep {__FILE_LOCATION__} intact, reverted below (#1087).
    modelfile = (
        modelfile.replace("{__FILE_LOCATION__}", FILE_LOCATION_REPLACER)
        .replace("{__EOS_TOKEN__}", EOS_TOKEN_REPLACER)
        .replace("{", LEFT_BRACKET_REPLACER)
        .replace("}", RIGHT_BRACKET_REPLACER)
    )

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


@contextmanager
def _hub_cache_prewarm_disabled(disable):
    """Turn the base-model cache pre-warm off for one export, then restore it.

    The pre-warm is an optimization for the NEXT export, so on a disk that
    cannot hold the cached base and the export at once, the export wins. The
    old value goes back afterwards, including on an exception, so nothing
    leaks into the rest of the session.
    """
    if not disable:
        yield
        return
    key = "UNSLOTH_PREWARM_HUB_CACHE"
    previous = os.environ.get(key, None)
    os.environ[key] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


# The zoo's merge guard compares against int(free * 0.95), and model_16bit_bytes counts tensors only, with
# config, tokenizer and index on top. Ask the redirect for that same effective figure, else a "just big
# enough" directory is kept and the merge then refuses.
_MERGE_FREE_SPACE_RESERVE = 0.95

# torchao weight-only fp8 / int8: one byte per quantized weight.
_TORCHAO_SIBLING_WEIGHT_BITS = 8


def _compressed_scheme_weight_bits(scheme):
    """Bits per weight of a compressed-tensors sibling checkpoint.

    Four for every scheme that quantizes weights to 4 bits (`W4*`, and any
    `*FP4` microscaling variant), eight for the rest. Rounded the way
    unsloth_zoo rounds its GGUF table: over-estimating costs headroom,
    under-estimating costs the whole export.
    """
    scheme = str(scheme).upper()
    return 4 if scheme.startswith("W4") or "FP4" in scheme else 8


def _compressed_ignore_patterns(model):
    """The compressed recipe's own `ignore` list for this model, or an empty list.

    Read out of `_compressed_quantize.py` rather than restated here, so the
    sizing cannot drift from the recipe the way it already had. That module is
    otherwise only ever *executed* by file path; its module-level imports are
    stdlib only, so importing it costs nothing and imports no llm-compressor.

    Any failure (module renamed, symbol gone, a config that will not answer)
    returns nothing, which leaves the estimate exactly as it was.
    """
    try:
        from unsloth._compressed_quantize import compressed_ignore_patterns
        return list(compressed_ignore_patterns(getattr(model, "config", None)))
    except Exception:
        return []


def _matches_ignore_pattern(name, module, patterns):
    """Mirror compressed-tensors' `is_match` for one module against `ignore`.

    `compressed_tensors.utils.match._match_name` treats a `re:` prefix as
    `re.match(pattern, name)` - anchored at the start of the fully qualified
    module name, and not required to reach its end - and everything else as
    `target == name`. `_match_class` additionally matches a plain entry
    against the names of the module's parent classes, which is why the MRO is
    walked here too.

    An unparseable pattern matches nothing rather than raising.
    """
    for pattern in patterns:
        try:
            if isinstance(pattern, str) and pattern.startswith("re:"):
                if re.match(pattern.removeprefix("re:"), name) is not None:
                    return True
            elif pattern == name:
                return True
            elif any(cls.__name__ == pattern for cls in type(module).__mro__):
                return True
        except Exception:
            continue
    return False


def _named_parameters(module):
    """`(name, parameter)` pairs, named where the module can say.

    `logical_numel` needs the name to spot MXFP4 packing, which has no
    `quant_state` to give it away and is only identifiable by the
    `_blocks` / `_scales` suffix. `torch.nn.Module` always offers
    `named_parameters`; the fall back to `parameters()` keeps any other object
    that answers the older call working, just without the MXFP4 case.
    """
    try:
        return list(module.named_parameters())
    except Exception:
        return [("", parameter) for parameter in module.parameters()]


def _unquantized_parameter_bytes(model, ignore_patterns = ()):
    """Bytes a weight-only export leaves at 16 bits in the sibling checkpoint.

    compressed-tensors and torchao both quantize `Linear` weights only, so the
    input embeddings and an untied `lm_head` stay 16-bit. They are the
    dominant exclusion (a 4.5B model measured 0.5B of them), so pricing the
    whole sibling at the scheme's width under-sizes it by the embedding share.

    `ignore_patterns` is the recipe's `ignore` list, and every module it names
    stays 16-bit as well: the vision tower of a VLM, a Qwen3-Next hybrid's
    linear-attention blocks, an MTP head, the MoE router gates. Those are the
    same order of magnitude as the embeddings, and pricing them at 4 or 8 bits
    under-sizes the merge, which is the direction that loses the redirect.
    Empty for torchao, which quantizes with no ignore list of its own.

    Every parameter is measured with `logical_numel`, the same helper
    `model_16bit_bytes` sizes the merge with, and NOT with `numel()`. A model
    loaded in 4 bits holds `Params4bit`, whose `numel()` is the packed uint8
    count and roughly half the logical one; MXFP4 blocks are worth twice
    theirs. Believing `numel()` would price an ignored subtree at half what
    the export writes for it while the merge it is subtracted from was priced
    logically, and the two have to agree.

    Tied embeddings are one tensor and are counted once, and a module already
    counted through an embedding getter or through an enclosing ignored module
    is not counted again. Zero when the model does not answer, which leaves
    the estimate exactly as it was.
    """
    total = 0
    seen = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            weight = getattr(getattr(model, getter)(), "weight", None)
            if weight is None or id(weight) in seen:
                continue
            seen.add(id(weight))
            total += logical_numel(weight) * 2
        except Exception:
            continue
    if not ignore_patterns:
        return total
    try:
        named_modules = list(model.named_modules())
    except Exception:
        # A model whose modules cannot be walked keeps the embeddings-only figure.
        return total
    for name, module in named_modules:
        try:
            if not _matches_ignore_pattern(name, module, ignore_patterns):
                continue
            for parameter_name, parameter in _named_parameters(module):
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                total += logical_numel(parameter, parameter_name) * 2
        except Exception:
            continue
    return total


def _quantized_sibling_bytes(
    model,
    merge_bytes,
    weight_bits,
    ignore_patterns = (),
):
    """Bytes of the quantized sibling written to `save_directory + "-<suffix>"`.

    `merge_bytes` is the 16-bit checkpoint size. Only the part of it that a
    weight-only scheme actually quantizes shrinks; the rest is copied across
    at 16 bits.
    """
    unquantized = min(_unquantized_parameter_bytes(model, ignore_patterns), merge_bytes)
    return int((merge_bytes - unquantized) * weight_bits / 16) + unquantized


def _full_model_checkpoint_bytes(model, state_dict = None):
    """Bytes `save_pretrained` writes for a model with no adapter to merge.

    Measured from the parameters' real storage rather than assumed to be a
    16-bit merge, because this checkpoint is written with no cast: an fp32
    model costs four bytes per parameter, a 4-bit one its packed storage.
    Zero when the model cannot be measured, which leaves the caller with
    today's behaviour.

    A caller-supplied `state_dict` is measured INSTEAD of the model, because
    it is what `save_pretrained` writes and the two can disagree by a whole
    factor: only `"16bit" in save_method` casts the dict, so a `"lora"` save
    on a model with no adapter forwards an fp32 dict verbatim while the
    resident parameters are fp16, and sizing the model would price the export
    at half. Undercounting here is not a crash but a missed `/tmp` redirect,
    which is how the 20GB Kaggle working directory fills instead.
    """
    try:
        # `is not None`, matching unsloth_generic_save, which forwards the dict on that test. An empty dict
        # reaches save_pretrained and writes no tensors, so sizing the resident model would price tens of
        # gigabytes for a save that writes none.
        if state_dict is not None:
            total = 0
            for tensor in state_dict.values():
                tensor = getattr(tensor, "data", tensor)
                total += tensor.numel() * tensor.element_size()
            return int(total)
        total = 0
        for parameter in model.parameters():
            total += parameter.numel() * parameter.element_size()
        return int(total)
    except Exception:
        return 0


def _cast_16bit_state_dict_bytes(state_dict):
    """Bytes a `"16bit"` save writes for a caller-supplied `state_dict`.

    `unsloth_generic_save` casts every FLOATING entry to bf16/fp16 and leaves
    the rest alone, then hands the dictionary straight to `save_pretrained`.
    So the floats are two bytes per element whatever they arrived as, and an
    integer or bool entry keeps its own width.

    `numel()` and not `logical_numel`, because these are the tensors the
    writer writes: a packed 4-bit entry is stored exactly as it stands here,
    not expanded.
    """
    total = 0
    for tensor in state_dict.values():
        tensor = getattr(tensor, "data", tensor)
        try:
            floating = bool(tensor.is_floating_point())
        except Exception:
            floating = False
        total += tensor.numel() * (2 if floating else tensor.element_size())
    return int(total)


def _same_filesystem(left, right):
    """True when two paths sit on the same mount.

    Neither path has to exist. The nearest existing ancestor answers, exactly
    as `free_bytes` resolves the disk it measures, because that ancestor is
    the filesystem a write to the path really lands on. A destination is
    routinely absent here: `_preflight_merge_disk` runs before anything is
    created, so stat-ing it directly raised, the callers' broad handlers
    swallowed the probe, and a first export got no TMPDIR warning at all.

    Raises when either side cannot be identified, which is what every caller
    already reads as "cannot tell".
    """
    left_id = _filesystem_id(left)
    right_id = _filesystem_id(right)
    if left_id is None or right_id is None:
        raise OSError(f"Unsloth: cannot identify the filesystem holding `{left}` or `{right}`.")
    return left_id == right_id


def _destination_holds_torchao_staging(destination, need_bytes, staging_bytes):
    """Can the redirect target hold the torchao staging merge as well?

    `_unsloth_save_torchao` merges into `tempfile.mkdtemp()` and removes it
    only once quantization has finished, so where the tempfile default and the
    redirect target are the same filesystem - which is exactly a Kaggle kernel,
    where both are /tmp - the staging checkpoint and the quantized sibling sit
    on the destination at the same time.

    Checked here rather than added to `need_bytes`, because the staging
    directory never lands in /kaggle/working: charging it there would relocate
    exports that fit into /tmp, which is not kept as notebook output. So this
    can only ever cancel a redirect, never cause one, and it cancels one only
    when the destination could not have held the export anyway - in which case
    staying put leaves the working directory's own guard to raise the real
    error.
    """
    import tempfile
    try:
        if staging_bytes <= 0:
            return True
        if not _same_filesystem(tempfile.gettempdir(), destination):
            return True
        free = free_bytes(destination)
        if free is None:
            return True
        return free >= need_bytes + staging_bytes
    except Exception:
        return True


def _warn_if_sibling_filesystem_is_short(save_directory, suffix, sibling_bytes):
    """Say so when the quantized sibling lands on a disk nobody measured.

    Compressed and torchao exports build their output lexically, as
    `save_directory + "-<suffix>"`, so the sibling is a child of the PARENT of
    `save_directory`. That is the same filesystem the preflight probed, unless
    `save_directory` is itself a symlink or a mount point. When it is, the
    probe answers for the target and the sibling is written on the other side:
    a `model` symlinked into a roomy /tmp passes here while `model-fp8` fills
    a 20GB working directory.

    A warning rather than a different redirect, for the same reason as the
    staging one. `kaggle_tmp_redirect` relocates `save_directory`, and the
    caller derives the sibling from whatever this function returns, so a
    redirect already moves both. The uncovered case is the one where NO
    redirect fires, and there is nothing to cancel there.
    """
    try:
        if sibling_bytes <= 0 or not suffix:
            return
        sibling = f"{save_directory}-{suffix}"
        # The directory the sibling is created IN, which is what has the space.
        holder = os.path.dirname(os.path.abspath(sibling)) or "."
        if _same_filesystem(holder, save_directory):
            return
        free = free_bytes(holder)
        if free is None or free >= sibling_bytes:
            return
        print(
            f"Unsloth: `{holder}` has {free / 1024**3:.1f}GB free and the quantized "
            f"`{os.path.basename(sibling)}` needs about {sibling_bytes / 1024**3:.1f}GB there.\n"
            f"`{save_directory}` resolves to a different filesystem, and the sibling is "
            f"written next to the name rather than next to the target, so the room at "
            f"`{save_directory}` does not help.\n"
            f"Save to a path on the roomy filesystem itself rather than through a link."
        )
    except Exception:
        return


def _warn_if_torchao_staging_filesystem_is_short(destination, staging_bytes):
    """Say so when TMPDIR cannot hold the staging merge either.

    `_destination_holds_torchao_staging` asks whether the DESTINATION can hold
    the staging directory as well, which is the whole question while
    `tempfile` resolves onto the same filesystem. When it does not, that
    function returns True and nothing has measured the staging filesystem at
    all, so a 4GB tmpfs is handed a 60GB merge and `_unsloth_save_torchao`
    dies inside `tempfile.mkdtemp` without naming TMPDIR as the reason.

    A warning, not a refusal: the preflight never raises, and unlike the
    destination check this cannot be answered by cancelling the redirect.
    The staging merge is written to TMPDIR whether or not the export was
    relocated, so declining the move leaves the identical failure and puts
    the output on the smaller disk as well.
    """
    import tempfile
    try:
        if staging_bytes <= 0:
            return
        staging_directory = tempfile.gettempdir()
        if _same_filesystem(staging_directory, destination):
            return
        free = free_bytes(staging_directory)
        # The staging merge is written by merge_and_overwrite_lora, which refuses it unless free * 0.95 covers
        # it, so the bare size leaves a 5% band where the merge dies silently.
        needed = math.ceil(staging_bytes / _MERGE_FREE_SPACE_RESERVE)
        if free is None or free >= needed:
            return
        print(
            f"Unsloth: `{staging_directory}` has {free / 1024**3:.1f}GB free and the "
            f"16-bit staging merge needs about {needed / 1024**3:.1f}GB there.\n"
            f"The torchao export merges into a temporary directory before it quantizes, "
            f"and that directory is on a different filesystem from `{destination}`, so "
            f"the room at the destination does not help.\n"
            f"Point TMPDIR at a filesystem with the space if the export runs out."
        )
    except Exception:
        return


def _warn_if_a_cancelled_redirect_leaves_no_room(save_directory, sibling_bytes):
    """Say so when the export is handed back a filesystem that cannot hold it.

    Cancelling the torchao redirect returns the quantized sibling to
    `save_directory`, and outside `UNSLOTH_KAGGLE_USE_TMP=1` the only reason
    the redirect fired at all is that that filesystem measured too small for
    it. Nothing downstream covers the sibling: the torchao merge is staged in
    TMPDIR, so `merge_and_overwrite_lora`'s `free * 0.95` measures the staging
    disk and never this one, and the sibling is written at the very end of a
    long quantization. Cancelling is still the right move - /tmp cannot hold
    the staging merge and the sibling together, so relocating fails too - but
    it must not be silent for the minutes the quantization takes.

    A warning and not a refusal, for the same reason as the two warnings
    beside it: this preflight never raises, and the sibling is an estimate.
    """
    try:
        if sibling_bytes <= 0:
            return
        free = free_bytes(save_directory)
        if free is None or free >= sibling_bytes:
            return
        print(
            f"Unsloth: `{save_directory}` has {free / 1024**3:.1f}GB free and the quantized "
            f"checkpoint needs about {sibling_bytes / 1024**3:.1f}GB there.\n"
            f"The export was left here rather than moved to a larger filesystem, because "
            f"that filesystem cannot hold the temporary 16-bit merge as well.\n"
            f"Free space here, or point TMPDIR at a filesystem with room for the merge, "
            f"before the quantization runs: it is written at the end of the export."
        )
    except Exception:
        return


def _merge_writer_disposition(model, save_method):
    """What `save_pretrained_merged`'s writer does with a supplied `state_dict`.

    Returns `(forwards, verbatim)` for the `_preflight_merge_disk` arguments of
    the same meaning. Three writers sit behind that one entrypoint and they do
    not agree:

      - a compressed-tensors or torchao export hands the dictionary to
        `unsloth_generic_save(save_method = "merged_16bit")`, which casts every
        floating entry to two bytes and writes it. That checkpoint is the
        staging (torchao) or kept (compressed) merge, so it is sized from the
        dictionary, not from the resident model,
      - `unsloth_save_model` on an architecture that walks `.model.layers`
        rebuilds the dictionary from the merged layers and drops whatever it
        was handed, so sizing the caller's would price a save that is not
        happening,
      - `unsloth_save_model` on any other architecture takes its generic
        fallback and calls `save_pretrained(**save_pretrained_settings)` with
        the caller's dictionary untouched AND uncast, so the bytes written are
        the dictionary's own, at its own dtypes.

    Never raises: an unreadable model or an unrecognised method reports the
    conservative `(False, False)`, which is the behaviour before this existed.
    """
    method = str(save_method).lower().strip().replace("-", "_").replace(" ", "_")
    try:
        compressed = _normalize_compressed_method(method)
        torchao = _normalize_torchao_method(method)
    except Exception:
        return False, False
    if compressed is not None or torchao is not None:
        return True, False
    if method != "merged_16bit":
        return False, False
    # A PeftModel in the generic fallback writes ADAPTERS, not a checkpoint, so it keeps its own sizing rather
    # than being priced a full model.
    if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        return False, False
    try:
        # The same predicate unsloth_save_model dispatches on, and no other.
        takes_generic_fallback = not hasattr(model, "model") or not hasattr(
            getattr(model, "model", None), "layers"
        )
    except Exception:
        return False, False
    return (True, True) if takes_generic_fallback else (False, False)


def _preflight_merge_disk(
    model,
    save_directory,
    save_method,
    push_to_hub = False,
    state_dict = None,
    forwards_state_dict = False,
    writes_model_verbatim = False,
    writer_runs_merge_guard = False,
):
    """Kaggle only: send a merge that cannot fit in /kaggle/working to /tmp.

    Never raises. A merge that is short of space on an ordinary machine is
    already handled by unsloth_zoo's own guard, which knows about shard
    streaming and the push_to_hub fallbacks; this exists purely so the ~20GB
    Kaggle working directory stops being the ceiling when a terabyte of
    overlay is mounted next to it.

    Skipped entirely when pushing to the hub, because there `save_directory`
    is a repo id like "user/model", not a filesystem path, and rewriting it
    would push to the wrong repository.

    `forwards_state_dict` says the writer behind this call hands a supplied
    `state_dict` to `save_pretrained` for a 16-bit save rather than building
    its own, casting its floating entries to two bytes on the way, as
    `unsloth_generic_save` does. `unsloth_save_model`'s merge path instead
    rebuilds the dictionary from the merged layers and drops whatever it was
    given, so sizing the caller's there would price a save that is not
    happening.

    `writes_model_verbatim` says the writer copies what it holds to
    `save_pretrained` with no cast at all, which is `unsloth_save_model`'s
    generic architecture fallback. Then the checkpoint is the dictionary's own
    bytes, at its own dtypes, or the resident parameters' own bytes when no
    dictionary was supplied, and never two bytes per logical parameter.
    `_merge_writer_disposition` decides both for the public entrypoint.

    `writer_runs_merge_guard` says the writer behind this call is
    `unsloth_generic_save`, whose PEFT branch is the one and only caller of
    `merge_and_overwrite_lora` here and therefore the only writer that brings
    its `free * 0.95` guard with it. It is deliberately separate from the two
    flags above, which decide SIZING: a compressed export is cast to two bytes
    by that same writer and is sized accordingly, yet it reserves nothing
    unless there is an adapter for the guard to merge.
    """
    if push_to_hub:
        return save_directory
    # unsloth_save_model normalizes spaces before dispatching, so "merged 16bit" is the same full merge as
    # "merged_16bit" and must be measured as one.
    method = str(save_method).lower().strip().replace("-", "_").replace(" ", "_")
    try:
        # Every compressed export keeps the 16-bit merge at save_directory and writes a quantized sibling
        # beside it, so all of them belong here, not just mxfp4.
        compressed = _normalize_compressed_method(method)
    except Exception:
        # An unsupported near-miss name raises later, where the message is.
        return save_directory
    # The torchao portable exports have the same shape: a quantized sibling at save_directory + "-<suffix>".
    torchao = _normalize_torchao_method(method)
    # "lora" on a model with no adapter writes the WHOLE model: both save paths fall back to save_pretrained,
    # so a full fine-tune asked for "lora" fills /kaggle/working like a merge. A real PeftModel writes
    # adapters only and is still skipped.
    is_peft = isinstance(model, (PeftModel, PeftModelForCausalLM))
    full_model_lora = method == "lora" and not is_peft
    # unsloth_generic_save reaches for model.state_dict() only when given none, so a model with no adapter is
    # saved from whatever dict it holds; a PeftModel goes to merge_and_overwrite_lora, which takes no state dict
    # at all.
    supplied_dict = state_dict if (forwards_state_dict and not is_peft) else None
    # Same reason the dict is only followed for a non-PEFT model: a PeftModel never reaches the fallback that
    # would copy one verbatim.
    verbatim = bool(writes_model_verbatim) and not is_peft
    if compressed is None and torchao is None and method != "merged_16bit" and not full_model_lora:
        return save_directory
    try:
        # A merge writes 2 bytes per parameter and no GGUF, so this is model_16bit_bytes, not the GGUF
        # estimate, which always prices an intermediate conversion. The no-adapter save writes the dictionary
        # it was handed, cast, so an empty one writes almost nothing and sizing the model would relocate it
        # off persistent Kaggle storage.
        if full_model_lora or verbatim:
            need = _full_model_checkpoint_bytes(model, state_dict)
        elif supplied_dict is not None:
            need = _cast_16bit_state_dict_bytes(supplied_dict)
        else:
            need = model_16bit_bytes(model)
        if need <= 0:
            return save_directory
        # What the torchao staging directory costs on whatever filesystem tempfile resolves to; zero for every
        # other export, which stages nothing.
        staging = 0
        # The lexical suffix the quantized sibling is written under, so its filesystem can be measured
        # separately below.
        sibling_suffix = ""
        sibling_bytes = 0
        # What merge_and_overwrite_lora writes HERE, the only part its free * 0.95 guard measures. Split out
        # from `need` because the reserve belongs on this alone: charging it around the whole estimate
        # relocates an export that fits. Every other writer here is a bare save_pretrained and reserves
        # nothing.
        guard_runs_here = is_peft and (compressed is not None or bool(writer_runs_merge_guard))
        merge_here = need if guard_runs_here else 0
        if torchao is not None:
            # _unsloth_save_torchao merges into a tempfile.mkdtemp staging dir, so only the 8-bit sibling lands
            # here and no merge guard runs against this filesystem.
            staging = need
            merge_here = 0
            # No ignore list: _unsloth_save_torchao quantizes with a bare Float8/Int8WeightOnlyConfig(), and
            # charging it the compressed recipe's exclusions would over-count and relocate an export that fits.
            need = _quantized_sibling_bytes(model, need, _TORCHAO_SIBLING_WEIGHT_BITS)
            sibling_suffix = torchao[1]
            sibling_bytes = need
        elif compressed is not None:
            sibling_bytes = _quantized_sibling_bytes(
                model,
                need,
                _compressed_scheme_weight_bits(compressed[0]),
                # Everything the recipe refuses to quantize is copied at 16 bits: vision tower, linear
                # attention, MTP, MoE gates.
                _compressed_ignore_patterns(model),
            )
            sibling_suffix = compressed[2]
            need += sibling_bytes
        # The reserve raises the ask for the merge alone; the rest is added at face value. max, not a sum: the
        # sibling coexists with the merge, so the peak is the whole estimate and it must also clear the
        # merge's reserved figure.
        need = max(need, math.ceil(merge_here / _MERGE_FREE_SPACE_RESERVE))
        new_directory, message = kaggle_tmp_redirect(
            save_directory,
            need_bytes = need,
            what = "model checkpoint" if (full_model_lora or verbatim) else "16-bit merge",
        )
    except Exception:
        return save_directory
    if message is not None:
        if not _destination_holds_torchao_staging(new_directory, need, staging):
            _warn_if_a_cancelled_redirect_leaves_no_room(save_directory, need)
            _warn_if_torchao_staging_filesystem_is_short(save_directory, staging)
            _warn_if_sibling_filesystem_is_short(save_directory, sibling_suffix, sibling_bytes)
            return save_directory
        print(message)
        _warn_if_torchao_staging_filesystem_is_short(new_directory, staging)
        _warn_if_sibling_filesystem_is_short(new_directory, sibling_suffix, sibling_bytes)
        return new_directory
    _warn_if_torchao_staging_filesystem_is_short(save_directory, staging)
    _warn_if_sibling_filesystem_is_short(save_directory, sibling_suffix, sibling_bytes)
    return save_directory


def _normalize_quantization_methods(quantization_method):
    """The list of GGUF types an export will actually write.

    Mirrors the normalisation `save_to_gguf` does, but cheaply and without
    validating, because this only feeds a size estimate: an unrecognised name
    is sized as q8_0 rather than rejected here, and the real validation still
    happens later where it always did.
    """
    if quantization_method is None:
        return []
    if isinstance(quantization_method, str):
        methods = [quantization_method]
    elif isinstance(quantization_method, (list, tuple)):
        methods = list(quantization_method)
    else:
        return []
    out = []
    for method in methods:
        if method is None:
            method = "q8_0"
        method = str(method).lower()
        if method == "not_quantized":
            method = "f16"
        elif method == "fast_quantized":
            method = "q8_0"
        elif method == "quantized":
            method = "q4_k_m"
        out.append(method)
    return out


def _imatrix_is_enabled(imatrix_file):
    """Whether an imatrix will really be applied, as `_resolve_imatrix_file` reads it.

    None and False both disable it. The preflight has to agree: an imatrix
    forces the two-pass route, so reading False as "enabled" sizes an
    intermediate GGUF that a direct-convertible export never writes, and can
    refuse an export that fits.
    """
    return imatrix_file is not None and imatrix_file is not False


def _gguf_writes_16bit_checkpoint(model):
    """Whether a GGUF export writes a full 16-bit checkpoint before converting.

    A PEFT model is merged into one. A non-PEFT model reuses an existing
    checkpoint when `_name_or_path` names a directory, and otherwise falls
    back to `save_pretrained`, which writes the same two bytes per parameter
    the merge would have. Sizing that fallback at zero is what lets an export
    pass the preflight and then fill the disk with the checkpoint.

    A module-level helper rather than a local, because the caller snapshots
    `locals()` into the kwargs of `unsloth_generic_save`.
    """
    if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        return True
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    try:
        return not (name_or_path and os.path.isdir(str(name_or_path)))
    except Exception:
        return True


def _fallback_checkpoint_extra_bytes(model):
    """Bytes the non-PEFT fallback checkpoint costs ON TOP of the 16-bit estimate.

    `estimate_gguf_export_bytes` budgets two bytes per logical parameter for
    the checkpoint, which is what a LoRA merge writes. The non-PEFT fallback
    calls `self.save_pretrained` with no cast, so a model loaded with
    `dtype = torch.float32` (a supported load) writes four and can fill a disk
    the preflight called big enough.

    Measured from the parameters' real storage, so a mixed-dtype model is not
    priced off its largest tensor, and clamped at zero: this can only ever ask
    for more room, never less, and an unmeasurable model adds nothing.
    """
    if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        return 0
    if not _gguf_writes_16bit_checkpoint(model):
        return 0
    try:
        actual = 0
        for parameter in model.parameters():
            actual += parameter.numel() * parameter.element_size()
        return max(0, actual - model_16bit_bytes(model))
    except Exception:
        return 0


def _gguf_output_directory(save_directory):
    """Where the GGUF files land: a SIBLING of `save_directory`, not a child.

    One definition for the preflight and for the export itself, so the disk
    that gets measured cannot drift from the disk that gets written.
    """
    return f"{save_directory}_gguf"


def _filesystem_id(path):
    """Device id of the filesystem `free_bytes(path)` would measure, or None.

    Mirrors `free_bytes`: lexical `abspath`, so a symlinked directory keeps
    its own name, then the nearest existing ancestor, because the GGUF
    sibling does not exist yet. `os.stat` follows symlinks, so this is the
    device `shutil.disk_usage` reports for the very same path. Windows fills
    `st_dev` from the volume serial number; a zero is the platform declining
    to answer and counts as unmeasurable.
    """
    try:
        probe = os.path.abspath(os.path.expanduser(str(path)))
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return os.stat(probe).st_dev or None
    except Exception:
        return None


def _on_separate_filesystems(directory, sibling):
    """True only when both paths are identified AND on different filesystems.

    Unmeasurable is False rather than a guess, so a path nothing can identify
    leaves the caller charging one filesystem for the whole export, which is
    what it did before this existed. Device ids and not "the sibling reports
    less free space": two `disk_usage` calls on ONE filesystem can disagree
    when something else writes between them, and reading that as two
    filesystems would charge a single-filesystem export the larger of the two
    halves instead of their sum.

    Separate from `_same_filesystem`, which resolves paths the same way but
    answers a different question for the torchao redirect: it raises when a
    path cannot be identified, because its callers want that to cancel the
    probe rather than to charge one filesystem for both halves.
    """
    left = _filesystem_id(directory)
    right = _filesystem_id(sibling)
    if left is None or right is None:
        return False
    return left != right


def _shares_filesystem(directory, sibling):
    """True only when both paths are identified AND on the same filesystem.

    Not `not _on_separate_filesystems(...)`: that reads an unidentifiable path
    as "together", which is right where the answer removes a charge and wrong
    where it adds one. This is the predicate for the adding case, so it says
    no to anything it cannot see, and neither predicate ever guesses.
    """
    left = _filesystem_id(directory)
    right = _filesystem_id(sibling)
    if left is None or right is None:
        return False
    return left == right


def _directory_is_writable(directory):
    """Can a file be created here? The same probe `convert_to_gguf` makes.

    `tempfile.mkstemp` is exclusive, so it never truncates an existing file
    and never follows a symlink. Anything that goes wrong reads as "no".
    """
    import tempfile
    try:
        handle, probe = tempfile.mkstemp(prefix = ".unsloth_write_test_", dir = directory)
        os.close(handle)
        os.remove(probe)
        return True
    except Exception:
        return False


def _gguf_conversion_directory(model_directory):
    """Where the intermediate GGUF is written, before it is moved.

    `unsloth_zoo.llama_cpp.convert_to_gguf` passes a BARE filename as
    `--outfile`, so llama.cpp resolves it against the process CWD; the only
    fallback to the input folder fires when that CWD cannot be written to.
    `save_to_gguf` then `shutil.move`s the finished file into the `_gguf`
    directory, which is a copy and not a rename when the two sit on different
    filesystems.

    So on a Kaggle kernel the intermediate - two bytes per parameter, the
    largest single staging artefact - still lands in the 20GB /kaggle/working
    even after the export has been redirected to /tmp, and that is the one
    filesystem the rest of this preflight never measures.

    None when the CWD cannot be read, and then nothing is charged.
    """
    try:
        cwd = os.getcwd()
    except Exception:
        return None
    return cwd if _directory_is_writable(cwd) else model_directory


def _gguf_model_input_directory(model, save_directory):
    """The folder the converter reads, which is not always `save_directory`.

    A non-PEFT model whose `_name_or_path` names a directory is converted from
    that checkpoint: `unsloth_save_pretrained_gguf` reassigns `save_directory`
    to it before calling `save_to_gguf`, so it is that path which arrives as
    `convert_to_gguf`'s `input_folder`. The same condition
    `_gguf_writes_16bit_checkpoint` uses to decide no merge is written.

    It matters only where the input folder is also written to, which is the
    unwritable-CWD fallback: the intermediate GGUF then lands beside the
    reused checkpoint rather than beside the requested output, and those two
    can be on different filesystems.
    """
    if isinstance(model, (PeftModel, PeftModelForCausalLM)):
        return save_directory
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    try:
        if name_or_path and os.path.isdir(str(name_or_path)):
            return str(name_or_path)
    except Exception:
        pass
    return save_directory


def _merge_reclamation_is_possible(save_directory):
    """Will `_free_merge_if_disk_is_tight` have a merge to reclaim?

    It never touches a file that was in `save_directory` before the export
    started, because that file is the caller's, not this export's. A directory
    already holding a checkpoint under the names a merge writes therefore
    yields nothing, and sizing that export as though the merge would go would
    pass one that really does peak at all three artefacts.

    A directory that does not exist yet is entirely this export's own, so
    True. One that cannot be listed is False, which is what the reclamation
    itself does with unreadable provenance.
    """
    try:
        names = os.listdir(save_directory)
    except FileNotFoundError:
        return True
    except Exception:
        return False
    try:
        return not _merge_weight_files(save_directory, names)
    except Exception:
        return False


def _hub_cache_directory():
    """Where `_prewarm_base_model_hub_cache` downloads the base model.

    Resolved from the live environment exactly as the pre-warm does, and not
    from huggingface_hub's frozen constants, so a runtime cache redirect is
    followed here too. Falls back to the constant, then to None, and None
    leaves every caller charging the cache where it charged it before.
    """
    try:
        from unsloth_zoo.hf_cache import _active_caches
        cache = _active_caches()[1]
        if cache is not None:
            return str(cache)
    except Exception:
        pass
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return str(HF_HUB_CACHE) or None
    except Exception:
        return None


def _gguf_source_dtype(model):
    """The initial-conversion dtype `save_to_gguf` will derive from the model.

    `estimate_gguf_export_bytes` omits a requested output that EQUALS the
    initial conversion, because that file is already on disk and gets no
    quantize pass. So the preflight has to name the same dtype the export
    will: told "f16" for a bf16 model asked for `["f16", "q4_k_m"]`, it charges
    the f16 file as the intermediate and nothing else, while `save_to_gguf`
    writes a bf16 intermediate AND a separate f16 output. That is one whole
    checkpoint unaccounted for (15.3GB on Qwen3-8B).

    Mirrors `save_to_gguf` step for step: `dtype_from_config`, mapped to the
    f16 / bf16 names it uses, then the same drop to f16 on hardware with no
    bf16. Returns "f16" whenever anything cannot be read, which is the same
    fallback the exporter prints and takes.
    """
    try:
        model_dtype = dtype_from_config(model.config)
        if type(model_dtype) is str:
            dtype = "bf16" if model_dtype == "bfloat16" else "f16"
        elif model_dtype == torch.bfloat16:
            dtype = "bf16"
        else:
            dtype = "f16"
    except Exception:
        return "f16"
    if dtype == "bf16":
        try:
            if not torch.cuda.is_bf16_supported():
                return "f16"
        except Exception:
            # save_to_gguf calls this unguarded, so a raise means no export at all and the figure is never used.
            # "f16" is the same width either way; only the name has to match.
            return "f16"
    return dtype


def _preflight_gguf_disk(
    model,
    save_directory,
    quantization_method,
    first_conversion = None,
    model_dtype = "f16",
    has_imatrix = False,
    needs_merge = True,
    merge_is_disposable = False,
):
    """Refuse a GGUF export that cannot fit, before it writes a single byte.

    Returns `(directory, prewarm_ok)`. `directory` differs from the input
    only when a Kaggle kernel's tiny working directory was swapped for the
    large /tmp overlay (and then it says so, once). `prewarm_ok` is False when
    the export fits only if the Hugging Face cache is not pre-warmed with the
    base model first.

    A GGUF export peaks at more than "the model, twice". It caches the
    full-precision base, writes the 16-bit HF merge, then an intermediate
    GGUF at the source dtype, then each requested quant, and every earlier
    artefact is still on disk while the next is written. Gemma4 (26B A4B)
    Vision, Gemma4 (31B) Vision and Qwen3 32B each trained, ran inference and
    completed `merged_16bit` before dying partway through a GGUF shard,
    because the check in front of them had sized the job at two copies.

    Dropping the pre-warm is tried before refusing, because it is a pure
    optimization for the NEXT export - the merge downloads what it needs
    either way - and an export that runs is worth more than a cache.

    `merge_is_disposable` says the merge is this export's own throwaway, so
    `_free_merge_if_disk_is_tight` may delete it once the intermediate GGUF
    exists. Then the three artefacts never coexist and the peak is the larger
    of the two phases, not their sum. Defaults off: charging the aggregate is
    what every caller got before this argument existed.

    Never blocks on a guess: an unmeasurable model or an unmeasurable disk
    returns the directory untouched. Set UNSLOTH_DISK_PREFLIGHT=0 to disable.
    """
    if os.environ.get("UNSLOTH_DISK_PREFLIGHT", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return save_directory, True

    try:
        methods = _normalize_quantization_methods(quantization_method)
        if first_conversion is None or not isinstance(first_conversion, str):
            first_conversion = _choose_first_conversion(
                methods, model_dtype, has_imatrix = has_imatrix
            )
        # save_to_gguf drops a bf16 initial conversion to f16 on hardware without bf16 AFTER resolving one, so
        # the drop must be applied after both branches, not only in the resolver. The estimate omits an output
        # equal to the initial conversion, so ["bf16"] at first_conversion "bf16" was priced as one file while
        # a T4 writes f16 plus bf16: 15.3GB missing on Qwen3-8B.
        if first_conversion == "bf16":
            try:
                if not torch.cuda.is_bf16_supported():
                    first_conversion = "f16"
            except Exception:
                # save_to_gguf asks unguarded, so a raise means no export runs and this is never used. f16 is
                # the wider of the two readings, never the narrower.
                first_conversion = "f16"
        need = estimate_gguf_export_bytes(
            model = model,
            quantization_methods = methods,
            first_conversion = first_conversion,
            needs_merge = needs_merge,
        )
        # The pre-warm runs only on the merge path and only while enabled. Kaggle and Colab return before it,
        # so pricing a cache copy there sends an export that fits in /kaggle/working to a /tmp that is not
        # kept as notebook output.
        prewarm_possible = (
            needs_merge
            and not IS_KAGGLE_ENVIRONMENT
            and not IS_COLAB_ENVIRONMENT
            and os.environ.get("UNSLOTH_PREWARM_HUB_CACHE", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        need_with_cache = (
            estimate_gguf_export_bytes(
                model = model,
                quantization_methods = methods,
                first_conversion = first_conversion,
                needs_merge = needs_merge,
                base_cache_copy = True,
            )
            if prewarm_possible
            else need
        )
        # The estimate prices the checkpoint at 2 bytes per parameter, but the non-PEFT fallback writes the
        # model's own dtype, so an fp32 model needs the difference. Zero for a 16-bit model.
        if need > 0 and needs_merge:
            extra = _fallback_checkpoint_extra_bytes(model)
            need += extra
            need_with_cache += extra
        # The same estimate without the checkpoint: the `_gguf` sibling's intermediate plus every quant. Used
        # only when that sibling sits on a smaller filesystem. Its own try, so an estimator that cannot answer
        # leaves the main guard standing.
        try:
            need_sibling = estimate_gguf_export_bytes(
                model = model,
                quantization_methods = methods,
                first_conversion = first_conversion,
                needs_merge = False,
            )
        except Exception:
            need_sibling = 0
        # The first phase a disposable merge peaks at: merge plus intermediate GGUF, before any quant.
        # quantization_methods = () still prices the intermediate, which must exist to be read.
        try:
            need_merge_phase = estimate_gguf_export_bytes(
                model = model,
                quantization_methods = (),
                first_conversion = first_conversion,
                needs_merge = needs_merge,
            ) + (extra if need > 0 and needs_merge else 0)
        except Exception:
            need_merge_phase = 0
        # Reclamation only helps when a quantize pass follows, and save_to_gguf skips a method equal to the
        # initial conversion, whose file is already on disk.
        has_quantize_pass = bool([m for m in dict.fromkeys(methods) if m != first_conversion])
        # The intermediate conversion on its own, for the filesystem it is written to before the move.
        try:
            need_conversion = estimate_gguf_export_bytes(
                model = model,
                quantization_methods = (),
                first_conversion = first_conversion,
                needs_merge = False,
            )
        except Exception:
            need_conversion = 0
    except Exception:
        # Sizing is best effort: a failure here must not stop an export that would otherwise have worked.
        return save_directory, True
    if need <= 0:
        return save_directory, True

    # Ask the redirect for the same figure the refusal reads. need_with_cache is the aggregate, but a
    # disposable merge is reclaimed before the quants, so a phased peak that fits in /kaggle/working was
    # relocated. Read before any move, and it can only lower the ask.
    phased_need = 0
    # The intermediate conversion is written to the process CWD, not into `save_directory` and not into the
    # `_gguf` sibling, and only afterwards moved. When that CWD is on its own filesystem, nothing above has
    # measured the disk the largest staging artefact actually lands on: on Kaggle it is the 20GB working
    # directory the redirect just moved the export away from.
    # The intermediate conversion is written to the working directory and only moved to the sibling afterwards,
    # so when that working directory is on THIS filesystem the checkpoint and the conversion are on it together.
    # Charging each of them alone lets a 60GB checkpoint and a 60GB conversion both pass on 100GB, and then it
    # fills. Only the split branch needs this: on one filesystem the aggregate already counts them both, and a
    # conversion sharing the sibling's disk is counted in `need_sibling`. `max` and not `+`, because the two
    # figures are two PHASES and not two artefacts. The merge runs first and its guard wants the checkpoint over
    # 0.95 with nothing else written yet; the conversion is written after, against the unreserved checkpoint.
    # Summing them charges the reserve on top of a conversion that does not exist while the guard runs: a 60GB
    # merge and a 60GB conversion on 122GB free clear both phases (63.2GB, then 120GB) and the sum asks 123.2GB
    # and refuses.
    if (
        merge_is_disposable
        and needs_merge
        and has_quantize_pass
        and need_merge_phase > 0
        and need_sibling > 0
        and not _on_separate_filesystems(save_directory, _gguf_output_directory(save_directory))
    ):
        redirect_peak = max(need_merge_phase, need_sibling)
        if redirect_peak < need:
            phased_need = redirect_peak + max(0, need_with_cache - need)

    redirect_need = need_with_cache
    if phased_need > 0 and _merge_reclamation_is_possible(save_directory):
        redirect_need = phased_need

    new_directory, message = kaggle_tmp_redirect(
        save_directory,
        need_bytes = redirect_need,
        what = "GGUF export",
    )
    # Asked the aggregate and declined, because this directory already holds weights reclamation may not
    # touch. After the move the merge is reclaimable, so re-ask at the phased peak: 141GB aggregate declines a
    # 130GB /tmp, but the real 123GB peak fits. Gated on this directory being too small for the aggregate,
    # else a lower ask could cancel the move for nothing.
    if message is None and 0 < phased_need < redirect_need:
        free_before_move = free_bytes(save_directory)
        if free_before_move is not None and free_before_move < need:
            new_directory, message = kaggle_tmp_redirect(
                save_directory,
                need_bytes = phased_need,
                what = "GGUF export",
            )
    if message is not None:
        print(message)
        save_directory = new_directory

    free = free_bytes(save_directory)

    # The quants and intermediate GGUF go to a SIBLING of save_directory, on the parent's filesystem, which is
    # the disk just measured unless save_directory is a mount or symlink. When split, charge each filesystem
    # only what it holds. One predicate, computed once, so the two halves cannot disagree.
    gguf_directory = _gguf_output_directory(save_directory)
    gguf_free = free_bytes(gguf_directory)
    separate_storage = (
        free is not None
        and gguf_free is not None
        and _on_separate_filesystems(save_directory, gguf_directory)
    )
    # Resolved before the split is priced, because where the conversion lands decides which filesystem it is
    # charged to.
    conversion_directory = _gguf_conversion_directory(
        _gguf_model_input_directory(model, save_directory)
    )

    # Cleared when the cache shares a filesystem with room for the export but not a cached base too: dropping
    # the optional half beats failing. The message travels with the flag, since more than one filesystem can
    # set it.
    sibling_prewarm_ok = True
    prewarm_drop_message = None
    # Resolved once, above the split, since the cache need not be on either filesystem. Zero whenever the pre-
    # warm cannot run, which leaves every branch below inert.
    cache_extra = max(0, need_with_cache - need)
    cache_directory = _hub_cache_directory() if cache_extra > 0 else None
    # The cache lands wherever HF_HOME points, not necessarily save_directory's disk. Charging it here drops
    # the pre-warm on a disk that had room, and the next export re-downloads the base. It only LOWERS the
    # figure and only the pre-warm reads it.
    cache_here = cache_extra
    if (
        cache_extra > 0
        and cache_directory is not None
        and _on_separate_filesystems(cache_directory, save_directory)
    ):
        cache_here = 0
    need_here = need
    need_here_with_cache = need + cache_here
    if separate_storage:
        # need_sibling is this estimate without the checkpoint, so the difference is the checkpoint alone; an
        # unsizable sibling leaves it 0, giving the previous behaviour. Both terms carry DISK_SLACK_BYTES and
        # it cancels, so merge_and_overwrite_lora's free * 0.95 reserve must be re-applied or this passes an
        # export the merge kills seconds later.
        checkpoint_here = max(0, need - need_sibling)
        need_here = checkpoint_here
        # The cache copy is not part of the merge and not what the zoo guard measures, so it rides on top of
        # the checkpoint. It need not land HERE either: with save_directory mounted elsewhere, ~/.cache and
        # the _gguf sibling stay on the parent disk.
        cache_sibling = 0
        if (
            cache_extra > 0
            and cache_here == 0
            and _shares_filesystem(cache_directory, gguf_directory)
        ):
            cache_sibling = cache_extra
        need_here_with_cache = checkpoint_here + cache_here
        writes_a_lora_merge = isinstance(model, (PeftModel, PeftModelForCausalLM))
        if writes_a_lora_merge and needs_merge and need_sibling > 0 and checkpoint_here > 0:
            # Four conditions, each dropping a charge for a guard that will not run: only a PEFT model reaches
            # merge_and_overwrite_lora's free * 0.95, a bare self.save_pretrained reserves nothing,
            # needs_merge = False writes no merge, and an unsizable sibling leaves a fallback figure.
            # min(need, ...) keeps the split's promise: cancel a redirect, never cause a refusal.
            reserved = min(need, math.ceil(checkpoint_here / _MERGE_FREE_SPACE_RESERVE))
            need_here = max(need_here, reserved)
            # The cache copy is written BEFORE the merge and is still there when the guard runs, so it comes
            # off the free space the guard sees. Riding on top of the reserved figure rather than beside it:
            # 16GB checkpoint + 14GB cache on 30.5GB passes a max of 30GB and then the merge refuses under its
            # own 5%. Added instead, this drops the pre-warm, not the export.
            need_here_with_cache = max(need_here_with_cache, reserved + cache_here)
        # The intermediate conversion lands in the working directory before the move, so on THIS filesystem
        # the checkpoint and conversion coexist and charging each alone lets 60GB + 60GB pass on 100GB. max,
        # not +: they are two phases, and summing them refuses 122GB free that clears both.
        if (
            need_conversion > 0
            and conversion_directory is not None
            and _shares_filesystem(conversion_directory, save_directory)
        ):
            need_here = max(need_here, checkpoint_here + need_conversion)
            need_here_with_cache = max(
                need_here_with_cache, checkpoint_here + need_conversion + cache_here
            )
        # No longer gated on the sibling being the TIGHTER of the two: now that the checkpoint is charged only
        # its own portion, a sibling roomier than save_directory but under need_sibling must be refused here,
        # since the aggregate comparison is gone.
        if gguf_free < need_sibling:
            raise RuntimeError(
                f"Unsloth: Not enough disk space to convert to GGUF.\n"
                f"The GGUF files are written to `{gguf_directory}`, which is on a different "
                f"filesystem from `{save_directory}` and has {gguf_free / 1024**3:.1f}GB free; "
                f"the intermediate `{first_conversion}` conversion and the quants need about "
                f"{need_sibling / 1024**3:.1f}GB there.\n"
                f"Options: free space on that filesystem, export fewer quantization methods, or "
                f"point `save_directory` at a path whose parent directory has the room.\n"
                f"To skip this check set the environment variable UNSLOTH_DISK_PREFLIGHT=0."
            )
        if cache_sibling > 0 and gguf_free < need_sibling + cache_sibling:
            sibling_prewarm_ok = False
            prewarm_drop_message = (
                f"Unsloth: Skipping the Hugging Face cache pre-warm - the Hugging Face "
                f"cache is on the same filesystem as `{gguf_directory}`, which has "
                f"{gguf_free / 1024**3:.1f}GB free: enough for the GGUF files "
                f"(~{need_sibling / 1024**3:.1f}GB) but not for a cached copy of the base "
                f"model as well. The next export will download the base again."
            )
    elif (
        merge_is_disposable
        and needs_merge
        and has_quantize_pass
        and need_merge_phase > 0
        and need_sibling > 0
        and _merge_reclamation_is_possible(save_directory)
    ):
        # _free_merge_if_disk_is_tight deletes this export's merge once the intermediate exists, so the three
        # artefacts never coexist and the peak is the larger of the two phases, not their sum:
        # Nemotron-3-Nano-30B-A3B peaks at 123GB where the 141GB aggregate refuses a 132GB disk. Single
        # filesystem only, since reclamation declines across devices.
        peak = max(need_merge_phase, need_sibling)
        if peak < need:
            # Can only lower the figure, so nothing the head allowed is refused here. The cache copy is not
            # reclaimed, so it rides on top of the peak as it did on the sum.
            need_here = peak
            need_here_with_cache = peak + cache_here

    # The intermediate conversion is written to the process CWD and only then moved, so when that CWD is its
    # own filesystem nothing above has measured the disk the largest staging artefact lands on -- on Kaggle,
    # the 20GB working directory the redirect just left.
    if (
        need_conversion > 0
        and conversion_directory is not None
        and _on_separate_filesystems(conversion_directory, gguf_directory)
    ):
        conversion_free = free_bytes(conversion_directory)
        # The pre-warm leaves the base model cached for the rest of the export, so on a shared filesystem the
        # cached base and the intermediate coexist, which cache_here and cache_sibling never charge. The pre-
        # warmer's own gate asks for two base copies, and an f32 conversion is two copies on its own: 38.1GB
        # free clears both checks and runs out 7.6GB short.
        cache_with_conversion = (
            cache_extra
            if cache_directory is not None
            and _shares_filesystem(cache_directory, conversion_directory)
            else 0
        )
        if conversion_free is not None and conversion_free < need_conversion:
            raise RuntimeError(
                f"Unsloth: Not enough disk space to convert to GGUF.\n"
                f"The intermediate `{first_conversion}` conversion is written to the current "
                f"working directory `{conversion_directory}` before it is moved to "
                f"`{gguf_directory}`, and that filesystem has "
                f"{conversion_free / 1024**3:.1f}GB free; the conversion needs about "
                f"{need_conversion / 1024**3:.1f}GB there.\n"
                f"Options: free space on that filesystem, or `os.chdir(...)` to a directory "
                f"on the same filesystem as the export.\n"
                f"`.push_to_hub_gguf(...)` does not avoid this one: it exports through a "
                f"temporary directory but never changes the working directory, so the "
                f"conversion is written here either way.\n"
                f"To skip this check set the environment variable UNSLOTH_DISK_PREFLIGHT=0."
            )
        if (
            conversion_free is not None
            and cache_with_conversion > 0
            and conversion_free < need_conversion + cache_with_conversion
        ):
            # Only reached once the conversion alone cleared the raise above, so this cannot turn a refusal into
            # a pass: it drops the optional half of an export that otherwise fits.
            sibling_prewarm_ok = False
            prewarm_drop_message = (
                f"Unsloth: Skipping the Hugging Face cache pre-warm - the Hugging Face "
                f"cache is on the same filesystem as the working directory "
                f"`{conversion_directory}`, which has "
                f"{conversion_free / 1024**3:.1f}GB free: enough for the intermediate "
                f"`{first_conversion}` conversion (~{need_conversion / 1024**3:.1f}GB) but "
                f"not for a cached copy of the base model as well. The next export will "
                f"download the base again."
            )

    if free is None or free >= need_here_with_cache:
        if not sibling_prewarm_ok and prewarm_possible and prewarm_drop_message:
            print(prewarm_drop_message)
        return save_directory, sibling_prewarm_ok

    if free >= need_here:
        if prewarm_possible:
            print(
                f"Unsloth: Skipping the Hugging Face cache pre-warm - "
                f"{free / 1024**3:.1f}GB free is enough for this GGUF export "
                f"(~{need_here / 1024**3:.1f}GB) but not for a cached copy of the base "
                f"model as well. The next export will download the base again."
            )
        return save_directory, False

    raise RuntimeError(
        f"Unsloth: Not enough disk space to convert to GGUF.\n"
        f"The export needs about {need_here / 1024**3:.1f}GB on the filesystem holding "
        f"`{save_directory}`, which has {free / 1024**3:.1f}GB free.\n"
        + (
            f"Only the 16-bit merge is charged here; the intermediate "
            f"`{first_conversion}` conversion and the quants go to a sibling directory "
            f"on another filesystem.\n"
            if separate_storage
            else f"It writes a 16-bit merge, then a `{first_conversion}` GGUF, then "
            f"{', '.join(_normalize_quantization_methods(quantization_method)) or 'no'} "
            f"quants, and the merge and the intermediate are both still on disk while "
            f"the quants are written.\n"
        )
        + f"Options: free space, export fewer quantization methods, point "
        f"`save_directory` at a bigger filesystem, or push straight to Hugging Face "
        f"with `.push_to_hub_gguf(...)`.\n"
        f"To skip this check set the environment variable UNSLOTH_DISK_PREFLIGHT=0."
    )


def _offloaded_parameter_hint(model):
    """Sentence to append when a save failed on offloaded (meta) parameters.

    Accelerate leaves offloaded parameters on the meta device, so saving dies
    inside accelerate with "'NoneType' object is not subscriptable" or "Cannot
    copy out of meta tensor", neither of which names the offload. Returns ""
    when no meta parameter is present, so unrelated failures are not
    mislabelled.
    """
    try:
        meta = []
        for name, tensor in model.named_parameters():
            if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                meta.append(name)
                # A 30B MoE has thousands; listing them would bury the error.
                if len(meta) >= 3:
                    break
        if not meta:
            return ""
        return (
            f" Unsloth: this model has parameters on the meta device "
            f"(offloaded because it did not fit the GPU), for example "
            f"{', '.join(meta)}. Saving needs the real weights, which the "
            f"offload hooks do not expose here. Re-run on a GPU large enough "
            f"to hold the model without offloading, or reload it with "
            f"`device_map` pinned to a single device before saving."
        )
    except Exception:
        # A diagnostic must never replace the real error with its own.
        return ""


def _model_basename(name_or_path, default = "model") -> str:
    """Leaf name of a model id or path, for use as a GGUF filename stem.

    Strips `\\` as well as `/` on every host: `os.path.basename` returns the whole
    `D:\\...` string on POSIX. A directory or drive left in the stem makes
    `os.path.join(gguf_directory, stem)` discard gguf_directory under ntpath, so the
    GGUF lands next to the base model (#7897); an empty stem gives a hidden
    `.Q4_K_M.gguf` that `glob.glob` cannot see.
    """
    if name_or_path is None:
        return default
    try:
        text = os.fspath(name_or_path)
    except TypeError:
        text = str(name_or_path)
    if not isinstance(text, str) or not text.strip():
        return default

    # A real directory wins: a POSIX directory name may legally contain a backslash.
    try:
        if os.path.isdir(text):
            base = os.path.basename(os.path.normpath(text))
            if base and base not in (".", ".."):
                return base
    except (OSError, ValueError):
        pass

    base = text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    # A bare drive ("D:") or "." would give a drive-relative or hidden output file.
    if not base or base in (".", "..") or (len(base) == 2 and base[1] == ":"):
        return default
    return base


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
    merge_is_disposable: bool = True,
    gguf_shard_size: Optional[str] = None,
):
    """
    Same as .save_pretrained(...) except 4bit weights are auto
    converted to float16 then converted to GGUF / llama.cpp format.

    imatrix_file: importance matrix for llama-quantize. None = off; a path = use that file
    (a *.gguf_file is renamed to *.gguf); True = download the upstream unsloth/<base>-GGUF
    imatrix. Required for the IQ low-bit quants (iq2_xxs, iq4_xs, ...).

    merge_is_disposable: the 16-bit merge written into `save_directory` exists only to feed
    the converter, so it may be reclaimed if the quants would otherwise not fit. Pass False
    to keep the weights when `save_directory` is part of the caller's own deliverable (the
    SentenceTransformer export writes its module directory there).

    gguf_shard_size: maximum final f32, f16 or bf16 GGUF shard size in MB or GB. Pass
    "0" for one file. None preserves the historical 50GB converter limit.

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
    save_directory = os.path.normpath(os.fspath(save_directory))
    gguf_shard_size = _resolve_gguf_shard_size(gguf_shard_size)

    # save_method="lora" exports the adapter itself as a GGUF LoRA, not a merged model.
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

    # base_model_name keeps the full id for create_ollama_modelfile's mapper lookup; only the filename stem is
    # trimmed.
    base_model_name = getattr(getattr(self, "config", None), "_name_or_path", None)
    try:
        base_model_name = get_model_name(base_model_name, load_in_4bit = False)
    except Exception:
        pass
    model_name = _model_basename(base_model_name)

    if push_to_hub:
        raise ValueError(
            "Unsloth: Please use .push_to_hub_gguf() instead of .save_pretrained_gguf() with push_to_hub=True"
        )

    is_vlm = _is_vlm(self)

    is_processor = is_vlm and isinstance(tokenizer, ProcessorMixin)

    is_gpt_oss = _is_gpt_oss(self)

    # Will this fit? Ask before the merge. Runs here because `arguments` below snapshots locals(), so a
    # redirected save_directory must be in place first. gpt-oss takes the mxfp4 route, not the
    # merge/convert/quantize one this sizes.
    _gguf_prewarm_ok = True
    if not is_gpt_oss:
        save_directory, _gguf_prewarm_ok = _preflight_gguf_disk(
            model = self,
            save_directory = save_directory,
            quantization_method = quantization_method,
            first_conversion = first_conversion,
            # Resolved rather than left at the default, which says "f16" while the export asks the config.
            model_dtype = _gguf_source_dtype(self),
            has_imatrix = _imatrix_is_enabled(imatrix_file),
            needs_merge = _gguf_writes_16bit_checkpoint(self),
            # The same flag save_to_gguf reclaims on. Where a non-PEFT model reuses its own checkpoint the
            # flag is cleared below on the same condition, so the two cannot disagree.
            merge_is_disposable = merge_is_disposable,
        )

    arguments = dict(locals())
    arguments["model"] = self
    arguments["tokenizer"] = tokenizer
    arguments["push_to_hub"] = False  # We handle upload ourselves
    # GPT-OSS needs the mxfp4 save method.
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
    del arguments["_gguf_prewarm_ok"]  # a local decision, not a save_pretrained kwarg
    del arguments["merge_is_disposable"]  # decides reclamation, not how the merge is written
    del arguments["gguf_shard_size"]  # only used by the gguf converter

    # Preserve the requested output before reusing a non-PEFT checkpoint as input. Same definition the
    # preflight sized, so it measured the disk these files land on.
    gguf_directory = _gguf_output_directory(save_directory)

    if is_processor:
        fix_bos_token, old_chat_template = fix_tokenizer_bos_token(tokenizer.tokenizer)
    else:
        fix_bos_token, old_chat_template = fix_tokenizer_bos_token(tokenizer)

    # Resolve the imatrix (download, validate, rename *.gguf_file) up front, so a bad path or an unavailable
    # upstream fails before the expensive merge and never reaches the IQ-quant gate.
    imatrix_path = _resolve_imatrix_file(self, imatrix_file, token, save_directory)

    # Settle ownership before a byte is written: reclamation may only take files this export made, and
    # afterwards the directory cannot say, since a reused output directory can already hold a finished sharded
    # save transformers neither removes nor distinguishes from ours.
    try:
        preexisting_weights = frozenset(os.listdir(save_directory))
    except FileNotFoundError:
        # Nothing there yet, so everything that appears is this export's own.
        preexisting_weights = frozenset()
    except OSError:
        # Provenance unreadable. Proving nothing, the reclamation takes nothing.
        preexisting_weights = None

    is_peft_model = isinstance(self, PeftModelForCausalLM) or isinstance(self, PeftModel)

    # The flag holds for both branches that write the weights; the middle branch reuses the user's own
    # checkpoint and clears it, whatever the caller asked for.
    if is_peft_model:
        print(f"Unsloth: Merging model weights to {'mxfp4' if is_gpt_oss else '16-bit'} format...")
        try:
            with _hub_cache_prewarm_disabled(not _gguf_prewarm_ok):
                unsloth_generic_save(**arguments)

        except Exception as e:
            raise RuntimeError(
                f"Failed to save/merge model: {_describe_exception(e)}"
                f"{_offloaded_parameter_hint(self)}"
            ) from e
    else:
        # Non-PEFT model: the checkpoint already exists, so point save_to_gguf at the original path instead of
        # re-saving into a temp subdir.
        original_path = getattr(self.config, "_name_or_path", None)
        if original_path and os.path.isdir(original_path):
            print(
                f"Unsloth: Model is not a PEFT model. Using existing checkpoint at {original_path}"
            )
            save_directory = original_path
            # The user's own checkpoint, not an intermediate: without this a save_pretrained_gguf on a tight
            # disk would delete the model it was handed.
            merge_is_disposable = False
            # Persist tokenizer fixes (BOS stripping) so the GGUF converter reads the corrected template.
            # Persist tokenizer fixes (e.g. BOS token stripping) to disk so the GGUF converter picks up the
            # corrected chat template.
            if tokenizer is not None:
                tokenizer.save_pretrained(save_directory)
        else:
            print("Unsloth: Model is not a PEFT model. Saving directly without LoRA merge...")
            os.makedirs(save_directory, exist_ok = True)
            # `gguf_directory` can point anywhere, and freeing bytes on one filesystem does nothing for a
            # quantize pass writing to another: without this the merge could be deleted for a destination it
            # cannot help, data gone and the export still out of space.
            try:
                self.save_pretrained(save_directory)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_directory)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save model: {_describe_exception(e)}"
                    f"{_offloaded_parameter_hint(self)}"
                ) from e

    if is_processor:
        tokenizer = tokenizer.tokenizer

    if fix_bos_token:
        tokenizer.chat_template = old_chat_template

    for _ in range(3):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        print(f"Unsloth: Could not determine dtype ({e}), defaulting to float16")
        model_dtype = "float16"

    print("Unsloth: Converting to GGUF format...")

    quantization_methods = []
    if quantization_method is not None:
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
            if quant_method is None:
                quant_method = "q8_0"
            else:
                quant_method = quant_method.lower()
            if quant_method == "not_quantized":
                quant_method = "f16"
            elif quant_method == "fast_quantized":
                quant_method = "q8_0"
            elif quant_method == "quantized":
                quant_method = "q4_k_m"
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
            gguf_directory = gguf_directory,
            merge_is_disposable = merge_is_disposable,
            preexisting_weights = preexisting_weights,
            gguf_shard_size = gguf_shard_size,
        )
    except Exception as e:
        if _gguf_child_was_oom_killed(e):
            raise RuntimeError(
                f"Unsloth: GGUF conversion was killed by the operating system "
                f"(SIGKILL), which almost always means the machine ran out of "
                f"host RAM. The converter holds tensors in RAM, so this is "
                f"about system memory rather than GPU memory or disk.\n"
                f"Try a smaller quantization, a machine with more RAM, or "
                f"convert from a saved 16bit checkpoint on a larger host.\n"
                f"Error: {_describe_exception(e)}"
            ) from e
        if IS_KAGGLE_ENVIRONMENT and _gguf_failure_looks_like_disk(e, save_directory):
            raise RuntimeError(
                f"Unsloth: GGUF conversion failed in Kaggle environment.\n"
                f"This is likely due to the 20GB disk space limit.\n"
                f"Try saving to /tmp directory or use a smaller model.\n"
                f"Error: {_describe_exception(e)}"
            ) from e
        else:
            raise RuntimeError(f"Unsloth: GGUF conversion failed: {_describe_exception(e)}") from e

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

    return {
        "save_directory": save_directory,
        "gguf_directory": gguf_directory,
        "gguf_files": all_file_locations,
        "modelfile_location": modelfile_location,
        "want_full_precision": want_full_precision,
        "is_vlm": is_vlm_update,
        "fix_bos_token": fix_bos_token,
    }


# Errno 28 / ENOSPC and the wordings the various layers use for it.
_DISK_FULL_PATTERNS = (
    "no space left on device",
    "not enough free space",
    "disk quota exceeded",
    "errno 28",
    "insufficient disk",
    "write failed: no space",
)

# Kaggle allows 20GB. Below this headroom a failed conversion is plausibly about space; above it, blaming disk
# sends the user nowhere useful.
_DISK_HEADROOM_BYTES = 2 * 1024**3

# A SIGKILLed child is 128 + 9 to a shell, so llama-quantize under shell = True surfaces as "returned non-zero
# exit status 137" with no signal named.
_OOM_KILL_PATTERNS = (
    "sigkill",
    "exit status 137",
    "exit code 137",
    "exited with code 137",
    "exited with code -9",
)


def _iter_exception_chain(exc, max_links = 10):
    """The exception plus its explicit causes and implicit contexts.

    Every layer here re-raises as a plain RuntimeError, so `returncode` and the
    original wording only survive on the chained cause.
    """
    seen = set()
    queue = [exc]
    while queue and len(seen) < max_links:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        queue.append(getattr(current, "__cause__", None))
        queue.append(getattr(current, "__context__", None))


def _gguf_child_was_oom_killed(exc):
    """Was the converter killed by the kernel rather than failing on its own?

    llama.cpp's converter loads tensors in host RAM, and a large model exceeds
    what a free Colab or Kaggle VM has. The kernel OOM-killer takes the process
    and subprocess reports only

        Command '[...]' died with <Signals.SIGKILL: 9>

    which says nothing about memory. Gemma3N_(4B)-Audio hits this on both the
    plain and the high-RAM T4, having trained, inferred and merged cleanly.

    SIGKILL alone is the signal: a converter that fails on its own raises and
    exits non-zero, so a kill is either the OOM-killer or someone stopping the
    run by hand, and both are worth naming.

    llama-quantize runs under a shell, which reports the kill as exit status
    137 instead of a signal, and every layer re-raises as a plain RuntimeError,
    so the whole chain is checked rather than just the outermost exception.
    """
    for error in _iter_exception_chain(exc):
        if getattr(error, "returncode", None) in (-9, 137):
            return True
        text = f"{error}".lower()
        if any(pattern in text for pattern in _OOM_KILL_PATTERNS):
            return True
    return False


def _gguf_failure_looks_like_disk(
    exc,
    save_directory = None,
    needed_bytes = None,
    partial_output = None,
):
    """Is this GGUF failure plausibly about running out of disk?

    Two independent signals, either alone sufficient: each can be absent for a
    good reason. The message may name ENOSPC after the directory was cleaned
    up, and the disk may be genuinely full while a subprocess surfaced
    something vaguer. Never raises; an unreadable path just means "not disk".

    `needed_bytes` is what the write that failed was actually going to take.
    Room is a relation between the two, not a constant: a 400MB quant with 1.5GB
    free has all the room it needs, and a caller that knows the size says so
    rather than being measured against a fixed floor that has nothing to do with
    it. The floor remains for callers that cannot say.

    `partial_output` is the file the failed write was filling. llama-quantize
    streams straight into it (`llama-quant.cpp` opens the `ofstream` up front and
    writes each tensor as it finishes one), so a pass that dies partway leaves
    those bytes on disk -- out of the free space measured here, while
    `needed_bytes` still describes the whole output. Crediting them back asks
    "was there room for this output", not "is there room for a second copy of
    it": without it a 10GB export that starts with 12GB free and dies on an
    unsupported tensor after 5GB reads as a full disk and loses the rebuild
    advice that would have addressed the real failure.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(p in text for p in _DISK_FULL_PATTERNS):
        return True
    if getattr(exc, "errno", None) == 28:
        return True
    # The output directory holds the file, so it decides and the working directory is only a fallback: calling
    # a roomy output disk full because an unrelated filesystem is short hides the advice that would have fixed
    # the quantizer failure.
    threshold = needed_bytes if needed_bytes and needed_bytes > 0 else _DISK_HEADROOM_BYTES
    written, written_device = 0, None
    if partial_output:
        try:
            _stat = os.stat(partial_output)
            written, written_device = _stat.st_size, _stat.st_dev
        except OSError:
            # No partial output, or unreadable: nothing to credit back.
            written = 0
    for path in (save_directory, os.getcwd()):
        if not path:
            continue
        try:
            free = shutil.disk_usage(path).free
        except OSError:
            # Never let the diagnostic be the thing that raises.
            continue
        if written:
            # Only on the filesystem that actually holds the partial file: bytes on one device are not room on
            # another.
            try:
                if os.stat(path).st_dev == written_device:
                    free += written
            except OSError:
                pass
        return free < threshold
    return False


# The full-precision output types; everything else llama-quantize writes is quantized and carries its nominal
# width as the leading digit of the name.
_GGUF_BITS_PER_WEIGHT = {"f32": 32.0, "f16": 16.0, "bf16": 16.0}

# k- and i-quants store block scales and mins beside the weights, so a type is wider than its
# name; llama.cpp's 7B sizes keep that under a bit per weight (Q4_K_M near 4.5, Q6_K near 6.6),
# so 1.5 bounds every type.
_QUANT_OVERHEAD_BITS = 1.5

_QUANT_NOMINAL_BITS = re.compile(r"^i?q(\d+)")


def _gguf_type_bits(dtype):
    """Nominal bits a weight of a GGUF type, or None if the name is unknown."""
    name = str(dtype).lower()
    if name in _GGUF_BITS_PER_WEIGHT:
        return _GGUF_BITS_PER_WEIGHT[name]
    nominal = _QUANT_NOMINAL_BITS.match(name)
    return float(nominal.group(1)) if nominal else None


def _gguf_output_size_ratio(
    quant_method,
    first_conversion,
    upper_bound = True,
):
    """One output's size as a multiple of the base GGUF's, rounded either way.

    Both directions cost something. Charging every quantized pass a whole copy of
    the base deletes a merge an export with room to spare would have kept (a
    Q4_K_M off a 60GB base needs about 21GB); charging `f32` one copy under-counts
    by half, since f32 off an f16 base writes four bytes a weight against two, and
    under-counting costs the export outright.

    So price each pass by its own width, and measure the base the same way rather
    than assuming it: q8_0 is a direct-convert outtype, so `first_conversion` is
    not always 16-bit.

    `upper_bound` picks which way that price is rounded, because the two callers
    are hurt by opposite errors and cannot share one number. Reclamation must not
    under-count -- too small an estimate keeps a merge the quants then have no
    room for -- so it adds each k-quant's block overhead and charges an
    unrecognised type a whole copy of the base. Diagnosis must not over-count:
    an inflated estimate reports a full disk for a failure that was nothing of
    the sort and swallows the llama.cpp rebuild advice, so it takes each type's
    nominal width (Q4_K_M really lands near 4.5 bits, never below 4) and returns
    None for a type it cannot measure, which leaves the caller on the fixed floor.
    """
    base = _gguf_type_bits(first_conversion) or 16.0
    target = _gguf_type_bits(quant_method)
    if target is None:
        # Unrecognised: charge a whole copy of the base, as every method used to, or for a diagnosis admit the
        # size is unknown.
        return 1.0 if upper_bound else None
    if upper_bound and str(quant_method).lower() not in _GGUF_BITS_PER_WEIGHT:
        target += _QUANT_OVERHEAD_BITS
    return target / base


# The names a disposable merge is written under, and only those: both writers emit safetensors. Neither stem
# nor extension is decoration, since transformers clears stale shards only when both match, so another stem or
# an older .bin is a file save_pretrained never writes or removes -- and this helper deletes permanently.
_MERGE_WEIGHT_NAME = re.compile(r"^(model|consolidated)(-\d{5}-of-\d{5})?\.safetensors$")
# The index save_pretrained writes for a sharded save. Safetensors only, like the matcher: a
# pytorch_model.bin.index.json belongs to an earlier save transformers leaves in place, and reading it would
# hand its shards to the deletion.
_WEIGHT_INDEX_NAMES = ("model.safetensors.index.json",)
# One shard of a sharded save, under any stem. Applied only to names an index already listed: alone it is far
# too wide, and dropping it from the matcher is what stopped a user's backup-00001-of-00002 being read as the
# merge.
_INDEX_SHARD_NAME = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.safetensors$")


def _merge_weight_files(model_directory, names):
    """The weight files a 16-bit merge writes, out of everything in a directory.

    Deleting by extension alone is what this exists to avoid: the merge lands in
    a directory the caller named, routinely a training `output_dir` already
    holding `training_args.bin`, `optimizer.pt` or `rng_state.pth`, artifacts
    this export did not create and cannot recreate. So match the names
    `save_pretrained` actually produces -- the index names its shards outright
    when it wrote one, and the naming convention answers when it did not.
    """
    indexed, spent_indexes = set(), set()
    for index_name in _WEIGHT_INDEX_NAMES:
        if index_name not in names:
            continue
        try:
            with open(os.path.join(model_directory, index_name), encoding = "utf-8") as index_file:
                weight_map = json.load(index_file).get("weight_map") or {}
            # Basenames: only this directory is listed, so a path never matches.
            listed = {os.path.split(str(shard))[-1] for shard in weight_map.values()}
        except (OSError, ValueError, AttributeError):
            # A missing or malformed index just means the names decide instead.
            continue
        if _is_one_whole_shard_set(listed, names):
            indexed.update(listed)
            # The index goes with the shards it named, and `names` is already filtered to this export's files.
            # Leaving it behind points at deleted files, and the next export's snapshot calls it preexisting,
            # losing the only way a shard set under a missed stem is found again.
            spent_indexes.add(index_name)
    return sorted(
        n for n in names if n in indexed or n in spent_indexes or _MERGE_WEIGHT_NAME.match(n)
    )


def _is_one_whole_shard_set(listed, names):
    """Does this index describe one complete shard set that is all still here?

    The index is the one way a name the convention misses still gets reclaimed,
    which makes it the one way a file the convention *protects* gets deleted. An
    index left behind by an earlier save is the case that matters: transformers
    writes one only when a save shards, and its stale sweep never removes an
    index (`model.safetensors.index` does not match the shard shape it looks
    for), so an unsharded merge lands beside a previous save's index and inherits
    whatever that one names.

    A live index is self-consistent in a way a stale one has no reason to be: it
    lists `-00001-of-000NN` through `-000NN-of-000NN` under a single stem, and
    every shard is on disk because the save just wrote them. Requiring that
    rejects a mixed or partial listing, which is what a stale index beside a
    fresh save looks like, while still reclaiming a sharded merge written under
    a stem `_MERGE_WEIGHT_NAME` does not know.
    """
    if not listed:
        return False
    stems, totals = set(), set()
    for name in listed:
        shard = _INDEX_SHARD_NAME.match(name)
        if shard is None:
            return False
        stems.add(shard.group("stem"))
        totals.add(shard.group("total"))
    if len(stems) != 1 or len(totals) != 1:
        return False
    # "of-000NN" states the count, so a listing missing or adding shards is not the set it claims.
    if len(listed) != int(totals.pop()):
        return False
    return listed <= set(names)


def _free_merge_if_disk_is_tight(
    model_directory,
    gguf_directory,
    initial_files,
    quant_methods = (),
    first_conversion = None,
    merge_is_disposable = False,
    preexisting_weights = None,
):
    """Reclaim the intermediate 16-bit merge when the quants will not fit.

    Returns the bytes freed, 0 if nothing was touched. Never raises: it runs to
    make an export succeed and must not be the thing that fails it.

    `merge_is_disposable` is the whole safety story and defaults to off. It is
    true only when this export wrote `model_directory` itself as a throwaway on
    the way to the GGUF. A non-PEFT `save_pretrained_gguf` instead points the
    converter at the checkpoint the model was loaded from, where deleting weights
    would destroy the user's input model rather than an intermediate.

    Only the weight files go: config.json and the tokenizer are small and later
    steps (the Modelfile, a push) may still want them.
    """
    if not merge_is_disposable:
        return 0
    # merge_is_disposable says the export wrote this directory; this says which files it wrote. A caller that
    # cannot say gets no reclamation rather than a guess, since the guess is permanent.
    if preexisting_weights is None:
        return 0
    if not model_directory or not os.path.isdir(model_directory):
        return 0
    quant_methods = list(quant_methods)
    if not quant_methods:
        return 0
    try:
        # llama-quantize copies companions rather than quantizing them, so they are excluded from the output
        # and memory estimates.
        base_bytes = sum(
            os.path.getsize(f)
            for f in initial_files
            if os.path.isfile(f) and not _is_gguf_companion(f)
        )
    except OSError:
        return 0
    if base_bytes <= 0:
        return 0

    target_directory = gguf_directory or model_directory
    # gguf_directory can point anywhere, and freeing bytes on one filesystem does nothing for a pass writing
    # to another: the merge would be deleted for a destination it cannot help.
    try:
        if os.stat(model_directory).st_dev != os.stat(target_directory).st_dev:
            return 0
    except OSError:
        return 0

    # Every output stays on disk, so the passes add up. Overestimating costs an unnecessary deletion;
    # underestimating costs the export.
    needed = (
        base_bytes * sum(_gguf_output_size_ratio(m, first_conversion) for m in quant_methods)
        + _DISK_HEADROOM_BYTES
    )
    try:
        free = shutil.disk_usage(target_directory).free
    except OSError:
        return 0
    if free >= needed:
        return 0

    weights = []
    try:
        names = os.listdir(model_directory)
    except OSError:
        # Unreadable is no reason to fail an export that no longer needs it.
        return 0
    # Anything already here is the caller's, whatever it is named: that is what stops a reused directory
    # losing a finished sharded save or a consolidated.safetensors. It also drops a stale index from the
    # reading below, so its shards are never inherited.
    names = [name for name in names if name not in preexisting_weights]
    for name in _merge_weight_files(model_directory, names):
        path = os.path.join(model_directory, name)
        if os.path.isfile(path):
            weights.append(path)
    freed = 0
    for path in weights:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
        except OSError:
            continue
    if freed:
        print(
            f"Unsloth: Freed {freed / 1024**3:.1f}GB of intermediate 16-bit "
            f"weights from {model_directory} so the quantization has room "
            f"({free / 1024**3:.1f}GB free, about "
            f"{needed / 1024**3:.1f}GB needed). The GGUF files are already "
            f"written and do not need them."
        )
    return freed


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
    is_main_process: bool = True,
    gguf_shard_size: Optional[str] = None,
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
    if not is_main_process:
        return None

    # save_method="lora" exports the adapter itself as a GGUF LoRA, not a merged model.
    if save_method is not None and str(save_method).lower() == "lora":
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

    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    if use_temp_dir or use_temp_dir is None:
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix = "unsloth_gguf_")
        save_directory = temp_dir
        cleanup_temp = True
    else:
        save_directory = model_name
        cleanup_temp = False

    print(f"Unsloth: Converting model to GGUF format...")

    try:
        result = unsloth_save_pretrained_gguf(
            self = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            quantization_method = quantization_method,
            first_conversion = first_conversion,
            push_to_hub = False,  # Never push from here
            token = token,  # forwarded so imatrix_file=True can read a gated/private upstream
            is_main_process = is_main_process,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            temporary_location = temporary_location,
            maximum_memory_usage = maximum_memory_usage,
            imatrix_file = imatrix_file,
            gguf_shard_size = gguf_shard_size,
        )

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
        raise RuntimeError(f"Failed to convert model to GGUF: {_describe_exception(e)}") from e

    print("Unsloth: Uploading GGUF to Huggingface Hub...")

    try:
        from huggingface_hub import HfApi

        api = HfApi(token = token)

        if "/" not in repo_id:
            username = api.whoami()["name"]
            full_repo_id = f"{username}/{repo_id}"
        else:
            full_repo_id = repo_id

        api.create_repo(
            repo_id = full_repo_id,
            repo_type = "model",
            private = private,
            exist_ok = True,
        )

        for file_location in all_file_locations:
            original_name = os.path.basename(file_location)
            if cleanup_temp and "unsloth_gguf_" in original_name:
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
            original_name = os.path.basename(file)
            if cleanup_temp and "unsloth_gguf_" in original_name:
                quant_suffix = (
                    original_name.split(".", 1)[1] if "." in original_name else original_name
                )
                proper_name = f"{model_name}.{quant_suffix}"
            else:
                proper_name = original_name.replace(os.path.basename(save_directory), model_name)
            readme_content += f"- `{proper_name}`\n"

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
        raise RuntimeError(f"Failed to upload to Hugging Face Hub: {_describe_exception(e)}") from e

    finally:
        if cleanup_temp:
            print("Unsloth: Cleaning up temporary files...")
            for d in [save_directory, f"{save_directory}_gguf"]:
                if os.path.exists(d):
                    try:
                        shutil.rmtree(d)
                    except:
                        pass

    return full_repo_id


def save_lora_to_custom_dir(model, tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok = True)

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


# Upstream Unsloth GGUF repos ship a calibration imatrix under one of these names; the GGUF one is suffixed
# .gguf_file so the Hub does not list it as a model, and renamed locally.
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
        # The upstream imatrix lives in unsloth/<base>-GGUF, so map any org onto unsloth; an already-formed
        # -GGUF id is kept as-is.
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

    # imatrix_file=True auto-resolves from the upstream Unsloth GGUF repo; hf_hub_download is imported here
    # since nothing else needs it.
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

    # Resolve the dequantized base id, since the adapter usually references a 4bit repo.
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
    model_name = _model_basename(base_model_id)

    # Save the adapter: an isolated temp dir for a hub push, else save_directory itself, wrapped so it is always
    # cleaned up.
    # Save the adapter; for a hub push use an isolated temp dir, else save_directory itself.
    if push_to_hub:
        lora_dir = tempfile.mkdtemp(prefix = "unsloth-lora-gguf-")
    else:
        os.makedirs(save_directory, exist_ok = True)
        lora_dir = save_directory

    # Wrap so the isolated temp dir used for hub pushes is always cleaned up, even on failure.
    try:
        save_lora_to_custom_dir(model, tokenizer, lora_dir)

        # Ensure a full llama.cpp checkout, which ships convert_lora_to_gguf.py: a prebuilt install or a
        # reused CWD copy carries binaries but not the script.
        install_llama_cpp(just_clone_repo = True)
        converter = os.path.join(LLAMA_CPP_DEFAULT_DIR, "convert_lora_to_gguf.py")
        if not os.path.exists(converter):
            # A prebuilt llama.cpp install (or a reused CWD copy) carries binaries but not the converter script,
            # so force a dedicated source checkout that ships it.
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
        # Pass --trust-remote-code only when the model really came from custom code (the approved load
        # decision), not merely because its config carries an auto_map entry.
        if _loaded_via_remote_code(model):
            cmd.append("--trust-remote-code")

        # Expose the token to the converter so it can fetch a gated/private base config from the Hub.
        env = os.environ.copy()
        if isinstance(token, str) and token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

        print(f"Unsloth: Converting LoRA adapter at '{lora_dir}' to GGUF -> '{out_gguf}'")
        # Resolve the cache from the live env like the merge, not huggingface_hub's frozen constants: a runtime
        # cache redirect (read-only default, Unsloth) would else miss (#6890).
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


from .models.loader_utils import (
    get_model_name,
    _resolve_hub_repo_cached_file,
    _tokenizer_cache_dir,
    _tokenizer_wants_local_only,
)

# Imported lazily at the two call sites: an older zoo, before its bitsandbytes import became optional, would
# otherwise break `import unsloth` on a host without bnb.
from unsloth_zoo.llama_cpp import (
    install_llama_cpp,
    convert_to_gguf as _convert_to_gguf,
)


def _prewarm_base_model_hub_cache(
    model,
    save_method = "merged_16bit",
    token = None,
):
    """Download the 16-bit base weights into the persistent HF hub cache before the merge.

    merge_and_overwrite_lora fetches missing shards with hf_hub_download(local_dir = ...),
    which never populates the hub cache. When the merge directory is temporary (GGUF
    checkpoint exports delete it after conversion), every export re-downloads the full
    base model (#6890). Pre-warming the cache makes the first export download once and
    later exports copy from the cache. Best-effort: any failure or skip falls back to
    the streaming download. Disable with UNSLOTH_PREWARM_HUB_CACHE=0.
    """
    _false = ("0", "false", "no", "off")
    if os.environ.get("UNSLOTH_PREWARM_HUB_CACHE", "1").strip().lower() in _false:
        return
    if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
        return
    _true = ("1", "true", "yes", "on")
    if (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _true
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _true
    ):
        return
    # Only the 16bit / mxfp4 merges download the base model; merged_4bit and lora do not.
    if save_method not in ("merged_16bit", "mxfp4"):
        return
    if not isinstance(model, PeftModel):
        return

    try:
        # getattr so a model without a config / _name_or_path skips instead of raising.
        name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
        if not name_or_path:
            return
        try:
            model_name = get_model_name(name_or_path, load_in_4bit = False)
        except Exception:
            model_name = name_or_path
        if not model_name or os.path.isdir(model_name):
            return  # local checkpoints are copied, never downloaded

        # The merge may swap a gpt-oss "-BF16" repo for its MXFP4 variant, so skip it.
        if save_method == "mxfp4" and model_name.endswith("-BF16"):
            return

        from unsloth_zoo.saving_utils import determine_base_model_source

        model_name, is_local_path, _, base_is_quantized, quant_type = determine_base_model_source(
            model_name, token
        )
        if not model_name or is_local_path:
            return
        # Mirror the merge: an FP8 base with a 16bit sibling merges onto the sibling, so pre-warm the sibling
        # rather than the FP8 repo (#6890).
        if base_is_quantized and quant_type == "fp8" and save_method == "merged_16bit":
            try:
                from unsloth_zoo.saving_utils import _resolve_fp8_16bit_sibling
                sibling = _resolve_fp8_16bit_sibling(model_name, token)
            except Exception:
                sibling = None
            if sibling:
                model_name, is_local_path, _, base_is_quantized, quant_type = (
                    determine_base_model_source(sibling, token)
                )
                if not model_name or is_local_path:
                    return
        if base_is_quantized and quant_type in ("nf4", "fp4"):
            return  # the 16bit merge refuses these bases; nothing worth caching

        from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

        # Resolve the cache from the live env like the merge, not huggingface_hub's frozen constants, or a
        # runtime cache redirect misses (#6890).
        try:
            from unsloth_zoo.hf_cache import _active_caches
            _hub_cache = _active_caches()[1]
            hub_cache_dir = str(_hub_cache) if _hub_cache is not None else None
        except Exception:
            hub_cache_dir = None

        # Mirror the zoo's shard listing (drop consolidated.safetensors when real shards coexist) so the
        # cached set is a superset of what the merge looks up.
        shard_names = []
        total_size_in_bytes = 0
        for x in HfFileSystem(token = token).ls(model_name, detail = True):
            if x["name"].endswith(".safetensors"):
                shard_names.append((os.path.split(x["name"])[-1], int(x.get("size") or 0)))
        if any(name != "consolidated.safetensors" for name, _ in shard_names):
            shard_names = [x for x in shard_names if x[0] != "consolidated.safetensors"]
        if not shard_names:
            return

        try:
            for filename, _ in shard_names:
                hf_hub_download(
                    repo_id = model_name,
                    filename = filename,
                    cache_dir = hub_cache_dir,
                    local_files_only = True,
                    token = token,
                )
            return  # already fully cached
        except Exception:
            pass

        # Mirror the merge's index filter on the download path: some repos ship shards the index omits, so
        # keep only indexed ones or the disk gate over-counts.
        if len(shard_names) > 1:
            try:
                import json as _json

                _idx = hf_hub_download(
                    repo_id = model_name,
                    filename = "model.safetensors.index.json",
                    cache_dir = hub_cache_dir,
                    token = token,
                )
                with open(_idx, encoding = "utf-8") as _f:
                    _indexed = {
                        os.path.split(v)[-1] for v in _json.load(_f).get("weight_map", {}).values()
                    }
                if _indexed and not {n for n, _ in shard_names}.issubset(_indexed):
                    _kept = [x for x in shard_names if x[0] in _indexed]
                    if _kept:
                        shard_names = _kept
            except Exception:
                pass
        total_size_in_bytes = sum(size for _, size in shard_names)

        # The cache copy is extra disk on top of the merge working copy; there must be room for both.
        from huggingface_hub import constants as _hf_constants

        # abspath so a relative HF_HUB_CACHE walks up to an existing root, not "".
        cache_probe = os.path.abspath(
            os.path.expanduser(str(hub_cache_dir or _hf_constants.HF_HUB_CACHE))
        )
        while cache_probe and not os.path.exists(cache_probe):
            parent = os.path.dirname(cache_probe)
            if parent == cache_probe:
                break
            cache_probe = parent
        free_space = shutil.disk_usage(cache_probe).free if os.path.exists(cache_probe) else 0
        if free_space < 2 * total_size_in_bytes:
            print(
                f"Unsloth: Not enough free disk to keep `{model_name}` in the Hugging Face "
                f"cache (need ~{round(2 * total_size_in_bytes / 1024**3, 1)}GB free, have "
                f"{round(free_space / 1024**3, 1)}GB). Downloading straight to the merge "
                f"directory instead; the next export will re-download it."
            )
            return

        if total_size_in_bytes >= 0.1 * 1024**3:
            size_str = f"{round(total_size_in_bytes / 1024**3, 1)}GB"
        else:
            size_str = f"{max(1, round(total_size_in_bytes / 1024**2))}MB"
        print(
            f"Unsloth: Downloading `{model_name}` into the Hugging Face cache so future "
            f"exports skip the {size_str} download..."
        )
        snapshot_download(
            repo_id = model_name,
            allow_patterns = [name for name, _ in shard_names]
            + ["model.safetensors.index.json", "tokenizer.model"],
            cache_dir = hub_cache_dir,
            token = token,
        )
    except Exception as e:
        print(
            f"Unsloth: Could not pre-cache the base model weights ({e}). "
            f"Falling back to downloading into the merge directory."
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

    new_quantization_methods = []
    if quantization_method is not None:
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
            if quant_method is None:
                quant_method = "q8_0"
            else:
                quant_method = quant_method.lower()
            if quant_method == "not_quantized":
                quant_method = "f16"
            elif quant_method == "fast_quantized":
                quant_method = "q8_0"
            elif quant_method == "quantized":
                quant_method = "q4_k_m"
            new_quantization_methods.append(quant_method.lower())
    else:
        new_quantization_methods.append(quantization_type.lower())
    for quant_method in new_quantization_methods:
        if quant_method not in ALLOWED_QUANTS.keys():
            error = f"Unsloth: Quant method = [{quant_method}] not supported. Choose from below:\n"
            for key, value in ALLOWED_QUANTS.items():
                error += f"[{key}] => {value}\n"
            raise RuntimeError(error)

    for quantization_type in new_quantization_methods:
        metadata = _convert_to_gguf(
            save_directory,
            print_output = True,
            quantization_type = quantization_type,
        )
        if repo_id is not None:
            from unsloth_zoo.saving_utils import prepare_saving

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
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
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

    # Full-finetuned models have no adapters to merge, so fall back to save_pretrained, mirroring the torchao
    # and GGUF save paths.
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
        _prewarm_base_model_hub_cache(model, save_method = save_method, token = token)
        from unsloth_zoo.saving_utils import merge_and_overwrite_lora
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

    # Kaggle's working directory is ~20GB while /tmp on the same kernel is terabytes, so relative paths under
    # /kaggle/working that do not fit move there. Absolute paths, hub pushes and non-Kaggle machines are
    # untouched.
    save_directory = _preflight_merge_disk(
        self,
        save_directory,
        save_method,
        push_to_hub = push_to_hub,
        state_dict = state_dict,
        # unsloth_generic_save writes a supplied dictionary rather than the resident model when there is no
        # adapter, and it alone runs merge_and_overwrite_lora when there is one.
        forwards_state_dict = True,
        # And it is the writer that runs `merge_and_overwrite_lora` when there IS one, which no other entrypoint
        # here does.
        writer_runs_merge_guard = True,
    )

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

    _torchao = _normalize_torchao_method(save_method)
    if _torchao is not None:
        kind, suffix = _torchao
        _unsloth_save_torchao(
            model = self,
            save_directory = save_directory,
            tokenizer = tokenizer,
            kind = kind,
            suffix = suffix,
            push_to_hub = push_to_hub,
            token = token,
            is_main_process = is_main_process,
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
    del arguments["_torchao"]
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

    _torchao = _normalize_torchao_method(save_method)
    if _torchao is not None:
        kind, suffix = _torchao
        _unsloth_save_torchao(
            model = self,
            save_directory = repo_id,
            tokenizer = tokenizer,
            kind = kind,
            suffix = suffix,
            push_to_hub = True,
            token = token,
            is_main_process = True,
            private = private,
            commit_message = commit_message,
            commit_description = commit_description,
            create_pr = create_pr,
            revision = revision,
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
    del arguments["_torchao"]
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
    _convert_torchao_model(model)
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

    # TorchAO does not support safe_serialization reliably.
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

    is_vlm = _is_vlm(model)
    auto_model = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    auto_processor = AutoProcessor if is_vlm else AutoTokenizer

    tokenizer = auto_processor.from_pretrained(save_directory)
    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)

    # TorchAO must only use bfloat16 for loading; float16 fails.
    if HAS_TORCH_DTYPE:
        kwargs = {"torch_dtype": torch.bfloat16}
    else:
        kwargs = {"dtype": torch.bfloat16}

    # Else the original stays resident on every GPU while device_map="auto" below loads a second copy.
    model_restore = _offload_model_for_quantize_subprocess(model)
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    # The original stays offloaded until the quantized copy is saved AND released, else both are resident and
    # the restore OOMs.
    # The original stays offloaded until the quantized copy is saved AND released, else both are resident at
    # once and the restore OOMs.
    try:
        quantized_model = auto_model.from_pretrained(
            save_directory,
            device_map = "auto",
            quantization_config = quantization_config,
            **kwargs,
        )

        torchao_save_directory = save_directory + "-torchao"

        # TorchAO does not support safe_serialization right now; 0.14.0 seems broken.
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

    finally:
        # del here, not at the end of the try: if save_pretrained raises, the copy would still be resident
        # while the original is restored.
        quantized_model = None
        del quantized_model
        # A failed save leaves a live traceback whose frames hold the copy, so dropping the local alone does
        # not free its VRAM.
        _exc = sys.exc_info()[1]
        if _exc is not None:
            traceback.clear_frames(_exc.__traceback__)
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        _restore_model_after_quantize_subprocess(model, model_restore)

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


_DISPATCH_SNAPSHOT_ATTR = "_unsloth_dispatch_snapshot"


def _accelerate_move_guards():
    """The instance methods dispatch_model wraps to block moving an offloaded model."""
    try:
        from accelerate.hooks import _accelerate_added_attributes
        return tuple(_accelerate_added_attributes)
    except Exception:
        return ("to", "cuda", "npu", "xpu", "mlu", "sdaa", "musa")


_ACCELERATE_MOVE_GUARDS = _accelerate_move_guards()


def _accelerate_dispatch_root(model):
    """The module that really owns the accelerate dispatch.

    A PEFT wrapper only proxies ``_hf_hook``, so ``delattr`` fails and
    ``remove_hook_from_submodules`` raises before removing anything; ``hf_device_map``
    keys are relative to the inner root too. Walks real children, never ``__getattr__``.
    """
    node, seen = model, set()
    while id(node) not in seen:
        seen.add(id(node))
        if "hf_device_map" in getattr(node, "__dict__", {}):
            return node
        children = getattr(node, "__dict__", {}).get("_modules") or {}
        nxt = next(
            (
                children[a]
                for a in ("base_model", "model")
                if hasattr(children.get(a), "named_modules")
            ),
            None,
        )
        if nxt is None:
            return model
        node = nxt
    return model


def _snapshot_dispatch_state(root):
    """Hooks, tensor placements and instance forwards, so the dispatch can be replayed.

    Re-deriving it with ``dispatch_model`` is not equivalent: PEFT reparents each
    targeted ``Linear`` after transformers dispatched, so accelerate hooks modules that
    never had any (measured: 395 -> 1379) and the logits shift enough to reorder top-5.
    """
    hooks = [
        (name, mod.__dict__["_hf_hook"])
        for name, mod in root.named_modules()
        if "_hf_hook" in mod.__dict__
    ]
    # remove_duplicate=False: the default hides one half of every tied pair, exactly the half that needs re-
    # tying below.
    named = list(root.named_parameters(remove_duplicate = False)) + list(
        root.named_buffers(remove_duplicate = False)
    )
    places = {name: tensor.device for name, tensor in named}
    # Tied weights share one storage, but the CPU round trip repoints every tensor while tied_params_map is
    # keyed on the old pointer, so replaying hooks alone yields independent copies: double VRAM and divergent
    # updates. Skip meta tensors, since offloaded parameters all have pointer 0 and would collapse into one
    # fake tied group.
    groups = {}
    for name, tensor in named:
        if tensor.device.type == "meta":
            continue
        ptr = tensor.untyped_storage().data_ptr()
        if ptr:
            groups.setdefault(ptr, []).append(name)
    ties = [names for names in groups.values() if len(names) > 1]
    # Removing a hook restores forward = _old_forward, captured before unsloth patched the module, so a
    # remove/re-add permanently drops every fused kernel installed after the dispatch (measured on all 28
    # MLPs) and the to/cuda move guards. Record those too.
    attrs = ("forward", "_old_forward") + tuple(_ACCELERATE_MOVE_GUARDS)
    saved_attrs = {
        name: {a: mod.__dict__[a] for a in attrs if a in mod.__dict__}
        for name, mod in root.named_modules()
        if any(a in mod.__dict__ for a in attrs)
    }
    # Re-adding a hook runs init_hook -> set_module_tensor_to_device, building a fresh Parameter and dropping
    # .grad, so snapshot the gradients and reattach on restore.
    grads = {
        name: getattr(tensor, "grad", None)
        for name, tensor in root.named_parameters(remove_duplicate = False)
        if getattr(tensor, "grad", None) is not None
    }
    return hooks, places, saved_attrs, ties, grads


def _drop_accelerator_tied_param_cache(snapshot) -> None:
    """Drop the GPU tensors accelerate caches in each hook's ``tied_params_map``.

    Holding the hooks across the offload pins a GPU copy of the tied embedding (0.31 GB
    of 1.24 GB here). The entries are keyed on the pre-move ``data_ptr`` so they are
    stale anyway, and re-attaching repopulates them.
    """
    for _name, hook in snapshot[0]:
        cache = getattr(hook, "tied_params_map", None)
        if not cache:
            continue
        for ptr in list(cache):
            entry = cache[ptr]
            for device in list(entry):
                if str(device) != "cpu":
                    del entry[device]
            if not entry:
                del cache[ptr]


def _split_tensor_path(root, full_name):
    """``("model.embed_tokens.weight")`` -> ``(the module, "weight")``."""
    mod_name, _, attr = full_name.rpartition(".")
    try:
        return (root.get_submodule(mod_name) if mod_name else root), attr
    except AttributeError:
        return None, attr


def _lookup_tensor(root, full_name):
    mod, attr = _split_tensor_path(root, full_name)
    if mod is None:
        return None
    for store in ("_parameters", "_buffers"):
        found = (getattr(mod, store, None) or {}).get(attr)
        if found is not None:
            return found
    return None


def _share_tensor(root, full_name, leader) -> None:
    """Point ``full_name`` back at ``leader``, restoring a tie."""
    mod, attr = _split_tensor_path(root, full_name)
    if mod is None:
        return
    for store in ("_parameters", "_buffers"):
        target = getattr(mod, store, None)
        if target is None or attr not in target:
            continue
        current = target[attr]
        if current is None or current.device != leader.device or current.shape != leader.shape:
            return  # not actually the same tensor; leave it alone
        target[attr] = leader
        return


def _restore_dispatch_state(root, snapshot) -> None:
    """Replay ``_snapshot_dispatch_state``."""
    from accelerate.hooks import add_hook_to_module

    hooks, places, saved_attrs, ties, grads = snapshot
    for name, hook in hooks:
        add_hook_to_module(root.get_submodule(name) if name else root, hook)

    # Re-adding a hook rewraps whatever _old_forward now holds, so put the exact callables back, _old_forward
    # first.
    for name, values in saved_attrs.items():
        mod = root.get_submodule(name) if name else root
        for attr in ("_old_forward", "forward", *_ACCELERATE_MOVE_GUARDS):
            if attr in values:
                mod.__dict__[attr] = values[attr]

    # init_hook only re-places tensors the hooked module owns, so anything added after the dispatch (the LoRA
    # adapters) is still on CPU.
    for mod_name, mod in root.named_modules():
        for attr in ("_parameters", "_buffers"):
            store = getattr(mod, attr, None)
            if not store:
                continue
            for tensor_name, tensor in list(store.items()):
                if tensor is None:
                    continue
                full = f"{mod_name}.{tensor_name}" if mod_name else tensor_name
                want = places.get(full)
                if want is None or tensor.device == want:
                    continue
                if getattr(tensor, "quant_state", None) is not None:
                    # Only bitsandbytes' own .to() moves absmax/code/state2 with the data.
                    mod.to(want)
                else:
                    tensor.data = tensor.data.to(want)

    # Reattach the gradients init_hook discarded, on their weight's device.
    for name, grad in grads.items():
        tensor = _lookup_tensor(root, name)
        if tensor is not None and tensor.grad is None and tensor.shape == grad.shape:
            tensor.grad = grad.to(tensor.device)

    # Re-tie last, once every tensor is back on its own device.
    for names in ties:
        leader = _lookup_tensor(root, names[0])
        if leader is None:
            continue
        for follower in names[1:]:
            _share_tensor(root, follower, leader)
    # init_hook refilled tied_params_map with the pre-retie tensors, now unreferenced by the model but still
    # pinned by the map.
    if ties:
        _drop_accelerator_tied_param_cache(snapshot)


def _offload_model_for_quantize_subprocess(model):
    """Best-effort: move the model's weights off the GPU before the quantized export
    loads its own copy from disk, so the GPUs need not hold both at once. Returns an
    opaque token for ``_restore_model_after_quantize_subprocess`` (None if nothing moved).

    Two shapes are handled:
      * single-device CUDA/XPU model -> ``.to("cpu")``, restored with ``.to(device)``;
      * accelerate-dispatched model (a multi-GPU ``device_map`` shard, e.g. the Unsloth
        multi-GPU export load) -> hooks removed and moved to CPU, restored by replaying
        the dispatch. A plain ``.to("cpu")`` is invalid here, which is why the old
        single-device-only move left every GPU holding a full copy. A map spilling to
        CPU is still released, but disk/meta targets are left alone: accelerate keeps
        those parameters off the model, so moving would materialize the whole checkpoint.

    Quantized (bnb) models are attempted too rather than skipped: Unsloth exports load
    4-bit by DEFAULT, so skipping them left a shard on every GPU. transformers refuses
    ``.to()`` for some bitsandbytes builds and that refusal raises before anything moves,
    so the failure path restores the model and returns None, i.e. the old behaviour.
    """
    try:
        _has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        if not ((torch.cuda.is_available() or _has_xpu) and hasattr(model, "parameters")):
            return None
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            targets = {str(v).lower() for v in device_map.values()}
            # A cpu spill is safe to move, being in host RAM; disk/meta is not, since those parameters are off
            # the model and .to("cpu") would materialize the whole checkpoint.
            if not all(t.isdigit() or t.startswith(("cuda", "xpu")) or t == "cpu" for t in targets):
                return None
            if not any(t.isdigit() or t.startswith(("cuda", "xpu")) for t in targets):
                return None  # nothing on an accelerator: no GPU memory to reclaim
            from accelerate.hooks import remove_hook_from_submodules

            # A PEFT wrapper only proxies the hooks; they live on the inner root.
            root = _accelerate_dispatch_root(model)
            try:
                setattr(root, _DISPATCH_SNAPSHOT_ATTR, _snapshot_dispatch_state(root))
            except Exception as snap_exc:
                # A silent `return None` is indistinguishable from "nothing to move", which hides a real bug
                # behind a merely slower export.
                # Restore will fall back to re-deriving from the device_map.
                logger.warning_once(
                    f"Unsloth: could not snapshot the accelerate dispatch "
                    f"({type(snap_exc).__name__}: {snap_exc}); re-dispatching on restore."
                )
            remove_hook_from_submodules(root)
            try:
                model.to("cpu")
            except Exception:
                # The move failed after the hooks came off, so re-dispatch and leave the model usable rather
                # than hookless and half-moved across CPU/GPUs.
                _restore_model_after_quantize_subprocess(model, ("dispatch", dict(device_map)))
                return None
            snapshot = getattr(root, _DISPATCH_SNAPSHOT_ATTR, None)
            if snapshot is not None:
                _drop_accelerator_tied_param_cache(snapshot)
            return ("dispatch", dict(device_map))
        devices = {str(p.device) for p in model.parameters()}
        if len(devices) == 1 and next(iter(devices)).startswith(("cuda", "xpu")):
            device = next(model.parameters()).device
            try:
                model.to("cpu")
            except Exception:
                _restore_model_after_quantize_subprocess(model, ("device", device))
                return None
            return ("device", device)
    except Exception as exc:
        # A silent `return None` is indistinguishable from "nothing to move", hiding a real bug behind a merely
        # slower export.
        logger.warning_once(
            f"Unsloth: could not free the model's accelerator memory before the quantized "
            f"export ({type(exc).__name__}: {exc}); continuing with the model resident."
        )
        return None
    return None


def _restore_model_after_quantize_subprocess(model, restore_token) -> None:
    """Undo ``_offload_model_for_quantize_subprocess``; warns instead of raising."""
    if restore_token is None:
        return
    kind, value = restore_token
    try:
        if kind == "dispatch":
            root = _accelerate_dispatch_root(model)
            snapshot = root.__dict__.pop(_DISPATCH_SNAPSHOT_ATTR, None)
            if snapshot is not None:
                _restore_dispatch_state(root, snapshot)
            else:
                from accelerate import dispatch_model

                # skip_keys matters: without it accelerate moves every forward kwarg to the executing device,
                # wrong for device-invariant cache tensors.
                dispatch_model(
                    root,
                    device_map = value,
                    skip_keys = getattr(root, "_skip_keys_device_placement", None),
                )
        else:
            model.to(value)
    except Exception:
        logger.warning_once(
            "Unsloth: could not restore the model to its original device(s) after the "
            "quantized export; it may remain on CPU."
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

    # Only the main process installs deps, merges, quantizes and uploads, mirroring the non-PEFT save path;
    # other ranks return at once rather than race on dirs or run pip installs.
    if not is_main_process:
        return None

    # Prepare the quantization runtime BEFORE merging, so an unusable config fails fast instead of writing a
    # full 16bit checkpoint first. Under the llm-compressor-main shadow the subprocess validates itself, so
    # the workspace install / ceiling / scheme checks are skipped.
    _shadow_pythonpath = _compressed_quantize_pythonpath()
    if _shadow_pythonpath is None:
        install_llm_compressor()
        # llm-compressor cannot run under a newer transformers than its ceiling: the subprocess dies on a
        # cryptic TORCH_INIT_FUNCTIONS ImportError only AFTER the costly merge.
        _exceeds, _tf_ver = _transformers_exceeds_llm_compressor_ceiling()
        if _exceeds:
            raise RuntimeError(
                f"Unsloth: FP8/FP4 compressed-tensors export is not available for this model. It runs "
                f"under transformers {_tf_ver}, but llm-compressor supports transformers "
                f"<= {_LLM_COMPRESSOR_MAX_TRANSFORMERS}. Export to GGUF or 16-bit instead."
            )
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

    # Pick the local working dir: on a hub push save_directory is a repo id, so merge and quantize inside a
    # temp dir rather than writing ./<repo_id> into the cwd.
    repo_id, work_tmp, calib_tmp, model_restore = None, None, None, None
    if push_to_hub:
        repo_id = os.fspath(save_directory)
        work_tmp = tempfile.mkdtemp(prefix = "unsloth-compressed-")
        local_dir = os.path.join(work_tmp, os.path.basename(repo_id.rstrip("/")) or "model")
    else:
        # Drop trailing separators so the sibling "<dir>-<fmt>" output is not nested inside <dir>.
        local_dir = os.fspath(save_directory)
        local_dir = local_dir.rstrip("/\\") or local_dir

    # Wrap the body so the isolated temp dirs are always cleaned up, even when the merge, quantization,
    # validation, or hub upload raises.
    api = None
    try:
        # Validate Hub access up front, so a bad token or denied repo fails before the expensive merge and
        # quantization; create_repo is idempotent.
        # Validate Hub access up front (a bad token / denied repo should fail before the expensive merge and
        # quantization, matching the normal push path). create_repo is idempotent.
        if push_to_hub:
            from huggingface_hub import HfApi
            api = HfApi(token = token)
            api.create_repo(
                repo_id = repo_id,
                repo_type = "model",
                private = merge_kwargs.get("private", None),
                exist_ok = True,
            )

        # Merge to 16bit at local_dir via unsloth_generic_save, so adapters are merged and full-finetuned
        # models written in 16bit alike. The subprocess reloads this staging checkpoint with default weight
        # filenames, so never write variant-named shards here; the user's variant goes on the final compressed
        # checkpoint.
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

        # Detect a VLM from the in-memory config: a vision_config or a vision-named architecture. A bare
        # *ForConditionalGeneration also matches text seq2seq (T5/BART/Whisper), so it does not count on its
        # own.
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
        # trust_remote_code must reflect the approved load decision, not the config's static auto_map, so a
        # built-in-loadable model carrying auto_map cannot run unvetted code in the subprocess. Model and
        # tokenizer trust stay separate, as on the torchao path.
        model_trust = _loaded_via_remote_code(model)
        tok_trust = _loaded_via_remote_code(tokenizer)

        # Marshal the calibration dataset for the subprocess: None means the ultrachat default, a str/PathLike
        # is a local save_to_disk dir if it exists else a Hub id, a Dataset goes to temp.
        calib_kind, calib_value = "none", ""
        if needs_calibration and calibration_dataset is not None:
            if isinstance(calibration_dataset, (str, os.PathLike)):
                calib_value = os.fspath(calibration_dataset)
                calib_kind = "disk" if os.path.isdir(calib_value) else "hfid"
            elif hasattr(calibration_dataset, "save_to_disk"):
                # Only persist the samples needed, so multi-GB training sets are not fully copied.
                ds_to_save = calibration_dataset
                # A DatasetDict's len() is the split count, not rows, so pick one split first or every split
                # is saved to the temp dir.
                # A DatasetDict's len() is the split count, not rows; pick one split first so the row subsample
                # below applies and we do not save every split to the temp dir.
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

        # Quantize in a separate process: importing Unsloth patches transformers attention, which breaks the
        # forward llm-compressor runs for calibration. Invoke the converter by file path, not `-m`, so the
        # subprocess stays unpatched.
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
        if model_trust:
            cmd.append("--trust-remote-code")
        if tok_trust:
            cmd.append("--trust-remote-code-tokenizer")
        if variant:
            cmd += ["--variant", variant]

        # Free the in-memory model's CUDA memory before the subprocess loads its own copy, so the GPUs need
        # not hold both. Best-effort, restored in finally.
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

        # Clean PYTHONPATH means shadow only: torch still comes from site-packages while transformers 5.x and
        # llm-compressor main come from the shadow, and dropping the inherited PYTHONPATH removes any parent
        # transformers sidecar.
        if _shadow_pythonpath is not None:
            env["PYTHONPATH"] = _shadow_pythonpath

        print(
            f"Unsloth: Quantizing the merged model to {scheme} with llm-compressor "
            f"{'(llm-compressor-main shadow) ' if _shadow_pythonpath is not None else ''}"
            "(in a separate process)..."
        )
        try:
            subprocess.check_call(cmd, env = env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Unsloth: {scheme} quantization failed (llm-compressor subprocess exit "
                f"{e.returncode}). See the output above for details."
            )

        cfg_path = os.path.join(out_dir, "config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding = "utf-8") as f:
                cfg = json.load(f)
        if "quantization_config" not in cfg:
            raise RuntimeError(
                f"Unsloth: {scheme} export failed - no quantization_config written to {cfg_path}"
            )

        # Optional hub upload of the compressed artifact, not the intermediate 16bit one; the repo was created
        # and validated up front.
        # 8) Optional hub upload of the compressed artifact (not the intermediate 16bit one). The repo was
        # already created/validated up front, so just upload here.
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


def _unsloth_save_torchao(
    model,
    save_directory: Union[str, os.PathLike],
    tokenizer,
    kind: str,
    suffix: str,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    **merge_kwargs,
):
    """Export a device-agnostic torchao FP8 / INT8 "portable" checkpoint (no NVIDIA GPU needed).

    Merges LoRA to 16bit in a staging dir, then applies torchao weight-only quantization via
    `TorchAoConfig` into `save_directory + "-" + suffix`. No calibration, subprocess, or CUDA.
    `kind` is "fp8" (safetensors) or "int8" (.bin; torchao only whitelists float8 for safetensors).
    """
    import tempfile

    if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)):
        tokenizer = patch_saving_functions(tokenizer)
    if token is None:
        token = get_token()

    # Only the main process merges, quantizes and uploads; other ranks return at once.
    if not is_main_process:
        return None

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        TorchAoConfig,
    )
    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig

    if kind == "fp8":
        quant_type = Float8WeightOnlyConfig()
        safe_serialization = True
    elif kind == "int8":
        quant_type = Int8WeightOnlyConfig()
        safe_serialization = False  # torchao only supports safetensors for float8 configs
    else:
        raise RuntimeError(f"Unsloth: unknown torchao export kind '{kind}' (expected fp8/int8).")

    # Always merge into an isolated temp staging dir, never save_directory, so a co-selected 16-bit export
    # there is not overwritten. The torchao output is the sibling "<save_directory>-<suffix>", or the repo id
    # on a hub push.
    repo_id, work_tmp, model_restore = None, None, None
    work_tmp = tempfile.mkdtemp(prefix = "unsloth-torchao-")
    if push_to_hub:
        repo_id = os.fspath(save_directory)
        staging = os.path.join(work_tmp, os.path.basename(repo_id.rstrip("/")) or "model")
        out_dir = staging + "-" + suffix
    else:
        base = os.fspath(save_directory).rstrip("/\\") or os.fspath(save_directory)
        staging = os.path.join(work_tmp, os.path.basename(base) or "model")
        out_dir = base + "-" + suffix

    api = None
    try:
        if push_to_hub:
            from huggingface_hub import HfApi
            api = HfApi(token = token)
            api.create_repo(
                repo_id = repo_id,
                repo_type = "model",
                private = merge_kwargs.get("private", None),
                exist_ok = True,
            )

        # Merge to 16bit at a staging dir, LoRA and base alike. The reload reads default weight filenames, so
        # never write variant-named shards here.
        merge_kwargs.pop("variant", None)
        print(f"Unsloth: Merging to 16bit before torchao {kind} quantization...")
        merge_args = dict(merge_kwargs)
        merge_args.update(
            dict(
                model = model,
                tokenizer = tokenizer,
                save_directory = staging,
                save_method = "merged_16bit",
                push_to_hub = False,
                token = token,
                is_main_process = is_main_process,
            )
        )
        unsloth_generic_save(**merge_args)

        # Detect VLM and reload class: a bare *ForConditionalGeneration also matches text seq2seq, so key off
        # vision_config or a vision-named architecture only.
        is_vlm = False
        if hasattr(model, "config"):
            archs = getattr(model.config, "architectures", None) or []
            is_vlm = hasattr(model.config, "vision_config") or any(
                x.endswith("ForVisionText2Text") for x in archs
            )
        # trust_remote_code must reflect the approved load decision, not the staged config's auto_map, which
        # an attacker can set on a built-in-loadable model to run unvetted code past the gate.
        model_trust = _loaded_via_remote_code(model)
        tok_trust = _loaded_via_remote_code(tokenizer)
        # Reload with the class matching the checkpoint: an image-text VLM class (with a fallback for
        # Transformers lacking AutoModelForImageTextToText), the model's own class for encoder-decoder
        # seq2seq, otherwise causal-LM.
        if is_vlm:
            try:
                from transformers import AutoModelForImageTextToText as _reload_model
            except ImportError:
                from transformers import AutoModelForVision2Seq as _reload_model
            auto_model = _reload_model
        elif getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            import transformers as _tf
            auto_model = next(
                (
                    getattr(_tf, _arch)
                    for _arch in (getattr(model.config, "architectures", None) or [])
                    if getattr(_tf, _arch, None) is not None
                ),
                AutoModelForCausalLM,
            )
        else:
            auto_model = AutoModelForCausalLM
        auto_processor = AutoProcessor if is_vlm else AutoTokenizer

        # Free the in-memory model's accelerator memory before reloading from disk, else it sits beside the
        # copy and OOMs a device that fit the model once. Covers CUDA, XPU and multi-GPU dispatched shards,
        # which a plain .to("cpu") cannot move.
        _has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        model_restore = _offload_model_for_quantize_subprocess(model)
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _has_xpu:
                torch.xpu.empty_cache()

        # Reload the staged 16bit checkpoint with torchao applied: bfloat16 is required, and device_map="auto"
        # falls back to CPU, so this works on any hardware.
        print(f"Unsloth: Quantizing the merged model to torchao {kind}...")
        dtype_kw = {"torch_dtype": torch.bfloat16} if HAS_TORCH_DTYPE else {"dtype": torch.bfloat16}
        quantized_model = auto_model.from_pretrained(
            staging,
            device_map = "auto",
            quantization_config = TorchAoConfig(quant_type = quant_type),
            trust_remote_code = model_trust,
            **dtype_kw,
        )
        staged_tokenizer = auto_processor.from_pretrained(staging, trust_remote_code = tok_trust)

        quantized_model.save_pretrained(out_dir, safe_serialization = safe_serialization)
        staged_tokenizer.save_pretrained(out_dir)
        del quantized_model
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg_path = os.path.join(out_dir, "config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding = "utf-8") as f:
                cfg = json.load(f)
        if "quantization_config" not in cfg:
            raise RuntimeError(
                f"Unsloth: torchao {kind} export failed - no quantization_config written to "
                f"{cfg_path}"
            )

        # Optional hub upload of the quantized artifact; the temp staging is cleaned in finally.
        if push_to_hub:
            print(f"Unsloth: Uploading torchao {kind} checkpoint to '{repo_id}' ...")
            api.upload_folder(
                folder_path = out_dir,
                repo_id = repo_id,
                repo_type = "model",
                commit_message = merge_kwargs.get("commit_message", None),
                commit_description = merge_kwargs.get("commit_description", None),
                create_pr = merge_kwargs.get("create_pr", False),
                revision = merge_kwargs.get("revision", None),
            )
            datasets = merge_kwargs.get("datasets", None)
            if datasets:
                try:
                    from huggingface_hub import metadata_update
                    metadata_update(repo_id, {"datasets": datasets}, overwrite = True, token = token)
                except Exception as meta_err:
                    logger.warning_once(
                        f"Unsloth: could not update datasets metadata for {repo_id}: {meta_err}"
                    )

        result = repo_id if push_to_hub else out_dir
        print(
            f"Unsloth: Saved torchao {kind} checkpoint to '{result}'.\n"
            f"Unsloth: This is portable (produced on any device, no NVIDIA GPU required). Load it "
            f"with vLLM or transformers; FP8/INT8 acceleration is available on supported GPUs."
        )
        return result
    finally:
        # A raise pins the copy in the local and the live traceback, so free both or the restore below OOMs.
        quantized_model = None
        del quantized_model
        _exc = sys.exc_info()[1]
        if _exc is not None:
            traceback.clear_frames(_exc.__traceback__)
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        _restore_model_after_quantize_subprocess(model, model_restore)
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
        # PTQ path: the user provided a config, so the model must NOT have a QAT config unless PEFT.
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
        # QAT path: no config provided, so the model must have a QAT config.
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

    if not vision:
        if hasattr(model, "config"):
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
        model.push_to_hub_merged = types.MethodType(unsloth_generic_push_to_hub_merged, model)
        model.save_pretrained_merged = types.MethodType(
            unsloth_generic_save_pretrained_merged, model
        )
        model.push_to_hub_gguf = types.MethodType(unsloth_push_to_hub_gguf, model)
        model.save_pretrained_gguf = types.MethodType(unsloth_save_pretrained_gguf, model)
        model.save_pretrained_torchao = types.MethodType(unsloth_save_pretrained_torchao, model)
    return model
