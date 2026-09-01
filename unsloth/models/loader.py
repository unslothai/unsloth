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

from ._utils import (
    _prepare_model_for_qat,
    is_bfloat16_supported,
    is_vLLM_available,
    HAS_FLASH_ATTENTION,
    HAS_FLASH_ATTENTION_SOFTCAPPING,
    USE_MODELSCOPE,
    get_transformers_model_type,
    hf_login,
    # Single source of truth is _utils.py; re-exported so callers importing it from here keep
    # working and so _is_sdpa_excluded can honor it without a loader -> _utils cycle.
    DISABLE_SDPA_MODEL_NAMES,
)
from ._custom_dtype import register_custom_dtype
from .granite import FastGraniteModel
from .llama import FastLlamaModel, logger, _vllm_will_load_weights
from .mistral import FastMistralModel
from .qwen2 import FastQwen2Model
from .qwen3 import FastQwen3Model
from .qwen3_moe import FastQwen3MoeModel
from .cohere import FastCohereModel
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from .loader_utils import (
    DEFAULT_DEVICE_MAP,
    OFFLOAD_EMBEDDING_AUTO,
    _exclude_rope_inv_freq_from_ddp,
    _get_fp8_mode_and_check_settings,
    _offline_quantize_to_fp8,
    _tag_model_with_fp8_torchao_config,
    get_model_name,
    prepare_device_map,
    requested_device_map,
    _offline_aware_load,
    _resolve_checkpoint_tokenizer_name,
    _is_offline_related_error,
)
import os, contextlib, sys

try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        # For older versions of huggingface_hub.
        from huggingface_hub.utils._token import get_token
import importlib.util
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
    ALLOW_BITSANDBYTES,
)

# huggingface/transformers#26037 allows 4 bit loading.
from unsloth_zoo.utils import Version, _get_dtype
from unsloth_zoo.hf_utils import dtype_from_config
from unsloth_zoo.tiled_mlp import patch_tiled_mlp

transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2 = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32 = transformers_version > Version("4.45.0")
SUPPORTS_GRANITE = transformers_version >= Version("4.46.0")
SUPPORTS_QWEN3 = transformers_version >= Version("4.50.3")
SUPPORTS_QWEN3_MOE = transformers_version >= Version("4.50.3")
SUPPORTS_FALCON_H1 = transformers_version >= Version("4.53.0")
SUPPORTS_GEMMA3N = transformers_version >= Version("4.53.0")
SUPPORTS_GPTOSS = transformers_version >= Version("4.55.0")
SUPPORTS_GEMMA4 = transformers_version >= Version("5.5.0")
# Transformers v5 meta-device loading corrupts non-persistent buffers (inv_freq); see _fix_rope_inv_freq() below.
_NEEDS_ROPE_FIX = transformers_version >= Version("5.0.0")
if SUPPORTS_GEMMA:
    from .gemma import FastGemmaModel
if SUPPORTS_GEMMA2:
    from .gemma2 import FastGemma2Model
if SUPPORTS_FALCON_H1:
    from .falcon_h1 import FastFalconH1Model
import torch
from ._utils import (
    patch_compiling_bitsandbytes,
    patch_model_and_tokenizer,
    prepare_model_for_kbit_training,
    apply_unsloth_gradient_checkpointing,
    patch_compiled_autograd,
    process_vision_info,
    unsloth_compile_transformers,
    fast_inference_setup,
    _requested_float32,
    _mark_requested_float32,
    _mark_forced_float32,
    _mark_full_finetuning,
    _get_text_only_config,
    resolve_model_class,
    _is_family_text_decoder,
    _apply_text_only_key_mapping,
    set_task_config_attr,
    maybe_prefetch_hf_snapshot,
)

# Source of truth is unsloth_zoo.model_lists, re-exported for callers importing FORCE_FLOAT32
# from here. The fallback list is unioned in so a newer unsloth still forces float32 for these
# archs against an older zoo.
_FORCE_FLOAT32_FALLBACK = [
    "gemma3,",
    "gemma3text",  # Gemma3TextModel (EmbeddingGemma, standalone text-only Gemma3)
    "gemma3n",
    "gemma4",  # Gemma4 (gemma4 / gemma4_text): float16 NaNs grad norms in the backward
    "glm4_moe",  # GLM-4.x MoE (glm4_moe / glm4_moe_lite): float16 NaNs grad norms
    "gpt_oss",
    "qwen3_5",  # Qwen3.5 GDN layers produce NaN grad norms in float16 training
    "qwen3_moe",  # Qwen3-MoE (Qwen3-30B-A3B): float16 NaNs grad norms in the backward
]
try:
    from unsloth_zoo import FORCE_FLOAT32 as _ZOO_FORCE_FLOAT32
    FORCE_FLOAT32 = list(_ZOO_FORCE_FLOAT32)
except ImportError:
    FORCE_FLOAT32 = []
for _mt in _FORCE_FLOAT32_FALLBACK:
    if not any(_mt in _entry for _entry in FORCE_FLOAT32):
        FORCE_FLOAT32.append(_mt)

global DISABLE_COMPILE_MODEL_NAMES
# Must be alphabetically sorted for each entry.


def _strip_unsloth_bnb_4bit_suffix(model_name: str) -> str:
    """Remove Unsloth 4bit suffixes without lowercasing (HF cache dirs are case-sensitive)."""
    s = model_name
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if len(s) >= len(suffix) and s.lower().endswith(suffix.lower()):
            s = s[: -len(suffix)]
    return s


def _revision_for_resolved_repo(
    revision,
    model_name,
    old_model_name,
    mapper_moved_name = False,
):
    """Drop `revision` once the requested repo has been remapped to another one.

    A revision names a branch/tag/SHA on the repo the caller asked for, but from_pretrained
    may resolve model_name to a different repo (a pre-quantized mirror, an fp8 temp dir, a
    ModelScope snapshot, a -bnb-4bit strip), where that ref does not exist. Only the mapper
    substitution answers to use_exact_model_name, so only suggest it when it would help.
    """
    if revision is None or model_name == old_model_name:
        return revision
    remedy = (
        " Pass `use_exact_model_name = True` to load your repo as-is." if mapper_moved_name else ""
    )
    logger.warning_once(
        f"Unsloth: Ignoring revision = `{revision}` since `{old_model_name}` resolved to "
        f"`{model_name}`, which does not have that revision.{remedy}"
    )
    return None


def _revision_for_tokenizer_repo(
    tokenizer_name,
    model_name,
    old_model_name,
    revision,
    model_revision,
    is_peft = False,
):
    """Pick the revision for whichever repo the tokenizer is actually read from.

    It is not always the base model's: an adapter-hosted tokenizer is a separate repo with
    its own history, so it keeps the caller's ref even though the base model does not. An
    unset tokenizer_name follows the resolved model_name and so takes whatever the model
    load itself uses (None on a PEFT load, whose ref belongs to the adapter).

    On a plain load the tokenizer belongs to the same model as the weights, so it follows
    model_revision even when the caller named its repo directly: a remap has already dropped
    the pin off the weights, and a pinned tokenizer beside a mirror's default-branch weights
    is the ref mismatch this whole gate exists to avoid.
    """
    repo = tokenizer_name if tokenizer_name else model_name
    if is_peft and repo == old_model_name:
        return revision
    if repo == model_name:
        return model_revision
    return None


def _config_get(
    config,
    field_name,
    default = None,
):
    if isinstance(config, dict):
        return config.get(field_name, default)
    return getattr(config, field_name, default)


def _loaded_skip_modules(model_config):
    """The skip list the load actually used, for the synthetic config stamped after it.

    Whatever ended up on the loaded model is the authority: a pre-quantized checkpoint
    brings its own list and transformers prefers it over any runtime config, while
    on-the-fly quantization gets the one Unsloth built. None (transformers picked the
    output head itself) and [] (told to exclude nothing) are different instructions on
    reload, so neither is normalized away.
    """
    return _config_get(
        getattr(model_config, "quantization_config", None) or {},
        "llm_int8_skip_modules",
    )


def _config_diff(config):
    if isinstance(config, dict):
        return config
    to_diff_dict = getattr(config, "to_diff_dict", None)
    if callable(to_diff_dict):
        try:
            diff = to_diff_dict()
            if isinstance(diff, dict):
                return diff
        except Exception:
            pass
    return {}


def _has_sequence_classification_architecture(config):
    architectures = _config_get(config, "architectures", None) or []
    return any(str(arch).endswith("ForSequenceClassification") for arch in architectures)


def _get_user_task_config_attrs(user_config):
    if user_config is None:
        return {}
    diff = _config_diff(user_config)
    attrs = {}
    for key in ("id2label", "label2id", "problem_type"):
        if key in diff:
            attrs[key] = _config_get(user_config, key, diff.get(key))
    if isinstance(user_config, dict) and "num_labels" in user_config:
        attrs["num_labels"] = user_config["num_labels"]
    elif _has_sequence_classification_architecture(user_config):
        num_labels = _config_get(user_config, "num_labels", None)
        if num_labels is not None:
            attrs["num_labels"] = num_labels
    elif "id2label" in attrs:
        try:
            attrs["num_labels"] = len(attrs["id2label"])
        except TypeError:
            pass
    return attrs


DISABLE_COMPILE_MODEL_NAMES = [
    "aya_vision",
    "modernbert",
    "granite,llava_next",  # Granite-vision 3
]

# Architectures with gated-deltanet (linear attention) layers. Unsloth bundles the
# flash-linear-attention Triton kernels, so no install is needed; transformers falls back to
# the much slower pure PyTorch path only when they cannot be enabled.
FLA_MODEL_TYPE_PREFIXES = ("qwen3_next", "qwen3_5", "kimi_linear", "olmo_hybrid")
_fla_advised = False


def _maybe_advise_fla_install(model_types):
    """One-time note when a gated-deltanet model loads without the fast kernels.

    The kernels ship with Unsloth (no install needed); this fires only when they
    could not be enabled on this platform (e.g. no CUDA, torch < 2.7 or
    triton < 3.3), or when Unsloth deliberately disabled them because they are
    known-broken on this GPU / Triton combination, i.e. exactly when transformers
    uses the slow pure PyTorch path.
    """
    global _fla_advised
    if _fla_advised:
        return
    if model_types is None:
        return
    if isinstance(model_types, str):
        model_types = [model_types]  # a lone string would otherwise iterate chars
    try:
        if not any(
            isinstance(t, str) and t.startswith(FLA_MODEL_TYPE_PREFIXES) for t in model_types
        ):
            return
        from transformers.utils.import_utils import is_flash_linear_attention_available
        if is_flash_linear_attention_available():
            return  # bundled (or user-installed) fast kernels are active
    except Exception:
        return
    _fla_advised = True
    # Prefer unsloth_zoo's reason when it disabled the kernels on purpose: the generic text blames
    # CUDA / torch / triton minimums, all satisfied on an H100 that hit the fla #640 Triton
    # miscompile, so it would point that user in the wrong direction.
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import fla_unavailable_reason
        reason = fla_unavailable_reason()
    except Exception:
        reason = None
    if reason:
        print(reason)
        return
    print(
        "Unsloth: This model uses gated-deltanet linear attention layers. Unsloth\n"
        "bundles the flash-linear-attention kernels, but they could not be enabled\n"
        "on this setup (they need CUDA with torch >= 2.7 and triton >= 3.3), so\n"
        "transformers will use a slower pure PyTorch path."
    )


def _fix_rope_inv_freq(model):
    """Fix inv_freq corruption caused by transformers v5 meta-device loading.

    v5 inits on meta then replaces all non-persistent buffers with uninitialized
    memory. Vanilla restores inv_freq via _init_weights() (needs original_inv_freq),
    but Unsloth rotary classes lack that attr, so inv_freq stays corrupted -> wrong
    positional encodings and 5-11x higher training loss. Here we recompute inv_freq
    from base/dim, apply scaling, and rebuild cos/sin caches. No-op on v4.
    """
    if not _NEEDS_ROPE_FIX:
        return model

    for name, module in model.named_modules():
        # Unsloth's LlamaRotaryEmbedding and subclasses (Extended, LinearScaling, Granite). Native v5
        # rotary classes carry original_inv_freq, which v5's _init_weights() uses to restore inv_freq.
        if (
            hasattr(module, "inv_freq")
            and hasattr(module, "base")
            and hasattr(module, "dim")
            and hasattr(module, "_apply_inv_freq_scaling")
            and hasattr(module, "multi_gpu_cos_cached")
        ):
            if hasattr(module, "_unsloth_recompute_inv_freq"):
                # Restore config scaling (llama3/yarn); unscaled here broke v5.
                inv_freq = module._unsloth_recompute_inv_freq()
            else:
                inv_freq = 1.0 / (
                    module.base
                    ** (
                        torch.arange(0, module.dim, 2, dtype = torch.int64, device = "cpu").float()
                        / module.dim
                    )
                )
                inv_freq = module._apply_inv_freq_scaling(inv_freq)
            module.inv_freq = inv_freq
            for device_idx in range(len(module.multi_gpu_cos_cached)):
                if module.multi_gpu_cos_cached[device_idx] is not None:
                    module._set_cos_sin_cache(
                        seq_len = module.current_rope_size,
                        device = torch.device(device_idx),
                        dtype = torch.get_default_dtype(),
                    )

        # LongRopeRotaryEmbedding (Phi-3.5 style with short_inv_freq + long_inv_freq).
        elif (
            hasattr(module, "short_inv_freq")
            and hasattr(module, "long_inv_freq")
            and hasattr(module, "base")
            and hasattr(module, "dim")
        ):
            config = getattr(model, "config", None)
            rope_scaling = getattr(config, "rope_scaling", None) if config else None
            if rope_scaling is not None:
                short_factor = rope_scaling.get("short_factor", None)
                long_factor = rope_scaling.get("long_factor", None)
                if short_factor is not None and long_factor is not None:
                    inv_freq_shape = (
                        torch.arange(0, module.dim, 2, dtype = torch.int64, device = "cpu").float()
                        / module.dim
                    )
                    sf = torch.tensor(short_factor, device = "cpu", dtype = torch.float32)
                    lf = torch.tensor(long_factor, device = "cpu", dtype = torch.float32)
                    module.short_inv_freq = 1.0 / (sf * module.base**inv_freq_shape)
                    module.long_inv_freq = 1.0 / (lf * module.base**inv_freq_shape)

                    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
                    t = torch.arange(
                        module.original_max_position_embeddings,
                        device = module.short_inv_freq.device,
                        dtype = torch.int64,
                    ).float()
                    freqs = torch.outer(t, module.short_inv_freq)
                    emb = torch.cat((freqs, freqs), dim = -1)
                    for device_idx in range(len(module.multi_gpu_short_cos_cached)):
                        if module.multi_gpu_short_cos_cached[device_idx] is not None:
                            device_obj = torch.device(device_idx)
                            module.multi_gpu_short_cos_cached[device_idx] = (
                                emb.cos() * module.scaling_factor
                            ).to(dtype = dtype, device = device_obj, non_blocking = True)
                            module.multi_gpu_short_sin_cached[device_idx] = (
                                emb.sin() * module.scaling_factor
                            ).to(dtype = dtype, device = device_obj, non_blocking = True)
    return model


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,  # 4bit QLoRA
        load_in_8bit = False,  # 8bit  LoRA
        load_in_16bit = False,  # 16bit LoRA
        full_finetuning = False,
        token = None,
        device_map = DEFAULT_DEVICE_MAP,
        # Planner hints for device_map = "unsloth"; see resolve_unsloth_device_map.
        device_map_planner_kwargs = None,
        rope_scaling = None,
        fix_tokenizer = True,
        trust_remote_code = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab = None,
        revision = None,
        use_exact_model_name = False,
        offload_embedding = OFFLOAD_EMBEDDING_AUTO,
        float32_mixed_precision = None,  # Forces float32 mixed precision
        fast_inference = False,  # uses vLLM
        gpu_memory_utilization = 0.5,
        float8_kv_cache = False,
        random_state = 3407,
        max_lora_rank = 64,
        disable_log_stats = True,
        qat_scheme = None,
        load_in_fp8 = False,  # fp8 LoRA (True, False, 'block')
        unsloth_tiled_mlp = False,
        text_only = False,
        *args,
        **kwargs,
    ):
        # Respect a user-provided quantization_config (e.g. BitsAndBytesConfig).
        quantization_config = kwargs.get("quantization_config", None)
        if quantization_config is not None:
            if isinstance(quantization_config, dict):
                q_load_in_4bit = quantization_config.get("load_in_4bit", False)
                q_load_in_8bit = quantization_config.get("load_in_8bit", False)
            else:
                q_load_in_4bit = getattr(quantization_config, "load_in_4bit", False)
                q_load_in_8bit = getattr(quantization_config, "load_in_8bit", False)
            if q_load_in_4bit:
                load_in_4bit = True
                load_in_8bit = False
            if q_load_in_8bit:
                load_in_8bit = True
                load_in_4bit = False

        # Login to allow private models. Before normalization: dtype is also derived below from a 4bit
        # config's compute dtype, which is not a request for the whole model.
        user_float32 = _requested_float32(dtype)
        token = hf_login(token)
        # Align dtype with bnb_4bit_compute_dtype if provided and dtype is unset.
        if dtype is None and quantization_config is not None:
            bnb_compute_dtype = None
            if isinstance(quantization_config, dict):
                if quantization_config.get("load_in_4bit", False):
                    bnb_compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", None)
            else:
                if getattr(quantization_config, "load_in_4bit", False):
                    bnb_compute_dtype = getattr(quantization_config, "bnb_4bit_compute_dtype", None)
            if isinstance(bnb_compute_dtype, str):
                bnb_compute_dtype = getattr(torch, bnb_compute_dtype, None)
            if isinstance(bnb_compute_dtype, torch.dtype):
                dtype = bnb_compute_dtype

        # Distributed-safe placement for quantized models: under torchrun each rank must load on its
        # own device, else Accelerate raises device relocation errors on quantized weights.
        is_quantized = load_in_4bit or load_in_8bit or load_in_fp8
        device_map = requested_device_map(device_map)
        if is_quantized and isinstance(device_map, str):
            distributed_device_map, is_dist = prepare_device_map()
            if is_dist:
                # One whole model per rank; sharding one across the ranks' GPUs too would have every rank
                # fighting for the same cards.
                device_map = distributed_device_map

        # @_offline_aware_load already forced offline when needed; delegations inherit it.
        if load_in_8bit or full_finetuning or qat_scheme is not None:
            delegated, tokenizer = FastModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                load_in_16bit = load_in_16bit,
                full_finetuning = full_finetuning,
                token = token,
                device_map = device_map,
                device_map_planner_kwargs = device_map_planner_kwargs,
                rope_scaling = rope_scaling,  # [TODO] No effect
                fix_tokenizer = fix_tokenizer,  # [TODO] No effect
                trust_remote_code = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab = resize_model_vocab,  # [TODO] No effect
                revision = revision,
                return_logits = False,
                fullgraph = True,  # No graph breaks
                use_exact_model_name = use_exact_model_name,
                offload_embedding = offload_embedding,
                float32_mixed_precision = float32_mixed_precision,
                # Pass vLLM/inference parameters.
                fast_inference = fast_inference,
                gpu_memory_utilization = gpu_memory_utilization,
                float8_kv_cache = float8_kv_cache,
                random_state = random_state,
                max_lora_rank = max_lora_rank,
                disable_log_stats = disable_log_stats,
                qat_scheme = qat_scheme,
                load_in_fp8 = load_in_fp8,
                unsloth_tiled_mlp = unsloth_tiled_mlp,
                text_only = text_only,
                *args,
                **kwargs,
            )
            return _mark_requested_float32(delegated, user_float32), tokenizer

        if isinstance(dtype, str) and dtype in ["float16", "bfloat16"]:
            dtype = getattr(torch, dtype)
        assert (
            dtype is None
            or dtype == torch.float16
            or dtype == torch.bfloat16
            or dtype == torch.float32
        )

        if fast_inference:
            if importlib.util.find_spec("vllm") is None:
                raise ImportError(
                    "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
                    "You can do this in a terminal via `pip install vllm`"
                )
            if DEVICE_TYPE_TORCH == "cuda":
                for i in range(DEVICE_COUNT):
                    # DGX Spark vLLM breaks.
                    if "NVIDIA GB10" in str(torch.cuda.get_device_name(i)).upper():
                        print(
                            "Unsloth: DGX Spark detected - `fast_inference=True` is currently broken as of January 2026.\n"
                            "Defaulting to native Unsloth inference."
                        )
                        fast_inference = False
                        break

        # bitsandbytes unusable (absent, or unstable on some AMD stacks). A capability check, so not
        # gated on use_exact_model_name, which only suppresses repo-name remapping.
        if not ALLOW_BITSANDBYTES:
            # A user config sets load_in_4bit/8bit above and is forwarded in kwargs, so clearing the flags
            # alone still rebuilds the bnb quantizer downstream. Drop it only when it asks for bnb: a
            # GPTQ / AWQ / fp8 / torchao config must pass through untouched.
            _quant_cfg = kwargs.get("quantization_config", None)
            if isinstance(_quant_cfg, dict):
                _wants_bnb = bool(
                    _quant_cfg.get("load_in_4bit", False) or _quant_cfg.get("load_in_8bit", False)
                )
            elif _quant_cfg is not None:
                _wants_bnb = bool(
                    getattr(_quant_cfg, "load_in_4bit", False)
                    or getattr(_quant_cfg, "load_in_8bit", False)
                )
            else:
                _wants_bnb = False
            if (
                load_in_4bit
                or load_in_8bit
                or _wants_bnb
                or model_name.lower().endswith("-bnb-4bit")
            ):
                print(
                    "Unsloth: `bitsandbytes` is unavailable here - disabling 4bit/8bit. "
                    "16bit LoRA and full finetuning still work."
                )
            # 8bit is bitsandbytes too: leaving either set sends the request to Transformers, which builds
            # the bnb quantizer and fails there.
            load_in_4bit = False
            load_in_8bit = False
            if _wants_bnb:
                kwargs.pop("quantization_config", None)

        # Find FP8, BnB 4bit, other mapped names
        old_model_name = model_name
        fp8_mode = None
        if not use_exact_model_name:
            new_model_name = get_model_name(
                model_name,
                load_in_4bit = load_in_4bit,
                load_in_fp8 = load_in_fp8,
                token = token,
                trust_remote_code = trust_remote_code,
            )
            if new_model_name is None and load_in_fp8 != False:
                fp8_mode = _get_fp8_mode_and_check_settings(
                    load_in_fp8,
                    fast_inference,
                    full_finetuning,
                    load_in_4bit,
                    load_in_8bit,
                    load_in_16bit,
                )
                # Still the caller's repo here, so their ref is the one to quantize from.
                model_name = _offline_quantize_to_fp8(
                    model_name, fp8_mode, text_only = text_only, revision = revision
                )
            else:
                assert new_model_name is not None
                model_name = new_model_name
                # If the mapper resolved to a pre-quantized FP8 model, disable on-the-fly quantization to avoid
                # quantizing twice.
                if load_in_fp8 != False and new_model_name != old_model_name:
                    load_in_fp8 = False
        # Only this block honours use_exact_model_name; the transforms below do not.
        mapper_moved_name = model_name != old_model_name

        # AMD Instinct GPUs need blocksize = 128 on bitsandbytes < 0.49.2, and our pre-quants use blocksize = 64.
        if not ALLOW_PREQUANTIZED_MODELS and model_name.lower().endswith(
            ("-unsloth-bnb-4bit", "-bnb-4bit")
        ):
            model_name = _strip_unsloth_bnb_4bit_suffix(model_name)
        # '-bf16' hub repos load bf16; a local dir keeps the requested quant unless 16bit is set. Say
        # so: dropping the flags silently resurfaces as an OOM whose message never mentions
        # quantization.
        if model_name.lower().endswith("-bf16") and (
            load_in_16bit or not os.path.isdir(os.path.expanduser(model_name))
        ):
            # A user quantization_config stays in **kwargs and still quantizes. load_in_4bit defaults to
            # True, so a set flag is no proof of a request; an explicit load_in_16bit is.
            if (
                not load_in_16bit
                and (load_in_4bit or load_in_8bit or load_in_fp8 != False)
                and kwargs.get("quantization_config", None) is None
            ):
                print(
                    f"Unsloth: `{model_name}` is a 16bit (-bf16) checkpoint, so "
                    f"4bit/8bit/fp8 loading is disabled and the model will "
                    f"load in 16bit, which needs far more VRAM. Unsloth loads "
                    f"4bit by default; point at the 4bit repo instead if you "
                    f"wanted that."
                )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_16bit = True

        if USE_MODELSCOPE and not os.path.exists(model_name):
            from modelscope import snapshot_download
            model_name = snapshot_download(model_name)

        # Gate before the probe below, or a pinned 4bit load fails against the mirror.
        base_revision = _revision_for_resolved_repo(
            revision, model_name, old_model_name, mapper_moved_name
        )
        # The PeftConfig probe reads the adapter repo, which peft loads in-process, so it keeps the ref
        # even when vLLM takes the base model's away just after.
        adapter_revision = base_revision
        # vLLM fetches the default branch, so the pin is dead for the weights. Drop it before the
        # probe: model_types picks the architecture class off that config, so a ref the weights are not
        # at dispatches the wrong one. llama.py's predicate spares the in-process fallbacks.
        if base_revision is not None and _vllm_will_load_weights(
            fast_inference, kwargs.get("num_labels")
        ):
            logger.warning_once(
                f"Unsloth: Ignoring revision = `{base_revision}` since vLLM loads weights "
                "from the default branch. Use `fast_inference = False` to load a pinned revision."
            )
            base_revision = None

        from huggingface_hub.utils import (
            disable_progress_bars,
            enable_progress_bars,
            are_progress_bars_disabled,
        )

        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()

        autoconfig_error = None
        peft_error = None
        autoconfig_exc = None
        peft_exc = None
        model_config = None
        peft_config = None
        local_files_only = kwargs.get("local_files_only", False)

        try:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                revision = base_revision,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
            is_model = True
        except ImportError:
            raise
        except Exception as error:
            autoconfig_error = str(error)
            autoconfig_exc = error
            if "architecture" in autoconfig_error:
                if "qwen3_5" in autoconfig_error:
                    raise ImportError(
                        f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3.5.\n"
                        f"The minimum required version is 5.2.0.\n"
                        f'Try `pip install --upgrade "transformers>=5.2.0"`\n'
                        f"to obtain the latest transformers build, then restart this session."
                    )
                raise ValueError(
                    f"`{model_name}` is not supported yet in `transformers=={transformers_version}`.\n"
                    f"Please update transformers via `pip install --upgrade transformers` and try again."
                )
            is_model = False
        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token = token,
                revision = adapter_revision,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
            is_peft = True
        except ImportError:
            raise
        except Exception as error:
            peft_error = str(error)
            peft_exc = error
            if "architecture" in peft_error:
                raise ValueError(
                    f"`{model_name}` is not supported yet in `transformers=={transformers_version}`.\n"
                    f"Please update transformers via `pip install --upgrade transformers` and try again."
                )
            is_peft = False

        # Old transformers versions check
        both_exist = (is_model and is_peft) and not SUPPORTS_LLAMA32

        # Error out if both LoRA and normal model config exist.
        if both_exist:
            raise RuntimeError(
                "Unsloth: Your repo has a LoRA adapter and a base model.\n"
                "You have 2 files `config.json` and `adapter_config.json`.\n"
                "We must only allow one config file.\n"
                "Please separate the LoRA and base models to 2 repos."
            )
        if not is_model and not is_peft:
            error = autoconfig_error if autoconfig_error is not None else peft_error
            # Old transformers version
            if "rope_scaling" in error.lower() and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support new RoPE scaling methods.\n"
                    f"This includes Llama 3.1. The minimum required version is 4.43.2\n"
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            # Chain an offline-related cause if either probe had one, so @_offline_aware_load still retries from cache.
            _cause = next(
                (
                    e
                    for e in (autoconfig_exc, peft_exc)
                    if e is not None and _is_offline_related_error(e)
                ),
                autoconfig_exc or peft_exc,
            )
            raise RuntimeError(combined_error) from _cause

        model_types = get_transformers_model_type(
            peft_config if peft_config is not None else model_config,
            trust_remote_code = trust_remote_code,
        )
        if len(model_types) == 1:
            model_type = model_types[0]
        else:
            # Leave as tuple if more than one arch.
            model_type = model_types

        # New transformers need to check manually.
        if SUPPORTS_LLAMA32 and is_model and is_peft:
            if os.path.isdir(model_name):
                exist_adapter_config = os.path.exists(
                    os.path.join(model_name, "adapter_config.json")
                )
                exist_config = os.path.exists(os.path.join(model_name, "config.json"))
                both_exist = exist_adapter_config and exist_config
            else:
                # Both AutoConfig and PeftConfig loaded from this remote repo, so config.json and
                # adapter_config.json both exist and no extra HfFileSystem call is needed.
                both_exist = True

        if is_peft:
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(
                    model_name,
                    load_in_4bit = load_in_4bit,
                    load_in_fp8 = load_in_fp8,
                    token = token,
                    trust_remote_code = trust_remote_code,
                )
            # Are pre-quantized models allowed? AMD Instinct GPUs need blocksize = 128 on bitsandbytes
            # < 0.49.2, and our pre-quants use 64.
            if not ALLOW_PREQUANTIZED_MODELS and model_name.lower().endswith(
                ("-unsloth-bnb-4bit", "-bnb-4bit")
            ):
                model_name = _strip_unsloth_bnb_4bit_suffix(model_name)
            # '-bf16' hub repos load bf16; a local dir keeps the requested quant unless 16bit is set. Say
            # so: dropping the flags silently resurfaces as an OOM that never mentions quantization.
            if model_name.lower().endswith("-bf16") and (
                load_in_16bit or not os.path.isdir(os.path.expanduser(model_name))
            ):
                # A user quantization_config stays in **kwargs and still quantizes. load_in_4bit defaults to
                # True, so a set flag is no proof of a request; an explicit load_in_16bit is.
                if (
                    not load_in_16bit
                    and (load_in_4bit or load_in_8bit or load_in_fp8 != False)
                    and kwargs.get("quantization_config", None) is None
                ):
                    print(
                        f"Unsloth: `{model_name}` is a 16bit (-bf16) checkpoint, so "
                        f"4bit/8bit/fp8 loading is disabled and the model will "
                        f"load in 16bit, which needs far more VRAM. Unsloth loads "
                        f"4bit by default; point at the 4bit repo instead if you "
                        f"wanted that."
                    )
                load_in_4bit = False
                load_in_8bit = False
                load_in_fp8 = False
                load_in_16bit = True

            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )

        if not was_disabled:
            enable_progress_bars()

        if model_type == "llama":
            scaling_type = None
            if getattr(model_config, "rope_scaling", None) is not None:
                scaling_type1 = model_config.rope_scaling.get("type", None)
                scaling_type2 = model_config.rope_scaling.get("rope_type", None)
                scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2

            if scaling_type == "llama3" and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Llama 3.1.\n"
                    f"The minimum required version is 4.43.2\n"
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )

            dispatch_model = FastLlamaModel

        elif model_type == "mistral":
            dispatch_model = FastMistralModel
        elif model_type == "gemma":
            if not SUPPORTS_GEMMA:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"
                    f"The minimum required version is 4.38.\n"
                    f'Try `pip install --upgrade "transformers>=4.38"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            dispatch_model = FastGemmaModel
        elif model_type == "gemma2":
            if not SUPPORTS_GEMMA2:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"
                    f"The minimum required version is 4.42.3.\n"
                    f'Try `pip install --upgrade "transformers>=4.42.3"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            # Also check for softcapping support in flash-attn which is faster!
            if is_bfloat16_supported() and not HAS_FLASH_ATTENTION:
                print(
                    "Unsloth: If you want to finetune Gemma 2, install flash-attn to make it faster!\n"
                    "To install flash-attn, do the below:\n"
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
            elif HAS_FLASH_ATTENTION and not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"
                    "To update flash-attn, do the below:\n"
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )

            dispatch_model = FastGemma2Model
        elif model_type == "qwen2":
            dispatch_model = FastQwen2Model
        elif model_type == "qwen3":  # or model_type == "qwen3_moe":
            if not SUPPORTS_QWEN3 or not SUPPORTS_QWEN3_MOE:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3.\n"
                    f"The minimum required version is 4.50.3.\n"
                    f'Try `pip install --upgrade "transformers>=4.50.3"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            dispatch_model = FastQwen3Model if model_type == "qwen3" else FastQwen3MoeModel
        # Optimized Cohere and Granite paths are disabled until their errors match.
        else:
            delegated, tokenizer = FastModel.from_pretrained(
                model_name = old_model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                load_in_16bit = load_in_16bit,
                full_finetuning = full_finetuning,
                token = token,
                device_map = device_map,
                device_map_planner_kwargs = device_map_planner_kwargs,
                rope_scaling = rope_scaling,  # [TODO] No effect
                fix_tokenizer = fix_tokenizer,  # [TODO] No effect
                trust_remote_code = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab = resize_model_vocab,  # [TODO] No effect
                revision = revision,
                return_logits = False,
                fullgraph = True,  # No graph breaks
                use_exact_model_name = use_exact_model_name,
                offload_embedding = offload_embedding,
                float32_mixed_precision = float32_mixed_precision,
                # Pass vLLM/inference parameters.
                fast_inference = fast_inference,
                gpu_memory_utilization = gpu_memory_utilization,
                float8_kv_cache = float8_kv_cache,
                random_state = random_state,
                max_lora_rank = max_lora_rank,
                disable_log_stats = disable_log_stats,
                qat_scheme = qat_scheme,
                load_in_fp8 = load_in_fp8,
                unsloth_tiled_mlp = unsloth_tiled_mlp,
                text_only = text_only,
                *args,
                **kwargs,
            )
            return _mark_requested_float32(delegated, user_float32), tokenizer

        use_gradient_checkpointing = apply_unsloth_gradient_checkpointing(
            use_gradient_checkpointing, max_seq_length, dtype
        )

        # Keep the local checkpoint dir as tokenizer when self-sufficient; see _resolve_checkpoint_tokenizer_name.
        tokenizer_name = _resolve_checkpoint_tokenizer_name(old_model_name, kwargs)

        if fast_inference:
            fast_inference, model_name = fast_inference_setup(model_name, model_config)

        # model_name can move once more here. Skip for PEFT: model_name is then the base model, and
        # `revision` names the adapter PeftModel.from_pretrained loads below.
        if not is_peft:
            base_revision = _revision_for_resolved_repo(
                base_revision, model_name, old_model_name, mapper_moved_name
            )
        # A PEFT load resolved model_name to the base, which the caller's ref is not for.
        model_revision = base_revision if not is_peft else None

        load_in_4bit_kwargs = load_in_4bit
        load_in_8bit_kwargs = load_in_8bit
        if quantization_config is not None and not fast_inference:
            load_in_4bit_kwargs = False
            load_in_8bit_kwargs = False

        # Mirror FastModel: bitsandbytes < 0.46.0 needs dynamo disabled. Best effort, never crash the
        # load (an old zoo without the #710 fix raises NameError here on Python 3.13).
        try:
            patch_compiling_bitsandbytes()
        except Exception as e:
            print(f"Unsloth: Could not patch bitsandbytes for torch.compile - {e}")

        # The optimized path never carried offload_embedding, so a request for one is dropped rather
        # than honoured; say so instead of leaving the caller to infer it from memory use. "auto"
        # stays quiet: it promises a decision, and off is one.
        if offload_embedding != OFFLOAD_EMBEDDING_AUTO and offload_embedding:
            print(
                "Unsloth: Not offloading embeddings; the optimized path for this "
                "architecture does not support it. Pass `device_map` or use FastModel "
                "if you need the offload."
            )

        model, tokenizer = dispatch_model.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = _get_dtype(dtype),
            load_in_4bit = load_in_4bit_kwargs,
            token = token,
            device_map = device_map,
            device_map_planner_kwargs = device_map_planner_kwargs,
            rope_scaling = rope_scaling,
            fix_tokenizer = fix_tokenizer,
            model_patcher = dispatch_model,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            revision = model_revision,
            tokenizer_revision = _revision_for_tokenizer_repo(
                tokenizer_name, model_name, old_model_name, revision, model_revision, is_peft
            ),
            fast_inference = fast_inference,
            gpu_memory_utilization = gpu_memory_utilization,
            float8_kv_cache = float8_kv_cache,
            random_state = random_state,
            max_lora_rank = max_lora_rank,
            disable_log_stats = disable_log_stats,
            load_in_fp8 = load_in_fp8,
            *args,
            **kwargs,
        )

        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)
            # resize_token_embeddings rebuilds the embedding and drops _hf_hook, so this is the last module
            # swap of all, after every repair above.
            try:
                from unsloth.models.vision import _repair_dispatch_hooks
                _repaired = _repair_dispatch_hooks(model)
                if _repaired:
                    logger.info(
                        f"Unsloth: re-attached dispatch hooks to {_repaired} module(s) "
                        "left unhooked by the vocabulary resize."
                    )
            except Exception as _exc:
                logger.warning(
                    f"Unsloth: could not check the dispatch hooks after resizing "
                    f"the vocabulary ({type(_exc).__name__}: {_exc})."
                )

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(
                [
                    "unsloth",
                ]
            )
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(
                [
                    "unsloth",
                ]
            )

        if load_in_4bit:
            # Fix up the bitsandbytes config, but respect a user-provided quantization_config.
            if quantization_config is None:
                # load_in_4bit is the requested flag, not the effective one: a non-bnb checkpoint
                # (MXFP4/gptq/awq) had bnb disabled by check_and_disable, so a synthetic bnb config would
                # corrupt its real one.
                try:
                    from unsloth_zoo.utils import get_quant_type
                    _stamp_bnb = get_quant_type(model.config) in (None, "bitsandbytes")
                except Exception:
                    _stamp_bnb = True
                if _stamp_bnb:
                    compute_dtype = dtype_from_config(model.config)
                    quantization_config = {
                        # compute_dtype is sometimes not a string.
                        "bnb_4bit_compute_dtype": compute_dtype,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "llm_int8_enable_fp32_cpu_offload": False,
                        "llm_int8_has_fp16_weight": False,
                        # Whatever the load really used. None here describes a layout that never existed, and saving it
                        # makes the adapter unreloadable.
                        "llm_int8_skip_modules": _loaded_skip_modules(model.config),
                        "llm_int8_threshold": 6.0,
                        "load_in_4bit": True,
                        "load_in_8bit": False,
                        "quant_method": "bitsandbytes",
                    }
                    model.config.update({"quantization_config": quantization_config})
            else:
                if hasattr(quantization_config, "to_dict"):
                    model.config.update({"quantization_config": quantization_config.to_dict()})
                elif isinstance(quantization_config, dict):
                    model.config.update({"quantization_config": quantization_config})

        if load_in_fp8 != False:
            _tag_model_with_fp8_torchao_config(model, fp8_mode)

        if is_peft:
            # From huggingface/peft#184
            # Warm the adapter repo: PeftModel downloads it in-process and can hang on Xet, and it always
            # loads in-process, so warm it even under fast_inference.
            _prefetched = maybe_prefetch_hf_snapshot(
                old_model_name,
                token = token,
                revision = revision,
                cache_dir = kwargs.get("cache_dir"),
                local_files_only = local_files_only,
                # Adapter always loads in-process via PeftModel, so warm it even under fast_inference.
                fast_inference = False,
                force_download = kwargs.get("force_download", False),
                # Leave use_safetensors auto, since inheriting the base format could skip a safetensors-only
                # adapter; adapter_only restricts the warm to adapter files plus root aux.
                adapter_only = True,
            )
            # The child did the forced download; clear the flag so the load reuses the warm cache.
            if _prefetched and kwargs.get("force_download", False):
                kwargs["force_download"] = False
            # Forward cache_dir so the load reads the warmed adapter. No subfolder: that targets the base
            # checkpoint, and adapters live at the root.
            peft_load_kwargs = {}
            if kwargs.get("cache_dir") is not None:
                peft_load_kwargs["cache_dir"] = kwargs["cache_dir"]
            model = PeftModel.from_pretrained(
                model,
                old_model_name,
                token = token,
                revision = revision,
                local_files_only = local_files_only,
                is_trainable = True,
                trust_remote_code = trust_remote_code,
                **peft_load_kwargs,
            )
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
            try:
                from .vision import _lift_endpoint_hooks_onto_adapters
                _lift_endpoint_hooks_onto_adapters(model)
            except Exception:
                pass  # never block loading on a placement nicety
            # Re-evaluate grouped MoE now the adapter is attached: an expert-LoRA block falls back to the
            # original loop, an attention-only adapter keeps the grouped path.
            try:
                from unsloth_zoo.temporary_patches.moe_grouped_modulelist import (
                    auto_enable_grouped_moe,
                )
                auto_enable_grouped_moe(model)
            except Exception:
                pass  # optional speedup; never block model loading

        # Patch Tiled MLP; enable by setting UNSLOTH_TILED_MLP to "arctic", "target", or "target:{GB}".
        patch_tiled_mlp_choice = os.environ.get(
            "UNSLOTH_TILED_MLP", "arctic" if unsloth_tiled_mlp else "0"
        )
        if patch_tiled_mlp_choice != "0" or unsloth_tiled_mlp:
            patch_tiled_mlp(model, patch_options_str = patch_tiled_mlp_choice)

        model = _fix_rope_inv_freq(model)
        model = _exclude_rope_inv_freq_from_ddp(model)
        # This path never sets UNSLOTH_FORCE_FLOAT32, so answer False rather than let the trainer read
        # whatever a later load writes to the environment.
        model = _mark_forced_float32(model, False)
        # Full finetuning delegates to FastModel above, so this path is always LoRA.
        model = _mark_full_finetuning(model, False)
        return _mark_requested_float32(model, user_float32), tokenizer


from ..kernels import (
    patch_loss_functions,
    post_patch_loss_function,
)
from .vision import FastBaseModel
from .diffusion import FastDiffusionModel, is_diffusion_model_type
from transformers import (
    AutoModelForCausalLM,
)

try:
    from transformers import AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText
except:
    from transformers import AutoModelForVision2Seq


class FastModel(FastBaseModel):
    @staticmethod
    def _prepare_for_qat(model, qat_scheme):
        model = _prepare_model_for_qat(model, qat_scheme)
        return model

    @staticmethod
    def get_peft_model(model, *args, **kwargs):
        # Route text-diffusion models (slow path) to the transformers-only PEFT helper.
        if getattr(model, "_unsloth_slow_diffusion", False):
            return FastDiffusionModel.get_peft_model(model, *args, **kwargs)
        return FastBaseModel.get_peft_model(model, *args, **kwargs)

    @staticmethod
    def for_inference(model):
        if getattr(model, "_unsloth_slow_diffusion", False):
            return FastDiffusionModel.for_inference(model)
        return FastBaseModel.for_inference(model)

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if getattr(model, "_unsloth_slow_diffusion", False):
            return FastDiffusionModel.for_training(model, use_gradient_checkpointing)
        return FastBaseModel.for_training(model, use_gradient_checkpointing)

    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,  # 4bit QLoRA
        load_in_8bit = False,  # 8bit  LoRA
        load_in_16bit = False,  # 16bit LoRA
        full_finetuning = False,
        token = None,
        device_map = DEFAULT_DEVICE_MAP,
        # Planner hints for device_map = "unsloth"; see resolve_unsloth_device_map.
        device_map_planner_kwargs = None,
        rope_scaling = None,  # [TODO] No effect
        fix_tokenizer = True,  # [TODO] No effect
        trust_remote_code = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab = None,  # [TODO] No effect
        revision = None,
        return_logits = False,
        fullgraph = True,  # No graph breaks
        use_exact_model_name = False,
        auto_model = None,
        whisper_language = None,
        whisper_task = None,
        unsloth_force_compile = False,
        offload_embedding = OFFLOAD_EMBEDDING_AUTO,
        float32_mixed_precision = None,  # Forces float32 mixed precision
        # Add the missing vLLM/inference parameters
        fast_inference = False,  # uses vLLM
        gpu_memory_utilization = 0.5,
        float8_kv_cache = False,
        random_state = 3407,
        max_lora_rank = 64,
        disable_log_stats = True,
        qat_scheme = None,
        load_in_fp8 = False,  # fp8 LoRA (True, False, 'block')
        unsloth_tiled_mlp = False,
        target_parameters = None,  # For MoE expert parameters
        text_only = False,
        *args,
        **kwargs,
    ):
        user_config = kwargs.pop("config", None)
        # Respect a user-provided quantization_config (e.g. BitsAndBytesConfig).
        quantization_config = kwargs.get("quantization_config", None)
        if quantization_config is not None:
            if isinstance(quantization_config, dict):
                q_load_in_4bit = quantization_config.get("load_in_4bit", False)
                q_load_in_8bit = quantization_config.get("load_in_8bit", False)
            else:
                q_load_in_4bit = getattr(quantization_config, "load_in_4bit", False)
                q_load_in_8bit = getattr(quantization_config, "load_in_8bit", False)
            if q_load_in_4bit:
                load_in_4bit = True
                load_in_8bit = False
            if q_load_in_8bit:
                load_in_8bit = True
                load_in_4bit = False

        # Login to allow private models. Before normalization: dtype is also derived below from a 4bit
        # config's compute dtype, which is not a request for the whole model.
        user_float32 = _requested_float32(dtype)
        token = hf_login(token)
        if whisper_language is not None:
            assert type(whisper_language) is str
        if whisper_task is not None:
            assert type(whisper_task) is str
        # Align dtype with bnb_4bit_compute_dtype if provided and dtype is unset.
        if dtype is None and quantization_config is not None:
            bnb_compute_dtype = None
            if isinstance(quantization_config, dict):
                if quantization_config.get("load_in_4bit", False):
                    bnb_compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", None)
            else:
                if getattr(quantization_config, "load_in_4bit", False):
                    bnb_compute_dtype = getattr(quantization_config, "bnb_4bit_compute_dtype", None)
            if isinstance(bnb_compute_dtype, str):
                bnb_compute_dtype = getattr(torch, bnb_compute_dtype, None)
            if isinstance(bnb_compute_dtype, torch.dtype):
                dtype = bnb_compute_dtype
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        assert dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert load_in_fp8 in (True, False, "block")

        patch_compiled_autograd()
        # Same best-effort wrapper as the FastLanguageModel path: unsloth_zoo's patch imports
        # bitsandbytes unconditionally, so on a host without it this raised before the capability
        # fallback could take the 16bit path.
        try:
            patch_compiling_bitsandbytes()
        except Exception as e:
            print(f"Unsloth: Could not patch bitsandbytes for torch.compile - {e}")

        if full_finetuning and (load_in_4bit or load_in_8bit):
            print(
                "Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA."
            )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_16bit = False

        # bitsandbytes unusable (absent, or unstable on some AMD stacks). A capability check, so not
        # gated on use_exact_model_name, which only suppresses repo-name remapping.
        if not ALLOW_BITSANDBYTES:
            # A user config sets load_in_4bit/8bit above and is forwarded in kwargs, so clearing the flags
            # alone still rebuilds the bnb quantizer downstream. Drop it only when it asks for bnb.
            _quant_cfg = kwargs.get("quantization_config", None)
            if isinstance(_quant_cfg, dict):
                _wants_bnb = bool(
                    _quant_cfg.get("load_in_4bit", False) or _quant_cfg.get("load_in_8bit", False)
                )
            elif _quant_cfg is not None:
                _wants_bnb = bool(
                    getattr(_quant_cfg, "load_in_4bit", False)
                    or getattr(_quant_cfg, "load_in_8bit", False)
                )
            else:
                _wants_bnb = False
            if (
                load_in_4bit
                or load_in_8bit
                or _wants_bnb
                or model_name.lower().endswith("-bnb-4bit")
            ):
                print(
                    "Unsloth: `bitsandbytes` is unavailable here - disabling 4bit/8bit. "
                    "16bit LoRA and full finetuning still work."
                )
            # 8bit is bitsandbytes too: leaving either set sends the request to Transformers, which builds
            # the bnb quantizer and fails there.
            load_in_4bit = False
            load_in_8bit = False
            if _wants_bnb:
                kwargs.pop("quantization_config", None)

        if (
            int(load_in_4bit) + int(load_in_8bit) + int(load_in_16bit) + int(load_in_fp8 != False)
            >= 2
        ):
            raise RuntimeError(
                "Unsloth: Can only load in 4bit or 8bit or 16bit, not a combination!\n"
                "Also, we by default set `load_in_4bit = True`.\n"
                "If you want 8bit finetuning, set both `load_in_4bit = False` and `load_in_8bit = True`\n"
                "If you want 16bit LoRA finetuning, set `load_in_16bit = True`"
            )

        if qat_scheme is not None and not full_finetuning:
            raise ValueError(
                "Specifying `qat_scheme` in `FastLanguageModel.from_pretrained(...)` is only "
                "compatible with `full_finetuning=True`. If you wish to use QAT with LoRA, "
                "please pass in `qat_scheme` in `FastLanguageModel.get_peft_model(...)` instead."
            )
        if qat_scheme == "phone-deployment":
            qat_scheme = "int8-int4"

        # Distributed-safe placement for quantized models: under torchrun each rank must load on its
        # own device, else Accelerate raises device relocation errors.
        is_quantized = load_in_4bit or load_in_8bit or load_in_fp8
        device_map = requested_device_map(device_map)
        if is_quantized and isinstance(device_map, str):
            distributed_device_map, is_dist = prepare_device_map()
            if is_dist:
                # One whole model per rank; sharding one across the ranks' GPUs too would have every rank
                # fighting for the same cards.
                device_map = distributed_device_map

        if fast_inference:
            if importlib.util.find_spec("vllm") is None:
                raise ImportError(
                    "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
                    "You can do this in a terminal via `pip install vllm`"
                )
            if DEVICE_TYPE_TORCH == "cuda":
                for i in range(DEVICE_COUNT):
                    # DGX Spark vLLM breaks.
                    if "NVIDIA GB10" in str(torch.cuda.get_device_name(i)).upper():
                        print(
                            "Unsloth: DGX Spark detected - `fast_inference=True` is currently broken as of January 2026.\n"
                            "Defaulting to native Unsloth inference."
                        )
                        fast_inference = False
                        break

        # Find FP8, BnB 4bit, other mapped names
        old_model_name = model_name
        fp8_mode = None
        if not use_exact_model_name:
            new_model_name = get_model_name(
                model_name, load_in_4bit = load_in_4bit, load_in_fp8 = load_in_fp8
            )
            if new_model_name is None and load_in_fp8 != False:
                fp8_mode = _get_fp8_mode_and_check_settings(
                    load_in_fp8,
                    fast_inference,
                    full_finetuning,
                    load_in_4bit,
                    load_in_8bit,
                    load_in_16bit,
                )
                # Still the caller's repo here, so their ref is the one to quantize from.
                model_name = _offline_quantize_to_fp8(
                    model_name, fp8_mode, text_only = text_only, revision = revision
                )
            else:
                assert new_model_name is not None
                model_name = new_model_name
                # If the mapper resolved to a pre-quantized FP8 model, disable on-the-fly quantization to avoid
                # quantizing twice.
                if load_in_fp8 != False and new_model_name != old_model_name:
                    load_in_fp8 = False
        # Only this block honours use_exact_model_name; the transforms below do not.
        mapper_moved_name = model_name != old_model_name

        # Are pre-quantized models allowed? AMD Instinct GPUs need blocksize = 128 on bitsandbytes
        # < 0.49.2, and our pre-quants use 64.
        if not ALLOW_PREQUANTIZED_MODELS and model_name.lower().endswith(
            ("-unsloth-bnb-4bit", "-bnb-4bit")
        ):
            model_name = _strip_unsloth_bnb_4bit_suffix(model_name)
        # '-bf16' hub repos load bf16; a local dir keeps the requested quant unless 16bit is set. Say
        # so: dropping the flags silently resurfaces as an OOM that never mentions quantization.
        if model_name.lower().endswith("-bf16") and (
            load_in_16bit or not os.path.isdir(os.path.expanduser(model_name))
        ):
            # A user quantization_config stays in **kwargs and still quantizes. load_in_4bit defaults to
            # True, so a set flag is no proof of a request; an explicit load_in_16bit is.
            if (
                not load_in_16bit
                and (load_in_4bit or load_in_8bit or load_in_fp8 != False)
                and kwargs.get("quantization_config", None) is None
            ):
                print(
                    f"Unsloth: `{model_name}` is a 16bit (-bf16) checkpoint, so "
                    f"4bit/8bit/fp8 loading is disabled and the model will "
                    f"load in 16bit, which needs far more VRAM. Unsloth loads "
                    f"4bit by default; point at the 4bit repo instead if you "
                    f"wanted that."
                )
            load_in_4bit = False
            load_in_8bit = False
            load_in_fp8 = False
            load_in_16bit = True

        if USE_MODELSCOPE and not os.path.exists(model_name):
            from modelscope import snapshot_download
            model_name = snapshot_download(model_name)

        # Gate before the probe below, or a pinned 4bit load fails against the mirror.
        base_revision = _revision_for_resolved_repo(
            revision, model_name, old_model_name, mapper_moved_name
        )
        # The PeftConfig probe reads the adapter repo, which peft loads in-process, so it keeps the ref
        # even when vLLM takes the base model's away just after.
        adapter_revision = base_revision
        # vLLM fetches the default branch, so the pin is dead for the weights. Drop it here, not at
        # dispatch: model_types, auto_model and the text-only decision all come off the config probed
        # below, so a ref the weights are not at dispatches the wrong model.
        if base_revision is not None and fast_inference and is_vLLM_available():
            logger.warning_once(
                f"Unsloth: Ignoring revision = `{base_revision}` since vLLM loads weights "
                "from the default branch. Use `fast_inference = False` to load a pinned revision."
            )
            base_revision = None

        from huggingface_hub.utils import (
            disable_progress_bars,
            enable_progress_bars,
            are_progress_bars_disabled,
        )

        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()

        autoconfig_error = None
        peft_error = None
        autoconfig_exc = None
        peft_exc = None
        model_config = None
        peft_config = None
        # @_offline_aware_load already forced offline when needed; nested calls inherit it.
        local_files_only = kwargs.get("local_files_only", False)

        # Text-diffusion slow-path dispatch, factored so the normal route and the legacy-config
        # fallback share one call site.
        def _dispatch_diffusion():
            model, tokenizer = FastDiffusionModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                load_in_16bit = load_in_16bit,
                full_finetuning = full_finetuning,
                token = token,
                device_map = device_map,
                device_map_planner_kwargs = device_map_planner_kwargs,
                trust_remote_code = trust_remote_code,
                revision = base_revision,
                **kwargs,
            )
            # Returns before the FORCE_FLOAT32 scan and no diffusion type is on that list, so False.
            # Stamped, not left unset, or the trainer reads whatever an earlier load wrote.
            model = _mark_forced_float32(model, False)
            model = _mark_full_finetuning(model, full_finetuning)
            return _mark_requested_float32(model, user_float32), tokenizer

        try:
            model_config = user_config
            if model_config is None:
                model_config = AutoConfig.from_pretrained(
                    model_name,
                    token = token,
                    revision = base_revision,
                    trust_remote_code = trust_remote_code,
                    local_files_only = local_files_only,
                )
            is_model = True
        except ImportError:
            raise
        except Exception as error:
            autoconfig_error = str(error)
            autoconfig_exc = error
            # Legacy text-diffusion configs use model_type "diffusion_gemma", which current transformers
            # does not register (it ships "diffusion_gemma4"), so AutoConfig raises before dispatch. Route
            # straight to the diffusion slow path, whose loader aliases the legacy type.
            if "diffusion_gemma" in autoconfig_error and is_diffusion_model_type("diffusion_gemma"):
                return _dispatch_diffusion()
            if "architecture" in autoconfig_error:
                if "qwen3_5" in autoconfig_error:
                    raise ImportError(
                        f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3.5.\n"
                        f"The minimum required version is 5.2.0.\n"
                        f'Try `pip install --upgrade "transformers>=5.2.0"`\n'
                        f"to obtain the latest transformers build, then restart this session."
                    )
                raise ValueError(
                    f"`{model_name}` is not supported yet in `transformers=={transformers_version}`.\n"
                    f"Please update transformers via `pip install --upgrade transformers` and try again."
                )
            is_model = False
        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token = token,
                revision = adapter_revision,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
            is_peft = True
        except ImportError:
            raise
        except Exception as error:
            peft_error = str(error)
            peft_exc = error
            if "architecture" in peft_error:
                raise ValueError(
                    f"`{model_name}` is not supported yet in `transformers=={transformers_version}`.\n"
                    f"Please update transformers via `pip install --upgrade transformers` and try again."
                )
            is_peft = False
        # Old transformers versions check
        both_exist = (is_model and is_peft) and not SUPPORTS_LLAMA32
        # Error out if both LoRA and normal model config exist.
        if both_exist:
            raise RuntimeError(
                "Unsloth: Your repo has a LoRA adapter and a base model.\n"
                "You have 2 files `config.json` and `adapter_config.json`.\n"
                "We must only allow one config file.\n"
                "Please separate the LoRA and base models to 2 repos."
            )
        if not is_model and not is_peft:
            error = autoconfig_error if autoconfig_error is not None else peft_error
            # Old transformers version
            if "rope_scaling" in error.lower() and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support new RoPE scaling methods.\n"
                    f"This includes Llama 3.1. The minimum required version is 4.43.2\n"
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'
                    f"to obtain the latest transformers build, then restart this session."
                )
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            # Chain an offline-related cause if either probe had one, so @_offline_aware_load still retries from cache.
            _cause = next(
                (
                    e
                    for e in (autoconfig_exc, peft_exc)
                    if e is not None and _is_offline_related_error(e)
                ),
                autoconfig_exc or peft_exc,
            )
            raise RuntimeError(combined_error) from _cause

        model_types = get_transformers_model_type(
            peft_config if peft_config is not None else model_config,
            trust_remote_code = trust_remote_code,
        )
        model_types_all = ",".join(model_types) + ","
        _maybe_advise_fla_install(model_types)

        # Text-diffusion models (DiffusionGemma) take a transformers-only slow path: a custom
        # block-diffusion generate over a novel backbone, so Unsloth's autoregressive kernel/compile
        # patching is skipped and the unmodified HF model is loaded, keeping 4bit/8bit and PEFT LoRA.
        if is_diffusion_model_type(model_types):
            return _dispatch_diffusion()

        lowered_model_name = model_name.lower()
        # Build UNSLOTH_MODEL_NAME fresh from THIS load's types and flags: prepending the inherited
        # value would let a stale "_load_in_4bit_" push gpt-oss onto the BnB router patch on a later
        # 16bit load. The raw model name is excluded so a path containing a sentinel is not misread.
        # Encode the EFFECTIVE bnb state: a non-bnb checkpoint has load_in_4bit disabled later by
        # check_and_disable, so the requested flag would route a native MXFP4 gpt-oss onto the BnB
        # router patch. Early best-effort; sync_unsloth_model_name_bnb_flags is authoritative.
        try:
            from unsloth_zoo.utils import get_quant_type
            _bnb_compatible_quant = get_quant_type(model_config) in (None, "bitsandbytes")
        except Exception:
            _bnb_compatible_quant = True
        string = model_types_all
        if load_in_4bit and _bnb_compatible_quant:
            string += "_load_in_4bit_"
        if load_in_8bit and _bnb_compatible_quant:
            string += "_load_in_8bit_"
        if load_in_16bit:
            string += "_load_in_16bit_"
        if load_in_fp8:
            string += "load_in_fp8"
        os.environ["UNSLOTH_MODEL_NAME"] = string

        LATEST = "\nPlease use transformers via `pip install --no-deps git+https://github.com/huggingface/transformers.git`"
        NIGHTLY = (
            '\nPlease use nightly transformers via pip install --upgrade "transformers>=4.49.0"`'
        )
        if "pixtral" in model_types_all and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Pixtral only works on transformers >= 4.49.0." + LATEST)
        elif "qwen2_5" in model_types_all and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Qwen 2.5 only works on transformers >= 4.49.0." + LATEST)
        # Gemma 4 must come before Gemma 3N, and Gemma 3N before Gemma 3.
        elif "gemma4" in model_types_all:
            if not SUPPORTS_GEMMA4:
                raise RuntimeError("Unsloth: Gemma 4 requires transformers >= 5.5.0" + LATEST)
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
            os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
        # Gemma 3N must be before Gemma 3
        elif "gemma3n" in model_types_all:
            if transformers_version < Version("4.53.0"):
                raise RuntimeError(
                    "Unsloth: Gemma 3N only works on transformers >= 4.53.0" + LATEST
                )
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
            register_custom_dtype(
                "float16;torch.float16;torch.float16;"
                "if name.endswith('norm'): "
                "module._pre_set_compute_dtype = torch.float32\n"
                ";"
                "from unsloth_zoo.temporary_patches.gemma3n import patch_Gemma3nConv_Embed_forwards; patch_Gemma3nConv_Embed_forwards()"
            )
            # Set norms to float32 since they get upcasted anyway; common to gemma-3 and gemma-3n.
            # Granite-4 rms norms are stored as 16 bit, but we upcast
            os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
        elif "gemma3" in model_types_all:
            if transformers_version < Version("4.50.0.dev0"):
                raise RuntimeError(
                    "Unsloth: Gemma 3 only works on transformers >= 4.50.0." + NIGHTLY
                )
            os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
            # ROCm/HIP: the Gemma3 compiled forward gives NaN on RDNA GPUs (gfx1100-gfx1151), so disable
            # torch.compile for the model forward; loss compilation is fine (#3385).
            from unsloth.kernels.utils import is_rdna

            if is_rdna():
                os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"
        elif "cohere2" in model_types_all and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError(
                "Unsloth: Cohere's Command model only works on transformers >= 4.50.0." + NIGHTLY
            )
        elif "csm" in model_types_all:
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"  # Inference is too slow
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"  # Sesame fails
            # Set down projection compute dtype to be float32 for float16 machines Set norms to float32 since
            # anyways they get upcasted to float32
            register_custom_dtype(
                "all;torch.float32;torch.float16;"
                "if name.endswith(('_proj', 'fc1', 'fc2', 'codebook', 'head')): module.to(torch.float16)"
                ";"
            )
        elif "granitemoehybrid" in model_types_all:
            # Granite-4 rms norms are stored as 16 bit, but we upcast.
            # Set norms to float32 since anyways they get upcasted to float32
            os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
        elif "olmo2" in model_types_all and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: OLMo-2 only works on transformers >= 4.50.0." + NIGHTLY)
        elif "olmo3" in model_types_all and transformers_version < Version("4.57.0.dev0"):
            raise RuntimeError("Unsloth: OLMo-3 only works on transformers >= 4.57.0." + LATEST)
        elif "falcon_h1" in model_types_all:
            # Falcon must use float32 Triton (TRITON_F32_DEFAULT = 'ieee') since Mamba kernels error on lower precision.
            register_custom_dtype(
                "float16;torch.float32;torch.float16;"
                "if name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'head')): module.to(torch.float16)"
                ";"
                "os.environ['TRITON_F32_DEFAULT'] = 'ieee'"
            )
        elif "nemotron_h" in model_types_all:
            # NemotronH (hybrid Mamba-2 + Transformer) uses the same Mamba kernels as Falcon-H1, which need
            # float32 Triton precision.
            register_custom_dtype(
                "float16;torch.float32;torch.float16;"
                "if name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'head')): module.to(torch.float16)"
                ";"
                "os.environ['TRITON_F32_DEFAULT'] = 'ieee'"
            )
        elif "gpt_oss" in model_types_all:
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
            # Use the EFFECTIVE bnb state, not the raw flag: a native MXFP4 checkpoint loaded with the
            # default load_in_4bit=True has bnb disabled later, so the raw flag picks the wrong dtype path.
            if not (load_in_4bit and _bnb_compatible_quant):
                # Only upcast MoE biases for MXFP4, not BnB.
                register_custom_dtype(
                    "all;None;None;"
                    "x = 'gate_up_proj_bias'\n"
                    "if hasattr(module, x): "
                    "setattr(module, x, torch.nn.Parameter(getattr(module, x).to(torch.float32)) if isinstance(getattr(module, x), torch.nn.Parameter) else getattr(module, x).to(torch.float32))\n"
                    ""
                    "x = 'down_proj_bias'\n"
                    "if hasattr(module, x): "
                    "setattr(module, x, torch.nn.Parameter(getattr(module, x).to(torch.float32)) if isinstance(getattr(module, x), torch.nn.Parameter) else getattr(module, x).to(torch.float32))\n"
                    ""
                    ";"
                )
            else:
                register_custom_dtype(
                    "torch.float16;torch.bfloat16;torch.float16;"
                    "if ('down_projs' in name) and hasattr(module, 'weight') and "
                    "torch.amax(dequantize_module_weight(module)) >= 0:"
                    "module._pre_set_compute_dtype = torch.float32\n"
                    ""
                    "if ('mlp.router' in name) and hasattr(module, 'weight'):"
                    "module._pre_set_compute_dtype = torch.float32\n"
                    ";"
                )
            os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"] = "1"
        else:
            for check_model_name in DISABLE_COMPILE_MODEL_NAMES:
                if check_model_name in lowered_model_name:
                    os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"
                    os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
                    if transformers_version < Version("4.50.0.dev0"):
                        raise RuntimeError(
                            f"Unsloth: {check_model_name} only works on transformers >= 4.50.0."
                            + NIGHTLY
                        )
                    break

        if auto_model is not None:
            # All other models need to disable static cache.
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"

        # New transformers need to check manually.
        if SUPPORTS_LLAMA32 and is_model and is_peft:
            if os.path.isdir(model_name):
                exist_adapter_config = os.path.exists(
                    os.path.join(model_name, "adapter_config.json")
                )
                exist_config = os.path.exists(os.path.join(model_name, "config.json"))
                both_exist = exist_adapter_config and exist_config
            else:
                # Both AutoConfig and PeftConfig loaded from this remote repo, so config.json and
                # adapter_config.json both exist and no extra HfFileSystem call is needed.
                both_exist = True

        if is_peft:
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(model_name, load_in_4bit)
            # Are pre-quantized models allowed? AMD Instinct GPUs need blocksize = 128 on bitsandbytes
            # < 0.49.2, and our pre-quants use 64.
            if not ALLOW_PREQUANTIZED_MODELS and model_name.lower().endswith(
                ("-unsloth-bnb-4bit", "-bnb-4bit")
            ):
                model_name = _strip_unsloth_bnb_4bit_suffix(model_name)
            # '-bf16' hub repos load bf16; a local dir keeps the requested quant unless 16bit is set. Say
            # so: dropping the flags silently resurfaces as an OOM that never mentions quantization.
            if model_name.lower().endswith("-bf16") and (
                load_in_16bit or not os.path.isdir(os.path.expanduser(model_name))
            ):
                # A user quantization_config stays in **kwargs and still quantizes. load_in_4bit defaults to
                # True, so a set flag is no proof of a request; an explicit load_in_16bit is.
                if (
                    not load_in_16bit
                    and (load_in_4bit or load_in_8bit or load_in_fp8 != False)
                    and kwargs.get("quantization_config", None) is None
                ):
                    print(
                        f"Unsloth: `{model_name}` is a 16bit (-bf16) checkpoint, so "
                        f"4bit/8bit/fp8 loading is disabled and the model will "
                        f"load in 16bit, which needs far more VRAM. Unsloth loads "
                        f"4bit by default; point at the 4bit repo instead if you "
                        f"wanted that."
                    )
                load_in_4bit = False
                load_in_8bit = False
                load_in_fp8 = False
                load_in_16bit = True

            if user_config is not None:
                model_config = user_config
            else:
                model_config = AutoConfig.from_pretrained(
                    model_name,
                    token = token,
                    trust_remote_code = trust_remote_code,
                    local_files_only = local_files_only,
                )

        if not was_disabled:
            enable_progress_bars()

        do_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
        if do_logging:
            redirector = contextlib.nullcontext()
        else:
            redirector = contextlib.redirect_stdout(open(os.devnull, "w", encoding = "utf-8"))

        model_types = ["siglip"] + model_types
        # Set forced float32 env flag
        os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"
        do_forced_float32 = False
        for model_type_arch in model_types:
            if model_type_arch != "siglip":
                break
        for disable_name in FORCE_FLOAT32:
            # Add a comma to model_types_all so an exact match at the end still matches.
            if (
                disable_name.lower() == model_type_arch.lower().replace("-", "").replace("_", "")
                or disable_name.lower() in model_types_all
            ) and ((dtype == torch.float16) or not SUPPORTS_BFLOAT16):
                os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
                do_forced_float32 = True
                dtype = torch.bfloat16  # Change to bfloat16 loading
                break
        use_gradient_checkpointing = apply_unsloth_gradient_checkpointing(
            use_gradient_checkpointing, max_seq_length, dtype
        )
        with redirector:
            patch_loss_functions(torch_compile = False)
            model_types, supports_sdpa = unsloth_compile_transformers(
                dtype = dtype,
                model_name = model_name,
                model_types = model_types,
                token = token,
                sdpa_dynamic_mask = True,
                sdpa_bool_masks = True,
                sdpa_gqa_replace = True,
                sdpa_dynamic_compile = True,
                compile_attention = True,
                disable_causal_masks = True,
                compile_torch_modules = True,
                compile_custom_modules = True,
                compile_function_calls = True,
                fuse_lm_head = True,
                gradient_checkpointing = True,
                manual_replacements = True,
                fast_lora_forwards = True,
                fast_residual_stream = False,
                accurate_accumulation = True,
                epilogue_fusion = True,
                max_autotune = False,
                shape_padding = True,
                cudagraphs = False,
                debug = False,
                fullgraph = fullgraph,
                import_from_cache = False,
                disable = False,
                return_logits = return_logits,
                trust_remote_code = trust_remote_code,
                unsloth_force_compile = unsloth_force_compile,
            )
        # Fix SDPA issues
        for model_type in DISABLE_SDPA_MODEL_NAMES:
            if model_type in model_types_all:
                supports_sdpa = False

        # Keep the local checkpoint dir as tokenizer when self-sufficient; a VLM also needs local
        # processor files, else fall back to the base repo so its cached processor loads.
        _ckpt_arch = getattr(model_config, "architectures", None) or []
        _ckpt_is_vlm = any(x.endswith("ForConditionalGeneration") for x in _ckpt_arch) or hasattr(
            model_config, "vision_config"
        )
        tokenizer_name = _resolve_checkpoint_tokenizer_name(
            old_model_name, kwargs, require_processor = _ckpt_is_vlm
        )

        # Capture task intent before text_only can replace a parent VLM config with its nested text config.
        task_config_attrs = _get_user_task_config_attrs(user_config)
        for _cfg_key in ("num_labels", "id2label", "label2id", "problem_type"):
            _cfg_val = kwargs.get(_cfg_key, None)
            if _cfg_val is not None:
                task_config_attrs[_cfg_key] = _cfg_val
        _num_labels = task_config_attrs.get("num_labels", None)
        for _cfg_key, _cfg_val in task_config_attrs.items():
            set_task_config_attr(model_config, _cfg_key, _cfg_val)

        architectures = getattr(model_config, "architectures", None)
        if architectures is None:
            architectures = []
        is_vlm = any(x.endswith("ForConditionalGeneration") for x in architectures)
        is_vlm = is_vlm or hasattr(model_config, "vision_config")
        load_text_only = text_only and auto_model is None
        text_only_decoder = False
        if load_text_only:
            if hasattr(model_config, "vision_config"):
                text_config = _get_text_only_config(model_config, old_model_name)
                # Skip the vision tower only for families with their own text decoder (Gemma 3); others would
                # load random weights, so keep the full model.
                text_class = resolve_model_class(AutoModelForCausalLM, text_config)
                if text_class is None or not _is_family_text_decoder(
                    getattr(model_config, "model_type", ""),
                    getattr(text_config, "model_type", ""),
                ):
                    load_text_only = False
                else:
                    logger.warning_once(
                        f"Loading {old_model_name} as text-only; vision/audio towers skipped. "
                        "Use FastVisionModel for multimodal inputs."
                    )
                    # Remap VLM text weights (tf >= 5) while model_config is still the parent (#5816).
                    _apply_text_only_key_mapping(kwargs, model_config, text_config)
                    model_config = text_config
                    is_vlm = False
                    # model_config is no longer the repo's config, so anything rebuilding it from model_name (the
                    # device-map planner) sees a different model.
                    text_only_decoder = True
            else:
                is_vlm = False
        # If num_labels is set, use AutoModelForSequenceClassification.
        for _cfg_key, _cfg_val in task_config_attrs.items():
            set_task_config_attr(model_config, _cfg_key, _cfg_val)
        if auto_model is None:
            if _num_labels is not None:
                from transformers import AutoModelForSequenceClassification
                auto_model = AutoModelForSequenceClassification
            elif is_vlm:
                # Some repo-code VL models register only a generic auto class (Nemotron-VL uses
                # AutoModelForCausalLM, DeepSeek-OCR AutoModel), so the VLM auto class raises "Unrecognized
                # configuration class". Fall back to what the repo registered, matching the CONCRETE class
                # name, since transformers resolves remote code by that exact name.
                _auto_map = getattr(model_config, "auto_map", {}) or {}
                _vlm_class_name = AutoModelForVision2Seq.__name__
                _has_vlm_class = _vlm_class_name in _auto_map
                if not _has_vlm_class and "AutoModelForCausalLM" in _auto_map:
                    auto_model = AutoModelForCausalLM
                elif not _has_vlm_class and "AutoModel" in _auto_map:
                    from transformers import AutoModel
                    auto_model = AutoModel
                else:
                    auto_model = AutoModelForVision2Seq
            else:
                auto_model = AutoModelForCausalLM

        load_in_4bit_kwargs = load_in_4bit
        load_in_8bit_kwargs = load_in_8bit
        if quantization_config is not None and not fast_inference:
            load_in_4bit_kwargs = False
            load_in_8bit_kwargs = False

        # FastBaseModel remaps again via fast_inference_setup. Skip for PEFT: model_name is then the
        # base model, and `revision` names the adapter PeftModel loads below.
        if not is_peft:
            base_revision = _revision_for_resolved_repo(
                base_revision, model_name, old_model_name, mapper_moved_name
            )
        # A PEFT load resolved model_name to the base, which the caller's ref is not for.
        model_revision = base_revision if not is_peft else None

        model, tokenizer = FastBaseModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = _get_dtype(dtype),
            load_in_4bit = load_in_4bit_kwargs,
            load_in_8bit = load_in_8bit_kwargs,
            load_in_16bit = load_in_16bit,
            full_finetuning = full_finetuning,
            token = token,
            device_map = device_map,
            device_map_planner_kwargs = device_map_planner_kwargs,
            trust_remote_code = trust_remote_code,
            revision = model_revision,
            tokenizer_revision = _revision_for_tokenizer_repo(
                tokenizer_name, model_name, old_model_name, revision, model_revision, is_peft
            ),
            model_types = model_types,
            tokenizer_name = tokenizer_name,
            auto_model = auto_model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            supports_sdpa = supports_sdpa,
            whisper_language = whisper_language,
            whisper_task = whisper_task,
            auto_config = model_config,
            auto_config_from_caller = user_config is not None,
            # resize_token_embeddings below replaces the embedding module and hooks do not follow, so an
            # offload installed during the load would leave a CPU embedding feeding a GPU decoder. An
            # explicit request is left alone.
            offload_embedding = (
                False
                if resize_model_vocab is not None and offload_embedding == OFFLOAD_EMBEDDING_AUTO
                else offload_embedding
            ),
            float32_mixed_precision = float32_mixed_precision,
            # Pass vLLM/inference parameters.
            fast_inference = fast_inference,
            gpu_memory_utilization = gpu_memory_utilization,
            float8_kv_cache = float8_kv_cache,
            random_state = random_state,
            max_lora_rank = max_lora_rank,
            disable_log_stats = disable_log_stats,
            load_in_fp8 = load_in_fp8,
            text_only = load_text_only,
            text_only_decoder = text_only_decoder,
            *args,
            **kwargs,
        )

        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)
            # resize_token_embeddings rebuilds the embedding and drops _hf_hook, so this is the last module
            # swap of all, after every repair above.
            try:
                from unsloth.models.vision import _repair_dispatch_hooks
                _repaired = _repair_dispatch_hooks(model)
                if _repaired:
                    logger.info(
                        f"Unsloth: re-attached dispatch hooks to {_repaired} module(s) "
                        "left unhooked by the vocabulary resize."
                    )
            except Exception as _exc:
                logger.warning(
                    f"Unsloth: could not check the dispatch hooks after resizing "
                    f"the vocabulary ({type(_exc).__name__}: {_exc})."
                )

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(
                [
                    "unsloth",
                ]
            )
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(
                [
                    "unsloth",
                ]
            )

        if load_in_4bit:
            # Fix up the bitsandbytes config, but respect a user-provided quantization_config.
            if quantization_config is None:
                # load_in_4bit is the requested flag, not the effective one: a non-bnb checkpoint had bnb
                # disabled by check_and_disable, so a synthetic bnb config would corrupt its real one.
                try:
                    from unsloth_zoo.utils import get_quant_type
                    _stamp_bnb = get_quant_type(model.config) in (None, "bitsandbytes")
                except Exception:
                    _stamp_bnb = True
                if _stamp_bnb:
                    compute_dtype = dtype_from_config(model.config)
                    quantization_config = {
                        # compute_dtype is sometimes not a string.
                        "bnb_4bit_compute_dtype": compute_dtype,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "llm_int8_enable_fp32_cpu_offload": False,
                        "llm_int8_has_fp16_weight": False,
                        # Whatever the load really used. None here describes a layout that never existed, and saving it
                        # makes the adapter unreloadable.
                        "llm_int8_skip_modules": _loaded_skip_modules(model.config),
                        "llm_int8_threshold": 6.0,
                        "load_in_4bit": True,
                        "load_in_8bit": False,
                        "quant_method": "bitsandbytes",
                    }
                    model.config.update({"quantization_config": quantization_config})
            else:
                if hasattr(quantization_config, "to_dict"):
                    model.config.update({"quantization_config": quantization_config.to_dict()})
                elif isinstance(quantization_config, dict):
                    model.config.update({"quantization_config": quantization_config})

        if load_in_fp8 != False:
            _tag_model_with_fp8_torchao_config(model, fp8_mode)

        if is_peft:
            # From huggingface/peft#184

            # Gemma4 ClippableLinear wraps nn.Linear and PEFT cannot inject LoRA on it directly, so patch
            # PEFT to target the inner .linear child (same patch as vision.py). See huggingface/peft#3129.
            _clippable_linear_cls = None
            try:
                from transformers.models.gemma4.modeling_gemma4 import (
                    Gemma4ClippableLinear as _clippable_linear_cls,
                )
            except ImportError:
                pass

            if _clippable_linear_cls is not None:
                from peft.tuners.lora.model import LoraModel as _LoraModel

                _original_car = _LoraModel._create_and_replace

                def _patched_car(
                    self,
                    peft_config,
                    adapter_name,
                    target,
                    target_name,
                    parent,
                    current_key = None,
                    **kwargs,
                ):
                    if isinstance(target, _clippable_linear_cls):
                        return _original_car(
                            self,
                            peft_config,
                            adapter_name,
                            target.linear,
                            "linear",
                            target,
                            current_key = current_key,
                            **kwargs,
                        )
                    return _original_car(
                        self,
                        peft_config,
                        adapter_name,
                        target,
                        target_name,
                        parent,
                        current_key = current_key,
                        **kwargs,
                    )

                _LoraModel._create_and_replace = _patched_car

            # Warm the adapter repo: PeftModel downloads it in-process and can hang on Xet, and it always
            # loads in-process, so warm it even under fast_inference.
            _prefetched = maybe_prefetch_hf_snapshot(
                old_model_name,
                token = token,
                revision = revision,
                cache_dir = kwargs.get("cache_dir"),
                local_files_only = local_files_only,
                # Adapter always loads in-process via PeftModel, so warm it even under fast_inference.
                fast_inference = False,
                force_download = kwargs.get("force_download", False),
                # Leave use_safetensors auto, since inheriting the base format could skip a safetensors-only
                # adapter; adapter_only restricts the warm to adapter files plus root aux.
                adapter_only = True,
            )
            # The child did the forced download; clear the flag so the load reuses the warm cache.
            if _prefetched and kwargs.get("force_download", False):
                kwargs["force_download"] = False
            # Forward cache_dir so the load reads the warmed adapter. No subfolder: that targets the base
            # checkpoint, and adapters live at the root.
            peft_load_kwargs = {}
            if kwargs.get("cache_dir") is not None:
                peft_load_kwargs["cache_dir"] = kwargs["cache_dir"]
            try:
                model = PeftModel.from_pretrained(
                    model,
                    old_model_name,
                    token = token,
                    revision = revision,
                    local_files_only = local_files_only,
                    is_trainable = True,
                    trust_remote_code = trust_remote_code,
                    **peft_load_kwargs,
                )
            finally:
                # Always restore the original PEFT method, even if loading fails.
                if _clippable_linear_cls is not None:
                    _LoraModel._create_and_replace = _original_car

            model = FastBaseModel.post_patch_model(
                model, use_gradient_checkpointing, trust_remote_code = trust_remote_code
            )
            try:
                from .vision import _lift_endpoint_hooks_onto_adapters
                _lift_endpoint_hooks_onto_adapters(model)
            except Exception:
                pass  # never block loading on a placement nicety
            # Re-evaluate grouped MoE now the adapter is attached: an expert-LoRA block falls back to the
            # original loop, an attention-only adapter keeps the grouped path.
            try:
                from unsloth_zoo.temporary_patches.moe_grouped_modulelist import (
                    auto_enable_grouped_moe,
                )
                auto_enable_grouped_moe(model)
            except Exception:
                pass  # optional speedup; never block model loading

        if qat_scheme is not None:
            print("Unsloth: Applying QAT to mitigate quantization degradation")
            model = FastModel._prepare_for_qat(model, qat_scheme)

        # Patch Tiled MLP; enable by setting UNSLOTH_TILED_MLP to "arctic", "target", or "target:{GB}".
        patch_tiled_mlp_choice = os.environ.get(
            "UNSLOTH_TILED_MLP", "arctic" if unsloth_tiled_mlp else "0"
        )
        if patch_tiled_mlp_choice != "0" or unsloth_tiled_mlp:
            patch_tiled_mlp(model, patch_options_str = patch_tiled_mlp_choice)

        model = _fix_rope_inv_freq(model)
        model = _exclude_rope_inv_freq_from_ddp(model)
        model = _mark_forced_float32(model, do_forced_float32)
        model = _mark_full_finetuning(model, full_finetuning)
        return _mark_requested_float32(model, user_float32), tokenizer


class FastVisionModel(FastModel):
    pass


class FastTextModel(FastModel):
    pass
