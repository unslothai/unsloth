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
"""
FastDiffusionModel: a transformers-only slow path for text-diffusion models (e.g. DiffusionGemma).

These models use a block-diffusion sampling loop (custom generate) and a novel backbone, so we skip
Unsloth's autoregressive kernel/compile patching and load the unmodified HF model (outputs stay
bit-identical to transformers), keeping only the safe conveniences: 4bit/8bit loading, PEFT LoRA, the
(model, tokenizer) API, and for_inference/for_training. Extend DIFFUSION_MODEL_TYPES as more land.
"""

import os
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from ._utils import is_bfloat16_supported, maybe_prefetch_hf_snapshot
from .llama import logger

__all__ = ["FastDiffusionModel", "DIFFUSION_MODEL_TYPES", "is_diffusion_model_type"]

# transformers model_type strings routed to this slow path
DIFFUSION_MODEL_TYPES = ("diffusion_gemma", "diffusion_gemma4")

# Default LoRA targets: standard nn.Linear modules in the shared Gemma-4 backbone. The 128 MoE experts
# are fused 3D Parameters (gate_up_proj/down_proj), not nn.Linear, so PEFT LoRA cannot target them.
DIFFUSION_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # attention
    "gate_proj",
    "up_proj",
    "down_proj",  # dense (non-expert) MLP
]

# Vision tower uses a custom Linear with the same suffix names; exclude it so only the text path is wrapped.
DIFFUSION_LORA_EXCLUDE = r".*(vision_tower|embed_vision).*"


def is_diffusion_model_type(model_types):
    """model_types: str or iterable -> True if any is a known diffusion model_type."""
    if isinstance(model_types, str):
        model_types = (model_types,)
    return any(mt in DIFFUSION_MODEL_TYPES for mt in model_types)


def _resolve_diffusion_model_class(config):
    """Resolve the HF model class for a diffusion checkpoint from config.architectures."""
    import transformers

    archs = getattr(config, "architectures", None) or []
    for arch in archs:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    # Fallbacks across naming revisions.
    for name in (
        "DiffusionGemmaForBlockDiffusion",
        "DiffusionGemma4ModelForBlockDiffusion",
        "DiffusionGemma4ForBlockDiffusion",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        f"Unsloth: could not resolve a diffusion model class from architectures={archs}. "
        "Ensure you have the transformers build that ships the DiffusionGemma implementation."
    )


def _load_diffusion_config(
    model_name,
    token,
    trust_remote_code,
    revision,
    local_files_only,
    cache_dir = None,
):
    """Load the config, aliasing the legacy ``diffusion_gemma`` model_type to the ``diffusion_gemma4``
    classes current transformers ships. AutoConfig raises on the legacy type; catch that, rewrite the
    type/arch names in-memory, and rebuild."""
    try:
        return AutoConfig.from_pretrained(
            model_name,
            token = token,
            trust_remote_code = trust_remote_code,
            revision = revision,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
    except ValueError as e:
        if "diffusion_gemma" not in str(e):
            raise
        import json
        from transformers.utils import cached_file

        cfg_path = cached_file(
            model_name,
            "config.json",
            token = token,
            revision = revision,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
        with open(cfg_path, encoding = "utf-8") as f:
            cd = json.load(f)
        cd["model_type"] = "diffusion_gemma4"
        cd.setdefault("architectures", ["DiffusionGemma4ModelForBlockDiffusion"])
        if isinstance(cd.get("text_config"), dict):
            cd["text_config"]["model_type"] = "diffusion_gemma4_text"
        if isinstance(cd.get("vision_config"), dict):
            cd["vision_config"]["model_type"] = "diffusion_gemma4_vision"
        from transformers import DiffusionGemma4Config

        return DiffusionGemma4Config.from_dict(cd)


class FastDiffusionModel:
    """transformers-only slow path for text-diffusion models."""

    @staticmethod
    def from_pretrained(
        model_name = "google/diffusiongemma-26B-A4B-it",
        max_seq_length = None,  # API-compat; diffusion uses canvas_length
        dtype = None,
        load_in_4bit = False,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        token = None,
        device_map = "auto",
        trust_remote_code = False,
        attn_implementation = "eager",  # exact match with the reference golden logits
        revision = None,
        return_tokenizer = True,
        **kwargs,
    ):
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        assert dtype in (torch.float16, torch.bfloat16, torch.float32)

        # Honor an explicit local_files_only; else fall back to the offline env vars.
        local_files_only = kwargs.pop("local_files_only", None)
        if local_files_only is None:
            local_files_only = (
                os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            )

        cache_dir = kwargs.get("cache_dir")

        config = _load_diffusion_config(
            model_name,
            token,
            trust_remote_code,
            revision,
            local_files_only,
            cache_dir = cache_dir,
        )
        model_type = getattr(config, "model_type", None)
        if not is_diffusion_model_type(model_type):
            raise RuntimeError(
                f"Unsloth: FastDiffusionModel only supports diffusion model_types {DIFFUSION_MODEL_TYPES}, "
                f"got '{model_type}'. Use FastModel/FastLanguageModel for autoregressive models."
            )

        model_cls = _resolve_diffusion_model_class(config)

        # Prefetch the whole repo root so the weight load is a cache hit. No subfolder: the pipeline
        # loads every component subfolder, so narrowing would leave unet/vae/text_encoder to Xet.
        maybe_prefetch_hf_snapshot(
            model_name,
            token = token,
            revision = revision,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            fast_inference = False,
            force_download = kwargs.get("force_download", False),
            use_safetensors = kwargs.get("use_safetensors"),
            # Forward variant (e.g. "fp16") so the warm keeps variant weights.
            variant = kwargs.get("variant"),
        )

        load_kwargs = dict(
            dtype = dtype,
            device_map = device_map,
            token = token,
            trust_remote_code = trust_remote_code,
            attn_implementation = attn_implementation,
            revision = revision,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
        # Match the load's weight format to the warm (None/auto already matches).
        if kwargs.get("use_safetensors") is not None:
            load_kwargs["use_safetensors"] = kwargs["use_safetensors"]
        # Forward variant to the real load so it reads the warmed variant weights.
        if kwargs.get("variant") is not None:
            load_kwargs["variant"] = kwargs["variant"]

        # Optional bitsandbytes quant. The MoE experts (3D Parameters) are not nn.Linear so bnb skips
        # them; only attention + dense MLP Linears quantize, lm_head/embeddings stay full precision.
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                qcfg = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_compute_dtype = dtype,
                    llm_int8_skip_modules = [
                        "lm_head",
                        "embed_tokens",
                        "experts",
                        "self_conditioning",
                        "router",
                    ],
                )
            else:
                qcfg = BitsAndBytesConfig(load_in_8bit = True)
            load_kwargs["quantization_config"] = qcfg

        print(f"==((  Unsloth: FastDiffusionModel (slow / transformers-only path)  ))==")
        print(f"   Model: {model_name}  | class: {model_cls.__name__}  | model_type: {model_type}")
        print(
            f"   dtype: {dtype} | 4bit: {load_in_4bit} | 8bit: {load_in_8bit} | attn: {attn_implementation}"
        )

        model = model_cls.from_pretrained(model_name, **load_kwargs).eval()
        # Mark before any early return so get_peft_model/for_* route to the slow path.
        model._unsloth_slow_diffusion = True

        if not return_tokenizer:
            return model, None

        # Prefer the processor (chat template + tokenizer); fall back to a bare tokenizer. Returned as
        # "tokenizer" to match the Unsloth (model, tokenizer) contract.
        try:
            tokenizer = AutoProcessor.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                revision = revision,
                local_files_only = local_files_only,
                cache_dir = cache_dir,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                revision = revision,
                local_files_only = local_files_only,
                cache_dir = cache_dir,
            )

        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = None,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
        task_type = None,
        **kwargs,
    ):
        """Attach a PEFT LoRA to the diffusion backbone (attention + dense MLP). No fused kernels."""
        from peft import LoraConfig, get_peft_model as peft_get_peft_model

        if target_modules is None:
            target_modules = DIFFUSION_LORA_TARGETS

        # NOTE: use_dora (and any other LoraConfig kwarg outside this allowlist,
        # e.g. use_rslora) is silently dropped here. Studio does not reach this
        # path today, so it's untested/unsupported on diffusion models; add it
        # to the allowlist below if that changes.
        lora_kwargs = dict(
            r = r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = bias,
            target_modules = target_modules,
            task_type = task_type,  # None: diffusion has no standard CAUSAL_LM head
            **{k: v for k, v in kwargs.items() if k in ("modules_to_save", "init_lora_weights")},
        )
        # Exclude the vision tower's custom (non-Linear) modules that share suffix names.
        exclude = kwargs.get("exclude_modules", DIFFUSION_LORA_EXCLUDE)
        try:
            lora_config = LoraConfig(exclude_modules = exclude, **lora_kwargs)
        except TypeError:
            # Older PEFT without exclude_modules: scope the target to the text decoder by regex.
            lora_kwargs["target_modules"] = (
                r".*model\.decoder\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
            )
            lora_config = LoraConfig(**lora_kwargs)
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        model = peft_get_peft_model(model, lora_config)
        model._unsloth_slow_diffusion = True
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
        for _, m in model.named_modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        model.train()
        if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
