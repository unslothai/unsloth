# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DDP defaults for Unsloth trainers.

transformers' Trainer passes ``find_unused_parameters=True`` to DDP whenever the model is not a
``PreTrainedModel`` and ``ddp_find_unused_parameters`` is None, which is every PEFT model. Under reentrant
gradient checkpointing that crashes on the first backward ("marked as ready twice"): the checkpointed
forward runs without grad, DDP marks every LoRA weight unused, then the recompute marks them again.
transformers already picks False for a ``PreTrainedModel`` under checkpointing; this does the same for a
plain LoRA adapter on a dense text model, where every trainable parameter gets a gradient each step.
Without checkpointing True works (and tolerates an adapter a step never runs), so it is left alone.
"""

import os

__all__ = ["unsloth_default_ddp_find_unused_parameters"]

_AUTO_FLAG = "_unsloth_auto_ddp_find_unused_parameters"
# LoRA tensors that are used whenever their base layer runs.
_LORA_PARAM_MARKERS = (".lora_A.", ".lora_B.", ".lora_magnitude_vector.")
# Modules a step can skip (unrouted experts, a vision tower on a text batch, MTP heads), or that
# Unsloth's fused loss / embedding paths may read by weight and bypass the LoRA forward.
_SKIPPABLE_NAME_MARKERS = (
    "expert",
    "vision",
    "visual",
    "audio",
    "image",
    "projector",
    "mtp",
    "lm_head",
    "embed",
)
_SUBMODEL_CONFIG_KEYS = ("vision_config", "audio_config", "vision_tower", "speech_config")
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_tok",
)


def _is_distributed():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > 1
    except Exception:
        pass
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _configs(config):
    seen = []
    stack = [config]
    while stack:
        c = stack.pop()
        if c is None or any(c is s for s in seen):
            continue
        seen.append(c)
        for key in ("text_config", "language_config", "llm_config"):
            stack.append(getattr(c, key, None))
    return seen


def _config_may_skip(config):
    for c in _configs(config):
        for key in _SUBMODEL_CONFIG_KEYS:
            if getattr(c, key, None) is not None:
                return True
        for key in _MOE_CONFIG_KEYS:
            value = getattr(c, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return True
        if (getattr(c, "layerdrop", 0) or 0) > 0:
            return True
    return False


def _gradient_checkpointing_on(args, model):
    """True unless checkpointing is off both in the args (Trainer turns it on in train()) and on the model."""
    if getattr(args, "gradient_checkpointing", False):
        return True
    try:
        return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    except Exception:
        return True


def _lora_uses_every_trainable_param(model):
    """True only when we can show every trainable tensor is a LoRA weight run on each forward."""
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        # transformers already decides for these (False under gradient checkpointing).
        return False
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        return False
    cfg = next(iter(peft_config.values()))
    if (
        str(getattr(getattr(cfg, "peft_type", None), "value", getattr(cfg, "peft_type", "")))
        != "LORA"
    ):
        return False
    for attr in ("modules_to_save", "target_parameters", "trainable_token_indices"):
        if getattr(cfg, attr, None):
            return False
    base = getattr(model, "get_base_model", lambda: None)()
    if _config_may_skip(getattr(base, "config", None) or getattr(model, "config", None)):
        return False
    n_trainable = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_trainable += 1
        if not any(m in name for m in _LORA_PARAM_MARKERS):
            return False
        lowered = name.lower()
        if any(m in lowered for m in _SKIPPABLE_NAME_MARKERS):
            return False
    return n_trainable > 0


def unsloth_default_ddp_find_unused_parameters(args, model):
    """Set ``args.ddp_find_unused_parameters = False`` for a plain LoRA model under DDP with gradient
    checkpointing when the user left it None.

    Kept at None (transformers' True) for anything a step may skip: MoE / expert LoRA, vision or audio
    towers, MTP heads, lm_head / embedding adapters, modules_to_save, several adapters, non-LoRA PEFT.
    Returns the value now on ``args``.
    """
    if args is None or not hasattr(args, "ddp_find_unused_parameters"):
        return None
    if getattr(args, _AUTO_FLAG, False):
        # Reused args object: our earlier default is not a user choice, re-decide for this model.
        args.ddp_find_unused_parameters = None
        object.__setattr__(args, _AUTO_FLAG, False)
    if args.ddp_find_unused_parameters is not None:
        return args.ddp_find_unused_parameters
    if os.environ.get("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", "1") == "0":
        return None
    if not _is_distributed():
        return None
    try:
        safe = _gradient_checkpointing_on(args, model) and _lora_uses_every_trainable_param(model)
    except Exception:
        safe = False
    if safe:
        args.ddp_find_unused_parameters = False
        object.__setattr__(args, _AUTO_FLAG, True)
    return args.ddp_find_unused_parameters
