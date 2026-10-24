# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# CPU backend: reached only when unsloth_zoo found no accelerator (or UNSLOTH_FORCE_CPU=1). Like the
# MLX branch it never imports the GPU model / kernel stack: stock transformers + PEFT + TRL, with
# unsloth_zoo.cpu choosing dtype, 4-bit, chunked cross entropy and trainer defaults.

import os
import torch

from ._version import __version__
from unsloth_zoo.cpu import (
    FastCPUModel,
    cpu_capabilities,
    cpu_bf16_supported,
    install_cpu_training_defaults,
)

DEVICE_TYPE = "cpu"
DEVICE_TYPE_TORCH = "cpu"
DEVICE_COUNT = 0

install_cpu_training_defaults()

print(
    f"Unsloth {__version__}: no GPU found, training on CPU ({__import__('platform').machine()}, "
    f"{torch.get_num_threads()} threads, bfloat16 = {cpu_bf16_supported()}). CPU training is much "
    f"slower than GPU training; supported: text LoRA / full finetuning with SFTTrainer. "
    f"Set UNSLOTH_DISABLE_CPU_FALLBACK=1 to raise instead."
)


class FastLanguageModel(FastCPUModel):
    pass


FastModel = FastLanguageModel
FastTextModel = FastLanguageModel


class FastVisionModel(FastLanguageModel):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise NotImplementedError(
            "Unsloth: vision / audio models are not supported on the CPU backend yet."
        )


def is_bfloat16_supported(including_emulation = False):
    return True if including_emulation else cpu_bf16_supported()


def is_vLLM_available():
    return False


def get_chat_template(*args, **kwargs):
    from .chat_templates import get_chat_template as _get_chat_template
    return _get_chat_template(*args, **kwargs)


def apply_chat_template(*args, **kwargs):
    from .chat_templates import apply_chat_template as _apply_chat_template
    return _apply_chat_template(*args, **kwargs)


def standardize_data_formats(*args, **kwargs):
    from unsloth_zoo.dataset_utils import standardize_data_formats as _standardize_data_formats
    return _standardize_data_formats(*args, **kwargs)


def standardize_sharegpt(*args, **kwargs):
    return standardize_data_formats(*args, **kwargs)


def train_on_responses_only(*args, **kwargs):
    from .chat_templates import train_on_responses_only as _train_on_responses_only
    if _train_on_responses_only is None:
        from unsloth_zoo.dataset_utils import train_on_responses_only as _train_on_responses_only
    return _train_on_responses_only(*args, **kwargs)


try:
    from trl import SFTConfig as _SFTConfig, SFTTrainer as _SFTTrainer
except ImportError:
    # trl is not a default dependency on Apple Silicon: importing unsloth must still work there.
    class _SFTConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("Unsloth: CPU training needs TRL. Install it with `pip install trl`.")

    _SFTTrainer = _SFTConfig


class UnslothTrainingArguments(_SFTConfig):
    def __init__(
        self,
        embedding_learning_rate = None,
        q_galore_config = None,
        *args,
        loraplus_lr_ratio = None,
        **kwargs,
    ):
        # Same positional order as the GPU class; Q-GaLore and LoRA+ are GPU-only, so they must stay unset.
        if q_galore_config is not None or loraplus_lr_ratio is not None:
            raise NotImplementedError(
                "Unsloth: q_galore_config / loraplus_lr_ratio are not supported on the CPU backend."
            )
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)


class UnslothTrainer(_SFTTrainer):
    def create_optimizer(self, *args, **kwargs):
        embedding_lr = getattr(self.args, "embedding_learning_rate", None)
        if embedding_lr is None or self.optimizer is not None:
            return super().create_optimizer(*args, **kwargs)
        # Same split as the GPU UnslothTrainer: trained embed_tokens / lm_head copies get their own lr.
        model = self.model
        decay = set(self.get_decay_parameter_names(model))
        custom = getattr(self, "optimizer_cls_and_kwargs", None)
        if custom is not None:
            optimizer_cls, optimizer_kwargs = custom[0], dict(custom[1])
        else:
            optimizer_cls, optimizer_kwargs = _SFTTrainer.get_optimizer_cls_and_kwargs(
                self.args, model
            )
        if any(key in optimizer_kwargs for key in ("params", "model", "optimizer_dict")):
            raise NotImplementedError(
                f"Unsloth: embedding_learning_rate cannot be combined with optim = {self.args.optim!r} on CPU."
            )
        embedding_ids = set()
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            module = getattr(model, getter, lambda: None)()
            if getattr(module, "weight", None) is not None:
                embedding_ids.add(id(module.weight))
        # Older transformers models can return None from get_output_embeddings yet still own an lm_head.
        for owner in (model, getattr(getattr(model, "base_model", None), "model", None)):
            weight = getattr(getattr(owner, "lm_head", None), "weight", None)
            if weight is not None:
                embedding_ids.add(id(weight))
        groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_embedding = (
                name.endswith("modules_to_save.default.weight") or id(param) in embedding_ids
            )
            key = (is_embedding, name in decay)
            groups.setdefault(key, []).append(param)
        self.optimizer = optimizer_cls(
            [
                {
                    "params": params,
                    "lr": embedding_lr if is_embedding else self.args.learning_rate,
                    "weight_decay": self.args.weight_decay if decays else 0.0,
                }
                # Same order as the GPU trainer's groups, so its checkpoints resume here.
                for (is_embedding, decays), params in sorted(
                    groups.items(), key = lambda kv: (kv[0][0], not kv[0][1])
                )
            ],
            **optimizer_kwargs,
        )
        return self.optimizer


class UnslothVisionDataCollator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Unsloth: vision / audio models are not supported on the CPU backend yet."
        )


def _install_cpu_trainer_module():
    # `from unsloth.trainer import ...` would otherwise load unsloth/trainer.py and the GPU kernel stack.
    import sys
    import types
    import importlib.machinery

    module_name = __name__.rsplit(".", 1)[0] + ".trainer"
    trainer = types.ModuleType(module_name)
    trainer.__package__ = module_name.rsplit(".", 1)[0]
    trainer.__spec__ = importlib.machinery.ModuleSpec(module_name, loader = None)
    trainer.UnslothTrainer = UnslothTrainer
    trainer.UnslothTrainingArguments = UnslothTrainingArguments
    trainer.UnslothVisionDataCollator = UnslothVisionDataCollator
    sys.modules[module_name] = trainer
    package = sys.modules.get(trainer.__package__)
    if package is not None:
        setattr(package, "trainer", trainer)


_install_cpu_trainer_module()
del _install_cpu_trainer_module

__all__ = [
    "FastLanguageModel",
    "FastModel",
    "FastTextModel",
    "FastVisionModel",
    "is_bfloat16_supported",
    "is_vLLM_available",
    "get_chat_template",
    "apply_chat_template",
    "standardize_data_formats",
    "standardize_sharegpt",
    "train_on_responses_only",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "UnslothVisionDataCollator",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "__version__",
]
