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
        raise NotImplementedError("Unsloth: vision / audio models are not supported on the CPU backend yet.")


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


def _unsloth_trainer_classes():
    from trl import SFTConfig, SFTTrainer

    class UnslothTrainingArguments(SFTConfig):
        def __init__(self, *args, embedding_learning_rate = None, **kwargs):
            self.embedding_learning_rate = embedding_learning_rate
            super().__init__(*args, **kwargs)

    class UnslothTrainer(SFTTrainer):
        pass

    return UnslothTrainer, UnslothTrainingArguments


UnslothTrainer, UnslothTrainingArguments = _unsloth_trainer_classes()
del _unsloth_trainer_classes

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
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
]
