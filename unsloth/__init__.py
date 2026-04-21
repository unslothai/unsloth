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

import os, platform, importlib.util

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Detect Apple Silicon + MLX before any torch/numpy imports
_IS_MLX = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
)

if _IS_MLX:
    import unsloth_zoo
    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx_loader import FastMLXModel
    from .dataprep.raw_text import RawTextDataLoader, TextPreprocessor
    __version__ = unsloth_zoo.__version__
    DEVICE_TYPE = "mlx"

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def get_peft_model(*args, **kwargs):
            return FastMLXModel.get_peft_model(*args, **kwargs)

        @staticmethod
        def for_inference(*args, **kwargs):
            return args[0] if args else None

    class FastVisionModel(FastLanguageModel):
        @staticmethod
        def for_training(*args, **kwargs):
            return args[0] if args else None

    FastTextModel = FastLanguageModel
    FastModel = FastLanguageModel
    FastVisionModel = FastLanguageModel

    class FastSentenceTransformer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

        @staticmethod
        def get_peft_model(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

    def is_bfloat16_supported():
        return True

    is_bf16_supported = is_bfloat16_supported

    class UnslothVisionDataCollator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Unsloth: UnslothVisionDataCollator is not used on MLX. "
                "Use the MLX trainer/data path instead."
            )

else:
    # GPU path: load everything from _gpu_init
    from ._gpu_init import *
    from ._gpu_init import __version__
