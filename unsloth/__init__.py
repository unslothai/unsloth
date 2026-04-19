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
    __version__ = unsloth_zoo.__version__

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def get_peft_model(*args, **kwargs):
            return FastMLXModel.get_peft_model(*args, **kwargs)

        @staticmethod
        def for_inference(*args, **kwargs):
            raise NotImplementedError("Unsloth: for_inference not yet supported on MLX.")

    FastModel = FastLanguageModel
    FastVisionModel = FastLanguageModel

else:
    # GPU path: load everything from _gpu_init
    from ._gpu_init import *
    from ._gpu_init import __version__
