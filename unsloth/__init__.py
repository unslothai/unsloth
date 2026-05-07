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
    try:
        import unsloth_zoo
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires `unsloth-zoo` with MLX modules. "
            "Reinstall with `pip install unsloth-zoo` or rerun install.sh."
        ) from _e
    # The mlx_trainer / mlx_loader submodules ship with unsloth-zoo's MLX
    # support. An older installed unsloth-zoo (e.g. from PyPI before the
    # MLX release lands) will satisfy `import unsloth_zoo` but be missing
    # these submodules. Surface the same friendly install hint instead of
    # a raw ImportError on the submodule path.
    try:
        from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx_loader import FastMLXModel
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires an unsloth-zoo build that includes "
            "`unsloth_zoo.mlx_trainer` and `unsloth_zoo.mlx_loader`. Upgrade with "
            "`pip install -U unsloth-zoo` or rerun install.sh."
        ) from _e

    # Load raw_text helpers without executing dataprep/__init__.py, which
    # imports synthetic.py -> torch and would defeat the torch-free MLX path.
    from pathlib import Path as _Path

    _raw_text_path = _Path(__file__).resolve().parent / "dataprep" / "raw_text.py"
    _raw_text_spec = importlib.util.spec_from_file_location(
        "unsloth._mlx_raw_text", _raw_text_path
    )
    if _raw_text_spec is None or _raw_text_spec.loader is None:
        raise ImportError("Unsloth: could not load MLX raw_text dataprep helpers.")
    _raw_text = importlib.util.module_from_spec(_raw_text_spec)
    _raw_text_spec.loader.exec_module(_raw_text)
    RawTextDataLoader = _raw_text.RawTextDataLoader
    TextPreprocessor = _raw_text.TextPreprocessor
    del _raw_text, _raw_text_spec, _raw_text_path, _Path

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
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("text_only", False)
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def for_training(*args, **kwargs):
            return args[0] if args else None

    FastTextModel = FastLanguageModel
    FastModel = FastLanguageModel

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
        try:
            import mlx.core as mx

            name = mx.device_info().get("device_name", "") or ""
            return not name.startswith(("Apple M1", "Apple M2"))
        except Exception:
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
