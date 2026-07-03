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

import os, importlib.util, platform

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# ── Windows console UTF-8 safety ─────────────────────────────────────────────
# Legacy Windows consoles (cp1252) can't encode Unsloth's emoji/box-drawing
# glyphs and crash with UnicodeEncodeError. Force stdout/stderr to UTF-8 only on
# Windows and only when not already UTF-8; no-op elsewhere. errors="replace"
# guarantees we never crash on an unencodable glyph.
if platform.system() == "Windows":
    import sys as _sys
    for _name in ("stdout", "stderr"):
        _s = getattr(_sys, _name, None)
        try:
            _enc = (getattr(_s, "encoding", None) or "").lower()
            if _s is not None and hasattr(_s, "reconfigure") and "utf" not in _enc:
                _s.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass


def _is_mlx_available():
    # Transitional import barrier: keep non-Apple-Silicon imports from touching
    # unsloth_zoo until unsloth_zoo.mlx is import-safe on GPU hosts. Then this
    # can collapse back to the centralized zoo runtime call below.
    if (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") == "1"
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or importlib.util.find_spec("mlx") is None
    ):
        return False
    try:
        from unsloth_zoo.mlx import is_mlx_available
    except ImportError:
        return False
    return is_mlx_available()


def _apply_mlx_trainer_compat(MLXTrainer, MLXTrainingConfig):
    # Keep the externally-observed dataclass field order stable when newer
    # unsloth-zoo builds insert load_best_model_at_end ahead of dataset fields.
    fields = getattr(MLXTrainingConfig, "__dataclass_fields__", None)
    if (
        isinstance(fields, dict)
        and "dataset_text_field" in fields
        and "load_best_model_at_end" in fields
    ):
        ordered_names = list(fields.keys())
        dataset_idx = ordered_names.index("dataset_text_field")
        load_best_idx = ordered_names.index("load_best_model_at_end")
        if load_best_idx < dataset_idx:
            ordered_names.pop(load_best_idx)
            dataset_idx = ordered_names.index("dataset_text_field")
            ordered_names.insert(dataset_idx + 1, "load_best_model_at_end")
            MLXTrainingConfig.__dataclass_fields__ = {
                name: fields[name] for name in ordered_names
            }

    # Newer MLX trainer paths use a dedicated dataset accessor for batch planning.
    # Older trainers and local test doubles only expose train_dataset.
    if not hasattr(MLXTrainer, "_train_dataset_for_batches"):
        def _train_dataset_for_batches(self):
            return getattr(self, "train_dataset", None)
        MLXTrainer._train_dataset_for_batches = _train_dataset_for_batches

    return MLXTrainer, MLXTrainingConfig


# Detect Apple Silicon + MLX before any torch/numpy imports
_IS_MLX = _is_mlx_available()

if _IS_MLX:
    try:
        import unsloth_zoo
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires `unsloth-zoo` with MLX modules. "
            "Reinstall with `pip install unsloth-zoo` or rerun install.sh."
        ) from _e
    # An older unsloth-zoo satisfies `import unsloth_zoo` but lacks the
    # mlx.trainer / mlx.loader submodules. Surface a friendly install hint
    # instead of a raw ImportError on the submodule path.
    try:
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx.loader import FastMLXModel
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires an unsloth-zoo build that includes "
            "`unsloth_zoo.mlx.trainer` and `unsloth_zoo.mlx.loader`. Upgrade with "
            "`pip install -U unsloth-zoo` or rerun install.sh."
        ) from _e
    MLXTrainer, MLXTrainingConfig = _apply_mlx_trainer_compat(
        MLXTrainer, MLXTrainingConfig
    )

    # Load raw_text helpers without executing dataprep/__init__.py, which
    # imports synthetic.py -> torch and would defeat the torch-free MLX path.
    from pathlib import Path as _Path

    _raw_text_path = _Path(__file__).resolve().parent / "dataprep" / "raw_text.py"
    _raw_text_spec = importlib.util.spec_from_file_location("unsloth._mlx_raw_text", _raw_text_path)
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
