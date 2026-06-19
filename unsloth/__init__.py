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


class _UnslothDeviceStats:
    """Portable device metadata used by notebook memory-reporting cells."""

    def __init__(
        self,
        name,
        total_memory = 0,
    ):
        """Store a display name and total memory in bytes."""
        self.name = name
        self.total_memory = int(total_memory or 0)


def _bytes_to_gb(value):
    """Convert byte counts to GiB rounded"""
    return round(float(value or 0) / 1024 / 1024 / 1024, 3)


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
        from unsloth_zoo.mlx.trainer import (
            MLXTrainer,
            MLXTrainingConfig,
            _normalize_mlx_optimizer_name,
        )
        from unsloth_zoo.mlx.loader import FastMLXModel
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires an unsloth-zoo build that includes "
            "`unsloth_zoo.mlx.trainer` and `unsloth_zoo.mlx.loader`. Upgrade with "
            "`pip install -U unsloth-zoo` or rerun install.sh."
        ) from _e

    import dataclasses as _dataclasses
    import importlib.machinery as _machinery
    import sys as _sys
    import types as _types
    import warnings as _warnings

    __version__ = unsloth_zoo.__version__
    DEVICE_TYPE = "mlx"

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

    def get_gpu_memory_stats():
        """Return MLX device stats, peak memory, and total memory in GiB."""
        import mlx.core as mx

        info = mx.device_info()
        total = info.get("memory_size") or info.get("max_recommended_working_set_size") or 0
        get_peak_memory = getattr(mx, "get_peak_memory", None)
        if get_peak_memory is None and hasattr(mx, "metal"):
            get_peak_memory = getattr(mx.metal, "get_peak_memory", None)
        peak = get_peak_memory() if callable(get_peak_memory) else 0
        stats = _UnslothDeviceStats(info.get("device_name", "Apple GPU"), total)
        max_memory = _bytes_to_gb(total) or 1.0
        return stats, _bytes_to_gb(peak), max_memory

    def clear_gpu_memory():
        """Clear MLX's cached GPU memory for notebook cleanup cells."""
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    _MLX_TRAINING_CONFIG_FIELDS = {_field.name for _field in _dataclasses.fields(MLXTrainingConfig)}
    _MLX_TRAINING_ARGUMENT_ALIASES = {
        "max_length": "max_seq_length",
    }
    _MLX_COMPAT_EXTRA_ARGUMENTS = frozenset(
        (
            "bf16",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataset_kwargs",
            "disable_tqdm",
            "eval_strategy",
            "fp16",
            "full_determinism",
            "gradient_checkpointing_kwargs",
            "hub_model_id",
            "hub_token",
            "logging_strategy",
            "neftune_noise_alpha",
            "optim_args",
            "padding_free",
            "push_to_hub",
            "remove_unused_columns",
            "save_on_each_node",
            "save_safetensors",
            "save_strategy",
            "torch_compile",
        )
    )
    _MLX_IMPLEMENTED_EXTRA_ARGUMENTS = frozenset(("warmup_ratio",))
    _MLX_ALLOWED_EXTRA_ARGUMENTS = _MLX_COMPAT_EXTRA_ARGUMENTS | _MLX_IMPLEMENTED_EXTRA_ARGUMENTS
    _MLX_UNSUPPORTED_TASK_ARGUMENTS = frozenset(
        (
            "assistant_only_loss",
            "completion_only_loss",
        )
    )

    def _is_mlx_no_save_strategy(value):
        """Return whether a Transformers save strategy disables checkpointing."""
        if hasattr(value, "value"):
            value = value.value
        strategy = str(value or "").strip().lower()
        strategy = strategy.rsplit(".", 1)[-1]
        return strategy in ("no", "none", "false")

    def _normalize_mlx_training_value(key, value):
        """Normalize TRL/Transformers argument values to MLX-compatible values."""
        if key == "optim":
            return _normalize_mlx_optimizer_name(value)
        return value

    def _mlx_training_argument_values(args):
        """Extract MLX-compatible fields from a TRL/Transformers args object."""
        values = {}
        for field in _dataclasses.fields(MLXTrainingConfig):
            if hasattr(args, field.name):
                values[field.name] = _normalize_mlx_training_value(
                    field.name,
                    getattr(args, field.name),
                )
        for alias, target in _MLX_TRAINING_ARGUMENT_ALIASES.items():
            if target not in values and hasattr(args, alias):
                values[alias] = getattr(args, alias)
        for name in _MLX_ALLOWED_EXTRA_ARGUMENTS:
            if hasattr(args, name):
                values[name] = getattr(args, name)
        for name in _MLX_UNSUPPORTED_TASK_ARGUMENTS:
            if hasattr(args, name):
                value = getattr(args, name)
                if value is not None and value is not False:
                    values[name] = value
        if _is_mlx_no_save_strategy(values.get("save_strategy", None)):
            values["save_steps"] = 0
        return values

    def _split_mlx_trainer_kwargs(kwargs):
        """Split SFTTrainer kwargs into MLXTrainer kwargs, config overrides, ignored."""
        trainer_kwargs = {}
        config_kwargs = {}
        ignored_kwargs = {}
        for key, value in kwargs.items():
            if key in _MLX_TRAINER_KWARGS:
                trainer_kwargs[key] = value
                continue
            target = _MLX_TRAINING_ARGUMENT_ALIASES.get(key, key)
            if target in _MLX_TRAINING_CONFIG_FIELDS or key in _MLX_ALLOWED_EXTRA_ARGUMENTS:
                config_kwargs[key] = value
            else:
                ignored_kwargs[key] = value
        return trainer_kwargs, config_kwargs, ignored_kwargs

    def _is_mlx_training_args_like(value):
        """Return whether a positional value looks like training arguments."""
        if isinstance(value, (MLXTrainingConfig, dict, str, os.PathLike)):
            return True
        return any(
            hasattr(value, name)
            for name in (
                "output_dir",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "max_steps",
                "learning_rate",
            )
        )

    def _should_use_trl_positional_schema(args):
        """Return whether positional trainer args match TRL SFTTrainer layout."""
        if len(args) < 2:
            return False
        if _is_mlx_training_args_like(args[1]):
            return True
        # TRL callers often pass explicit defaults:
        # SFTTrainer(model, None, None, train_dataset, ...)
        return len(args) >= 3 and args[1] is None and (args[2] is None or callable(args[2]))

    def _assign_mlx_positional_kwarg(kwargs, name, value):
        """Assign a positional trainer argument while preserving Python errors."""
        if name in kwargs:
            raise TypeError(
                f"UnslothTrainer.__init__() got multiple values for argument " f"{name!r}"
            )
        kwargs[name] = value

    def _normalize_mlx_trainer_init_args(args, kwargs):
        """Map TRL-style or MLX-style positional trainer args into kwargs."""
        kwargs = dict(kwargs)
        if len(args) == 0:
            return kwargs

        use_trl_schema = _should_use_trl_positional_schema(args)
        positional_names = (
            _TRL_SFT_TRAINER_POSITIONAL_KWARGS if use_trl_schema else _MLX_TRAINER_POSITIONAL_KWARGS
        )
        if len(args) > len(positional_names):
            raise TypeError(
                f"UnslothTrainer.__init__() takes at most "
                f"{len(positional_names)} positional arguments on MLX "
                f"({len(args)} given)"
            )
        for name, value in zip(positional_names, args):
            _assign_mlx_positional_kwarg(kwargs, name, value)
        return kwargs

    def _is_meaningful_mlx_extra_value(value):
        """Return whether an inert compatibility arg may alter expected behavior."""
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes)) and len(value) == 0:
            return False
        if isinstance(value, (dict, list, tuple, set, frozenset)) and len(value) == 0:
            return False
        return True

    def _warn_ignored_mlx_training_args(extra_kwargs):
        """Warn when TrainingArguments kwargs are accepted but inert on MLX."""
        names = sorted(
            key
            for key, value in extra_kwargs.items()
            if (key in _MLX_COMPAT_EXTRA_ARGUMENTS and _is_meaningful_mlx_extra_value(value))
        )
        if not names:
            return
        _warnings.warn(
            "Unsloth MLX: accepting but not applying unsupported "
            "TrainingArguments kwargs: "
            f"{', '.join(names)}. These options are not implemented by "
            "MLXTrainer yet.",
            RuntimeWarning,
            stacklevel = 3,
        )

    def _is_meaningful_mlx_trainer_kwarg(key, value):
        """Return whether an unsupported trainer kwarg would change behavior."""
        if key == "optimizers" and value == (None, None):
            return False
        return _is_meaningful_mlx_extra_value(value)

    def _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs):
        """Fail on unsupported trainer kwargs that would change training semantics."""
        names = sorted(
            key
            for key, value in ignored_kwargs.items()
            if _is_meaningful_mlx_trainer_kwarg(key, value)
        )
        if not names:
            return
        raise NotImplementedError(
            "Unsloth MLX: unsupported SFTTrainer kwargs cannot be ignored safely: "
            f"{', '.join(names)}. Remove these kwargs or use a supported MLX "
            "trainer configuration."
        )

    def _raise_unknown_mlx_training_args(extra_kwargs):
        """Fail on unknown TrainingArguments/SFTConfig kwargs on MLX."""
        names = sorted(key for key in extra_kwargs if key not in _MLX_ALLOWED_EXTRA_ARGUMENTS)
        if not names:
            return
        raise NotImplementedError(
            "Unsloth MLX: unsupported TrainingArguments/SFTConfig kwargs: "
            f"{', '.join(names)}. Remove these kwargs or use fields implemented "
            "by MLXTrainingConfig."
        )

    def _positive_mlx_context_length(value):
        """Return a positive integer context length or None."""
        if value is None or isinstance(value, bool):
            return None
        try:
            length = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if length <= 0:
            return None
        return length

    def _positive_mlx_training_number(value):
        """Return a positive training scalar or None."""
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if number <= 0:
            return None
        return number

    def _set_mlx_cuda_style_context_length(args, length):
        """Set both MLX and TRL context length names after CUDA-style resolution."""
        args.max_seq_length = length
        args.max_length = length
        args._unsloth_mlx_max_length_value = length
        return args

    class UnslothTrainingArguments(MLXTrainingConfig):
        """MLX-compatible public training arguments for Unsloth notebooks."""

        def __init__(self, *args, **kwargs):
            """Accept TRL-style keyword args and keep MLX-supported values active."""
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
            elif len(args) == 1 and isinstance(args[0], (str, os.PathLike)):
                kwargs = {"output_dir": os.fspath(args[0]), **kwargs}
            elif args:
                raise TypeError(
                    "UnslothTrainingArguments on MLX accepts keyword arguments, "
                    "a dict, or a single positional output_dir."
                )

            max_length_value = kwargs.get("max_length", None)
            max_seq_length_explicit = (
                _positive_mlx_context_length(kwargs.get("max_seq_length", None)) is not None
            )
            if "max_length" in kwargs and "max_seq_length" not in kwargs:
                kwargs["max_seq_length"] = kwargs["max_length"]
            if "num_train_epochs" in kwargs and "max_steps" not in kwargs:
                kwargs["max_steps"] = -1

            dataset_order_explicit = "dataset_order" in kwargs or bool(
                kwargs.get("preserve_dataset_order", False)
            )
            grad_clip_explicit = any(
                name in kwargs for name in ("max_grad_norm", "max_grad_value", "max_grad_leaf_norm")
            )
            warmup_ratio = kwargs.get("warmup_ratio", None)
            warmup_steps_supplied = "warmup_steps" in kwargs
            warmup_steps_value = kwargs.get("warmup_steps", None)
            warmup_steps_explicit = False
            if warmup_steps_supplied:
                try:
                    warmup_steps_explicit = int(warmup_steps_value) > 0
                except (TypeError, ValueError):
                    warmup_steps_explicit = True
            filtered_kwargs = {}
            extra_kwargs = {}
            for key, value in kwargs.items():
                target = _MLX_TRAINING_ARGUMENT_ALIASES.get(key, key)
                if key != target and target in kwargs:
                    continue
                value = _normalize_mlx_training_value(target, value)
                if target in _MLX_TRAINING_CONFIG_FIELDS:
                    filtered_kwargs[target] = value
                elif key in _MLX_UNSUPPORTED_TASK_ARGUMENTS and not _is_meaningful_mlx_extra_value(
                    value
                ):
                    continue
                else:
                    extra_kwargs[key] = value

            _raise_unknown_mlx_training_args(extra_kwargs)

            if _is_mlx_no_save_strategy(extra_kwargs.get("save_strategy", None)):
                filtered_kwargs["save_steps"] = 0

            if warmup_ratio is not None and not warmup_steps_explicit:
                import math as _math
                max_steps = filtered_kwargs.get(
                    "max_steps",
                    getattr(MLXTrainingConfig, "max_steps", 60),
                )
                try:
                    if int(max_steps) > 0:
                        filtered_kwargs["warmup_steps"] = max(
                            0,
                            _math.ceil(int(max_steps) * float(warmup_ratio)),
                        )
                except (TypeError, ValueError):
                    pass

            super().__init__(**filtered_kwargs)
            self._unsloth_mlx_dataset_order_explicit = dataset_order_explicit
            self._unsloth_mlx_max_seq_length_explicit = max_seq_length_explicit
            self._unsloth_mlx_max_length_value = max_length_value
            if "max_length" in kwargs:
                self.max_length = max_length_value
            self._unsloth_mlx_grad_clip_explicit = grad_clip_explicit
            self._unsloth_mlx_warmup_steps_explicit = warmup_steps_explicit
            self._unsloth_mlx_extra_args = extra_kwargs
            for key, value in extra_kwargs.items():
                setattr(self, key, value)
            _warn_ignored_mlx_training_args(extra_kwargs)

    def _resolve_mlx_cuda_style_max_seq_length(args, model = None):
        """Mirror Unsloth CUDA SFTTrainer's max_length/max_seq_length bridge."""
        model_max_seq_length = _positive_mlx_context_length(
            getattr(model, "max_seq_length", None),
        )
        args_max_seq_length = _positive_mlx_context_length(
            getattr(args, "max_seq_length", None),
        )
        args_max_seq_length_explicit = getattr(
            args,
            "_unsloth_mlx_max_seq_length_explicit",
            None,
        )
        if args_max_seq_length_explicit is None:
            default_max_seq_length = getattr(MLXTrainingConfig, "max_seq_length", 2048)
            args_max_seq_length_explicit = (
                args_max_seq_length is not None and args_max_seq_length != default_max_seq_length
            )
        if not args_max_seq_length_explicit:
            args_max_seq_length = None

        if args_max_seq_length is None and model_max_seq_length is not None:
            args_max_seq_length = model_max_seq_length
        elif (
            args_max_seq_length is not None
            and model_max_seq_length is not None
            and args_max_seq_length > model_max_seq_length
        ):
            print(
                "Unsloth: You set `max_seq_length` as "
                f"{args_max_seq_length} but the maximum the model supports is "
                f"{model_max_seq_length}. We shall reduce it."
            )
            args_max_seq_length = model_max_seq_length

        if args_max_seq_length is not None:
            _set_mlx_cuda_style_context_length(args, args_max_seq_length)
            return args

        model_max_length = model_max_seq_length
        if model_max_length is None:
            model_max_length = _positive_mlx_context_length(
                getattr(model, "max_length", None),
            )
        if model_max_length is not None:
            _set_mlx_cuda_style_context_length(args, model_max_length)
            return args

        args_max_length = _positive_mlx_context_length(
            getattr(args, "max_length", None),
        )
        if args_max_length is None:
            args_max_length = _positive_mlx_context_length(
                getattr(args, "_unsloth_mlx_max_length_value", None),
            )
        if args_max_length is not None:
            _set_mlx_cuda_style_context_length(args, args_max_length)
            if model is not None:
                setattr(model, "max_seq_length", args_max_length)
            return args

        _set_mlx_cuda_style_context_length(args, 1024)
        return args

    def _apply_unsloth_trainer_mlx_defaults(
        args,
        model = None,
        max_seq_length_explicit = False,
    ):
        """Apply notebook-compatible MLX defaults used only by UnslothTrainer."""
        if not getattr(args, "preserve_dataset_order", False) and not getattr(
            args, "_unsloth_mlx_dataset_order_explicit", False
        ):
            default_order = getattr(MLXTrainingConfig, "dataset_order", "default")
            if getattr(args, "dataset_order", default_order) in (None, default_order):
                args.dataset_order = "torch_randperm"

        if isinstance(args, UnslothTrainingArguments) and not getattr(
            args, "_unsloth_mlx_grad_clip_explicit", False
        ):
            max_grad_norm = _positive_mlx_training_number(
                getattr(args, "max_grad_norm", None),
            )
            max_grad_value = _positive_mlx_training_number(
                getattr(args, "max_grad_value", None),
            )
            max_grad_leaf_norm = _positive_mlx_training_number(
                getattr(args, "max_grad_leaf_norm", None),
            )
            if max_grad_norm is None and max_grad_value is None and max_grad_leaf_norm is None:
                args.max_grad_norm = 1.0

        if not max_seq_length_explicit:
            _resolve_mlx_cuda_style_max_seq_length(args, model = model)
        return args

    def _coerce_mlx_training_args(args, overrides = None):
        """Return an MLXTrainingConfig from None, dicts, or trainer args objects."""
        overrides = overrides or {}
        if isinstance(args, MLXTrainingConfig) and not overrides:
            return args
        dataset_order_explicit = None
        max_seq_length_explicit = None
        max_length_value = None
        grad_clip_explicit = None
        if args is None:
            values = {}
        elif isinstance(args, dict):
            values = dict(args)
        elif isinstance(args, (str, os.PathLike)):
            values = {"output_dir": os.fspath(args)}
        else:
            dataset_order_explicit = getattr(
                args,
                "_unsloth_mlx_dataset_order_explicit",
                False,
            )
            max_seq_length_explicit = getattr(
                args,
                "_unsloth_mlx_max_seq_length_explicit",
                None,
            )
            max_length_value = getattr(
                args,
                "_unsloth_mlx_max_length_value",
                getattr(args, "max_length", None),
            )
            grad_clip_explicit = getattr(
                args,
                "_unsloth_mlx_grad_clip_explicit",
                None,
            )
            values = _mlx_training_argument_values(args)
            if hasattr(args, "max_length"):
                values["max_length"] = getattr(args, "max_length")
        values.update(overrides)
        coerced = UnslothTrainingArguments(**values)
        if (
            dataset_order_explicit is not None
            and "dataset_order" not in overrides
            and "preserve_dataset_order" not in overrides
        ):
            coerced._unsloth_mlx_dataset_order_explicit = dataset_order_explicit
        if (
            max_seq_length_explicit is not None
            and "max_seq_length" not in overrides
            and "max_length" not in overrides
        ):
            coerced._unsloth_mlx_max_seq_length_explicit = max_seq_length_explicit
        if max_length_value is not None and "max_length" not in overrides:
            coerced._unsloth_mlx_max_length_value = max_length_value
            coerced.max_length = max_length_value
        if (
            grad_clip_explicit is not None
            and "max_grad_norm" not in overrides
            and "max_grad_value" not in overrides
            and "max_grad_leaf_norm" not in overrides
        ):
            coerced._unsloth_mlx_grad_clip_explicit = grad_clip_explicit
        return coerced

    _MLX_TRAINER_POSITIONAL_KWARGS = (
        "model",
        "tokenizer",
        "train_dataset",
        "eval_dataset",
        "dataset_text_field",
        "max_seq_length",
        "packing",
        "data_collator",
        "args",
        "formatting_func",
        "processor",
    )
    _TRL_SFT_TRAINER_POSITIONAL_KWARGS = (
        "model",
        "args",
        "data_collator",
        "train_dataset",
        "eval_dataset",
        "processing_class",
        "compute_loss_func",
        "compute_metrics",
        "callbacks",
        "optimizers",
        "optimizer_cls_and_kwargs",
        "preprocess_logits_for_metrics",
        "peft_config",
        "formatting_func",
    )
    _MLX_TRAINER_KWARGS = frozenset(_MLX_TRAINER_POSITIONAL_KWARGS)

    class UnslothTrainer(MLXTrainer):
        """Backend-aware public trainer that routes supported SFT notebooks to MLX."""

        def __init__(self, *args, **kwargs):
            """Normalize common TRL SFTTrainer kwargs, then initialize MLXTrainer."""
            kwargs = _normalize_mlx_trainer_init_args(args, kwargs)
            processing_class = kwargs.pop("processing_class", None)
            processor_from_processing_class = False
            if processing_class is not None:
                if kwargs.get("processor", None) is None:
                    kwargs["processor"] = processing_class
                    processor_from_processing_class = True
                if kwargs.get("tokenizer", None) is None:
                    kwargs["tokenizer"] = getattr(
                        processing_class,
                        "tokenizer",
                        processing_class,
                    )
            kwargs.setdefault("tokenizer", None)

            data_collator = kwargs.pop("data_collator", None)
            if data_collator is not None:
                if not isinstance(data_collator, UnslothVisionDataCollator):
                    raise NotImplementedError(
                        "Unsloth MLX: custom data_collator is not supported by "
                        "MLXTrainer. Pass the dataset directly or use the MLX "
                        "trainer's native batching path."
                    )
                collator_processor = getattr(data_collator, "processor", None)
                if collator_processor is not None and (
                    kwargs.get("processor", None) is None or processor_from_processing_class
                ):
                    kwargs["processor"] = collator_processor
                    if kwargs.get("tokenizer", None) is None:
                        kwargs["tokenizer"] = getattr(
                            collator_processor,
                            "tokenizer",
                            collator_processor,
                        )

            trainer_kwargs, config_kwargs, ignored_kwargs = _split_mlx_trainer_kwargs(kwargs)
            _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs)
            trainer_kwargs["args"] = _coerce_mlx_training_args(
                trainer_kwargs.get("args"),
                config_kwargs,
            )
            trainer_kwargs["args"] = _apply_unsloth_trainer_mlx_defaults(
                trainer_kwargs["args"],
                model = trainer_kwargs.get("model"),
                max_seq_length_explicit = (trainer_kwargs.get("max_seq_length") is not None),
            )

            super().__init__(**trainer_kwargs)
            if trainer_kwargs.get("max_seq_length") is not None:
                _set_mlx_cuda_style_context_length(
                    self.args,
                    self.args.max_seq_length,
                )
            self._unsloth_mlx_ignored_trainer_kwargs = ignored_kwargs

    class UnslothVisionDataCollator:
        def __init__(
            self,
            model = None,
            processor = None,
            *args,
            **kwargs,
        ):
            self.model = model
            self.processor = processor
            self.args = args
            self.kwargs = kwargs

        def __call__(self, features):
            raise NotImplementedError(
                "Unsloth: UnslothVisionDataCollator is a compatibility placeholder "
                "on MLX. Pass the dataset to UnslothTrainer; MLXTrainer performs "
                "vision batching internally."
            )

    def get_chat_template(*args, **kwargs):
        """Apply an Unsloth chat template through a lazy MLX-safe import."""
        from .chat_templates import get_chat_template as _get_chat_template
        return _get_chat_template(*args, **kwargs)

    def apply_chat_template(*args, **kwargs):
        """Format a dataset with an Unsloth chat template through a lazy import."""
        from .chat_templates import apply_chat_template as _apply_chat_template
        return _apply_chat_template(*args, **kwargs)

    def standardize_data_formats(*args, **kwargs):
        """Normalize ShareGPT-style datasets through the shared zoo helper."""
        from unsloth_zoo.dataset_utils import standardize_data_formats as _standardize_data_formats
        return _standardize_data_formats(*args, **kwargs)

    def standardize_sharegpt(*args, **kwargs):
        """Alias ShareGPT standardization to the shared dataset-format helper."""
        return standardize_data_formats(*args, **kwargs)

    def train_on_responses_only(*args, **kwargs):
        """Mask non-response tokens through the shared zoo dataset helper."""
        from unsloth_zoo.dataset_utils import train_on_responses_only as _train_on_responses_only
        return _train_on_responses_only(*args, **kwargs)

    def _install_mlx_trl_sft_shim():
        """Expose SFTTrainer/SFTConfig aliases for old notebooks after importing unsloth."""
        _trl = _sys.modules.get("trl")
        if _trl is None:
            try:
                import trl as _trl
            except ImportError:
                _trl = _types.ModuleType("trl")
                _trl.__version__ = "0.0.0+unsloth-mlx"
                _trl.__package__ = "trl"
                _trl.__path__ = []
                _trl.__spec__ = _machinery.ModuleSpec("trl", loader = None, is_package = True)
                _sys.modules["trl"] = _trl

        _trl.SFTTrainer = UnslothTrainer
        _trl.SFTConfig = UnslothTrainingArguments
        _trl.__UNSLOTH_MLX_COMPAT__ = True

    def _install_mlx_unsloth_trainer_shim():
        """Expose a lightweight unsloth.trainer module on MLX."""
        module_name = f"{__name__}.trainer"
        _trainer = _types.ModuleType(module_name)
        _trainer.__package__ = __name__
        _trainer.__spec__ = _machinery.ModuleSpec(module_name, loader = None)
        _trainer.MLXTrainer = MLXTrainer
        _trainer.MLXTrainingConfig = MLXTrainingConfig
        _trainer.UnslothTrainer = UnslothTrainer
        _trainer.UnslothTrainingArguments = UnslothTrainingArguments
        _trainer.UnslothVisionDataCollator = UnslothVisionDataCollator
        _sys.modules[module_name] = _trainer
        globals()["trainer"] = _trainer

    _install_mlx_trl_sft_shim()
    _install_mlx_unsloth_trainer_shim()

else:
    # GPU path: load everything from _gpu_init
    from ._gpu_init import *
    from ._gpu_init import __version__

    def get_gpu_memory_stats():
        """Return CUDA/ROCm/XPU device stats, peak memory, and total memory in GiB."""
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                props = torch.xpu.get_device_properties(0)
                peak = (
                    torch.xpu.max_memory_reserved()
                    if hasattr(torch.xpu, "max_memory_reserved")
                    else torch.xpu.max_memory_allocated()
                )
                total = getattr(props, "total_memory", 0)
                return props, _bytes_to_gb(peak), _bytes_to_gb(total) or 1.0
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                peak = torch.cuda.max_memory_reserved()
                total = getattr(props, "total_memory", 0)
                return props, _bytes_to_gb(peak), _bytes_to_gb(total) or 1.0
        except Exception:
            pass
        stats = _UnslothDeviceStats("Unknown GPU", 0)
        return stats, 0.0, 1.0

    def clear_gpu_memory():
        """Clear cached GPU memory on CUDA, ROCm, or XPU when available."""
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
