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

# Relax Metal's context-store timeout before MLX modules can initialize Metal.
# Keep an explicit user value authoritative.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")

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
    """Portable device metadata used by backend memory-reporting helpers."""

    def __init__(
        self,
        name,
        total_memory = 0,
    ):
        """Store a display name and total memory in bytes."""
        self.name = name
        self.total_memory = int(total_memory or 0)
        self.major = 0
        self.minor = 0
        self.multi_processor_count = 0


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
            _is_vlm_model,
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
    import inspect as _inspect
    import importlib.machinery as _machinery
    import sys as _sys
    import types as _types
    import warnings as _warnings

    __version__ = unsloth_zoo.__version__
    DEVICE_TYPE = "mlx"
    _MLX_TRAINER_ACCEPTS_VAR_KWARGS = False
    _MLX_TRAINER_SUPPORTED_KWARGS = frozenset()
    try:
        _MLX_TRAINER_INIT_PARAMETERS = _inspect.signature(MLXTrainer.__init__).parameters
        _MLX_TRAINER_ACCEPTS_VAR_KWARGS = any(
            param.kind is _inspect.Parameter.VAR_KEYWORD
            for param in _MLX_TRAINER_INIT_PARAMETERS.values()
        )
        _MLX_TRAINER_SUPPORTED_KWARGS = frozenset(
            name
            for name, param in _MLX_TRAINER_INIT_PARAMETERS.items()
            if name != "self"
            and param.kind
            in (
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                _inspect.Parameter.KEYWORD_ONLY,
            )
        )
    except (TypeError, ValueError):
        pass

    def _mlx_trainer_supports_kwarg(name):
        """Return whether the installed zoo MLXTrainer accepts a kwarg."""
        return _MLX_TRAINER_ACCEPTS_VAR_KWARGS or name in _MLX_TRAINER_SUPPORTED_KWARGS

    def _is_mlx_cuda_device_target(device):
        """Return True when a torch .to/.cuda target asks for CUDA on MLX."""
        if device is None:
            return False
        return str(device).lower().startswith("cuda")

    def _patch_mlx_batch_encoding_to_cuda():
        """Treat tokenizer_output.to("cuda") as a no-op on the MLX backend."""
        try:
            from transformers.tokenization_utils_base import BatchEncoding
        except Exception:
            return

        original_to = getattr(BatchEncoding, "to", None)
        if original_to is None or getattr(original_to, "_unsloth_mlx_cuda_noop", False):
            return

        def batch_encoding_to(
            self,
            device = None,
            *args,
            **kwargs,
        ):
            target = kwargs.get("device", device)
            if _is_mlx_cuda_device_target(target):
                return self
            # device given by keyword: don't also pass the positional None, or the
            # original raises "multiple values for 'device'" (e.g. .to(device="cpu")).
            if "device" in kwargs:
                return original_to(self, *args, **kwargs)
            return original_to(self, device, *args, **kwargs)

        batch_encoding_to._unsloth_mlx_cuda_noop = True
        batch_encoding_to._unsloth_original_to = original_to
        BatchEncoding.to = batch_encoding_to

    _patch_mlx_batch_encoding_to_cuda()

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
        """Clear MLX's cached GPU memory for compatibility cleanup helpers."""
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is None and hasattr(mx, "metal"):
            clear_cache = getattr(mx.metal, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    def _patch_mlx_torch_cuda_compat_api():
        """Expose CUDA-shaped torch helpers for compatibility callers on MLX."""
        try:
            import torch
        except Exception:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and not getattr(cuda, "_unsloth_mlx_cuda_compat_api", False):

            def get_device_properties(device = None):
                """Return MLX device stats through torch.cuda's compatibility API."""
                return get_gpu_memory_stats()[0]

            def get_device_name(device = None):
                """Return the MLX device name through torch.cuda's compatibility API."""
                return get_device_properties(device).name

            def max_memory_reserved(device = None):
                """Return MLX peak memory in bytes for torch.cuda compatibility API."""
                return int(get_gpu_memory_stats()[1] * 1024 * 1024 * 1024)

            def empty_cache():
                """Clear MLX cache through torch.cuda.empty_cache()."""
                clear_gpu_memory()

            def _mlx_active_memory_bytes():
                """Current active MLX memory in bytes (not the peak high-water mark)."""
                import mlx.core as mx

                get_active = getattr(mx, "get_active_memory", None)
                if get_active is None and hasattr(mx, "metal"):
                    get_active = getattr(mx.metal, "get_active_memory", None)
                return int(get_active()) if callable(get_active) else 0

            def memory_current(device = None):
                """Return CURRENT MLX memory in bytes. torch.cuda.memory_reserved /
                memory_allocated report live usage, not the peak (that is max_*)."""
                return _mlx_active_memory_bytes()

            def mem_get_info(device = None):
                """Return (free, total) bytes for torch.cuda compatibility API.
                Free uses CURRENT active memory, not the peak high-water mark, so
                a capacity check stays accurate after a transient spike."""
                total = int(get_gpu_memory_stats()[2] * 1024 * 1024 * 1024)
                return (max(total - _mlx_active_memory_bytes(), 0), total)

            def reset_peak_memory_stats(device = None):
                """Reset MLX's peak-memory counter so a later max_memory_reserved /
                max_memory_allocated scopes to the run, not earlier model-load peaks."""
                import mlx.core as mx

                reset = getattr(mx, "reset_peak_memory", None)
                if reset is None and hasattr(mx, "metal"):
                    reset = getattr(mx.metal, "reset_peak_memory", None)
                if callable(reset):
                    reset()

            def synchronize(device = None):
                """Wait for queued MLX work when torch.cuda.synchronize() is called."""
                import mlx.core as mx

                sync = getattr(mx, "synchronize", None)
                if callable(sync):
                    sync()

            cuda.get_device_properties = get_device_properties
            cuda.get_device_name = get_device_name
            cuda.max_memory_reserved = max_memory_reserved
            cuda.max_memory_allocated = max_memory_reserved
            cuda.memory_reserved = memory_current
            cuda.memory_allocated = memory_current
            cuda.empty_cache = empty_cache
            cuda.mem_get_info = mem_get_info
            cuda.reset_peak_memory_stats = reset_peak_memory_stats
            cuda.synchronize = synchronize
            cuda.current_device = lambda: 0
            cuda.device_count = lambda: 1
            cuda.set_device = lambda device = None: None
            cuda.get_device_capability = lambda device = None: (0, 0)
            cuda.is_bf16_supported = lambda *args, **kwargs: is_bfloat16_supported()
            cuda._unsloth_mlx_cuda_compat_api = True

        tensor_to = getattr(torch.Tensor, "to", None)
        if tensor_to is not None and not getattr(tensor_to, "_unsloth_mlx_cuda_noop", False):

            def _coerce_mlx_dtype_to_torch(value):
                """Map MLX dtype objects to their torch dtype equivalents."""
                try:
                    import mlx.core as mx
                except Exception:
                    return value
                dtype_map = {
                    mx.bool_: torch.bool,
                    mx.int8: torch.int8,
                    mx.int16: torch.int16,
                    mx.int32: torch.int32,
                    mx.int64: torch.int64,
                    mx.uint8: torch.uint8,
                    mx.float16: torch.float16,
                    mx.float32: torch.float32,
                    mx.bfloat16: torch.bfloat16,
                }
                mapped = dtype_map.get(value, None)
                if mapped is not None:
                    return mapped
                dtype_name = str(value).rsplit(".", 1)[-1]
                name_map = {
                    "bool_": torch.bool,
                    "int8": torch.int8,
                    "int16": torch.int16,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "uint8": torch.uint8,
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }
                return name_map.get(dtype_name, value)

            def mlx_tensor_to(self, *args, **kwargs):
                """Ignore CUDA device targets while preserving dtype conversions."""
                args = list(args)
                kwargs = dict(kwargs)
                removed_cuda_device = False
                if args and _is_mlx_cuda_device_target(args[0]):
                    args.pop(0)
                    removed_cuda_device = True
                if _is_mlx_cuda_device_target(kwargs.get("device", None)):
                    kwargs.pop("device", None)
                    removed_cuda_device = True
                if removed_cuda_device and not args:
                    cuda_only_kwargs = ("non_blocking", "copy", "memory_format")
                    if all(key in cuda_only_kwargs for key in kwargs):
                        return self
                if removed_cuda_device and not args and not kwargs:
                    return self
                if args:
                    args[0] = _coerce_mlx_dtype_to_torch(args[0])
                if "dtype" in kwargs:
                    kwargs["dtype"] = _coerce_mlx_dtype_to_torch(kwargs["dtype"])
                return tensor_to(self, *args, **kwargs)

            mlx_tensor_to._unsloth_mlx_cuda_noop = True
            mlx_tensor_to._unsloth_original_to = tensor_to
            torch.Tensor.to = mlx_tensor_to

        tensor_cuda = getattr(torch.Tensor, "cuda", None)
        if tensor_cuda is not None and not getattr(tensor_cuda, "_unsloth_mlx_cuda_noop", False):

            def mlx_tensor_cuda(self, *args, **kwargs):
                """Treat tensor.cuda() as a no-op on MLX."""
                return self

            mlx_tensor_cuda._unsloth_mlx_cuda_noop = True
            mlx_tensor_cuda._unsloth_original_cuda = tensor_cuda
            torch.Tensor.cuda = mlx_tensor_cuda

    _patch_mlx_torch_cuda_compat_api()

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
            "ddp_find_unused_parameters",
            "disable_tqdm",
            "eval_strategy",
            "evaluation_strategy",
            "fp16",
            "full_determinism",
            "gradient_checkpointing_kwargs",
            "hub_model_id",
            "hub_token",
            "log_level",
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
    _MLX_IMPLEMENTED_EXTRA_ARGUMENTS = frozenset(
        (
            "image_size",
            "preserve_dataset_order",
            "warmup_ratio",
        )
    )
    _MLX_ALLOWED_EXTRA_ARGUMENTS = _MLX_COMPAT_EXTRA_ARGUMENTS | _MLX_IMPLEMENTED_EXTRA_ARGUMENTS
    _MLX_UNSUPPORTED_TASK_ARGUMENTS = frozenset(
        (
            "assistant_only_loss",
            "completion_only_loss",
        )
    )

    def _is_mlx_no_save_strategy(value):
        if hasattr(value, "value"):
            value = value.value
        strategy = str(value or "").strip().lower()
        strategy = strategy.rsplit(".", 1)[-1]
        return strategy in ("no", "none", "false")

    _MLX_ADAMW_OPTIMIZER_ALIASES = frozenset(
        (
            "adamw_8bit",
            "paged_adamw_8bit",
            "adamw_bnb_8bit",
            "paged_adamw_32bit",
            "adamw_torch",
            "adamw_torch_fused",
            "paged_adamw",
            "adamw_32bit",
            "adamw_hf",
            "adamw_anyprecision",
            "adamw_apex_fused",
        )
    )

    def _normalize_mlx_training_value(key, value):
        if key == "eval_steps" and value is None:
            return 0
        if key == "num_train_epochs" and value is not None and not isinstance(value, bool):
            try:
                epochs = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if epochs.is_integer():
                    return int(epochs)
        if key == "lr_scheduler_type" and hasattr(value, "value"):
            return value.value
        if key != "optim":
            return value
        try:
            return _normalize_mlx_optimizer_name(value)
        except ValueError:
            # Older unsloth-zoo lacks CUDA/TRL optimizer aliases; map common
            # adamw_* names so notebook defaults (optim="adamw_8bit") still work.
            opt = str(getattr(value, "value", value) or "adamw").strip().lower()
            opt = opt.rsplit(".", 1)[-1].replace("-", "_")
            if opt in _MLX_ADAMW_OPTIMIZER_ALIASES:
                return "adamw"
            raise

    def _mlx_training_argument_values(args):
        values = {}
        for field in _dataclasses.fields(MLXTrainingConfig):
            if hasattr(args, field.name):
                values[field.name] = _normalize_mlx_training_value(
                    field.name,
                    getattr(args, field.name),
                )
        for alias, target in _MLX_TRAINING_ARGUMENT_ALIASES.items():
            if target not in values and hasattr(args, alias):
                values[target if target in _MLX_ALLOWED_EXTRA_ARGUMENTS else alias] = getattr(
                    args, alias
                )
        for name in _MLX_ALLOWED_EXTRA_ARGUMENTS:
            if hasattr(args, name):
                values[name] = getattr(args, name)
        for name in _MLX_UNSUPPORTED_TASK_ARGUMENTS:
            if hasattr(args, name):
                value = getattr(args, name)
                if (
                    name == "completion_only_loss"
                    and value is not None
                    and name in _MLX_TRAINING_CONFIG_FIELDS
                ):
                    values[name] = value
                elif value is not None and value is not False:
                    values[name] = value
        if _is_mlx_no_save_strategy(values.get("save_strategy", None)):
            values["save_steps"] = 0
        return values

    def _split_mlx_trainer_kwargs(kwargs):
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
        if len(args) < 2:
            return False
        if _is_mlx_training_args_like(args[1]):
            return True
        # TRL callers often pass explicit defaults:
        # SFTTrainer(model, None, None, train_dataset, ...)
        return len(args) >= 3 and args[1] is None and (args[2] is None or callable(args[2]))

    def _assign_mlx_positional_kwarg(kwargs, name, value):
        if name in kwargs:
            raise TypeError(
                f"UnslothTrainer.__init__() got multiple values for argument " f"{name!r}"
            )
        kwargs[name] = value

    def _normalize_mlx_trainer_init_args(args, kwargs):
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
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes)) and len(value) == 0:
            return False
        if isinstance(value, (dict, list, tuple, set, frozenset)) and len(value) == 0:
            return False
        return True

    def _warn_ignored_mlx_training_args(extra_kwargs):
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
        if key == "optimizers" and value == (None, None):
            return False
        return _is_meaningful_mlx_extra_value(value)

    def _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs):
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
        names = sorted(key for key in extra_kwargs if key not in _MLX_ALLOWED_EXTRA_ARGUMENTS)
        if not names:
            return
        raise NotImplementedError(
            "Unsloth MLX: unsupported TrainingArguments/SFTConfig kwargs: "
            f"{', '.join(names)}. Remove these kwargs or use fields implemented "
            "by MLXTrainingConfig."
        )

    def _positive_mlx_context_length(value):
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
        args.max_seq_length = length
        args.max_length = length
        args._unsloth_mlx_max_length_value = length
        return args

    class UnslothTrainingArguments(MLXTrainingConfig):
        """MLX-compatible public training arguments for Unsloth notebooks."""

        def __init__(self, *args, **kwargs):
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
            # Only the canonical max_seq_length marks context length explicit; TRL
            # max_length stays a compatibility alias and defers to the model's
            # context length when one is available.
            max_seq_length_explicit = (
                _positive_mlx_context_length(kwargs.get("max_seq_length", None)) is not None
            )
            if "max_length" in kwargs and "max_seq_length" not in kwargs:
                kwargs["max_seq_length"] = kwargs["max_length"]
            elif (
                "max_length" in kwargs
                and _positive_mlx_context_length(kwargs.get("max_seq_length", None)) is not None
            ):
                max_length_value = kwargs["max_seq_length"]
            if "num_train_epochs" in kwargs and "max_steps" not in kwargs:
                kwargs["max_steps"] = -1

            dataset_order_explicit = "dataset_order" in kwargs or bool(
                kwargs.get("preserve_dataset_order", False)
            )
            append_eos_explicit = "append_eos" in kwargs
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
                if target in _MLX_UNSUPPORTED_TASK_ARGUMENTS:
                    if (
                        target == "completion_only_loss"
                        and value is not None
                        and target in _MLX_TRAINING_CONFIG_FIELDS
                    ):
                        filtered_kwargs[target] = value
                    elif _is_meaningful_mlx_extra_value(value):
                        extra_kwargs[key] = value
                    continue
                if target in _MLX_TRAINING_CONFIG_FIELDS:
                    filtered_kwargs[target] = value
                else:
                    extra_kwargs[target if target in _MLX_ALLOWED_EXTRA_ARGUMENTS else key] = value

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
            self._unsloth_mlx_append_eos_explicit = append_eos_explicit
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
        if (
            not getattr(args, "streaming", False)
            and not getattr(args, "preserve_dataset_order", False)
            and not getattr(args, "_unsloth_mlx_dataset_order_explicit", False)
        ):
            default_order = getattr(MLXTrainingConfig, "dataset_order", "default")
            if getattr(args, "dataset_order", default_order) in (None, default_order):
                args.dataset_order = "torch_randperm"

        if isinstance(args, UnslothTrainingArguments) and not getattr(
            args, "_unsloth_mlx_append_eos_explicit", False
        ):
            args.append_eos = False

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
        overrides = overrides or {}
        if isinstance(args, MLXTrainingConfig) and not overrides:
            return args
        dataset_order_explicit = None
        append_eos_explicit = None
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
            append_eos_explicit = getattr(
                args,
                "_unsloth_mlx_append_eos_explicit",
                None,
            )
            max_seq_length_explicit = getattr(
                args,
                "_unsloth_mlx_max_seq_length_explicit",
                None,
            )
            if max_seq_length_explicit is None:
                args_max_seq_length = _positive_mlx_context_length(
                    getattr(args, "max_seq_length", None),
                )
                default_max_seq_length = getattr(MLXTrainingConfig, "max_seq_length", 2048)
                max_seq_length_explicit = (
                    args_max_seq_length is not None
                    and args_max_seq_length != default_max_seq_length
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
        if append_eos_explicit is not None and "append_eos" not in overrides:
            coerced._unsloth_mlx_append_eos_explicit = append_eos_explicit
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
        "callbacks",
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

    def _filter_supported_mlx_trainer_kwargs(trainer_kwargs):
        """Drop inert/empty kwargs unsupported by this zoo MLXTrainer."""
        unsupported = {
            key: value
            for key, value in trainer_kwargs.items()
            if not _mlx_trainer_supports_kwarg(key)
        }
        names = sorted(
            key for key, value in unsupported.items() if _is_meaningful_mlx_extra_value(value)
        )
        if names:
            subject = ", ".join(names)
            verb = "requires" if len(names) == 1 else "require"
            raise NotImplementedError(
                "Unsloth MLX: "
                f"{subject} {verb} an unsloth-zoo build with "
                "matching MLXTrainer support. Upgrade unsloth-zoo together "
                "with unsloth."
            )
        for key in unsupported:
            trainer_kwargs.pop(key, None)
        return trainer_kwargs

    def _is_mlx_native_text_collator(collator):
        """HF pad/copy collators are redundant on MLX; match by class name."""
        for klass in type(collator).__mro__:
            name = klass.__name__
            if name in (
                "DataCollatorForSeq2Seq",
                "DataCollatorWithPadding",
                "DefaultDataCollator",
            ):
                return True
            if name == "DataCollatorForLanguageModeling":
                # Plain causal padding is fine; MLM masking changes semantics.
                return not bool(getattr(collator, "mlm", False))
        return False

    _MLX_VISION_COLLATOR_FORWARDED_KWARGS = frozenset(
        ("completion_only_loss", "formatting_func", "max_seq_length")
    )
    _MLX_VISION_COLLATOR_IMAGE_KWARGS = frozenset(("image_size", "resize"))
    _MLX_VISION_COLLATOR_POSITIONAL_KWARGS = (
        "max_seq_length",
        "formatting_func",
        "resize",
        "ignore_index",
        "train_on_responses_only",
        "instruction_part",
        "response_part",
        "force_match",
        "num_proc",
        "completion_only_loss",
        "pad_to_multiple_of",
        "resize_dimension",
        "snap_to_patch_size",
        "last_response_only",
    )
    _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS = {
        "ignore_index": -100,
        "train_on_responses_only": False,
        "instruction_part": None,
        "response_part": None,
        "force_match": True,
        "num_proc": None,
        "pad_to_multiple_of": None,
        "resize_dimension": 0,
        "snap_to_patch_size": False,
        "last_response_only": False,
    }

    def _is_default_mlx_vision_collator_value(key, value):
        """Return whether an unsupported collator value is the CUDA default."""
        if key not in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS:
            return False
        default = _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS[key]
        if default is None:
            return value is None
        if isinstance(default, bool):
            return value is default
        return value == default and type(value) is type(default)

    def _has_mlx_training_arg_value(args, key):
        """Return whether training args already carry an explicit config value."""
        if args is None or isinstance(args, (str, os.PathLike)):
            return False
        if isinstance(args, dict):
            return key in args
        return getattr(args, key, None) is not None

    def _raise_unsupported_mlx_vision_collator_kwargs(collator_kwargs):
        """Reject VLM collator kwargs that cannot be ignored safely on MLX."""
        unsupported = sorted(
            key
            for key, value in collator_kwargs.items()
            if (
                key not in _MLX_VISION_COLLATOR_FORWARDED_KWARGS
                and key not in _MLX_VISION_COLLATOR_IMAGE_KWARGS
                and (
                    (
                        key in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS
                        and not _is_default_mlx_vision_collator_value(key, value)
                    )
                    or (
                        key not in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS
                        and _is_meaningful_mlx_extra_value(value)
                    )
                )
            )
        )
        if unsupported:
            raise NotImplementedError(
                "Unsloth MLX: unsupported UnslothVisionDataCollator kwargs "
                f"cannot be ignored safely: {', '.join(unsupported)}."
            )

    class UnslothTrainer(MLXTrainer):
        """Backend-aware public trainer that routes supported SFT notebooks to MLX."""

        def __init__(self, *args, **kwargs):
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
                if isinstance(data_collator, UnslothVisionDataCollator):
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
                    collator_kwargs = getattr(data_collator, "kwargs", None) or {}
                    collator_explicit_kwargs = getattr(
                        data_collator,
                        "_unsloth_mlx_explicit_kwargs",
                        set(collator_kwargs),
                    )
                    collator_image_size = collator_kwargs.get(
                        "image_size",
                        collator_kwargs.get("resize", None),
                    )
                    if isinstance(collator_image_size, list):
                        collator_image_size = tuple(collator_image_size)
                    if (
                        isinstance(collator_image_size, str)
                        and collator_image_size.lower() == "max"
                    ):
                        collator_image_size = "max"
                    if "image_size" not in kwargs and (
                        isinstance(collator_image_size, int)
                        or collator_image_size == "max"
                        or (
                            isinstance(collator_image_size, tuple)
                            and len(collator_image_size) == 2
                            and all(isinstance(x, int) for x in collator_image_size)
                        )
                    ):
                        kwargs["image_size"] = collator_image_size
                    for collator_key in _MLX_VISION_COLLATOR_FORWARDED_KWARGS:
                        collator_defaulted_value = collator_key not in collator_explicit_kwargs
                        if collator_defaulted_value and _has_mlx_training_arg_value(
                            kwargs.get("args"), collator_key
                        ):
                            continue
                        if (
                            collator_key in collator_kwargs
                            and collator_key not in kwargs
                            and collator_kwargs[collator_key] is not None
                        ):
                            kwargs[collator_key] = collator_kwargs[collator_key]
                    _raise_unsupported_mlx_vision_collator_kwargs(collator_kwargs)
                elif _is_mlx_native_text_collator(data_collator):
                    pass  # redundant on MLX; MLXTrainer batches/masks/pads natively
                else:
                    raise NotImplementedError(
                        "Unsloth MLX: custom data_collator is not supported by "
                        "MLXTrainer. Pass the dataset directly or use the MLX "
                        "trainer's native batching path."
                    )

            trainer_kwargs, config_kwargs, ignored_kwargs = _split_mlx_trainer_kwargs(kwargs)
            _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs)
            trainer_kwargs = _filter_supported_mlx_trainer_kwargs(trainer_kwargs)
            trainer_kwargs["args"] = _coerce_mlx_training_args(
                trainer_kwargs.get("args"),
                config_kwargs,
            )
            if getattr(
                trainer_kwargs["args"], "completion_only_loss", None
            ) is True and not _is_vlm_model(trainer_kwargs.get("model")):
                raise NotImplementedError(
                    "Unsloth MLX: completion_only_loss=True is only supported "
                    "for VLM training. For text SFT, call train_on_responses_only "
                    "after constructing the trainer."
                )
            if getattr(
                trainer_kwargs["args"], "train_on_completions", None
            ) is True and not _is_vlm_model(trainer_kwargs.get("model")):
                raise NotImplementedError(
                    "Unsloth MLX: train_on_completions=True is only supported "
                    "for VLM training. For text SFT, call train_on_responses_only "
                    "after constructing the trainer."
                )
            trainer_kwargs["args"] = _apply_unsloth_trainer_mlx_defaults(
                trainer_kwargs["args"],
                model = trainer_kwargs.get("model"),
                max_seq_length_explicit = (trainer_kwargs.get("max_seq_length") is not None),
            )

            super().__init__(**trainer_kwargs)
            self.processing_class = (
                processing_class
                if processing_class is not None
                else self.processor or self.tokenizer
            )
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
            explicit_kwargs = set(kwargs)
            if len(args) > len(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS):
                raise TypeError(
                    "UnslothVisionDataCollator on MLX accepts at most "
                    f"{len(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS)} positional "
                    "options after model and processor."
                )
            for key, value in zip(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS, args):
                if key in kwargs:
                    raise TypeError(
                        f"UnslothVisionDataCollator got multiple values for argument {key!r}"
                    )
                kwargs[key] = value
                explicit_kwargs.add(key)
            if "completion_only_loss" not in kwargs:
                kwargs["completion_only_loss"] = True
            self.model = model
            self.processor = processor
            self.args = ()
            self.kwargs = kwargs
            self._unsloth_mlx_explicit_kwargs = explicit_kwargs

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

    def _safe_mlx_trl_star_exports(_trl):
        """Return importable TRL star exports plus the MLX SFT shims."""
        exports = list(getattr(_trl, "__all__", ()))
        safe_exports = []
        for name in exports:
            try:
                getattr(_trl, name)
            except Exception:
                continue
            safe_exports.append(name)
        for name in ("SFTConfig", "SFTTrainer"):
            if name not in safe_exports:
                safe_exports.append(name)
        return safe_exports

    # trl trainers with no MLX implementation yet. Swap them for stubs that fail
    # with a clear message instead of importing the real torch/CUDA trainer and
    # crashing deep inside it, so an unmigrated GRPO/DPO/ORPO notebook is legible.
    _MLX_UNSUPPORTED_TRL_TRAINERS = (
        "GRPOTrainer",
        "DPOTrainer",
        "ORPOTrainer",
        "KTOTrainer",
        "PPOTrainer",
        "RewardTrainer",
    )

    def _make_mlx_unsupported_trl_trainer(name):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"Unsloth: {name} is not yet supported on the MLX (Apple Silicon) "
                f"backend. Only SFT training runs on MLX today; use a CUDA/ROCm GPU "
                f"for {name}."
            )

        return type(name, (), {"__init__": __init__, "_unsloth_mlx_unsupported": True})

    class _MLXSFTConfig(UnslothTrainingArguments):
        """`trl.SFTConfig` alias that keeps TRL's default training length.

        TRL/HF SFTConfig defaults to num_train_epochs=3 (max_steps=-1); the
        native MLX config defaults to max_steps=60. An unmigrated notebook that
        builds SFTConfig without an explicit length would otherwise silently run
        60 MLX steps under this alias, so seed the TRL epoch default when neither
        max_steps nor num_train_epochs is given (epoch mode is MLX-supported).
        """

        def __init__(self, *args, **kwargs):
            keys = set(kwargs)
            if len(args) == 1 and isinstance(args[0], dict):
                keys |= set(args[0])
            if not ({"max_steps", "num_train_epochs"} & keys):
                kwargs.setdefault("num_train_epochs", 3)
            super().__init__(*args, **kwargs)

    def _install_mlx_trl_sft_shim():
        """Install MLX-backed TRL SFT shims without replacing the TRL module."""
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
        _trl.SFTConfig = _MLXSFTConfig
        # Only retarget trainers the installed trl actually exposes (don't invent
        # attributes); idempotent so re-importing unsloth is a no-op.
        # Decide what to stub from trl's declared exports (__all__) and already
        # materialized attrs only. A getattr probe here would trigger trl's lazy
        # trainer import, pulling torch and breaking `import unsloth` on torch-free
        # MLX just to check existence.
        _trl_exports = set(getattr(_trl, "__all__", ()) or ())
        # Stub every non-SFT trainer trl exposes, not just a fixed list, so newer
        # trainers (RLOOTrainer, ...) also fail with a clear MLX message instead
        # of importing the real torch trainer. Names come from __all__ so we never
        # resolve them (that would trigger trl's lazy import and pull torch).
        _unsupported = set(_MLX_UNSUPPORTED_TRL_TRAINERS) | {
            _n for _n in _trl_exports if _n.endswith("Trainer") and _n != "SFTTrainer"
        }
        for _name in _unsupported:
            _current = vars(_trl).get(_name)
            if getattr(_current, "_unsloth_mlx_unsupported", False):
                continue
            if _name in _trl_exports or _current is not None:
                setattr(_trl, _name, _make_mlx_unsupported_trl_trainer(_name))
        _trl.__all__ = _safe_mlx_trl_star_exports(_trl)
        _trl.__UNSLOTH_MLX_COMPAT__ = True

    def _install_mlx_unsloth_trainer_shim():
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
