"""Tests for the MLX public trainer compatibility surface."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import platform
import sys
import types
import warnings

import pytest

_MLX_SKIP_REASON = "MLX public trainer API is only active on the MLX backend"


def _import_mlx_unsloth():
    """Import unsloth and skip when the current platform is not using MLX."""
    # Skip before importing unsloth so non-MLX hosts missing optional GPU deps
    # (e.g. bitsandbytes) skip cleanly instead of erroring at collection.
    if not (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    ):
        pytest.skip(_MLX_SKIP_REASON)
    unsloth = importlib.import_module("unsloth")
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        pytest.skip(_MLX_SKIP_REASON)
    return unsloth


class _DummyModel:
    """Small model stub that satisfies MLXTrainer constructor probes."""

    def trainable_parameters(self):
        """Return no trainable parameters for constructor-only tests."""
        return {}


class _DummyVLMModel(_DummyModel):
    """Small VLM model stub for MLX vision trainer constructor probes."""

    _is_vlm_model = True


def test_mlx_exports_unsloth_trainer_api():
    """MLX imports should expose the public Unsloth trainer API."""
    unsloth = _import_mlx_unsloth()
    from unsloth import (
        RawTextDataLoader,
        TextPreprocessor,
        UnslothTrainer,
        UnslothTrainingArguments,
        clear_gpu_memory,
        get_gpu_memory_stats,
    )

    assert RawTextDataLoader is unsloth.RawTextDataLoader
    assert TextPreprocessor is unsloth.TextPreprocessor
    assert UnslothTrainer is unsloth.UnslothTrainer
    assert UnslothTrainingArguments is unsloth.UnslothTrainingArguments
    assert get_gpu_memory_stats is unsloth.get_gpu_memory_stats
    assert clear_gpu_memory is unsloth.clear_gpu_memory
    assert issubclass(UnslothTrainer, unsloth.MLXTrainer)
    assert issubclass(UnslothTrainingArguments, unsloth.MLXTrainingConfig)
    assert importlib.util.find_spec("unsloth.memory") is None


def test_non_mlx_exports_public_trainer_api_when_available():
    """GPU/ROCm imports should keep exporting the public Unsloth trainer API."""
    try:
        unsloth = importlib.import_module("unsloth")
    except ImportError as exc:
        # Non-MLX import pulls the optional GPU stack (numpy/torch/unsloth-zoo,
        # bitsandbytes/triton, and _gpu_init can re-raise missing deps as
        # ImportError). Skip when any of it is unavailable rather than failing
        # collection on CPU/ROCm/XPU review hosts.
        pytest.skip(f"non-MLX import dependency unavailable: {exc}")
    if getattr(unsloth, "DEVICE_TYPE", None) == "mlx":
        pytest.skip("non-MLX export smoke test only runs on GPU/ROCm backends")

    assert unsloth.UnslothTrainer is not None
    assert unsloth.UnslothTrainingArguments is not None
    assert callable(unsloth.get_gpu_memory_stats)
    assert callable(unsloth.clear_gpu_memory)
    assert importlib.util.find_spec("unsloth.memory") is None


def test_mlx_training_arguments_accept_trl_style_kwargs():
    """TRL/SFTConfig-style kwargs should normalize without breaking MLX config."""
    unsloth = _import_mlx_unsloth()

    with pytest.warns(RuntimeWarning, match = "bf16.*dataset_kwargs"):
        args = unsloth.UnslothTrainingArguments(
            max_length = 123,
            max_steps = 10,
            warmup_ratio = 0.2,
            remove_unused_columns = False,
            dataset_kwargs = {"skip_prepare_dataset": True},
            bf16 = True,
        )

    assert args.max_seq_length == 123
    assert args.warmup_steps == 2
    assert args.remove_unused_columns is False
    assert args.dataset_kwargs == {"skip_prepare_dataset": True}
    assert args.bf16 is True
    assert args.warmup_ratio == 0.2
    assert args._unsloth_mlx_max_seq_length_explicit is False
    assert args._unsloth_mlx_warmup_steps_explicit is False


def test_mlx_training_arguments_do_not_warn_for_implemented_or_falsey_extras():
    """Implemented and falsey inert compatibility kwargs should stay quiet."""
    unsloth = _import_mlx_unsloth()
    supported_eval_kwargs = {}
    if "eval_strategy" in unsloth._MLX_TRAINING_CONFIG_FIELDS:
        supported_eval_kwargs = {"eval_strategy": "no", "eval_delay": 1}

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        args = unsloth.UnslothTrainingArguments(
            warmup_ratio = 0.2,
            max_steps = 10,
            padding_free = False,
            remove_unused_columns = False,
            assistant_only_loss = False,
            completion_only_loss = False,
            **supported_eval_kwargs,
        )

    assert args.warmup_steps == 2
    assert args.padding_free is False
    assert args.remove_unused_columns is False
    assert args.completion_only_loss is False
    if supported_eval_kwargs:
        assert args.eval_strategy == "no"
        assert args.eval_delay == 1
    assert caught == []


def test_mlx_training_arguments_prefer_canonical_max_seq_length():
    """Canonical MLX config fields should win over compatibility aliases."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments(max_seq_length = 456, max_length = 123)
    dict_args = unsloth.UnslothTrainingArguments(
        {"max_length": 123, "max_seq_length": 456},
    )

    assert args.max_seq_length == 456
    assert args.max_length == 456
    assert args._unsloth_mlx_max_length_value == 456
    assert dict_args.max_seq_length == 456
    assert dict_args.max_length == 456
    assert dict_args._unsloth_mlx_max_length_value == 456
    assert args._unsloth_mlx_max_seq_length_explicit is True
    assert dict_args._unsloth_mlx_max_seq_length_explicit is True


def test_mlx_training_arguments_preserve_explicit_positive_warmup_steps():
    """Explicit warmup_steps should take precedence over warmup_ratio."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments(
        max_steps = 10,
        warmup_steps = 5,
        warmup_ratio = 0.1,
    )

    assert args.warmup_steps == 5
    assert args._unsloth_mlx_warmup_steps_explicit is True


def test_mlx_clear_gpu_memory_uses_metal_fallback(monkeypatch):
    """Older MLX releases expose cache clearing under mx.metal.clear_cache."""
    unsloth = _import_mlx_unsloth()
    import mlx.core as mx

    called = []
    metal = getattr(mx, "metal", None) or type("Metal", (), {})()
    monkeypatch.delattr(mx, "clear_cache", raising = False)
    monkeypatch.setattr(mx, "metal", metal, raising = False)
    monkeypatch.setattr(metal, "clear_cache", lambda: called.append("metal"), raising = False)

    unsloth.clear_gpu_memory()

    assert called == ["metal"]


def test_mlx_training_arguments_preserve_explicit_epoch_training():
    """Epoch-based configs should not inherit the MLX max_steps default."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments(num_train_epochs = 1, warmup_ratio = 0.1)
    default_args = unsloth.UnslothTrainingArguments()

    assert args.num_train_epochs == 1
    assert args.max_steps == -1
    assert args.warmup_ratio == 0.1
    assert args._unsloth_mlx_warmup_steps_explicit is False
    assert default_args.max_steps == unsloth.MLXTrainingConfig.max_steps


def test_mlx_training_arguments_keep_mlx_dataset_order_default():
    """Training arguments alone should not override MLX's native data order."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments(max_steps = 1)
    explicit_default = unsloth.UnslothTrainingArguments(
        max_steps = 1,
        dataset_order = "default",
    )

    assert args.dataset_order == "default"
    assert args._unsloth_mlx_dataset_order_explicit is False
    assert args._unsloth_mlx_max_seq_length_explicit is False
    assert explicit_default.dataset_order == "default"
    assert explicit_default._unsloth_mlx_dataset_order_explicit is True


def test_mlx_training_arguments_warn_on_meaningful_inert_kwargs():
    """Unsupported TrainingArguments knobs should not be silently ignored."""
    unsloth = _import_mlx_unsloth()

    with pytest.warns(RuntimeWarning, match = "push_to_hub.*save_strategy"):
        args = unsloth.UnslothTrainingArguments(
            save_strategy = "steps",
            push_to_hub = True,
            padding_free = False,
        )

    assert args.save_strategy == "steps"
    assert args.push_to_hub is True
    assert args.padding_free is False


def test_mlx_training_arguments_reject_unknown_kwargs():
    """Unknown SFTConfig flags should fail instead of becoming inert attributes."""
    unsloth = _import_mlx_unsloth()

    with pytest.raises(NotImplementedError, match = "assistant_only_loss"):
        unsloth.UnslothTrainingArguments(assistant_only_loss = True)

    completion_args = unsloth.UnslothTrainingArguments(completion_only_loss = True)
    assert completion_args.completion_only_loss is True


def test_mlx_training_arguments_reject_unsupported_object_flags():
    """Object-style SFTConfig flags should not be silently dropped."""
    unsloth = _import_mlx_unsloth()

    class ArgsObject:
        max_steps = 1
        assistant_only_loss = True

    with pytest.raises(NotImplementedError, match = "assistant_only_loss"):
        unsloth._coerce_mlx_training_args(ArgsObject())

    class CompletionArgsObject:
        max_steps = 1
        completion_only_loss = True

    completion_args = unsloth._coerce_mlx_training_args(CompletionArgsObject())
    assert completion_args.completion_only_loss is True


def test_mlx_training_arguments_accept_output_dir_positional():
    """A single positional output_dir should match TrainingArguments behavior."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments("custom-outputs", max_steps = 3)

    assert args.output_dir == "custom-outputs"
    assert args.max_steps == 3


def test_mlx_training_arguments_normalize_optim_and_object_aliases():
    """Common notebook optimizer names and object aliases should normalize."""
    unsloth = _import_mlx_unsloth()

    class Scheduler:
        value = "cosine"

    class ArgsObject:
        optim = "adamw_8bit"
        eval_steps = None
        lr_scheduler_type = Scheduler()
        max_length = 321
        max_steps = 10
        num_train_epochs = 3.0
        save_steps = 500
        save_strategy = "no"
        warmup_ratio = 0.1
        warmup_steps = 0

    with pytest.warns(RuntimeWarning, match = "save_strategy"):
        args = unsloth._coerce_mlx_training_args(ArgsObject())

    assert args.optim == "adamw"
    assert args.eval_steps == 0
    assert args.lr_scheduler_type == "cosine"
    assert args.max_seq_length == 321
    assert args.num_train_epochs == 3
    assert type(args.num_train_epochs) is int
    assert args.save_steps == 0
    assert args.warmup_steps == 1
    assert args._unsloth_mlx_warmup_steps_explicit is False


def test_mlx_training_arguments_accept_supported_notebook_kwargs():
    """Supported SFT notebooks should be able to pass their current args."""
    unsloth = _import_mlx_unsloth()

    with pytest.warns(
        RuntimeWarning,
        match = "bf16.*dataset_kwargs.*gradient_checkpointing_kwargs.*save_strategy",
    ):
        args = unsloth.UnslothTrainingArguments(
            bf16 = True,
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            dataset_text_field = "text",
            embedding_learning_rate = 5e-5,
            fp16 = False,
            gradient_accumulation_steps = 8,
            gradient_checkpointing = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            learning_rate = 1e-4,
            logging_steps = 2,
            lr_scheduler_type = "cosine",
            max_grad_norm = 0.3,
            max_length = 1024,
            max_steps = 10,
            num_train_epochs = 1,
            optim = "paged_adamw_8bit",
            output_dir = "outputs",
            padding_free = False,
            per_device_train_batch_size = 1,
            remove_unused_columns = False,
            report_to = "none",
            save_strategy = "steps",
            seed = 123,
            warmup_ratio = 0.1,
            weight_decay = 0.01,
        )

    assert args.dataset_num_proc == 4
    assert args.dataset_text_field == "text"
    assert args.embedding_learning_rate == 5e-5
    assert args.gradient_accumulation_steps == 8
    assert args.gradient_checkpointing is True
    assert args.learning_rate == 1e-4
    assert args.logging_steps == 2
    assert args.lr_scheduler_type == "cosine"
    assert args.max_grad_norm == 0.3
    assert args.max_seq_length == 1024
    assert args.max_steps == 10
    assert args.num_train_epochs == 1
    assert args.optim == "adamw"
    assert args.output_dir == "outputs"
    assert args.per_device_train_batch_size == 1
    assert args.report_to == "none"
    assert args.seed == 123
    assert args.warmup_ratio == 0.1
    assert args.warmup_steps == 1
    assert args.weight_decay == 0.01
    assert args.dataset_kwargs == {"skip_prepare_dataset": True}
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert args.save_strategy == "steps"


def test_mlx_training_arguments_honor_direct_no_save_strategy():
    """Direct kwargs should map save_strategy=no to save_steps=0."""
    unsloth = _import_mlx_unsloth()

    with pytest.warns(RuntimeWarning, match = "save_strategy"):
        args = unsloth.UnslothTrainingArguments(
            save_strategy = "no",
            save_steps = 500,
        )

    assert args.save_steps == 0


def test_mlx_trainer_accepts_common_sft_kwargs():
    """UnslothTrainer should accept common SFTTrainer kwargs on MLX."""
    unsloth = _import_mlx_unsloth()

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        trainer = unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            args = {"max_steps": 1},
            dataset_num_proc = 8,
            max_length = 456,
            optim = "adamw_bnb_8bit",
            processing_class = object(),
        )

    assert trainer.args.max_steps == 1
    assert trainer.args.dataset_num_proc == 8
    assert trainer.args.max_seq_length == 456
    assert trainer.args.max_grad_norm == 1.0
    assert trainer.args.optim == "adamw"
    assert trainer.args.dataset_order == "torch_randperm"
    assert trainer._unsloth_mlx_ignored_trainer_kwargs == {}
    assert caught == []


def test_mlx_trainer_preserves_explicit_dataset_order():
    """UnslothTrainer should only set torch_randperm when order is implicit."""
    unsloth = _import_mlx_unsloth()

    explicit_default = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(
            max_steps = 1,
            dataset_order = "default",
        ),
    )
    explicit_sequential = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(
            max_steps = 1,
            dataset_order = "sequential",
        ),
    )
    implicit_with_override = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1),
        dataset_num_proc = 4,
    )
    implicit_streaming = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, streaming = True),
    )
    explicit_no_clip = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(
            max_steps = 1,
            max_grad_norm = 0.0,
        ),
    )

    assert explicit_default.args.dataset_order == "default"
    assert explicit_sequential.args.dataset_order == "sequential"
    assert implicit_with_override.args.dataset_order == "torch_randperm"
    assert implicit_streaming.args.dataset_order == "default"
    assert implicit_with_override.args.max_grad_norm == 1.0
    assert explicit_no_clip.args.max_grad_norm == 0.0


def test_mlx_trainer_uses_model_context_length_when_implicit():
    """UnslothTrainer should mirror CUDA's max_length bridge precedence."""
    unsloth = _import_mlx_unsloth()
    model = _DummyModel()
    model.max_seq_length = 321
    max_length_model = _DummyModel()
    max_length_model.max_seq_length = 321
    none_model = _DummyModel()
    none_model.max_seq_length = 321
    explicit_seq_model = _DummyModel()
    explicit_seq_model.max_seq_length = 321
    clamped_seq_model = _DummyModel()
    clamped_seq_model.max_seq_length = 321
    model_max_length = _DummyModel()
    model_max_length.max_length = 777
    metadata_model = _DummyModel()
    metadata_model.config = type("Config", (), {"max_position_embeddings": 888})()
    metadata_tokenizer = type("Tokenizer", (), {"model_max_length": 999})()
    explicit_max_length_no_model = _DummyModel()
    trainer_override_model = _DummyModel()
    trainer_override_model.max_seq_length = 321
    config_override_model = _DummyModel()
    config_override_model.max_seq_length = 432

    implicit = unsloth.UnslothTrainer(
        model = model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1),
    )
    max_length_args = unsloth.UnslothTrainer(
        model = max_length_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, max_length = 123),
    )
    none_args = unsloth.UnslothTrainer(
        model = none_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, max_seq_length = None),
    )
    explicit_seq = unsloth.UnslothTrainer(
        model = explicit_seq_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, max_seq_length = 123),
    )
    clamped_seq = unsloth.UnslothTrainer(
        model = clamped_seq_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, max_seq_length = 654),
    )
    model_max_length_only = unsloth.UnslothTrainer(
        model = model_max_length,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1),
    )
    metadata_ignored = unsloth.UnslothTrainer(
        model = metadata_model,
        tokenizer = metadata_tokenizer,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1),
    )
    explicit_max_length = unsloth.UnslothTrainer(
        model = explicit_max_length_no_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1, max_length = 123),
    )
    trainer_override = unsloth.UnslothTrainer(
        model = trainer_override_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(max_steps = 1),
        max_seq_length = 654,
    )
    config_with_override = unsloth.UnslothTrainer(
        model = config_override_model,
        tokenizer = None,
        train_dataset = [],
        args = unsloth.MLXTrainingConfig(max_steps = 1),
        dataset_num_proc = 4,
    )

    assert implicit.args.max_seq_length == 321
    assert implicit.args.max_length == 321
    assert max_length_args.args.max_seq_length == 321
    assert max_length_args.args.max_length == 321
    assert none_args.args.max_seq_length == 321
    assert none_args.args.max_length == 321
    assert explicit_seq.args.max_seq_length == 123
    assert explicit_seq.args.max_length == 123
    assert clamped_seq.args.max_seq_length == 321
    assert clamped_seq.args.max_length == 321
    assert model_max_length_only.args.max_seq_length == 777
    assert model_max_length_only.args.max_length == 777
    assert metadata_ignored.args.max_seq_length == 1024
    assert metadata_ignored.args.max_length == 1024
    assert explicit_max_length.args.max_seq_length == 123
    assert explicit_max_length.args.max_length == 123
    assert trainer_override.args.max_seq_length == 654
    assert trainer_override.args.max_length == 654
    assert config_with_override.args.max_seq_length == 432
    assert config_with_override.args.max_length == 432


def test_mlx_trainer_processing_class_overrides_explicit_none_tokenizer():
    """TRL passes tokenizer=None while processing_class carries the tokenizer."""
    unsloth = _import_mlx_unsloth()
    tokenizer = object()

    class Processor:
        pass

    processor = Processor()
    processor.tokenizer = tokenizer

    trainer = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        args = {"max_steps": 1},
        processing_class = processor,
    )

    assert trainer.processor is processor
    assert trainer.tokenizer is tokenizer


def test_mlx_trainer_vision_collator_processor_overrides_processing_class():
    """Vision notebooks pass the tokenizer as processing_class and processor in collator."""
    unsloth = _import_mlx_unsloth()
    tokenizer = object()

    class Processor:
        pass

    processor = Processor()
    processor.tokenizer = tokenizer
    collator = unsloth.UnslothVisionDataCollator(_DummyVLMModel(), processor)

    trainer = unsloth.UnslothTrainer(
        model = _DummyVLMModel(),
        tokenizer = None,
        train_dataset = [],
        args = {"max_steps": 1},
        processing_class = tokenizer,
        data_collator = collator,
    )

    assert trainer.processor is processor
    assert trainer.tokenizer is tokenizer


def test_mlx_trainer_preserves_explicit_processor_over_vision_collator():
    """Explicit processor kwargs should stay authoritative over collator metadata."""
    unsloth = _import_mlx_unsloth()
    tokenizer = object()
    explicit_processor = object()

    class Processor:
        pass

    collator_processor = Processor()
    collator_processor.tokenizer = tokenizer
    collator = unsloth.UnslothVisionDataCollator(_DummyVLMModel(), collator_processor)

    trainer = unsloth.UnslothTrainer(
        model = _DummyVLMModel(),
        tokenizer = None,
        train_dataset = [],
        args = {"max_steps": 1},
        processor = explicit_processor,
        processing_class = tokenizer,
        data_collator = collator,
    )

    assert trainer.processor is explicit_processor
    assert trainer.tokenizer is tokenizer


def test_mlx_trainer_forwards_vision_collator_positional_defaults():
    """Vision collator CUDA-style positionals should route into MLX args."""
    unsloth = _import_mlx_unsloth()
    collator = unsloth.UnslothVisionDataCollator(
        _DummyVLMModel(),
        object(),
        2048,
        None,
        "max",
        -100,
        False,
        None,
        None,
        True,
        None,
        False,
    )

    trainer = unsloth.UnslothTrainer(
        model = _DummyVLMModel(),
        tokenizer = None,
        train_dataset = [],
        args = {"max_steps": 1},
        data_collator = collator,
    )

    assert trainer.args.max_seq_length == 2048
    assert trainer.args.image_size == "max"
    assert trainer.args.completion_only_loss is False


def test_mlx_vision_collator_default_does_not_override_explicit_args():
    """Implicit collator defaults should not override explicit trainer args."""
    unsloth = _import_mlx_unsloth()
    collator = unsloth.UnslothVisionDataCollator(_DummyVLMModel(), object())

    trainer = unsloth.UnslothTrainer(
        model = _DummyVLMModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(
            max_steps = 1,
            completion_only_loss = False,
        ),
        data_collator = collator,
    )

    assert trainer.args.completion_only_loss is False


def test_mlx_trainer_rejects_unsafe_unsupported_sft_kwargs():
    """Unsupported kwargs that change training semantics should fail on MLX."""
    unsloth = _import_mlx_unsloth()

    with pytest.raises(NotImplementedError, match = "peft_config"):
        unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            peft_config = object(),
        )


def test_mlx_trainer_rejects_compute_metrics():
    """compute_metrics is still unsupported by MLXTrainer."""
    unsloth = _import_mlx_unsloth()

    with pytest.raises(NotImplementedError, match = "compute_metrics"):
        unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            compute_metrics = lambda *_: None,
        )


def test_mlx_trainer_accepts_callbacks():
    """Callbacks are routed to MLXTrainer when the zoo backend supports them."""
    unsloth = _import_mlx_unsloth()
    from transformers import TrainerCallback

    if not unsloth._mlx_trainer_supports_kwarg("callbacks"):
        pytest.skip("requires unsloth-zoo MLXTrainer callback support")

    class Callback(TrainerCallback):
        pass

    trainer = unsloth.UnslothTrainer(
        model = _DummyModel(),
        tokenizer = None,
        train_dataset = [],
        callbacks = [Callback()],
    )
    assert any(isinstance(cb, Callback) for cb in trainer.callback_handler.callbacks)


def test_mlx_trainer_rejects_callbacks_with_old_zoo(monkeypatch):
    """Older unsloth-zoo builds should fail clearly instead of TypeError."""
    unsloth = _import_mlx_unsloth()
    from transformers import TrainerCallback

    monkeypatch.setattr(
        unsloth,
        "_mlx_trainer_supports_kwarg",
        lambda name: name != "callbacks",
    )

    with pytest.raises(NotImplementedError, match = "callbacks require"):
        unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            callbacks = [TrainerCallback()],
        )


def test_mlx_trainer_rejects_custom_data_collator():
    """MLXTrainer owns batching; custom SFT data collators must not be ignored."""
    unsloth = _import_mlx_unsloth()

    with pytest.raises(NotImplementedError, match = "data_collator"):
        unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            data_collator = object(),
        )


def test_mlx_trainer_rejects_text_completion_only_loss():
    """Text MLX training should not silently ignore completion_only_loss=True."""
    unsloth = _import_mlx_unsloth()

    with pytest.raises(NotImplementedError, match = "completion_only_loss=True"):
        unsloth.UnslothTrainer(
            model = _DummyModel(),
            tokenizer = None,
            train_dataset = [],
            args = unsloth.UnslothTrainingArguments(
                max_steps = 1,
                completion_only_loss = True,
            ),
        )


def test_mlx_trainer_allows_vlm_completion_only_loss():
    """VLM MLX training supports completion_only_loss during collation."""
    unsloth = _import_mlx_unsloth()

    class VLMModel(_DummyModel):
        _is_vlm_model = True

    trainer = unsloth.UnslothTrainer(
        model = VLMModel(),
        tokenizer = None,
        train_dataset = [],
        args = unsloth.UnslothTrainingArguments(
            max_steps = 1,
            completion_only_loss = True,
        ),
    )

    assert trainer.args.completion_only_loss is True


def test_mlx_trainer_accepts_trl_style_positional_args():
    """TRL-style positional `(model, args, ...)` should not be read as tokenizer."""
    unsloth = _import_mlx_unsloth()

    args = unsloth.UnslothTrainingArguments("trl-outputs", max_steps = 2)
    trainer = unsloth.UnslothTrainer(
        _DummyModel(),
        args,
        train_dataset = [],
        tokenizer = None,
    )

    assert trainer.args is args
    assert trainer.args.output_dir == "trl-outputs"
    assert trainer.train_dataset == []


def test_mlx_trainer_accepts_trl_none_placeholder_positionals():
    """Explicit TRL default placeholders should preserve later positional args."""
    unsloth = _import_mlx_unsloth()
    dataset = [{"text": "hello"}]
    processing_class = object()

    trainer = unsloth.UnslothTrainer(
        _DummyModel(),
        None,
        None,
        dataset,
        None,
        processing_class,
    )

    assert getattr(trainer.train_dataset, "_dataset", trainer.train_dataset) is dataset
    assert getattr(trainer, "_mlx_train_dataset_for_batches", dataset) is dataset
    assert trainer.tokenizer is processing_class
    assert trainer.args.max_steps == 60


def test_mlx_trainer_accepts_short_trl_none_placeholder_positionals():
    """Short TRL placeholder calls should keep the fourth arg as train_dataset."""
    unsloth = _import_mlx_unsloth()
    dataset = [{"text": "hello"}]

    trainer = unsloth.UnslothTrainer(
        _DummyModel(),
        None,
        None,
        dataset,
    )

    assert trainer.train_dataset is dataset
    assert trainer.eval_dataset is None
    assert trainer.args.max_steps == 60


def test_mlx_trainer_accepts_short_trl_placeholders_with_keyword_dataset():
    """Short TRL placeholders should not conflict with keyword train_dataset."""
    unsloth = _import_mlx_unsloth()
    dataset = [{"text": "hello"}]

    trainer = unsloth.UnslothTrainer(
        _DummyModel(),
        None,
        None,
        train_dataset = dataset,
    )

    assert trainer.train_dataset is dataset
    assert trainer.eval_dataset is None
    assert trainer.args.max_steps == 60


def test_mlx_trainer_preserves_mlx_positional_schema_with_none_tokenizer():
    """MLX-style `(model, tokenizer, train_dataset, ...)` should still work."""
    unsloth = _import_mlx_unsloth()
    dataset = [{"text": "hello"}]

    trainer = unsloth.UnslothTrainer(
        _DummyModel(),
        None,
        dataset,
        None,
    )

    assert trainer.tokenizer is None
    assert trainer.train_dataset is dataset
    assert trainer.eval_dataset is None


def test_mlx_compatibility_shims_are_installed():
    """Old notebook imports should resolve to the MLX public API after unsloth import."""
    unsloth = _import_mlx_unsloth()

    trl = importlib.import_module("trl")
    trainer_module = importlib.import_module("unsloth.trainer")
    chat_templates = importlib.import_module("unsloth.chat_templates")
    dataset_utils = importlib.import_module("unsloth_zoo.dataset_utils")

    assert importlib.util.find_spec("trl") is not None
    assert importlib.util.find_spec("unsloth.trainer") is not None
    assert unsloth.trainer is trainer_module
    assert unsloth.chat_templates is chat_templates
    assert trl.SFTTrainer is unsloth.UnslothTrainer
    assert issubclass(trl.SFTConfig, unsloth.UnslothTrainingArguments)
    assert trainer_module.UnslothTrainer is unsloth.UnslothTrainer
    assert trainer_module.UnslothVisionDataCollator is unsloth.UnslothVisionDataCollator
    assert chat_templates.train_on_responses_only is dataset_utils.train_on_responses_only
    assert callable(unsloth.train_on_responses_only)


def test_mlx_trl_shim_preserves_existing_trl_module(monkeypatch):
    """The MLX TRL shim should patch, not replace, an already-loaded TRL module."""
    unsloth = _import_mlx_unsloth()
    trl = types.ModuleType("trl")
    trl.__path__ = ["real-trainer-package"]
    trl.existing_marker = object()
    trl.ExistingExport = object()
    trl.__all__ = ["ExistingExport", "BrokenExport"]

    def _raise_for_broken_export(name):
        if name == "BrokenExport":
            raise RuntimeError("optional dependency missing")
        raise AttributeError(name)

    trl.__getattr__ = _raise_for_broken_export
    monkeypatch.setitem(sys.modules, "trl", trl)

    unsloth._install_mlx_trl_sft_shim()

    assert sys.modules["trl"] is trl
    assert trl.__path__ == ["real-trainer-package"]
    assert trl.SFTTrainer is unsloth.UnslothTrainer
    assert issubclass(trl.SFTConfig, unsloth.UnslothTrainingArguments)
    assert trl.__UNSLOTH_MLX_COMPAT__ is True
    assert "ExistingExport" in trl.__all__
    assert "BrokenExport" not in trl.__all__
    assert "SFTTrainer" in trl.__all__
    assert "SFTConfig" in trl.__all__


def test_mlx_trl_shim_installs_real_trl_or_stub(monkeypatch):
    """The MLX TRL shim should prefer real TRL and stub only if unavailable."""
    unsloth = _import_mlx_unsloth()
    monkeypatch.delitem(sys.modules, "trl", raising = False)
    real_trl_available = importlib.util.find_spec("trl") is not None

    unsloth._install_mlx_trl_sft_shim()
    trl = importlib.import_module("trl")

    if real_trl_available:
        assert trl.__version__ != "0.0.0+unsloth-mlx"
    else:
        assert trl.__version__ == "0.0.0+unsloth-mlx"
    assert trl.SFTTrainer is unsloth.UnslothTrainer
    assert issubclass(trl.SFTConfig, unsloth.UnslothTrainingArguments)
    assert trl.__UNSLOTH_MLX_COMPAT__ is True


def test_mlx_trl_star_import_exports_public_shims():
    """Existing `from trl import *` callers should receive MLX SFT shims."""
    unsloth = _import_mlx_unsloth()
    namespace = {}

    exec("from trl import *", namespace)

    assert namespace["SFTTrainer"] is unsloth.UnslothTrainer
    assert issubclass(namespace["SFTConfig"], unsloth.UnslothTrainingArguments)


def test_mlx_rl_trainers_stub_with_clear_error(monkeypatch):
    """GRPO/DPO/ORPO trainers have no MLX path, so the shim retargets the ones trl
    exposes to a clear NotImplementedError instead of a confusing CUDA crash, and
    never invents trainers trl does not have."""
    unsloth = _import_mlx_unsloth()
    trl = types.ModuleType("trl")
    trl.__path__ = ["real-trainer-package"]

    class _RealTrainer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the real torch/CUDA trainer must not run on MLX")

    trl.GRPOTrainer = _RealTrainer
    trl.DPOTrainer = _RealTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    unsloth._install_mlx_trl_sft_shim()

    for name in ("GRPOTrainer", "DPOTrainer"):
        assert getattr(trl, name) is not _RealTrainer
        with pytest.raises(NotImplementedError) as exc:
            getattr(trl, name)(model = None, args = None)
        assert "MLX" in str(exc.value) and name in str(exc.value)
    # trainers trl never exposed must not be invented
    assert not hasattr(trl, "PPOTrainer")
    # idempotent: a second install keeps the same stub
    stub = trl.GRPOTrainer
    unsloth._install_mlx_trl_sft_shim()
    assert trl.GRPOTrainer is stub


def test_mlx_rl_trainer_stub_is_lazy_import_safe(monkeypatch):
    """Stubbing unsupported trl trainers must not resolve them: trl lazy-imports
    pull torch, so on a torch-free MLX install a getattr probe would crash
    `import unsloth`. The shim reads __all__/vars metadata and never triggers
    trl's __getattr__ for a trainer it is about to replace."""
    unsloth = _import_mlx_unsloth()
    trl = types.ModuleType("trl")
    trl.__path__ = ["real-trainer-package"]
    trl.__all__ = ["SFTTrainer", "SFTConfig", "GRPOTrainer", "DPOTrainer"]
    resolved = []

    def _lazy_getattr(name):
        resolved.append(name)
        raise ImportError(f"lazy import of {name} would pull torch")

    trl.__getattr__ = _lazy_getattr
    monkeypatch.setitem(sys.modules, "trl", trl)

    unsloth._install_mlx_trl_sft_shim()  # must not raise despite the lazy trl

    # trainers declared in __all__ are stubbed WITHOUT ever resolving the real one
    assert resolved == []
    for name in ("GRPOTrainer", "DPOTrainer"):
        with pytest.raises(NotImplementedError):
            getattr(trl, name)(model = None)


def test_mlx_stubs_trl_trainers_outside_fixed_set(monkeypatch):
    """Any non-SFT trainer trl exports (e.g. a newer RLOOTrainer not in the fixed
    list) must be stubbed too, so no torch trainer slips through on MLX."""
    unsloth = _import_mlx_unsloth()
    trl = types.ModuleType("trl")
    trl.__path__ = ["real-trainer-package"]
    trl.__all__ = ["SFTTrainer", "SFTConfig", "RLOOTrainer"]
    monkeypatch.setitem(sys.modules, "trl", trl)

    unsloth._install_mlx_trl_sft_shim()

    with pytest.raises(NotImplementedError) as exc:
        trl.RLOOTrainer(model = None)
    assert "MLX" in str(exc.value) and "RLOOTrainer" in str(exc.value)
    # SFT stays usable; only non-SFT trainers are stubbed
    assert trl.SFTTrainer is unsloth.UnslothTrainer


def test_mlx_preserve_dataset_order_is_accepted():
    """preserve_dataset_order=True must be accepted (it is a real MLX config field),
    not rejected as an unknown/unsupported argument."""
    unsloth = _import_mlx_unsloth()
    args = unsloth.UnslothTrainingArguments(
        output_dir = "mlx-out",
        max_steps = 10,
        preserve_dataset_order = True,
    )
    assert getattr(args, "preserve_dataset_order", False) is True


def test_mlx_sftconfig_alias_keeps_trl_epoch_default(monkeypatch):
    """`trl.SFTConfig` (aliased on MLX) keeps TRL's default training length: with
    no explicit max_steps/num_train_epochs it runs TRL's 3 epochs, not the native
    MLX 60-step default. An explicit length is authoritative and untouched."""
    unsloth = _import_mlx_unsloth()
    trl = types.ModuleType("trl")
    trl.__path__ = ["real-trainer-package"]
    monkeypatch.setitem(sys.modules, "trl", trl)

    unsloth._install_mlx_trl_sft_shim()

    # no explicit length -> TRL epoch default (3 epochs, step cap disabled)
    cfg = trl.SFTConfig(output_dir = "mlx-out")
    assert cfg.num_train_epochs == 3
    assert cfg.max_steps == -1
    # explicit step / epoch counts stay exactly as written
    assert trl.SFTConfig(output_dir = "mlx-out", max_steps = 17).max_steps == 17
    assert trl.SFTConfig(output_dir = "mlx-out", num_train_epochs = 2).num_train_epochs == 2


def test_mlx_vision_collator_is_constructor_compatible():
    """Vision notebooks should be able to instantiate the collator placeholder."""
    unsloth = _import_mlx_unsloth()

    collator = unsloth.UnslothVisionDataCollator("model", "processor", flag = True)

    assert collator.model == "model"
    assert collator.processor == "processor"
    assert collator.kwargs == {"completion_only_loss": True, "flag": True}


def test_mlx_train_on_responses_only_returns_shared_mask_function():
    """The MLX public shim should expose the shared response-mask helper."""
    unsloth = _import_mlx_unsloth()

    class Tokenizer:
        def __call__(
            self,
            text,
            add_special_tokens = False,
        ):
            return types.SimpleNamespace(
                input_ids = {
                    "<user>": [1],
                    "<assistant>": [2],
                }[text]
            )

        def convert_tokens_to_ids(self, token):
            return token

    mask_fn = unsloth.train_on_responses_only(
        None,
        instruction_part = "<user>",
        response_part = "<assistant>",
        tokenizer = Tokenizer(),
        return_function = True,
    )
    masked = mask_fn(
        {
            "input_ids": [[1, 10, 2, 20, 21, 1, 11]],
        }
    )

    assert masked == {"labels": [[-100, -100, -100, 20, 21, -100, -100]]}

    last_mask_fn = unsloth.train_on_responses_only(
        None,
        instruction_part = "<user>",
        response_part = "<assistant>",
        tokenizer = Tokenizer(),
        return_function = True,
        last_response_only = True,
    )
    last_masked = last_mask_fn(
        {
            "input_ids": [[1, 10, 2, 20, 1, 11, 2, 30]],
        }
    )

    assert last_masked == {"labels": [[-100, -100, -100, -100, -100, -100, -100, 30]]}


def test_mlx_get_chat_template_uses_light_tokenizer_patch(monkeypatch):
    """MLX notebooks should not import CUDA-heavy tokenizer/save helpers."""
    _import_mlx_unsloth()
    from unsloth.chat_templates import get_chat_template
    import unsloth_zoo.tokenizer_utils as tokenizer_utils

    class Tokenizer:
        is_fast = True
        padding_side = "right"
        eos_token = "<eos>"
        bos_token = "<bos>"
        unk_token = "<unk>"
        pad_token = "<pad>"
        added_tokens_decoder = {}

    def fake_patch_tokenizer(model, tokenizer):
        return model, tokenizer

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("unsloth.models") or name.startswith("unsloth.save"):
            raise AssertionError(f"unexpected CUDA-heavy import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(tokenizer_utils, "patch_tokenizer", fake_patch_tokenizer)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    tokenizer = get_chat_template(
        Tokenizer(),
        chat_template = ("{{ messages }}", "<eos>"),
    )

    assert tokenizer.chat_template == "{{ messages }}"
    assert tokenizer.padding_side == "right"


def test_mlx_gpu_memory_stats_helper_shape():
    """The portable memory helper should return CUDA-shaped values."""
    unsloth = _import_mlx_unsloth()

    stats, used, total = unsloth.get_gpu_memory_stats()

    assert isinstance(stats.name, str)
    assert hasattr(stats, "total_memory")
    assert isinstance(used, float)
    assert total > 0


def test_mlx_torch_cuda_compatibility_shim():
    """Existing CUDA memory and move calls should run on MLX."""
    unsloth = _import_mlx_unsloth()
    torch = pytest.importorskip("torch")
    from transformers.tokenization_utils_base import BatchEncoding

    stats, used, total = unsloth.get_gpu_memory_stats()
    cuda_stats = torch.cuda.get_device_properties(0)

    assert cuda_stats.name == stats.name
    assert cuda_stats.total_memory == stats.total_memory
    assert torch.cuda.get_device_name(0) == stats.name
    assert torch.cuda.max_memory_reserved() == int(used * 1024 * 1024 * 1024)
    assert torch.cuda.max_memory_allocated() == torch.cuda.max_memory_reserved()
    # current (non-max) APIs report live active memory, not the peak high-water
    # mark, and never exceed it.
    assert 0 <= torch.cuda.memory_reserved() <= torch.cuda.max_memory_reserved()
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    assert torch.cuda.get_device_capability() == (0, 0)
    assert total > 0

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    assert total_bytes == int(total * 1024 * 1024 * 1024)
    assert 0 <= free_bytes <= total_bytes

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_device(0)

    tensor = torch.tensor([1, 2, 3])
    assert tensor.to("cuda") is tensor
    assert tensor.cuda() is tensor
    assert tensor.to(device = "cuda") is tensor
    assert tensor.to("cuda", dtype = torch.float32).dtype == torch.float32

    batch = BatchEncoding({"input_ids": tensor})
    assert batch.to("cuda") is batch
    assert batch.to(device = "cuda") is batch
