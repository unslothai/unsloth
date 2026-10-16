# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 1.0+ refuses padding-free without packing while max_length is set; truncate via max_seq_length."""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("unsloth") is None:
    pytest.skip("trl or unsloth not installed", allow_module_level = True)

# Spoof CUDA before any unsloth import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()

import torch  # noqa: E402


def _eager_compile(
    model = None,
    *args,
    **kwargs,
):
    if callable(model):
        return model
    return lambda fn: fn


@pytest.fixture(scope = "module", autouse = True)
def _cpu_only_torch():
    """Scoped to this module via MonkeyPatch, since import-time patching leaks into later GPU tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "compile", _eager_compile)
        # torch.accelerator only exists from torch 2.6 onwards.
        if hasattr(torch, "accelerator"):
            mp.setattr(torch.accelerator, "is_available", lambda *a, **k: False)
        yield mp


_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_MODEL_MAX_SEQ_LENGTH = 128
_USER_MAX_LENGTH = 64


@pytest.fixture(scope = "module", autouse = True)
def patched_sft(_cpu_only_torch):
    """UNSLOTH_ALLOW_CPU=1 skips both SFT patch halves, so enable them explicitly in _gpu_init's order."""
    global torch  # the `import torch._dynamo` below would otherwise shadow it
    import unsloth  # noqa: F401

    # `import unsloth` reinstalls the real torch.compile; the dynamo kill switch is global too.
    _cpu_only_torch.setattr(torch, "compile", _eager_compile)
    try:
        import torch._dynamo
        _cpu_only_torch.setattr(torch._dynamo.config, "disable", True)
    except Exception:
        pass

    import trl

    if trl.SFTTrainer.__name__ != "UnslothSFTTrainer":
        from unsloth.models.rl import _patch_trl_rl_trainers
        from unsloth.trainer import _patch_trl_trainer

        _patch_trl_rl_trainers("sft_trainer")
        _patch_trl_trainer()


@pytest.fixture(scope = "module")
def trl_has_guard(patched_sft):
    from trl.trainer import sft_trainer
    return "`max_length` is not enforced" in inspect.getsource(sft_trainer)


def _load_plain(model_max_seq_length = _MODEL_MAX_SEQ_LENGTH):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        # No dtype kwarg: `dtype=` fails at the 4.52.4 floor, `torch_dtype=` is deprecated from 4.57.6.
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_MODEL).to(torch.float32)
    except OSError as e:
        pytest.skip(f"could not fetch {_MODEL} (network/hub): {str(e)[:150]}")
    got = next(model.parameters()).dtype
    assert got == torch.float32, (
        f"the cast after load left the model in {got}, not float32. These tests compare "
        f"losses, so a silent dtype change is a silent change of what they measure."
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.max_seq_length = model_max_seq_length
    return model.to("cpu"), tok


def _build(
    tmp_path,
    dataset = None,
    model_max_seq_length = _MODEL_MAX_SEQ_LENGTH,
    eval_dataset = None,
    **config_kwargs,
):
    """Construct the Unsloth-patched SFTTrainer over a long, truncatable dataset."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    assert SFTTrainer.__name__ == "UnslothSFTTrainer", "SFT patch did not apply"
    model, tok = _load_plain(model_max_seq_length)
    ds = (
        dataset(tok)
        if callable(dataset)
        else Dataset.from_list([{"text": "The quick brown fox. " * 200}] * 4)
    )
    cfg = SFTConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataset_text_field = "text",
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        **config_kwargs,
    )
    return SFTTrainer(
        model = model,
        processing_class = tok,
        args = cfg,
        train_dataset = ds,
        eval_dataset = eval_dataset,
    )


def _longest(trainer):
    return max(len(x) for x in trainer.train_dataset["input_ids"])


def test_default_sft_construction_does_not_trip_the_guard(tmp_path, trl_has_guard):
    """The reported break: a plain SFTTrainer(), no padding_free / max_length given."""
    trainer = _build(tmp_path)
    args = trainer.args

    assert args.padding_free is True, "padding-free should still auto-enable"
    assert args.packing is False
    assert args.max_seq_length == _MODEL_MAX_SEQ_LENGTH
    if trl_has_guard:
        assert args.max_length is None
    else:
        assert args.max_length == _MODEL_MAX_SEQ_LENGTH
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH


def test_explicit_max_length_resolves_the_same_on_every_trl(tmp_path, trl_has_guard):
    """An explicit max_length below the model cap must still truncate on every TRL after the handoff."""
    trainer = _build(tmp_path, max_length = _USER_MAX_LENGTH)
    args = trainer.args

    assert args.padding_free is True
    assert args.max_seq_length == _USER_MAX_LENGTH
    if trl_has_guard:
        assert args.max_length is None
    else:
        assert args.max_length == _USER_MAX_LENGTH
    assert _longest(trainer) == _USER_MAX_LENGTH


def _trl_default_max_length():
    import dataclasses
    for field in dataclasses.fields(_pristine_sft_config_cls()):
        if field.name == "max_length":
            return field.default
    return None


# 1025 is the first cap above TRL's 1024 default; 2048 is from_pretrained's default.
@pytest.mark.parametrize("model_cap", [1025, 2048, 8192])
def test_an_untouched_max_length_default_does_not_cap_the_model_context(
    tmp_path, trl_has_guard, model_cap
):
    """An untouched max_length default of 1024 must not be read as a user cap on the model context."""
    from datasets import Dataset

    default_max_length = _trl_default_max_length()
    assert default_max_length is None or default_max_length > 0, (
        "this TRL defaults max_length to something falsy, so the regression "
        "cannot be reproduced here"
    )

    def _long_text(tok):
        return Dataset.from_list([{"text": "The quick brown fox. " * 4000}] * 4)

    trainer = _build(tmp_path, dataset = _long_text, model_max_seq_length = model_cap)
    args = trainer.args

    assert args.max_seq_length == model_cap
    if trl_has_guard:
        assert args.max_length is None
    else:
        assert args.max_length == model_cap
    assert _longest(trainer) == model_cap, (
        f"a {model_cap}-token context was truncated to {_longest(trainer)}; the "
        "untouched config default was read as an explicit request"
    )


@pytest.mark.parametrize("path", ["cli", "clone"])
def test_a_rebuilt_default_config_still_does_not_cap_the_model(tmp_path, trl_has_guard, path):
    """Read omission from the value: HfArgumentParser and dataclasses.replace forge provenance markers."""
    import dataclasses

    from datasets import Dataset
    from trl import SFTConfig

    default_max_length = _trl_default_max_length()
    if default_max_length is None or default_max_length <= 0:
        pytest.skip("this TRL has no positive max_length default, so there is nothing to confuse")

    if path == "cli":
        from transformers import HfArgumentParser
        (cfg,) = HfArgumentParser((SFTConfig,)).parse_args_into_dataclasses(
            ["--output_dir", str(tmp_path)]
        )
    else:
        cfg = dataclasses.replace(SFTConfig(output_dir = str(tmp_path)), learning_rate = 1e-4)
    assert cfg.max_length == default_max_length

    model, tok = _load_plain(8192)
    text = "The quick brown fox. " * 4000
    for key, value in (
        ("per_device_train_batch_size", 2),
        ("max_steps", 1),
        ("report_to", "none"),
        ("save_strategy", "no"),
        ("use_cpu", True),
        ("dataset_text_field", "text"),
        ("fp16", False),
        ("bf16", False),
        ("optim", "adamw_torch"),
    ):
        setattr(cfg, key, value)
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model = model,
        processing_class = tok,
        args = cfg,
        train_dataset = Dataset.from_list([{"text": text}] * 4),
    )
    assert _longest(trainer) == 8192, (
        f"a config rebuilt via {path} carried its resolved default back in and capped an "
        "8192-token context at it"
    )


def test_an_explicit_max_length_equal_to_the_default_keeps_the_model_length(tmp_path):
    """An explicit max_length equal to the default is indistinguishable from untouched, a known limit."""
    from datasets import Dataset
    from trl import SFTConfig

    default_max_length = _trl_default_max_length()
    if default_max_length is None or default_max_length <= 0:
        pytest.skip("this TRL has no positive max_length default")

    model, tok = _load_plain(8192)
    cfg = SFTConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataset_text_field = "text",
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        max_length = default_max_length,
    )
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model = model,
        processing_class = tok,
        args = cfg,
        train_dataset = Dataset.from_list([{"text": "The quick brown fox. " * 4000}] * 4),
    )
    assert _longest(trainer) == 8192


def test_the_cap_reads_an_explicit_max_length_not_a_positive_one():
    """The behavioural test above needs a TRL whose default is positive; this one does not."""
    from unsloth.models import rl

    source = inspect.getsource(rl)
    assert "_unsloth_explicit_max_length" in source
    assert "_unsloth_default_max_length" in source
    assert (
        "min(model.max_seq_length, args.max_length) "
        "if (getattr(args, 'max_length', None) or 0) > 0" not in source
    )


def test_max_seq_length_still_beats_max_length(tmp_path, trl_has_guard):
    """max_seq_length must beat max_length: 4096 wins over 512 in the padding-free branch on TRL 1.0+."""
    from datasets import Dataset

    big, small = 4096, 512

    def _long_text(tok):
        return Dataset.from_list([{"text": "The quick brown fox. " * 4000}] * 4)

    trainer = _build(
        tmp_path,
        dataset = _long_text,
        model_max_seq_length = 8192,
        max_seq_length = big,
        max_length = small,
    )
    args = trainer.args

    assert args.max_seq_length == big
    if trl_has_guard:
        assert args.max_length is None
    else:
        assert args.max_length == big
    assert _longest(trainer) == big


def _tokenized_dataset(tok, with_labels = False):
    from datasets import Dataset

    ids = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(ids) > _MODEL_MAX_SEQ_LENGTH, "row must be overlength to be interesting"
    row = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if with_labels:
        row["labels"] = list(ids)
    return Dataset.from_list([dict(row) for _ in range(4)])


def _collated_width(trainer):
    rows = [trainer.train_dataset[i] for i in range(2)]
    return int(trainer.data_collator(rows)["input_ids"].shape[-1])


@pytest.mark.parametrize(
    "name, dataset",
    [
        ("input_ids", lambda tok: _tokenized_dataset(tok)),
        ("labels", lambda tok: _tokenized_dataset(tok, with_labels = True)),
    ],
)
def test_pretokenized_rows_are_truncated_so_the_cap_is_really_enforced(
    tmp_path, trl_has_guard, name, dataset
):
    """Zoo's prep leaves pre-tokenized rows untouched, so they must be truncated here to enforce the cap."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = dataset)
    args = trainer.args

    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH, f"{name}: rows were not truncated"
    assert args.max_length is None, f"{name}: the cap should be consumed by the truncation"
    assert args.max_seq_length == _MODEL_MAX_SEQ_LENGTH, f"{name}: the cap must be recorded"
    assert args.padding_free is True, f"{name}: padding-free no longer needs dropping"
    # Padding-free concatenates the batch, so the collated width is rows x cap.
    assert _collated_width(trainer) == 2 * _MODEL_MAX_SEQ_LENGTH


def test_a_with_transform_dataset_keeps_its_cap(tmp_path, trl_has_guard):
    """A with_transform split recreates rows on read, so map() cannot cap them and max_length must stay."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    from datasets import Dataset

    def _transformed(tok):
        ids = tok("The quick brown fox. " * 200)["input_ids"]
        assert len(ids) > _MODEL_MAX_SEQ_LENGTH
        base = Dataset.from_list([{"input_ids": list(ids), "attention_mask": [1] * len(ids)}] * 4)
        return base.with_transform(
            lambda batch: {
                "input_ids": [list(ids)] * len(batch["input_ids"]),
                "attention_mask": [[1] * len(ids)] * len(batch["input_ids"]),
            }
        )

    # TRL's collator never truncates; truncation lives only in _prepare_dataset.
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _transformed)


def test_a_raw_eval_split_is_left_for_the_tokenizer(tmp_path, trl_has_guard):
    """A raw conversational eval split must not be sliced as tokens, or its messages get cut off the end."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    from datasets import Dataset

    text = "The quick brown fox. " * 200
    raw = Dataset.from_list([{"text": text}] * 4)
    trainer = _build(tmp_path, dataset = _tokenized_dataset, eval_dataset = {"validation": raw})

    assert trainer.args.max_length is None
    split = trainer.eval_dataset["validation"]
    if "text" in (split.column_names or []):
        assert all(r["text"] == text for r in split), "a raw column was sliced"


def test_a_torch_formatted_dataset_is_still_truncated(tmp_path, trl_has_guard):
    """Batched map() gets tensors under set_format, so list checks or truthiness tests leave rows uncut."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")

    def _formatted(tok):
        ds = _tokenized_dataset(tok)
        ds.set_format("torch")
        return ds

    trainer = _build(tmp_path, dataset = _formatted)
    assert trainer.args.max_length is None
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH, "formatted rows were not truncated"


def test_every_named_eval_split_is_truncated(tmp_path, trl_has_guard):
    """Each split in a named eval dict must be truncated; skipping dicts left evaluation uncapped."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    evals = {"validation": _tokenized_dataset(tok), "test": _tokenized_dataset(tok)}
    trainer = _build(tmp_path, dataset = _tokenized_dataset, eval_dataset = evals)

    assert trainer.args.max_length is None
    for name, split in trainer.eval_dataset.items():
        assert (
            max(len(x) for x in split["input_ids"]) == _MODEL_MAX_SEQ_LENGTH
        ), f"{name}: eval split was not truncated"


def _short_tokenized_dataset(tok):
    """Pre-tokenized and already within the cap: nothing to enforce."""
    from datasets import Dataset

    ids = tok("The quick brown fox.")["input_ids"][: _MODEL_MAX_SEQ_LENGTH // 2]
    return Dataset.from_list(
        [{"input_ids": list(ids), "attention_mask": [1] * len(ids)} for _ in range(4)]
    )


def test_unprepared_datasets_keep_their_length_cap(tmp_path, trl_has_guard):
    """Unprepared rows are not truncated, so padding-free is dropped and max_length kept for the
    collator."""
    trainer = _build(
        tmp_path,
        dataset = _short_tokenized_dataset,
        dataset_kwargs = {"skip_prepare_dataset": True},
    )
    args = trainer.args

    assert (
        args.max_length == _MODEL_MAX_SEQ_LENGTH
    ), "the length cap must not be cleared for an unprepared dataset"
    if trl_has_guard:
        assert (
            args.padding_free is False
        ), "padding-free must be dropped, since it disables truncation"
    assert _longest(trainer) <= _MODEL_MAX_SEQ_LENGTH


def _transformed_dataset(tok):
    """A with_transform dataset: column_names reports the backing text column while rows yield input_ids."""
    from datasets import Dataset

    ids = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(ids) > _MODEL_MAX_SEQ_LENGTH, "row must be overlength to be interesting"
    base = Dataset.from_list([{"text": "The quick brown fox. " * 200}] * 4)
    return base.with_transform(
        lambda batch: {
            "input_ids": [list(ids)] * len(batch["text"]),
            "attention_mask": [[1] * len(ids)] * len(batch["text"]),
        }
    )


def test_transformed_datasets_are_refused_rather_than_run_uncapped(tmp_path, trl_has_guard):
    """On-access tokenizing transforms cannot be truncated, so they are refused, not run uncapped."""
    if not trl_has_guard:
        trainer = _build(tmp_path, dataset = _transformed_dataset)
        assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH
        return
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _transformed_dataset)


def _short_transformed_dataset(tok):
    """The same shape, but the rows fit. Nothing is wrong here."""
    from datasets import Dataset

    ids = tok("The quick brown fox.")["input_ids"][: _MODEL_MAX_SEQ_LENGTH // 2]
    base = Dataset.from_list([{"text": "The quick brown fox."}] * 4)
    return base.with_transform(
        lambda batch: {
            "input_ids": [list(ids)] * len(batch["text"]),
            "attention_mask": [[1] * len(ids)] * len(batch["text"]),
        }
    )


def test_a_transformed_dataset_within_the_cap_is_not_refused(tmp_path, trl_has_guard):
    """The refusal is on an OBSERVED overlength row, not on the dataset shape."""
    trainer = _build(tmp_path, dataset = _short_transformed_dataset)
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH, "the cap must survive"
    if trl_has_guard:
        assert trainer.args.padding_free is False


def test_a_tokenized_eval_split_that_cannot_be_truncated_is_refused(tmp_path, trl_has_guard):
    """The train split truncates cleanly, so the cap was consumed on its word
    alone. A transformed eval split already yields `input_ids`, so prep never
    re-tokenizes it and evaluation ran over the cap."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            dataset = _tokenized_dataset,
            eval_dataset = _transformed_dataset(tok),
        )


def test_a_transformed_eval_split_within_the_cap_is_not_refused(tmp_path, trl_has_guard):
    """A with_transform eval split is not refused, but rebuilds its rows per read, so the cap is
    unproven."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    trainer = _build(
        tmp_path,
        dataset = _tokenized_dataset,
        eval_dataset = _short_transformed_dataset(tok),
    )
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH, "the cap must survive"
    assert trainer.args.padding_free is False


def test_the_truncation_map_resolves_the_serial_worker_sentinel():
    """The config layer writes "run in-process" as `dataset_num_proc = 1`, and
    datasets >= 4.1 builds a Pool(1) for it. Every other map site converts that
    back through the helper; this one forwarded the raw sentinel and could fork a
    tokenizer worker on the host that asked for none."""
    block = _padding_free_codegen_block()
    assert "get_dataset_num_proc" in block, "the map site must resolve through the helper"
    assert "_unsloth_map_kw['num_proc'] = _unsloth_nproc" in block
    assert "_unsloth_map_kw['num_proc'] = getattr(args, 'dataset_num_proc'" not in block


def _pristine_sft_config_cls():
    """Returns TRL's own SFTConfig, not the subclass PatchFastRL swaps into trl.SFTConfig."""
    # Check the marker: the generated subclass is renamed to TRL's name for pickling.
    from trl import SFTConfig

    cls = SFTConfig
    while "_unsloth_patched_rl_config" in cls.__dict__ or cls.__name__.startswith("Unsloth"):
        cls = cls.__bases__[0]
    return cls


def test_pristine_trl_config_without_max_seq_length_still_truncates(tmp_path, trl_has_guard):
    """Pristine SFTConfig lacks max_seq_length, so the cap must be copied, not gated on hasattr."""
    from datasets import Dataset

    config_cls = _pristine_sft_config_cls()
    if hasattr(config_cls(output_dir = str(tmp_path)), "max_seq_length"):
        pytest.skip(
            "this TRL still declares max_seq_length, so the regression it guards cannot "
            "exist here; the cap-copy path is covered by the max_length tests above"
        )

    model, tok = _load_plain()
    text = "The quick brown fox. " * 200
    untruncated = len(tok(text)["input_ids"])
    assert untruncated > _MODEL_MAX_SEQ_LENGTH, "row must be overlength to be interesting"

    cfg = config_cls(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataset_text_field = "text",
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        max_length = _MODEL_MAX_SEQ_LENGTH,
        padding_free = True,
    )
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model = model,
        processing_class = tok,
        args = cfg,
        train_dataset = Dataset.from_list([{"text": text}] * 4),
    )

    if trl_has_guard:
        assert trainer.args.max_length is None
        assert trainer.args.max_seq_length == _MODEL_MAX_SEQ_LENGTH
        assert trainer.args.padding_free is True
    else:
        assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH

    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH, "dataset prep stopped truncating"
    assert _collated_width(trainer) <= 2 * _MODEL_MAX_SEQ_LENGTH, (
        "overlength rows reached the model: padding-free flattens the batch, so an "
        f"untruncated pair collates to {2 * untruncated} tokens"
    )


def _padding_free_codegen_block():
    """The emitted padding-free branch, sliced out of rl.py's generator."""
    from unsloth.models import rl

    source = inspect.getsource(rl)
    start = source.index("if getattr(args, 'padding_free', False) is True")
    return source[start : source.index("extra_args += max_length_check", start)]


def test_generator_copies_the_cap_without_a_hasattr_gate():
    """The max_seq_length copy must be unconditional; hasattr is False on every pristine SFTConfig."""
    block = _padding_free_codegen_block()

    assert "args.max_seq_length = args.max_length" in block
    assert "hasattr(args, 'max_seq_length')" not in block
    # TRL's guard is `args.max_length is not None`, so 0 would still raise.
    assert "args.max_length = None" in block


def test_padding_free_off_keeps_max_length(tmp_path):
    """Nothing is cleared when padding-free is not in play."""
    trainer = _build(tmp_path, padding_free = False)

    assert trainer.args.padding_free is False
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH


def test_packing_keeps_max_length(tmp_path):
    """TRL's guard only fires without packing, so packing runs keep max_length."""
    trainer = _build(tmp_path, packing = True)

    assert trainer.args.packing is True
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH


def test_generator_only_emits_the_none_for_a_trl_that_guards():
    """The codegen edit is gated on the guard text, so old TRLs are untouched."""
    from unsloth.models import rl

    source = inspect.getsource(rl)
    assert '"`max_length` is not enforced" in old_RLTrainer_source' in source
    assert "_unsloth_prep_truncates" in source
    assert "skip_prepare_dataset" in source
    assert "_unsloth_requested_max_length" not in source


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "When `padding_free=True` without packing, `max_length` is not enforced.",
            True,
        ),
        ("Some other max_length problem", False),
        ("padding_free is unsupported here", False),
    ],
)
def test_padding_free_error_matcher(message, expected):
    from unsloth.trainer import _should_skip_auto_padding_free_error
    assert _should_skip_auto_padding_free_error(ValueError(message)) is expected


def _late_overlength_dataset(tok):
    """Row 0 fits, row 3 does not. Only the first row was ever inspected."""
    from datasets import Dataset

    short = tok("hi")["input_ids"]
    long = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(long) > _MODEL_MAX_SEQ_LENGTH
    # Keyed off row text: with_transform receives arbitrary slices, so batch index is meaningless.
    base = Dataset.from_list([{"text": t} for t in ("s", "s", "s", "L")])
    return base.with_transform(
        lambda batch: {
            "input_ids": [list(long if t == "L" else short) for t in batch["text"]],
            "attention_mask": [[1] * len(long if t == "L" else short) for t in batch["text"]],
        }
    )


def test_a_later_overlength_row_is_not_hidden_by_a_short_first_one(tmp_path, trl_has_guard):
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _late_overlength_dataset)


def test_the_cap_check_reads_the_whole_split():
    """A map-style split is read in full; a stream cannot be rewound, so a
    bounded prefix is all there is and the generated code says so."""
    block = _padding_free_codegen_block()
    assert "_UNSLOTH_SCAN_ROWS" in block
    assert "if len(_row['input_ids']) > _unsloth_cap: return False" in block
    assert (
        "return len(_row['input_ids']) <= _unsloth_cap" not in block
    ), "that early return inspected only the first row"


def test_a_raw_train_split_does_not_excuse_a_tokenized_eval_split(tmp_path, trl_has_guard):
    """The truncation decision reads only the train split, leaving a pre-tokenized eval split uncapped."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    trainer = _build(tmp_path, eval_dataset = _tokenized_dataset(tok))

    assert trainer.args.max_length is None, "the cap was consumed"
    assert (
        max(len(x) for x in trainer.eval_dataset["input_ids"]) == _MODEL_MAX_SEQ_LENGTH
    ), "the eval split was left over the cap"


def _tokenized_stream(tok, rows = 4096):
    """A pre-tokenized IterableDataset whose overlength row is past the scan."""
    from datasets import Dataset

    long_ids = tok("The quick brown fox. " * 200)["input_ids"]
    short_ids = long_ids[: _MODEL_MAX_SEQ_LENGTH // 2]

    def _gen():
        for i in range(rows):
            ids = long_ids if i == rows - 1 else short_ids
            yield {"input_ids": list(ids), "attention_mask": [1] * len(ids)}

    return Dataset.from_generator(_gen).to_iterable_dataset()


def test_a_pretokenized_stream_is_truncated_without_num_proc(tmp_path, trl_has_guard):
    """IterableDataset.map takes no num_proc, and its lazy map caps every row, unlike a prefix scan."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    trainer = _build(tmp_path, dataset = _tokenized_stream, dataset_num_proc = 4)

    assert trainer.args.max_length is None
    widths = [len(row["input_ids"]) for row in trainer.train_dataset]
    assert max(widths) == _MODEL_MAX_SEQ_LENGTH, "a row past the scan stayed long"


def test_an_unrewritable_stream_is_refused_not_assumed(tmp_path, trl_has_guard):
    """A stream the truncation cannot rewrite is unverifiable, not verified: the
    prefix scan called the first 1024 fitting rows proof, and nothing downstream
    truncates a pre-tokenized row."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]

    def _opaque_stream(tok):
        stream = _tokenized_stream(tok)
        stream._unsloth_hide_columns = True
        type(stream).column_names = property(lambda self: None)
        return stream

    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _opaque_stream)


def test_keep_end_truncation_keeps_the_end(tmp_path, trl_has_guard):
    """keep_end must keep the suffix, like TRL's [-max_length:]; a prefix trains on the wrong half."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    ids = tok("The quick brown fox. " * 200)["input_ids"]

    def _tail_marked(tok):
        from datasets import Dataset
        row = {"input_ids": list(ids), "attention_mask": [1] * len(ids)}
        return Dataset.from_list([dict(row) for _ in range(4)])

    trainer = _build(tmp_path, dataset = _tail_marked, truncation_mode = "keep_end")

    kept = trainer.train_dataset[0]["input_ids"]
    assert len(kept) == _MODEL_MAX_SEQ_LENGTH
    assert kept == ids[-_MODEL_MAX_SEQ_LENGTH:], "kept the start, not the end"


def test_keep_start_is_still_the_default(tmp_path, trl_has_guard):
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    trainer = _build(tmp_path, dataset = _tokenized_dataset)
    assert trainer.train_dataset[0]["input_ids"] == ids[:_MODEL_MAX_SEQ_LENGTH]


def test_a_packed_split_is_not_truncated_at_all(tmp_path, trl_has_guard):
    """seq_lengths describes documents, not tokens, so a packed split is refused rather than truncated."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]

    def _packed(tok):
        from datasets import Dataset

        ids = tok("The quick brown fox. " * 200)["input_ids"]
        row = {
            "input_ids": list(ids),
            "attention_mask": [1] * len(ids),
            "seq_lengths": [50, 100, len(ids) - 150],
        }
        return Dataset.from_list([dict(row) for _ in range(4)])

    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _packed)


def test_rows_left_fully_masked_are_dropped(tmp_path, trl_has_guard):
    """TRL filters these right after truncating: a row whose prompt alone fills
    the cap has every label at -100 and contributes no loss."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]

    def _tail_labelled(tok):
        from datasets import Dataset

        ids = tok("The quick brown fox. " * 200)["input_ids"]
        labels = [-100] * (len(ids) - 8) + list(ids[-8:])
        rows = [
            {"input_ids": list(ids), "attention_mask": [1] * len(ids), "labels": list(labels)}
            for _ in range(3)
        ]
        short = list(ids[: _MODEL_MAX_SEQ_LENGTH // 2])
        rows.append({"input_ids": short, "attention_mask": [1] * len(short), "labels": list(short)})
        return Dataset.from_list(rows)

    trainer = _build(tmp_path, dataset = _tail_labelled)

    for row in trainer.train_dataset:
        assert any(l != -100 for l in row["labels"]), "a fully masked row survived"
    assert len(trainer.train_dataset) == 1, "only the short row keeps any signal"


def test_a_column_that_is_not_per_token_is_left_alone(tmp_path, trl_has_guard):
    """Row-length matching, not a blanket slice: a per-row list that is not a
    token sequence must survive untouched."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]

    def _with_sidecar(tok):
        from datasets import Dataset

        ids = tok("The quick brown fox. " * 200)["input_ids"]
        row = {"input_ids": list(ids), "attention_mask": [1] * len(ids), "doc_spans": [1, 2, 3]}
        return Dataset.from_list([dict(row) for _ in range(4)])

    trainer = _build(tmp_path, dataset = _with_sidecar)
    if "doc_spans" in trainer.train_dataset.column_names:
        assert trainer.train_dataset[0]["doc_spans"] == [1, 2, 3]


def _scalar_torch_formatted_dataset(tok):
    """A 0-dim tensor has __len__ and raises on it, so a scalar id column must not be read as a sequence."""
    from datasets import Dataset

    ids = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(ids) > _MODEL_MAX_SEQ_LENGTH
    ds = Dataset.from_list(
        [
            {"input_ids": list(ids), "attention_mask": [1] * len(ids), "sample_id": i}
            for i in range(4)
        ]
    )
    return ds.with_format("torch")


def test_a_scalar_column_does_not_defeat_truncation(tmp_path, trl_has_guard):
    """The token columns are truncatable, so the run must not die on the id."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _scalar_torch_formatted_dataset)
    assert _longest(trainer) <= _MODEL_MAX_SEQ_LENGTH


def _mask_row(tok, mask_column, long):
    """Under keep_start a cut completion leaves an all-zero mask, which the collator turns into all -100."""
    prompt = tok("The quick brown fox. " * (200 if long else 1))["input_ids"]
    completion = tok(" answer")["input_ids"]
    ids = list(prompt) + list(completion)
    assert (len(ids) > _MODEL_MAX_SEQ_LENGTH) == bool(long)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        mask_column: [0] * len(prompt) + [1] * len(completion),
    }


def _mask_supervised_dataset(tok):
    """Two rows lose supervision under the cap and two keep it, so the filter itself is tested."""
    from datasets import Dataset
    return Dataset.from_list(
        [_mask_row(tok, "completion_mask", long) for long in (True, True, False, False)]
    )


def _has_supervised_token(row, mask_column):
    """A prepared row is supervised via its mask column, or its labels once TRL 1.7 or zoo drops masks."""
    if mask_column in row:
        return any(m != 0 for m in row[mask_column])
    assert "labels" in row, f"the row carries neither {mask_column} nor labels: {sorted(row)}"
    return any(label != -100 for label in row["labels"])


def test_rows_whose_mask_is_truncated_away_are_dropped(tmp_path, trl_has_guard):
    """Same rule the `labels` filter already applies, for the other two spellings."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    # Set completion_only_loss explicitly: TRL resolves None from the TRAIN sample, which lacks
    # prompt/completion here, so the mask would not be supervision.
    trainer = _build(tmp_path, dataset = _mask_supervised_dataset, completion_only_loss = True)
    assert len(trainer.train_dataset) == 2, "the rows that kept their completion were dropped too"
    for row in trainer.train_dataset:
        assert _has_supervised_token(
            row, "completion_mask"
        ), "a row with no supervised token survived truncation"


def _assistant_mask_dataset(tok):
    """The same shape, supervised by `assistant_masks` instead."""
    from datasets import Dataset
    return Dataset.from_list(
        [_mask_row(tok, "assistant_masks", long) for long in (True, True, False, False)]
    )


def _all_unsupervised_dataset(tok):
    """Every row loses its supervision to the truncation."""
    from datasets import Dataset
    return Dataset.from_list([_mask_row(tok, "completion_mask", True) for _ in range(4)])


def test_assistant_masks_are_filtered_even_with_the_loss_mode_off(tmp_path, trl_has_guard):
    """assistant_masks apply on presence alone, so filter them even when assistant_only_loss is off."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _assistant_mask_dataset, assistant_only_loss = False)
    assert len(trainer.train_dataset) == 2, "the rows that kept their completion were dropped too"
    for row in trainer.train_dataset:
        assert _has_supervised_token(
            row, "assistant_masks"
        ), "a row TRL will label all -100 survived truncation"


def test_a_cap_below_all_supervision_is_a_clear_error(tmp_path, trl_has_guard):
    """An emptied split must raise a clear error, not a bare StopIteration from TRL's __init__."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    with pytest.raises(ValueError, match = "no supervised token"):
        _build(
            tmp_path,
            dataset = _all_unsupervised_dataset,
            completion_only_loss = True,
        )


def test_skip_prepare_dataset_does_not_excuse_an_overlength_row(tmp_path, trl_has_guard):
    """It was the one way to a silently uncapped run: TRL then neither truncates
    nor builds its collator with a truncation length, so the oversized rows
    reach the model with `max_length` set and ignored."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            dataset = _tokenized_dataset,
            dataset_kwargs = {"skip_prepare_dataset": True},
        )


def test_the_codegen_carries_the_third_round_fixes():
    """Version-independent: the behavioural tests above only run on a TRL that
    has the guard, so pin the three changes in the emitted source as well."""
    block = _padding_free_codegen_block()

    # A 0-dim tensor has __len__ and raises on it, so hasattr is the wrong probe.
    assert "hasattr(_first, '__len__')" not in block
    assert "try:    len(_first)" in block

    assert "'assistant_masks' in _unsloth_cols" in block
    assert "getattr(args, 'assistant_only_loss'" not in block

    assert "getattr(args, 'completion_only_loss', None) is not False" not in block
    # Resolved from the TRAIN sample, matching the collator.
    assert "'prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample" in block

    # Masks apply sequentially, so a row survives only where they all agree.
    assert (
        "_unsloth_supervision = (['labels'] if 'labels' in _unsloth_cols else []) + _unsloth_masks"
        in block
    )
    assert (
        "any(all((_x != -100) if _n == 'labels' else _x for _n, _x in zip(_c, _v)) "
        "for _v in zip(*[_e[_n] for _n in _c]))" in block
    )

    assert "if not _unsloth_skip_prepare and not (_unsloth_within_cap" not in block
    assert "if not (_unsloth_within_cap(train_dataset)" in block


def _stub_trainer_class(prepares_late = False):
    """prepares_late selects TRL 1.7.0+ behaviour, where evaluate prepares a split passed straight to it."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    if prepares_late:

        class Stub:
            def _prepare_dataset(self, dataset, *args, **kw):
                return dataset

            def evaluate(
                self,
                eval_dataset = None,
                **kw,
            ):
                seen["ds"] = self._prepare_dataset(eval_dataset)

            def predict(
                self,
                test_dataset = None,
                **kw,
            ):
                seen["ds"] = test_dataset

    else:

        class Stub:
            def evaluate(
                self,
                eval_dataset = None,
                **kw,
            ):
                seen["ds"] = eval_dataset

            def predict(
                self,
                test_dataset = None,
                **kw,
            ):
                seen["ds"] = test_dataset

    _wrap_sft_evaluate_cap(Stub)
    return Stub, seen


class _Args:
    def __init__(self, max_seq_length, max_length):
        self.max_seq_length = max_seq_length
        self.max_length = max_length


def test_evaluate_caps_a_pretokenized_split_handed_over_later():
    """Splits handed to evaluate() later must be capped by the wrapper; nothing else enforces the cap."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    cap = _MODEL_MAX_SEQ_LENGTH
    assert max(len(r) for r in late["input_ids"]) > cap

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)

    got = seen["ds"]
    assert max(len(r) for r in got["input_ids"]) <= cap
    # Per-token sidecars must move with input_ids, or the mask misaligns.
    assert all(len(a) == len(i) for a, i in zip(got["attention_mask"], got["input_ids"]))


def test_a_retained_max_length_does_not_excuse_a_late_split():
    """A retained max_length proves nothing for a split handed to evaluate() later; prep never sees it."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, _MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = late)
    assert max(len(r) for r in seen["ds"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_a_raw_late_split_is_still_left_alone_with_max_length_set():
    """The control for the change above: no `input_ids` means there is nothing
    to cut, and prep will tokenize it with the cap applied there."""
    from datasets import Dataset

    raw = Dataset.from_list([{"text": "The quick brown fox. " * 200}] * 2)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, _MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = raw)
    assert seen["ds"] is raw


def test_evaluate_leaves_a_raw_text_split_alone():
    """No `input_ids` means prep will tokenize it, with the cap applied there."""
    from datasets import Dataset

    raw = Dataset.from_list([{"text": "The quick brown fox. " * 200}] * 2)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(eval_dataset = raw)
    assert seen["ds"] is raw


def test_evaluate_caps_every_split_of_a_dict():
    _, tok = _load_plain()
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(eval_dataset = {"a": _tokenized_dataset(tok), "b": _tokenized_dataset(tok)})
    for split in seen["ds"].values():
        assert max(len(r) for r in split["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_wrapping_evaluate_twice_is_a_no_op():
    """The patch runs again on a second FastLanguageModel call in one process."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    Stub, _ = _stub_trainer_class()
    first = Stub.evaluate
    _wrap_sft_evaluate_cap(Stub)
    assert Stub.evaluate is first


def test_a_none_completion_only_loss_does_not_filter_a_pretokenized_split():
    """None completion_only_loss resolves from dataset shape, so a pre-tokenized split is not filtered."""
    block = _padding_free_codegen_block()
    # Bounded by the next anchor, not a byte count, so added comments cannot shift the window.
    i = block.index("_unsloth_completion_only")
    window = block[i : block.index("args._unsloth_completion_only_loss", i)]
    assert "is None" in window
    assert "'prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample" in window
    assert "args._unsloth_completion_only_loss = _unsloth_completion_only" in block


def test_the_predict_entry_point_is_capped_too():
    """`predict(test_dataset = ...)` comes from the base Trainer and reaches
    the same collator by the same route as `evaluate`."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["eval"] = eval_dataset

        def predict(
            self,
            test_dataset = None,
            **kw,
        ):
            seen["predict"] = test_dataset

    _wrap_sft_evaluate_cap(Stub)
    assert getattr(Stub.predict, "_unsloth_eval_cap_wrapped", False)

    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.predict(test_dataset = late)
    assert max(len(r) for r in seen["predict"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_a_trainer_without_predict_is_not_broken():
    """Not every generated trainer has one; absence must not raise."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    class OnlyEvaluate:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            return eval_dataset

    _wrap_sft_evaluate_cap(OnlyEvaluate)
    assert not hasattr(OnlyEvaluate, "predict")


def test_evaluate_caps_an_iterable_split():
    """On a stream dataset[0] reads 0 as a column name, so the cap failed silently and rows went
    uncapped."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok).to_iterable_dataset()
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(eval_dataset = late)

    got = seen["ds"]
    assert got is not late, "the stream came back untouched"
    rows = list(got)
    assert rows and all(len(r["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH for r in rows)
    assert all(len(r["attention_mask"]) == len(r["input_ids"]) for r in rows)


def test_evaluate_honours_keep_end():
    """keep_end must keep the suffix in evaluate() too, matching TRL's [-max_length:] slice."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    tail = [row[-_MODEL_MAX_SEQ_LENGTH:] for row in late["input_ids"]]

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.args.truncation_mode = "keep_end"
    stub.evaluate(eval_dataset = late)

    assert seen["ds"]["input_ids"] == tail


def test_evaluate_drops_rows_left_with_no_supervision():
    """Truncation can leave a row with no supervised label, and TRL's filter does not run on this path."""
    from datasets import Dataset

    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    cap = _MODEL_MAX_SEQ_LENGTH
    assert len(ids) > cap
    doomed = {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * cap + ids[cap:],
    }
    fine = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}
    late = Dataset.from_list([doomed, fine])

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)

    got = seen["ds"]
    assert len(got) == 1, "the row with no supervised token left should be gone"
    assert any(label != -100 for label in got[0]["labels"])


def test_evaluate_leaves_a_packed_split_alone():
    """seq_lengths describes documents, not tokens, so slicing a packed row misaligns position ids."""
    from datasets import Dataset

    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    packed = Dataset.from_list(
        [{"input_ids": ids, "seq_lengths": [len(ids) // 2, len(ids) - len(ids) // 2]}] * 2
    )

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(eval_dataset = packed)
    assert seen["ds"] is packed


def test_the_codegen_leaves_a_packed_eval_split_to_the_packer():
    """eval_packing is set apart from packing; cutting rows first would truncate the corpus being packed."""
    import inspect

    from unsloth.models import rl

    block = inspect.getsource(rl)
    assert (
        "_unsloth_eval_packing = getattr(args, 'packing', False) if getattr(args, 'eval_packing', None) is None else getattr(args, 'eval_packing')"
        in block
    )
    # Packing needs max_length, so the split is spared rather than max_length cleared.
    assert "if _unsloth_eval_packing or not _unsloth_known_mode:" in block
    assert "_unsloth_capped = False\\n" in block
    assert (
        "_unsloth_scan_eval = None if _unsloth_eval_packing else "
        "(eval_dataset if 'eval_dataset' in locals() else None)" in block
    )


def test_the_codegen_does_not_raise_on_a_split_it_left_to_the_packer():
    """A split left to the packer is overlength on purpose, so the overflow scan must skip it."""
    block = _padding_free_codegen_block()
    assert (
        "_unsloth_scan_eval = None if _unsloth_eval_packing else "
        "(eval_dataset if 'eval_dataset' in locals() else None)" in block
    )
    assert "_unsloth_splits_within_cap(_unsloth_scan_eval)" in block
    assert "_unsloth_within_cap(train_dataset) and" in block
    packing_at = block.index("_unsloth_eval_packing = getattr(args, 'packing'")
    skip_at = block.index("if not _unsloth_skip_prepare:")
    assert packing_at < skip_at, "the fallback reads it even when that block is skipped"


def test_the_codegen_refuses_an_unknown_truncation_mode():
    """Unknown truncation modes are refused, since TRL's SFT path never reads truncation_mode."""
    block = _padding_free_codegen_block()
    assert "_unsloth_known_mode = _unsloth_truncation_mode in ('keep_start', 'keep_end')" in block
    assert "_unsloth_capped = _unsloth_known_mode" in block


def _trl_sft_late_hooks():
    """Read from the source file, since unsloth has already patched the live SFTTrainer class."""
    import ast

    import trl.trainer.sft_trainer as sft_module

    tree = ast.parse(Path(inspect.getsourcefile(sft_module)).read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SFTTrainer")
    methods = [m for m in body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    late = {"evaluate", "predict", "get_eval_dataloader", "get_test_dataloader"}
    prepares = [
        m.name
        for m in methods
        if any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "_prepare_dataset"
            for c in ast.walk(m)
        )
    ]
    return {m.name for m in methods} & late, prepares


def test_eval_packing_on_a_late_split_follows_whether_trl_packs_it():
    """Under eval_packing a late split is packer-owned only from TRL 1.7.0, when evaluate() prepares it."""
    _, tok = _load_plain()

    # Fresh split per case: a skipped cut marks the object as capped, which would skew reuse.
    def _late():
        split = _tokenized_dataset(tok)
        assert max(len(r) for r in split["input_ids"]) > _MODEL_MAX_SEQ_LENGTH
        return split

    def _run(
        prepares_late,
        eval_packing,
        strategy = "wrapped",
    ):
        Stub, seen = _stub_trainer_class(prepares_late = prepares_late)
        stub = Stub()
        stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
        stub.args.eval_packing = eval_packing
        stub.args.packing_strategy = strategy
        stub.evaluate(eval_dataset = _late())
        return max(len(r) for r in seen["ds"]["input_ids"])

    for strategy in ("wrapped", "bfd_split"):
        assert _run(True, True, strategy) > _MODEL_MAX_SEQ_LENGTH, (
            f"{strategy}: the split was cut at the cap before TRL's packer "
            "could redistribute the overflow"
        )
        assert (
            _run(False, True, strategy) <= _MODEL_MAX_SEQ_LENGTH
        ), f"{strategy}: an uncapped split reached the collator"

    for prepares_late in (False, True):
        assert _run(prepares_late, False) <= _MODEL_MAX_SEQ_LENGTH


def _packing_aware_stub():
    """Mirrors TRL: only a split passed to evaluate is prepared; other splits reach get_eval_dataloader."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Base:
        def get_eval_dataloader(self, eval_dataset = None):
            seen["dataloader"] = (
                self.eval_dataset[eval_dataset]
                if isinstance(eval_dataset, str)
                else eval_dataset
                if eval_dataset is not None
                else self.eval_dataset
            )
            return seen["dataloader"]

        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            override = eval_dataset is not None
            return self.get_eval_dataloader(eval_dataset if override else self.eval_dataset)

    class Stub(Base):
        def _prepare_dataset(self, dataset, *a, **kw):
            seen["prepared"] = dataset
            return dataset.map(lambda e: e)

        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            if (
                not self._skip_prepare_dataset
                and eval_dataset is not None
                and not isinstance(eval_dataset, str)
            ):
                eval_dataset = self._prepare_dataset(eval_dataset)
            return super().evaluate(eval_dataset = eval_dataset, **kw)

    _wrap_sft_evaluate_cap(Stub)
    return Stub, seen


def test_a_split_no_packer_reaches_is_still_capped_under_eval_packing():
    """Deferring to the packer must not mark the split capped, or its overlength rows reach the collator."""
    _, tok = _load_plain()
    cap = _MODEL_MAX_SEQ_LENGTH

    def _run(call, **flags):
        Stub, seen = _packing_aware_stub()
        stub = Stub()
        stub.args = _Args(cap, None)
        stub.args.eval_packing = flags.get("eval_packing")
        stub.args.packing = flags.get("packing", False)
        stub._skip_prepare_dataset = flags.get("skip_prepare", False)
        stub.eval_dataset = _tokenized_dataset(tok)
        call(stub)
        return seen

    for flags in (
        {"eval_packing": True},
        {"packing": True},
        {"eval_packing": True, "packing": True},
    ):
        seen = _run(lambda s: s.evaluate(), **flags)
        assert "prepared" not in seen, "TRL does not prepare a stored split"
        assert (
            max(len(r) for r in seen["dataloader"]["input_ids"]) <= cap
        ), f"{flags}: an overlength stored split reached the collator"

    Stub, seen = _packing_aware_stub()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.args.eval_packing = True
    stub.args.packing = False
    stub._skip_prepare_dataset = False
    stub.eval_dataset = {"validation": _tokenized_dataset(tok)}
    stub.evaluate(eval_dataset = "validation")
    assert "prepared" not in seen
    assert (
        max(len(r) for r in seen["dataloader"]["input_ids"]) <= cap
    ), "an overlength named split reached the collator"

    seen = _run(
        lambda s: s.evaluate(eval_dataset = s.eval_dataset),
        eval_packing = True,
        skip_prepare = True,
    )
    assert "prepared" not in seen
    assert (
        max(len(r) for r in seen["dataloader"]["input_ids"]) <= cap
    ), "skip_prepare_dataset + eval_packing let an overlength split through"

    seen = _run(lambda s: s.evaluate(eval_dataset = s.eval_dataset), eval_packing = True)
    assert (
        max(len(r) for r in seen["prepared"]["input_ids"]) > cap
    ), "the split was cut before TRL's packer could redistribute the overflow"


def test_the_installed_trl_is_on_the_side_of_1_7_0_that_its_version_says():
    """The source-level half: which shape the TRL actually installed here has."""
    from packaging.version import Version

    import trl

    late_hooks, prepares = _trl_sft_late_hooks()
    # _prepare_dataset is reached only from __init__ and evaluate; a third caller needs auditing.
    assert set(prepares) <= {"__init__", "evaluate"}, prepares
    assert not late_hooks - {"evaluate"}, late_hooks
    packs_late = "evaluate" in prepares
    assert packs_late == ("evaluate" in late_hooks), (prepares, late_hooks)
    assert packs_late == (Version(trl.__version__) >= Version("1.7.0")), (
        trl.__version__,
        prepares,
    )


def test_predict_still_caps_under_eval_packing_on_every_trl():
    """`predict` is the base Trainer's on every TRL, so nothing packs its split
    and the cap has to apply there whatever `evaluate` does."""
    late_hooks, _ = _trl_sft_late_hooks()
    assert "predict" not in late_hooks, late_hooks
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.args.eval_packing = True
    stub.predict(test_dataset = late)
    got = seen["ds"]
    assert max(len(r) for r in got["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_predict_caps_a_split_under_eval_packing():
    """`predict()` is the base Trainer's, and never runs TRL's prep at all."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["eval"] = eval_dataset

        def predict(self, test_dataset, **kw):
            seen["test"] = test_dataset

    _wrap_sft_evaluate_cap(Stub)

    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.args.eval_packing = True
    stub.predict(late)
    assert max(len(r) for r in seen["test"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_evaluate_intersects_labels_with_the_masks():
    """Labels and masks must be intersected, since disjoint supervision still leaves every label at -100."""
    from datasets import Dataset

    cap = _MODEL_MAX_SEQ_LENGTH
    length = cap + 8
    # Each filter alone says keep; the intersection says the row is empty.
    crossed = {
        "input_ids": list(range(length)),
        "labels": [7] + [-100] * (length - 1),
        "assistant_masks": [0, 1] + [0] * (length - 2),
    }
    agreeing = {
        "input_ids": list(range(length)),
        "labels": [7, 7] + [-100] * (length - 2),
        "assistant_masks": [1, 1] + [0] * (length - 2),
    }
    late = Dataset.from_list([crossed, agreeing])

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)

    got = seen["ds"]
    assert len(got) == 1, "the row whose label and mask never agree should be gone"
    assert got[0]["assistant_masks"][0] == 1


def test_evaluate_uses_the_trainer_resolved_completion_only_mode():
    """Completion-only mode is the trainer's one-time resolution, not re-derived per eval split."""
    from datasets import Dataset

    cap = _MODEL_MAX_SEQ_LENGTH
    length = cap + 8
    doomed = {
        "input_ids": list(range(length)),
        "completion_mask": [0] * cap + [1] * (length - cap),
    }
    late = Dataset.from_list([doomed])

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.args._unsloth_completion_only_loss = True
    stub.evaluate(eval_dataset = late)
    assert len(seen["ds"]) == 0, "the mask truncated to all zeros, so the row has no supervision"

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)
    assert len(seen["ds"]) == 1


def test_evaluate_caps_a_split_that_carries_no_column_metadata():
    """A split with no column_names must not be read as raw text, or a pre-tokenized split stays
    uncapped."""
    _, tok = _load_plain()
    backing = _tokenized_dataset(tok)
    rows = [dict(backing[i]) for i in range(len(backing))]

    class NoMetadata:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __iter__(self):
            return iter(self._rows)

        def __getitem__(self, i):
            return self._rows[i]

        def map(self, fn):
            return NoMetadata([{**r, **fn(r)} for r in self._rows])

        def filter(self, fn):
            return NoMetadata([r for r in self._rows if fn(r)])

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(eval_dataset = NoMetadata(rows))

    got = seen["ds"]
    assert all(len(r["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH for r in got)


def test_evaluate_caps_a_split_with_no_map(monkeypatch):
    """Splits without .map(), like a plain list, must be capped, not passed through by the broad catch."""
    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    cap = _MODEL_MAX_SEQ_LENGTH
    assert len(ids) > cap

    class MapLess:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, i):
            return self._rows[i]

    rows = [{"input_ids": list(ids), "attention_mask": [1] * len(ids)} for _ in range(3)]

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = MapLess(rows))

    got = seen["ds"]
    assert len(got) == 3
    assert all(len(r["input_ids"]) <= cap for r in got)
    assert all(len(r["attention_mask"]) == len(r["input_ids"]) for r in got)
    assert len(got[0]["input_ids"]) <= cap


def test_evaluate_caps_a_with_transform_split():
    """with_transform reports backing columns, not yielded ones, and map() writes a table nobody reads."""
    from datasets import Dataset

    _, tok = _load_plain()
    cap = _MODEL_MAX_SEQ_LENGTH
    text = "The quick brown fox. " * 200
    backing = Dataset.from_list([{"text": text} for _ in range(3)])

    def transform(batch):
        ids = [tok(t)["input_ids"] for t in batch["text"]]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}

    shaped = backing.with_transform(transform)
    assert "input_ids" not in (shaped.column_names or ()), "metadata must still say `text`"

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = shaped)

    got = seen["ds"]
    assert got is not shaped, "the transformed split came back untouched"
    assert all(len(r["input_ids"]) <= cap for r in got)


def _torch_stream(rows):
    """A `torch.utils.data.IterableDataset` with no `map` and no length."""
    import torch.utils.data

    class Stream(torch.utils.data.IterableDataset):
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    return Stream(rows)


def test_a_capped_stream_is_still_iterable_style():
    """The DataLoader picks its kind by isinstance, so a capped stream must subclass IterableDataset."""
    import torch.utils.data

    cap = _MODEL_MAX_SEQ_LENGTH
    length = cap + 16
    rows = [{"input_ids": list(range(length)), "attention_mask": [1] * length} for _ in range(3)]

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = _torch_stream(rows))

    got = seen["ds"]
    assert isinstance(got, torch.utils.data.IterableDataset), "the stream lost its kind"
    loader = torch.utils.data.DataLoader(got, batch_size = None, collate_fn = lambda x: x)
    read = list(loader)
    assert len(read) == 3
    assert all(len(r["input_ids"]) <= cap for r in read)
    assert all(len(r["attention_mask"]) == len(r["input_ids"]) for r in read)


def test_a_short_split_is_still_filtered_for_supervision():
    """Being under the cap does not skip the supervision filter the constructor applies unconditionally."""
    from datasets import Dataset

    cap = _MODEL_MAX_SEQ_LENGTH
    short = cap // 2
    empty = {
        "input_ids": list(range(short)),
        "attention_mask": [1] * short,
        "labels": [-100] * short,
    }
    fine = {
        "input_ids": list(range(short)),
        "attention_mask": [1] * short,
        "labels": list(range(short)),
    }
    late = Dataset.from_list([empty, fine])
    assert max(len(r) for r in late["input_ids"]) <= cap, "nothing here needs cutting"

    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)

    got = seen["ds"]
    assert len(got) == 1, "the row with no supervised token should be gone"
    assert any(label != -100 for label in got[0]["labels"])


def test_a_short_and_fully_supervised_split_comes_back_untouched():
    """The filter that drops nothing must not hand back a copy either."""
    from datasets import Dataset

    cap = _MODEL_MAX_SEQ_LENGTH
    short = cap // 2
    supervised = Dataset.from_list(
        [{"input_ids": list(range(short)), "labels": list(range(short))}] * 2
    )
    bare = Dataset.from_list([{"input_ids": list(range(short)), "attention_mask": [1] * short}] * 2)

    for late in (supervised, bare):
        Stub, seen = _stub_trainer_class()
        stub = Stub()
        stub.args = _Args(cap, None)
        stub.evaluate(eval_dataset = late)
        assert seen["ds"] is late


def _stub_with_stored_eval():
    """A stub whose `evaluate()` falls back to `self.eval_dataset`, as HF does."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["ds"] = self.eval_dataset if eval_dataset is None else eval_dataset
            # Read during the call: a named split is capped for the call and restored after.
            stored = getattr(self, "eval_dataset", None)
            if isinstance(eval_dataset, str) and isinstance(stored, dict):
                seen["resolved"] = stored.get(eval_dataset)

    _wrap_sft_evaluate_cap(Stub)
    return Stub, seen


def test_evaluate_caps_the_split_stored_on_the_trainer():
    """Splits installed after construction escape the constructor's cap, so evaluate() must cap them."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    cap = _MODEL_MAX_SEQ_LENGTH
    assert max(len(r) for r in late["input_ids"]) > cap

    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.eval_dataset = late
    stub.evaluate()

    assert max(len(r) for r in seen["ds"]["input_ids"]) <= cap
    assert stub.eval_dataset is late


def test_the_stored_split_is_restored_even_when_evaluate_raises():
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)

    from unsloth.models.rl import _wrap_sft_evaluate_cap

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            raise RuntimeError("boom")

    _wrap_sft_evaluate_cap(Stub)
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.eval_dataset = late
    with pytest.raises(RuntimeError):
        stub.evaluate()
    assert stub.eval_dataset is late


def test_a_stored_dict_of_splits_is_capped_by_name():
    """HF recurses over a dict of stored splits by NAME, so the capped split has
    to be reachable under the same key rather than passed down as an override."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stored = {"a": _tokenized_dataset(tok), "b": _tokenized_dataset(tok)}
    stub.eval_dataset = stored
    stub.evaluate()

    got = seen["ds"]
    assert sorted(got) == ["a", "b"]
    for split in got.values():
        assert max(len(r) for r in split["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH
    assert stub.eval_dataset is stored


def test_a_split_is_only_scanned_once():
    """The scan materialises the whole input_ids column, so it must run once, not on every eval."""
    _, tok = _load_plain()
    backing = _tokenized_dataset(tok)
    reads = []

    class Counting:
        def __init__(self, inner):
            self._inner = inner

        def __len__(self):
            return len(self._inner)

        def __getitem__(self, key):
            reads.append(key)
            return self._inner[key]

        def __iter__(self):
            return iter(self._inner)

        def __getattr__(self, attribute):
            return getattr(self._inner, attribute)

    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.eval_dataset = Counting(backing)
    stub.evaluate()
    first = seen["ds"]
    after_one = len(reads)
    stub.evaluate()

    assert len(reads) == after_one, "the split was scanned again"
    assert seen["ds"] is first, "the same split gave a different answer"


def _late_cap_helpers():
    """`evaluate`/`predict` wrapped onto a stub, so the late cap can be driven
    without standing up a real trainer."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["ds"] = eval_dataset

        def predict(
            self,
            test_dataset = None,
            **kw,
        ):
            seen["ds"] = test_dataset

    _wrap_sft_evaluate_cap(Stub)
    return Stub, seen


class _EvalArgs:
    def __init__(
        self,
        cap,
        max_length = None,
    ):
        self.max_seq_length = cap
        self.max_length = max_length
        self.eval_packing = None
        self.packing = False
        self.completion_only_loss = None
        self.assistant_only_loss = False
        self.truncation_mode = "keep_start"


def test_the_capped_wrappers_are_picklable():
    """A DataLoader worker under `spawn` pickles the split. Defined inside
    `_wrap_sft_evaluate_cap`, the classes carry a `<locals>` qualname and worker
    startup dies before a single row is evaluated."""
    import pickle

    from unsloth.models import rl

    for name in ("_CappedBase", "_CappedRows"):
        cls = getattr(rl, name)
        assert "<locals>" not in cls.__qualname__, f"{name} is not at module scope"

    rows = [{"input_ids": list(range(8)), "attention_mask": [1] * 8}]
    wrapper = rl._CappedRows(rows, slice(None, 4), (), ("input_ids", "attention_mask"))
    revived = pickle.loads(pickle.dumps(wrapper))
    assert [r["input_ids"] for r in revived] == [[0, 1, 2, 3]]


def test_the_stream_wrapper_is_picklable_too():
    import pickle

    from unsloth.models import rl

    rows = [{"input_ids": list(range(8)), "attention_mask": [1] * 8}]
    stream = rl._capped_stream(rows, slice(None, 4), (), ("input_ids", "attention_mask"))
    assert "<locals>" not in type(stream).__qualname__
    revived = pickle.loads(pickle.dumps(stream))
    assert [r["input_ids"] for r in revived] == [[0, 1, 2, 3]]


def test_probing_a_generator_does_not_eat_its_first_row():
    """`iter(gen) is gen`, so reading a row off it consumes that row for good
    and the split silently evaluates one example short."""
    from unsloth.models.rl import _column_names

    def _gen():
        for i in range(3):
            yield {"input_ids": [i] * 4, "attention_mask": [1] * 4}

    names, source, _probed = _column_names(_gen())
    assert "input_ids" in names
    assert [r["input_ids"][0] for r in source] == [0, 1, 2], "the first row was eaten"


def test_probing_a_rewindable_split_hands_it_straight_back():
    from datasets import Dataset

    from unsloth.models.rl import _column_names

    ds = Dataset.from_list([{"input_ids": [1, 2], "attention_mask": [1, 1]}])
    names, source, _probed = _column_names(ds)
    assert "input_ids" in names and source is ds


def test_predict_keeps_every_row_it_was_given():
    """`predict` returns one prediction per row IN ORDER. Dropping the
    unsupervised ones silently shortens and shifts the output relative to the
    dataset the caller zipped it back onto."""
    from datasets import Dataset

    ids = list(range(8))
    rows = Dataset.from_list(
        [
            {"input_ids": ids, "attention_mask": [1] * 8, "labels": list(ids)},
            {"input_ids": ids, "attention_mask": [1] * 8, "labels": [-100] * 8},
        ]
    )
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)

    stub.predict(test_dataset = rows)
    assert len(seen["ds"]) == 2, "predict dropped a row it must return a prediction for"

    stub.evaluate(eval_dataset = rows)
    assert len(seen["ds"]) == 1


def test_the_memo_does_not_serve_a_stale_cap_for_a_mutable_split():
    """The same list reused across two calls, with rows appended in between."""
    long_row = list(range(_MODEL_MAX_SEQ_LENGTH * 2))

    def _row():
        return {"input_ids": list(long_row), "attention_mask": [1] * len(long_row)}

    rows = [_row()]
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)

    stub.evaluate(eval_dataset = rows)
    assert len(list(seen["ds"])) == 1

    rows.append(_row())
    stub.evaluate(eval_dataset = rows)
    assert len(list(seen["ds"])) == 2, "the memo served a cap taken before the append"


def test_the_memo_still_reuses_an_unchanged_datasets_split():
    """The whole point of the memo: an eval every N steps must not rescan."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)

    stub.evaluate(eval_dataset = late)
    first = seen["ds"]
    stub.evaluate(eval_dataset = late)
    assert seen["ds"] is first


def test_capping_the_train_split_cannot_undo_the_unknown_mode_refusal():
    """Train-split capping must AND into _unsloth_capped, not overwrite the unknown-mode refusal."""
    block = _padding_free_codegen_block()
    assert "train_dataset, _unsloth_split_ok = _unsloth_cap_split(train_dataset)" in block
    assert "_unsloth_capped = _unsloth_capped and _unsloth_split_ok" in block
    assert "train_dataset, _unsloth_capped = _unsloth_cap_split" not in block


def test_the_scan_refuses_a_single_pass_stream_instead_of_draining_it():
    """A stream whose two iter() calls return the same object is single-pass, so the scan refuses it."""
    block = _padding_free_codegen_block()
    assert "_unsloth_rows = iter(_ds)" in block
    assert "if _unsloth_rows is iter(_ds): return False" in block
    assert "for _unsloth_row in _ds:" not in block
    scan_at = block.index("_unsloth_rows = iter(_ds)")
    loop_at = block.index("for _row in _unsloth_rows:")
    assert scan_at < loop_at, "the guard has to run before anything is read"


def test_a_one_shot_stream_survives_the_cap_scan():
    """The behaviour the codegen assertions above stand for, run for real."""
    scanned = {"rows": 0}

    def _stream():
        for i in range(4):
            scanned["rows"] += 1
            yield {"input_ids": [1] * (i + 1)}

    rows = _stream()
    probe = iter(rows)
    assert probe is iter(rows), "a generator is its own iterator"
    assert scanned["rows"] == 0, "the guard must not read a row"
    assert [r["input_ids"] for r in rows] == [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]


def test_the_max_length_seed_rewrite_is_required():
    """Required: without the max_length seed rewrite, a cleared None stops raw datasets being truncated."""
    import re as _re

    import pytest as _pytest

    from unsloth.models import rl_replacements

    with _pytest.raises(RuntimeError, match = "required source edit"):
        rl_replacements._replace_or_fallback(
            "def f():\n    pass\n",
            '    max_seq_length = getattr(args, "max_length", 0)',
            '    max_seq_length = getattr(args, "max_length", 0) or 0',
            fallback_pattern = _re.compile(r"^nothing matches this$", _re.MULTILINE),
            fallback_new = r"x",
            where = "sft_prepare_dataset max_length seed",
            required = True,
        )


def test_an_optional_rewrite_still_only_warns():
    """The control: the worker-count edit must keep degrading quietly."""
    import re as _re

    from unsloth.models import rl_replacements

    source = "def f():\n    pass\n"
    assert (
        rl_replacements._replace_or_fallback(
            source,
            "not present either",
            "x",
            fallback_pattern = _re.compile(r"^nothing matches this$", _re.MULTILINE),
            fallback_new = r"x",
            where = "dataset_num_proc",
        )
        == source
    )


def _shared_iterator_split(rows):
    """Single-pass stream whose __iter__ returns one stored generator; iter(x) is x does not catch it."""
    import torch.utils.data as _tud

    class _Shared(_tud.IterableDataset):
        def __init__(self, rows):
            self._it = iter(list(rows))

        def __iter__(self):
            return self._it

    return _Shared(rows)


def test_the_schema_probe_replays_a_shared_iterator_row():
    """`iterator is dataset` is true for a bare generator and false for a split
    whose `__iter__` returns a stored one, so the probed row was dropped and the
    split started at row 2."""
    from unsloth.models.rl import _column_names

    rows = [{"input_ids": [i]} for i in range(3)]
    names, source, _probed = _column_names(_shared_iterator_split(rows))

    assert "input_ids" in names, "the probe still has to read the schema"
    assert [r["input_ids"] for r in source] == [[0], [1], [2]], "row 0 was eaten"


def test_a_rewindable_stream_is_not_chained():
    """The control. A `datasets.IterableDataset` restarts, so chaining the
    probed row on to a fresh pass would duplicate it."""
    from datasets import Dataset

    from unsloth.models.rl import _column_names

    split = Dataset.from_dict({"input_ids": [[0], [1], [2]]}).to_iterable_dataset()
    names, source, _probed = _column_names(split)

    assert "input_ids" in names
    assert [r["input_ids"] for r in source] == [[0], [1], [2]], "row 0 duplicated"


def test_the_completion_only_probe_does_not_eat_a_training_row():
    """That probe read the first TRAINING example and chained nothing back, so
    a one-shot stream trained from row 2. Columns first, and a row only when
    reading one is free."""
    block = _padding_free_codegen_block()
    assert "_unsloth_probe is train_dataset or _unsloth_probe is iter(train_dataset)" in block
    assert "next(iter(train_dataset), None)" not in block, "the destructive probe is back"
    names_at = block.index("getattr(train_dataset, 'column_names', None)")
    probe_at = block.index("_unsloth_probe = iter(train_dataset)")
    assert names_at < probe_at


def test_a_transformed_split_is_not_memoized_by_fingerprint():
    """_fingerprint covers only the backing table, so a transformed split must not be memoized by it."""
    from datasets import Dataset

    ids = list(range(8))
    other = list(reversed(ids))
    backing = Dataset.from_dict(
        {
            "input_ids": [list(ids), list(other)],
            "attention_mask": [[1] * 8, [1] * 8],
            "labels": [list(ids), list(other)],
        }
    )
    supervised = {"both": True}

    def _mask_second(batch):
        # Keyed on row contents: transforms get arbitrary batches.
        out = dict(batch)
        out["labels"] = [
            [-100] * 8 if (not supervised["both"] and list(row) == other) else list(lab)
            for row, lab in zip(batch["input_ids"], batch["labels"])
        ]
        return out

    split = backing.with_transform(_mask_second)
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)

    stub.evaluate(eval_dataset = split)
    assert len(seen["ds"]) == 2, "both rows are supervised on the first pass"

    supervised["both"] = False
    stub.evaluate(eval_dataset = split)
    assert len(seen["ds"]) == 1, "the memo served a cap taken before the change"


def test_evaluate_caps_the_split_a_string_key_names():
    """Caps the named split, swapping it back afterwards so the caller's dict is never written through."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    original = _tokenized_dataset(tok)
    assert max(len(r) for r in original["input_ids"]) > _MODEL_MAX_SEQ_LENGTH
    stub.eval_dataset = {"validation": original}
    stub.evaluate("validation")

    assert seen["ds"] == "validation", "the key itself must still be handed down"
    during = seen["resolved"]
    assert max(len(r) for r in during["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH
    assert stub.eval_dataset["validation"] is original, "the original must survive"


def test_a_named_split_can_still_be_capped_from_the_other_end():
    """Restoring the original lets keep_end after keep_start on the same key still produce the suffix."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    original = _tokenized_dataset(tok)
    stub.eval_dataset = {"validation": original}

    stub.evaluate("validation")
    head = list(seen["resolved"]["input_ids"])

    stub.args.truncation_mode = "keep_end"
    stub.evaluate("validation")
    tail = list(seen["resolved"]["input_ids"])

    assert max(len(r) for r in tail) <= _MODEL_MAX_SEQ_LENGTH
    assert any(
        len(row) > _MODEL_MAX_SEQ_LENGTH and h != t
        for row, h, t in zip(original["input_ids"], head, tail)
    ), "an overlength row must give a different suffix than prefix"


def test_an_unknown_string_key_is_handed_straight_back():
    """A key that is not in the stored dict, or no dict at all, is HF's problem
    to report: capping must not turn it into a different error."""
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.eval_dataset = {"validation": None}
    stub.evaluate("missing")

    assert seen["ds"] == "missing"


def _transformed_tokenized_dataset(tok):
    """A split whose backing table says `text` while it yields long `input_ids`."""
    from datasets import Dataset

    ids = tok("The quick brown fox. " * 200)["input_ids"]
    ds = Dataset.from_list([{"text": "x"}] * 4)
    return ds.with_transform(
        lambda batch: {
            "input_ids": [list(ids)] * len(batch["text"]),
            "attention_mask": [[1] * len(ids)] * len(batch["text"]),
        }
    )


def _transformed_short_dataset(tok):
    """The same transform, yielding rows that already fit the cap."""
    from datasets import Dataset

    ids = tok("The quick brown fox.")["input_ids"]
    assert len(ids) <= _MODEL_MAX_SEQ_LENGTH
    ds = Dataset.from_list([{"text": "x"}] * 4)
    return ds.with_transform(
        lambda batch: {
            "input_ids": [list(ids)] * len(batch["text"]),
            "attention_mask": [[1] * len(ids)] * len(batch["text"]),
        }
    )


class _SharedIteratorStream:
    """A single-pass stream with no `column_names`: `iter()` hands back the same
    exhausting generator every time, so a probe read is a row the run loses."""

    def __init__(self, rows):
        self._rows = iter(rows)

    def __iter__(self):
        return self._rows


def test_a_transformed_tokenized_split_keeps_its_cap(tmp_path, trl_has_guard):
    """Transformed splits keep max_length and raise on overlength rows rather than go uncapped."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            dataset = _transformed_tokenized_dataset,
            padding_free = True,
            max_length = _MODEL_MAX_SEQ_LENGTH,
        )


def test_a_transformed_split_within_the_cap_keeps_max_length_and_trains(tmp_path, trl_has_guard):
    """The other half: the same shape with nothing overlength must not be cleared
    either, and must not raise. This is what shows the raise above is about the
    rows and not about the transform."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(
        tmp_path,
        dataset = _transformed_short_dataset,
        padding_free = True,
        max_length = _MODEL_MAX_SEQ_LENGTH,
    )
    assert trainer.args.max_length is not None, "nothing else enforces the cap"


def test_an_unprobeable_tokenized_stream_keeps_its_cap(tmp_path, trl_has_guard):
    """A stream with no schema and no spare row cannot be ruled tokenized, and
    clearing the cap on that guess leaves padding-free training uncapped."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    rows = [{"input_ids": list(ids), "attention_mask": [1] * len(ids)}] * 4
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            dataset = lambda _tok: _SharedIteratorStream(rows),
            padding_free = True,
            max_length = _MODEL_MAX_SEQ_LENGTH,
        )


def test_the_schema_read_distrusts_a_transform_and_an_unprobeable_stream():
    """The schema read distrusts transforms, and a stream that cannot spare a row must refuse."""
    block = _padding_free_codegen_block()
    assert "_unsloth_transformed" in block, "the transform is not detected at all"
    assert (
        "None if _unsloth_transformed else getattr(train_dataset, 'column_names', None)" in block
    ), "a transformed split is still read off its backing columns"
    guard = "if _unsloth_probe_cols is train_dataset or _unsloth_probe_cols is iter(train_dataset):"
    assert guard in block
    refusal = block.index("_unsloth_prep_truncates = False", block.index(guard))
    assert (
        refusal - block.index(guard) < 200
    ), "an unprobeable stream still claims preparation will truncate it"


def test_a_none_valued_token_column_does_not_defeat_the_late_cap():
    """A None-valued token column must not defeat the late cap; the allow-list judged names, not values."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_SEQ := _MODEL_MAX_SEQ_LENGTH, None)
    rows = _tokenized_dataset(tok).to_list()
    for row in rows:
        row["token_type_ids"] = None
    from datasets import Dataset

    stub.evaluate(Dataset.from_list(rows))

    got = seen["ds"]
    assert max(len(r) for r in got["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_a_token_major_two_dimensional_column_is_sliced():
    """`[seq_len, channels]` is one vector PER TOKEN, so its first axis is the
    token axis and `[:cap]` cuts it correctly. Leaving it alone handed a custom
    collator the old sequence length beside capped tokens."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    rows = _tokenized_dataset(tok).to_list()
    for row in rows:
        row["position_ids"] = [[i, 0] for i in range(len(row["input_ids"]))]
    from datasets import Dataset

    stub.evaluate(Dataset.from_list(rows))

    got = seen["ds"]
    assert max(len(r) for r in got["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH
    assert len(got["position_ids"][0]) == len(got["input_ids"][0])


def test_a_channel_major_two_dimensional_column_is_left_alone():
    """Channel-major columns like mrope position_ids have channels on axis 0 and must not be cut there."""
    _, tok = _load_plain()
    Stub, seen = _stub_with_stored_eval()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    rows = _tokenized_dataset(tok).to_list()
    for row in rows:
        row["position_ids"] = [list(range(len(row["input_ids"]))) for _ in range(3)]
    from datasets import Dataset

    stub.evaluate(Dataset.from_list(rows))

    got = seen["ds"]
    assert max(len(r) for r in got["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH
    assert len(got["position_ids"][0]) == 3, "a channel axis was cut as if it were tokens"


@pytest.mark.parametrize(
    "method, keyword",
    [
        ("get_eval_dataloader", "eval_dataset"),
        ("get_test_dataloader", "test_dataset"),
    ],
)
def test_the_dataloader_builders_cap_a_late_split_too(method, keyword):
    """Both are public API and neither goes through `evaluate`/`predict`, so a
    caller building a dataloader directly reached the padding-free collator with
    `args.max_length` already cleared and nothing capping the split."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    _, tok = _load_plain()
    seen = {}

    def _builder(
        self,
        split = None,
        **kw,
    ):
        seen["ds"] = split

    Stub = type("Stub", (), {method: _builder})
    _wrap_sft_evaluate_cap(Stub)
    assert getattr(
        getattr(Stub, method), "_unsloth_eval_cap_wrapped", False
    ), f"{method} was never wrapped"

    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    getattr(stub, method)(_tokenized_dataset(tok))

    assert max(len(r) for r in seen["ds"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_the_pretokenized_probe_does_not_eat_a_one_shot_row():
    """`next(iter(_ds))` on a single-pass stream is a row the run then trains
    without: read raw it declares the split safe and training starts at row 2,
    read tokenized it rejects a caller-owned stream it has already mutated."""
    block = _padding_free_codegen_block()
    # Sliced to the next def, not a fixed window, so added comments cannot hide the line.
    start = block.index("def _unsloth_pretokenized")
    body = block[start : block.index("def _unsloth_cap_split", start)]
    assert (
        "_probe is _ds or _probe is iter(_ds)" in body
    ), "the pretokenized probe still reads a row off a one-shot stream"
    assert body.index("column_names") < body.index(
        "iter(_ds)"
    ), "the schema is not consulted before a row is taken"


def test_capping_a_one_shot_stream_twice_does_not_eat_its_rows():
    """Capping a one-shot stream twice, in evaluate() and get_eval_dataloader, must not consume its rows."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    rows = [{"input_ids": list(ids), "attention_mask": [1] * len(ids)} for _ in range(4)]
    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            return self.get_eval_dataloader(eval_dataset)

        def get_eval_dataloader(
            self,
            eval_dataset = None,
            **kw,
        ):
            split = self.eval_dataset if eval_dataset is None else eval_dataset
            seen["rows"] = list(split)
            return split

    _wrap_sft_evaluate_cap(Stub)
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.eval_dataset = None
    stub.evaluate(_SharedIteratorStream(rows))

    assert len(seen["rows"]) == len(rows), "the second cap ate rows off the stream"
    assert all(len(r["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH for r in seen["rows"])


def test_a_capped_split_is_handed_straight_back_to_the_second_pass():
    """The signature is what stops the second pass, and it must be OUR mark:
    `_CappedBase.__getattr__` forwards anything it does not hold to the split
    inside, so an unmarked wrapper around a marked split would answer for it."""
    from unsloth.models import rl

    inner = rl._CappedRows.__new__(rl._CappedRows)
    inner.__dict__[rl._CAP_SIGNATURE_ATTR] = (16, True)
    outer = rl._CappedRows.__new__(rl._CappedRows)
    outer.__dict__["_inner"] = inner

    assert rl._cap_signature(inner) == (16, True)
    assert rl._cap_signature(outer) is None, "the outer wrapper read the inner one's mark"
    assert getattr(outer, rl._CAP_SIGNATURE_ATTR) == (
        16,
        True,
    ), "premise: a plain getattr does forward, which is why __dict__ is read"


def test_an_unknown_truncation_mode_leaves_the_split_alone():
    """An unknown truncation_mode must leave the split untouched, not cut it silently from the start."""
    block = _padding_free_codegen_block()
    for guarded in (
        "if _unsloth_known_mode and not _unsloth_prep_truncates:",
        "if _unsloth_eval_packing or not _unsloth_known_mode:",
    ):
        assert guarded in block, f"the split is still rewritten under an unknown mode: {guarded}"
    assert "_unsloth_capped = _unsloth_known_mode" in block


def test_the_transform_rule_is_read_by_both_schema_probes():
    """Both schema probes must share one transform rule, since column_names describes the backing table."""
    block = _padding_free_codegen_block()
    assert (
        block.count("_unsloth_is_transformed(") >= 3
    ), "the rule is not defined once and read by both probes"
    start = block.index("def _unsloth_pretokenized")
    body = block[start : block.index("def _unsloth_cap_split", start)]
    assert body.index("_unsloth_is_transformed(_ds)") < body.index(
        "column_names"
    ), "the late probe still trusts the backing columns of a transformed split"


def test_a_transformed_eval_split_keeps_its_cap(tmp_path, trl_has_guard):
    """The same split on the eval side, which is the path `_unsloth_pretokenized`
    decides: `_unsloth_truncatable` refuses to rewrite it, so the answer here is
    what clears `max_length` or holds it."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    tok = _load_plain()[1]
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            eval_dataset = _transformed_tokenized_dataset(tok),
            padding_free = True,
            max_length = _MODEL_MAX_SEQ_LENGTH,
        )
    trainer = _build(
        tmp_path,
        eval_dataset = _transformed_short_dataset(tok),
        padding_free = True,
        max_length = _MODEL_MAX_SEQ_LENGTH,
    )
    assert trainer.args.max_length is not None, "nothing else truncates the yielded rows"


def test_a_one_shot_stream_slices_every_aligned_column():
    """Every aligned column must be sliced with input_ids, or labels fall out of step with the tokens."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    rows = [
        {"input_ids": list(ids), "attention_mask": [1] * len(ids), "labels": list(ids)}
        for _ in range(3)
    ]
    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["rows"] = list(eval_dataset)

    _wrap_sft_evaluate_cap(Stub)
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.evaluate(_SharedIteratorStream(rows))

    assert len(seen["rows"]) == len(rows), "the probe ate a row"
    for row in seen["rows"]:
        width = len(row["input_ids"])
        assert width <= _MODEL_MAX_SEQ_LENGTH
        for name in ("attention_mask", "labels"):
            assert len(row[name]) == width, f"{name} is not aligned with input_ids"


def test_an_unfiltered_map_style_split_is_not_scanned_up_front():
    """With no supervision columns every row survives, so building an identity
    index read and transformed the whole split before the dataloader could
    start -- a second on-access tokenization pass for no information."""
    from unsloth.models.rl import _CappedRows

    reads = []

    class Split:
        def __len__(self):
            return 500

        def __getitem__(self, i):
            reads.append(i)
            return {"input_ids": list(range(40))}

    capped = _CappedRows(Split(), slice(None, 8), (), ("input_ids",))
    assert not reads, f"constructor read {len(reads)} rows before anything asked"
    assert len(capped) == 500
    assert len(capped[0]["input_ids"]) == 8
    assert reads == [0], "indexing did not map straight through"


def test_a_filtered_split_still_drops_its_unsupervised_rows():
    """The control: supervision present means the index is real, and the rows
    with no supervised token still go."""
    from unsloth.models.rl import _CappedRows

    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, -100]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]

    class Split:
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    capped = _CappedRows(Split(), slice(None, 3), ("labels",), ("input_ids", "labels"))
    assert len(capped) == 1
    assert capped[0]["labels"] == [4, 5, 6]


def test_switching_truncation_mode_re_caps_the_same_split():
    """The memo keyed on identity, cap and filtering mode but not on the SLICE,
    so evaluating with keep_start and then keep_end handed back the cached
    prefixes for both."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)

    stub.args.truncation_mode = "keep_start"
    stub.evaluate(eval_dataset = late)
    starts = list(seen["ds"]["input_ids"][0])

    stub.args.truncation_mode = "keep_end"
    stub.evaluate(eval_dataset = late)
    ends = list(seen["ds"]["input_ids"][0])

    full = list(late["input_ids"][0])
    assert starts == full[:_MODEL_MAX_SEQ_LENGTH]
    assert ends == full[-_MODEL_MAX_SEQ_LENGTH:]
    assert starts != ends, "keep_end reused the cached keep_start prefix"


def test_the_late_cap_refuses_a_truncation_mode_it_cannot_honour():
    """The construction path already refuses a third value; the late cap took
    it as keep_start and cut every row from the side the caller ruled out."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
    stub.args.truncation_mode = "keep_middle"
    stub.evaluate(eval_dataset = late)
    assert seen["ds"] is late, "the split was cut under a mode we cannot honour"


def _cap_scan_shapes():
    """Pins the generated __init__ cap scan to rl.pretokenized_within_cap, since the generated copy
    can drift."""
    from datasets import Dataset

    fits = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]]})
    over = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5, 6]]})
    raw = Dataset.from_dict({"text": ["a", "b"]})

    def one_shot():
        yield {"input_ids": [1, 2]}

    return [
        (None, True),
        (fits, True),
        (over, False),
        (raw, True),
        ([{"input_ids": [1, 2]}], True),
        ([{"input_ids": [1, 2, 3, 4]}], False),
        (one_shot(), False),
    ]


@pytest.mark.parametrize("dataset, expected", _cap_scan_shapes())
def test_the_importable_cap_scan_matches_the_generated_one(dataset, expected):
    from unsloth.models.rl import pretokenized_within_cap
    assert pretokenized_within_cap(dataset, 3) is expected


@pytest.mark.parametrize("dataset, expected", _cap_scan_shapes())
def test_the_generated_cap_scan_matches_the_importable_one(dataset, expected):
    """The inline copy, extracted from the generator and executed as written."""
    import inspect as _inspect
    import re
    from unsloth.models import rl

    source = _inspect.getsource(rl)
    start = source.index('"    def _unsloth_within_cap(_ds):\\n"')
    end = source.index('"    def _unsloth_splits_within_cap(_ev):\\n"')
    lines = re.findall(r'^\s*"(.*?)\\n"\s*$', source[start:end], re.M)
    namespace = {"_unsloth_cap": 3, "_UNSLOTH_SCAN_ROWS": 1024}
    exec("\n".join(line[4:] for line in lines), namespace)
    assert namespace["_unsloth_within_cap"](dataset) is expected


def test_an_unscannable_split_never_reads_as_capped():
    """A split that raises mid-scan has proven nothing, and the caller is about
    to decide whether anything downstream enforces the cap."""
    from unsloth.models.rl import pretokenized_within_cap, splits_within_cap

    class Angry:
        def __len__(self):
            return 2

        def __iter__(self):
            yield {"input_ids": [1]}
            raise RuntimeError("no")

    assert pretokenized_within_cap(Angry(), 3) is False
    assert splits_within_cap({"a": Angry()}, 3) is False


def test_every_eval_split_counts_towards_the_cap():
    from datasets import Dataset
    from unsloth.models.rl import splits_within_cap

    fits = Dataset.from_dict({"input_ids": [[1, 2]]})
    over = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]})
    assert splits_within_cap({"a": fits}, 3) is True
    assert splits_within_cap({"a": fits, "b": over}, 3) is False


def _padding_free_fallback(
    train = None,
    evals = None,
    max_length = 3,
):
    """Runs the padding-free retry on a rejecting init; returns its call count, or the propagated error."""
    from types import SimpleNamespace
    from unsloth.trainer import (
        _bound_splits,
        _cap_is_enforceable_without_padding_free,
    )

    def original_init(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        **kw,
    ):
        pass

    config = SimpleNamespace(max_length = max_length, padding_free = True)
    kwargs = {"train_dataset": train, "eval_dataset": evals}
    bound_train, bound_evals = _bound_splits(original_init, (None, config), kwargs)
    assert bound_train is train and bound_evals is evals
    return _cap_is_enforceable_without_padding_free(config, bound_train, bound_evals)


def test_the_padding_free_fallback_refuses_a_split_it_cannot_cap():
    """Retrying with padding-free off would run uncapped, so such a split must be refused."""
    from datasets import Dataset

    over = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5, 6]]})
    assert _padding_free_fallback(train = over) is False
    assert _padding_free_fallback(evals = {"validation": over}) is False


def test_the_padding_free_fallback_still_runs_when_the_cap_holds():
    """Raw text and already-short rows are both fine: prep truncates the first
    and the second needs no truncating. The fallback must not become a wall."""
    from datasets import Dataset

    assert _padding_free_fallback(train = Dataset.from_dict({"input_ids": [[1, 2]]})) is True
    assert _padding_free_fallback(train = Dataset.from_dict({"text": ["hello"]})) is True
    assert _padding_free_fallback(train = None, max_length = None) is True


def test_the_fallback_reads_splits_through_the_signature():
    """TRL has moved these parameters between releases; a positional index reads
    the data collator on the version that did."""
    from unsloth.trainer import _bound_splits

    def moved(
        self,
        model = None,
        processing_class = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
    ):
        pass

    train, evals = _bound_splits(moved, (None, "tok", "args", "TRAIN", "EVAL"), {})
    assert (train, evals) == ("TRAIN", "EVAL")


def test_completion_only_ignores_the_columns_of_a_transformed_split():
    """Completion-only mode must read yielded columns of a with_transform split, not its backing table."""
    import inspect as _inspect
    from unsloth.models import rl

    source = _inspect.getsource(rl)
    guard = (
        "_unsloth_train_sample = {} if _unsloth_is_transformed(train_dataset) else dict.fromkeys("
    )
    assert guard in source
    assert "_unsloth_train_sample = next(_unsloth_probe, None) or {}" in source


def _row(ids, **extra):
    row = {"input_ids": list(ids), "attention_mask": [1] * len(ids)}
    row.update(extra)
    return row


def test_a_later_row_that_cannot_take_the_slice_does_not_raise():
    """Every row must be validated, not just the probed one, since an optional column can be None later."""
    from unsloth.models.rl import _CappedRows

    rows = [
        _row(range(10), token_type_ids = [0] * 10),
        _row(range(10), token_type_ids = None),
    ]

    class Split:
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

        def __iter__(self):
            return iter(rows)

    capped = _CappedRows(
        Split(), slice(None, 4), (), ("input_ids", "attention_mask", "token_type_ids")
    )
    out = list(capped)
    assert [len(r["input_ids"]) for r in out] == [4, 4]
    assert out[0]["token_type_ids"] == [0] * 4
    assert out[1]["token_type_ids"] is None, "an unsliceable value must be left alone, not cut"


def test_a_misaligned_later_row_keeps_its_own_length():
    """Same probe, different drift: a column that is aligned in row 0 and a
    different width in row 1. Cutting it there would report a mask for tokens
    the row never had."""
    from unsloth.models.rl import _CappedRows

    # Longer than the tokens, so cutting it is visible.
    rows = [_row(range(10), labels = [1] * 10), _row(range(10), labels = [1] * 20)]
    capped = _CappedRows(rows, slice(None, 4), (), ("input_ids", "labels"))
    out = list(capped)
    assert out[0]["labels"] == [1] * 4
    assert out[1]["labels"] == [1] * 20, "a misaligned value was cut to a width it never had"


def test_input_ids_comes_first_so_every_column_is_measured():
    """`_column_names` returns a SET, and the `map` path reads the width off
    `input_ids` as it walks this list. A run that ordered `labels` first sliced
    the labels having compared them to nothing at all."""
    from unsloth.models.rl import _sliceable_per_token

    names = ("labels", "attention_mask", "input_ids")
    kept = _sliceable_per_token(None, names, 4, _row(range(10), labels = [1] * 10))
    assert kept[0] == "input_ids", kept


def test_a_custom_per_token_column_rides_along_with_the_slice():
    """`loss_mask` is not on the allow-list, so it kept its full length while
    `input_ids` was cut and a custom collator got mismatched rows."""
    from unsloth.models.rl import _sliceable_per_token

    probed = _row(range(10), loss_mask = [1] * 10)
    kept = _sliceable_per_token(None, set(probed), 4, probed)
    assert "loss_mask" in kept


def test_a_coincidentally_long_text_column_does_not_ride_along():
    """Alignment alone is not proof: a list of ten strings is ten long too.
    Only a flat vector of scalars is a per-token field."""
    from unsloth.models.rl import _sliceable_per_token

    probed = _row(range(10), messages = [{"role": "user"}] * 10, tags = ["a"] * 10, text = "0123456789")
    kept = _sliceable_per_token(None, set(probed), 4, probed)
    assert "messages" not in kept and "tags" not in kept and "text" not in kept


def test_a_mark_is_not_trusted_after_the_split_is_mutated():
    """A cap mark is not trusted once the split is mutated, or new longer rows skip the rescan."""
    from unsloth.models import rl

    class Split:
        _fingerprint = "before"

    split = Split()
    rl._mark_capped(split, 16, True)
    assert rl._cap_still_holds(split, 16, True)
    split._fingerprint = "after"
    assert not rl._cap_still_holds(split, 16, True), "a moved fingerprint still read as capped"


def test_an_unfingerprintable_split_is_never_trusted_by_its_mark():
    """The memo excludes these on purpose because their rows can change under a
    stable identity. The mark has to reach the same conclusion, or it becomes
    the way around the memo."""
    from unsloth.models import rl

    class Plain:
        pass

    plain = rl._mark_capped(Plain(), 16, True)
    assert not rl._cap_still_holds(plain, 16, True)

    class Transformed:
        _fingerprint = "x"
        format = {"type": "custom"}

    assert not rl._cap_still_holds(rl._mark_capped(Transformed(), 16, True), 16, True)


def test_our_own_wrapper_is_still_handed_straight_back():
    """The mark exists to stop the paired wrappers capping one call twice, and
    over a one-shot stream the second pass is destructive. A wrapper holds a
    fixed slice and cannot drift, so it is trusted without a fingerprint."""
    from unsloth.models import rl

    wrapper = rl._CappedRows([], slice(None, 4), (), ("input_ids",))
    rl._mark_capped(wrapper, 16, True)
    assert rl._cap_still_holds(wrapper, 16, True)
    assert not rl._cap_still_holds(wrapper, 32, True), "a different cap must still rescan"


def test_the_late_evaluation_memo_is_bounded():
    """Every entry pins the original split AND the capped copy for the trainer's
    lifetime. A caller building a fresh validation subset each epoch grew this
    dictionary without bound until the host ran out of memory."""
    from unsloth.models import rl

    _, tok = _load_plain()
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    for _ in range(rl._EVAL_CAP_MEMO_MAX * 3):
        stub.evaluate(eval_dataset = _tokenized_dataset(tok))
    memo = getattr(stub, "_unsloth_eval_cap_memo", {})
    assert 0 < len(memo) <= rl._EVAL_CAP_MEMO_MAX, len(memo)


def test_a_nullable_value_does_not_break_the_construction_time_truncation():
    """A column judged from its first row must not break truncation when a later row is None."""
    block = _padding_free_codegen_block()
    assert "_unsloth_cut_value(_v, _r)" in block, "the batch map still slices unguarded"
    import inspect as _inspect
    import re
    from unsloth.models import rl

    source = _inspect.getsource(rl)
    start = source.index('"        def _unsloth_cut_value(_v, _r):\\n"')
    end = source.index('"        def _unsloth_truncate_rows(_batch):\\n"')
    lines = re.findall(r'^\s*"(.*?)\\n"\s*$', source[start:end], re.M)
    scope = {"_unsloth_slice": slice(None, 4)}
    exec("\n".join(line[8:] for line in lines), scope)
    cut = scope["_unsloth_cut_value"]
    ids = list(range(10))
    assert cut([1] * 10, ids) == [1] * 4
    assert cut(None, ids) is None, "a None value was sliced"
    assert cut(7, ids) == 7, "a scalar value was sliced"
    assert cut([1] * 3, ids) == [1] * 3, "a misaligned value was cut"


def test_completion_only_reads_the_columns_the_split_actually_yields():
    """Completion-only detection must use the columns set_format yields, not the backing table's."""
    block = _padding_free_codegen_block()
    for fragment in (
        "_unsloth_shown = _unsloth_fmt.get('columns')",
        "if _unsloth_fmt.get('output_all_columns') or not _unsloth_shown:",
    ):
        assert fragment in block, fragment
    from datasets import Dataset

    ds = Dataset.from_list([{"prompt": "a", "completion": "b", "input_ids": [1, 2]}])
    # datasets<4 numpy/torch formatters import torchvision.io.VideoReader, gone in torchvision 0.28.
    ds.set_format(None, columns = ["input_ids"], output_all_columns = False)
    assert "completion" in ds.column_names
    assert ds.format.get("columns") == ["input_ids"]
    assert "completion" not in ds[0]


def test_the_fallback_does_not_scan_an_eval_packed_split():
    """Disabling padding-free keeps `max_length`, and TRL's eval packer owns and
    chunks the overflow, so an overlength row in a packed eval split is not an
    unenforced cap. The generated exact-match path already excludes those."""
    from unsloth.trainer import _cap_is_enforceable_without_padding_free as enforceable

    long_rows = [{"input_ids": list(range(64))}]
    short = [{"input_ids": [1, 2]}]

    class Config:
        max_length = 8
        packing = False
        eval_packing = None

    config = Config()
    assert not enforceable(config, short, long_rows), "premise: unpacked evals are scanned"
    config.eval_packing = True
    assert enforceable(config, short, long_rows)
    config.eval_packing, config.packing = None, True
    assert enforceable(config, long_rows, long_rows)


def test_a_zoo_that_already_normalizes_the_seed_is_left_alone():
    """A zoo that already normalizes the seed is left alone, rather than failing every SFT trainer."""
    from unsloth.models import rl_replacements as R

    old = '    max_seq_length = getattr(args, "max_length", 0)'
    new = '    max_seq_length = getattr(args, "max_length", 0) or 0'
    done = "def f():\n" + new + "\n"
    assert (
        R._replace_or_fallback(
            done,
            old,
            new,
            fallback_pattern = R._ZOO_MAX_LENGTH_SEED,
            fallback_new = r'\g<indent>max_seq_length = getattr(args, "max_length", 0) or 0',
            where = "test",
            required = True,
        )
        == done
    )
    todo = "def f():\n" + old + "\n"
    assert (
        R._replace_or_fallback(
            todo,
            old,
            new,
            fallback_pattern = R._ZOO_MAX_LENGTH_SEED,
            fallback_new = r'\g<indent>max_seq_length = getattr(args, "max_length", 0) or 0',
            where = "test",
            required = True,
        )
        == done
    )


def test_a_single_quoted_normalized_seed_is_recognised():
    """The idempotence check must also accept single-quoted seeds, or required fails every SFT trainer."""
    from unsloth.models import rl_replacements as R

    old = '    max_seq_length = getattr(args, "max_length", 0)'
    new = '    max_seq_length = getattr(args, "max_length", 0) or 0'
    kwargs = dict(
        fallback_pattern = R._ZOO_MAX_LENGTH_SEED,
        fallback_new = r'\g<indent>max_seq_length = getattr(args, "max_length", 0) or 0',
        where = "test",
        required = True,
    )
    single = "def f():\n    max_seq_length = getattr(args, 'max_length', 0) or 0\n"
    assert R._replace_or_fallback(single, old, new, **kwargs) == single
    assert new not in single and not R._ZOO_MAX_LENGTH_SEED.search(single)


def test_the_late_cap_prefers_the_trainers_resolved_completion_mode():
    """The late cap uses the completion mode TRL resolved on the training sample, not the split's schema."""
    _, tok = _load_plain()
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    for name in ("_unsloth_completion_only_loss", "completion_only_loss"):
        if hasattr(stub.args, name):
            setattr(stub.args, name, None)
    stub.completion_only_loss = True

    stub.evaluate(eval_dataset = _tokenized_dataset(tok))
    assert getattr(stub.args, "_unsloth_resolved_completion_only") is True

    stub.completion_only_loss = False
    stub.evaluate(eval_dataset = _tokenized_dataset(tok))
    assert getattr(stub.args, "_unsloth_resolved_completion_only") is False


def test_the_pre_truncation_rewrite_runs_under_the_rank_window():
    """Every rank reaches this before TRL's `_prepare_dataset`, and TRL runs its
    own preparation maps under `main_process_first`. Without it, eight ranks each
    start `num_proc` workers against one Arrow cache."""
    block = _padding_free_codegen_block()
    for fragment in (
        "def _unsloth_rank_first():",
        "from accelerate import PartialState",
        "return PartialState().main_process_first()",
        "with _unsloth_rank_first():",
        "return _unsloth_cap_one(_ds)",
    ):
        assert fragment in block, fragment
    assert "def _unsloth_cap_one(_ds):" in block
    assert block.index("def _unsloth_cap_split(_ds):") < block.index("def _unsloth_cap_one(_ds):")


def test_the_rank_window_degrades_to_a_no_op():
    """A single process, or an accelerate that cannot build a PartialState, must
    still cap. The helper is executed as written."""
    import inspect as _inspect
    import re
    from unsloth.models import rl

    source = _inspect.getsource(rl)
    start = source.index('"        def _unsloth_rank_first():\\n"')
    end = source.index('"        def _unsloth_cap_split(_ds):\\n"')
    lines = re.findall(r'^\s*"(.*?)\\n"\s*$', source[start:end], re.M)
    scope = {}
    exec("\n".join(line[8:] for line in lines), scope)
    with scope["_unsloth_rank_first"]():
        pass
