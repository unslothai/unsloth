# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Padding-free + `max_length` handshake across TRL versions.

TRL >= 1.0.0 refuses padding-free without packing while `args.max_length` is set,
which every default SFT run tripped. rl.py now hands those TRLs the `None` they
ask for and truncates through `max_seq_length` instead. Pinned below: the
resolved length is unchanged, and the swap only happens when Unsloth's dataset
prep really tokenizes -- otherwise padding-free is dropped and `max_length` kept.
Assertions branch on whether the installed TRL carries the guard.
"""

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

# Spoof CUDA before any unsloth import, so CPU-only runners can still patch.
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


torch.compile = _eager_compile
if hasattr(torch, "accelerator"):
    torch.accelerator.is_available = lambda *a, **k: False

_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# Model-level length the trainer resolves from when the user names none.
_MODEL_MAX_SEQ_LENGTH = 128
# Smaller than the model cap, so honouring it is visible.
_USER_MAX_LENGTH = 64


@pytest.fixture(scope = "module", autouse = True)
def patched_sft():
    """Import unsloth, and force both halves of the SFT patch on.

    `UNSLOTH_ALLOW_CPU=1` (which CPU-only CI sets) skips both, so ask explicitly,
    in `_gpu_init`'s order: the codegen swaps `trl.SFTTrainer` out wholesale, so
    the `__init__` wrapper has to go on afterwards. Both are no-ops once applied.
    """
    global torch  # the `import torch._dynamo` below would otherwise shadow it
    import unsloth  # noqa: F401

    torch.compile = _eager_compile
    try:
        import torch._dynamo
        torch._dynamo.config.disable = True
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
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_MODEL, dtype = torch.float32)
    except OSError as e:
        pytest.skip(f"could not fetch {_MODEL} (network/hub): {str(e)[:150]}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.max_seq_length = model_max_seq_length
    return model.to("cpu"), tok


def _build(
    tmp_path,
    dataset = None,
    model_max_seq_length = _MODEL_MAX_SEQ_LENGTH,
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
    return SFTTrainer(model = model, processing_class = tok, args = cfg, train_dataset = ds)


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
        # What TRL >= 1.0.0 asks for; truncation moves to Unsloth's dataset prep.
        assert args.max_length is None
    else:
        assert args.max_length == _MODEL_MAX_SEQ_LENGTH
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH


def test_explicit_max_length_resolves_the_same_on_every_trl(tmp_path, trl_has_guard):
    """The swap only moves the already-resolved length across to `max_seq_length`;
    it must never reinstate the raw user `max_length`."""
    trainer = _build(tmp_path, max_length = _USER_MAX_LENGTH)
    args = trainer.args

    assert args.padding_free is True
    assert args.max_seq_length == _MODEL_MAX_SEQ_LENGTH
    if trl_has_guard:
        assert args.max_length is None
    else:
        assert args.max_length == _MODEL_MAX_SEQ_LENGTH
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH


def test_max_seq_length_still_beats_max_length(tmp_path, trl_has_guard):
    """`max_seq_length=4096, max_length=512` must truncate at 4096, not 512.

    Re-reading the raw user `max_length` inside the padding-free branch would
    invert that precedence, on TRL >= 1.0.0 only.
    """
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
    """A pre-tokenized dataset is truncated here, exactly as TRL would have done.

    TRL enforces `max_length` in `_prepare_dataset` (`truncate_dataset`), not in the
    collator: the LM collator it builds is constructed without a `max_length`. The
    Zoo's prep returns already-tokenized rows untouched, so leaving the rows long
    meant nothing enforced the cap and a 402-token row reached the model under a
    128-token request. Truncating restores TRL's own contract, and padding-free can
    then stay on rather than being dropped.
    """
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = dataset)
    args = trainer.args

    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH, f"{name}: rows were not truncated"
    # Truncation happened, so the cap is spent and padding-free keeps its speed win.
    assert args.max_length is None, f"{name}: the cap should be consumed by the truncation"
    assert args.max_seq_length == _MODEL_MAX_SEQ_LENGTH, f"{name}: the cap must be recorded"
    assert args.padding_free is True, f"{name}: padding-free no longer needs dropping"
    # Padding-free CONCATENATES the batch into one flat sequence, so the collated width is
    # rows x cap, not the cap. Two rows at exactly 2 x 128 is the proof that neither row
    # exceeded it; asserting 128 here would be asserting that padding-free was off.
    assert _collated_width(trainer) == 2 * _MODEL_MAX_SEQ_LENGTH


@pytest.mark.parametrize(
    "name, dataset, config_kwargs",
    [
        (
            "skip_prepare_dataset",
            lambda tok: _tokenized_dataset(tok),
            {"dataset_kwargs": {"skip_prepare_dataset": True}},
        ),
    ],
)
def test_unprepared_datasets_keep_their_length_cap(
    tmp_path, trl_has_guard, name, dataset, config_kwargs
):
    """`skip_prepare_dataset` means the user asked for the dataset to be left alone.

    TRL skips truncation there too, so touching the rows would break the promise the
    flag makes. Drop padding-free instead and keep the cap for the collator.
    """
    trainer = _build(tmp_path, dataset = dataset, **config_kwargs)
    args = trainer.args

    assert (
        args.max_length == _MODEL_MAX_SEQ_LENGTH
    ), f"{name}: the length cap must not be cleared for an unprepared dataset"
    if trl_has_guard:
        assert (
            args.padding_free is False
        ), f"{name}: padding-free must be dropped, since it disables truncation"
    # The rows themselves stay long: the user owns preparation here.
    assert _longest(trainer) > _MODEL_MAX_SEQ_LENGTH
    if getattr(trainer.data_collator, "max_length", None) is not None:
        assert (
            _collated_width(trainer) == _MODEL_MAX_SEQ_LENGTH
        ), f"{name}: overlength rows reached the model"


def _transformed_dataset(tok):
    """A dataset that tokenizes on access through `with_transform`.

    `column_names` still says `["text"]` while rows yield `input_ids`, so a check
    that trusts `column_names` wrongly concludes the tokenize pass will truncate.
    """
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


def test_transformed_datasets_keep_their_length_cap(tmp_path, trl_has_guard):
    """An on-access tokenizing transform is an unprepared dataset too."""
    trainer = _build(tmp_path, dataset = _transformed_dataset)
    args = trainer.args
    rows = [trainer.train_dataset[i] for i in range(2)]

    assert trainer.train_dataset.column_names == ["text"], "transform should hide the schema"
    assert "input_ids" in rows[0], "rows should already be tokenized"
    assert args.max_length == _MODEL_MAX_SEQ_LENGTH, "the length cap must not be cleared"
    if trl_has_guard:
        assert args.padding_free is False, "padding-free must be dropped, it disables truncation"
    assert max(len(r["input_ids"]) for r in rows) > _MODEL_MAX_SEQ_LENGTH
    if getattr(trainer.data_collator, "max_length", None) is not None:
        assert (
            _collated_width(trainer) == _MODEL_MAX_SEQ_LENGTH
        ), "overlength rows reached the model"


def _pristine_sft_config_cls():
    """TRL's own SFTConfig, not Unsloth's generated subclass of it.

    `PatchFastRL` rebinds `trl.SFTConfig` to `UnslothSFTConfig`, which re-adds a
    `max_seq_length` field no TRL from 0.22.2 to 1.9.2 declares. A caller who
    imported SFTConfig before `import unsloth` still passes the pristine class.
    """
    from trl import SFTConfig
    return SFTConfig.__bases__[0] if SFTConfig.__name__.startswith("Unsloth") else SFTConfig


def test_pristine_trl_config_without_max_seq_length_still_truncates(tmp_path, trl_has_guard):
    """A config with `max_length` and no `max_seq_length` must keep its cap.

    Gating the copy into `max_seq_length` on `hasattr` skipped it for every
    pristine `trl.SFTConfig`, so the cap was cleared and never stored.
    """
    from datasets import Dataset

    config_cls = _pristine_sft_config_cls()
    assert not hasattr(
        config_cls(output_dir = str(tmp_path)), "max_seq_length"
    ), "this TRL declares max_seq_length, so the regression cannot be reproduced here"

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

    # The cap has to land somewhere the Zoo's sft_prepare_dataset reads it.
    if trl_has_guard:
        assert trainer.args.max_length is None
        assert trainer.args.max_seq_length == _MODEL_MAX_SEQ_LENGTH
        assert trainer.args.padding_free is True
    else:
        # No guard, so the swap is not emitted at all and `max_length` carries it.
        assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH

    # What actually matters: the rows, and the batch the model would see.
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
    """The copy into `max_seq_length` must not be conditional.

    `hasattr(args, 'max_seq_length')` is False for every pristine TRL SFTConfig,
    so a gate there is an unconditional skip, not a safety check.
    """
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
    # The swap is conditional on Unsloth's prep actually truncating ...
    assert "_unsloth_prep_truncates" in source
    assert "skip_prepare_dataset" in source
    # ... and never re-reads the raw user `max_length` over the resolved one.
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
