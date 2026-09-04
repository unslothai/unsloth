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


@pytest.fixture(scope = "module", autouse = True)
def _cpu_only_torch():
    """Hold the eager-compile spoof for this module only.

    `torch.compile` and `torch.accelerator.is_available` are process-wide, and
    nothing here needs them before the module's own tests start, so scope them:
    left in place at import time they follow the session into any GPU test
    collected after this file.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "compile", _eager_compile)
        # torch.accelerator only exists from torch 2.6 onwards.
        if hasattr(torch, "accelerator"):
            mp.setattr(torch.accelerator, "is_available", lambda *a, **k: False)
        yield mp


_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# Model-level length the trainer resolves from when the user names none.
_MODEL_MAX_SEQ_LENGTH = 128
# Smaller than the model cap, so honouring it is visible.
_USER_MAX_LENGTH = 64


@pytest.fixture(scope = "module", autouse = True)
def patched_sft(_cpu_only_torch):
    """Import unsloth, and force both halves of the SFT patch on.

    `UNSLOTH_ALLOW_CPU=1` (which CPU-only CI sets) skips both, so ask explicitly,
    in `_gpu_init`'s order: the codegen swaps `trl.SFTTrainer` out wholesale, so
    the `__init__` wrapper has to go on afterwards. Both are no-ops once applied.
    """
    global torch  # the `import torch._dynamo` below would otherwise shadow it
    import unsloth  # noqa: F401

    # Through the module's MonkeyPatch: `import unsloth` reinstalls the real torch.compile over the passthrough, and
    # dynamo's kill switch is global too.
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
    # Padding-free CONCATENATES the batch into one flat sequence, so the collated width is rows x cap, not the cap. Two
    # rows at exactly 2 x 128 is the proof that neither row exceeded it; asserting 128 here would be asserting that
    # padding-free was off.
    assert _collated_width(trainer) == 2 * _MODEL_MAX_SEQ_LENGTH


def test_a_with_transform_dataset_keeps_its_cap(tmp_path, trl_has_guard):
    """A transform recreates rows on read, so map() cannot cap them.

    If the BACKING table carries input_ids the schema check says "tokenized", but
    Dataset.map writes that table while the transform keeps handing back the original
    overlength rows. Clearing max_length there would train uncapped with padding-free
    still on, which is the worst of both.
    """
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    from datasets import Dataset

    def _transformed(tok):
        ids = tok("The quick brown fox. " * 200)["input_ids"]
        assert len(ids) > _MODEL_MAX_SEQ_LENGTH
        base = Dataset.from_list([{"input_ids": list(ids), "attention_mask": [1] * len(ids)}] * 4)
        # Backing schema HAS input_ids, and the transform re-inflates every row on read.
        return base.with_transform(
            lambda batch: {
                "input_ids": [list(ids)] * len(batch["input_ids"]),
                "attention_mask": [[1] * len(ids)] * len(batch["input_ids"]),
            }
        )

    # Dropping padding-free keeps `max_length` for TRL's collator, and that collator has never truncated on any TRL from
    # 0.22.2 to main: truncation lives only in _prepare_dataset, which returns pre-tokenized rows untouched.
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _transformed)


def test_a_raw_eval_split_is_left_for_the_tokenizer(tmp_path, trl_has_guard):
    """A raw conversational eval split must not be sliced as if it were tokens.

    `messages: list[dict]` is a per-row sequence too, so a blanket truncation would cut
    conversation turns off the end and silently corrupt evaluation.
    """
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    from datasets import Dataset

    text = "The quick brown fox. " * 200
    raw = Dataset.from_list([{"text": text}] * 4)
    trainer = _build(tmp_path, dataset = _tokenized_dataset, eval_dataset = {"validation": raw})

    assert trainer.args.max_length is None
    split = trainer.eval_dataset["validation"]
    # The raw split is untouched: no truncation ran over it, so the tokenizer pass that
    # follows sees exactly what the user passed. A blanket map would have sliced it.
    if "text" in (split.column_names or []):
        assert all(r["text"] == text for r in split), "a raw column was sliced"


def test_a_torch_formatted_dataset_is_still_truncated(tmp_path, trl_has_guard):
    """A set_format dataset hands batched map() tensors, not lists.

    `if _col` on a tensor raises on the ambiguous truth value, and an isinstance(list)
    check simply leaves the column alone; either way the rows stay long while the cap
    is cleared, which is the exact failure this branch exists to prevent.
    """
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")

    def _formatted(tok):
        ds = _tokenized_dataset(tok)
        ds.set_format("torch")
        return ds

    trainer = _build(tmp_path, dataset = _formatted)
    # The train split was tokenized and capped, so the cap is consumed.
    assert trainer.args.max_length is None
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH, "formatted rows were not truncated"


def test_every_named_eval_split_is_truncated(tmp_path, trl_has_guard):
    """TRL accepts a dict of named eval splits, and a dict has no .map of its own.

    Skipping them while still clearing max_length would leave evaluation running at the
    full length the cap was meant to stop.
    """
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
    """`skip_prepare_dataset` means the user asked for the dataset to be left
    alone, so the rows are not touched: drop padding-free and keep the cap for
    the collator.

    On rows that already fit. This used to pass an OVERLENGTH split, which is
    the one configuration nothing downstream can enforce -- TRL skips both its
    truncation and its collator truncation length -- so it now raises, and
    `test_skip_prepare_dataset_does_not_excuse_an_overlength_row` covers that.
    Asserting both here would have been asserting two opposite things.
    """
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
    # The rows themselves are untouched: the user owns preparation here.
    assert _longest(trainer) <= _MODEL_MAX_SEQ_LENGTH


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


def test_transformed_datasets_are_refused_rather_than_run_uncapped(tmp_path, trl_has_guard):
    """An on-access tokenizing transform cannot be truncated, and nothing below
    would enforce the cap either, so this is a hard error and not a warning.

    Before this handshake existed TRL's own guard already refused the same
    configuration; turning padding-free off must not quietly turn that into a
    run that trains on rows longer than the user asked for."""
    if not trl_has_guard:
        # No guard in this TRL: the block is never generated, so nothing changes.
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
    """Not refused, but not counted as enforcement either.

    A `with_transform` split rebuilds its rows on every read, so sitting under
    the cap once proves nothing about the next read. The train-side test above
    already keeps `max_length` for exactly that reason; treating the eval side
    as enforced was the inconsistency, and it consumed the cap while leaving
    padding-free on with nothing downstream to truncate."""
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
    """TRL's own SFTConfig, not Unsloth's generated subclass of it.

    `PatchFastRL` rebinds `trl.SFTConfig` to `UnslothSFTConfig`, which re-adds a
    `max_seq_length` field no TRL from 0.22.2 to 1.9.2 declares. A caller who
    imported SFTConfig before `import unsloth` still passes the pristine class.
    """
    # Go by the marker rather than the name: the generated subclass is renamed onto TRL's
    # own name so that instances of it keep pickling, so `Unsloth` need not appear in
    # `__name__` at all; the name check below only catches whatever kept the old name.
    from trl import SFTConfig

    cls = SFTConfig
    while "_unsloth_patched_rl_config" in cls.__dict__ or cls.__name__.startswith("Unsloth"):
        cls = cls.__bases__[0]
    return cls


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


def _late_overlength_dataset(tok):
    """Row 0 fits, row 3 does not. Only the first row was ever inspected."""
    from datasets import Dataset

    short = tok("hi")["input_ids"]
    long = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(long) > _MODEL_MAX_SEQ_LENGTH
    # Keyed off the row's own text, not its position: `with_transform` is handed whatever slice was asked for, so a
    # batch index says nothing about which row of the dataset it is.
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


# --- what the fourth review round found -------------------------------------


def test_a_raw_train_split_does_not_excuse_a_tokenized_eval_split(tmp_path, trl_has_guard):
    """`_unsloth_prep_truncates` is decided from the train split. A raw train set
    beside a pre-tokenized eval set skipped the whole truncation block, cleared
    `max_length`, and left evaluation uncapped: preparation does not re-tokenize
    rows that already carry `input_ids`."""
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
    """`IterableDataset.map` takes no `num_proc`, so forwarding the auto-sized one
    raised TypeError, the catch restored the stream, and construction died on
    `cannot be enforced`. The lazy map also caps every row the stream will ever
    yield, which the 1024-row prefix scan could not promise."""
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
        # No column_names, so `_unsloth_truncatable` refuses to rewrite it.
        stream._unsloth_hide_columns = True
        type(stream).column_names = property(lambda self: None)
        return stream

    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(tmp_path, dataset = _opaque_stream)


# --- what the fifth review round found ---------------------------------------


def test_keep_end_truncation_keeps_the_end(tmp_path, trl_has_guard):
    """TRL slices `[-max_length:]` for `truncation_mode = 'keep_end'`, which is
    what callers use when the completion sits at the tail of a long prompt.
    Always keeping the prefix while consuming `max_length` trained on the wrong
    half of every row, with nothing downstream to correct it."""
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
    """`seq_lengths` holds document lengths, not tokens. Slicing it by the cap
    left it describing the pre-truncation row, so padding-free built position
    ids for more tokens than `input_ids` still held. TRL skips truncation under
    packing for the same reason, so the split is refused rather than cut."""
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
        # Only the tail carries labels, so keep_start truncation masks it away.
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


# --- what the third review round found ---------------------------------------


def _scalar_torch_formatted_dataset(tok):
    """A tokenized split with a scalar id column, read as torch tensors.

    Batched, that column is a 1-D tensor whose elements are 0-dimensional. A
    0-dim tensor HAS `__len__` and raises on it, so the column read as a
    sequence and the later `len()` threw.
    """
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
    """One row supervised by `mask_column`, with the completion at the END.

    `long` puts the completion past the cap, so `keep_start` truncation cuts it
    away and leaves an all-zero mask that TRL's collator turns into all -100: no
    supervised token left. A short row keeps its completion and must survive.
    """
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
    """Two rows the truncation strips of supervision, two it leaves alone.

    A mix on purpose: dropping EVERY row leaves an empty split, which is its own
    error (see `test_a_cap_below_all_supervision_is_a_clear_error`), so a split
    that still has something to train on is what pins the filter itself.
    """
    from datasets import Dataset
    return Dataset.from_list(
        [_mask_row(tok, "completion_mask", long) for long in (True, True, False, False)]
    )


def test_rows_whose_mask_is_truncated_away_are_dropped(tmp_path, trl_has_guard):
    """Same rule the `labels` filter already applies, for the other two spellings."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    # completion_only_loss explicitly, because the collator's mode is what decides whether this mask is supervision
    # at all. TRL resolves a None from the TRAIN sample, and this split has no prompt/completion columns, so the
    # effective mode would be False, the collator would ignore the mask, and filtering on it would be deleting rows
    # that still carry full-sequence supervision.
    trainer = _build(tmp_path, dataset = _mask_supervised_dataset, completion_only_loss = True)
    assert len(trainer.train_dataset) == 2, "the rows that kept their completion were dropped too"
    for row in trainer.train_dataset:
        assert any(
            m != 0 for m in row["completion_mask"]
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
    """The two masks are not gated alike, and gating both on their flag was
    wrong. DataCollatorForLanguageModeling applies `assistant_masks` on presence
    alone -- only `completion_mask` is behind `completion_only_loss` -- so with
    `assistant_only_loss` at its default False an all-zero mask still becomes
    all -100 and the row carries no supervised token."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _assistant_mask_dataset, assistant_only_loss = False)
    assert len(trainer.train_dataset) == 2, "the rows that kept their completion were dropped too"
    for row in trainer.train_dataset:
        assert any(
            m != 0 for m in row["assistant_masks"]
        ), "a row TRL will label all -100 survived truncation"


def test_a_cap_below_all_supervision_is_a_clear_error(tmp_path, trl_has_guard):
    """An emptied split must not be handed onwards.

    Every TRL 1.x resolves `completion_only_loss` and `_is_vision_dataset` from
    `next(iter(train_dataset))` in `__init__`, so a split the supervision filter
    emptied came back out as a bare `StopIteration` naming nothing the caller
    could act on. Say what happened and what to change instead.
    """
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

    # Supervision is carried three ways and only `labels` was filtered.
    assert "'assistant_masks' in _unsloth_cols" in block
    assert "getattr(args, 'assistant_only_loss'" not in block

    # A None `completion_only_loss` is resolved from the dataset the way TRL resolves it, not read as "on".
    assert "getattr(args, 'completion_only_loss', None) is not False" not in block
    # Resolved from the TRAIN sample now, not this split's columns, because that is what the collator does and the two
    # have to agree.
    assert "'prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample" in block

    # The masks apply onto the same labels one after another, so a row survives only where they all agree, and `labels`
    assert (
        "_unsloth_supervision = (['labels'] if 'labels' in _unsloth_cols else []) + _unsloth_masks"
        in block
    )
    assert (
        "any(all((_x != -100) if _n == 'labels' else _x for _n, _x in zip(_c, _v)) "
        "for _v in zip(*[_e[_n] for _n in _c]))" in block
    )

    # skip_prepare_dataset must not exempt the overlength check.
    assert "if not _unsloth_skip_prepare and not (_unsloth_within_cap" not in block
    assert "if not (_unsloth_within_cap(train_dataset)" in block


def _stub_trainer_class(prepares_late = False):
    """A minimal class carrying `evaluate`, wrapped the way rl.py wraps it.

    `prepares_late` picks which side of TRL 1.7.0 the stub imitates, since that
    is what `_trl_prepares_late_evals` reads off the class: up to 1.6 `evaluate`
    was the base Trainer's and never prepared anything, from 1.7.0 it calls
    `_prepare_dataset` on a split passed straight to it.
    """
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
                # The 1.7.0 shape, and the string the probe looks for.
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
    """The init-time splits are capped and `max_length` is then cleared, so a
    pre-tokenized split passed to `evaluate()` afterwards had nothing enforcing
    the cap: TRL prepares it with `max_length = None` and Zoo's prep leaves rows
    that already carry `input_ids` alone. Version-independent: the wrapper is
    exercised directly, since the surrounding branch needs a guarded TRL."""
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
    # The per-token sidecars move with `input_ids`, or the mask stops lining up.
    assert all(len(a) == len(i) for a, i in zip(got["attention_mask"], got["input_ids"]))


def test_a_retained_max_length_does_not_excuse_a_late_split():
    """This test used to assert the opposite, on the premise that `max_length`
    being set means TRL truncates in its own prep. That premise does not hold
    for the path this wrapper exists for: `_prepare_dataset` runs only from
    `__init__`, so a split handed to `evaluate()` afterwards is never prepared,
    and TRL's collator does not truncate rows that already carry `input_ids`.
    A retained `max_length` is also exactly what the construction block leaves
    behind when it turns padding-free OFF, so reading it as proof of
    enforcement let an overlength late split reach the model uncapped."""
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
    """TRL resolves `None` from the dataset shape:

        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample

    (sft_trainer.py:736). A pre-tokenized split has neither column, so the
    effective mode is False and the collator ignores `completion_mask`
    entirely. Reading `None` as "on" deleted rows that still had valid
    full-sequence supervision, and could empty the split outright."""
    block = _padding_free_codegen_block()
    # Bounded by the next anchor rather than by a byte count: a comment added inside the block used to push the
    # assertions out of a fixed window.
    i = block.index("_unsloth_completion_only")
    window = block[i : block.index("args._unsloth_completion_only_loss", i)]
    assert "is None" in window
    # From the training sample, which is the sample TRL reads.
    assert "'prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample" in window
    # And published so the late evaluate()/predict() cap uses the same value.
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


# ── the late cap has to agree with the construction-time cap ─────────────────
def test_evaluate_caps_an_iterable_split():
    """A stream reached the collator uncapped, and did so silently.

    `dataset[0]` does not raise on a stream: `datasets` 4.x reads the 0 as a
    COLUMN name and hands back an `IterableColumn`, whose `len()` then threw
    TypeError into the wrapper's catch, which returns the original. So neither
    this wrapper nor TRL enforced anything, which is the whole failure this
    wrapper exists to prevent.
    """
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
    """TRL slices [-max_length:] for `keep_end`, and so does the cap at init.

    Always keeping the prefix evaluates the wrong half of every long row for the
    callers who set this, which is exactly the ones whose completion sits at the
    tail of a long prompt.
    """
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
    """Cutting a long prompt can leave every label at -100.

    TRL filters those right after its own truncation, but `args.max_length` is
    None on this path so TRL does not run, and a batch made of such rows has no
    supervised token at all: a NaN loss rather than a small one.
    """
    from datasets import Dataset

    _, tok = _load_plain()
    ids = tok("The quick brown fox. " * 200)["input_ids"]
    cap = _MODEL_MAX_SEQ_LENGTH
    assert len(ids) > cap
    # One row supervised only past the cap, one supervised from the start.
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
    """`seq_lengths` describes documents, not tokens.

    Slicing `input_ids` under a `seq_lengths` that still describes the longer row
    makes padding-free build position ids for tokens the row no longer has. The
    construction-time cap refuses this shape for the same reason.
    """
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
    """`eval_packing` is resolved separately from `packing`.

    The branch is gated on `not args.packing`, so packing = False with
    eval_packing = True still reaches it, and TRL then PACKS that eval split
    rather than truncating it. Wrapped packing concatenates the whole token
    stream before chunking, so cutting each row at the cap first evaluates on a
    truncated corpus.
    """
    import inspect

    from unsloth.models import rl

    block = inspect.getsource(rl)
    assert (
        "_unsloth_eval_packing = getattr(args, 'packing', False) if getattr(args, 'eval_packing', None) is None else getattr(args, 'eval_packing')"
        in block
    )
    # And it drops the enforcement claim rather than the split: packing needs `max_length`,
    # so clearing it would make TRL raise instead.
    # `or not _unsloth_known_mode`: a mode this cannot honour spares the split too.
    assert "if _unsloth_eval_packing or not _unsloth_known_mode:" in block
    assert "_unsloth_capped = False\\n" in block
    assert (
        "_unsloth_scan_eval = None if _unsloth_eval_packing else "
        "(eval_dataset if 'eval_dataset' in locals() else None)" in block
    )


# ── round five: what the eval packer owns, and what supervision means ────────
def test_the_codegen_does_not_raise_on_a_split_it_left_to_the_packer():
    """Sparing the split and then scanning it is a hard error, not a fallback.

    Leaving an eval split for TRL's packer sets `_unsloth_capped = False`, which
    drops through to the branch that scans every split and raises on an
    overlength row. That split is overlength ON PURPOSE, so the scan turned a
    working eval-packing run into a ValueError and denied `wrapped` and
    `bfd_split` the overflow they exist to handle.
    """
    block = _padding_free_codegen_block()
    # And the fallback scan must not then raise on the split it just spared.
    assert (
        "_unsloth_scan_eval = None if _unsloth_eval_packing else "
        "(eval_dataset if 'eval_dataset' in locals() else None)" in block
    )
    assert "_unsloth_splits_within_cap(_unsloth_scan_eval)" in block
    # The train split is still scanned: nothing packs that one.
    assert "_unsloth_within_cap(train_dataset) and" in block
    # And it is resolved outside the truncation block, which skip_prepare_dataset skips.
    packing_at = block.index("_unsloth_eval_packing = getattr(args, 'packing'")
    skip_at = block.index("if not _unsloth_skip_prepare:")
    assert packing_at < skip_at, "the fallback reads it even when that block is skipped"


def test_the_codegen_refuses_an_unknown_truncation_mode():
    """keep_start and keep_end are the only two slices there are.

    TRL's SFT path never reads `truncation_mode` at all (it belongs to the
    preference trainers), so nothing downstream would catch a third value and
    mapping it to the default silently cuts from the side the caller asked us
    not to. Drop the enforcement claim instead.
    """
    block = _padding_free_codegen_block()
    assert "_unsloth_known_mode = _unsloth_truncation_mode in ('keep_start', 'keep_end')" in block
    assert "_unsloth_capped = _unsloth_known_mode" in block


def _trl_sft_late_hooks():
    """Which of the late entry points TRL's own SFTTrainer defines, and which of
    its methods call `_prepare_dataset`.

    Read from the source FILE, because by now unsloth has patched the live class
    with this very wrapper, among others.
    """
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
    """Who owns a late eval split changed in TRL 1.7.0, so this follows the TRL.

    Up to 1.6 no packer ever saw a split handed over after construction:
    `_prepare_dataset` ran from `__init__` and nowhere else, and SFTTrainer
    overrode none of the late entry points, so skipping the cap under
    `eval_packing` handed the collator raw overlength rows with `max_length`
    already cleared. From 1.7.0 `SFTTrainer.evaluate` prepares a split passed
    straight to it, packing included, and every strategy redistributes the
    overflow, so cutting rows at the cap first throws those tokens away and the
    packer has to keep the split instead.
    """
    _, tok = _load_plain()

    # A fresh split per case on purpose: a skipped cut MARKS the caller's own object as capped, which is what stops the
    # paired `get_eval_dataloader` wrapper cutting it seconds later, so reusing one split across cases would measure
    # that mark instead of the decision under test.
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

    # Both sides of the change, deterministically, so this pins the wrapper's logic rather than whichever TRL happens to
    # be installed.
    for strategy in ("wrapped", "bfd_split"):
        assert _run(True, True, strategy) > _MODEL_MAX_SEQ_LENGTH, (
            f"{strategy}: the split was cut at the cap before TRL's packer "
            "could redistribute the overflow"
        )
        assert (
            _run(False, True, strategy) <= _MODEL_MAX_SEQ_LENGTH
        ), f"{strategy}: an uncapped split reached the collator"

    # With `eval_packing` off, the cap applies on both sides: nothing packs it.
    for prepares_late in (False, True):
        assert _run(prepares_late, False) <= _MODEL_MAX_SEQ_LENGTH


def _packing_aware_stub():
    """The two real classes around the wrapper, in their actual shapes.

    `packs_late` is read off the CLASS, but TRL only prepares a split that was
    passed to `evaluate`:

        if not self._skip_prepare_dataset and eval_dataset is not None
           and not isinstance(eval_dataset, str):

    (trl 1.9.2 sft_trainer.py:1675, unchanged since 1.7.0). Everything else --
    a split stored on the trainer, a string key, a `skip_prepare_dataset` run --
    goes straight to `Trainer.evaluate`, which hands that same object to
    `self.get_eval_dataloader(eval_dataset)` (transformers 4.57.6
    trainer.py:4467 and 4481). That builder is the entry point that caps.
    """
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Base:  # transformers.Trainer
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

    class Stub(Base):  # trl >= 1.7 SFTTrainer
        def _prepare_dataset(self, dataset, *a, **kw):
            seen["prepared"] = dataset
            return dataset.map(lambda e: e)  # packing always yields a NEW object

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
    """Deferring to the packer must not MARK the split as capped.

    `evaluate()` with nothing passed, and `evaluate("name")`, never reach TRL's
    prep: its guard needs `eval_dataset is not None and not isinstance(..., str)`.
    The stored split then arrives at `get_eval_dataloader` as the very object
    `evaluate` marked, and `_cap_still_holds` read that mark and handed the
    collator the overlength rows -- with `max_length` cleared and nothing else
    truncating, on a plain `SFTConfig(packing = True)`.
    """
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

    # Stored split, `evaluate()` with no argument. TRL never prepares it.
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

    # A named split: `evaluate("name")` is excluded from TRL's prep by name.
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

    # `skip_prepare_dataset`: the split IS passed, and TRL still never packs it.
    seen = _run(
        lambda s: s.evaluate(eval_dataset = s.eval_dataset),
        eval_packing = True,
        skip_prepare = True,
    )
    assert "prepared" not in seen
    assert (
        max(len(r) for r in seen["dataloader"]["input_ids"]) <= cap
    ), "skip_prepare_dataset + eval_packing let an overlength split through"

    # The control, and the deferral this branch exists for: when TRL really does
    # prepare the split, the packer must still receive the FULL rows.
    seen = _run(lambda s: s.evaluate(eval_dataset = s.eval_dataset), eval_packing = True)
    assert (
        max(len(r) for r in seen["prepared"]["input_ids"]) > cap
    ), "the split was cut before TRL's packer could redistribute the overflow"


def test_the_installed_trl_is_on_the_side_of_1_7_0_that_its_version_says():
    """The source-level half: which shape the TRL actually installed here has."""
    from packaging.version import Version

    import trl

    late_hooks, prepares = _trl_sft_late_hooks()
    # Whichever side of the change this TRL is on, `_prepare_dataset` is reached only from `__init__` and from
    # `evaluate`. A third caller would be a third path for the late cap to audit.
    assert set(prepares) <= {"__init__", "evaluate"}, prepares
    # `predict` and the two dataloader builders are the base Trainer's on every TRL, which is why only `evaluate` is
    # ever given `packs_late`.
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
    """A label and a mask lighting up in DIFFERENT positions is still all -100.

    The collator applies the masks onto the labels, so filtering each on its own
    keeps exactly the row that ends up with no supervised token.
    """
    from datasets import Dataset

    cap = _MODEL_MAX_SEQ_LENGTH
    length = cap + 8
    # Supervised label at position 0, assistant mask on at position 1: each filter
    # on its own says "keep", the intersection says the row is empty.
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
    """TRL resolves the mode ONCE, from the training sample.

        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample

    So prompt/completion training data makes the collator apply
    `completion_mask` to a pre-tokenized eval split that carries neither column.
    Resolving per split read that as full-sequence loss and kept rows whose mask
    truncated to all zeros.
    """
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
    # What the trainer resolved from the TRAIN split, which this eval split cannot see.
    stub.args._unsloth_completion_only_loss = True
    stub.evaluate(eval_dataset = late)
    assert len(seen["ds"]) == 0, "the mask truncated to all zeros, so the row has no supervision"

    # Without the trainer's answer this split alone reads as full-sequence loss.
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(cap, None)
    stub.evaluate(eval_dataset = late)
    assert len(seen["ds"]) == 1


def test_evaluate_caps_a_split_that_carries_no_column_metadata():
    """A custom map-style split has no `column_names`.

    Reading that as "raw text, prep will tokenize it" left a pre-tokenized split
    uncapped on a path where `args.max_length` is already None, so nothing else
    cut it either.
    """
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
    """A `torch.utils.data.Dataset` or a plain list has no `.map()`.

    Calling it raised AttributeError into the broad catch, which returns the
    original, so the split reached the collator uncapped on a path where
    `args.max_length` is already None.
    """
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
    # Indexing is the other way the collator reads it.
    assert len(got[0]["input_ids"]) <= cap


def test_evaluate_caps_a_with_transform_split():
    """`with_transform` reports its BACKING columns while yielding `input_ids`.

    Trusting `column_names` read the split as raw text, and mapping it would be
    wrong anyway: the rows are rebuilt on every read, so a map writes a table
    nobody reads.
    """
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


# ── round nine: what the wrapper hands back, and what it never sees ──────────
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
    """A DataLoader picks its kind with `isinstance`, not with `hasattr`.

    Trainer skips building a sampler only for an `IterableDataset`, so wrapping
    a stream in a plain object got it a `SequentialSampler`, which asks for the
    `len()` the stream never had. The eval call then died on the wrapper instead
    of yielding the capped rows it was built to yield.
    """
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
    # And the loader the trainer would build reads it without asking for a length.
    loader = torch.utils.data.DataLoader(got, batch_size = None, collate_fn = lambda x: x)
    read = list(loader)
    assert len(read) == 3
    assert all(len(r["input_ids"]) <= cap for r in read)
    assert all(len(r["attention_mask"]) == len(r["input_ids"]) for r in read)


def test_a_short_split_is_still_filtered_for_supervision():
    """Being under the cap is not the same as being supervised.

    A row whose labels are all -100 is a NaN loss whether or not anything had to
    be cut off it, and the construction-time cap filters those unconditionally.
    Returning early on the length alone was the one detail where the late cap
    and the constructor disagreed.
    """
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
    # Two shapes: one the filter runs over and keeps every row of, and one with no supervision column at all, where
    # there is nothing to run.
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
            # What `get_eval_dataloader` resolves a string key to, read DURING
            # the call. A named split is capped for the call and restored after,
            # so the stored dict alone cannot show whether the cap was applied.
            stored = getattr(self, "eval_dataset", None)
            if isinstance(eval_dataset, str) and isinstance(stored, dict):
                seen["resolved"] = stored.get(eval_dataset)

    _wrap_sft_evaluate_cap(Stub)
    return Stub, seen


def test_evaluate_caps_the_split_stored_on_the_trainer():
    """`evaluate()` with nothing passed reads `self.eval_dataset`.

    A caller can install or replace that split after construction, which is
    exactly where the constructor's cap can no longer see it, and `max_length`
    is already cleared by then.
    """
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
    # Swapped in for the call only:
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
    """`evaluate()` runs at every eval step of a training run.

    The scan that decides whether anything needs cutting materialises the whole
    `input_ids` column, so doing it again for every eval turns the stored-split
    fallback into a per-eval pass over the dataset.
    """
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


# --- round 7: pickling, mutation, predict's contract, single-pass probes -----


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

    # evaluate still drops it: a loss over an all -100 row is meaningless.
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


# ── round nine: the refusal, the required rewrite, and the one-shot stream ────
def test_capping_the_train_split_cannot_undo_the_unknown_mode_refusal():
    """`_unsloth_capped` is seeded from the mode and then had to survive.

    The eval branches combine with `and`; the train branch assigned over the
    seed, so a `truncation_mode` that is neither keep_start nor keep_end printed
    "not being enforced" and was then served as keep_start anyway, with
    `max_length` cleared and padding-free left on.
    """
    block = _padding_free_codegen_block()
    assert "train_dataset, _unsloth_split_ok = _unsloth_cap_split(train_dataset)" in block
    assert "_unsloth_capped = _unsloth_capped and _unsloth_split_ok" in block
    assert "train_dataset, _unsloth_capped = _unsloth_cap_split" not in block


def test_the_scan_refuses_a_single_pass_stream_instead_of_draining_it():
    """Reading a one-shot stream to check it IS consuming it.

    The trainer would then receive an exhausted split, or one short by the whole
    scanned prefix. Two `iter()` calls returning the same object is what marks
    one, and the answer is the same as for an unfinished prefix: not proven.
    """
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
    # What the generated helper does, in the same order.
    probe = iter(rows)
    assert probe is iter(rows), "a generator is its own iterator"
    assert scanned["rows"] == 0, "the guard must not read a row"
    assert [r["input_ids"] for r in rows] == [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]


def test_the_max_length_seed_rewrite_is_required():
    """Not the optional worker-count edit this helper was written for.

    The generated trainer clears `args.max_length`, so an unrewritten seed reads
    that `None` rather than the `0` that makes the guard fall through, and a raw
    dataset stops being truncated. Both anchors missing has to fail loudly.
    """
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


# ── round ten: the same one-shot signal, everywhere it is probed ─────────────
def _shared_iterator_split(rows):
    """An `IterableDataset` whose `__iter__` hands back one stored generator.

    Not `self`, so `iter(x) is x` misses it, and still single-pass: the object
    a second `iter()` returns is the same one, which is the signal that works.
    """
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
    # And the cheap path is tried first.
    names_at = block.index("getattr(train_dataset, 'column_names', None)")
    probe_at = block.index("_unsloth_probe = iter(train_dataset)")
    assert names_at < probe_at


def test_a_transformed_split_is_not_memoized_by_fingerprint():
    """`_fingerprint` covers the backing table, not the transform.

    A transform closing over mutable state yields different rows under an
    unchanged fingerprint, so the memo replayed a filter decided against the old
    ones: rows kept that should now be dropped, or dropped that should be kept.
    """
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
        # Keyed on the row's own contents: a transform is handed whatever batch
        # the reader asks for, so positions in the batch say nothing about
        # positions in the split.
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

    # Same object, same `_fingerprint`; only the transform's closure moved.
    supervised["both"] = False
    stub.evaluate(eval_dataset = split)
    assert len(seen["ds"]) == 1, "the memo served a cap taken before the change"


def test_evaluate_caps_the_split_a_string_key_names():
    """`evaluate(eval_dataset = "validation")` is the supported way to pick one
    split out of a stored dict: `get_eval_dataloader` resolves it as
    `self.eval_dataset[eval_dataset]`. Capping the KEY is a no-op, so the split
    it names reached the collator over `max_seq_length` with `max_length = None`.

    Swapped in for the call and back out again, never written through: the
    caller keeps its uncapped original, so a later `truncation_mode` change can
    still take the suffix instead of re-capping a saved prefix.
    """
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
    """The whole point of restoring the original. `keep_end` after `keep_start`
    on the same key must produce the SUFFIX, which is impossible if the first
    call left its prefix behind in the stored dict.
    """
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
    """`column_names` describes the BACKING table, so a `with_transform` split
    storing `text` and yielding `input_ids` answered "raw" and the cap was
    cleared for rows preparation then never truncates.

    Raising is how the cap is kept for this shape. `_unsloth_truncatable` refuses
    to rewrite a transform, so padding-free turns off and `max_length` stays for
    TRL's collator -- which does not truncate rows that already carry
    `input_ids`. An overlength row is therefore reported rather than served
    uncapped, and reaching that raise at all is proof the cap was not cleared:
    the scan it comes from only runs on the branch where `max_length` is kept.
    """
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
    # Same reasoning as the transformed split above: the cap is kept, so the
    # overlength rows are reported instead of being served uncapped.
    with pytest.raises(ValueError, match = "cannot be enforced"):
        _build(
            tmp_path,
            dataset = lambda _tok: _SharedIteratorStream(rows),
            padding_free = True,
            max_length = _MODEL_MAX_SEQ_LENGTH,
        )


def test_the_schema_read_distrusts_a_transform_and_an_unprobeable_stream():
    """The source-level half of the two tests above, which only run on a TRL that
    ships the guard. A `with_transform` split must not be believed on its backing
    `column_names`, and a stream that cannot spare a row must refuse rather than
    assume preparation will truncate it."""
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
    """The allow-list treated every token-shaped NAME as sliceable. An optional
    column stored as `token_type_ids = None` made the late cap's `map` raise, and
    the broad catch around it handed the caller its uncapped split straight back.
    """
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
    """`[3, seq_len]`, which is what `position_ids` is under mrope. Its first
    axis is the channel axis, so cutting there would drop whole channels. The
    length test is what tells the two apart: only the token-major layout is as
    long as `input_ids`."""
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
    # Sliced to the next def, not a fixed window: a comment added inside the probe
    # used to push the line under test out of the window and pass the test blind.
    start = block.index("def _unsloth_pretokenized")
    body = block[start : block.index("def _unsloth_cap_split", start)]
    assert (
        "_probe is _ds or _probe is iter(_ds)" in body
    ), "the pretokenized probe still reads a row off a one-shot stream"
    assert body.index("column_names") < body.index(
        "iter(_ds)"
    ), "the schema is not consulted before a row is taken"


# ── round fourteen: capping once, and not cutting what we cannot honour ──────


def test_capping_a_one_shot_stream_twice_does_not_eat_its_rows():
    """`evaluate()` caps the split and stores it, then Transformers calls
    `get_eval_dataloader`, which this module also wraps -- so one call reaches
    the cap twice. `_CappedStream.__iter__` hands out a fresh generator over the
    same exhausting source rather than rewinding, so the second pass's schema and
    per-token probes read the first rows off instead of replaying them."""
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
            # What Trainer.evaluate does: hand the stored split to the builder.
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
    """Seeding `_unsloth_capped` false only drops the ENFORCEMENT claim. The
    slice still ran, with keep_end false for any unknown value, so the fallback
    scanned an already-trimmed split, found it within the cap and merely turned
    padding-free off -- every row silently cut from the start right after warning
    that the mode could not be honoured."""
    block = _padding_free_codegen_block()
    for guarded in (
        "if _unsloth_known_mode and not _unsloth_prep_truncates:",
        "if _unsloth_eval_packing or not _unsloth_known_mode:",
    ):
        assert guarded in block, f"the split is still rewritten under an unknown mode: {guarded}"
    # And the refusal it protects is still seeded, not assigned over.
    assert "_unsloth_capped = _unsloth_known_mode" in block


def test_the_transform_rule_is_read_by_both_schema_probes():
    """`column_names` describes the BACKING table. The construction-time probe
    already distrusted it for a transform; the late `_unsloth_pretokenized` one
    did not, so a `with_transform` split storing `text` and yielding overlength
    `input_ids` was reported raw, marked safe, and had its cap cleared."""
    block = _padding_free_codegen_block()
    assert (
        block.count("_unsloth_is_transformed(") >= 3
    ), "the rule is not defined once and read by both probes"
    # Sliced to the next def, not a fixed window:
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
    # And held, not cleared, when the yielded rows do fit.
    trainer = _build(
        tmp_path,
        eval_dataset = _transformed_short_dataset(tok),
        padding_free = True,
        max_length = _MODEL_MAX_SEQ_LENGTH,
    )
    assert trainer.args.max_length is not None, "nothing else truncates the yielded rows"


# ── round fifteen: alignment, laziness, the cache key, and unknown modes ─────


def test_a_one_shot_stream_slices_every_aligned_column():
    """`_column_names` already read a row off the stream, and discarding it left
    `_sliceable_per_token` with nothing to measure, so it cut `input_ids` alone
    and left `labels`/`attention_mask` overlength -- supervision that no longer
    lines up with the tokens it describes."""
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
    """Split shapes whose verdict the two cap scans must agree on.

    The generated `__init__` inlines its own copy of the scan because that module
    is standalone; `rl.pretokenized_within_cap` is the importable twin. A drift
    between them is invisible until a real run goes uncapped, so pin them here.
    """
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
        (raw, True),  # not tokenized: prep still truncates it
        ([{"input_ids": [1, 2]}], True),
        ([{"input_ids": [1, 2, 3, 4]}], False),
        (one_shot(), False),  # single-pass: unverifiable reads False
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
    """Run the auto-padding-free retry with an init that rejects padding-free.

    Returns the number of `original_init` calls, or the propagated error.
    """
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
    """The retry only runs when the exact source match in `rl.py` missed TRL's
    guard, which means the truncation block was never generated either. Turning
    padding-free off keeps `max_length` for a collator that does not truncate, so
    retrying would turn a hard error into a silently uncapped run."""
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
    """`with_transform` reports its BACKING table. A transform storing `text` but
    yielding `prompt`/`completion` resolved False here while TRL, reading a
    yielded row, resolved True and applied `completion_mask` -- so the cap
    filters kept rows whose completion had been truncated away entirely."""
    import inspect as _inspect
    from unsloth.models import rl

    source = _inspect.getsource(rl)
    guard = (
        "_unsloth_train_sample = {} if _unsloth_is_transformed(train_dataset) else dict.fromkeys("
    )
    assert guard in source
    # And the probe it falls through to still reads a row rather than giving up.
    assert "_unsloth_train_sample = next(_unsloth_probe, None) or {}" in source


def _row(ids, **extra):
    row = {"input_ids": list(ids), "attention_mask": [1] * len(ids)}
    row.update(extra)
    return row


def test_a_later_row_that_cannot_take_the_slice_does_not_raise():
    """`_sliceable_per_token` probes ONE row. An optional column that is a list
    there and None further in used to raise inside the dataloader -- a failure
    the caller would not have had without the cap. The `map` path already
    validates per row; the read-side wrapper has to as well."""
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

    # Longer than the tokens, not shorter, so cutting it is visible: a shorter
    # value comes back unchanged from the slice either way.
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

    # Worst case spelled out, since a set's own order is stable within a run.
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
    """Three of the four `_cap` outcomes mark the CALLER'S object and hand it
    back. Mutating it -- a `set_transform` that starts yielding longer rows --
    left the mark in place, so the rescan was skipped and the new rows went
    through uncapped."""
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
        # A fresh object each time, as a per-epoch validation subset would be.
        stub.evaluate(eval_dataset = _tokenized_dataset(tok))
    memo = getattr(stub, "_unsloth_eval_cap_memo", {})
    assert 0 < len(memo) <= rl._EVAL_CAP_MEMO_MAX, len(memo)


# ── round nineteen ───────────────────────────────────────────────────────────
def test_a_nullable_value_does_not_break_the_construction_time_truncation():
    """`_unsloth_is_sequence_column` judges the COLUMN from its first row. An
    optional field that is a list there and None further in raised TypeError out
    of `len`, the enclosing handler restored the overlength split, and a run
    that could have been truncated died on "cannot be enforced"."""
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
    """`set_format(columns = [...], output_all_columns = False)` yields only the
    named columns while `column_names` still answers with the whole backing
    table. Reading the table resolved completion-only True off a `completion`
    the rows never hand over, TRL resolved False from a yielded row, and the cap
    then filtered on a mask the collator ignores."""
    block = _padding_free_codegen_block()
    for fragment in (
        "_unsloth_shown = _unsloth_fmt.get('columns')",
        "if _unsloth_fmt.get('output_all_columns') or not _unsloth_shown:",
    ):
        assert fragment in block, fragment
    # Premise: `datasets` really does keep the backing names under a narrowed format.
    from datasets import Dataset

    ds = Dataset.from_list([{"prompt": "a", "completion": "b", "input_ids": [1, 2]}])
    ds.set_format("numpy", columns = ["input_ids"], output_all_columns = False)
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
    # `None` means "whatever packing is", which is TRL's own default.
    config.eval_packing, config.packing = None, True
    assert enforceable(config, long_rows, long_rows)


def test_a_zoo_that_already_normalizes_the_seed_is_left_alone():
    """A newer unsloth_zoo adopting the replacement itself matched neither
    anchor, and `required = True` then failed every SFT trainer over behaviour
    already present. `old` is also a PREFIX of `new` here, so the wide anchor
    matched the normalized line and appended a second `or 0`."""
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
    # And the edit itself still applies to an un-normalized source.
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


# ── round twenty ─────────────────────────────────────────────────────────────
def test_a_single_quoted_normalized_seed_is_recognised():
    """The narrow regex accepts either quote style, so the idempotence check has
    to as well. A Zoo carrying the replacement single-quoted matched neither the
    literal nor the `$`-anchored regex, and `required = True` then raised on
    every SFT trainer over behaviour already present."""
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
    # Premise: neither anchor sees it on its own.
    assert new not in single and not R._ZOO_MAX_LENGTH_SEED.search(single)


def test_the_late_cap_prefers_the_trainers_resolved_completion_mode():
    """The generated block does not always run -- a TRL whose guard did not
    match, or padding-free off from the start -- but TRL has still resolved
    `completion_only_loss` from the training sample. Falling through to the late
    split's own schema read False off one carrying only `input_ids` and
    `completion_mask`, kept rows whose completion was cut away entirely, and the
    collator turned them into all -100."""
    _, tok = _load_plain()
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    # Exactly the state the generated block would have left empty.
    for name in ("_unsloth_completion_only_loss", "completion_only_loss"):
        if hasattr(stub.args, name):
            setattr(stub.args, name, None)
    stub.completion_only_loss = True

    stub.evaluate(eval_dataset = _tokenized_dataset(tok))
    assert getattr(stub.args, "_unsloth_resolved_completion_only") is True

    # And False is an answer too, not an absence.
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
    # The map AND the filter have to be inside it, so the whole body moved.
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
    # Extracted from the generator and executed as written, like the cap scan.
    exec("\n".join(line[8:] for line in lines), scope)
    with scope["_unsloth_rank_first"]():
        pass  # must not raise, whatever accelerate is or is not here
