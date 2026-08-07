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
    # Padding-free CONCATENATES the batch into one flat sequence, so the collated width is
    # rows x cap, not the cap. Two rows at exactly 2 x 128 is the proof that neither row
    # exceeded it; asserting 128 here would be asserting that padding-free was off.
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

    # Dropping padding-free keeps `max_length` for TRL's collator, and that collator
    # has never truncated on any TRL from 0.22.2 to main: truncation lives only in
    # _prepare_dataset, which returns pre-tokenized rows untouched. So there is no
    # enforcement path left here, and construction must fail rather than run uncapped.
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

    # The train split was tokenized and capped, so the cap is consumed.
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


def _late_overlength_dataset(tok):
    """Row 0 fits, row 3 does not. Only the first row was ever inspected."""
    from datasets import Dataset

    short = tok("hi")["input_ids"]
    long = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(long) > _MODEL_MAX_SEQ_LENGTH
    # Keyed off the row's own text, not its position: `with_transform` is handed
    # whatever slice was asked for, so a batch index says nothing about which
    # row of the dataset it is.
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
    ds = Dataset.from_list([
        {"input_ids": list(ids), "attention_mask": [1] * len(ids), "sample_id": i}
        for i in range(4)
    ])
    return ds.with_format("torch")


def test_a_scalar_column_does_not_defeat_truncation(tmp_path, trl_has_guard):
    """The token columns are truncatable, so the run must not die on the id."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _scalar_torch_formatted_dataset)
    assert _longest(trainer) <= _MODEL_MAX_SEQ_LENGTH


def _mask_supervised_dataset(tok):
    """Supervision carried by `completion_mask`, with the completion at the END.

    `keep_start` truncation cuts it away entirely, leaving an all-zero mask that
    TRL's collator turns into all -100: no supervised token in the row.
    """
    from datasets import Dataset

    prompt = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(prompt) > _MODEL_MAX_SEQ_LENGTH
    completion = tok(" answer")["input_ids"]
    ids = list(prompt) + list(completion)
    mask = [0] * len(prompt) + [1] * len(completion)
    return Dataset.from_list([
        {"input_ids": ids, "attention_mask": [1] * len(ids), "completion_mask": mask}
        for _ in range(4)
    ])


def test_rows_whose_mask_is_truncated_away_are_dropped(tmp_path, trl_has_guard):
    """Same rule the `labels` filter already applies, for the other two spellings."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _mask_supervised_dataset)
    for row in trainer.train_dataset:
        assert any(m != 0 for m in row["completion_mask"]), (
            "a row with no supervised token survived truncation"
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

    # Supervision is carried three ways, and only `labels` was filtered.
    assert "'labels', 'completion_mask', 'assistant_masks'" in block

    # skip_prepare_dataset must not exempt the overlength check.
    assert "if not _unsloth_skip_prepare and not (_unsloth_within_cap" not in block
    assert "if not (_unsloth_within_cap(train_dataset)" in block
