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
    return Dataset.from_list(
        [
            {"input_ids": ids, "attention_mask": [1] * len(ids), "completion_mask": mask}
            for _ in range(4)
        ]
    )


def test_rows_whose_mask_is_truncated_away_are_dropped(tmp_path, trl_has_guard):
    """Same rule the `labels` filter already applies, for the other two spellings."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    # completion_only_loss explicitly, because the collator's mode is what decides
    # whether this mask is supervision at all. TRL resolves a None from the TRAIN
    # sample, and this split has no prompt/completion columns, so the effective mode
    # would be False, the collator would ignore the mask, and filtering on it would
    # be deleting rows that still carry full-sequence supervision.
    trainer = _build(tmp_path, dataset = _mask_supervised_dataset, completion_only_loss = True)
    for row in trainer.train_dataset:
        assert any(
            m != 0 for m in row["completion_mask"]
        ), "a row with no supervised token survived truncation"


def _assistant_mask_dataset(tok):
    """The same shape, supervised by `assistant_masks` instead."""
    from datasets import Dataset

    prompt = tok("The quick brown fox. " * 200)["input_ids"]
    assert len(prompt) > _MODEL_MAX_SEQ_LENGTH
    completion = tok(" answer")["input_ids"]
    ids = list(prompt) + list(completion)
    mask = [0] * len(prompt) + [1] * len(completion)
    return Dataset.from_list(
        [
            {"input_ids": ids, "attention_mask": [1] * len(ids), "assistant_masks": mask}
            for _ in range(4)
        ]
    )


def test_assistant_masks_are_filtered_even_with_the_loss_mode_off(tmp_path, trl_has_guard):
    """The two masks are not gated alike, and gating both on their flag was
    wrong. DataCollatorForLanguageModeling applies `assistant_masks` on presence
    alone -- only `completion_mask` is behind `completion_only_loss` -- so with
    `assistant_only_loss` at its default False an all-zero mask still becomes
    all -100 and the row carries no supervised token."""
    if not trl_has_guard:
        pytest.skip("no guard in this TRL: the block under test is not generated at all")
    trainer = _build(tmp_path, dataset = _assistant_mask_dataset, assistant_only_loss = False)
    for row in trainer.train_dataset:
        assert any(
            m != 0 for m in row["assistant_masks"]
        ), "a row TRL will label all -100 survived truncation"


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

    # Supervision is carried three ways and only `labels` was filtered. The two
    # masks are gated the way TRL's collator gates them, which is not the same
    # for both: `completion_mask` behind the RESOLVED `completion_only_loss`,
    # `assistant_masks` on presence alone.
    assert "'assistant_masks' in _unsloth_cols" in block
    assert "getattr(args, 'assistant_only_loss'" not in block

    # A None `completion_only_loss` is resolved from the dataset the way TRL
    # resolves it, not read as "on".
    assert "getattr(args, 'completion_only_loss', None) is not False" not in block
    # Resolved from the TRAIN sample now, not this split's columns, because that is
    # what the collator does and the two have to agree.
    assert "'prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample" in block

    # The masks apply onto the same labels one after another, so a row survives only
    # where they all agree -- and `labels` is in that intersection, not a filter of its
    # own: a row supervised at one position and masked in at another passes two separate
    # filters and still goes out all -100.
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


def _stub_trainer_class():
    """A minimal class carrying `evaluate`, wrapped the way rl.py wraps it."""
    from unsloth.models.rl import _wrap_sft_evaluate_cap

    seen = {}

    class Stub:
        def evaluate(
            self,
            eval_dataset = None,
            **kw,
        ):
            seen["ds"] = eval_dataset

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


def test_evaluate_leaves_a_split_alone_when_trl_still_holds_the_cap():
    """`max_length` set means TRL truncates in its own prep; capping here too
    would be a second, invisible truncation on a path that already works."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _stub_trainer_class()
    stub = Stub()
    stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, _MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = late)
    assert seen["ds"] is late


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
    i = block.index("_unsloth_completion_only")
    window = block[i : i + 600]
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
    # And it drops the enforcement claim rather than the split: packing needs
    # `max_length`, so clearing it would make TRL raise instead.
    assert "if _unsloth_eval_packing:" in block
    assert "_unsloth_capped = False\\n" in block
    # And the fallback scan must not then raise on the split it just spared.
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


def test_evaluate_leaves_a_split_the_eval_packer_will_take():
    """The late cap has to make the same call as the constructor.

    `evaluate()` re-prepares the split it is given, so under eval packing TRL
    packs it, and every strategy owns the overflow: `wrapped` concatenates the
    stream before chunking, `bfd_split` turns an overlength example into more
    chunks. Cutting rows first evaluates on a truncated corpus.
    """
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    for strategy in ("wrapped", "bfd_split"):
        Stub, seen = _stub_trainer_class()
        stub = Stub()
        stub.args = _Args(_MODEL_MAX_SEQ_LENGTH, None)
        stub.args.eval_packing = True
        stub.args.packing_strategy = strategy
        stub.evaluate(eval_dataset = late)
        assert seen["ds"] is late, f"{strategy}: the packer's split was truncated first"


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


# --- round 6: the late cap's four gaps ---------------------------------------


def _late_cap_helpers():
    """`_cap`, `_CappedRows` and friends out of the closure `_wrap_sft_evaluate_cap`
    builds them in, by wrapping a stub and reading what it captured."""
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
        eval_packing = None,
        packing = False,
    ):
        self.max_seq_length = cap
        self.max_length = max_length
        self.eval_packing = eval_packing
        self.packing = packing
        self.completion_only_loss = None
        self.assistant_only_loss = False
        self.truncation_mode = "keep_start"


def test_a_streaming_late_split_stays_iterable_style():
    """The read-side wrapper is map-style: it defines `__len__`/`__getitem__`,
    and `isinstance(it, IterableDataset)` is False, so Trainer picks a map-style
    sampler and asks a stream for a length it cannot give -- raising before one
    capped row is yielded."""
    import torch.utils.data as tud

    long_row = list(range(_MODEL_MAX_SEQ_LENGTH * 2))

    class _Stream(tud.IterableDataset):
        column_names = ["input_ids", "attention_mask"]

        def __iter__(self):
            for _ in range(3):
                yield {"input_ids": list(long_row), "attention_mask": [1] * len(long_row)}

    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = _Stream())

    got = seen["ds"]
    assert isinstance(got, tud.IterableDataset), "the cap turned a stream into a map-style dataset"
    rows = list(got)
    assert rows and all(len(r["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH for r in rows)


def test_predict_is_capped_even_when_eval_packing_is_on():
    """`predict` comes from the base Trainer and never runs TRL's eval-packing
    prep, so deferring to a packer that will not run leaves the split neither
    packed nor capped."""
    _, tok = _load_plain()
    late = _tokenized_dataset(tok)
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH, eval_packing = True)

    stub.predict(test_dataset = late)
    assert max(len(r) for r in seen["ds"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH

    # evaluate keeps deferring: TRL's packer really does own that split.
    stub.evaluate(eval_dataset = late)
    assert seen["ds"] is late


def test_evaluate_caps_a_split_installed_on_the_trainer():
    """`evaluate()` with no argument falls back to `self.eval_dataset`, which
    the construction-time cap never saw either."""
    _, tok = _load_plain()
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    stub.eval_dataset = _tokenized_dataset(tok)

    stub.evaluate()
    assert seen["ds"] is not None, "evaluate() was handed nothing to cap"
    assert max(len(r) for r in seen["ds"]["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH


def test_an_already_short_late_split_still_drops_unsupervised_rows():
    """Nothing to truncate, but `args.max_length` is None on this path so TRL's
    own post-truncation filter does not run: a row that arrived already all
    -100 reaches the collator and reports a NaN loss."""
    from datasets import Dataset

    ids = list(range(8))
    good = {"input_ids": ids, "attention_mask": [1] * 8, "labels": list(ids)}
    dead = {"input_ids": ids, "attention_mask": [1] * 8, "labels": [-100] * 8}
    short = Dataset.from_list([dict(good), dict(dead), dict(good)])
    assert max(len(r) for r in short["input_ids"]) <= _MODEL_MAX_SEQ_LENGTH

    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = short)

    got = seen["ds"]
    assert len(got) == 2, "the all -100 row survived"
    for row in got:
        assert any(x != -100 for x in row["labels"])


def test_an_already_short_split_with_nothing_to_drop_is_handed_back_as_is():
    """The filter must not cost a rewrite when there is nothing to remove."""
    from datasets import Dataset

    ids = list(range(8))
    short = Dataset.from_list([{"input_ids": ids, "attention_mask": [1] * 8}] * 3)
    Stub, seen = _late_cap_helpers()
    stub = Stub()
    stub.args = _EvalArgs(_MODEL_MAX_SEQ_LENGTH)
    stub.evaluate(eval_dataset = short)
    assert seen["ds"] is short
