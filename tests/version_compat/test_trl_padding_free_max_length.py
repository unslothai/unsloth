# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Padding-free + `max_length` handshake across TRL versions.

TRL >= 1.0.0 refuses to build an SFTTrainer when padding-free is on, packing is
off and `args.max_length` is set:

    ValueError: When `padding_free=True` without packing, `max_length` is not
    enforced. Either enable packing ..., or set `max_length=None`.

Unsloth auto-enables padding-free whenever the user leaves `padding_free` at its
`None` default, and rl.py's `max_length_check` codegen always wrote
`args.max_length`, so every default SFT run tripped that guard. rl.py now hands
those TRLs the `None` they ask for and keeps truncating through
`max_seq_length`, which is what Unsloth's own dataset prep reads.

Two things the swap must not break, both pinned below:

* the resolved length is unchanged, so `max_seq_length` still wins over
  `max_length` exactly as it does on TRL < 1.0.0;
* the swap only happens when Unsloth's dataset prep really runs its truncating
  tokenize pass. For an already-tokenized dataset, or for
  `dataset_kwargs={"skip_prepare_dataset": True}`, nothing would truncate, so
  padding-free is turned off and `max_length` kept instead.

The assertions branch on whether the installed TRL actually carries the guard,
so the same file pins the new behaviour on TRL >= 1.0.0 and the untouched old
behaviour on TRL < 1.0.0.
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

# Apply the CUDA spoof before any unsloth-touching import, so a CPU-only runner
# can still import unsloth and generate the patched trainer.
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
# Deliberately smaller than _MODEL_MAX_SEQ_LENGTH, so honouring it is visible.
_USER_MAX_LENGTH = 64


@pytest.fixture(scope = "module", autouse = True)
def patched_sft():
    """Import unsloth, and make sure the SFT trainer really did get patched.

    `import unsloth` alone is not enough: under `UNSLOTH_ALLOW_CPU=1` (which
    CPU-only CI sets so unsloth imports without a GPU) both halves of the SFT
    patch are deliberately skipped, so drift detectors keep seeing the pristine
    upstream TRL classes -- `rl.PatchFastRL` returns before
    `patch_trl_rl_trainers`, and `_gpu_init` guards `_patch_trl_trainer`, which
    installs the `__init__` wrapper that resolves packing / padding-free. These
    tests are about the patched trainer, so ask for both explicitly, in
    `_gpu_init`'s order: the codegen swaps `trl.SFTTrainer` out wholesale, so
    the wrapper has to go on afterwards or it is thrown away. Both are
    no-ops once applied, so this stays inert wherever the import did the work.
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
    """The resolved truncation length does not depend on the TRL version.

    Unsloth's precedence (documented by the `max_length_check` codegen: the
    model / args `max_seq_length` wins) already resolved `max_length` before the
    padding-free swap runs, and the swap only moves that same number across to
    `max_seq_length`. It must never reinstate the raw user `max_length`.
    """
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

    The bridge above the padding-free block copies `max_seq_length` into
    `max_length`, so `max_seq_length` wins. Re-reading the user's raw
    `max_length` inside the padding-free branch would silently invert that and
    train on a quarter of the requested context on TRL >= 1.0.0 only.
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
    "name, dataset, config_kwargs",
    [
        ("input_ids", lambda tok: _tokenized_dataset(tok), {}),
        ("labels", lambda tok: _tokenized_dataset(tok, with_labels = True), {}),
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
    """Nothing truncates these, so `max_length` must survive.

    Unsloth's `sft_prepare_dataset` skips its tokenize pass entirely when the
    dataset already carries `input_ids` / `labels`, and TRL skips the whole
    prepare step for `skip_prepare_dataset=True`. Clearing `args.max_length`
    there would tell TRL the caller supplied truncated rows, and overlength rows
    would reach padding-free training untouched. Drop padding-free instead and
    let TRL's collator enforce the cap.
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
    # The rows themselves stay long: enforcement lives in the collator.
    assert _longest(trainer) > _MODEL_MAX_SEQ_LENGTH
    if getattr(trainer.data_collator, "max_length", None) is not None:
        assert (
            _collated_width(trainer) == _MODEL_MAX_SEQ_LENGTH
        ), f"{name}: overlength rows reached the model"


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
