# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Default packing keeps optimizer steps per epoch: the batch size and gradient accumulation shrink by the
documents per packed row, or the split stays unpacked when even 1 x 1 would cut the steps too far. An explicit
packing = True is left as the user set it."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
import unsloth.trainer as trainer_module  # noqa: E402


class _Model(torch.nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return input_ids


def _patched(monkeypatch, docs_per_row):
    seen = {"prepare": [], "trainer_batch": []}

    class _StubSFTTrainer:
        # TRL's order: prepare (and pack) the splits, then Trainer.__init__ reads the batch arguments.
        def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            **kwargs,
        ):
            self.model, self.args, self.data_collator = model, args, data_collator
            self.train_dataset = self._prepare_dataset(
                train_dataset, None, args, args.packing, None, "train"
            )
            if eval_dataset is not None:
                eval_packing = args.packing if args.eval_packing is None else args.eval_packing
                self.eval_dataset = self._prepare_dataset(
                    eval_dataset, None, args, eval_packing, None, "eval"
                )
            seen["trainer_batch"].append(
                (
                    args.per_device_train_batch_size,
                    args.gradient_accumulation_steps,
                    len(self.train_dataset),
                )
            )

        def _prepare_dataset(
            self, dataset, processing_class, args, packing, formatting_func, dataset_name
        ):
            seen["prepare"].append((dataset_name, bool(packing)))
            if not packing:
                return dataset
            return datasets.Dataset.from_dict(
                {"input_ids": [[1]] * math.ceil(len(dataset) / docs_per_row)}
            )

    for name in ("enable_sample_packing", "enable_padding_free_metadata"):
        monkeypatch.setattr(trainer_module, name, lambda m, t: None)
    monkeypatch.setattr(trainer_module, "install_token_budget", lambda t: None)
    monkeypatch.setattr(
        trainer_module, "_responses_only_masks_packed_rows", lambda: True, raising = False
    )
    monkeypatch.delenv("UNSLOTH_DEFAULT_PACKING", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module.SFTTrainer, seen


def _config(
    batch_size = 2,
    grad_accum = 4,
    **kw,
):
    base = dict(
        packing = None,
        padding_free = None,
        eval_packing = None,
        max_length = 1024,
        remove_unused_columns = True,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        world_size = 1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _docs(n = 600):
    return datasets.Dataset.from_dict({"text": [f"doc {i}" for i in range(n)]})


@pytest.mark.parametrize(
    "documents, rows, batch, accum, expected",
    [
        (600, 89, 2, 4, (1, 1)),  # the measured Qwen3-0.6B / alpaca run: 6.7 docs per row
        (600, 600, 2, 4, (2, 4)),  # one document per row: nothing to change
        (1000, 100, 4, 8, (3, 1)),  # 10 docs per row, 32 docs per step -> 3 rows per step
        (1000, 100, 8, 10, (8, 1)),  # 80 docs per step -> 8 rows
        (600, 89, 1, 1, None),  # 6.7 docs per row already beats 1 doc per step: stay unpacked
        (600, 89, 2, 2, None),  # 4 docs per step: 1 row of 6.7 would cut steps 1.7x
        (600, 120, 2, 3, (1, 1)),  # 5 docs per row, 6 per step: 1 row gives 120 steps, not 100
    ],
)
def test_default_packing_batch(documents, rows, batch, accum, expected):
    assert trainer_module._default_packing_batch(documents, rows, batch, accum) == expected
    if expected is not None:
        nb, na = expected
        # optimizer steps per epoch never drop by more than the allowed factor
        assert (documents / (batch * accum)) / (
            rows / (nb * na)
        ) <= trainer_module._DEFAULT_PACKING_MAX_FEWER_STEPS


def test_epoch_run_packs_at_matched_documents_per_step(monkeypatch, capsys):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config()
    SFT(model = _Model(), args = config, train_dataset = _docs())
    # Trainer.__init__ sees 1 x 1 over 89 packed rows: 89 steps per epoch, not 12 (75 unpacked).
    assert seen["trainer_batch"] == [(1, 1, 89)]
    assert config.packing is True
    out = capsys.readouterr().out
    assert "2 x 4 -> 1 x 1" in out and "89 steps per epoch, 75 unpacked" in out


def test_epoch_run_that_cannot_keep_its_steps_stays_unpacked(monkeypatch, capsys):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config(batch_size = 1, grad_accum = 1)
    SFT(model = _Model(), args = config, train_dataset = _docs(), eval_dataset = _docs(10))
    assert seen["prepare"] == [("train", True), ("train", False), ("eval", False)]
    assert seen["trainer_batch"] == [(1, 1, 600)]
    assert config.packing is False and config.padding_free is True and config.eval_packing is None
    assert "Default packing skipped" in capsys.readouterr().out


def test_explicit_packing_keeps_the_users_batch(monkeypatch):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config(packing = True)
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert seen["trainer_batch"] == [(2, 4, 89)]


def test_a_reused_config_gets_its_batch_back(monkeypatch):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config()
    SFT(model = _Model(), args = config, train_dataset = _docs())
    SFT(model = _Model(), args = config, train_dataset = _docs(), data_collator = lambda rows: rows)
    assert seen["trainer_batch"] == [(1, 1, 89), (2, 4, 600)]
    assert (config.per_device_train_batch_size, config.gradient_accumulation_steps) == (2, 4)
    # an unpacked fallback stays Unsloth's choice: a reused config decides again
    config = _config(batch_size = 1, grad_accum = 1)
    SFT(model = _Model(), args = config, train_dataset = _docs())
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert seen["trainer_batch"][-2:] == [(1, 1, 600), (1, 1, 600)]
    assert seen["prepare"][-2:] == [("train", True), ("train", False)]
