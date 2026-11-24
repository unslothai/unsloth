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


def _clear_launcher_env(monkeypatch):
    # an ambient torchrun / accelerate launch must not send ordinary cases down the fixed-batch path
    for name in ("WORLD_SIZE", "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(name, raising = False)


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
    _clear_launcher_env(monkeypatch)
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


@pytest.mark.parametrize(
    "trigger",
    ["world_size_env", "world_size_arg", "deepspeed", "fsdp", "fsdp_config", "accelerate_fsdp_env"],
)
def test_distributed_deepspeed_or_fsdp_skip_default_packing(monkeypatch, capsys, trigger):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config(deepspeed = None, fsdp = "", fsdp_config = None)
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising = False)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising = False)
    if trigger == "world_size_env":
        monkeypatch.setenv("WORLD_SIZE", "2")
    elif trigger == "world_size_arg":
        config.world_size = 2
    elif trigger == "deepspeed":
        config.deepspeed = "ds_config.json"
    elif trigger == "fsdp":
        config.fsdp = "full_shard auto_wrap"
    elif trigger == "fsdp_config":
        config.fsdp_config = {"fsdp_version": 2}
    else:
        monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    SFT(model = _Model(), args = config, train_dataset = _docs())
    # never packed, batch untouched: the padding-free behaviour from before default packing
    assert seen["prepare"] == [("train", False)]
    assert seen["trainer_batch"] == [(2, 4, 600)]
    assert config.packing is False and config.padding_free is True
    out = capsys.readouterr().out
    assert "Default packing skipped" in out and (
        "DeepSpeed" in out or "FSDP" in out or "world size 2" in out
    )


def test_transformers_default_fsdp_config_is_not_fsdp(monkeypatch):
    # transformers 4.x fills fsdp_config with these defaults on every run
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising = False)
    config = _config(
        fsdp = [],
        fsdp_config = {
            "min_num_params": 0,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False,
        },
    )
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert seen["trainer_batch"] == [(1, 1, 89)]


def test_fallback_keeps_the_unpacked_length_cap(monkeypatch):
    # The generated SFTTrainer caps pre-tokenized rows at max_length before _prepare_dataset, only when
    # `padding_free and not packing` (unsloth/models/rl.py); Zoo's unpacked prep leaves input_ids rows alone.
    # The fallback must therefore restart init unpacked, not just re-prepare under the packed init.
    seen = {"inits": [], "max_row": []}

    class _GeneratedSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            **kwargs,
        ):
            seen["inits"].append(bool(args.packing))
            self.model, self.args, self.data_collator = model, args, data_collator
            if args.padding_free and not args.packing and args.max_length:
                train_dataset = train_dataset.map(
                    lambda r: {"input_ids": r["input_ids"][: args.max_length]}
                )
            self.train_dataset = self._prepare_dataset(
                train_dataset, None, args, args.packing, None, "train"
            )
            seen["max_row"].append(max(len(x) for x in self.train_dataset["input_ids"]))

        def _prepare_dataset(
            self, dataset, processing_class, args, packing, formatting_func, dataset_name
        ):
            if not packing:
                return dataset  # Zoo: already-tokenized rows pass through untouched
            return datasets.Dataset.from_dict(
                {"input_ids": [[1] * args.max_length] * math.ceil(len(dataset) / 50)}
            )

    for name in ("enable_sample_packing", "enable_padding_free_metadata"):
        monkeypatch.setattr(trainer_module, name, lambda m, t: None)
    monkeypatch.setattr(trainer_module, "install_token_budget", lambda t: None)
    monkeypatch.setattr(
        trainer_module, "_responses_only_masks_packed_rows", lambda: True, raising = False
    )
    monkeypatch.delenv("UNSLOTH_DEFAULT_PACKING", raising = False)
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    module = SimpleNamespace(SFTTrainer = _GeneratedSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    rows = [[5] * 20] * 600 + [[7] * 3000]  # many short documents and one overlength row
    config = _config(batch_size = 1, grad_accum = 1, max_length = 1024)
    trainer = module.SFTTrainer(
        model = _Model(), args = config, train_dataset = datasets.Dataset.from_dict({"input_ids": rows})
    )
    assert config.packing is False and len(trainer.train_dataset) == 601
    assert seen["max_row"][-1] == 1024  # capped, as an unpacked run is
    assert seen["inits"][-1] is False


def _real_steps(
    items,
    batch,
    accum,
    drop_last = False,
):
    # transformers: len(dataloader) // accum + remainder, over floor / ceil(items / batch) micro-batches
    loader = items // batch if drop_last else -(-items // batch)
    return -(-loader // accum)


@pytest.mark.parametrize("drop_last", [False, True])
def test_chosen_batch_keeps_the_real_integer_step_count(drop_last):
    # The bound must hold on the trainer's integer step count, not on fractional rows per step
    # (9 documents in 4 rows at 2 x 4: rows per step 3.56 rounds to 4 -> 1 step instead of 2).
    limit = trainer_module._DEFAULT_PACKING_MAX_FEWER_STEPS
    for documents in range(1, 70):
        for rows in range(1, documents + 1):
            for batch, accum in ((2, 4), (8, 2), (4, 2), (3, 3), (1, 4)):
                chosen = trainer_module._default_packing_batch(
                    documents, rows, batch, accum, drop_last = drop_last
                )
                unpacked = _real_steps(documents, batch, accum, drop_last)
                if chosen is None or unpacked == 0:
                    continue
                nb, na = chosen
                assert nb <= batch and na <= accum, (documents, rows, batch, accum, chosen)
                packed = _real_steps(rows, nb, na, drop_last)
                assert packed >= 1, (documents, rows, batch, accum, chosen)  # never an empty loader
                assert unpacked <= limit * packed, (documents, rows, batch, accum, chosen)


@pytest.mark.parametrize(
    "documents, rows, batch, accum",
    [(160, 130, 8, 2), (700, 612, 4, 2), (440, 400, 8, 4)],
)
def test_poorly_factorable_targets_do_not_raise_gradient_accumulation(
    documents, rows, batch, accum
):
    nb, na = trainer_module._default_packing_batch(documents, rows, batch, accum)
    assert nb <= batch and na <= accum


def test_unchanged_batch_does_not_print_a_transition(monkeypatch, capsys):
    SFT, seen = _patched(monkeypatch, docs_per_row = 1)
    SFT(model = _Model(), args = _config(), train_dataset = _docs())
    out = capsys.readouterr().out
    assert seen["trainer_batch"] == [(2, 4, 600)] and "2 x 4 -> 2 x 4" not in out


def test_skip_keeps_an_explicit_eval_packing_false(monkeypatch):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config(batch_size = 1, grad_accum = 1, eval_packing = False)
    SFT(model = _Model(), args = config, train_dataset = _docs(), eval_dataset = _docs(10))
    assert config.packing is False and config.eval_packing is False


def test_skip_runs_the_ordinary_padding_choice(monkeypatch, capsys):
    # Skipped default packing must build the trainer exactly as an unpacked run: the padding choice
    # (UNSLOTH_PADDING_CHOICE / UNSLOTH_DISABLE_AUTO_PADDING_FREE) decides, not packing's padding_free.
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    monkeypatch.setattr(
        trainer_module, "_auto_padding_choice", lambda *a, **k: (False, "test prefers padded")
    )
    config = _config(batch_size = 1, grad_accum = 1)
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert config.packing is False and config.padding_free is None
    assert config.remove_unused_columns is True
    out = capsys.readouterr().out
    assert "Default packing skipped" in out and "Using padding-free" not in out


def test_fixed_batch_skip_does_not_promise_padding_free(monkeypatch, capsys):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    monkeypatch.setattr(
        trainer_module, "_auto_padding_choice", lambda *a, **k: (False, "test prefers padded")
    )
    config = _config(world_size = 2)
    SFT(model = _Model(), args = config, train_dataset = _docs())
    out = capsys.readouterr().out
    assert "Default packing skipped" in out and "Using padding-free" not in out
    assert config.padding_free is None


def test_reusing_a_config_leaves_the_first_trainer_alone(monkeypatch):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config()
    first = SFT(model = _Model(), args = config, train_dataset = _docs())
    # the user's config keeps the user's batch; the trainer trains at the chosen one
    assert (config.per_device_train_batch_size, config.gradient_accumulation_steps) == (2, 4)
    assert (first.args.per_device_train_batch_size, first.args.gradient_accumulation_steps) == (
        1,
        1,
    )
    second = SFT(model = _Model(), args = config, train_dataset = _docs(), data_collator = lambda r: r)
    assert (first.args.per_device_train_batch_size, first.args.gradient_accumulation_steps) == (
        1,
        1,
    )
    assert (second.args.per_device_train_batch_size, second.args.gradient_accumulation_steps) == (
        2,
        4,
    )
    # a one-field edit after the first trainer is the user's: it is kept
    config.per_device_train_batch_size = 4
    SFT(model = _Model(), args = config, train_dataset = _docs(), data_collator = lambda r: r)
    assert (config.per_device_train_batch_size, config.gradient_accumulation_steps) == (4, 4)


def _raising_trainer(monkeypatch, error, when):
    """A trainer that packs the train split, then raises `error` when `when(args)` holds."""
    seen = {"prepare": [], "inits": []}

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            **kw,
        ):
            seen["inits"].append(bool(args.packing))
            self.model, self.args, self.data_collator = model, args, data_collator
            self.train_dataset = self._prepare_dataset(
                train_dataset, None, args, args.packing, None, "train"
            )
            if when(args):
                raise error

        def _prepare_dataset(self, dataset, processing_class, args, packing, formatting_func, name):
            seen["prepare"].append(bool(packing))
            if not packing:
                return dataset
            return datasets.Dataset.from_dict({"input_ids": [[1]] * math.ceil(len(dataset) / 6.75)})

    for name in ("enable_sample_packing", "enable_padding_free_metadata"):
        monkeypatch.setattr(trainer_module, name, lambda m, t: None)
    monkeypatch.setattr(trainer_module, "install_token_budget", lambda t: None)
    monkeypatch.setattr(
        trainer_module, "_responses_only_masks_packed_rows", lambda: True, raising = False
    )
    monkeypatch.delenv("UNSLOTH_DEFAULT_PACKING", raising = False)
    _clear_launcher_env(monkeypatch)
    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module.SFTTrainer, seen


def test_a_constructor_error_after_the_shrink_restores_the_batch(monkeypatch):
    SFT, seen = _raising_trainer(monkeypatch, RuntimeError("CUDA out of memory"), lambda a: True)
    config = _config()
    with pytest.raises(RuntimeError):
        SFT(model = _Model(), args = config, train_dataset = _docs())
    assert (config.per_device_train_batch_size, config.gradient_accumulation_steps) == (2, 4)


def test_value_error_fallback_keeps_eval_opt_out_and_reelects_on_reuse(monkeypatch):
    attempts = []

    def when(args):
        attempts.append(args.packing)
        return len(attempts) == 1 and args.packing  # only the first trainer refuses packing

    SFT, seen = _raising_trainer(monkeypatch, ValueError("cannot pack here"), when)
    monkeypatch.setattr(
        trainer_module, "_auto_padding_choice", lambda *a, **k: (False, "test prefers padded")
    )
    config = _config(eval_packing = False)
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert config.packing is False and config.eval_packing is False
    assert config.padding_free is None  # the ordinary padding choice ran
    assert (config.per_device_train_batch_size, config.gradient_accumulation_steps) == (2, 4)
    # still Unsloth's choice: a reused config packs again on a trainer that accepts it
    trainer = SFT(model = _Model(), args = config, train_dataset = _docs())
    assert config.packing is True and trainer.args.per_device_train_batch_size == 1


def test_an_unrelated_error_after_the_skip_is_not_retried(monkeypatch):
    SFT, seen = _raising_trainer(monkeypatch, ValueError("bad dataset"), lambda a: not a.packing)
    config = _config(batch_size = 1, grad_accum = 1)
    with pytest.raises(ValueError, match = "bad dataset"):
        SFT(model = _Model(), args = config, train_dataset = _docs())
    assert seen["prepare"] == [True, False]


@pytest.mark.parametrize(
    "trigger", ["deepspeed_plugin", "accelerate_num_steps", "per_gpu_batch", "process_group"]
)
def test_more_fixed_batch_setups_skip_default_packing(monkeypatch, trigger):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    config = _config()
    if trigger == "deepspeed_plugin":
        config.deepspeed_plugin = object()
    elif trigger == "accelerate_num_steps":
        config.accelerator_config = SimpleNamespace(gradient_accumulation_kwargs = {"num_steps": 4})
    elif trigger == "process_group":  # launched without WORLD_SIZE, process group already up
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 2)
    else:
        config.per_gpu_train_batch_size = 2
    SFT(model = _Model(), args = config, train_dataset = _docs())
    assert seen["prepare"] == [("train", False)] and seen["trainer_batch"] == [(2, 4, 600)]


def test_the_prepare_guard_is_detached_after_init(monkeypatch):
    SFT, seen = _patched(monkeypatch, docs_per_row = 6.75)
    trainer = SFT(model = _Model(), args = _config(), train_dataset = _docs())
    assert "_prepare_dataset" not in trainer.__dict__
