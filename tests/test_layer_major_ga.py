# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""offload_layers layer-major gradient accumulation through transformers.Trainer (CPU, tiny Llama + LoRA, a stand-in
block swap): the trained LoRA weights and logged losses are bitwise those of standard accumulation, windows run only
with an active swap and an eligible trainer, UNSLOTH_OFFLOAD_LAYER_MAJOR = 0 runs none, and auto turns on only after
measured copy stalls."""

import importlib.util
import os
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _zoo_engine_path():
    explicit = os.environ.get("UNSLOTH_ZOO_DIR")
    if explicit:
        return os.path.join(explicit, "unsloth_zoo", "block_swap_layer_major.py")
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "block_swap_layer_major.py")


@pytest.fixture
def ga(monkeypatch):
    path = _zoo_engine_path()
    if path is None or not os.path.exists(path):
        pytest.skip("unsloth_zoo without block_swap_layer_major")
    spec = importlib.util.spec_from_file_location("unsloth_zoo.block_swap_layer_major", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    # The engine is a leaf module: registering it skips unsloth_zoo's package import (GPU set-up) on this CPU test.
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap_layer_major", engine)
    spec = importlib.util.spec_from_file_location(
        "layer_major_ga_under_test", os.path.join(_ROOT, "unsloth", "models", "_layer_major_ga.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run_layer_major is engine.run_layer_major
    # Patch transformers.Trainer itself, as Unsloth does (monkeypatch puts the originals back after the test).
    monkeypatch.setattr(transformers.Trainer, "training_step", transformers.Trainer.training_step)
    monkeypatch.setattr(
        transformers.Trainer, "get_batch_samples", transformers.Trainer.get_batch_samples
    )
    mod.patch_layer_major_gradient_accumulation(transformers.Trainer)
    assert getattr(transformers.Trainer.training_step, "_unsloth_layer_major", False)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return mod, engine


class _Swap:
    """Stand-in for unsloth_zoo's BlockSwap: two of four layers swapped, holds recorded, optional stall totals."""

    def __init__(self, stall = None):
        self.indices, self.total, self.log = [1, 2], 4, []
        self._tot = {"stall_ms": 0.0, "compute_ms": 0.0}
        self.stall = stall

    def hold(self, idx):
        self.log.append(("hold", idx))

    def unhold(self, idx):
        self.log.append(("unhold", idx))

    def _harvest(self):
        if self.stall is not None:
            self._tot["stall_ms"] += self.stall
            self._tot["compute_ms"] += 100.0


def _model(dropout = 0.0):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
        max_position_embeddings = 256,
    )
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r = 4,
            lora_alpha = 8,
            lora_dropout = dropout,
            init_lora_weights = False,
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.enable_input_require_grads()
    return model


class _Data(torch.utils.data.Dataset):
    def __init__(self, n = 24):
        g = torch.Generator().manual_seed(1)
        self.rows = [
            torch.randint(0, 256, (8 + int(torch.randint(0, 12, (1,), generator = g)),), generator = g)
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"input_ids": self.rows[i]}


def _collate(rows):
    rows = [{"input_ids": torch.as_tensor(r["input_ids"])} for r in rows]
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.zeros(len(rows), n, dtype = torch.long)
    mask = torch.zeros(len(rows), n, dtype = torch.long)
    for k, r in enumerate(rows):
        ids[k, : len(r["input_ids"])] = r["input_ids"]
        mask[k, : len(r["input_ids"])] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


def _train(
    mod,
    tmp_path,
    monkeypatch,
    mode,
    swap = True,
    steps = 2,
    dropout = 0.0,
    stall = None,
    trainer_base = None,
):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", mode)
    base = trainer_base or transformers.Trainer
    T = type("T", (base,), {})
    model = _model(dropout)
    sw = _Swap(stall)
    if swap:
        model.get_base_model().model.layers._unsloth_block_swap = sw
    args = transformers.TrainingArguments(
        output_dir = str(tmp_path / mode),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 3,
        max_steps = steps,
        learning_rate = 1e-2,
        logging_steps = 1,
        report_to = [],
        save_strategy = "no",
        use_cpu = True,
        seed = 0,
        dataloader_num_workers = 0,
        optim = "adamw_torch",
        lr_scheduler_type = "constant",
        disable_tqdm = True,
    )
    trainer = T(model = model, args = args, train_dataset = _Data(), data_collator = _collate)
    _LAST[:] = [trainer]
    trainer.train()
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return weights, losses, sw


_LAST = []


def _windows(engine):
    return engine.layer_major_stats()["windows"]


def test_trainer_matches_standard_accumulation_bitwise(ga, tmp_path, monkeypatch):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    w0, l0, _ = _train(mod, tmp_path, monkeypatch, "0")
    assert _windows(engine) == 0
    w1, l1, sw = _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 2
    assert l0 == l1 and len(l0) == 2
    assert w0.keys() == w1.keys()
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k
    # Each window holds every swapped layer once on the way down and once on the way up.
    assert sw.log.count(("hold", 1)) == 2 * 2 and sw.log.count(("hold", 2)) == 2 * 2


def test_instance_level_training_step_wrappers_see_each_micro_batch_once(ga, tmp_path, monkeypatch):
    """A diagnostic wrapper on trainer.training_step (UNSLOTH_BATCH_REPORT installs one) counts the trainer
    loop's calls: each micro-batch once, in order, with the window on."""
    mod, engine = ga
    seen = []
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        inner = self.training_step

        def counting(model, inputs, *a, **k):
            seen.append(int(inputs["input_ids"].shape[1]))
            return inner(model, inputs, *a, **k)

        counting._unsloth_layer_major_observer = counting  # read-only, as UNSLOTH_BATCH_REPORT's
        self.training_step = counting

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 2
    first = list(seen)
    seen.clear()
    _train(mod, tmp_path, monkeypatch, "0")
    assert seen == first and len(first) == 6


def test_token_budget_is_ineligible(ga, tmp_path, monkeypatch, capsys):
    """UNSLOTH_TOKEN_BUDGET splits micro-batches in an instance-level training_step: never under a window."""
    mod, engine = ga
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._unsloth_token_budget_installed = True

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0
    assert "UNSLOTH_TOKEN_BUDGET" in capsys.readouterr().out


def test_rewritten_inputs_after_a_window_raise_instead_of_counting_twice(ga, tmp_path, monkeypatch):
    """An instance-level wrapper that hands training_step new dicts for later micro-batches (after the window
    accumulated them) is an error, not a second backward."""
    mod, engine = ga
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        inner = self.training_step
        calls = []

        def rewriting(model, inputs, *a, **k):
            calls.append(1)
            if len(calls) % 3 == 2:
                inputs = dict(inputs)
            return inner(model, inputs, *a, **k)

        rewriting._unsloth_layer_major_observer = rewriting  # claims to be read-only, but is not
        self.training_step = rewriting

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    with pytest.raises(RuntimeError, match = "already ran micro-batch 1"):
        _train(mod, tmp_path, monkeypatch, "1")


def test_no_block_swap_means_no_window(ga, tmp_path, monkeypatch):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", swap = False)
    assert _windows(engine) == 0


def test_lora_dropout_is_ineligible(ga, tmp_path, monkeypatch, capsys):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", dropout = 0.1)
    assert _windows(engine) == 0
    assert "dropout" in capsys.readouterr().out


def test_trainer_overriding_compute_loss_is_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga

    class Custom(transformers.Trainer):
        def compute_loss(self, *args, **kwargs):
            return super().compute_loss(*args, **kwargs)

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", trainer_base = Custom)
    assert _windows(engine) == 0


@pytest.mark.parametrize("stall, want", [(None, 0), (1.0, 0), (50.0, 2)])
def test_auto_turns_on_only_after_measured_stalls(ga, tmp_path, monkeypatch, stall, want):
    """auto: steps 1-2 standard (probe), from step 3 on only if step 2's copies stalled > 5%."""
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "auto", steps = 4, stall = stall)
    assert _windows(engine) == want


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {f"t{i}": i for i in range(256)}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "t0"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object = tok, pad_token = "t0", eos_token = "t1", unk_token = "t0"
    )


def _sft(
    mod,
    tmp_path,
    monkeypatch,
    mode,
    trainer_cls = None,
    abort_once = False,
):
    trl = pytest.importorskip("trl")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", mode)
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")  # TRL's entropy metric reads the logits
    T = (
        trainer_cls or trl.SFTTrainer
    )  # the real SFTTrainer chain over the patched transformers.Trainer
    model = _model()
    model.get_base_model().model.layers._unsloth_block_swap = _Swap()
    data = _Data()
    try:
        import datasets
    except ImportError:
        pytest.skip("datasets")
    ds = datasets.Dataset.from_dict({"input_ids": [r.tolist() for r in data.rows]})
    args = trl.SFTConfig(
        output_dir = str(tmp_path / f"sft{mode}"),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 3,
        max_steps = 2,
        learning_rate = 1e-2,
        logging_steps = 1,
        report_to = [],
        save_strategy = "no",
        use_cpu = True,
        seed = 0,
        dataloader_num_workers = 0,
        optim = "adamw_torch",
        lr_scheduler_type = "constant",
        disable_tqdm = True,
        dataset_kwargs = {"skip_prepare_dataset": True},
        remove_unused_columns = False,
    )
    trainer = T(
        model = model,
        args = args,
        train_dataset = ds,
        data_collator = _collate,
        processing_class = _tokenizer(),
    )
    if abort_once:
        # One out-of-memory error in the window's decoder backward (after the loss parts ran): the step is
        # retried the standard way.
        armed = [True]

        def oom(module, args, out):
            if (
                armed[0]
                and torch.is_grad_enabled()
                and mod._TLS.__dict__.get("passthrough") is None
            ):
                armed[0] = False
                raise torch.OutOfMemoryError("simulated")

        model.get_base_model().model.layers[1].register_forward_hook(oom)
    trainer.train()
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    logs = [
        {
            k: v
            for k, v in h.items()
            if k in ("loss", "entropy", "mean_token_accuracy", "num_tokens")
        }
        for h in trainer.state.log_history
        if "loss" in h
    ]
    return weights, logs


def test_trl_sft_trainer_matches_standard_accumulation_bitwise(ga, tmp_path, monkeypatch):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    w0, l0 = _sft(mod, tmp_path, monkeypatch, "0")
    assert _windows(engine) == 0
    w1, l1 = _sft(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 2
    assert l0 == l1 and len(l0) == 2
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


def test_sft_abort_restores_metrics_and_token_counts(ga, tmp_path, monkeypatch):
    """A window aborted in its decoder backward (TRL already counted tokens and metrics in the loss parts):
    the retry logs what standard accumulation logs."""
    mod, engine = ga
    w0, l0 = _sft(mod, tmp_path, monkeypatch, "0")
    engine.layer_major_stats(reset = True)
    w1, l1 = _sft(mod, tmp_path, monkeypatch, "1", abort_once = True)
    assert engine.layer_major_stats()["aborts"] == 1
    assert l0 == l1 and len(l0) == 2
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


def test_custom_sft_subclass_override_is_ineligible(ga, tmp_path, monkeypatch):
    trl = pytest.importorskip("trl")

    class CustomSFTTrainer(trl.SFTTrainer):
        def training_step(self, *args, **kwargs):
            return super().training_step(*args, **kwargs)

    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _sft(mod, tmp_path, monkeypatch, "1", trainer_cls = CustomSFTTrainer)
    assert _windows(engine) == 0


def test_unmarked_instance_wrappers_are_ineligible(ga, tmp_path, monkeypatch):
    """An instance-level training_step wrapper that edits a micro-batch in place before the trainer step: the
    window would have trained later micro-batches before it ran, so it falls back (and matches standard)."""
    mod, engine = ga
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        inner = self.training_step

        def masking(model, inputs, *a, **k):
            inputs["labels"][:, 1] = -100
            return inner(model, inputs, *a, **k)

        self.training_step = masking

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    w0, l0, _ = _train(mod, tmp_path, monkeypatch, "0")
    engine.layer_major_stats(reset = True)
    w1, l1, _ = _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0
    assert l0 == l1
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


def test_substep_callbacks_are_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga
    grads = []

    class Watch(transformers.TrainerCallback):
        def on_substep_end(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            grads.append(
                sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            )

    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.add_callback(Watch())

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    _train(mod, tmp_path, monkeypatch, "0")
    base, grads[:] = list(grads), []
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0
    assert grads == base


def test_optimizers_that_step_inside_backward_are_ineligible(ga):
    import types

    mod, engine = ga
    model = _model()
    layers = model.get_base_model().model.layers
    stub = types.SimpleNamespace(
        args = types.SimpleNamespace(optim = "adalomo", n_gpu = 1), accelerator = None
    )
    assert "every backward" in mod.layer_major_ineligible(stub, model, layers, _Swap())
    stub.args.optim = "adamw_torch"
    assert mod.layer_major_ineligible(stub, model, layers, _Swap()) is None
    p = next(p for p in model.parameters() if p.requires_grad)
    p.register_post_accumulate_grad_hook(lambda t: None)
    assert "inside backward" in mod.layer_major_ineligible(stub, model, layers, _Swap())


def test_vram_budget_uses_the_capped_allocator_budget(ga, monkeypatch):
    mod, engine = ga
    model = _model()
    layers = model.get_base_model().model.layers
    samples = [{"input_ids": torch.zeros(2, 2048, dtype = torch.long)} for _ in range(3)]
    need = engine.boundary_bytes([2 * 2048] * 3, 64, len(layers), 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mod, "_usable_cuda_bytes", lambda device: int(need / 0.6) + 1)
    assert mod._fits(model, layers, samples)[0] is True
    monkeypatch.setattr(mod, "_usable_cuda_bytes", lambda device: int(need / 0.6) - 1)
    assert mod._fits(model, layers, samples)[0] is False


def test_prefetched_batches_are_not_kept(ga, tmp_path, monkeypatch):
    import weakref

    mod, engine = ga
    refs = []
    real = transformers.Trainer.get_batch_samples

    def recording(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        refs[:] = [weakref.ref(b["input_ids"]) for b in out[0]]
        return out

    monkeypatch.setattr(transformers.Trainer, "get_batch_samples", recording)
    import gc

    for mode, swap in (("0", True), ("1", False), ("1", True)):
        _train(mod, tmp_path, monkeypatch, mode, swap = swap)
        gc.collect()
        # The trainer is still alive (_LAST): it must not hold the last step's micro-batches.
        assert _LAST and refs and all(r() is None for r in refs), (mode, swap)
    _LAST.clear()
