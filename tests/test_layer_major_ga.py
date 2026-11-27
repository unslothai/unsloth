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
    monkeypatch.setattr(transformers.Trainer, "train", transformers.Trainer.train)
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


def _model(dropout = 0.0, arch = "Llama"):
    torch.manual_seed(0)
    cfg = getattr(transformers, f"{arch}Config")(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
        max_position_embeddings = 256,
    )
    cfg._attn_implementation = "eager"
    model = getattr(transformers, f"{arch}ForCausalLM")(cfg)
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
    arch = "Llama",
    collate = None,
    model_hook = None,
    **extra_args,
):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", mode)
    base = trainer_base or transformers.Trainer
    T = type("T", (base,), {})
    model = _model(dropout, arch)
    if model_hook is not None:
        model = model_hook(model) or model
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
        **{k: v for k, v in extra_args.items() if k not in ("trainer_kwargs",)},
    )
    trainer = T(
        model = model,
        args = args,
        train_dataset = _Data(),
        data_collator = collate or _collate,
        **extra_args.get("trainer_kwargs", {}),
    )
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
        # As install_token_budget leaves a budget (its splitting training_step aside).
        self._unsloth_token_budget_installed = True
        self._unsloth_token_budget_splits = 0

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0
    assert "UNSLOTH_TOKEN_BUDGET" in capsys.readouterr().out


def test_token_balance_keeps_windows(ga, tmp_path, monkeypatch):
    """UNSLOTH_TOKEN_BALANCE (on by default under DDP) only regroups rows in an instance-level get_train_dataloader:
    windows still run, bitwise equal to standard accumulation over the same balanced batches."""
    mod, engine = ga
    spec = importlib.util.spec_from_file_location(
        "_lmga_token_budget", os.path.join(_ROOT, "unsloth", "utils", "token_budget.py")
    )
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "exact")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        budget.install_token_budget(self)
        assert "get_train_dataloader" in self.__dict__

    monkeypatch.setattr(transformers.Trainer, "__init__", init)
    engine.layer_major_stats(reset = True)
    w0, l0, _ = _train(mod, tmp_path, monkeypatch, "0")
    w1, l1, _ = _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 2
    assert l0 == l1 and len(l0) == 2
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


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
        # Injected inside the layer body (module forward hooks from outside the trusted packages are refused).
        armed = [True]
        mlp = model.get_base_model().model.layers[1].mlp
        real = mlp.forward

        def oom(*args, **kwargs):
            if (
                armed[0]
                and torch.is_grad_enabled()
                and mod._TLS.__dict__.get("passthrough") is None
            ):
                armed[0] = False
                raise torch.OutOfMemoryError("simulated")
            return real(*args, **kwargs)

        mlp.forward = oom
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
    assert mod._window_blocker(stub, model, layers) is None
    p = next(p for p in model.parameters() if p.requires_grad)
    p.register_post_accumulate_grad_hook(lambda t: None)
    assert "inside backward" in mod._window_blocker(stub, model, layers)


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


# ---------------------------------------------------------------- round 2 review fixes (each fails without its fix)
def _with_init(monkeypatch, after):
    real_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        after(self)

    monkeypatch.setattr(transformers.Trainer, "__init__", init)


def _same(a, b):
    (w0, l0, _), (w1, l1, _) = a, b
    assert l0 == l1 and w0.keys() == w1.keys()
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


def test_compiled_models_are_ineligible(ga):
    import types

    mod, engine = ga
    stub = types.SimpleNamespace(args = types.SimpleNamespace(optim = "adamw_torch", n_gpu = 1))
    for how in ("compile", "optimized", "forward"):
        model = _model()
        base = model.get_base_model()
        layers = base.model.layers
        assert mod._window_blocker(stub, model, layers) is None
        if how == "compile":
            base.compile(backend = "eager")
        elif how == "optimized":
            model = torch.compile(model, backend = "eager")
        else:
            base.model.forward = torch.compile(base.model.forward, backend = "eager")
        assert "torch.compile" in mod._window_blocker(stub, model, layers), how
    # Compiled decoder layers or modules inside them: refused too (only Unsloth's block compile is proven).
    for how in ("layer", "submodule"):
        model = _model()
        layers = model.get_base_model().model.layers
        if how == "layer":
            layers[1].compile(backend = "eager")
        else:
            attn = layers[2].self_attn
            attn.forward = torch.compile(attn.forward, backend = "eager")
        assert "decoder layer" in mod._window_blocker(stub, model, layers), how


def test_compiled_model_trains_the_standard_way(ga, tmp_path, monkeypatch):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _train(
        mod,
        tmp_path,
        monkeypatch,
        "1",
        steps = 1,
        model_hook = lambda m: m.get_base_model().compile(backend = "eager"),
    )
    assert _windows(engine) == 0


def test_custom_compute_loss_func_keeps_standard_accumulation(ga, tmp_path, monkeypatch):
    """compute_loss_func runs in each micro-batch's loss part; one that reads the accelerator's sync state (a
    regulariser once per step) would see the first micro-batch's state for all of them under a window."""
    mod, engine = ga
    seen = []

    def make():
        trainer_ref = []

        def loss_func(
            outputs,
            labels,
            num_items_in_batch = None,
        ):
            logits = outputs.logits[..., :-1, :].float()
            loss = (
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels[..., 1:].reshape(-1),
                    ignore_index = -100,
                    reduction = "sum",
                )
                / num_items_in_batch
            )
            sync = trainer_ref[0].accelerator.sync_gradients
            seen.append(sync)
            if sync:
                loss = loss * 1.5  # weighs the step's last micro-batch more
            return loss

        return trainer_ref, loss_func

    runs = []
    for mode in ("0", "1"):
        ref, fn = make()
        _with_init(monkeypatch, lambda self: ref.append(self))
        seen.clear()
        engine.layer_major_stats(reset = True)
        runs.append(
            _train(mod, tmp_path, monkeypatch, mode, trainer_kwargs = {"compute_loss_func": fn})
        )
        runs[-1] = runs[-1] + (list(seen), _windows(engine))
    (w0, l0, s0, seen0, n0), (w1, l1, s1, seen1, n1) = runs
    assert n1 == 0 and seen1 == seen0 == [False, False, True] * 2
    _same((w0, l0, s0), (w1, l1, s1))


def test_backward_hooks_installed_after_the_auto_probe_are_ineligible(ga, tmp_path, monkeypatch):
    """A post-accumulate-grad hook (an optimizer step inside backward) installed from a callback at step 3, after
    auto's two probe steps: no window from then on."""
    mod, engine = ga

    class Install(transformers.TrainerCallback):
        def on_step_begin(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if state.global_step == 2:
                p = next(p for p in model.parameters() if p.requires_grad)
                p.register_post_accumulate_grad_hook(lambda t: None)

    _with_init(monkeypatch, lambda self: self.add_callback(Install()))
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "auto", steps = 4, stall = 50.0)
    assert _windows(engine) == 0


def test_instance_assigned_substep_callbacks_are_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga
    grads = []

    def watch(
        args,
        state,
        control,
        model = None,
        **kwargs,
    ):
        grads.append(
            sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        )

    def add(self):
        callback = transformers.TrainerCallback()
        callback.on_substep_end = watch
        self.add_callback(callback)

    _with_init(monkeypatch, add)
    _train(mod, tmp_path, monkeypatch, "0")
    base, grads[:] = list(grads), []
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0
    assert grads == base and len(base) == 4


def test_first_window_after_evaluation_runs(ga, tmp_path, monkeypatch):
    mod, engine = ga
    eval_args = dict(
        eval_strategy = "steps",
        eval_steps = 1,
        per_device_eval_batch_size = 2,
        trainer_kwargs = {"eval_dataset": _Data(4)},
    )
    a = _train(mod, tmp_path, monkeypatch, "0", steps = 3, **eval_args)
    engine.layer_major_stats(reset = True)
    b = _train(mod, tmp_path, monkeypatch, "1", steps = 3, **eval_args)
    assert _windows(engine) == 3
    _same(a, b)


def test_tokenizer_batch_encodings_are_accepted(ga, tmp_path, monkeypatch):
    mod, engine = ga

    def collate(rows):
        return transformers.BatchEncoding(_collate(rows))

    a = _train(mod, tmp_path, monkeypatch, "0", collate = collate)
    engine.layer_major_stats(reset = True)
    b = _train(mod, tmp_path, monkeypatch, "1", collate = collate)
    assert _windows(engine) == 2
    _same(a, b)


def test_standalone_training_steps_keep_no_prefetched_batch(ga, tmp_path, monkeypatch):
    """A trainer whose training_step never reaches the patched one (TRL's OnlineDPOTrainer): nothing consumes
    the stashed micro-batches, so none are stashed."""
    import gc
    import weakref

    mod, engine = ga
    refs = []
    real = transformers.Trainer.get_batch_samples

    def recording(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        refs[:] = [weakref.ref(b["input_ids"]) for b in out[0]]
        return out

    monkeypatch.setattr(transformers.Trainer, "get_batch_samples", recording)

    class Standalone(transformers.Trainer):
        def training_step(
            self,
            model,
            inputs,
            num_items_in_batch = None,
        ):
            model.train()
            loss = model(**inputs).loss / self.args.gradient_accumulation_steps
            loss.backward()
            return loss.detach()

    _train(mod, tmp_path, monkeypatch, "1", trainer_base = Standalone)
    gc.collect()
    assert _LAST and refs and all(r() is None for r in refs)
    assert "_unsloth_layer_major_batch" not in _LAST[0].__dict__
    _LAST.clear()


@pytest.mark.parametrize("mode, want", [("auto", 0), ("1", 2)])
def test_decoders_outside_the_auto_list_need_an_explicit_opt_in(
    ga, tmp_path, monkeypatch, capsys, mode, want
):
    mod, engine = ga
    if not hasattr(transformers, "OlmoForCausalLM"):
        pytest.skip("no Olmo in this transformers")
    base = _train(mod, tmp_path, monkeypatch, "0", arch = "Olmo")
    capsys.readouterr()
    engine.layer_major_stats(reset = True)
    run = _train(
        mod, tmp_path, monkeypatch, mode, arch = "Olmo", steps = 2 if mode == "1" else 4, stall = 50.0
    )
    out = capsys.readouterr().out
    assert _windows(engine) == want
    if mode == "auto":
        assert out.count("opt-in for OlmoDecoderLayer") == 1
    else:
        assert "opt-in" not in out
        _same(base, run)


def test_auto_list_covers_the_dense_families(ga):
    mod, engine = ga
    for arch, name in (
        ("Llama", "LlamaDecoderLayer"),
        ("Qwen2", "Qwen2DecoderLayer"),
        ("Mistral", "MistralDecoderLayer"),
    ):
        model = _model(arch = arch)
        layers = model.get_base_model().model.layers
        assert mod._auto_layers_ok(model, layers) == (True, None), arch
    model = _model(arch = "Olmo")
    assert mod._auto_layers_ok(model, model.get_base_model().model.layers) == (
        False,
        "OlmoDecoderLayer",
    )


# ---------------------------------------------------------------- round 3 review fixes (each fails without its fix)
def test_wrapped_training_step_overrides_are_ineligible(ga, tmp_path, monkeypatch):
    """functools.wraps copies the patched method's marker onto a user's override: identity decides."""
    import functools

    mod, engine = ga

    class Wrapped(transformers.Trainer):
        @functools.wraps(transformers.Trainer.training_step)
        def training_step(self, model, inputs, *args, **kwargs):
            inputs["input_ids"] = (inputs["input_ids"] + 1) % 256  # the same dict, rewritten
            return super().training_step(model, inputs, *args, **kwargs)

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", trainer_base = Wrapped)
    assert _windows(engine) == 0


def test_dropout_outside_the_decoder_is_ineligible(ga, tmp_path, monkeypatch, capsys):
    mod, engine = ga

    def head_dropout(model):
        head = model.get_base_model().lm_head
        head.drop = torch.nn.Dropout(0.3)  # as lora_dropout on a head adapter
        head.register_forward_hook(lambda module, args, out: module.drop(out))

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", model_hook = head_dropout)
    assert _windows(engine) == 0
    assert "Dropout(p = 0.3)" in capsys.readouterr().out


def test_gradient_hooks_on_trainable_parameters_are_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga

    def hook(model):
        p = next(p for p in model.parameters() if p.requires_grad)
        p.register_hook(lambda g: g * 1.5)

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", model_hook = hook)
    assert _windows(engine) == 0


def test_module_backward_hooks_are_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga

    def hook(model):
        model.get_base_model().lm_head.register_full_backward_hook(lambda m, gi, go: None)

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", model_hook = hook)
    assert _windows(engine) == 0


def test_class_level_compiled_forward_is_ineligible(ga):
    import types

    mod, engine = ga
    stub = types.SimpleNamespace(args = types.SimpleNamespace(optim = "adamw_torch", n_gpu = 1))
    model = _model()
    inner = model.get_base_model().model
    layers = inner.layers
    assert mod._window_blocker(stub, model, layers) is None
    cls = type(inner)
    inner.__class__ = type(
        cls.__name__, (cls,), {"forward": torch.compile(cls.forward, backend = "eager")}
    )
    assert "torch.compile" in mod._window_blocker(stub, model, layers)


def test_an_abort_does_not_outlive_its_train_call(ga, tmp_path, monkeypatch):
    mod, engine = ga
    real = mod.run_layer_major
    calls = []

    def once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None, "simulated abort"
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "run_layer_major", once)
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 0 and len(calls) == 1
    trainer = _LAST[0]
    assert not [k for k in trainer.__dict__ if k.startswith("_unsloth_layer_major_")]
    trainer.train()
    assert _windows(engine) == 2
    _LAST.clear()


# ---------------------------------------------------------------- round 4: positive eligibility (each fails before)
def test_prepare_inputs_overrides_are_ineligible(ga, tmp_path, monkeypatch):
    """_prepare_inputs runs inside training_step under the current micro-batch's sync state; any Trainer method a
    non-audited class overrides keeps the step standard."""
    mod, engine = ga
    seen = []

    class Prep(transformers.Trainer):
        def _prepare_inputs(self, inputs):
            sync = self.accelerator.sync_gradients
            seen.append(sync)
            inputs = super()._prepare_inputs(inputs)
            if sync:
                inputs["labels"] = inputs["labels"].clone()
                inputs["labels"][:, :2] = -100
            return inputs

    a = _train(mod, tmp_path, monkeypatch, "0", trainer_base = Prep)
    first, seen[:] = list(seen), []
    engine.layer_major_stats(reset = True)
    b = _train(mod, tmp_path, monkeypatch, "1", trainer_base = Prep)
    assert _windows(engine) == 0 and seen == first == [False, False, True] * 2
    _same(a, b)


def test_trainer_classes_are_matched_by_identity_not_name(ga, tmp_path, monkeypatch):
    mod, engine = ga

    def training_step(self, model, inputs, *args, **kwargs):
        inputs["input_ids"] = (inputs["input_ids"] + 1) % 256
        return transformers.Trainer.training_step(self, model, inputs, *args, **kwargs)

    Lookalike = type("UnslothTrainer", (transformers.Trainer,), {"training_step": training_step})
    Lookalike.__module__ = "unsloth_finetune"
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", trainer_base = Lookalike)
    assert _windows(engine) == 0


@pytest.mark.parametrize("kind", ["Dropout1d", "AlphaDropout", "FeatureAlphaDropout"])
def test_every_dropout_kind_is_ineligible(ga, tmp_path, monkeypatch, capsys, kind):
    mod, engine = ga
    if not hasattr(torch.nn, kind):
        pytest.skip(kind)

    def add(model):
        emb = model.get_base_model().model.embed_tokens
        emb.drop = getattr(torch.nn, kind)(0.3)
        emb.register_forward_hook(lambda module, args, out: module.drop(out))

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", model_hook = add)
    assert _windows(engine) == 0
    assert f"{kind}(p = 0.3)" in capsys.readouterr().out


def test_neftune_is_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", neftune_noise_alpha = 5.0)
    assert _windows(engine) == 0


@pytest.mark.parametrize("kind", ["function_mode", "dispatch_mode", "autocast"])
def test_caller_side_modes_are_ineligible(ga, tmp_path, monkeypatch, kind):
    """Entered around training_step (here from callbacks): thread-local, so the workers would run without them."""
    from torch.utils._python_dispatch import TorchDispatchMode

    mod, engine = ga

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            return func(*args, **(kwargs or {}))

    class FunctionPassthrough(torch.overrides.TorchFunctionMode):
        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            return func(*args, **(kwargs or {}))

    class Enter(transformers.TrainerCallback):
        ctx = None

        def on_step_begin(self, args, state, control, **kwargs):
            if kind == "function_mode":
                self.ctx = FunctionPassthrough()
            elif kind == "dispatch_mode":
                self.ctx = Passthrough()
            else:
                self.ctx = torch.autocast("cpu", dtype = torch.bfloat16, enabled = True)
            self.ctx.__enter__()

        def on_step_end(self, args, state, control, **kwargs):
            self.ctx.__exit__(None, None, None)

    _with_init(monkeypatch, lambda self: self.add_callback(Enter()))
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", steps = 1)
    assert _windows(engine) == 0


def test_off_by_default_with_an_opt_in_hint(ga, tmp_path, monkeypatch, capsys):
    """Opt-in: with UNSLOTH_OFFLOAD_LAYER_MAJOR unset no window runs (one line says how to opt in), with 1 they do."""
    mod, engine = ga
    assert mod._DEFAULT_MODE == "off"
    monkeypatch.delenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", raising = False)
    assert mod._mode() == "off"
    monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", "")
    assert mod._mode() == "off"
    monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", "1")
    assert mod._mode() == "on"
    base = _train(mod, tmp_path, monkeypatch, "0")
    capsys.readouterr()
    real_setenv = monkeypatch.setenv
    monkeypatch.setattr(
        monkeypatch,
        "setenv",
        lambda k, v, *a: monkeypatch.delenv(k, raising = False)
        if k == "UNSLOTH_OFFLOAD_LAYER_MAJOR"
        else real_setenv(k, v, *a),
    )
    engine.layer_major_stats(reset = True)
    run = _train(mod, tmp_path, monkeypatch, "unset")
    monkeypatch.undo()
    out = capsys.readouterr().out
    assert _windows(engine) == 0
    assert out.count("It is opt-in: set UNSLOTH_OFFLOAD_LAYER_MAJOR=1") == 1
    _same(base, run)


def test_only_known_setup_wrappers_may_shadow_trainer_methods(ga):
    import types

    mod, engine = ga
    stub = transformers.Trainer.__new__(transformers.Trainer)
    stub.train = types.MethodType(transformers.Trainer.train, stub)
    stub._prepare_dataset = lambda *a, **k: None
    assert mod._instance_overrides(stub) is None
    stub._prepare_inputs = types.MethodType(transformers.Trainer._prepare_inputs, stub)
    assert mod._instance_overrides(stub) == "_prepare_inputs"


# ---------------------------------------------------------------- round 5 review fixes (each fails without its fix)
def test_a_stop_request_keeps_the_step_standard(ga, tmp_path, monkeypatch):
    """A callback asking the epoch to stop at a step's start: Transformers runs that one micro-batch, then leaves."""
    mod, engine = ga

    class Stop(transformers.TrainerCallback):
        done = False

        def on_step_begin(self, args, state, control, **kwargs):
            # Once: the next epoch's steps then carry whatever the stopped step accumulated.
            if state.global_step == 1 and not self.done:
                self.done = True
                control.should_epoch_stop = True

    _with_init(monkeypatch, lambda self: self.add_callback(Stop()))
    # 4 steps per epoch; max_steps 6 runs into the second epoch after the stop.
    a = _train(mod, tmp_path, monkeypatch, "0", steps = 6)
    engine.layer_major_stats(reset = True)
    b = _train(mod, tmp_path, monkeypatch, "1", steps = 6)
    assert _windows(engine) >= 1  # the steps before the stop
    _same(a, b)


def test_optimizers_with_a_train_hook_are_ineligible(ga, tmp_path, monkeypatch):
    mod, engine = ga
    seen = []

    class Hooked(torch.optim.AdamW):
        def train(self):
            seen.append(any(p.grad is not None for g in self.param_groups for p in g["params"]))

    a = _train(
        mod,
        tmp_path,
        monkeypatch,
        "0",
        trainer_kwargs = {"optimizer_cls_and_kwargs": (Hooked, {"lr": 1e-2})},
    )
    first, seen[:] = list(seen), []
    engine.layer_major_stats(reset = True)
    b = _train(
        mod,
        tmp_path,
        monkeypatch,
        "1",
        trainer_kwargs = {"optimizer_cls_and_kwargs": (Hooked, {"lr": 1e-2})},
    )
    assert _windows(engine) == 0 and seen == first
    _same(a, b)


def test_untrusted_forward_hooks_are_ineligible(ga, tmp_path, monkeypatch, capsys):
    mod, engine = ga

    def hook(model):
        emb = model.get_base_model().model.embed_tokens
        emb.register_forward_hook(lambda module, args, out: out * (1 + 0.01 * torch.rand(())))

    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1", model_hook = hook)
    assert _windows(engine) == 0
    assert "forward hook on" in capsys.readouterr().out
    # The hooks transformers / PEFT install (input-grad hook) are trusted: windows still run.
    engine.layer_major_stats(reset = True)
    _train(mod, tmp_path, monkeypatch, "1")
    assert _windows(engine) == 2


def test_instance_training_step_overrides_keep_no_prefetched_batch(ga, tmp_path, monkeypatch):
    import gc
    import types
    import weakref

    mod, engine = ga
    refs = []
    real = transformers.Trainer.get_batch_samples

    def recording(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        refs[:] = [weakref.ref(b["input_ids"]) for b in out[0]]
        return out

    monkeypatch.setattr(transformers.Trainer, "get_batch_samples", recording)

    def standalone(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        model.train()
        loss = model(**inputs).loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.detach()

    alive = []

    class AtEnd(transformers.TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            gc.collect()
            alive.append(sum(r() is not None for r in refs))

    def install(self):
        self.training_step = types.MethodType(standalone, self)
        self.add_callback(AtEnd())

    _with_init(monkeypatch, install)
    _train(mod, tmp_path, monkeypatch, "0")
    _train(mod, tmp_path, monkeypatch, "1")
    # Still inside train() (on_train_end): no more micro-batches alive than standard accumulation keeps (its own
    # loop locals on some transformers versions).
    assert refs and len(alive) == 2 and alive[1] == alive[0], alive
    assert "_unsloth_layer_major_batch" not in _LAST[0].__dict__
    _LAST.clear()
