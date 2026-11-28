# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full finetuning auto-fit: the optimizer step inside backward (optimizers/auto_fit.py). CPU; bitsandbytes on CUDA."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

ROOT = Path(__file__).resolve().parents[1]
GB = 2**30


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def af(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUTO_FIT_OPTIMIZER", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", raising = False)
    monkeypatch.delenv("UNSLOTH_AUTO_FIT_CLIP", raising = False)
    monkeypatch.delenv("UNSLOTH_AUTO_FIT_EXACT_CLIP", raising = False)
    return _load("_test_auto_fit", "unsloth/optimizers/auto_fit.py")


@pytest.fixture
def sr(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    return _load("_test_auto_fit_sr", "unsloth/optimizers/bf16_stochastic_rounding.py")


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 32,
        tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model._unsloth_full_finetuning = True
    return model


def _dataset():
    from datasets import Dataset

    g = torch.Generator().manual_seed(1)
    rows = []
    for _ in range(12):
        ids = torch.randint(0, 96, (12,), generator = g).tolist()
        rows.append({"input_ids": ids, "labels": list(ids)})
    return Dataset.from_list(rows)


def _train(
    af,
    sr,
    model,
    monkeypatch,
    mode,
    clip,
    optim = "adamw_torch",
    steps = 4,
    tmp_path = None,
    ga = 1,
    save_steps = 0,
    resume = None,
):
    from transformers import Trainer, TrainingArguments

    monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", mode)

    class T(Trainer):
        pass

    sr.patch_trainer_bf16_stochastic_rounding(T)
    args = TrainingArguments(
        output_dir = str(tmp_path or "/tmp/_g125_auto_fit"),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = ga,
        max_steps = steps,
        learning_rate = 1e-3,
        optim = optim,
        max_grad_norm = clip,
        lr_scheduler_type = "cosine",
        warmup_steps = 1,
        report_to = [],
        use_cpu = True,
        logging_steps = 1,
        save_strategy = "steps" if save_steps else "no",
        save_steps = save_steps or 500,
        seed = 0,
        dataloader_num_workers = 0,
        disable_tqdm = True,
    )
    trainer = T(model = model, args = args, train_dataset = _dataset())
    af.install_auto_fit_optimizer(trainer)
    trainer.train(resume_from_checkpoint = resume)
    return trainer


def _weights_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize("clip", [0.0, 1e9])
def test_in_backward_is_bitwise_the_standard_adamw_step(af, sr, monkeypatch, clip, tmp_path):
    """GA = 1, no clipping (or a max_grad_norm the norm never reaches): the same weights bit for bit, with
    bfloat16 stochastic rounding, a warmup + cosine schedule and tied embeddings."""
    base = _tiny_model()
    standard = copy.deepcopy(base)
    reference = _train(af, sr, standard, monkeypatch, "0", clip, tmp_path = tmp_path)
    head = copy.deepcopy(base)
    trainer = _train(af, sr, head, monkeypatch, "in_backward", clip, tmp_path = tmp_path)
    wrapper = trainer._unsloth_in_backward_optimizer
    n = len([p for p in head.parameters() if p.requires_grad])
    # Stochastic rounding covers both (transformers 5 builds the scheduler over accelerate's wrapper of ours).
    mark = "_unsloth_bf16_stochastic_rounding"
    assert getattr(af._unwrap(reference.optimizer), mark, None) == n
    assert getattr(wrapper.optimizer, mark, None) == n
    assert (
        wrapper.hook_steps == 4 * n and wrapper.late_steps == 0
    )  # every weight stepped from its hook
    assert not _weights_equal(base, head)  # it trained
    assert _weights_equal(standard, head)


@pytest.mark.parametrize("batch_bytes", [None, 0])
def test_each_gradient_is_freed_inside_backward(af, batch_bytes):
    """Ready gradients wait for their batch's step: at most the largest weight's size of them is alive next to the
    one arriving (batch_bytes = 0: only the one being stepped, plus a tied one still accumulating)."""
    model = _tiny_model().float()
    base = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    wrapper = af.InBackwardOptimizer(base, batch_bytes = batch_bytes)
    params = [p for p in model.parameters() if p.requires_grad]
    size = lambda q: q.numel() * q.element_size()
    live, live_bytes = [], []

    def record(_p):
        alive = [q for q in params if q.grad is not None]
        live.append(len(alive))
        live_bytes.append(sum(size(q) for q in alive))

    for p in params:
        p.register_post_accumulate_grad_hook(record)
    ids = torch.randint(0, 96, (2, 12))
    with wrapper.armed():
        model(input_ids = ids, labels = ids).loss.backward()
    assert all(p.grad is None for p in params)  # all stepped by the end of backward
    wrapper.step()
    assert all(p.grad is None for p in params)
    largest = max(size(p) for p in params)
    if batch_bytes == 0:
        # Our hook ran first and freed the previous gradients: only this one's (plus the tied head's) remain.
        assert max(live) <= 2, live
    else:
        assert max(live_bytes) <= 2 * largest < sum(size(p) for p in params), live_bytes
        assert max(live) > 2  # batched
    assert len(base.state) == len(params)


def _clip_step(wrapper, params, grads, batches):
    """One step whose gradients arrive in `batches` (one backward graph task each, as one checkpointed block's
    backward). With SGD at lr 1 and no weight decay, the coefficient each weight was clipped by is -delta / grad."""
    before = [p.detach().clone() for p in params]
    with wrapper.armed():
        for batch in batches:
            sum((params[i] * grads[i]).sum() for i in batch).backward()
    wrapper.step()
    return [((before[i] - params[i].detach()) / grads[i]).mean().item() for i in range(len(params))]


def _oracle(steps, max_norm, mode):
    """float64 reference of every weight's coefficient: torch's max / (norm + 1e-6), at most 1, with the norm each
    mode uses for a batch stepped once the earlier batches (and it) have arrived."""
    out, prev_sq, prev_total = [], None, None
    for grads, batches in steps:
        sq = [float((g.double() ** 2).sum()) for g in grads]
        total = sum(sq)
        coef = [None] * len(grads)
        seen = seen_prev = 0.0
        for batch in batches:
            seen += sum(sq[i] for i in batch)
            if prev_sq is not None:
                seen_prev += sum(prev_sq[i] for i in batch)
            for i in batch:
                if mode == "per_tensor":
                    norm = sq[i] ** 0.5
                elif mode == "exact" or prev_total is None:
                    norm = (total if mode == "exact" else seen) ** 0.5
                elif mode == "lagged":
                    norm = max(seen, prev_total) ** 0.5
                else:
                    norm = max(seen, seen * prev_total / seen_prev) ** 0.5
                coef[i] = min(1.0, max_norm / (norm + 1e-6))
        out.append(coef)
        prev_sq, prev_total = sq, total
    return out


def _clip_case(seed = 0, scales = (1.0, 0.4, 2.5)):
    g = torch.Generator().manual_seed(seed)
    shapes = [(5, 3), (7,), (4, 4), (2,), (6, 2), (3,)]
    steps = []
    for k, scale in enumerate(scales):
        grads = [
            torch.randn(s, generator = g, dtype = torch.float64) * scale * (1 + i)
            for i, s in enumerate(shapes)
        ]
        order = list(range(len(shapes)))[::-1] if k % 2 else list(range(len(shapes)))
        steps.append((grads, [order[:2], order[2:3], order[3:]]))
    return shapes, steps


@pytest.mark.parametrize("mode", ["estimate", "lagged", "per_tensor"])
def test_clip_coefficients_match_the_fp64_oracle(af, mode):
    """Batches stepped mid-backward clip with the mode's norm: estimate = this step's squares so far x the previous
    step's total / its squares over the same tensors (never below the norm seen so far), lagged = the previous
    total, per_tensor = each tensor's own. Without a previous step: the norm seen so far."""
    shapes, steps = _clip_case()
    params = [torch.nn.Parameter(torch.zeros(s, dtype = torch.float64)) for s in shapes]
    wrapper = af.InBackwardOptimizer(
        torch.optim.SGD(params, lr = 1.0), max_grad_norm = 1.0, clip = mode, batch_bytes = 1 << 30
    )
    got = [_clip_step(wrapper, params, grads, batches) for grads, batches in steps]
    want = _oracle(steps, 1.0, mode)
    for step_got, step_want in zip(got, want):
        for a, b in zip(step_got, step_want):
            assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (mode, step_got, step_want)
    exact = _oracle(steps, 1.0, "exact")
    assert any(
        abs(a - b) > 1e-3 for a, b in zip(got[1], exact[1])
    )  # an estimate, not the exact norm
    # Never clipped less than the norm already seen implies (and so never less than each tensor's own norm).
    prefix = _oracle([(steps[1][0], steps[1][1])], 1.0, "estimate")[0]
    assert all(a <= b + 1e-12 for a, b in zip(got[1], prefix)) or mode == "per_tensor"


def test_estimate_is_exact_when_gradients_keep_their_relative_sizes(af):
    """Gradients that scale together from one step to the next (any order of arrival) are clipped by exactly the
    global norm in every batch, the first ones included."""
    shapes, steps = _clip_case(scales = (1.0, 1.0, 1.0))
    base = steps[0][0]
    steps = [
        ([g * alpha for g in base], batches) for alpha, (_, batches) in zip((1.0, 3.0, 0.2), steps)
    ]
    params = [torch.nn.Parameter(torch.zeros(s, dtype = torch.float64)) for s in shapes]
    wrapper = af.InBackwardOptimizer(
        torch.optim.SGD(params, lr = 1.0), max_grad_norm = 1.0, batch_bytes = 1 << 30
    )
    assert wrapper.clip == "estimate"  # the default
    got = [_clip_step(wrapper, params, grads, batches) for grads, batches in steps]
    exact = _oracle(steps, 1.0, "exact")
    for step_got, step_want in zip(got[1:], exact[1:]):
        assert max(abs(a - b) for a, b in zip(step_got, step_want)) < 1e-12
    assert (
        max(abs(a - b) for a, b in zip(got[0], exact[0])) > 1e-3
    )  # no previous step: only the norm seen so far


def test_measuring_pass_gives_the_exact_global_coefficient(af):
    """measuring(): nothing is stepped, every gradient is freed as it arrives; the next pass clips every batch by the
    exact global norm, and the logged norm is the global pre-clip norm."""
    shapes, steps = _clip_case()
    params = [torch.nn.Parameter(torch.zeros(s, dtype = torch.float64)) for s in shapes]
    wrapper = af.InBackwardOptimizer(
        torch.optim.SGD(params, lr = 1.0), max_grad_norm = 1.0, clip = "exact", batch_bytes = 1 << 30
    )
    exact = _oracle(steps, 1.0, "exact")
    for (grads, batches), want in zip(steps, exact):
        assert wrapper.wants_measure()
        before = [p.detach().clone() for p in params]
        with wrapper.measuring():
            for batch in batches:
                sum((params[i] * grads[i]).sum() for i in batch).backward()
                assert all(params[i].grad is None for i in batch)
        assert all(torch.equal(p.detach(), b) for p, b in zip(params, before))  # nothing stepped
        got = _clip_step(wrapper, params, grads, batches)
        assert max(abs(a - b) for a, b in zip(got, want)) < 1e-12
        norm = sum(float((g**2).sum()) for g in grads) ** 0.5
    assert wrapper.measured_steps == 3
    assert abs(wrapper._prev_total.item() ** 0.5 - norm) < 1e-9


@pytest.mark.parametrize("mode", ["estimate", "exact", "lagged", "per_tensor"])
def test_clipping_below_max_grad_norm_is_bitwise_the_unclipped_step(af, mode):
    shapes, steps = _clip_case(scales = (1e-3, 2e-3, 5e-4))
    params = [torch.nn.Parameter(torch.zeros(s, dtype = torch.float64)) for s in shapes]
    wrapper = af.InBackwardOptimizer(
        torch.optim.SGD(params, lr = 1.0), max_grad_norm = 1.0, clip = mode, batch_bytes = 1 << 30
    )
    for grads, batches in steps:
        before = [p.detach().clone() for p in params]
        _clip_step(wrapper, params, grads, batches)
        assert all(torch.equal(p.detach(), b - g) for p, b, g in zip(params, before, grads))


def test_reported_norm_is_the_global_pre_clip_norm(af):
    a = torch.nn.Parameter(torch.zeros(4))
    b = torch.nn.Parameter(torch.zeros(3))
    wrapper = af.InBackwardOptimizer(
        torch.optim.SGD([a, b], lr = 1.0), max_grad_norm = 1.0, batch_bytes = 1 << 30
    )
    ga = torch.tensor([3.0, 4.0, 0.0, 0.0])
    gb = torch.tensor([0.3, 0.4, 0.0])
    with wrapper.armed():
        (
            (a * ga).sum() + (b * gb).sum()
        ).backward()  # one batch: the whole step, so the exact global norm
    norm = (25 + 0.25) ** 0.5
    assert torch.allclose(a.detach(), -ga / (norm + 1e-6), atol = 1e-7)
    assert torch.allclose(b.detach(), -gb / (norm + 1e-6), atol = 1e-7)
    assert abs(wrapper.total_norm().item() - norm) < 1e-5
    wrapper.step()
    assert wrapper.total_norm().item() == 0.0


def test_clip_mode_from_the_environment(af, monkeypatch):
    for value, mode in (
        ("", "estimate"),
        ("exact", "exact"),
        ("LAGGED", "lagged"),
        ("per_tensor", "per_tensor"),
    ):
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", value)
        assert af.clip_mode() == mode
    monkeypatch.delenv("UNSLOTH_AUTO_FIT_CLIP")
    monkeypatch.setenv("UNSLOTH_AUTO_FIT_EXACT_CLIP", "1")
    assert af.clip_mode() == "exact"


def _max_diff(a, b):
    return max(
        (x.float() - y.float()).abs().max().item()
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )


def _fp32_model(dropout = 0.0):
    model = _tiny_model().float()
    model.config.attention_dropout = dropout
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = dropout
    return model


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_exact_clip_matches_the_standard_clipped_step(af, sr, monkeypatch, dropout, tmp_path):
    """UNSLOTH_AUTO_FIT_CLIP=exact: a gradient-only pass measures the global norm, then the in-backward step clips
    with it: the standard clip_grad_norm_ step up to float32 rounding, with dropout drawing the same masks."""
    monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", "exact")
    base = _fp32_model(dropout)
    standard = copy.deepcopy(base)
    reference = _train(af, sr, standard, monkeypatch, "0", 0.05, tmp_path = tmp_path)
    head = copy.deepcopy(base)
    trainer = _train(af, sr, head, monkeypatch, "in_backward", 0.05, tmp_path = tmp_path)
    wrapper = trainer._unsloth_in_backward_optimizer
    assert wrapper.measured_steps == 4
    norms = [h["grad_norm"] for h in reference.state.log_history if "grad_norm" in h]
    assert min(norms) > 0.05 * 4  # clipping engaged on every step
    assert _max_diff(base, head) > 1e-3
    assert _max_diff(standard, head) < 2e-6, _max_diff(standard, head)
    got = [h["grad_norm"] for h in trainer.state.log_history if "grad_norm" in h]
    assert all(abs(a - b) <= 1e-4 * b for a, b in zip(got, norms))
    losses = lambda t: [h["loss"] for h in t.state.log_history if "loss" in h]
    assert all(abs(a - b) <= 1e-4 for a, b in zip(losses(trainer), losses(reference)))


def test_estimated_clip_tracks_the_standard_clipped_step(af, sr, monkeypatch, tmp_path):
    """Default: only the first step is measured (no previous step to estimate from); later steps estimate the
    global norm. Much closer to the standard step than clipping each tensor by its own norm."""
    base = _fp32_model()
    standard = copy.deepcopy(base)
    _train(af, sr, standard, monkeypatch, "0", 0.05, steps = 6, tmp_path = tmp_path)
    diffs = {}
    for mode in ("estimate", "per_tensor"):
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", mode)
        head = copy.deepcopy(base)
        trainer = _train(af, sr, head, monkeypatch, "in_backward", 0.05, steps = 6, tmp_path = tmp_path)
        assert trainer._unsloth_in_backward_optimizer.measured_steps == (
            1 if mode == "estimate" else 0
        )
        diffs[mode] = _max_diff(standard, head)
    assert diffs["estimate"] * 2 < diffs["per_tensor"], diffs


def test_measuring_pass_leaves_no_trace(af):
    """The gradient-only pass restores the random state and the trainer's counters and logged metrics."""

    class Trainer:
        pass

    from collections import defaultdict

    trainer = Trainer()
    trainer._total_train_tokens = 10
    train = defaultdict(list, {"num_tokens": [10], "loss": [1.0]})
    trainer._metrics = {"train": train, "eval": defaultdict(list)}
    torch.manual_seed(5)
    want = torch.rand(3)
    torch.manual_seed(5)
    with af._repeatable(trainer, None):
        torch.rand(7)
        trainer._total_train_tokens += 5
        trainer._metrics["train"]["num_tokens"] = [15]
        trainer._metrics["train"]["loss"].append(2.0)
        trainer._metrics["train"]["entropy"].append(0.5)
    assert torch.equal(torch.rand(3), want)
    assert trainer._total_train_tokens == 10
    assert trainer._metrics == {"train": {"num_tokens": [10], "loss": [1.0]}, "eval": {}}
    assert (
        trainer._metrics["train"] is train
    )  # TRL appends to a missing key: still a defaultdict(list)
    trainer._metrics["train"]["entropy"].append(0.1)


def test_trainers_that_generate_inside_the_step_are_not_repeated(
    af, sr, monkeypatch, capsys, tmp_path
):
    """GRPO / RLOO sample completions in _prepare_inputs (online trainers in training_step): running the step twice
    would draw new samples, so exact falls back to the estimate and the first step is not measured."""
    from transformers import Trainer

    class GRPOTrainer(Trainer):
        pass

    class SFTTrainer(Trainer):
        def training_step(self, *args, **kwargs):
            return super().training_step(*args, **kwargs)

    assert af._can_repeat_step(object.__new__(SFTTrainer))
    assert not af._can_repeat_step(object.__new__(GRPOTrainer))

    class Sampling(Trainer):
        def _prepare_inputs(self, inputs):
            return super()._prepare_inputs(inputs)

    assert not af._can_repeat_step(object.__new__(Sampling))
    model = _fp32_model()
    base = torch.optim.SGD(model.parameters(), lr = 1e-3)
    trainer = object.__new__(GRPOTrainer)
    trainer.training_step = lambda *a, **k: None
    wrapper = af.InBackwardOptimizer(base, max_grad_norm = 1.0, clip = "exact")
    af._arm_training_step(trainer, wrapper)
    assert wrapper.clip == "estimate" and not wrapper.calibrate
    assert "generates its inputs" in capsys.readouterr().out


class _NormNet(torch.nn.Module):
    """A forward that updates buffers in place (BatchNorm running statistics)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # No bias before BatchNorm: the norm subtracts it back out, so its gradient is zero up to rounding, and AdamW
        # turns that rounding into lr-sized steps that differ between two equivalent paths. running_mean includes
        # it, which made this comparison depend on the CPU's GEMM kernels rather than on the buffers.
        self.a = torch.nn.Linear(6, 8, bias = False)
        self.norm = torch.nn.BatchNorm1d(8)
        self.b = torch.nn.Linear(8, 1)

    def forward(self, x, labels):
        return {
            "loss": (self.b(torch.relu(self.norm(self.a(x)))).squeeze(-1) - labels).square().mean()
        }


def test_measuring_pass_restores_buffers_the_forward_updates(af, sr, monkeypatch, tmp_path):
    """BatchNorm's running statistics see each batch once, as in the standard step, not once per pass."""
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    g = torch.Generator().manual_seed(2)
    rows = [{"x": torch.randn(6, generator = g).tolist(), "labels": float(i % 3)} for i in range(16)]
    results = {}
    for mode in ("0", "in_backward"):
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", mode)
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", "exact")
        model = _NormNet()
        args = TrainingArguments(
            output_dir = str(tmp_path),
            per_device_train_batch_size = 4,
            max_steps = 3,
            learning_rate = 1e-2,
            max_grad_norm = 0.01,
            report_to = [],
            use_cpu = True,
            save_strategy = "no",
            seed = 0,
            dataloader_num_workers = 0,
            disable_tqdm = True,
        )
        trainer = Trainer(model = model, args = args, train_dataset = Dataset.from_list(rows))
        af.install_auto_fit_optimizer(trainer)
        trainer.train()
        if mode != "0":
            assert trainer._unsloth_in_backward_optimizer.measured_steps == 3
        results[mode] = model
    standard, head = results["0"], results["in_backward"]
    assert int(head.norm.num_batches_tracked) == int(standard.norm.num_batches_tracked) == 3
    assert torch.allclose(head.norm.running_mean, standard.norm.running_mean, atol = 1e-6)
    assert torch.allclose(head.norm.running_var, standard.norm.running_var, atol = 1e-6)
    assert _max_diff(standard, head) < 1e-5


def _lora_model():
    from peft import LoraConfig, get_peft_model

    model = _fp32_model()
    model._unsloth_full_finetuning = False
    return get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"])
    )


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason = "needs peft")
@pytest.mark.parametrize("clip", [0.0, 0.05])
def test_forced_in_backward_on_lora_is_the_standard_step(af, sr, monkeypatch, clip, tmp_path):
    """LoRA / QLoRA never engage automatically (the adapter gradients are a few MB: nothing to gain), but forcing
    the in-backward step on a LoRA model steps only the adapters, as the standard step does."""
    monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", "exact")
    torch.manual_seed(0)
    base = _lora_model()
    for name, p in base.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(
                p, std = 0.02
            )  # B starts at zero: make every adapter receive a gradient
    standard = copy.deepcopy(base)
    _train(af, sr, standard, monkeypatch, "0", clip, tmp_path = tmp_path)
    head = copy.deepcopy(base)
    trainer = _train(af, sr, head, monkeypatch, "in_backward", clip, tmp_path = tmp_path)
    wrapper = trainer._unsloth_in_backward_optimizer
    trainable = [p for p in head.parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, p in head.named_parameters() if p.requires_grad)
    assert wrapper.hook_steps == 4 * len(trainable) and wrapper.late_steps == 0
    frozen = [n for n, p in base.named_parameters() if not p.requires_grad]
    hs, bs = dict(head.named_parameters()), dict(base.named_parameters())
    assert all(torch.equal(hs[n], bs[n]) for n in frozen)
    if clip == 0:
        assert _weights_equal(standard, head)
    else:
        assert _max_diff(standard, head) < 2e-6


def test_second_backward_before_step_raises(af):
    p = torch.nn.Parameter(torch.ones(3))
    wrapper = af.InBackwardOptimizer(torch.optim.SGD([p], lr = 0.1))
    with wrapper.armed():
        p.sum().backward()
        with pytest.raises(RuntimeError, match = "second gradient"):
            p.sum().backward()


def test_unarmed_and_direct_gradients_step_at_optimizer_step(af):
    """A gradient outside training_step (or assigned to .grad directly) waits for step(), as before."""
    p = torch.nn.Parameter(torch.ones(3))
    wrapper = af.InBackwardOptimizer(torch.optim.SGD([p], lr = 0.5))
    p.sum().backward()
    assert torch.equal(p.detach(), torch.ones(3)) and p.grad is not None
    wrapper.step()
    assert torch.equal(p.detach(), torch.full((3,), 0.5)) and p.grad is None
    assert wrapper.late_steps == 1


def test_state_dict_is_the_wrapped_optimizers(af):
    model = torch.nn.Linear(4, 3)
    base = torch.optim.AdamW(model.parameters(), lr = 1e-2)
    wrapper = af.InBackwardOptimizer(base)
    with wrapper.armed():
        model(torch.randn(2, 4)).sum().backward()
    wrapper.step()
    saved = wrapper.state_dict()
    assert saved["param_groups"] == base.state_dict()["param_groups"]
    fresh = torch.optim.AdamW(copy.deepcopy(model).parameters(), lr = 1e-2)
    fresh.load_state_dict(saved)  # a standard optimizer resumes from it
    wrapper.load_state_dict(saved)  # and the wrapper still steps after a load
    with wrapper.armed():
        model(torch.randn(2, 4)).sum().backward()
    wrapper.step()
    assert int(base.state[model.weight]["step"]) == 2


def test_global_step_hooks_fire_once_per_step(af):
    from torch.optim.optimizer import register_optimizer_step_post_hook

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(model.parameters(), lr = 1e-3))
    calls = []
    handle = register_optimizer_step_post_hook(lambda opt, a, k: calls.append(type(opt).__name__))
    try:
        with wrapper.armed():
            model(torch.randn(3, 4)).sum().backward()
        wrapper.step()
    finally:
        handle.remove()
    assert calls == ["InBackwardOptimizer"]


class _Args:
    def __init__(self, **kw):
        self.gradient_accumulation_steps = 1
        self.world_size = 1
        self.n_gpu = 1
        self.fp16 = False
        self.optim = "adamw_8bit"
        self.max_grad_norm = 1.0
        self.per_device_train_batch_size = 1
        self.max_length = 512
        self.__dict__.update(kw)


class _Trainer:
    def __init__(self, **kw):
        self.args = _Args(**kw)
        self.optimizer_cls_and_kwargs = None


def _plan_model():
    # 24 x 1M bfloat16 weights: standard needs grads + 8-bit state, in backward only the state.
    model = torch.nn.Sequential(
        *[torch.nn.Linear(1000, 1000, bias = False, dtype = torch.bfloat16) for _ in range(24)]
    )
    model._unsloth_full_finetuning = True
    return model


def _decide_with(
    af,
    monkeypatch,
    usable,
    mode = "auto",
    **kw,
):
    monkeypatch.setattr(af, "_usable_bytes", lambda params: usable)
    monkeypatch.setattr(af, "_lomo_available", lambda: True)
    return af._decide(_Trainer(**kw), _plan_model(), mode)


def test_trigger_fires_only_on_a_predicted_oom(af, monkeypatch):
    plan = af.plan_auto_fit(_Trainer(), _plan_model(), usable_bytes = 0)
    assert plan["lomo"] < plan["in_backward"] < plan["standard"]
    between = (plan["in_backward"] + plan["standard"]) // 2
    tight = (plan["lomo"] + plan["in_backward"]) // 2
    assert _decide_with(af, monkeypatch, plan["standard"] + 1)[0] is None  # fits: unchanged, silent
    assert _decide_with(af, monkeypatch, plan["standard"] + 1)[1] is None
    tier, message = _decide_with(af, monkeypatch, between)
    assert tier == "in_backward" and "UNSLOTH_AUTO_FIT_OPTIMIZER=0" in message
    tier, message = _decide_with(af, monkeypatch, tight)
    assert tier == "sgd" and "SGD" in message
    tier, message = _decide_with(af, monkeypatch, plan["lomo"] - 1)
    assert tier is None and "Neither" in message
    # Gradient accumulation: unchanged, with the hint only when it would run out of memory.
    tier, message = _decide_with(af, monkeypatch, between, gradient_accumulation_steps = 4)
    assert tier is None and "gradient_accumulation_steps = 1" in message
    assert _decide_with(af, monkeypatch, plan["standard"] + 1, gradient_accumulation_steps = 4) == (
        None,
        None,
    )
    for blocked in (
        dict(world_size = 2),
        dict(fp16 = True),
        dict(optim = "adamw_torch_layerwise"),
        dict(auto_find_batch_size = True),
        dict(torch_compile = True),
        dict(enable_jit_checkpoint = True),
    ):
        assert _decide_with(af, monkeypatch, between, **blocked) == (None, None), blocked
    # LoRA / frozen models: never automatic.
    monkeypatch.setattr(af, "_full_finetuning", lambda model: False)
    assert _decide_with(af, monkeypatch, between) == (None, None)
    assert _decide_with(af, monkeypatch, between, mode = "in_backward")[0] == "in_backward"


def test_distributed_training_is_refused_except_fsdp2(af, monkeypatch):
    """FSDP2 (accelerate's plugin with fsdp_version 2, or Unsloth's native FSDP2) can step inside backward, with
    gradient accumulation too; DDP, FSDP1, DeepSpeed, tensor / pipeline / context parallelism, HSDP and FSDP2 CPU
    offload keep their reasons."""
    from types import SimpleNamespace

    monkeypatch.setattr(af, "_fsdp_calls_grad_hooks", lambda: True)
    for name in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE"):
        monkeypatch.delenv(name, raising = False)

    def trainer(
        plugin = None,
        config = None,
        **kw,
    ):
        t = _Trainer(world_size = 2, **kw)
        t.accelerator = SimpleNamespace(
            state = SimpleNamespace(fsdp_plugin = plugin), parallelism_config = config
        )
        t.is_fsdp_enabled = plugin is not None
        return t

    model = _plan_model()
    fsdp2 = SimpleNamespace(fsdp_version = 2, cpu_offload = None)
    assert af._distributed(trainer(fsdp2), model) == ("fsdp2", None)
    assert af._blocker(trainer(fsdp2, gradient_accumulation_steps = 4), model, False) is None
    assert "gradient_accumulation" in af._blocker(
        _Trainer(gradient_accumulation_steps = 4), model, False
    )

    class CPUOffloadPolicy:
        pass

    cases = [
        (trainer(), "DDP"),
        (trainer(SimpleNamespace(fsdp_version = 1)), "FSDP1"),
        (trainer(SimpleNamespace(fsdp_version = 2, cpu_offload = CPUOffloadPolicy())), "CPU offload"),
        (trainer(fsdp2, SimpleNamespace(tp_size = 2)), "tensor parallelism"),
        (trainer(fsdp2, SimpleNamespace(dp_replicate_size = 2)), "HSDP"),
        (trainer(fsdp2, SimpleNamespace(cp_size = 2)), "context parallelism"),
    ]
    deepspeed = trainer(fsdp2)
    deepspeed.is_deepspeed_enabled = True
    cases.append((deepspeed, "DeepSpeed"))
    for t, why in cases:
        kind, reason = af._distributed(t, model)
        assert kind is None and why in reason, (why, reason)
        assert why in af._blocker(t, model, True)
    monkeypatch.setenv("UNSLOTH_PP_SIZE", "2")
    assert "pipeline" in af._distributed(trainer(fsdp2), model)[1]
    monkeypatch.delenv("UNSLOTH_PP_SIZE")
    monkeypatch.setattr(af, "_fsdp_calls_grad_hooks", lambda: False)
    assert "torch 2.5" in af._distributed(trainer(fsdp2), model)[1]


def test_planner_budgets_whole_fsdp2_weights_before_sharding(af):
    """accelerate shards after the optimizer is built: the root group (outside the decoder blocks) stays whole for
    the step, resharding blocks are whole two at a time, and without reshard_after_forward every block stays whole."""
    from types import SimpleNamespace

    model = _tiny_model().float()
    full = {n: p.numel() * p.element_size() for n, p in model.named_parameters()}
    root = sum(v for n, v in full.items() if "layers." not in n)
    block = sum(v for n, v in full.items() if "layers.0." in n)
    trainer = SimpleNamespace(
        accelerator = SimpleNamespace(
            state = SimpleNamespace(
                fsdp_plugin = SimpleNamespace(fsdp_version = 2, reshard_after_forward = True)
            )
        )
    )
    assert af._fsdp_gathered(trainer, model) == (root, 2 * block)
    trainer.accelerator.state.fsdp_plugin.reshard_after_forward = False
    assert af._fsdp_gathered(trainer, model) == (root + 2 * block, 0)
    # Each group is one in-backward batch: the planner budgets the largest group, not the largest weight.
    assert af._fsdp_group_bytes(model) == [block, block, root]


def test_free_memory_reserves_the_shard_of_host_resident_weights(af, monkeypatch):
    """Planning before accelerate moves the model to the GPU: the rank's weight shard still has to arrive there."""
    params = [torch.nn.Parameter(torch.zeros(1000, 250)) for _ in range(4)]  # 4 MB on the host
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (100 << 20, 200 << 20))
    monkeypatch.setattr(af, "_min_over_ranks", lambda value, device: value)
    assert af._usable_bytes(params, world = 2) == (100 << 20) - 4 * 1000 * 250 * 4 // 2


def test_planner_is_per_rank_under_fsdp2(af, monkeypatch):
    from types import SimpleNamespace

    model = _plan_model()
    single = af.plan_auto_fit(_Trainer(), model, usable_bytes = 0)
    monkeypatch.setattr(af, "_distributed", lambda trainer, model: ("fsdp2", None))
    monkeypatch.setattr(af, "_world_size", lambda: 4)
    sharded = af.plan_auto_fit(
        SimpleNamespace(args = _Args(), optimizer_cls_and_kwargs = None), model, usable_bytes = 0
    )
    assert sharded["grads"] == single["grads"] // 4 and sharded["states"] == single["states"] // 4
    assert sharded["in_backward"] < sharded["standard"] < single["standard"]
    # Gradient accumulation: the earlier micro-batches' gradients are held in every tier (shards when reduced each
    # micro-batch, else FSDP2's whole accumulators under no_sync).
    for each, held in ((True, sharded["grads"]), (False, sharded["grads"] * 4)):
        monkeypatch.setattr(
            af, "_fsdp_reduces_every_micro_batch", lambda trainer, model, each = each: each
        )
        ga = af.plan_auto_fit(
            SimpleNamespace(
                args = _Args(gradient_accumulation_steps = 2), optimizer_cls_and_kwargs = None
            ),
            model,
            usable_bytes = 0,
        )
        assert ga["held"] == held and ga["in_backward"] == sharded["in_backward"] + held


def test_kill_switch_and_forced_modes(af, monkeypatch):
    for value, mode in (
        ("0", "off"),
        ("off", "off"),
        ("", "auto"),
        ("1", "auto"),
        ("LOMO", "lomo"),
    ):
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", value)
        assert af.auto_fit_mode() == mode
    monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", "in_backward")
    assert af.auto_fit_mode() == "in_backward"


def test_kill_switch_keeps_the_standard_optimizer(af, sr, monkeypatch, tmp_path):
    trainer = _train(af, sr, _tiny_model(), monkeypatch, "0", 1.0, steps = 1, tmp_path = tmp_path)
    assert not hasattr(trainer, "_unsloth_in_backward_optimizer")
    assert not getattr(trainer, "_unsloth_updates_in_backward", False)


def test_forced_in_backward_refuses_gradient_accumulation(af, sr, monkeypatch, tmp_path, capsys):
    trainer = _train(
        af, sr, _tiny_model(), monkeypatch, "in_backward", 1.0, steps = 1, tmp_path = tmp_path, ga = 2
    )
    assert not hasattr(trainer, "_unsloth_in_backward_optimizer")
    assert "gradient_accumulation_steps > 1" in capsys.readouterr().out


def test_lomo_tier_is_lomos_update(af, sr, monkeypatch, tmp_path):
    """UNSLOTH_AUTO_FIT_OPTIMIZER=lomo: torch SGD stepped inside backward over the Trainer's groups. Bitwise the
    standard optim="sgd" step, and lomo-optim's LOMO (optim="lomo") up to bfloat16 rounding of p - lr * g (LOMO
    adds in float32 and rounds once, torch's bfloat16 add_ rounds its own way), LR schedule included."""
    lomo = pytest.importorskip("lomo_optim")
    base = _tiny_model()
    head = copy.deepcopy(base)
    trainer = _train(af, sr, head, monkeypatch, "lomo", 0.0, steps = 3, tmp_path = tmp_path)
    wrapper = trainer._unsloth_in_backward_optimizer
    assert type(wrapper.optimizer) is torch.optim.SGD and trainer._unsloth_updates_in_backward
    assert not _weights_equal(base, head)
    sgd = copy.deepcopy(base)
    _train(af, sr, sgd, monkeypatch, "0", 0.0, optim = "sgd", steps = 3, tmp_path = tmp_path)
    assert _weights_equal(sgd, head)
    reference = copy.deepcopy(base)
    lomo_trainer = _train(
        af, sr, reference, monkeypatch, "0", 0.0, optim = "lomo", steps = 3, tmp_path = tmp_path
    )
    assert isinstance(af._unwrap(lomo_trainer.optimizer), lomo.Lomo)
    for (name, x), y in zip(reference.state_dict().items(), head.state_dict().values()):
        assert torch.allclose(x.float(), y.float(), rtol = 2**-7, atol = 1e-6), name


def test_sgd_tier_does_not_load_adamw_moments(af):
    model = torch.nn.Linear(4, 3)
    adamw = torch.optim.AdamW(model.parameters(), lr = 1e-2)
    model(torch.randn(2, 4)).sum().backward()
    adamw.step()
    wrapper = af.InBackwardOptimizer(af._sgd_from(adamw))
    wrapper.load_state_dict(adamw.state_dict())  # resuming an AdamW checkpoint with the SGD tier
    assert len(wrapper.optimizer.state) == 0
    assert wrapper.param_groups[0]["lr"] == 1e-2 and wrapper.param_groups[0]["momentum"] == 0.0
    before = model.weight.detach().clone()
    with wrapper.armed():
        model(torch.randn(2, 4)).sum().backward()
    wrapper.step()  # SGD's own group options survived the load
    assert not torch.equal(before, model.weight)


def test_token_budget_and_layer_major_see_the_in_backward_step(af):
    tb = _load("_test_auto_fit_tb", "unsloth/utils/token_budget.py")
    trainer = _Trainer(optim = "adamw_8bit")
    trainer._unsloth_updates_in_backward = True
    assert "every backward" in tb._exactness_blocker(trainer)
    assert af.updates_in_backward(trainer)
    source = (ROOT / "unsloth/models/_layer_major_ga.py").read_text(encoding = "utf-8")
    assert '"_unsloth_updates_in_backward"' in source


class _ConvNet(torch.nn.Module):
    def __init__(self, experts = True):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            8, 8, 4, groups = 8, bias = False
        )  # [8, 1, 4], as Inkling's *_sconv
        self.proj = torch.nn.Linear(8, 8)
        # fused experts [E, out, in]
        self.experts = torch.nn.Parameter(torch.randn(3, 8, 8) * 0.1) if experts else None

    def forward(self, x):
        h = self.proj(self.conv(x.transpose(1, 2)).transpose(1, 2))
        if self.experts is None:
            return h.square().mean()
        return torch.einsum("bsd,eod->bseo", h, self.experts).square().mean()


def _adalomo(monkeypatch, model):
    lomo = pytest.importorskip("lomo_optim")
    # A CPU model: AdaLomo would put its state on the default CUDA device.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    return lomo.AdaLomo(model, lr = 1e-3)


@pytest.mark.parametrize(
    "experts, message",
    [
        (False, r"output with shape \[8\] doesn't match the broadcast shape \[8, 8\]"),
        (True, "must match"),
    ],
)
def test_adalomo_three_dimensional_weights_crash_without_the_fix(monkeypatch, experts, message):
    """lomo-optim itself: the depthwise conv gives the Inkling-2 error ([2304] vs [2304, 2304] there)."""
    torch.manual_seed(0)
    model = _ConvNet(experts = experts)
    optimizer = _adalomo(monkeypatch, model)
    with pytest.raises(RuntimeError, match = message):
        optimizer.fused_backward(model(torch.randn(2, 6, 8)), 1e-3)


def test_adalomo_state_fix(af, monkeypatch):
    torch.manual_seed(0)
    model = _ConvNet()
    optimizer = _adalomo(monkeypatch, model)
    kept_row = optimizer.exp_avg_sq_row["proj.weight"]
    assert af.fix_adalomo_state(optimizer) == 4  # conv + experts, rows and cols
    assert optimizer.exp_avg_sq_row["proj.weight"] is kept_row  # 2-D state untouched
    assert optimizer.exp_avg_sq_row["conv.weight"].shape == (8, 1)
    assert optimizer.exp_avg_sq_col["experts"].shape == (3, 8)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        optimizer.fused_backward(model(torch.randn(2, 6, 8)), 1e-3)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all() and not torch.equal(p, before[name]), name
    assert af.fix_adalomo_state(optimizer) == 0  # idempotent


@pytest.mark.skipif(not has_real_cuda(), reason = "bitsandbytes 8-bit optimizers need CUDA")
@pytest.mark.parametrize("name", ["AdamW8bit", "PagedAdamW8bit"])
def test_bitsandbytes_in_backward_matches_the_standard_step(af, name):
    bnb = pytest.importorskip("bitsandbytes")
    torch.manual_seed(0)
    base = torch.nn.Sequential(
        torch.nn.Embedding(512, 256), torch.nn.Linear(256, 256), torch.nn.Linear(256, 512)
    ).cuda()
    manager = bnb.optim.GlobalOptimManager.get_instance()

    def run(in_backward):
        model = copy.deepcopy(base)
        manager.register_module_override(model[0], "weight", {"optim_bits": 32})
        optimizer = getattr(bnb.optim, name)(model.parameters(), lr = 1e-3)
        stepper = af.InBackwardOptimizer(optimizer) if in_backward else optimizer
        g = torch.Generator(device = "cuda").manual_seed(3)
        for _ in range(3):
            ids = torch.randint(0, 512, (4, 16), device = "cuda", generator = g)
            if in_backward:
                with stepper.armed():
                    model(ids).float().square().mean().backward()
            else:
                model(ids).float().square().mean().backward()
            stepper.step()
            stepper.zero_grad()
        return model, optimizer

    standard, s_opt = run(False)
    head, h_opt = run(True)
    assert _weights_equal(standard, head)
    # The embedding kept its 32-bit override, the others 8-bit state.
    assert h_opt.state[head[0].weight]["state1"].dtype == torch.float32
    assert h_opt.state[head[1].weight]["state1"].dtype == torch.uint8


class _Blocks(torch.nn.Module):
    """Two blocks under reentrant checkpointing (as Unsloth's offloaded checkpointing runs one backward per
    block); `shared` reuses one weight in both blocks."""

    def __init__(self, shared):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(8, 8)
        self.b = self.a if shared else torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint

        x = x.requires_grad_()
        h = checkpoint(lambda t: torch.tanh(self.a(t)), x, use_reentrant = True)
        h = checkpoint(lambda t: torch.tanh(self.b(t)), h, use_reentrant = True)
        return self.head(h).square().mean()


def test_reentrant_checkpointing_matches_the_standard_step(af):
    x = torch.randn(4, 8)
    standard, head = _Blocks(False), _Blocks(False)
    opt = torch.optim.AdamW(standard.parameters(), lr = 1e-2)
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(head.parameters(), lr = 1e-2))
    for _ in range(3):
        standard(x).backward()
        opt.step()
        opt.zero_grad()
        with wrapper.armed():
            head(x).backward()
        wrapper.step()
    assert _weights_equal(standard, head)


def test_a_weight_reused_across_checkpointed_blocks_raises_without_the_model(af):
    """Each block's backward would step the shared weight before the earlier block recomputes with it."""
    model = _Blocks(True)
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(model.parameters(), lr = 1e-2))
    with wrapper.armed(), pytest.raises(RuntimeError, match = "second gradient"):
        model(torch.randn(4, 8)).backward()


def test_a_weight_reused_across_checkpointed_blocks_is_stepped_at_the_end(af):
    """With the model, its module running twice in the forward marks the weight reused (as a looped block or
    Inkling-2's loop layers): it keeps its gradient until optimizer.step(), so the result is the standard one."""
    x = torch.randn(4, 8)
    standard, head = _Blocks(True), _Blocks(True)
    opt = torch.optim.AdamW(standard.parameters(), lr = 1e-2)
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(head.parameters(), lr = 1e-2), model = head)
    for _ in range(3):
        standard(x).backward()
        opt.step()
        opt.zero_grad()
        with wrapper.armed():
            head(x).backward()
        wrapper.step()
    assert _weights_equal(standard, head)
    assert wrapper.deferred_steps == 3 * 2  # the shared Linear's weight and bias, every step


def test_optimizer_enum_names_are_read_by_value(af):
    """TrainingArguments stores optim as an OptimizerNames enum: `adamw_bnb_8bit` prints as
    `OptimizerNames.ADAMW_BNB`, which must still be planned as 8-bit state."""
    from transformers.training_args import OptimizerNames

    model = _plan_model()
    by_string = af.plan_auto_fit(_Trainer(optim = "adamw_8bit"), model, usable_bytes = 0)
    for name in ("adamw_bnb_8bit", "adamw_8bit"):
        plan = af.plan_auto_fit(_Trainer(optim = OptimizerNames(name)), model, usable_bytes = 0)
        assert plan["states"] == by_string["states"], name
    torch_plan = af.plan_auto_fit(
        _Trainer(optim = OptimizerNames("adamw_torch")), model, usable_bytes = 0
    )
    assert torch_plan["states"] == 2 * by_string["grads"]  # two bfloat16 moments


def test_a_new_optimizer_drops_the_previous_in_backward_step(af, sr, monkeypatch, tmp_path):
    model = _tiny_model()
    trainer = _train(af, sr, model, monkeypatch, "in_backward", 0.0, steps = 1, tmp_path = tmp_path)
    old = trainer._unsloth_in_backward_optimizer
    monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", "0")
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.create_optimizer()
    assert "_unsloth_in_backward_optimizer" not in trainer.__dict__
    assert not getattr(trainer, "_unsloth_updates_in_backward", False)
    assert old._handles == []  # hooks removed
    assert type(af._unwrap(trainer.optimizer)) is torch.optim.AdamW


def test_an_active_grad_scaler_blocks_both_tiers(af):
    trainer = _Trainer()
    trainer.accelerator = type("A", (), {"scaler": object(), "mixed_precision": "no"})()
    assert "fp16" in af._blocker(trainer, _plan_model(), forced = True)


class _Looped(torch.nn.Module):
    """One block run twice under reentrant checkpointing; its Linear is read functionally (F.linear on the child's
    weight, never calling the child), as Inkling-2's loop convolutions are."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.block = torch.nn.Module()
        self.block.proj = torch.nn.Linear(8, 8)
        self.block.forward = lambda t: torch.tanh(
            torch.nn.functional.linear(t, self.block.proj.weight, self.block.proj.bias)
        )
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint

        h = x.requires_grad_()
        for _ in range(2):
            h = checkpoint(self.block, h, use_reentrant = True)
        return self.head(h).square().mean()


def test_a_functionally_read_weight_in_a_looped_block_is_stepped_at_the_end(af):
    x = torch.randn(4, 8)
    standard, head = _Looped(), _Looped()
    opt = torch.optim.AdamW(standard.parameters(), lr = 1e-2)
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(head.parameters(), lr = 1e-2), model = head)
    for _ in range(3):
        standard(x).backward()
        opt.step()
        opt.zero_grad()
        with wrapper.armed():
            head(x).backward()
        wrapper.step()
    assert _weights_equal(standard, head)
    assert wrapper.deferred_steps == 3 * 2


def test_planner_budgets_tied_and_looped_gradients(af):
    model = _tiny_model()  # tied embedding and head
    params = af._trainable(model)
    tied = model.get_input_embeddings().weight
    assert af._reused_bytes(model, params) == tied.numel() * tied.element_size()
    model.config.loop_config = {
        "num_passes": 3,
        "core_start": 1,
        "core_end": 2,
    }  # layer 1 runs three times
    layer = sum(p.numel() * p.element_size() for p in model.model.layers[1].parameters())
    assert af._reused_bytes(model, params) == tied.numel() * tied.element_size() + layer


def test_resuming_keeps_the_optimizer_family(af, sr, monkeypatch, tmp_path):
    """A run trained with the SGD tier records it in its checkpoints; resuming in the default mode keeps SGD (its
    groups lack AdamW's options), even where the standard step would now fit (no CUDA here: no prediction)."""
    import json

    first = _train(
        af, sr, _tiny_model(), monkeypatch, "lomo", 0.0, steps = 2, tmp_path = tmp_path, save_steps = 1
    )
    checkpoint = tmp_path / "checkpoint-2"
    assert json.loads((checkpoint / "unsloth_auto_fit.json").read_text()) == {"tier": "sgd"}
    resumed = _train(
        af,
        sr,
        _tiny_model(),
        monkeypatch,
        "",
        0.0,
        steps = 3,
        tmp_path = tmp_path,
        resume = str(checkpoint),
    )
    assert type(resumed._unsloth_in_backward_optimizer.optimizer) is torch.optim.SGD
    assert resumed.state.global_step == 3
    del first


def test_reentrant_checkpointing_steps_one_batch_per_block(af, monkeypatch):
    """Each checkpointed block's backward is its own graph task: its gradients (and the head's, ready before it)
    are stepped together when it ends, not one launch sequence per weight."""
    x = torch.randn(4, 8)
    model = _Blocks(False)
    wrapper = af.InBackwardOptimizer(
        torch.optim.AdamW(model.parameters(), lr = 1e-2), batch_bytes = 1 << 20
    )
    batches = []
    step_params = wrapper._step_params
    monkeypatch.setattr(
        wrapper, "_step_params", lambda params: (batches.append(len(params)), step_params(params))
    )
    with wrapper.armed():
        model(x).backward()
        assert all(p.grad is None for p in model.parameters())  # stepped inside backward
    wrapper.step()
    assert [b for b in batches if b] == [4, 2]  # head + block b, then block a
    assert wrapper.hook_steps == 6 and wrapper.late_steps == 0


class _Trial(torch.nn.Module):
    """A block whose first call also runs a throwaway forward under Unsloth's block-trial flag (as compiled
    decoder blocks check their region on the first call), then the real one."""

    def __init__(self, flag):
        super().__init__()
        torch.manual_seed(0)
        self.flag = flag
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        self.flag[0] = True
        try:
            with torch.enable_grad():
                self.proj(x.detach().requires_grad_())  # the trial: no gradient reaches the weight
        finally:
            self.flag[0] = False
        return self.head(torch.tanh(self.proj(x))).square().mean()


def test_a_block_trial_forward_is_not_a_reuse(af, monkeypatch):
    """Unsloth's first-call block trial ran every block's modules twice on step 1, so every layer's gradient was
    kept until optimizer.step() (all layer gradients alive at once on the first step)."""
    import sys
    import types

    flag = [False]
    monkeypatch.setitem(
        sys.modules, "unsloth.models._block_compile", types.SimpleNamespace(_IN_TRIAL = flag)
    )
    x = torch.randn(4, 8)
    standard, head = _Trial(flag), _Trial(flag)
    opt = torch.optim.AdamW(standard.parameters(), lr = 1e-2)
    wrapper = af.InBackwardOptimizer(torch.optim.AdamW(head.parameters(), lr = 1e-2), model = head)
    for _ in range(2):
        standard(x).backward()
        opt.step()
        opt.zero_grad()
        with wrapper.armed():
            head(x).backward()
        wrapper.step()
    assert _weights_equal(standard, head)
    assert wrapper.deferred_steps == 0 and wrapper.hook_steps == 2 * 4


def test_reuse_counting_does_not_break_a_compiled_block(af, monkeypatch):
    """The reuse counter's forward pre-hooks sit on every module that holds a weight, including the ones inside
    Unsloth's compiled decoder blocks: they must compile to nothing instead of splitting the graph."""
    import sys

    pytest.importorskip("torch._dynamo")
    monkeypatch.setitem(sys.modules, af.__name__, af)  # dynamo imports the hook's module by name

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(8, 8)
            self.b = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.b(torch.tanh(self.a(x)))

    torch.manual_seed(0)
    model = torch.nn.Sequential(Block(), torch.nn.Linear(8, 1))
    wrapper = af.InBackwardOptimizer(torch.optim.SGD(model.parameters(), lr = 1e-2), model = model)
    torch._dynamo.reset()
    compiled = torch.compile(model[0], backend = "eager", fullgraph = True)
    x = torch.randn(4, 8)
    with wrapper.armed():
        model[1](compiled(x)).square().mean().backward()  # fullgraph: a graph break raises
    wrapper.step()
    assert wrapper.hook_steps == 6 and wrapper.deferred_steps == 0
    torch._dynamo.reset()


def test_deferred_weights_are_stepped_in_bounded_batches(af, monkeypatch):
    """The weights left for optimizer.step() (a looped block's) are stepped in batches of at most batch_bytes, so a
    multi-tensor step's temporaries never span every deferred weight at once."""
    x = torch.randn(4, 8)
    standard, head = _Looped(), _Looped()
    opt = torch.optim.AdamW(standard.parameters(), lr = 1e-2)
    weight = head.block.proj.weight
    wrapper = af.InBackwardOptimizer(
        torch.optim.AdamW(head.parameters(), lr = 1e-2),
        model = head,
        batch_bytes = weight.numel() * weight.element_size(),
    )
    sizes = []
    step_params = wrapper._step_params
    monkeypatch.setattr(
        wrapper,
        "_step_params",
        lambda params: (
            sizes.append(
                sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
            ),
            step_params(params),
        ),
    )
    for _ in range(2):
        standard(x).backward()
        opt.step()
        opt.zero_grad()
        with wrapper.armed():
            head(x).backward()
        wrapper.step()
    assert _weights_equal(standard, head)
    assert wrapper.deferred_steps == 2 * 2
    assert max(sizes) <= wrapper.batch_bytes


def test_ready_batches_never_exceed_batch_bytes(af, monkeypatch):
    """A gradient that would push the waiting batch past batch_bytes steps that batch first; only a weight larger
    than batch_bytes is stepped alone, so a multi-tensor update never spans more than the planner budgets."""
    model = _tiny_model().float()
    params = [p for p in model.parameters() if p.requires_grad]
    size = lambda q: q.numel() * q.element_size()
    limit = 3 * max(size(p) for p in params if p.dim() == 2 and p.shape[0] != 96)
    wrapper = af.InBackwardOptimizer(
        torch.optim.AdamW(model.parameters(), lr = 1e-3), batch_bytes = limit
    )
    batches = []
    step_params = wrapper._step_params
    monkeypatch.setattr(
        wrapper,
        "_step_params",
        lambda ps: (batches.append([size(p) for p in ps if p.grad is not None]), step_params(ps)),
    )
    ids = torch.randint(0, 96, (2, 12))
    with wrapper.armed():
        model(input_ids = ids, labels = ids).loss.backward()
    wrapper.step()
    batches = [b for b in batches if b]
    assert len(batches) > 1 and any(len(b) > 1 for b in batches)
    assert all(sum(b) <= limit or len(b) == 1 for b in batches), batches


def test_planner_budgets_the_batch_workspace(af, monkeypatch):
    """While a batch steps, the gradient that arrived is alive too, and torch's foreach AdamW without the one-pass
    kernel adds a moment-sized temporary over the batch: one more largest weight in the in-backward budget."""
    model = _plan_model()
    largest = max(p.numel() * p.element_size() for p in model.parameters())
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    one_pass = af.plan_auto_fit(_Trainer(optim = "adamw_torch"), model, usable_bytes = 0)
    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "0")
    foreach = af.plan_auto_fit(_Trainer(optim = "adamw_torch"), model, usable_bytes = 0)
    assert foreach["in_backward"] - one_pass["in_backward"] == largest
    assert one_pass["in_backward"] - one_pass["states"] - (256 << 20) == 2 * largest


class _CompiledNet(torch.nn.Module):
    """A forward through a dynamic-shape torch.compile'd function (as Unsloth's compiled blocks are)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(6, 8)
        self.b = torch.nn.Linear(8, 1)
        self.body = torch.compile(self._body, backend = "eager", dynamic = True)

    def _body(self, x, labels):
        return (self.b(torch.relu(self.a(x))).squeeze(-1) - labels).square().mean()

    def forward(self, x, labels):
        return {"loss": self.body(x, labels)}


def test_measuring_pass_does_not_recompile_the_real_pass(af, sr, monkeypatch, tmp_path):
    """The measuring pass and the stepping pass hit the same compiled graphs: the measuring pass's inputs are new
    tensor objects that are not views (a graph compiled on a view guards on its `_base`, which the real pass's
    tensors do not have, so every graph recompiled once at the first step)."""
    import sys

    from datasets import Dataset
    from torch._dynamo.utils import counters
    from transformers import Trainer, TrainingArguments

    # A module of its own name: dynamo imports the module of code it traces by name, and torch 2.6 keeps the module
    # an earlier test registered under the fixture's name.
    af = _load("_test_auto_fit_recompile", "unsloth/optimizers/auto_fit.py")
    monkeypatch.setitem(sys.modules, af.__name__, af)
    g = torch.Generator().manual_seed(2)
    rows = [{"x": torch.randn(6, generator = g).tolist(), "labels": float(i % 3)} for i in range(16)]
    graphs = {}
    for mode, clip in (("0", "estimate"), ("in_backward", "estimate"), ("in_backward", "exact")):
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_OPTIMIZER", mode)
        monkeypatch.setenv("UNSLOTH_AUTO_FIT_CLIP", clip)
        torch._dynamo.reset()
        counters.clear()
        args = TrainingArguments(
            output_dir = str(tmp_path),
            per_device_train_batch_size = 4,
            max_steps = 3,
            learning_rate = 1e-2,
            max_grad_norm = 0.5,
            report_to = [],
            use_cpu = True,
            save_strategy = "no",
            seed = 0,
            dataloader_num_workers = 0,
            disable_tqdm = True,
        )
        trainer = Trainer(model = _CompiledNet(), args = args, train_dataset = Dataset.from_list(rows))
        af.install_auto_fit_optimizer(trainer)
        trainer.train()
        if mode != "0":
            assert trainer._unsloth_in_backward_optimizer.measured_steps == (
                3 if clip == "exact" else 1
            )
        graphs[mode, clip] = counters["stats"]["unique_graphs"]
    torch._dynamo.reset()
    assert graphs["0", "estimate"] >= 1
    assert (
        graphs["in_backward", "estimate"]
        == graphs["in_backward", "exact"]
        == graphs["0", "estimate"]
    ), graphs
