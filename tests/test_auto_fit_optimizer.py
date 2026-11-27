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


def test_each_gradient_is_freed_inside_backward(af):
    """At most the gradient being stepped (and a tied one still accumulating) is alive during backward."""
    model = _tiny_model().float()
    base = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    wrapper = af.InBackwardOptimizer(base)
    params = [p for p in model.parameters() if p.requires_grad]
    live = []
    for p in params:
        p.register_post_accumulate_grad_hook(
            lambda _p: live.append(sum(q.grad is not None for q in params))
        )
    ids = torch.randint(0, 96, (2, 12))
    with wrapper.armed():
        model(input_ids = ids, labels = ids).loss.backward()
    wrapper.step()
    assert all(p.grad is None for p in params)
    # Our hook ran first and freed the previous gradients: only this one's (plus the tied head's) remain.
    assert max(live) <= 2, live
    assert len(base.state) == len(params)


def test_per_tensor_clipping(af):
    """max_grad_norm clips each gradient by its own norm (torch's coefficient with that norm)."""
    a = torch.nn.Parameter(torch.zeros(4))
    b = torch.nn.Parameter(torch.zeros(3))
    base = torch.optim.SGD([a, b], lr = 1.0)
    wrapper = af.InBackwardOptimizer(base, max_grad_norm = 1.0)
    ga = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm 5 -> clipped to 1
    gb = torch.tensor([0.3, 0.4, 0.0])  # norm 0.5 -> untouched
    with wrapper.armed():
        ((a * ga).sum() + (b * gb).sum()).backward()
    assert torch.allclose(a.detach(), -ga / (5 + 1e-6), atol = 1e-7)
    assert torch.equal(b.detach(), -gb)
    assert abs(wrapper.total_norm().item() - (25 + 0.25) ** 0.5) < 1e-5  # pre-clip global norm
    wrapper.step()
    assert wrapper.total_norm().item() == 0.0


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
