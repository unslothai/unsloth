# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the checkpointing budget in the SDXL, MiniMax-H3 and MiniMax-H3-Image trainers, its step-down after an
out-of-memory error (instead of checkpointing every block), and planning past a block that does not fit."""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

import core.training.diffusion_gc_budget as gb

GIB = 1 << 30


def _planned(class_cost, blocks):
    budget = gb.BudgetedCheckpoint()
    budget.class_cost = dict(class_cost)
    for idx, (cls, proxy) in enumerate(blocks):
        budget.block_class[idx] = cls
        budget.block_proxy[idx] = proxy
    return budget


# --- plan + step-down arithmetic ---------------------------------------------------------------------------------


def test_plan_keeps_cheaper_blocks_after_one_that_does_not_fit():
    # (36 - 4 - 4) * 0.85 = 23.8 GiB: block 1 (30 GiB) cannot fit, the three 4 GiB blocks after it still do.
    budget = _planned({"A": 4.0, "B": 30.0}, [("A", GIB), ("B", GIB), ("A", GIB), ("A", GIB)])
    assert budget.plan(4 * GIB, 36 * GIB, 0, 40 * GIB) == 3
    assert budget.keep == frozenset({0, 2, 3})
    assert budget.kept_bytes == 12 * GIB
    assert budget.spare_bytes == int(23.8 * GIB) - 12 * GIB


def test_step_down_halves_the_kept_bytes_until_nothing_is_left():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 8)
    assert budget.plan(0, 80 * GIB, 0, 80 * GIB) == 8
    kept = []
    while budget.step_down():
        kept.append(sorted(budget.keep))
        assert budget.can_retry and not budget.disabled
    assert kept == [[0, 1, 2, 3], [0, 1], [0]]
    assert budget.step_downs == 3
    assert budget.disabled and budget.keep == frozenset() and not budget.can_retry
    assert budget.spare_bytes == 0
    assert budget.step_down() is False


def test_step_down_drops_at_least_one_block():
    # One kept block holds most of the bytes: half of them still drops a block.
    budget = _planned({"A": 10.0, "B": 1.0}, [("A", GIB), ("B", GIB)])
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    assert budget.keep == frozenset({0, 1})
    assert budget.step_down() is True
    assert budget.keep == frozenset({0}) or budget.keep == frozenset({1})
    assert len(budget.keep) == 1


def test_step_down_while_measuring_checkpoints_every_block():
    budget = gb.BudgetedCheckpoint()
    assert budget.probing and budget.can_retry
    assert budget.step_down() is False
    assert budget.disabled and not budget.can_retry


def test_step_down_scales_like_the_plan():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 8)
    budget.probe_area = 100
    # Every block costs 2 GiB at the largest bucket: 4 fit in 8 GiB of headroom, the step-down keeps 2 (4 GiB).
    budget.plan(0, 80 * GIB, 0, 80 * GIB, max_canvas_area = 200)
    budget.keep = frozenset(range(4))
    budget.kept_bytes = 8 * GIB
    assert budget.step_down() is True
    assert budget.keep == frozenset({0, 1}) and budget.kept_bytes == 4 * GIB


# --- BudgetedSteps: the loop helper the three trainers use ----------------------------------------------------------


class _Optim:
    def __init__(self):
        self.zeroed = 0

    def zero_grad(self, set_to_none = True):
        self.zeroed += 1


def _loop(budget, fail_calls):
    """A 3-micro-batch step whose forward draws RNG; calls in ``fail_calls`` raise out of memory after their draw."""
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        r = torch.rand(3)
        if calls["n"] in fail_calls:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return x + r

    warnings = []
    opt = _Optim()
    steps = gb.BudgetedSteps(budget, "cpu", opt, warn = warnings.append)
    torch.manual_seed(0)
    steps.start_step()
    results = []
    for i in range(3):
        out = steps.forward_backward(fn, (torch.full((3,), float(i)),))
        if out is gb.OUT_OF_MEMORY:
            results = steps.rerun(fn)
        else:
            results.append(out)
    return results, torch.rand(1), warnings, opt, steps


@pytest.mark.parametrize("fail_calls", [{2}, {3}, {2, 3}, {3, 4, 5}])
def test_rerun_replays_the_step_bit_identically(fail_calls):
    ref, ref_after, _w, _o, _s = _loop(_planned({"B": 1.0}, [("B", GIB)] * 8), set())
    budget = _planned({"B": 1.0}, [("B", GIB)] * 8)
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    got, after, warnings, opt, steps = _loop(budget, fail_calls)
    assert len(got) == 3
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
    # The RNG is left where one clean pass leaves it.
    assert torch.equal(after, ref_after)
    # Two failures inside one rerun step down twice; each step-down zeroes the step's partial gradients.
    assert opt.zeroed == len(warnings) == budget.step_downs == steps.reruns
    assert all("Ran out of GPU memory" in w for w in warnings)
    assert budget.keep and not budget.disabled


def test_no_budget_is_a_plain_call():
    results, *_ = _loop(None, set())
    assert len(results) == 3
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _loop(None, {1})


def test_out_of_memory_without_anything_to_drop_is_raised():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 8)
    budget.plan(0, 0, 0, 80 * GIB)  # no room: nothing kept
    assert not budget.can_retry
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _loop(budget, {2})


def test_other_errors_are_never_retried():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 8)
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    steps = gb.BudgetedSteps(budget, "cpu", _Optim())

    def boom(_x):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match = "shape mismatch"):
        steps.forward_backward(boom, (torch.zeros(1),))
    assert budget.step_downs == 0


def test_after_update_plans_once(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 5)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 3)
    monkeypatch.setattr(
        gb,
        "plan_from_device",
        lambda b, peak, alloc, area = None, held = 0: seen.append((peak, alloc, area))
        or b.plan(0, 80 * GIB, 0, 80 * GIB),
    )
    budget = _planned({"B": 1.0}, [("B", GIB)] * 2)
    steps = gb.BudgetedSteps(budget, "cuda", _Optim())
    for _ in range(3):
        steps.before_update()
        steps.after_update(7)
    assert seen == [(5, 3, 7)]


# --- the three trainers route checkpointing through the budget --------------------------------------------------


@pytest.mark.parametrize(
    "module, function",
    [
        ("core.training.diffusion_lora_trainer", "run_diffusion_lora_training"),
        ("core.training.diffusion_h3_trainer", "_train_h3"),
        ("core.training.diffusion_h3_image_trainer", "_train"),
    ],
)
def test_trainer_uses_the_budget_and_its_replay(module, function):
    import importlib

    mod = importlib.import_module(module)
    src = inspect.getsource(getattr(mod, function))
    # The plain path stays the fallback (ROCm, non-CUDA, UNSLOTH_DIFFUSION_GC_MODE=plain|partial).
    assert (
        "enable_budgeted_checkpointing(" in src
        and "enable_diffusion_gradient_checkpointing(" in src
    )
    for call in (
        "BudgetedSteps(",
        ".forward_backward(",
        ".rerun(",
        ".before_update()",
        ".after_update(",
    ):
        assert call in src, call


# --- end to end on CPU: tiny SDXL and tiny MiniMax-H3-Image -------------------------------------------------------


def _force_budget(monkeypatch, trainer_module, budgets):
    """CPU stand-in for CUDA: the trainer gets a real BudgetedCheckpoint, planned to keep every block."""
    counter = iter(range(0, 1 << 50, 1 << 20))
    monkeypatch.setattr(gb, "_memory_allocated", lambda: next(counter))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)

    def enable(model, device):
        b = gb.BudgetedCheckpoint()
        model.enable_gradient_checkpointing(gradient_checkpointing_func = b)
        budgets.append(b)
        return b

    monkeypatch.setattr(trainer_module, "enable_budgeted_checkpointing", enable)
    monkeypatch.setattr(gb, "plan_from_device", lambda b, *a, **k: b.plan(0, 80 * GIB, 0, 80 * GIB))


def _failing(real, fail_calls):
    """``real`` that runs out of memory on the calls in ``fail_calls`` (after drawing its RNG, as a real failure has)."""
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        if calls["n"] in fail_calls:
            torch.rand(100)
            real(*a, **k)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(*a, **k)

    return wrapped


@pytest.fixture
def tiny_sdxl():
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
    except Exception as exc:  # noqa: BLE001 -- offline CI
        pytest.skip(f"tiny SDXL pipe unavailable: {exc}")


def _run_sdxl(
    tmp_path,
    monkeypatch,
    tiny_sdxl,
    *,
    budget,
    fail_calls = frozenset(),
):
    from PIL import Image

    from core.training import diffusion_lora_trainer as trainer
    from core.training.diffusion_train_common import DiffusionLoraConfig

    data = tmp_path / "data"
    data.mkdir(parents = True, exist_ok = True)
    for i in range(3):
        Image.new("RGB", (64, 64), (200, 40 * i, 30)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(f"a tile {i}")
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: tmp_path / "catalog")
    (tmp_path / "catalog").mkdir(exist_ok = True)
    monkeypatch.setattr(trainer, "_assert_trusted_base_model", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    budgets = []
    if budget:
        _force_budget(monkeypatch, trainer, budgets)
    from diffusers import UNet2DConditionModel

    # One U-Net forward per micro-batch.
    monkeypatch.setattr(
        UNet2DConditionModel, "forward", _failing(UNet2DConditionModel.forward, set(fail_calls))
    )
    events = []
    out = tmp_path / "out"
    cfg = DiffusionLoraConfig(
        base_model = tiny_sdxl,
        model_family = "sdxl",
        data_dir = str(data),
        output_dir = str(out),
        resolution = 64,
        train_steps = 3,
        gradient_accumulation_steps = 2,
        learning_rate = 1e-2,
        lora_rank = 4,
        lora_dropout = 0.1,
        cache_variants = 1,
        seed = 0,
        compile_transformer = "off",
    )
    trainer.run_diffusion_lora_training(cfg, on_event = events.append)
    from safetensors.torch import load_file

    lora = load_file(str(out / "pytorch_lora_weights.safetensors"))
    losses = [e["loss"] for e in events if e.get("type") == "progress"]
    warnings = [e["message"] for e in events if e.get("type") == "warning"]
    return lora, losses, warnings, budgets


@pytest.mark.allow_network
def test_sdxl_budget_keeps_blocks_and_matches_full_checkpointing(tmp_path, monkeypatch, tiny_sdxl):
    torch.use_deterministic_algorithms(True)
    try:
        ref_lora, ref_losses, _w, _b = _run_sdxl(
            tmp_path / "plain", monkeypatch, tiny_sdxl, budget = False
        )
        monkeypatch.undo()
        lora, losses, _w, budgets = _run_sdxl(
            tmp_path / "budget", monkeypatch, tiny_sdxl, budget = True
        )
        monkeypatch.undo()
        # Step 2's second micro-batch, after the plan kept every block.
        oom_lora, oom_losses, warnings, oom_budgets = _run_sdxl(
            tmp_path / "oom", monkeypatch, tiny_sdxl, budget = True, fail_calls = {4}
        )
    finally:
        torch.use_deterministic_algorithms(False)
    b = budgets[0]
    assert b.planned and b.total_blocks > 1 and len(b.keep) == b.total_blocks
    assert set(b.class_cost) >= {"ResnetBlock2D"}
    assert losses == ref_losses
    for k in ref_lora:
        assert torch.equal(ref_lora[k], lora[k]), k
    assert any("Ran out of GPU memory" in w for w in warnings)
    ob = oom_budgets[0]
    assert ob.step_downs == 1 and ob.keep and not ob.disabled
    assert oom_losses == ref_losses
    for k in ref_lora:
        assert torch.equal(ref_lora[k], oom_lora[k]), k


@pytest.fixture(scope = "module")
def h3_image_release(tmp_path_factory):
    for mod in ("diffusers", "transformers", "accelerate", "peft"):
        pytest.importorskip(mod)
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_budget_release")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return root, ref, te


@pytest.fixture(scope = "module")
def h3_image_dataset(tmp_path_factory):
    import numpy as np
    from PIL import Image

    d = tmp_path_factory.mktemp("h3img_budget_data")
    rng = np.random.default_rng(0)
    for i in range(2):
        pic = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        Image.fromarray(pic).save(d / f"a{i}.png")
        (d / f"a{i}.txt").write_text(f"a cat {i}")
    return d


def _run_h3_image(
    release,
    dataset,
    out,
    monkeypatch,
    *,
    budget,
    fail_calls = frozenset(),
):
    from core.training import diffusion_h3_image_trainer as trainer
    from core.training.diffusion_train_common import DiffusionLoraConfig

    root, ref, te = release
    monkeypatch.setattr(
        trainer, "load_text_encoder", lambda cfg, r: (te, ref["tokenizer"], ref["processor"])
    )
    monkeypatch.setattr(trainer, "release_text_encoder", lambda m: None)
    monkeypatch.setattr(trainer, "resolve_train_device", lambda: "cpu")
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, "off")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    budgets = []
    if budget:
        _force_budget(monkeypatch, trainer, budgets)
    # One denoiser call per training micro-batch (no previews, no eval here).
    monkeypatch.setattr(
        trainer, "denoise_velocity", _failing(trainer.denoise_velocity, set(fail_calls))
    )
    events = []
    cfg = DiffusionLoraConfig(
        base_model = str(root),
        data_dir = str(dataset),
        output_dir = str(out),
        model_family = "minimax-h3-image",
        resolution = 64,
        train_steps = 3,
        gradient_accumulation_steps = 2,
        learning_rate = 3e-3,
        lora_rank = 4,
        lora_dropout = 0.1,
        lr_scheduler = "constant",
        seed = 0,
    )
    trainer.run_h3_image_lora_training(cfg, on_event = events.append)
    losses = [e["loss"] for e in events if e.get("type") == "progress"]
    warnings = [e["message"] for e in events if e.get("type") == "warning"]
    from safetensors.torch import load_file

    files = sorted(out.glob("*.safetensors"))
    lora = load_file(str(files[0])) if files else {}
    return lora, losses, warnings, budgets


@pytest.mark.allow_network
def test_h3_image_budget_keeps_blocks_and_matches_full_checkpointing(
    h3_image_release, h3_image_dataset, tmp_path, monkeypatch
):
    torch.use_deterministic_algorithms(True)
    try:
        ref_lora, ref_losses, _w, _b = _run_h3_image(
            h3_image_release, h3_image_dataset, tmp_path / "plain", monkeypatch, budget = False
        )
        monkeypatch.undo()
        lora, losses, _w, budgets = _run_h3_image(
            h3_image_release, h3_image_dataset, tmp_path / "budget", monkeypatch, budget = True
        )
        b = budgets[0]
        n_blocks = b.total_blocks
        monkeypatch.undo()
        # Step 2's second micro-batch, after the plan kept every block.
        oom_lora, oom_losses, warnings, oom_budgets = _run_h3_image(
            h3_image_release,
            h3_image_dataset,
            tmp_path / "oom",
            monkeypatch,
            budget = True,
            fail_calls = {4},
        )
    finally:
        torch.use_deterministic_algorithms(False)
    assert ref_lora and ref_lora.keys() == lora.keys()
    assert b.planned and n_blocks > 1 and len(b.keep) == n_blocks
    assert losses == ref_losses
    for k in ref_lora:
        assert torch.equal(ref_lora[k], lora[k]), k
    assert any("Ran out of GPU memory" in w for w in warnings)
    assert oom_budgets[0].step_downs == 1
    assert oom_losses == ref_losses
    for k in ref_lora:
        assert torch.equal(ref_lora[k], oom_lora[k]), k
