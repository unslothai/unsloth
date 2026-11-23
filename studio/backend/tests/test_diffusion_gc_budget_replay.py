# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the out-of-memory rerun in the real DiT loop (tiny Qwen-Image-2.1, skipped when unreachable): only the
current micro-batch is held, and a rerun is bit-identical (adapter and RNG) to a run without the error."""

from __future__ import annotations

import sys
import weakref

import pytest
import torch

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_gc_budget as gb
from core.training.diffusion_train_common import DiffusionLoraConfig

FAMILY = "qwen-image-2.1"
TINY_PIPE = "hf-internal-testing/tiny-qwenimage21-pipe"
GIB = 1 << 30


@pytest.fixture
def tiny_pipe_dir():
    try:
        from diffusers import QwenImage21Pipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers Qwen-Image-2.1 classes unavailable: {exc}")
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(TINY_PIPE)
    except Exception as exc:  # noqa: BLE001 -- offline CI
        pytest.skip(f"{TINY_PIPE} unavailable: {exc}")


def _run(
    tmp_path,
    monkeypatch,
    tiny_pipe_dir,
    *,
    budget,
    oom_at = None,
    accum = 4,
    steps = 3,
    dropout = 0.1,
    precision = "bf16",
    **cfg_extra,
):
    """One CPU training run; returns (final LoRA tensors, per-call record of forward inputs, alive-at-forward counts)."""
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir(parents = True, exist_ok = True)
    for i in range(4):
        Image.new("RGB", (64, 64), (200, 30 * i, 40)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(f"a red tile {i}")
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: tmp_path / "catalog")
    (tmp_path / "catalog").mkdir(exist_ok = True)
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: precision)
    monkeypatch.setattr(dit, "_assert_trusted_base_model", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    counter = iter(range(0, 1 << 50, 1 << 20))
    monkeypatch.setattr(gb, "_memory_allocated", lambda: next(counter))

    budgets = []
    if budget:

        def enable(model, device):
            b = gb.BudgetedCheckpoint()
            model.enable_gradient_checkpointing(gradient_checkpointing_func = b)
            budgets.append(b)
            return b

        def plan(b, *a, **k):
            b.plan(0, 80 * GIB, 0, 80 * GIB)

        monkeypatch.setattr(dit, "enable_budgeted_checkpointing", enable)
        monkeypatch.setattr(dit, "_plan_gc_budget", plan)
    else:
        monkeypatch.setattr(dit, "enable_budgeted_checkpointing", lambda model, device: None)
        monkeypatch.setattr(
            dit, "enable_diffusion_gradient_checkpointing", lambda model, mode = None: "plain"
        )

    spec = dit._SPECS[FAMILY]
    load, fwd = spec.load_transformer, spec.forward
    monkeypatch.setattr(spec, "load_transformer", lambda *a, **k: load(*a, **k).float())
    calls, seen, alive, replaying = {"n": 0}, [], [], [False]
    if hasattr(gb, "StepReplay"):
        orig_replay = gb.StepReplay.replay

        def replay(self, fn):
            replaying[0] = True
            try:
                return orig_replay(self, fn)
            finally:
                replaying[0] = False

        monkeypatch.setattr(gb.StepReplay, "replay", replay)

    failed_refs, failed_alive_at_replay = [], []

    def _tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (tuple, list)):
            for o in obj:
                yield from _tensors(o)

    def forward(transformer, noisy, *a, **k):
        calls["n"] += 1
        if replaying[0] and failed_refs and not failed_alive_at_replay:
            # First replayed forward: which of the failed micro-batch's tensors (and its partial output) still live.
            failed_alive_at_replay.append(sum(1 for r in failed_refs if r() is not None))
        if not replaying[0]:
            # Earlier micro-batches' noisy latents still referenced by the loop when this forward starts.
            alive.append(sum(1 for r in seen if r() is not None))
            seen.append(weakref.ref(noisy))
        if calls["n"] == oom_at or (isinstance(oom_at, (set, frozenset)) and calls["n"] in oom_at):
            # A real failure consumes RNG (dropout masks drawn) before the allocator gives up.
            torch.rand(1000)
            out = fwd(transformer, noisy, *a, **k)
            failed_refs.extend(weakref.ref(t) for t in _tensors((noisy, a[:3], out)))
            del out
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return fwd(transformer, noisy, *a, **k)

    monkeypatch.setattr(spec, "forward", forward)
    records = []
    orig_sampled = dit._sample_timesteps

    def sample_timesteps(*a, **k):
        out = orig_sampled(*a, **k)
        records.append(out[0].clone())
        return out

    monkeypatch.setattr(dit, "_sample_timesteps", sample_timesteps)
    events = []
    out_dir = tmp_path / "out"
    cfg = DiffusionLoraConfig(
        base_model = tiny_pipe_dir,
        model_family = FAMILY,
        data_dir = str(data),
        output_dir = str(out_dir),
        resolution = 64,
        train_steps = steps,
        train_batch_size = 1,
        gradient_accumulation_steps = accum,
        learning_rate = 1e-2,
        lora_rank = 4,
        lora_dropout = dropout,
        cache_variants = 1,
        seed = 0,
        **cfg_extra,
    )
    dit.run_dit_lora_training(cfg, on_event = events.append)
    from safetensors.torch import load_file

    lora = load_file(str(out_dir / "pytorch_lora_weights.safetensors"))
    losses = [e["loss"] for e in events if e.get("type") == "progress"]
    warnings = [e["message"] for e in events if e.get("type") == "warning"]
    _run.failed_alive_at_replay = failed_alive_at_replay
    _run.failed_refs = len(failed_refs)
    return lora, losses, records, alive, warnings, budgets


@pytest.mark.allow_network
@pytest.mark.parametrize("budget", [False, True])
def test_accumulation_holds_one_micro_batch_on_the_device(
    budget, tmp_path, monkeypatch, tiny_pipe_dir
):
    _lora, _losses, _rec, alive, _w, budgets = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = budget
    )
    # 3 steps x 4 micro-batches; when each forward starts, no earlier micro-batch's tensors are still held.
    assert len(alive) == 12 and max(alive) == 0
    if budget:
        assert budgets[0].planned and budgets[0].keep


@pytest.mark.allow_network
def test_after_a_step_down_nothing_is_kept_for_a_rerun(tmp_path, monkeypatch, tiny_pipe_dir):
    _lora, _losses, _rec, alive, warnings, budgets = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = True, oom_at = 6
    )
    assert any("Ran out of GPU memory" in w for w in warnings)
    b = budgets[0]
    # Step 2 ran out with every block kept: half stay kept (not every block checkpointed), and a rerun stays possible.
    assert b.step_downs == 1 and not b.disabled and b.can_retry
    assert 0 < len(b.keep) <= b.total_blocks // 2
    # The replayed forwards run from host copies and are not counted; the originals are never held past their forward.
    assert len(alive) == 12 and max(alive) == 0


@pytest.mark.allow_network
def test_out_of_memory_during_measurement_checkpoints_every_block(
    tmp_path, monkeypatch, tiny_pipe_dir
):
    _lora, _losses, _rec, alive, warnings, budgets = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = True, oom_at = 2
    )
    assert any("every block is checkpointed" in w for w in warnings)
    assert budgets[0].disabled and not budgets[0].can_retry
    assert len(alive) == 12 and max(alive) == 0


@pytest.mark.allow_network
def test_repeated_out_of_memory_steps_down_until_it_fits(tmp_path, monkeypatch, tiny_pipe_dir):
    torch.use_deterministic_algorithms(True)
    try:
        ref_lora, ref_losses, *_ = _run(tmp_path / "ref", monkeypatch, tiny_pipe_dir, budget = True)
        monkeypatch.undo()
        # Call 6 fails (step 2), and so does the rerun's first forward (call 7): two step-downs within one step.
        lora, losses, _rec, _alive, warnings, budgets = _run(
            tmp_path / "oom", monkeypatch, tiny_pipe_dir, budget = True, oom_at = {6, 7}
        )
    finally:
        torch.use_deterministic_algorithms(False)
    oom_warnings = [w for w in warnings if "Ran out of GPU memory" in w]
    assert len(oom_warnings) == 2
    b = budgets[0]
    assert b.step_downs + int(b.disabled) == 2
    assert losses == ref_losses
    for k in ref_lora:
        assert torch.equal(ref_lora[k], lora[k]), k


@pytest.mark.allow_network
@pytest.mark.parametrize("oom_at", [2, 6, 8])
def test_a_rerun_after_out_of_memory_is_bit_identical(oom_at, tmp_path, monkeypatch, tiny_pipe_dir):
    torch.use_deterministic_algorithms(True)
    try:
        ref_lora, ref_losses, ref_rec, *_ = _run(
            tmp_path / "ref", monkeypatch, tiny_pipe_dir, budget = True
        )
        # Snapshot: the second run's patches wrap the first run's, which would keep appending to its record.
        ref_rec = list(ref_rec)
        monkeypatch.undo()
        lora, losses, rec, _alive, warnings, _b = _run(
            tmp_path / "oom", monkeypatch, tiny_pipe_dir, budget = True, oom_at = oom_at
        )
    finally:
        torch.use_deterministic_algorithms(False)
    assert any("Ran out of GPU memory" in w for w in warnings)
    assert losses == ref_losses
    assert ref_lora.keys() == lora.keys()
    for k in ref_lora:
        assert torch.equal(ref_lora[k], lora[k]), k
    # Every timestep draw after the failure matches the clean run (the failed forward's draws were rewound).
    assert len(rec) == len(ref_rec)
    assert all(torch.equal(a, b) for a, b in zip(rec, ref_rec))


@pytest.mark.allow_network
@pytest.mark.parametrize("oom_at", [2, 8])
def test_the_failed_micro_batch_is_released_before_the_replay(
    oom_at, tmp_path, monkeypatch, tiny_pipe_dir
):
    _lora, _losses, _rec, _alive, warnings, _b = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = True, oom_at = oom_at
    )
    assert any("Ran out of GPU memory" in w for w in warnings)
    # noisy latents, timesteps, sigmas, embeddings and the partial output of the failed forward: all freed.
    assert _run.failed_refs >= 4
    assert _run.failed_alive_at_replay == [0]


@pytest.mark.allow_network
@pytest.mark.parametrize("oom_at", [None, 2])
def test_the_rerun_restarts_the_peak_the_cache_plan_reads(
    oom_at, tmp_path, monkeypatch, tiny_pipe_dir
):
    calls = []
    real = dit.TrainingPeakMemory

    class Recorder(real):
        def pause(self):
            calls.append("pause")
            super().pause()

        def resume(self):
            calls.append("resume")
            super().resume()

    monkeypatch.setattr(dit, "TrainingPeakMemory", Recorder)
    _lora, _losses, _rec, _alive, warnings, _b = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = True, oom_at = oom_at
    )
    if oom_at is None:
        assert calls == []
    else:
        assert any("Ran out of GPU memory" in w for w in warnings)
        assert calls == ["pause", "resume"]


@pytest.mark.allow_network
def test_a_preview_drops_the_fp8_backward_copies_and_keeps_their_budget(
    tmp_path, monkeypatch, tiny_pipe_dir
):
    import core.training.diffusion_frozen_fp8 as f8

    # An fp8 run whose base stays fp32 on CPU: only the backward weight cache's bookkeeping is under test.
    monkeypatch.setattr(dit, "_load_fp8_prequant_transformer", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_apply_fp8_training", lambda *a, **k: True)
    orig_plan = f8.Fp8BackwardWeights.plan

    def plan(self, nbytes):
        orig_plan(self, nbytes)
        # Stand in for the copies step 2's backward would make.
        q = torch.zeros(8, 8)
        self.entries[q.data_ptr()] = (q, q.t())
        self.used = q.numel()

    monkeypatch.setattr(f8.Fp8BackwardWeights, "plan", plan)
    at_preview = []

    def sample_round(plan_, step, *a, **k):
        w = f8.BACKWARD_WEIGHTS
        at_preview.append((step, len(w.entries), w.used, w.budget))

    monkeypatch.setattr(dit, "run_sample_round", sample_round)
    _run(
        tmp_path,
        monkeypatch,
        tiny_pipe_dir,
        budget = True,
        accum = 1,
        steps = 3,
        precision = "fp8",
        sample_every = 1,
        sample_prompts = ("a red tile",),
    )
    planned = [r for r in at_preview if r[3] > 0]
    assert planned, at_preview
    # Every preview after the plan starts with no copies held, and the budget survives for the next backward.
    assert all(n == 0 and used == 0 for _s, n, used, _b in planned), at_preview
    assert not f8.BACKWARD_WEIGHTS.active and f8.BACKWARD_WEIGHTS.entries == {}


@pytest.mark.allow_network
def test_an_fp8_conversion_that_cannot_fall_back_reloads_the_base_as_nf4(
    tmp_path, monkeypatch, tiny_pipe_dir
):
    """Auto's fp8 below the bf16 fit: a failed conversion must not move the bf16 rest onto the card; the base reloads
    as nf4 with a fresh adapter and the run trains on it."""
    monkeypatch.setattr(dit, "_load_fp8_prequant_transformer", lambda *a, **k: None)

    import weakref

    old = []

    def apply(
        transformer,
        on_event,
        family = None,
        device = None,
    ):
        # This frame holds the half-converted base; the raised exception's traceback keeps it alive.
        old.append(weakref.ref(transformer))
        raise dit.Fp8FallbackDoesNotFit("no fp8 here")

    monkeypatch.setattr(dit, "_apply_fp8_training", apply)
    spec = dit._SPECS[FAMILY]
    orig, asked, built, old_alive_at_reload = spec.load_transformer, [], [], []

    def load(cfg, device, dtype, base_precision):
        asked.append(base_precision)
        if base_precision == "nf4":
            import gc
            gc.collect()
            old_alive_at_reload.append(old[0]() is not None)
        # CPU has no bnb: the reload's nf4 stands in as the plain base. Only the reload is kept, so the test itself
        # holds no reference to the failed fp8 base.
        model = orig(cfg, device, dtype, "bf16")
        if base_precision == "nf4":
            built.append(model)
        return model

    monkeypatch.setattr(spec, "load_transformer", load)
    lora, losses, _rec, _alive, warnings, _b = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = False, accum = 1, steps = 2, precision = "fp8"
    )
    assert asked == ["fp8", "nf4"]
    # The failed base is released before the nf4 replacement loads, not held alongside it on the card.
    assert old_alive_at_reload == [False]
    assert any("training nf4 instead" in w for w in warnings), warnings
    assert len(losses) == 2 and lora
    # The adapter trained is the reloaded model's, not the dropped one's.
    assert any("lora" in n for n, _ in built[0].named_parameters())
