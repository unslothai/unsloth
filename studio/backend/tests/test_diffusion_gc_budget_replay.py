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
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: "bf16")
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
        if calls["n"] == oom_at:
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
def test_after_fall_back_nothing_is_kept_for_a_rerun(tmp_path, monkeypatch, tiny_pipe_dir):
    _lora, _losses, _rec, alive, warnings, budgets = _run(
        tmp_path, monkeypatch, tiny_pipe_dir, budget = True, oom_at = 6
    )
    assert any("Ran out of GPU memory" in w for w in warnings)
    assert budgets[0].disabled and not budgets[0].can_retry
    # The replayed forwards run from host copies and are not counted; the originals are never held past their forward.
    assert len(alive) == 12 and max(alive) == 0


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
    # Step 1's dequant cache plan reads the allocator peak: after a failed attempt it must be the rerun's own peak.
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
