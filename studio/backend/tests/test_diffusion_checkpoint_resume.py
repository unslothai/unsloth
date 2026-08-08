# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for diffusion training resume (NVIDIA QA P1-3).

A stop-and-save used to leave only the adapter, so restarting the same configuration began
again at step 1. These cover the pieces that fix it: the atomically written checkpoint
bundle, the optimizer / scheduler / RNG round-trip, the loop resuming at N+1 up to the same
TARGET total, the identity gate that rejects a mismatched checkpoint before the resident GPU
model is evicted, the ``can_resume`` reported for stopped / completed / errored runs, and the
LoRA sidecar recording the step actually reached.

No GPU and no diffusers/peft: the trainers' family-specific halves are exercised elsewhere,
while everything here runs against a two-parameter ``torch.nn.Linear`` through the same
shared helpers both trainers call. The one bitsandbytes case skips without CUDA.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training import diffusion_checkpoint as dc
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    PermutationBatchSampler,
    restore_resume_state,
    write_resume_checkpoint,
)
from routes.training import router as training_router


# ── fixtures / doubles ────────────────────────────────────────────────────────
def _identity(**overrides) -> dc.CheckpointIdentity:
    fields = {
        "family": "sdxl",
        "base_model": "stabilityai/sdxl-turbo",
        "lora_target_modules": ("to_k", "to_q", "to_v", "to_out.0"),
        "lora_rank": 16,
        "lora_alpha": 16,
        "precision": "bf16",
        "base_precision": "nf4",
        "resolution": 1024,
        "base_revision": "rev-deadbeef",
        "dataset_fingerprint": "ds-3-cafe",
    }
    fields.update(overrides)
    return dc.CheckpointIdentity(**fields)


class _Run:
    """A minimal stand-in for a trainer: one trainable tensor, a real optimizer, a real LR
    schedule and the shared permutation sampler, wired exactly as both trainers wire them so
    the resume helpers are exercised on the real objects rather than mocks."""

    def __init__(
        self,
        output_dir: Path,
        *,
        seed: int = 0,
        **cfg_overrides,
    ):
        torch.manual_seed(seed)
        self.model = torch.nn.Linear(4, 4, bias = False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        # A decaying lambda so a wrong schedule position shows up as a different LR.
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda = lambda step: 1.0 / (1.0 + 0.01 * step)
        )
        self.loop_rng = __import__("random").Random(seed)
        self.variant_rng = __import__("random").Random(seed + 1)
        self.sampler = PermutationBatchSampler(7, self.loop_rng)
        self.cfg = DiffusionLoraConfig(
            base_model = "stabilityai/sdxl-turbo",
            data_dir = "unused",
            output_dir = str(output_dir),
            train_steps = 500,
            **cfg_overrides,
        )
        self.identity = _identity()
        self.streams = {"loop": self.loop_rng, "variant": self.variant_rng}

    def step_once(self, grad_scale: float) -> None:
        self.optimizer.zero_grad(set_to_none = True)
        self.model.weight.grad = torch.full_like(self.model.weight, grad_scale)
        self.optimizer.step()
        self.lr_sched.step()

    def save(self, step: int, **kw):
        return write_resume_checkpoint(
            self.cfg,
            step = step,
            model = self.model,
            optimizer = self.optimizer,
            lr_scheduler = self.lr_sched,
            identity = self.identity,
            sampler = self.sampler,
            rng_streams = self.streams,
            **kw,
        )

    def restore(self):
        return restore_resume_state(
            self.cfg,
            model = self.model,
            optimizer = self.optimizer,
            lr_scheduler = self.lr_sched,
            identity = self.identity,
            sampler = self.sampler,
            rng_streams = self.streams,
        )


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run output directory inside the (per-test) Studio outputs root, since the resume
    path resolver refuses anything outside it."""
    from utils.paths import outputs_root

    d = outputs_root() / "my-lora-run"
    d.mkdir(parents = True, exist_ok = True)
    return d


# ── atomic write ──────────────────────────────────────────────────────────────
def test_kill_mid_write_leaves_no_valid_looking_checkpoint(run_dir, monkeypatch):
    # Simulate a hard kill between "everything written" and the atomic promote: rmtree is
    # neutralised so the staging directory survives untouched, exactly as SIGKILL would leave it.
    run = _Run(run_dir)
    monkeypatch.setattr(dc.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(dc, "_promote", lambda *a, **k: (_ for _ in ()).throw(SystemExit(1)))
    with pytest.raises(SystemExit):
        dc.save_checkpoint(
            output_dir = str(run_dir),
            step = 11,
            adapter_state = {"weight": run.model.weight.detach()},
            identity = run.identity,
            target_steps = 500,
            optimizer = run.optimizer,
            lr_scheduler = run.lr_sched,
        )

    # The staging directory really does hold a COMPLETE bundle, so this is a faithful mid-write kill...
    staged = list(run_dir.glob(".tmp-checkpoint-*"))
    assert staged and (staged[0] / dc.TRAINER_STATE_FILENAME).is_file()
    # ...and yet nothing that any scanner would offer as resumable exists.
    assert dc.list_checkpoints(run_dir) == []
    assert dc.latest_valid_checkpoint(run_dir) is None
    assert dc.describe_resume_state(str(run_dir))["can_resume"] is False
    assert not list(run_dir.glob("checkpoint-*"))


def test_failed_write_cleans_up_and_next_save_clears_stale_staging(run_dir, monkeypatch):
    run = _Run(run_dir)
    # A failure BEFORE the manifest (a full disk mid-optimizer-write) unwinds its own staging dir,
    # and the shared wrapper turns it into a reported reason instead of killing the training run.
    monkeypatch.setattr(
        dc, "_torch_save", lambda *a, **k: (_ for _ in ()).throw(OSError("no space left"))
    )
    path, error = run.save(5)
    assert path is None and "no space left" in error
    assert list(run_dir.glob(".tmp-checkpoint-*")) == []
    assert dc.list_checkpoints(run_dir) == []
    monkeypatch.undo()

    # A stale staging directory left by an earlier killed process is swept by the next good save.
    stale = run_dir / f"{dc._STAGING_PREFIX}9-abcd1234"
    stale.mkdir()
    (stale / "junk").write_text("x", encoding = "utf-8")
    path, error = run.save(6)
    assert error is None and path is not None
    assert not stale.exists()
    assert dc.latest_valid_checkpoint(run_dir)[1]["global_step"] == 6


def test_a_real_fsync_failure_fails_the_save_but_an_unsupported_one_does_not(run_dir, monkeypatch):
    # fsync is where delayed-allocation ENOSPC and writeback EIO surface, and the validator only
    # parses a safetensors HEADER, so a bundle whose bytes never reached the device would be
    # promoted and later read as valid. But Windows' _commit and several network filesystems
    # refuse the flush outright, which says nothing about the write and must not fail the run.
    run = _Run(run_dir)
    real_fsync = dc.os.fsync

    monkeypatch.setattr(dc.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(28, "ENOSPC")))
    path, error = run.save(1)
    assert path is None and "ENOSPC" in error
    assert dc.list_checkpoints(run_dir) == []

    monkeypatch.setattr(dc.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(1, "EPERM")))
    path, error = run.save(2)
    assert error is None and path is not None
    monkeypatch.setattr(dc.os, "fsync", real_fsync)
    assert dc.read_checkpoint(run_dir / "checkpoint-2") is not None


def test_re_saving_the_bundle_this_run_resumed_from_keeps_it(run_dir):
    # Resuming at N and stopping before N+1 re-saves byte-identical state, so keeping what is
    # there avoids _promote's one destructive branch (swapping a good bundle out to make room),
    # where a kill would leave the slot empty.
    run = _Run(run_dir)
    first, _ = run.save(9)
    before = (run_dir / "checkpoint-9" / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8")
    resumed = _Run(run_dir, resume_from_checkpoint = str(run_dir / "checkpoint-9"))
    second, error = resumed.save(9)
    assert error is None and second == first
    assert (run_dir / "checkpoint-9" / dc.TRAINER_STATE_FILENAME).read_text(
        encoding = "utf-8"
    ) == before


def test_a_same_step_bundle_from_another_run_is_overwritten(run_dir):
    # The dangerous half of the same arithmetic: resume checkpoint-10 in a folder that also
    # holds checkpoint-15 and stop at 15. The old shortcut saw "step 15 already exists" and
    # returned it, so checkpoint_saved named a bundle whose optimizer, scheduler, sampler and
    # RNG were from the EARLIER run and this run's state was dropped on the floor.
    seeded = _Run(run_dir)
    seeded.save(10)
    seeded.step_once(0.5)
    seeded.save(15)
    stale = (run_dir / "checkpoint-15" / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8")

    resumed = _Run(run_dir, seed = 7, resume_from_checkpoint = str(run_dir / "checkpoint-10"))
    resumed.step_once(1.5)
    path, error = resumed.save(15)

    assert error is None and path == str(run_dir / "checkpoint-15")
    written = dc.read_checkpoint(run_dir / "checkpoint-15")
    assert written is not None
    # It is this run's state now, not the one that was sitting there.
    assert (run_dir / "checkpoint-15" / dc.TRAINER_STATE_FILENAME).read_text(
        encoding = "utf-8"
    ) != stale
    from safetensors.torch import load_file

    restored = load_file(run_dir / "checkpoint-15" / dc.ADAPTER_FILENAME)
    assert torch.allclose(restored["weight"], resumed.model.weight)


def test_incomplete_or_inconsistent_bundles_are_rejected(run_dir):
    run = _Run(run_dir)
    run.save(4)
    good = run_dir / "checkpoint-4"
    assert dc.read_checkpoint(good) is not None

    # No manifest at all (the completion marker) -> not a checkpoint.
    (good / dc.TRAINER_STATE_FILENAME).unlink()
    assert dc.read_checkpoint(good) is None
    assert dc.latest_valid_checkpoint(run_dir) is None

    # A manifest whose step disagrees with its directory name is a rename, not a checkpoint.
    run.save(5)
    manifest_path = run_dir / "checkpoint-5" / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["global_step"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")
    assert dc.read_checkpoint(run_dir / "checkpoint-5") is None

    # A truncated tensor file fails the state-file probe even with a perfect manifest.
    run.save(6)
    (run_dir / "checkpoint-6" / dc.ADAPTER_FILENAME).write_bytes(b"")
    assert dc.read_checkpoint(run_dir / "checkpoint-6") is None
    state = dc.describe_resume_state(str(run_dir))
    assert state["can_resume"] is False
    assert "incomplete or corrupt" in state["resume_blocked_reason"]


def test_a_fresh_run_discards_an_earlier_runs_checkpoints(run_dir):
    # Training a NEW run into an output dir a previous run already used (same adapter name) must
    # not leave the old, higher-numbered bundles behind: latest_valid_checkpoint would pick one of
    # them and a later Resume would silently continue the wrong training.
    old = _Run(run_dir)
    old.save(40)
    assert dc.latest_valid_checkpoint(run_dir)[1]["global_step"] == 40

    fresh = _Run(run_dir, seed = 7)
    fresh.save(5, discard_existing = True)
    assert [p.name for p in dc.list_checkpoints(run_dir)] == ["checkpoint-5"]
    assert dc.describe_resume_state(str(run_dir))["checkpoint_step"] == 5


def test_a_checkpoint_with_no_adapter_tensors_is_refused(run_dir):
    # An empty safetensors file has no keys, so the bundle would fail its own validation and read
    # as "no checkpoint": a run that thinks it saved but cannot be resumed.
    with pytest.raises(ValueError, match = "no adapter tensors"):
        dc.save_checkpoint(
            output_dir = str(run_dir),
            step = 1,
            adapter_state = {},
            identity = _identity(),
            target_steps = 10,
        )
    assert dc.list_checkpoints(run_dir) == []


def test_non_finite_progress_is_dropped_from_the_manifest(run_dir):
    run = _Run(run_dir)
    run.save(2, progress = {"running_loss": float("nan"), "kept": 1.5})
    manifest = dc.read_checkpoint(run_dir / "checkpoint-2")
    assert manifest["progress"] == {"kept": 1.5}
    # And the file is strict JSON (no bare NaN token that a stricter parser would reject).
    raw = (run_dir / "checkpoint-2" / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert dc.load_checkpoint(run_dir / "checkpoint-2").running_loss == 0.0


def test_save_total_limit_keeps_only_the_newest_bundles(run_dir):
    run = _Run(run_dir, save_total_limit = 2)
    for step in (1, 2, 3, 4):
        run.save(step)
    kept = sorted(p.name for p in dc.list_checkpoints(run_dir))
    assert kept == ["checkpoint-3", "checkpoint-4"]


# ── state round-trip ──────────────────────────────────────────────────────────
def test_bundle_round_trips_optimizer_scheduler_sampler_and_rng(run_dir):
    reference = _Run(run_dir, seed = 3)
    for i in range(11):
        reference.step_once(0.01 * (i + 1))
        reference.sampler.next_batch(2)
    path, error = reference.save(11)
    assert error is None and Path(path).name == "checkpoint-11"

    # Keep training the reference, and separately resume a fresh run from the bundle.
    resumed = _Run(run_dir, seed = 999)  # a different seed: everything must come from the file
    resumed.cfg = __import__("dataclasses").replace(
        resumed.cfg, resume_from_checkpoint = str(run_dir)
    )
    loaded = resumed.restore()
    assert loaded is not None and loaded.step == 11

    # Adapter weights, LR-schedule position and sampler cycle all match immediately.
    assert torch.equal(resumed.model.weight, reference.model.weight)
    assert resumed.lr_sched.get_last_lr() == reference.lr_sched.get_last_lr()
    assert resumed.sampler.state_dict() == reference.sampler.state_dict()
    # ... and so do the RNG streams, which decide the next batch and the next noise draw.
    assert resumed.loop_rng.random() == reference.loop_rng.random()
    assert resumed.variant_rng.random() == reference.variant_rng.random()

    # The optimizer MOMENTS are what a naive "reload the adapter" resume loses: the same
    # gradient must therefore move both models to the same place, not just start from it.
    for i in range(11, 16):
        grad = 0.01 * (i + 1)
        reference.step_once(grad)
        resumed.step_once(grad)
        assert torch.allclose(resumed.model.weight, reference.model.weight, atol = 0, rtol = 0)
    assert resumed.lr_sched.get_last_lr() == reference.lr_sched.get_last_lr()


def test_a_foreign_optimizers_state_is_refused_instead_of_key_erroring(run_dir):
    # The trainers choose their optimizer from the HOST (bitsandbytes present, a fused kernel
    # available, UNSLOTH_DIFFUSION_FP32_OPTIM), not the config, so a checkpoint can arrive with
    # moments from a different implementation. Shapes and counts match, so load_state_dict
    # accepts them and the first step dies on a bare KeyError deep in the optimizer.
    run = _Run(run_dir)
    run.save(3)
    manifest_path = run_dir / "checkpoint-3" / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    assert manifest["optimizer_class"] == "torch.optim.adamw.AdamW"
    manifest["optimizer_class"] = "bitsandbytes.optim.adamw.AdamW8bit"
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    fresh = _Run(run_dir)
    fresh.cfg = __import__("dataclasses").replace(fresh.cfg, resume_from_checkpoint = str(run_dir))
    with pytest.raises(dc.ResumeError, match = "AdamW8bit"):
        fresh.restore()


def test_a_partial_adapter_restore_is_refused(run_dir):
    # A name mismatch would otherwise leave restored optimizer moments and a restored LR position
    # driving freshly initialised LoRA weights, while the run reports a normal resume.
    run = _Run(run_dir)
    run.save(3)
    from safetensors.torch import save_file

    adapter = run_dir / "checkpoint-3" / dc.ADAPTER_FILENAME
    save_file(
        {"weight": torch.zeros(4, 4), "renamed.weight": torch.zeros(4, 4)},
        str(adapter),
    )
    # Rewriting a bundle by hand changes the adapter's size, which read_checkpoint now checks
    # against the manifest. Keep it honest, or this exercises the truncation guard instead of
    # the name-mismatch guard it is here for.
    manifest_path = run_dir / "checkpoint-3" / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["file_sizes"]["adapter"] = adapter.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    fresh = _Run(run_dir)
    fresh.cfg = __import__("dataclasses").replace(fresh.cfg, resume_from_checkpoint = str(run_dir))
    with pytest.raises(ValueError, match = "not in this run"):
        fresh.restore()


def test_resuming_reapplies_the_live_lr_schedule(run_dir):
    # load_state_dict restores the rate the checkpoint was written with and never re-evaluates
    # the lambda, so without the re-apply the first resumed step runs at the OLD schedule's
    # value -- lr 0.0 when continuing a finished cosine run by raising the step count.
    from diffusers.optimization import get_scheduler

    def build(total: int):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4, bias = False)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
        return (
            model,
            optimizer,
            get_scheduler(
                "cosine", optimizer = optimizer, num_warmup_steps = 0, num_training_steps = total
            ),
        )

    model, optimizer, lr_sched = build(6)
    for _ in range(6):
        model.weight.grad = torch.full_like(model.weight, 0.01)
        optimizer.step()
        lr_sched.step()
    assert lr_sched.get_last_lr()[0] == pytest.approx(0.0, abs = 1e-12)
    dc.save_checkpoint(
        output_dir = str(run_dir),
        step = 6,
        adapter_state = {"weight": model.weight.detach()},
        identity = _identity(),
        target_steps = 6,
        optimizer = optimizer,
        lr_scheduler = lr_sched,
    )

    # Continue the same run with a raised target: the rate must come from the NEW 12-step curve.
    model2, optimizer2, lr_sched2 = build(12)
    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/sdxl-turbo",
        data_dir = "unused",
        output_dir = str(run_dir),
        train_steps = 12,
        resume_from_checkpoint = str(run_dir),
    )
    restored = restore_resume_state(
        cfg,
        model = model2,
        optimizer = optimizer2,
        lr_scheduler = lr_sched2,
        identity = _identity(),
    )
    assert restored.step == 6
    reference = build(12)[2]
    for _ in range(6):
        reference.step()
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(reference.get_last_lr()[0])
    assert optimizer2.param_groups[0]["lr"] > 0.0


def test_torch_rng_state_is_restored(run_dir):
    run = _Run(run_dir, seed = 5)
    run.save(1)
    expected = torch.randn(8)

    torch.manual_seed(12345)  # move the global RNG somewhere else entirely
    fresh = _Run(run_dir, seed = 5)
    fresh.cfg = __import__("dataclasses").replace(fresh.cfg, resume_from_checkpoint = str(run_dir))
    fresh.restore()
    assert torch.equal(torch.randn(8), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bitsandbytes AdamW8bit needs CUDA")
def test_bitsandbytes_adamw8bit_state_round_trips(run_dir):
    # The default diffusion optimizer is bnb AdamW8bit, whose moments are quantized uint8 plus
    # their maps. torch.save/load must carry them exactly, or a "resumed" run silently restarts
    # Adam from zero moments and spikes the loss.
    bnb = pytest.importorskip("bitsandbytes")
    if not isinstance(getattr(bnb.optim, "AdamW8bit", None), type):
        # unsloth_zoo replaces bitsandbytes with a raising stub on hosts without it; the suite
        # imports unsloth first, so that stub can be live even where real bnb is installed.
        pytest.skip("bitsandbytes is stubbed out in this environment")

    torch.manual_seed(0)
    # Above bitsandbytes' min_8bit_size (4096): a smaller parameter gets plain fp32 moments, so
    # the quantized state -- the part that actually needs a round-trip -- would never be written.
    param = torch.nn.Parameter(torch.randn(128, 64, device = "cuda"))
    optimizer = bnb.optim.AdamW8bit([param], lr = 1e-3)
    for i in range(4):
        param.grad = torch.full_like(param, 0.01 * (i + 1))
        optimizer.step()
    quant = next(iter(optimizer.state.values()))
    quant = quant.get("__bnb_optimizer_quant_state__", quant)
    assert any(getattr(v, "dtype", None) is torch.uint8 for v in quant.values())

    dc.save_checkpoint(
        output_dir = str(run_dir),
        step = 4,
        adapter_state = {"w": param.detach()},
        identity = _identity(),
        target_steps = 10,
        optimizer = optimizer,
    )
    loaded = dc.load_checkpoint(run_dir / "checkpoint-4")

    clone = torch.nn.Parameter(param.detach().clone())
    restored = bnb.optim.AdamW8bit([clone], lr = 1e-3)
    restored.load_state_dict(loaded.torch_state("optimizer"))
    for i in range(4, 8):
        grad = torch.full_like(param, 0.01 * (i + 1))
        param.grad = grad.clone()
        optimizer.step()
        clone.grad = grad.clone()
        restored.step()
    assert torch.equal(clone, param)


# ── the loop resumes at N+1 up to the same TARGET ─────────────────────────────
def _loop(run: _Run, *, stop_at: int | None = None) -> tuple[list[int], int]:
    """The exact shape of both trainers' loops: restore, then ``range(resumed, train_steps)``
    with ``done = opt_step + 1``."""
    restored = run.restore()
    resumed = restored.step if restored is not None else 0
    done = resumed
    seen: list[int] = []
    for opt_step in range(resumed, run.cfg.train_steps):
        run.step_once(0.001)
        run.sampler.next_batch(2)
        done = opt_step + 1
        seen.append(done)
        if stop_at is not None and done == stop_at:
            break
    return seen, done


def test_resume_at_step_11_with_target_500_runs_steps_12_to_500(run_dir):
    first = _Run(run_dir, seed = 1)
    seen, done = _loop(first, stop_at = 11)
    assert seen == list(range(1, 12)) and done == 11
    # Stop-and-save writes the bundle at the step actually reached.
    path, error = first.save(done)
    assert error is None and Path(path).name == "checkpoint-11"

    second = _Run(run_dir, seed = 1)
    second.cfg = __import__("dataclasses").replace(second.cfg, resume_from_checkpoint = str(run_dir))
    seen, done = _loop(second)
    # train_steps is the TARGET TOTAL, not an extra budget: 12..500, then the run is finished.
    assert seen[0] == 12
    assert seen[-1] == 500
    assert seen == list(range(12, 501))
    assert done == 500
    assert len(seen) == 489


def test_resume_of_a_finished_checkpoint_is_refused(run_dir):
    run = _Run(run_dir)
    run.save(500)
    with pytest.raises(dc.ResumeError, match = "already at step 500 of 500"):
        dc.preflight_resume(str(run_dir), identity = run.identity, target_steps = 500)


def test_both_trainers_loop_from_the_resumed_step():
    # The two real loops are GPU-only, so guard their bound structurally: a regression back to
    # ``range(cfg.train_steps)`` would silently retrain steps the checkpoint already covered.
    backend = Path(__file__).resolve().parent.parent
    for name in ("diffusion_lora_trainer.py", "diffusion_dit_trainer.py"):
        source = (backend / "core" / "training" / name).read_text(encoding = "utf-8")
        loops = [
            ast.unparse(node.iter)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "opt_step"
        ]
        assert loops == ["range(resumed, cfg.train_steps)"], f"{name}: {loops}"


# ── the identity gate ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"family": "flux.1"}, "different model family"),
        ({"base_model": "black-forest-labs/FLUX.1-dev"}, "different base model"),
        ({"lora_rank": 32}, "different LoRA rank"),
        ({"lora_alpha": 8}, "different LoRA alpha"),
        ({"lora_target_modules": ("to_q",)}, "different LoRA target modules"),
        ({"precision": "fp16"}, "different mixed precision"),
        ({"base_precision": "bf16"}, "different base precision"),
        ({"kind": "video"}, "different training type"),
        ({"dataset_fingerprint": "ds-4-beef"}, "training images have changed"),
    ],
)
def test_identity_mismatches_are_rejected_with_a_clear_reason(run_dir, overrides, expected):
    run = _Run(run_dir)
    run.save(3)
    with pytest.raises(dc.ResumeError, match = expected):
        dc.preflight_resume(str(run_dir), identity = _identity(**overrides), target_steps = 500)
    # The matching identity still passes, so the rejection is the mismatch and nothing else.
    path, step = dc.preflight_resume(str(run_dir), identity = _identity(), target_steps = 500)
    assert step == 3 and Path(path).name == "checkpoint-3"


def test_unknown_revision_or_dataset_on_either_side_is_not_a_mismatch(run_dir):
    # source_revision() reads "unresolved" for a repo that is not in the local Hub cache, and the
    # start route computes the identity before it has walked the dataset. Neither may look like a
    # changed base model / changed images, or every first resume would be refused.
    run = _Run(run_dir)
    run.identity = _identity(base_revision = "unresolved")
    run.save(3)
    dc.preflight_resume(
        str(run_dir), identity = _identity(base_revision = "rev-something"), target_steps = 500
    )
    dc.preflight_resume(
        str(run_dir), identity = _identity(dataset_fingerprint = None), target_steps = 500
    )


def test_resume_path_must_stay_inside_the_outputs_root(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(dc.ResumeError, match = "inside Unsloth outputs"):
        dc.resolve_resume_dir(str(outside))


def test_resume_of_a_run_without_checkpoints_is_refused(run_dir):
    with pytest.raises(dc.ResumeError, match = "No complete training checkpoint"):
        dc.preflight_resume(str(run_dir), identity = _identity(), target_steps = 500)


# ── route preflight: reject BEFORE evicting the GPU ───────────────────────────
class _FakeService:
    def __init__(self):
        self.calls: list[str] = []
        self.started_with: dict | None = None

    def reserve(self):
        self.calls.append("reserve")

    def unreserve(self):
        self.calls.append("unreserve")

    def is_active(self):
        return False

    def start(self, config):
        self.calls.append("start")
        self.started_with = config
        return "job-123"


class _FakeLLMBackend:
    def is_training_active(self):
        return False


_PAIRS = [("img.png", "a caption")]


@pytest.fixture
def client(monkeypatch):
    import routes.training as tr

    fake = _FakeService()
    freed: list[str] = []
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: fake
    )
    monkeypatch.setattr(tr, "get_training_backend", lambda: _FakeLLMBackend())
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: freed.append("freed"))
    monkeypatch.setattr(
        "core.training.diffusion_train_common.discover_image_caption_pairs",
        lambda data_dir, **kw: list(_PAIRS),
    )
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    c = TestClient(app)
    c._fake = fake  # type: ignore[attr-defined]
    c._freed = freed  # type: ignore[attr-defined]
    return c


def _request_identity(**overrides) -> dc.CheckpointIdentity:
    """The identity the start route computes for ``_RESUME_BODY`` + the stubbed dataset."""
    return _identity(
        **{
            "base_revision": "unresolved",
            "dataset_fingerprint": dc.dataset_fingerprint(_PAIRS),
            **overrides,
        }
    )


_RESUME_BODY = {
    "base_model": "stabilityai/sdxl-turbo",
    "data_dir": "uploads/my-images",
    "output_dir": "my-lora-run",
    "train_steps": 500,
    "resume_from_checkpoint": "my-lora-run",
}


def _write_bundle(run_dir: Path, step: int, identity: dc.CheckpointIdentity) -> None:
    dc.save_checkpoint(
        output_dir = str(run_dir),
        step = step,
        adapter_state = {"w": torch.zeros(2, 2)},
        identity = identity,
        target_steps = 500,
    )


def test_route_resume_accepts_a_matching_checkpoint_and_pins_it(client, run_dir):
    _write_bundle(run_dir, 11, _request_identity())
    r = client.post("/api/train/diffusion/start", json = _RESUME_BODY)
    assert r.status_code == 200, r.text
    # The exact bundle the preflight accepted is what the trainer gets, not "whatever is newest
    # in that folder by the time the child runs".
    pinned = Path(client._fake.started_with["resume_from_checkpoint"])
    assert pinned == run_dir / "checkpoint-11"
    assert client._freed == ["freed"]


def test_route_resume_mismatch_400s_without_evicting_the_gpu(client, run_dir):
    _write_bundle(run_dir, 11, _request_identity(family = "flux.1"))
    r = client.post("/api/train/diffusion/start", json = _RESUME_BODY)
    assert r.status_code == 400, r.text
    assert "different model family" in r.json()["detail"]
    # The whole point of preflighting here: the user's loaded Images pipeline survives a refusal.
    assert client._freed == []
    assert "start" not in client._fake.calls


def test_route_resume_dataset_change_400s_without_evicting_the_gpu(client, run_dir):
    # The dataset half of the identity is only knowable after discovery, which the route runs
    # inside the reservation -- still before the GPU teardown.
    _write_bundle(run_dir, 11, _request_identity(dataset_fingerprint = "ds-9-other"))
    r = client.post("/api/train/diffusion/start", json = _RESUME_BODY)
    assert r.status_code == 400, r.text
    assert "training images have changed" in r.json()["detail"]
    assert client._freed == []
    assert "start" not in client._fake.calls
    # The reservation is always released, so the next start is not locked out by the refusal.
    assert client._fake.calls.count("unreserve") == 1


def test_route_resume_missing_checkpoint_400s(client, run_dir):
    r = client.post("/api/train/diffusion/start", json = _RESUME_BODY)
    assert r.status_code == 400, r.text
    assert "No complete training checkpoint" in r.json()["detail"]
    assert client._freed == []


def test_route_resume_outside_outputs_400s(client, tmp_path):
    body = {**_RESUME_BODY, "resume_from_checkpoint": str(tmp_path / "escape")}
    r = client.post("/api/train/diffusion/start", json = body)
    assert r.status_code == 400, r.text
    assert client._freed == []


def test_route_start_without_resume_is_unchanged(client, run_dir):
    body = {k: v for k, v in _RESUME_BODY.items() if k != "resume_from_checkpoint"}
    r = client.post("/api/train/diffusion/start", json = body)
    assert r.status_code == 200, r.text
    assert client._fake.started_with.get("resume_from_checkpoint") is None


# ── can_resume in the run history ─────────────────────────────────────────────
@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    import core.training.diffusion_training_service as dts

    d = tmp_path / "runs" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(dts, "_runs_dir", lambda: d)
    return d


def _persist(job_id: str, status: str, output_dir: Path, **state):
    from core.training.diffusion_training_service import DiffusionTrainingService, _idle_state

    service = DiffusionTrainingService()
    snapshot = _idle_state()
    snapshot.update(
        job_id = job_id,
        status = status,
        output_dir = str(output_dir),
        lora_path = str(output_dir / "pytorch_lora_weights.safetensors"),
        step = 11,
        total_steps = 500,
        **state,
    )
    service._state = snapshot
    service._config = {"output_dir": str(output_dir), "base_model": "stabilityai/sdxl-turbo"}
    service._persist_run_record()


def test_can_resume_is_reported_per_run_state(runs_dir, run_dir):
    from core.training.diffusion_training_service import get_diffusion_run, list_diffusion_runs

    stopped_dir = run_dir
    _write_bundle(stopped_dir, 11, _identity())
    _persist("a" * 32, "stopped", stopped_dir)

    # A completed run reached its target, so it is never resumable -- even though a periodic
    # bundle from partway through is still sitting in its folder.
    from utils.paths import outputs_root

    completed_dir = outputs_root() / "done-run"
    completed_dir.mkdir(parents = True, exist_ok = True)
    _write_bundle(completed_dir, 450, _identity())
    _persist("b" * 32, "completed", completed_dir)

    # An errored run that DID reach a periodic checkpoint is exactly the case worth resuming.
    errored_dir = outputs_root() / "crashed-run"
    errored_dir.mkdir(parents = True, exist_ok = True)
    _write_bundle(errored_dir, 40, _identity())
    _persist("c" * 32, "error", errored_dir)

    # A stop whose checkpoint write FAILED must stay blocked, with the writer's reason shown,
    # even though an older bundle is still on disk (resuming it would silently lose steps).
    blocked_dir = outputs_root() / "blocked-run"
    blocked_dir.mkdir(parents = True, exist_ok = True)
    _write_bundle(blocked_dir, 7, _identity())
    _persist(
        "d" * 32,
        "stopped",
        blocked_dir,
        resume_blocked_reason = "Could not write a resume checkpoint at step 11: disk full",
    )

    by_id = {r["job_id"]: r for r in list_diffusion_runs(limit = 20)}
    assert by_id["a" * 32]["can_resume"] is True
    assert by_id["a" * 32]["checkpoint_step"] == 11
    assert by_id["a" * 32]["output_dir"] == str(stopped_dir)

    assert by_id["b" * 32]["can_resume"] is False
    assert "nothing left to train" in by_id["b" * 32]["resume_blocked_reason"]

    assert by_id["c" * 32]["can_resume"] is True
    assert by_id["c" * 32]["checkpoint_step"] == 40

    assert by_id["d" * 32]["can_resume"] is False
    assert "disk full" in by_id["d" * 32]["resume_blocked_reason"]

    # The detail endpoint agrees, and re-derives from disk: deleting the bundle takes the action
    # away rather than leaving a button that would 400.
    assert get_diffusion_run("a" * 32)["can_resume"] is True
    dc.shutil.rmtree(stopped_dir / "checkpoint-11")
    refreshed = get_diffusion_run("a" * 32)
    assert refreshed["can_resume"] is False
    assert "no resume checkpoint" in refreshed["resume_blocked_reason"]


def test_can_resume_is_false_when_the_output_folder_is_gone(runs_dir, run_dir):
    from core.training.diffusion_training_service import list_diffusion_runs

    _write_bundle(run_dir, 11, _identity())
    _persist("e" * 32, "stopped", run_dir)
    dc.shutil.rmtree(run_dir)
    (record,) = list_diffusion_runs(limit = 5)
    assert record["can_resume"] is False
    assert "no longer exists" in record["resume_blocked_reason"]


def test_lineage_is_recorded(runs_dir, run_dir):
    from core.training.diffusion_training_service import get_diffusion_run

    _write_bundle(run_dir, 20, _identity())
    from core.training.diffusion_training_service import DiffusionTrainingService, _idle_state

    service = DiffusionTrainingService()
    snapshot = _idle_state()
    snapshot.update(
        job_id = "f" * 32,
        status = "stopped",
        output_dir = str(run_dir),
        lora_path = str(run_dir / "pytorch_lora_weights.safetensors"),
        resumed_from_step = 11,
    )
    service._state = snapshot
    service._config = {"output_dir": str(run_dir), "resumed_from_job_id": "a" * 32}
    service._persist_run_record()

    record = get_diffusion_run("f" * 32)
    assert record["resumed_from_job_id"] == "a" * 32
    assert record["resumed_from_step"] == 11


def test_service_folds_the_resumed_and_checkpoint_events(runs_dir):
    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService()
    service._apply_event({"type": "resumed", "step": 11, "checkpoint_path": "/x/checkpoint-11"})
    snapshot = service.status()
    # The step resumed FROM is lineage, not a bundle this run wrote.
    assert snapshot["resumed_from_step"] == 11
    assert snapshot["checkpoint_step"] is None

    # Folded as each bundle lands, so a run that CRASHES after one is still known to be resumable.
    service._apply_event(
        {"type": "checkpoint_saved", "step": 40, "checkpoint_path": "/x/checkpoint-40"}
    )
    snapshot = service.status()
    assert snapshot["checkpoint_step"] == 40
    assert snapshot["checkpoint_path"] == "/x/checkpoint-40"
    assert snapshot["resume_blocked_reason"] is None

    # A failed write is sticky: the older bundle predates the work this run did.
    service._apply_event({"type": "checkpoint_failed", "step": 50, "message": "disk full"})
    assert service.status()["resume_blocked_reason"] == "disk full"
    # ...and a later good write clears it again.
    service._apply_event(
        {"type": "checkpoint_saved", "step": 60, "checkpoint_path": "/x/checkpoint-60"}
    )
    assert service.status()["resume_blocked_reason"] is None

    service._apply_event(
        {
            "type": "complete",
            "stopped": True,
            "output_dir": "/x",
            "lora_path": "/x/pytorch_lora_weights.safetensors",
            "resumed_from_step": 11,
        }
    )
    snapshot = service.status()
    assert snapshot["checkpoint_step"] == 60
    assert snapshot["resumed_from_step"] == 11
    assert snapshot["status"] == "stopped"


def test_a_checkpoint_write_failure_blocks_resume_on_a_crashed_run(runs_dir, run_dir):
    # The failure arrives on its own event, so it survives a run that ends in `error` rather than
    # `complete` -- the case where an older bundle is most likely to still be lying around.
    from core.training.diffusion_training_service import DiffusionTrainingService, get_diffusion_run

    _write_bundle(run_dir, 7, _identity())
    service = DiffusionTrainingService()
    service._state.update(job_id = "1" * 32, started_at = 0.0, total_steps = 500)
    service._config = {"output_dir": str(run_dir)}
    service._apply_event({"type": "checkpoint_failed", "step": 20, "message": "disk full"})
    service._apply_event({"type": "error", "message": "CUDA out of memory"})
    service._persist_run_record()

    record = get_diffusion_run("1" * 32)
    assert record["status"] == "error"
    assert record["can_resume"] is False
    assert record["resume_blocked_reason"] == "disk full"


def test_an_earlier_runs_checkpoints_are_not_offered_to_a_later_run(runs_dir, run_dir):
    # Two runs can share an output dir (same adapter name trained twice) and the EARLIER one's
    # bundles can carry higher step numbers. A later run must not advertise, and then resume,
    # another run's training state.
    import time as _time

    from core.training.diffusion_training_service import get_diffusion_run

    _write_bundle(run_dir, 400, _identity())
    later_start = _time.time() + 1.0
    _persist("2" * 32, "stopped", run_dir, started_at = later_start)

    record = get_diffusion_run("2" * 32)
    assert record["can_resume"] is False
    assert "left by an earlier run" in record["resume_blocked_reason"]

    # The run that actually wrote it still sees it.
    _persist("3" * 32, "stopped", run_dir, started_at = later_start - 60.0)
    assert get_diffusion_run("3" * 32)["checkpoint_step"] == 400


# ── the sidecar records the step actually reached ─────────────────────────────
def test_lora_sidecar_records_the_reached_step(tmp_path, monkeypatch):
    from core.inference import diffusion_lora as dl
    from core.training.diffusion_train_common import _publish_to_lora_catalog

    loras = tmp_path / "loras"
    loras.mkdir()
    monkeypatch.setattr(dl, "loras_dir", lambda: loras)

    source = tmp_path / "run" / "pytorch_lora_weights.safetensors"
    source.parent.mkdir(parents = True)
    source.write_bytes(b"weights")
    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/sdxl-turbo",
        data_dir = "unused",
        output_dir = str(tmp_path / "run"),
        train_steps = 500,
    ).normalized()

    # Stopped at step 11 of a 500-step run: the sidecar used to advertise the CONFIGURED 500.
    published = _publish_to_lora_catalog(str(source), cfg, 11)
    sidecar = json.loads(Path(published).with_suffix(".json").read_text(encoding = "utf-8"))
    assert sidecar["steps"] == 11

    # Omitted, it still falls back to train_steps for callers that do not know the reached step.
    fallback = _publish_to_lora_catalog(str(source), cfg)
    meta = json.loads(Path(fallback).with_suffix(".json").read_text(encoding = "utf-8"))
    assert meta["steps"] == 500


# ── config validation ─────────────────────────────────────────────────────────
def test_checkpoint_config_is_validated_and_normalized():
    def build(**kw):
        return DiffusionLoraConfig(
            base_model = "stabilityai/sdxl-turbo",
            data_dir = "unused",
            output_dir = "run",
            **kw,
        )

    # Off by default: no periodic checkpoints unless asked for.
    assert build().normalized().save_steps == 0
    assert build(save_steps = 50).normalized().save_steps == 50
    with pytest.raises(ValueError, match = "save_steps must be >= 0"):
        build(save_steps = -1).normalized()
    with pytest.raises(ValueError, match = "save_total_limit must be >= 0"):
        build(save_total_limit = -2).normalized()
    # A blank resume path is "fresh run", not the outputs root.
    assert build(resume_from_checkpoint = "   ").normalized().resume_from_checkpoint is None
    assert (
        build(resume_from_checkpoint = " outputs/run ").normalized().resume_from_checkpoint
        == "outputs/run"
    )


def test_sampler_state_round_trips_and_rejects_a_foreign_dataset_size():
    import random

    reference = PermutationBatchSampler(7, random.Random(0))
    reference.next_batch(5)
    state = reference.state_dict()
    # The two indices left in the current cycle. Beyond them the sampler reshuffles from its rng,
    # which the RNG snapshot restores separately, so this asserts only what the state owns.
    expected = reference.next_batch(2)

    # A sampler seeded differently still continues the SAME cycle once the state is loaded: the
    # order is stored, not just the position, so a partly consumed permutation is reproduced.
    clone = PermutationBatchSampler(7, random.Random(99))
    clone.load_state_dict(state)
    assert clone.next_batch(2) == expected

    # A state for a different dataset size is ignored rather than indexing out of range.
    other = PermutationBatchSampler(3, random.Random(0))
    other.load_state_dict(state)
    assert all(0 <= i < 3 for i in other.next_batch(10))


# ── the header parse is not a load ────────────────────────────────────────────


def test_a_truncated_tensor_storage_is_refused(run_dir):
    """The state-file validator walks a torch zip's pickle to STOP and wants one non-empty
    ``data/`` member. Truncating the moment STORAGES leaves both intact, and ``torch.load``
    then hands back uninitialized memory -- non-finite, order 1e22 -- which a resume feeds
    into the optimizer as Adam moments. The run diverges on its first step and reports a
    clean resume while doing it. The recorded size is what catches this."""
    import zipfile

    run = _Run(run_dir)
    run.step_once(0.5)
    run.save(2)
    optimizer_pt = run_dir / "checkpoint-2" / dc.OPTIMIZER_FILENAME
    assert dc.read_checkpoint(run_dir / "checkpoint-2") is not None

    with zipfile.ZipFile(optimizer_pt) as src:
        members = [(i, src.read(i.filename)) for i in src.infolist()]
    with zipfile.ZipFile(optimizer_pt, "w") as dst:
        for info, blob in members:
            # Every tensor storage clipped to one byte; the pickle and the header survive.
            if "/data/" in info.filename and len(blob) > 1:
                blob = blob[:1]
            dst.writestr(info.filename, blob)

    assert dc.read_checkpoint(run_dir / "checkpoint-2") is None
    fresh = _Run(run_dir)
    fresh.cfg = __import__("dataclasses").replace(fresh.cfg, resume_from_checkpoint = str(run_dir))
    with pytest.raises(dc.ResumeError):
        fresh.restore()


def test_a_bundle_torch_load_refuses_is_rejected_by_the_preflight(run_dir):
    """A ``.pt`` carrying a global outside the ``weights_only`` allowlist walks to STOP like
    any other, so the header check passes it. Before the preflight actually loaded, the start
    route returned 200, evicted the resident GPU model, and only then did the child die on a
    raw UnpicklingError -- exactly what the preflight exists to prevent."""
    run = _Run(run_dir)
    run.save(2)
    checkpoint = run_dir / "checkpoint-2"
    # eval is not in the weights_only allowlist; the zip still parses.
    torch.save({"boom": eval}, str(checkpoint / dc.SCHEDULER_FILENAME), pickle_protocol = 2)
    manifest_path = checkpoint / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["file_sizes"]["scheduler"] = (checkpoint / dc.SCHEDULER_FILENAME).stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    # The cheap gate still passes it -- that is the point.
    assert dc.read_checkpoint(checkpoint) is not None
    with pytest.raises(dc.ResumeError, match = "scheduler"):
        dc.preflight_resume(str(checkpoint), identity = _identity(), target_steps = 500)


# ── two runs sharing an output directory ──────────────────────────────────────


def test_a_finished_run_does_not_offer_its_successors_checkpoint(run_dir):
    """``not_before`` fences off an EARLIER run's bundles. Without the matching upper fence a
    finished run still sees every bundle a LATER run wrote into the shared folder and offers
    the newest of them as its own. The identity gate cannot catch it -- same family, base,
    dataset and LoRA shape -- so the earlier run would resume its successor's optimizer
    moments, LR position and RNG under its own config."""
    early = _Run(run_dir)
    early.save(10)
    early_manifest = json.loads(
        (run_dir / "checkpoint-10" / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8")
    )
    early_created = early_manifest["created_at"]
    early_started = early_created - 1.0

    later = _Run(run_dir)
    later.save(50)
    later_created = json.loads(
        (run_dir / "checkpoint-50" / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8")
    )["created_at"]
    assert later_created > early_created, "fixture did not order the two runs"
    # The first run ended between the two saves.
    early_ended = (early_created + later_created) / 2.0

    fenced = dc.describe_resume_state(
        str(run_dir), status = "stopped", started_at = early_started, ended_at = early_ended
    )
    assert fenced["checkpoint_step"] == 10
    assert fenced["checkpoint_path"].endswith("checkpoint-10")

    # The later run, with no upper fence of its own yet, still sees its own newest.
    live = dc.describe_resume_state(
        str(run_dir), status = "stopped", started_at = early_ended, ended_at = None
    )
    assert live["checkpoint_step"] == 50


def test_a_bundle_with_no_created_at_survives_the_upper_fence(run_dir):
    """An older bundle predates ``created_at`` and reads 0.0. Fencing it out by an end time
    would make every pre-existing run unresumable, which is the opposite of the point."""
    run = _Run(run_dir)
    run.save(4)
    manifest_path = run_dir / "checkpoint-4" / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    del manifest["created_at"]
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    found = dc.latest_valid_checkpoint(run_dir, not_after = 1.0)
    assert found is not None and dc.checkpoint_step(found[0]) == 4


# ── RNG restore across a change in visible devices ────────────────────────────


def test_cuda_rng_restores_the_devices_the_bundle_covers(monkeypatch):
    """A bundle written with one device visible, restored where two are.

    ``set_rng_state_all`` needs one state per visible device, and guarding that with
    all-or-nothing meant this case restored NOTHING -- including cuda:0, the only device the
    trainer touches. Every other part of the resume looked correct, so the run finished
    healthy and silently wrong: on a real B200 SDXL run all 192 LoRA tensors diverged from
    the uninterrupted control. The trainer is a spawned child inheriting
    CUDA_VISIBLE_DEVICES, so a change to the mask between the run and the Resume click was
    enough to trigger it."""
    restored: list[tuple[int, bytes]] = []

    fake_cuda = type(
        "FakeCuda",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "device_count": staticmethod(lambda: 2),
            "set_rng_state": staticmethod(
                lambda state, index: restored.append((index, bytes(state.tolist())))
            ),
            "set_rng_state_all": staticmethod(
                lambda states: pytest.fail("must not need every device")
            ),
        },
    )()
    monkeypatch.setattr(torch, "cuda", fake_cuda)

    only_device_zero = torch.tensor([7, 8, 9], dtype = torch.uint8)
    dc.restore_rng_state({}, {"torch_cuda_0": only_device_zero})

    assert restored == [(0, bytes([7, 8, 9]))], "cuda:0 was not restored"


def test_a_checkpoint_missing_a_live_tensor_is_refused():
    """The other direction, which the count alone could not see.

    Matching the number restored against the checkpoint's own key count proves every SAVED
    tensor landed somewhere. It says nothing about a live trainable parameter the checkpoint
    never had: a truncated or hand-edited adapter holding a strict SUBSET passed, and the full
    optimizer state was then loaded on top, so restored Adam moments drove freshly initialised
    weights while the run reported a clean resume.
    """
    from core.training.diffusion_train_common import (
        load_trainable_state_dict,
        trainable_state_dict,
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias = False), torch.nn.Linear(4, 4))
    for param in model.parameters():
        param.requires_grad_(True)
    full = trainable_state_dict(model)
    assert len(full) >= 2

    # Everything the checkpoint holds still lands, so the old count check was satisfied.
    partial = {k: v for k, v in full.items() if k != sorted(full)[0]}
    with pytest.raises(ValueError, match = "not in the checkpoint"):
        load_trainable_state_dict(model, partial)

    # ...and the complete set is still accepted, so the guard is not simply refusing everything.
    assert load_trainable_state_dict(model, full) == len(full)


# ── the writer must not destroy what it is replacing ──────────────────────────


def test_the_bundle_just_written_survives_pruning(run_dir):
    """Pruning is by STEP, and a resume legitimately writes a bundle that is not the highest
    numbered one in the folder. Resume 10, stop at 15 with 20 and 30 still present and a limit
    of 2, and the checkpoint just promoted is the one deleted -- while the service reports
    checkpoint_saved for a path that no longer exists and the run's own start fence stops those
    older bundles from making it resumable."""
    seed = _Run(run_dir, save_total_limit = 0)
    for step in (10, 20, 30):
        seed.save(step)

    run = _Run(run_dir, save_total_limit = 2)
    written, error = run.save(15)

    assert error is None, error
    assert Path(written).is_dir(), "the checkpoint this call reported was pruned by the same call"
    assert dc.read_checkpoint(Path(written)) is not None
    kept = {p.name for p in dc.list_checkpoints(run_dir)}
    assert "checkpoint-15" in kept
    assert len(kept) == 2, kept


def test_a_failed_write_leaves_the_previous_checkpoints_alone(run_dir, monkeypatch):
    """discard_existing cleared the directory BEFORE staging anything. A kill or an I/O error
    mid-write then left the new run with no checkpoint and every earlier run sharing the folder
    with none either -- the opposite of what this writer promises."""
    run = _Run(run_dir, save_total_limit = 0)
    run.save(7)

    def _boom(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(dc, "_save_tensors", _boom)
    # write_resume_checkpoint reports rather than raises, so the run survives a bad disk.
    written, error = run.save(1, discard_existing = True)
    assert written is None and error

    survivors = {p.name for p in dc.list_checkpoints(run_dir)}
    assert survivors == {"checkpoint-7"}, (
        "the pre-existing bundle was deleted before the replacement was written, so a write "
        f"failure lost it: {survivors}"
    )
    assert dc.read_checkpoint(run_dir / "checkpoint-7") is not None


def test_a_successful_discarding_write_still_replaces_the_old_bundles(run_dir):
    """The other half: deferring the delete must not turn discard_existing into a no-op."""
    run = _Run(run_dir, save_total_limit = 0)
    run.save(7)
    written, error = run.save(1, discard_existing = True)

    assert error is None, error
    assert {p.name for p in dc.list_checkpoints(run_dir)} == {"checkpoint-1"}
    assert Path(written).name == "checkpoint-1"


# ── a discard must not take the source checkpoint with it ─────────────────────


def test_clearing_this_runs_checkpoints_spares_the_one_it_resumed_from(run_dir):
    """A resumed run writes into the directory its source lives in, so a blanket clear on
    "stop without saving" deletes the bundle the user resumed from: an accidental resume
    followed by a discard leaves the ORIGINAL stopped run unresumable."""
    source = _Run(run_dir, save_total_limit = 0)
    source.save(11)
    preexisting = dc.list_checkpoints(run_dir)

    resumed = _Run(run_dir, save_total_limit = 0)
    resumed.save(14)
    resumed.save(17)

    dc.clear_own_checkpoints(run_dir, preexisting)

    survivors = {p.name for p in dc.list_checkpoints(run_dir)}
    assert survivors == {
        "checkpoint-11"
    }, f"the discard removed the source bundle as well as its own: {survivors}"


# ── EMA turned on by the resume ───────────────────────────────────────────────


def test_enabling_ema_on_a_resume_starts_from_the_restored_weights():
    """EMA is not part of the validated identity, so a resume can turn it on for a run that had
    it off. The trainer builds the EMA before restoring the adapter, so its shadow holds freshly
    initialised LoRA weights and the checkpoint carries no shadow to replace them -- every later
    update, and the exported EMA adapter, would blend the restored weights with that noise."""
    from core.training.diffusion_train_extras import LoRAEMA

    model = torch.nn.Linear(4, 4, bias = False)
    ema = LoRAEMA(model, decay = 0.99)
    initial = ema.state_dict()["weight"].clone()

    # ...the adapter is restored afterwards, exactly as the trainer orders it.
    with torch.no_grad():
        model.weight.copy_(torch.full_like(model.weight, 3.0))
    assert not torch.allclose(ema.state_dict()["weight"], model.weight)

    ema.reseed_from(model)

    assert torch.allclose(ema.state_dict()["weight"], model.weight)
    assert not torch.allclose(ema.state_dict()["weight"], initial)
    assert ema.updates == 0, "the warmup ramp restarts with the shadow"


# ── a run directory that happens to look like a bundle ────────────────────────


def test_an_output_directory_named_like_a_checkpoint_is_still_scanned(tmp_path, monkeypatch):
    """An adapter can legitimately be called "checkpoint-2026". Its output directory then
    matches the bundle pattern while holding checkpoint-11 rather than a trainer_state.json of
    its own, and the documented resume API rejected it with the real checkpoint sitting inside."""
    from utils.paths import outputs_root

    weird = outputs_root() / "checkpoint-2026"
    weird.mkdir(parents = True, exist_ok = True)
    run = _Run(weird, save_total_limit = 0)
    run.save(11)

    path, step = dc.preflight_resume(str(weird), identity = _identity(), target_steps = 500)

    assert Path(path).name == "checkpoint-11"
    assert step == 11


def test_an_image_replaced_in_place_changes_the_dataset_fingerprint(tmp_path):
    """Same filename, same caption, same byte length, different picture. On size alone the
    preflight accepted the dataset and the restored optimizer and scheduler carried an old
    experiment on against different training images, which is exactly what the fingerprint
    exists to refuse."""
    image = tmp_path / "cat.png"
    image.write_bytes(b"A" * 4096)
    entries = [(str(image), "a cat")]
    before = dc.dataset_fingerprint(entries)

    replaced = b"B" * 4096
    image.write_bytes(replaced)
    assert image.stat().st_size == 4096, "the fixture must keep the length identical"

    assert dc.dataset_fingerprint(entries) != before

    # ...and it is stable when nothing changed, or a resume would never be offered at all.
    assert dc.dataset_fingerprint(entries) == dc.dataset_fingerprint(entries)


def test_the_fingerprint_probe_does_not_read_whole_images(tmp_path):
    """It runs on the route thread before the resident model is evicted, so it must not scale
    with the dataset. Head and tail only, whatever the file size."""
    import time

    small = tmp_path / "small.png"
    small.write_bytes(b"\0" * (2 * dc._PROBE_BYTES))
    big = tmp_path / "big.png"
    # Distinct bytes throughout, so a whole-file read could not be optimised away.
    big.write_bytes(bytes(range(256)) * (64 * 1024 // 256) * 128)
    assert big.stat().st_size >= 64 * 1024 * 128

    def _elapsed(path):
        started = time.perf_counter()
        for _ in range(20):
            dc.dataset_fingerprint([(str(path), "caption")])
        return time.perf_counter() - started

    # A whole-file hash would scale with the 128x size difference; head+tail does not.
    assert _elapsed(big) < 8 * max(
        _elapsed(small), 1e-4
    ), "the probe appears to scale with file size, so it is reading more than head and tail"

    # The direct statement of the same thing: two files that differ only in the middle, past
    # the probe window on both ends, are indistinguishable -- which is the documented tradeoff.
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    body = b"\0" * (4 * dc._PROBE_BYTES)
    a.write_bytes(body)
    b.write_bytes(body[: 2 * dc._PROBE_BYTES] + b"X" + body[2 * dc._PROBE_BYTES + 1 :])
    assert dc._content_probe(a) == dc._content_probe(b)


def test_a_mid_sized_image_is_covered_end_to_end(tmp_path):
    """A file between one and two probe windows is read IN FULL, not head-only.

    The old gate only sampled a tail past 2 x _PROBE_BYTES, so anything from 64 KiB to
    128 KiB -- which is most JPEGs in a LoRA dataset -- contributed its first 64 KiB and
    nothing else. A same-length replacement sharing that head kept the fingerprint intact
    and the preflight accepted changed training images.
    """
    path = tmp_path / "mid.jpg"
    head = b"H" * dc._PROBE_BYTES
    path.write_bytes(head + b"A" * (dc._PROBE_BYTES // 2))
    entries = [(str(path), "a cat")]
    before = dc.dataset_fingerprint(entries)

    path.write_bytes(head + b"B" * (dc._PROBE_BYTES // 2))
    assert path.stat().st_size == dc._PROBE_BYTES + dc._PROBE_BYTES // 2
    assert dc.dataset_fingerprint(entries) != before


def test_a_failed_first_save_does_not_retire_the_discard(run_dir, monkeypatch):
    """A run that has written nothing yet still owns its output directory.

    A transient failure at the first save_steps interval used to flip wrote_checkpoint
    anyway, so the next successful save ran with discard_existing=False and an earlier run's
    higher-numbered bundle survived beside it -- and a later Resume by output directory
    picked that stale bundle over this run's state.
    """
    import core.training.diffusion_train_common as dtc

    # An earlier run of the same adapter name left a bundle behind, at a higher step.
    stale = _Run(run_dir)
    stale.save(40)
    assert (run_dir / "checkpoint-40").is_dir()

    wrote_checkpoint = False
    run = _Run(run_dir)
    failing = {"n": 1}
    real_save = dc.save_checkpoint

    def _save(**kwargs):
        if failing["n"] > 0:
            failing["n"] -= 1
            raise OSError(28, "No space left on device")
        return real_save(**kwargs)

    monkeypatch.setattr(dc, "save_checkpoint", _save)

    def _attempt(step):
        nonlocal wrote_checkpoint
        written, _error = dtc.write_resume_checkpoint(
            run.cfg,
            step = step,
            model = run.model,
            optimizer = run.optimizer,
            lr_scheduler = run.lr_sched,
            identity = run.identity,
            sampler = run.sampler,
            rng_streams = run.streams,
            discard_existing = not wrote_checkpoint,
        )
        if written:
            wrote_checkpoint = True
        return written

    assert _attempt(5) is None  # ENOSPC: nothing was written
    assert wrote_checkpoint is False
    assert _attempt(6) is not None

    # The stale higher-numbered bundle is gone, so a Resume by output directory picks step 6.
    assert not (run_dir / "checkpoint-40").exists()
    latest = dc.latest_valid_checkpoint(run_dir)
    assert latest is not None and dc.checkpoint_step(latest[0]) == 6


def test_a_partial_ema_shadow_is_refused_rather_than_half_restored():
    """A readable EMA file covering only SOME of the live shadows is not a resumable EMA.

    load_state_dict skips what it cannot match by design, so continuing here averages
    restored shadows for some parameters against freshly initialised ones for the rest --
    for every later update and for the exported EMA adapter -- while the run reports a
    clean resume.
    """
    from core.training.diffusion_train_extras import LoRAEMA

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias = False), torch.nn.Linear(4, 4))
    ema = LoRAEMA(model, decay = 0.99)
    full = ema.state_dict()
    assert len(full) > 1

    complete = ema.missing_from(full)
    assert complete == ()

    partial = {next(iter(full)): next(iter(full.values()))}
    missing = ema.missing_from(partial)
    assert missing and len(missing) == len(full) - 1

    # A shape change counts too: load_state_dict skips those on the same branch.
    mangled = {name: tensor[:1].clone() for name, tensor in full.items()}
    assert len(ema.missing_from(mangled)) == len(full)


def test_a_resume_refuses_a_checkpoint_whose_ema_is_incomplete(run_dir):
    """The end of the same fence, on the real restore path: the run stops with a reason
    rather than continuing on a half-restored average."""
    from core.training.diffusion_checkpoint import ResumeError
    from core.training.diffusion_train_extras import LoRAEMA
    from safetensors.torch import save_file

    run = _Run(run_dir)
    ema = LoRAEMA(run.model, decay = 0.99)
    path, error = run.save(3, ema = ema)
    assert error is None and path is not None

    # A wider model on the resume side: the saved shadow covers only part of it, which is the
    # shape a re-wrap (or a hand-edited bundle) produces.
    save_file(
        {"weight": torch.zeros(4, 4)},
        str(Path(path) / dc.EMA_FILENAME),
    )
    resumed = _Run(run_dir, resume_from_checkpoint = path)
    resumed.model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias = False), torch.nn.Linear(4, 4, bias = False)
    )
    wider = LoRAEMA(resumed.model, decay = 0.99)

    with pytest.raises(ResumeError) as excinfo:
        restore_resume_state(
            resumed.cfg,
            model = run.model,
            optimizer = resumed.optimizer,
            lr_scheduler = resumed.lr_sched,
            identity = resumed.identity,
            sampler = resumed.sampler,
            rng_streams = resumed.streams,
            ema = wider,
        )
    assert "EMA state is missing or mis-shaped" in str(excinfo.value)


def test_the_sdxl_trainer_binds_its_output_dir_before_scanning_checkpoints():
    """`out_dir` was assigned only inside two early-return branches and at export time, so the
    pre-run checkpoint snapshot read it before any binding existed: every normal SDXL run raised
    UnboundLocalError after the whole model and cache setup, fresh and resumed alike. The loop
    needs a GPU to drive, so the ordering is checked against the source."""
    import inspect

    from core.training import diffusion_lora_trainer

    source = inspect.getsource(diffusion_lora_trainer).splitlines()
    scan = next(i for i, line in enumerate(source) if "snapshot_checkpoints(out_dir)" in line)
    # The binding that covers it has to be at the loop's own indentation -- one nested inside an
    # `if` that returns does not run on the normal path.
    scan_indent = len(source[scan]) - len(source[scan].lstrip())
    bound = [
        i
        for i, line in enumerate(source[:scan])
        if line.strip().startswith("out_dir = ") and len(line) - len(line.lstrip()) <= scan_indent
    ]
    assert bound, "snapshot_checkpoints(out_dir) is reached before out_dir is bound"


def test_a_bundle_overwritten_by_this_run_is_discarded_with_it(run_dir):
    """Ownership by pathname alone kept a bundle this run REPLACED.

    Resume checkpoint-10 in a folder that also holds checkpoint-15, periodically save at 15
    (which overwrites it), then discard: the original is already gone, and matching on the name
    preserved the replacement as though it were the bundle it destroyed.
    """
    seeded = _Run(run_dir)
    seeded.save(10)
    seeded.step_once(0.5)
    seeded.save(15)
    preexisting = dc.snapshot_checkpoints(run_dir)
    assert {p.name for p, _ in preexisting} == {"checkpoint-10", "checkpoint-15"}

    resumed = _Run(run_dir, seed = 3, resume_from_checkpoint = str(run_dir / "checkpoint-10"))
    resumed.step_once(1.5)
    path, error = resumed.save(15)
    assert error is None and path == str(run_dir / "checkpoint-15")

    dc.clear_own_checkpoints(run_dir, preexisting)

    survivors = {p.name for p in dc.list_checkpoints(run_dir)}
    assert survivors == {
        "checkpoint-10"
    }, f"the replacement written by the discarded run survived: {survivors}"


def test_resuming_into_another_directory_is_not_treated_as_owning_it(run_dir, tmp_path):
    from utils.paths import outputs_root

    elsewhere = outputs_root() / "other-run"
    elsewhere.mkdir(parents = True, exist_ok = True)
    source = _Run(elsewhere)
    source.save(20)

    same_dir = _Run(run_dir, resume_from_checkpoint = str(run_dir / "checkpoint-3"))
    assert dc.resumed_into_this_dir(same_dir.cfg, run_dir) is True
    by_folder = _Run(run_dir, resume_from_checkpoint = str(run_dir))
    assert dc.resumed_into_this_dir(by_folder.cfg, run_dir) is True

    # ...but a bundle from ANOTHER directory says nothing about what is in this one.
    cross = _Run(run_dir, resume_from_checkpoint = str(elsewhere / "checkpoint-20"))
    assert dc.resumed_into_this_dir(cross.cfg, run_dir) is False
    fresh = _Run(run_dir)
    assert dc.resumed_into_this_dir(fresh.cfg, run_dir) is False


def test_a_bundle_without_optimizer_state_is_refused(run_dir):
    """Loading only the adapter and calling it a resume restarts Adam's moments from zero at
    step N while reporting a clean continue -- a fresh run with a warm learning rate."""
    from core.training.diffusion_checkpoint import ResumeError

    run = _Run(run_dir)
    path, error = run.save(4)
    assert error is None and path is not None
    (Path(path) / dc.OPTIMIZER_FILENAME).unlink()
    manifest_path = Path(path) / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["files"].pop("optimizer", None)
    manifest.get("file_sizes", {}).pop(dc.OPTIMIZER_FILENAME, None)
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    resumed = _Run(run_dir, resume_from_checkpoint = path)
    with pytest.raises(ResumeError, match = "no optimizer state"):
        resumed.restore()


def test_the_identity_covers_lora_dropout(run_dir):
    """Both trainers pass lora_dropout to the LoRA constructor, so it changes the stochastic
    forward the restored moments were produced against."""
    run = _Run(run_dir, lora_dropout = 0.0)
    path, error = run.save(5)
    assert error is None

    changed = _Run(run_dir, lora_dropout = 0.15, resume_from_checkpoint = path)
    reason = dc.identity_for_config(run.cfg).mismatch_reason(dc.identity_for_config(changed.cfg))
    assert reason is not None and "dropout" in reason.lower()

    # An identity from before the field existed reads as unknown, which must not be a mismatch.
    older = dc.CheckpointIdentity.from_dict(
        {**dc.identity_for_config(run.cfg).as_dict(), "lora_dropout": None}
    )
    assert older is not None
    assert older.mismatch_reason(dc.identity_for_config(changed.cfg)) is None


def test_the_resumed_event_seeds_the_live_counters(runs_dir):
    """Until the first post-resume progress event, step and total_steps stayed at their idle
    values: the UI showed 0/0 after restoring step 400 of 500, and an OOM on the first step
    persisted an errored run with both recorded as zero. The event carries them."""
    from core.training.diffusion_training_service import DiffusionTrainingService

    service = DiffusionTrainingService()
    service._apply_event(
        {
            "type": "resumed",
            "step": 400,
            "total_steps": 500,
            "checkpoint_path": "/x/checkpoint-400",
        }
    )
    snapshot = service.status()
    assert snapshot["step"] == 400
    assert snapshot["total_steps"] == 500
    # ...and still not a bundle this run wrote.
    assert snapshot["resumed_from_step"] == 400
    assert snapshot["checkpoint_step"] is None


# ── round 6: the resume must not overstate what it can honour ─────────────────


def test_a_resume_into_a_new_folder_that_died_first_is_still_resumable(run_dir, tmp_path):
    """An OOM on the first restored step leaves the new output dir nonexistent.

    The bundle it was validated against is still sitting in the source dir, and continuing from
    it is the obvious retry -- but the missing-folder refusal returned before the source
    fallback could be consulted, so the run read as unresumable."""
    from utils.paths import outputs_root

    source = _Run(run_dir)
    source.save(10)
    never_created = outputs_root() / "resumed-run-that-died"
    assert not never_created.exists()

    state = dc.describe_resume_state(
        str(never_created),
        status = "error",
        source_checkpoint = str(run_dir / "checkpoint-10"),
    )
    assert state["can_resume"] is True
    assert state["checkpoint_step"] == 10
    assert state["checkpoint_path"].endswith("checkpoint-10")

    # With no source to fall back to, the missing folder is still the answer.
    blocked = dc.describe_resume_state(str(never_created), status = "error")
    assert blocked["can_resume"] is False
    assert "no longer exists" in blocked["resume_blocked_reason"]


def test_a_checkpoint_with_no_sampler_state_is_refused(run_dir):
    """The RNG is restored to a point AFTER the saved permutation was drawn, so a fresh sampler
    generates a different order: images silently skipped or repeated under a resume that
    reported success. Every bundle this format writes carries sampler state."""
    run = _Run(run_dir)
    run.step_once(0.1)
    run.save(3)
    bundle = run_dir / "checkpoint-3"
    manifest_path = bundle / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["sampler"] = {"n": 999, "order": [1, 2, 3], "pos": 0}  # not this dataset
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    resumed = _Run(run_dir)
    resumed.cfg = type(resumed.cfg)(
        **{**resumed.cfg.__dict__, "resume_from_checkpoint": str(bundle)}
    )
    with pytest.raises(dc.ResumeError, match = "sampler"):
        resumed.restore()


def test_the_sampler_reports_whether_it_restored():
    sampler = PermutationBatchSampler(4, __import__("random").Random(0))
    saved = sampler.state_dict()
    assert PermutationBatchSampler(4, __import__("random").Random(1)).load_state_dict(saved) is True
    for bad in (None, {}, {"n": 4, "order": "nope", "pos": 0}, {"n": 9, "order": [0], "pos": 0}):
        assert PermutationBatchSampler(4, __import__("random").Random(1)).load_state_dict(bad) is False


def test_the_identity_records_the_precision_the_run_will_actually_use(monkeypatch, run_dir):
    """A pre-Ampere card resolves a bf16 request to fp16. Recording the REQUEST let an fp16
    bundle resume in bf16 on a newer card, continuing restored moments under different
    frozen-base numerics and calling it a clean resume."""
    from core.training import diffusion_train_common as dtc

    cfg = _Run(run_dir).cfg
    import torch as _torch

    monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)

    monkeypatch.setattr(dtc, "native_bf16_supported", lambda: False)
    old_card = dc.identity_for_config(cfg)
    monkeypatch.setattr(dtc, "native_bf16_supported", lambda: True)
    new_card = dc.identity_for_config(cfg)

    assert old_card.precision == "fp16" and new_card.precision == "bf16"
    assert old_card.mismatch_reason(new_card) is not None, (
        "moving the same request between the two cards must not read as the same run"
    )


def test_the_revision_is_pinned_once_the_base_is_on_disk(monkeypatch, run_dir):
    """The identity is built before the multi-GB load, so the first run of an uncached repo
    records "unresolved" and the bundle can never enforce which commit it trained on."""
    from core.training import diffusion_train_extras as dte

    cfg = _Run(run_dir).cfg
    monkeypatch.setattr(dte, "source_revision", lambda ref: "unresolved")
    unresolved = dc.identity_for_config(cfg)
    assert unresolved.base_revision == "unresolved"

    monkeypatch.setattr(dte, "source_revision", lambda ref: "rev-abc123")
    pinned = dc.with_resolved_revision(unresolved, cfg.base_model)
    assert pinned.base_revision == "rev-abc123"

    # Already pinned, or still unresolvable: left exactly as it was.
    monkeypatch.setattr(dte, "source_revision", lambda ref: "rev-def456")
    assert dc.with_resolved_revision(pinned, cfg.base_model).base_revision == "rev-abc123"
    monkeypatch.setattr(dte, "source_revision", lambda ref: "unresolved")
    assert dc.with_resolved_revision(unresolved, cfg.base_model).base_revision == "unresolved"


def test_both_trainers_pin_the_revision_after_the_load():
    """Source-ordered: the loop needs a GPU, so this asserts the call sits after the load event
    and before the restore that validates against it."""
    for name in ("diffusion_lora_trainer", "diffusion_dit_trainer"):
        src = (
            Path(dc.__file__).resolve().parent / f"{name}.py"
        ).read_text(encoding = "utf-8")
        pin = src.index("identity = with_resolved_revision(")
        load = src.index('"model_load_completed"')
        restore = src.index("restore_resume_state(")
        assert load < pin < restore, f"{name}: the revision must be pinned between the two"


def test_a_raised_target_survives_a_run_that_died_before_the_resumed_event(run_dir):
    """Raise the target to continue a checkpoint already at its original one, then die during
    model loading. total_steps is seeded by the "resumed" event, which never arrived, so zero
    sent the calculation back to the manifest's older target and the run reported that there
    was nothing left to train -- for the one request that had something left."""
    from core.training import diffusion_training_service as svc

    run = _Run(run_dir)
    run.save(500)

    record = {
        "status": "error",
        "output_dir": str(run_dir),
        "config": {"output_dir": str(run_dir), "train_steps": 800},
    }
    refreshed = svc._refresh_resume_state(dict(record))
    assert refreshed["can_resume"] is True, refreshed["resume_blocked_reason"]
    assert refreshed["checkpoint_step"] == 500

    # And the original target still stops at the top, so this is not a blanket "always resumable".
    at_target = svc._refresh_resume_state(
        {**record, "config": {"output_dir": str(run_dir), "train_steps": 500}}
    )
    assert at_target["can_resume"] is False
    assert "nothing left to train" in at_target["resume_blocked_reason"]
