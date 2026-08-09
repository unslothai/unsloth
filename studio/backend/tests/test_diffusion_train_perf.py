# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion training performance work.

Covers the new pure helpers and small policy functions that the perf PR adds:
the seed-deterministic latent-cache crop/flip plan, the per-family collate fns, the
index-based sigma gather, the new config validation + request-model fields, the
torch.compile policy, the stop save/cancel flag, and the ``preparing`` / ``warning``
service events. No GPU / model load: the collates and gathers run on CPU tensors, the
scheduler is default-initialised (no ``from_pretrained``), and the route/service tests
inject in-thread fakes exactly like ``test_diffusion_training.py``.
"""

from __future__ import annotations

import itertools

from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training.diffusion_dit_trainer import (
    _flux_collate,
    _gather_sigmas,
    _qwen_collate,
    _sample_timesteps,
    _should_compile,
    _zimage_collate,
)
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    LATENT_CACHE_OVER_BUDGET,
    _apply_perf_flags,
    _config_from_dict,
    _latent_cache_forced,
    _latent_cache_over_budget,
    _plan_cache_variants,
    _restore_perf_flags,
)
import core.training.diffusion_lora_trainer as sdxl_trainer
import core.training.diffusion_train_common as train_common
from core.training.diffusion_training_service import DiffusionTrainingService
from models.training import DiffusionTrainingStartRequest, DiffusionTrainingStopRequest
from routes.training import router as training_router

# A trainable SDXL base so normalized() resolves a family without a network call (pure name matching).
_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


def _cfg(**kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(base_model = _SDXL, data_dir = "d", output_dir = "o", **kw)


# ── _plan_cache_variants (pure, seed-deterministic) ───────────────────────────
def test_plan_cache_variants_deterministic_and_deduped():
    # Same seed gives a byte-identical plan (its own rng stream, so it is fully reproducible).
    p1 = _plan_cache_variants(3, 4, center_crop = False, random_flip = True, seed = 123)
    p2 = _plan_cache_variants(3, 4, center_crop = False, random_flip = True, seed = 123)
    assert p1 == p2
    assert len(p1) == 3

    # cache_variants=1 -> exactly one variant per image.
    p_one = _plan_cache_variants(3, 1, center_crop = False, random_flip = True, seed = 7)
    assert [len(v) for v in p_one] == [1, 1, 1]

    # A center crop with no flip collapses to one variant, the fixed (0.5, 0.5, False) center, however many draws are asked for.
    p_cc = _plan_cache_variants(2, 8, center_crop = True, random_flip = False, seed = 7)
    assert [len(v) for v in p_cc] == [1, 1]
    assert p_cc[0][0] == (0.5, 0.5, False)

    # A center crop WITH flip has at most two distinct variants (flip on/off; crop is fixed).
    p_cf = _plan_cache_variants(2, 8, center_crop = True, random_flip = True, seed = 7)
    assert all(len(v) <= 2 for v in p_cf)

    # Every crop fraction is a valid unit fraction the loader can map onto its crop range.
    for u_left, u_top, flip in itertools.chain.from_iterable(p1):
        assert 0.0 <= u_left < 1.0
        assert 0.0 <= u_top < 1.0
        assert isinstance(flip, bool)


# ── per-family collate fns ────────────────────────────────────────────────────
def test_flux_collate_shapes():
    # FLUX embeds are fixed length: 3 entries batch by a plain cat; text_ids are shared.
    entries = [(torch.randn(1, 512, 32), torch.randn(1, 16), torch.randn(512, 3)) for _ in range(3)]
    pe, pooled, text_ids = _flux_collate(entries, "cpu", torch.float32)
    assert pe.shape == (3, 512, 32)
    assert pooled.shape == (3, 16)
    assert text_ids.shape == (512, 3)
    # Position ids stay float32 regardless of the requested weight dtype.
    assert pe.dtype == torch.float32
    assert pooled.dtype == torch.float32
    assert text_ids.dtype == torch.float32


def test_qwen_collate_pads_and_masks():
    dim = 8
    # A short (mask=None) and a long (mask=ones) entry pad to the batch max, with the short sample's padded tail masked out.
    short = (torch.randn(1, 5, dim), None)
    long = (torch.randn(1, 9, dim), torch.ones(1, 9, dtype = torch.int64))
    pe, mask = _qwen_collate([short, long], "cpu", torch.float32)
    assert pe.shape == (2, 9, dim)
    assert mask.shape == (2, 9)
    assert torch.equal(mask[0, 5:], torch.zeros(4, dtype = mask.dtype))

    # A single unpadded sample with a None mask keeps the legacy None mask (no behaviour delta).
    pe1, mask1 = _qwen_collate([(torch.randn(1, 5, dim), None)], "cpu", torch.float32)
    assert pe1.shape == (1, 5, dim)
    assert mask1 is None

    # A single sample pinned to a compile pad bucket must pad AND expose a mask so the padded positions read as invalid.
    pe2, mask2 = _qwen_collate([(torch.randn(1, 5, dim), None)], "cpu", torch.float32, pad_to = 16)
    assert pe2.shape == (1, 16, dim)
    assert mask2 is not None
    assert torch.equal(mask2[0, 5:], torch.zeros(11, dtype = mask2.dtype))


def test_zimage_collate_list():
    # Z-Image uses list I/O: one tuple carrying a list of per-sample tensors, each cast to the requested dtype.
    entries = [(torch.randn(7, 2560),), (torch.randn(9, 2560),)]
    out = _zimage_collate(entries, "cpu", torch.float32)
    assert isinstance(out, tuple) and len(out) == 1
    (caps,) = out
    assert isinstance(caps, list) and len(caps) == 2
    assert all(t.dtype == torch.float32 for t in caps)


# ── index-based sigma gather ──────────────────────────────────────────────────
def test_gather_sigmas_matches_search_based_gather():
    # CI installs the backend test deps without diffusers, and the scheduler math is what we check, so skip there.
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    torch.manual_seed(0)
    sched = FlowMatchEulerDiscreteScheduler()  # default init, no from_pretrained / no network
    timesteps, indices = _sample_timesteps(sched, 16, "cpu")

    # The index path must return exactly what the old per-item timestep-matching search did.
    schedule_timesteps = sched.timesteps.to("cpu")
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    assert step_indices == indices.tolist()

    # _gather_sigmas takes the sigma TABLE (identity here: no flow shift), not the scheduler.
    sigma = _gather_sigmas(sched.sigmas, indices, "cpu", torch.float32, 4)
    assert sigma.ndim == 4
    expected = sched.sigmas[step_indices].flatten()
    while expected.ndim < 4:
        expected = expected.unsqueeze(-1)
    assert torch.equal(sigma, expected)


# ── config validation of the new perf fields ──────────────────────────────────
def test_config_validates_new_fields():
    # Defaults normalize cleanly and carry the new perf fields through.
    norm = _cfg().normalized()
    assert norm.cache_variants == 4
    assert norm.compile_transformer == "auto"
    assert norm.enable_tf32 is True
    assert norm.cache_latents is True

    # cache_variants is bounded to 1..16 inclusive.
    for bad in (0, 17):
        with pytest.raises(ValueError):
            _cfg(cache_variants = bad).normalized()

    # An unknown compile mode is rejected.
    with pytest.raises(ValueError):
        _cfg(compile_transformer = "banana").normalized()

    # compile_transformer is case/space-insensitive and stored lowered.
    assert _cfg(compile_transformer = " ON ").normalized().compile_transformer == "on"

    # The generic Studio dict path preserves the flags without inventing defaults.
    cfg = _config_from_dict(
        {
            "base_model": _SDXL,
            "data_dir": "d",
            "output_dir": "o",
            "enable_tf32": False,
            "cache_latents": False,
        }
    )
    assert cfg.enable_tf32 is False
    assert cfg.cache_latents is False

    # String flags from the generic Studio dict path are coerced: "false" is a truthy string, so an opt-out would silently no-op.
    cfg = _config_from_dict(
        {
            "base_model": _SDXL,
            "data_dir": "d",
            "output_dir": "o",
            "enable_tf32": "false",
            "cache_latents": "0",
        }
    )
    assert cfg.enable_tf32 is False
    assert cfg.cache_latents is False


# ── torch.compile policy ──────────────────────────────────────────────────────
def test_should_compile_policy():
    # off never compiles, even on cuda.
    assert _should_compile(_cfg(compile_transformer = "off"), False, "cuda") is False
    # on always compiles on cuda.
    assert _should_compile(_cfg(compile_transformer = "on"), False, "cuda") is True
    # auto stays off over a bitsandbytes base (graph breaks in the dequant path).
    assert _should_compile(_cfg(compile_transformer = "auto"), True, "cuda") is False
    # auto turns on for the dense bf16 base precision on cuda.
    assert (
        _should_compile(_cfg(compile_transformer = "auto"), False, "cuda", base_precision = "bf16")
        is True
    )
    # Any mode is a no-op on cpu.
    for mode in ("off", "on", "auto"):
        assert _should_compile(_cfg(compile_transformer = mode), False, "cpu") is False


# ── service stop save/cancel flag ─────────────────────────────────────────────
class _StopQueue:
    """Records what stop() puts on the wire (put-only for these tests)."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, x) -> None:
        self.items.append(x)


class _AliveProc:
    def is_alive(self) -> bool:
        return True


def test_service_stop_save_flag():
    svc = DiffusionTrainingService()
    # Nothing running -> stop is a no-op and returns False.
    assert svc.stop() is False

    # Attach a fake live proc + stop queue so stop() has a target.
    svc._proc = _AliveProc()
    q = _StopQueue()
    svc._stop_queue = q

    # save=False is the cancel path: the dict form {"save": False} goes on the queue.
    assert svc.stop(save = False) is True
    assert q.items[-1] == {"save": False}

    # The default (save) path keeps the bare-True wire format.
    assert svc.stop() is True
    assert q.items[-1] is True


# ── preparing / warning events + stopped completion messages ──────────────────
def test_apply_event_preparing_and_warning():
    svc = DiffusionTrainingService()
    svc._apply_event({"type": "preparing", "stage": "cache_latents", "done": 4, "total": 8})
    st = svc.status()
    assert st["status"] == "running"
    assert st["in_model_load"] is True
    assert "4/8" in st["message"]

    svc._apply_event({"type": "warning", "message": "compile disabled"})
    assert svc.status()["message"] == "compile disabled"

    # A stop with no saved adapter reports the no-adapter message and the stopped status.
    svc_no = DiffusionTrainingService()
    svc_no._apply_event({"type": "complete", "stopped": True, "lora_path": None})
    st_no = svc_no.status()
    assert st_no["status"] == "stopped"
    assert st_no["message"] == "Stopped (no adapter saved)."

    # A stop that DID save a partial adapter reports the partial-adapter message.
    svc_partial = DiffusionTrainingService()
    svc_partial._apply_event(
        {"type": "complete", "stopped": True, "lora_path": "/o/pytorch_lora_weights.safetensors"}
    )
    assert svc_partial.status()["message"] == "Stopped (partial adapter saved)."


# ── route: stop body forwards the save flag ───────────────────────────────────
class _FakeService:
    """Records the save flag the /diffusion/stop route forwards. A local copy of the
    test_diffusion_training.py pattern so the two suites stay decoupled."""

    def __init__(self) -> None:
        self._running = True
        self.stopped_with_save = None

    def stop(self, save = True):
        self.stopped_with_save = save
        was = self._running
        self._running = False
        return was


@pytest.fixture
def client(monkeypatch):
    fake = _FakeService()
    monkeypatch.setattr(
        "core.training.diffusion_training_service.get_diffusion_training_service", lambda: fake
    )
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    c = TestClient(app)
    c._fake = fake  # type: ignore[attr-defined]
    return c


def test_route_stop_save_body(client):
    # An explicit {"save": false} body forwards save=False to the service.
    r = client.post("/api/train/diffusion/stop", json = {"save": False})
    assert r.status_code == 200, r.text
    assert client._fake.stopped_with_save is False

    # A body-less POST defaults to save=True.
    r2 = client.post("/api/train/diffusion/stop")
    assert r2.status_code == 200, r2.text
    assert client._fake.stopped_with_save is True


# ── request models: new perf fields + stop schema ─────────────────────────────
def test_request_models_new_fields():
    req = DiffusionTrainingStartRequest(base_model = "b", data_dir = "d", output_dir = "o")
    assert req.cache_latents is True
    assert req.cache_variants == 4
    assert req.compile_transformer == "auto"
    assert req.enable_tf32 is True

    # cache_variants is validated against its 1..16 bound by pydantic.
    with pytest.raises(Exception):
        DiffusionTrainingStartRequest(
            base_model = "b", data_dir = "d", output_dir = "o", cache_variants = 32
        )

    # The stop request defaults to saving a partial adapter.
    assert DiffusionTrainingStopRequest().save is True


# ── perf flags round-trip on cpu ──────────────────────────────────────────────
def test_perf_flags_cpu_roundtrip():
    # On cpu (or a torch build without cuda) applying the perf flags is a no-op snapshot, and restoring it must not raise.
    snap = _apply_perf_flags(_cfg(), "cpu")
    assert isinstance(snap, dict)
    _restore_perf_flags(snap)  # no exception


def test_perf_flags_tf32_off_clears_flags():
    # enable_tf32=False is the strict-fp32 A/B mode: it must actively clear the TF32 flags (cudnn TF32 defaults ON) rather
    # than inherit ambient state, and restore must put them back. The flags are plain Python state, so no GPU is needed.
    import torch

    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    snap = _apply_perf_flags(_cfg(enable_tf32 = False), "cuda")
    try:
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        _restore_perf_flags(snap)
    after = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    assert after == before


# ── latent cache size gate ────────────────────────────────────────────────────
class _FakeLatentDist:
    def __init__(self, shape):
        self.mean = torch.zeros(shape, dtype = torch.float32)
        self.std = torch.ones(shape, dtype = torch.float32)


class _FakeEncoded:
    def __init__(self, shape):
        self.latent_dist = _FakeLatentDist(shape)


class _FakeVae:
    # Minimal VAE stand-in: encode() returns a posterior of the requested latent shape, so the builder measures a real per-variant size with no model or images.
    def __init__(self, shape):
        self._shape = shape

    def encode(self, pixel_values):
        return _FakeEncoded(self._shape)


def _fake_planned_loader(path, resolution, center_crop, u_left, u_top, flip):
    # The fake VAE ignores pixels; return a valid tensor + square SDXL time_ids.
    tensor = torch.zeros(3, resolution, resolution, dtype = torch.float32)
    return tensor, (resolution, resolution, 0, 0, resolution, resolution)


def _build_fake_sdxl_cache(monkeypatch, num_images, latent_shape):
    # center_crop + no flip collapses to one variant per image, so total_variants == num_images.
    monkeypatch.setattr(sdxl_trainer, "_load_image_tensor_planned", _fake_planned_loader)
    cfg = _cfg(cache_variants = 1, center_crop = True, random_flip = False).normalized()
    return sdxl_trainer._build_sdxl_latent_cache(
        _FakeVae(latent_shape),
        1.0,
        [f"img{i}.png" for i in range(num_images)],
        cfg,
        "cpu",
        torch.float32,
        None,
        lambda: False,
    )


def test_latent_cache_over_budget_boundary():
    # 32 bytes per variant x 4 variants = 128 bytes; exactly at budget is not "over".
    assert _latent_cache_over_budget(32, 4, budget_bytes = 200) is False
    assert _latent_cache_over_budget(32, 4, budget_bytes = 128) is False
    assert _latent_cache_over_budget(32, 4, budget_bytes = 127) is True
    # An empty plan can never overflow.
    assert _latent_cache_over_budget(1_000_000, 0, budget_bytes = 1) is False


def test_latent_cache_forced_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", raising = False)
    assert _latent_cache_forced() is False
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", "1")
    assert _latent_cache_forced() is True


def test_sdxl_cache_built_under_budget(monkeypatch):
    # Default (4 GiB) budget: a handful of tiny latents fits, so the full cache is returned.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", raising = False)
    cache = _build_fake_sdxl_cache(monkeypatch, num_images = 3, latent_shape = (1, 4, 8, 8))
    assert cache is not LATENT_CACHE_OVER_BUDGET and cache is not None
    assert len(cache) == 3
    assert all(len(variants) == 1 for variants in cache)


def test_sdxl_cache_gated_over_budget(monkeypatch):
    # A budget below one variant trips the gate on the first encode: the sentinel tells the caller to keep the VAE resident and encode per step.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", raising = False)
    monkeypatch.setattr(train_common, "_LATENT_CACHE_BUDGET_BYTES", 8)
    cache = _build_fake_sdxl_cache(monkeypatch, num_images = 3, latent_shape = (1, 4, 8, 8))
    assert cache is LATENT_CACHE_OVER_BUDGET


def test_sdxl_cache_force_bypasses_gate(monkeypatch):
    # An explicit force-on must be honoured verbatim even when the estimate is over budget.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", "1")
    monkeypatch.setattr(train_common, "_LATENT_CACHE_BUDGET_BYTES", 8)
    cache = _build_fake_sdxl_cache(monkeypatch, num_images = 3, latent_shape = (1, 4, 8, 8))
    assert cache is not LATENT_CACHE_OVER_BUDGET and cache is not None
    assert len(cache) == 3


def test_a_no_save_stop_survives_a_child_that_dies_before_reporting_it():
    """The trainer reports the discard on its completion event, but a child that OOMs or is
    killed after the request never emits one. The unexpected-exit path then recorded a plain
    error run with the last periodic checkpoint intact, so the history offered Resume from the
    very bundle the user asked to throw away. The intent is remembered in the parent."""

    class _DeadProc:
        def is_alive(self) -> bool:
            return False

    svc = DiffusionTrainingService()
    svc._proc = _AliveProc()
    svc._stop_queue = _StopQueue()
    svc._apply_event(
        {"type": "checkpoint_saved", "checkpoint_path": "/o/checkpoint-40", "step": 40}
    )
    assert svc.status()["resume_blocked_reason"] is None

    assert svc.stop(save = False) is True
    # ...and the child dies without a completion event.
    dead = _DeadProc()
    svc._proc = dead

    class _EmptyQueue:
        def get(self, timeout = None):
            raise RuntimeError("empty")

        def get_nowait(self):
            raise RuntimeError("empty")

    svc._pump_loop(_EmptyQueue(), dead)

    state = svc.status()
    assert state["status"] == "error"
    assert "stopped without saving" in (state["resume_blocked_reason"] or "")


def test_a_fresh_job_forgets_the_previous_no_save_stop():
    """The flag is per job. Carrying it across would block a resume the next run legitimately
    offers."""
    import inspect

    svc = DiffusionTrainingService()
    assert svc._discard_requested is False
    svc._proc = _AliveProc()
    svc._stop_queue = _StopQueue()
    assert svc.stop(save = False) is True
    assert svc._discard_requested is True
    # start() clears it for the next job, so a discarded run cannot block the next one's resume.
    assert "_discard_requested = False" in inspect.getsource(DiffusionTrainingService.start)


def test_a_discard_is_applied_to_a_terminal_error_too(tmp_path, monkeypatch):
    """An exception on the current step is a terminal `error`, and the pump returns on that path
    rather than through the dead-process branch -- so a stop-without-saving followed by a crash
    left the periodic checkpoint resumable in history."""
    import core.training.diffusion_checkpoint as dc

    bundle = tmp_path / "checkpoint-40"
    bundle.mkdir()
    (bundle / "keep.bin").write_bytes(b"x")

    class _Proc:
        def is_alive(self) -> bool:
            return True

    class _OneEvent:
        def __init__(self, ev):
            self._events = [ev]

        def get(self, timeout = None):
            if self._events:
                return self._events.pop(0)
            raise RuntimeError("empty")

    svc = DiffusionTrainingService()
    proc = _Proc()
    svc._proc = proc
    svc._stop_queue = _StopQueue()
    svc._apply_event({"type": "checkpoint_saved", "checkpoint_path": str(bundle), "step": 40})
    assert svc.stop(save = False) is True
    monkeypatch.setattr(svc, "_persist_run_record", lambda **_kw: None)

    svc._pump_loop(_OneEvent({"type": "error", "message": "CUDA out of memory"}), proc)

    state = svc.status()
    assert state["status"] == "error"
    assert "stopped without saving" in (state["resume_blocked_reason"] or "")
    # And the bundles the run wrote are gone: they hold optimizer state and the UI offers no
    # delete path once the run is marked discarded.
    assert not bundle.exists()
    assert dc is not None


def test_a_killed_discard_removes_only_this_runs_bundles(tmp_path):
    """The parent knows exactly which bundles it saw checkpoint_saved for, so an earlier run's
    leftovers in the same directory are untouched."""

    class _Dead:
        def is_alive(self) -> bool:
            return False

    class _Empty:
        def get(self, timeout = None):
            raise RuntimeError("empty")

        def get_nowait(self):
            raise RuntimeError("empty")

    earlier = tmp_path / "checkpoint-99"
    earlier.mkdir()
    mine = tmp_path / "checkpoint-40"
    mine.mkdir()

    svc = DiffusionTrainingService()
    svc._proc = _AliveProc()
    svc._stop_queue = _StopQueue()
    svc._apply_event({"type": "checkpoint_saved", "checkpoint_path": str(mine), "step": 40})
    assert svc.stop(save = False) is True
    dead = _Dead()
    svc._proc = dead
    svc._pump_loop(_Empty(), dead)

    assert not mine.exists(), "this run's bundle must go"
    assert earlier.is_dir(), "an earlier run's bundle is not this run's to delete"


def test_an_epoch_mode_target_does_not_fall_back_to_the_unused_step_count():
    """num_epochs overrides train_steps, which then still carries the request model's default of
    500. Using it for a run that died before its `resumed` event reported a 600-step checkpoint
    as 600/500 and refused the resume."""
    from core.training.diffusion_training_service import _resolved_total_steps

    assert (
        _resolved_total_steps({"total_steps": 1000}, {"num_epochs": 4, "train_steps": 500}) == 1000
    )
    assert _resolved_total_steps({}, {"num_epochs": 4, "train_steps": 500}) == 0
    # Step mode is unchanged: the configured count is the target.
    assert _resolved_total_steps({}, {"num_epochs": 0, "train_steps": 500}) == 500


def test_the_read_time_refresh_uses_the_same_epoch_rule():
    """The persisted record may carry the right target, but every read recomputes it -- and the
    read side was still falling back to the request model's unused train_steps in epoch mode,
    so a 600-step checkpoint of a run resolved to 1000 read as 600/500 and Resume was
    disabled again the moment the run was listed."""
    import inspect

    from core.training.diffusion_training_service import _refresh_resume_state

    source = inspect.getsource(_refresh_resume_state)
    assert "_resolved_total_steps(" in source, "the read side must use the shared rule"
    assert 'config.get("train_steps")' not in source, "and not the raw fallback it replaced"


def test_the_source_identity_is_seeded_before_the_child_starts():
    """The route has already validated and pinned a source bundle, so a resume that dies during
    the model load -- before the trainer can emit `resumed` -- still needs the timestamp its
    fallback is checked against, or the pathname alone offers back whatever later occupies the
    slot."""
    import inspect

    from core.training.diffusion_training_service import DiffusionTrainingService

    start = inspect.getsource(DiffusionTrainingService.start)
    assert "_seed_source_identity(config)" in start
    seeded = start.index("_seed_source_identity(config)")
    # AFTER the state reset, which replaces the whole dict and would drop the seed...
    assert seeded > start.index("self._state = _idle_state()")
    # ...and before the pump thread starts, which is the only other writer of that state.
    assert seeded < start.index("self._pump.start()")


def test_seeding_reads_the_bundles_own_timestamp(tmp_path, monkeypatch):
    import json as _json

    from core.training.diffusion_training_service import DiffusionTrainingService

    bundle = tmp_path / "checkpoint-10"
    bundle.mkdir()
    svc = DiffusionTrainingService()
    monkeypatch.setattr(
        "core.training.diffusion_checkpoint.read_checkpoint",
        lambda path: {"created_at": 1234.5} if Path(path) == bundle else None,
    )
    svc._seed_source_identity({"resume_from_checkpoint": str(bundle)})
    assert svc._state["resumed_source_created_at"] == 1234.5
    # An unreadable or absent bundle simply records nothing rather than raising.
    svc._seed_source_identity({"resume_from_checkpoint": str(tmp_path / "checkpoint-99")})
    assert _json.dumps(svc._state["resumed_source_created_at"]) in ("1234.5", "null")


def test_a_child_that_cleaned_up_is_not_cleaned_up_again(tmp_path, monkeypatch):
    """The trainer's own discard hands a displaced slot back to the bundle it replaced, so the
    path this run wrote to now holds ANOTHER run's checkpoint. The parent cleanup knows only
    pathnames, and repeating it deleted that restored original -- cancelling one branch
    destroyed a different run's resume point."""
    from core.training.diffusion_training_service import DiffusionTrainingService

    class _Proc:
        def is_alive(self) -> bool:
            return True

    class _OneEvent:
        def __init__(self, ev):
            self._events = [ev]

        def get(self, timeout = None):
            if self._events:
                return self._events.pop(0)
            raise RuntimeError("empty")

    # What the child restored into the slot this run had written over.
    restored = tmp_path / "checkpoint-10"
    restored.mkdir()
    (restored / "adapter.safetensors").write_bytes(b"other run")

    svc = DiffusionTrainingService()
    proc = _Proc()
    svc._proc = proc
    svc._stop_queue = _StopQueue()
    svc._apply_event({"type": "checkpoint_saved", "checkpoint_path": str(restored), "step": 10})
    assert svc.stop(save = False) is True
    monkeypatch.setattr(svc, "_persist_run_record", lambda **_kw: None)

    svc._pump_loop(
        _OneEvent({"type": "complete", "output_dir": str(tmp_path), "stopped": True,
                   "discarded": True}),
        proc,
    )

    assert restored.exists(), "the bundle the child handed back is not this run's to delete"
    # The state half of the discard still applies.
    state = svc.status()
    assert "stopped without saving" in (state["resume_blocked_reason"] or "")
    assert state["checkpoint_path"] is None


def test_a_checkpoint_makes_the_run_recoverable_before_it_ends(tmp_path, monkeypatch):
    """Previous runs is built from the run JSONs and nothing else, and only a terminal event
    wrote one. Studio being killed after a periodic save therefore left a resumable bundle on
    disk with no entry and no Resume action for it."""
    import inspect
    import json as _json

    from core.training import diffusion_training_service as svc_mod

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(svc_mod, "_runs_dir", lambda: runs)

    svc = svc_mod.DiffusionTrainingService()
    svc._state.update(job_id = "a" * 32, output_dir = str(tmp_path / "out"), status = "running")

    # A landed checkpoint asks for the record...
    svc._apply_event(
        {"type": "checkpoint_saved", "checkpoint_path": str(tmp_path / "checkpoint-40"),
         "step": 40}
    )
    assert svc._persist_interim is True
    # ...and the pump writes it, rather than waiting for a terminal event that may never come.
    assert "_persist_run_record(interim = True)" in inspect.getsource(
        svc_mod.DiffusionTrainingService._pump_loop
    )

    svc._persist_run_record(interim = True)
    written = runs / f"{'a' * 32}.json"
    assert written.exists(), "the checkpoint landed but nothing recorded the run"
    record = _json.loads(written.read_text(encoding = "utf-8"))
    assert record["job_id"] == "a" * 32
    # Interrupted is what it IS until the run ends; the terminal write replaces this file.
    assert record["status"] == "error"

    # ...and while the process is still on that job, the reader says so rather than offering a
    # Resume for a directory the live run is writing into.
    svc_mod._service = svc
    try:
        assert svc_mod._restate_live_job(dict(record))["status"] == "running"
    finally:
        svc_mod._service = None
