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
    _apply_perf_flags,
    _config_from_dict,
    _plan_cache_variants,
    _restore_perf_flags,
)
from core.training.diffusion_training_service import DiffusionTrainingService
from models.training import DiffusionTrainingStartRequest, DiffusionTrainingStopRequest
from routes.training import router as training_router

# A trainable SDXL base so DiffusionLoraConfig.normalized() resolves a family without a
# network call (resolve_trainable_family is pure name matching for this repo).
_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


def _cfg(**kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(base_model = _SDXL, data_dir = "d", output_dir = "o", **kw)


# ── _plan_cache_variants (pure, seed-deterministic) ───────────────────────────
def test_plan_cache_variants_deterministic_and_deduped():
    # Same seed -> byte-identical plan (its own rng stream, so it is fully reproducible).
    p1 = _plan_cache_variants(3, 4, center_crop = False, random_flip = True, seed = 123)
    p2 = _plan_cache_variants(3, 4, center_crop = False, random_flip = True, seed = 123)
    assert p1 == p2
    assert len(p1) == 3

    # cache_variants=1 -> exactly one variant per image.
    p_one = _plan_cache_variants(3, 1, center_crop = False, random_flip = True, seed = 7)
    assert [len(v) for v in p_one] == [1, 1, 1]

    # A center crop with no flip collapses to a single distinct variant no matter how many
    # draws are requested, and that variant is the fixed (0.5, 0.5, False) center.
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
    # A short (mask=None) and a long (mask=ones) entry -> pad to the batch max and build the
    # validity mask, with the padded tail of the short sample masked out.
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

    # A single sample pinned to a compile pad bucket must pad AND expose a mask so the padded
    # positions are attended to as invalid.
    pe2, mask2 = _qwen_collate([(torch.randn(1, 5, dim), None)], "cpu", torch.float32, pad_to = 16)
    assert pe2.shape == (1, 16, dim)
    assert mask2 is not None
    assert torch.equal(mask2[0, 5:], torch.zeros(11, dtype = mask2.dtype))


def test_zimage_collate_list():
    # Z-Image uses list I/O: the batch is one tuple carrying a list of per-sample tensors, each
    # cast to the requested dtype.
    entries = [(torch.randn(7, 2560),), (torch.randn(9, 2560),)]
    out = _zimage_collate(entries, "cpu", torch.float32)
    assert isinstance(out, tuple) and len(out) == 1
    (caps,) = out
    assert isinstance(caps, list) and len(caps) == 2
    assert all(t.dtype == torch.float32 for t in caps)


# ── index-based sigma gather ──────────────────────────────────────────────────
def test_gather_sigmas_matches_search_based_gather():
    from diffusers import FlowMatchEulerDiscreteScheduler

    torch.manual_seed(0)
    sched = FlowMatchEulerDiscreteScheduler()  # default init, no from_pretrained / no network
    timesteps, indices = _sample_timesteps(sched, 16, "cpu")

    # The index path must return exactly what the old per-item timestep-matching search did.
    schedule_timesteps = sched.timesteps.to("cpu")
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    assert step_indices == indices.tolist()

    sigma = _gather_sigmas(sched, indices, "cpu", torch.float32, 4)
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


# ── torch.compile policy ──────────────────────────────────────────────────────
def test_should_compile_policy():
    # off never compiles, even on cuda.
    assert _should_compile(_cfg(compile_transformer = "off"), False, "cuda") is False
    # on always compiles on cuda.
    assert _should_compile(_cfg(compile_transformer = "on"), False, "cuda") is True
    # auto stays off over a bitsandbytes base (graph breaks in the dequant path).
    assert _should_compile(_cfg(compile_transformer = "auto"), True, "cuda") is False
    # auto turns on for a dense (non-bnb) base on cuda.
    assert _should_compile(_cfg(compile_transformer = "auto"), False, "cuda") is True
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
    # On a cpu device (or a torch build without cuda), applying the perf flags is a no-op
    # snapshot path and restoring it must not raise.
    snap = _apply_perf_flags(_cfg(), "cpu")
    assert isinstance(snap, dict)
    _restore_perf_flags(snap)  # no exception
