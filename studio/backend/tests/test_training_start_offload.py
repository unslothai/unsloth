# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/train/start must run backend.start_training off the event loop.

start_training() runs the _free_vram_for_training before_spawn hook inline, and that
hook's diffusion/video unload() blocks on the engines' generation locks until an
in-flight denoise step reaches its cancel callback (seconds to tens of seconds for
video). Executed inline in the async route it would freeze every concurrent
status/cancel/UI request -- the same reason start_diffusion_training offloads
_free_gpu_for_diffusion_training via asyncio.to_thread. The backend guards the
overlapping-starts window this offload opens with a compare-and-set flag.
"""

import asyncio
import threading

import routes.training as tr
from models import TrainingStartRequest


class _FakeBackend:
    def __init__(self, result = True):
        self._result = result
        self.start_thread = None
        self.hook = None
        self.current_job_id = None

    def is_training_active(self):
        return False

    def start_training(
        self,
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        # The real backend runs before_spawn synchronously inside this call, so the
        # thread this method runs on is the thread the blocking VRAM hook runs on.
        self.start_thread = threading.current_thread()
        self.hook = before_spawn
        self.current_job_id = job_id
        return self._result


def _request() -> TrainingStartRequest:
    return TrainingStartRequest(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/data",
        # Skip the YAML trust_remote_code lookup (needs the model catalog on disk).
        trust_remote_code = True,
    )


def test_start_route_offloads_blocking_start(monkeypatch):
    fake = _FakeBackend()
    monkeypatch.setattr(tr, "get_training_backend", lambda: fake)
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)

    async def _run():
        return threading.current_thread(), await tr.start_training(
            request = _request(), current_subject = "test-user", via_api_key = False
        )

    loop_thread, resp = asyncio.run(_run())

    assert resp.status == "queued", resp
    # The VRAM-freeing hook was wired in and the blocking call left the loop thread.
    assert fake.hook is not None
    assert fake.start_thread is not None
    assert fake.start_thread is not loop_thread


def test_backend_start_guard_blocks_overlapping_starts():
    # With the route offloaded to worker threads, two overlapping /train/start requests
    # can reach TrainingBackend.start_training concurrently; the compare-and-set
    # _start_in_progress flag must let exactly one of them spawn.
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    first_entered = threading.Event()
    release_first = threading.Event()
    results = {}

    def _slow_impl(
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        first_entered.set()
        release_first.wait(timeout = 5.0)
        return True

    backend._start_training_impl = _slow_impl

    def _first():
        results["first"] = backend.start_training("job-a")

    t = threading.Thread(target = _first, daemon = True)
    t.start()
    assert first_entered.wait(timeout = 5.0)
    # Second start while the first is still inside the impl: refused by the guard,
    # without ever entering the impl.
    results["second"] = backend.start_training("job-b")
    release_first.set()
    t.join(timeout = 5.0)

    assert results["first"] is True
    assert results["second"] is False
    # The flag is cleared once the winning start returns, so a later start may proceed.
    assert backend._start_in_progress is False


def test_is_training_active_true_during_start_reservation():
    # While start_training holds the compare-and-set reservation but has not yet spawned
    # (before_spawn frees residents, then GPU auto-selection, then proc.start()), the LLM
    # training run must already read as active: /images/load, /video/load, and
    # /diffusion/start all gate on is_training_active(), so an idle reading in this window
    # would let another pipeline race the reserved run for the just-freed VRAM.
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    # Not reserved yet: idle.
    assert backend.is_training_active() is False

    entered = threading.Event()
    release = threading.Event()
    captured = {}

    def _slow_impl(job_id, *, before_spawn = None, **kwargs):
        entered.set()
        release.wait(timeout = 5.0)
        return True

    backend._start_training_impl = _slow_impl

    t = threading.Thread(target = lambda: backend.start_training("job-a"), daemon = True)
    t.start()
    assert entered.wait(timeout = 5.0)
    # Inside the pre-spawn window: reserved, so active even though no proc/progress is set.
    captured["in_window"] = backend.is_training_active()
    release.set()
    t.join(timeout = 5.0)

    assert captured["in_window"] is True
    # Reservation cleared once the start returns; with no live proc it reads idle again.
    assert backend.is_training_active() is False
