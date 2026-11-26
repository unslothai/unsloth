# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CPU MP4 encode of a finished clip runs after _generate_lock is released, and a queued clip hands the job slot
on before it, so the next clip's GPU work overlaps the encode. At most one clip encodes past its slot; Stop, a failed
encode and the per-id progress still reach the clip that is encoding."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import core.inference.video as video_mod  # noqa: E402
import core.inference.video_gallery as gallery_module  # noqa: E402
from core.inference.video import VideoBackend  # noqa: E402
from core.inference.video_families import VIDEO_CANCELLED_MSG  # noqa: E402
from test_video_backend import _load_gguf, fake_runtime  # noqa: E402,F401 -- fixture found by name


def _wait(
    cond,
    timeout = 10.0,
    what = "condition",
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


class _GatedEncoder:
    """A stub _encode_mp4 that records whether the generate lock was held, and blocks until released."""

    def __init__(
        self,
        backend,
        *,
        block = True,
    ):
        self.backend = backend
        self.calls: list[dict] = []
        self.entered = threading.Semaphore(0)
        self.gates: list[threading.Event] = []
        self.block = block
        self.fail: set[int] = set()
        self._mu = threading.Lock()

    def __call__(
        self,
        frames,
        fps,
        audio,
        pipe,
        *,
        audio_sample_rate = None,
    ):
        with self._mu:
            n = len(self.calls)
            gate = threading.Event()
            self.gates.append(gate)
            self.calls.append(
                {
                    "locked": self.backend._generate_lock.locked(),
                    "slot_busy": self.backend._generate_job_active,
                    "pipe": pipe,
                }
            )
        self.entered.release()
        if self.block:
            assert gate.wait(10), "encode gate never opened"
        if n in self.fail:
            raise OSError("disk full while muxing")
        return b"MP4-%d" % n

    def open(self, n):
        _wait(lambda: len(self.gates) > n, what = f"encode {n}")
        self.gates[n].set()


@pytest.fixture
def ltx(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "gallery").mkdir()
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path / "gallery")
    monkeypatch.delenv("UNSLOTH_VIDEO_ENCODE_OUTSIDE_LOCK", raising = False)
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    # The module-level readers (generation_in_flight, encode_account_in_flight) read the singleton.
    monkeypatch.setattr(video_mod, "_backend", backend)
    enc = _GatedEncoder(backend)
    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(enc))
    return backend, enc


def test_the_encode_runs_after_the_generate_lock_is_released(ltx):
    backend, enc = ltx
    enc.block = False
    result = backend.generate(prompt = "a fox", steps = 2)
    assert result["mp4_bytes"] == b"MP4-0"
    assert enc.calls[0]["locked"] is False
    # The encode holds no pipeline: an unload during it still frees the weights.
    assert enc.calls[0]["pipe"] is not backend._state.pipe
    assert backend._active_generate_cancel is None


def test_kill_switch_keeps_the_encode_under_the_lock(ltx, monkeypatch):
    backend, enc = ltx
    enc.block = False
    monkeypatch.setenv("UNSLOTH_VIDEO_ENCODE_OUTSIDE_LOCK", "0")
    assert backend.generate(prompt = "a fox", steps = 2)["mp4_bytes"] == b"MP4-0"
    assert enc.calls[0]["locked"] is True


def test_stop_during_a_direct_encode_discards_the_clip(ltx):
    backend, enc = ltx
    out: dict = {}

    def run():
        try:
            out["result"] = backend.generate(prompt = "a fox", steps = 2)
        except Exception as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target = run)
    t.start()
    assert enc.entered.acquire(timeout = 10)
    assert backend.cancel_generate() is True
    enc.open(0)
    t.join(10)
    assert str(out.get("error")) == VIDEO_CANCELLED_MSG
    assert backend._active_generate_cancel is None


def _outcome(backend, video_id):
    return backend.generate_outcome(video_id, video_mod.current_account_id())


def test_a_queued_clip_hands_the_slot_on_before_its_encode(ltx):
    backend, enc = ltx
    backend.begin_generate(prompt = "first", steps = 2, video_id = "vid-a")
    assert enc.entered.acquire(timeout = 10)
    # Clip A encodes on the CPU: the GPU lock and the job slot are both free, so the queue starts clip B now.
    assert enc.calls[0]["locked"] is False and enc.calls[0]["slot_busy"] is False
    # Still in flight for shutdown and the watchdog while it encodes.
    assert video_mod.generation_in_flight() is True
    progress = _outcome(backend, "vid-a")
    assert progress["active"] is True and progress["phase"] == "export"
    assert backend.generate_progress()["active"] is False
    backend.begin_generate(prompt = "second", steps = 2, video_id = "vid-b")
    # B finishes its GPU work while A still encodes: at most one pending encode, so B waits holding its slot.
    _wait(lambda: backend._gen.get("phase") == "export", what = "B to reach its export")
    time.sleep(0.05)
    assert len(enc.calls) == 1
    assert backend._generate_job_active is True
    with pytest.raises(RuntimeError):
        backend.begin_generate(prompt = "third", steps = 2, video_id = "vid-c")
    enc.open(0)
    assert enc.entered.acquire(timeout = 10)
    _wait(
        lambda: (_outcome(backend, "vid-a") or {}).get("phase") in ("completed", "failed"), what = "A"
    )
    assert _outcome(backend, "vid-a")["phase"] == "completed", _outcome(backend, "vid-a")
    assert _outcome(backend, "vid-a")["active"] is False
    assert _outcome(backend, "vid-b")["phase"] == "export"
    enc.open(1)
    _wait(lambda: (_outcome(backend, "vid-b") or {}).get("phase") == "completed", what = "B")
    # Both clips saved, each its own bytes; the progress record ends on the last clip.
    for vid, data in (("vid-a", b"MP4-0"), ("vid-b", b"MP4-1")):
        record = _outcome(backend, vid)["video"]
        assert gallery_module.get_record(vid) is not None
        assert record["id"] == vid
        path = gallery_module.owned_video_path(vid)
        assert path is not None and Path(path).read_bytes() == data
    gen = backend.generate_progress()
    assert gen["phase"] == "completed" and gen["video_id"] == "vid-b"
    assert backend._encode_tail is None and not video_mod.generation_in_flight()


def test_stop_by_id_reaches_a_clip_encoding_past_its_slot(ltx):
    backend, enc = ltx
    backend.begin_generate(prompt = "first", steps = 2, video_id = "vid-a")
    assert enc.entered.acquire(timeout = 10)
    backend.begin_generate(prompt = "second", steps = 2, video_id = "vid-b")
    assert backend.cancel_generate(expected_video_id = "vid-a") is True
    enc.open(0)
    _wait(lambda: (_outcome(backend, "vid-a") or {}).get("phase") == "failed", what = "A cancelled")
    assert _outcome(backend, "vid-a")["error"] == VIDEO_CANCELLED_MSG
    assert gallery_module.owned_video_path("vid-a") is None
    # B was not touched by A's Stop.
    assert enc.entered.acquire(timeout = 10)
    enc.open(1)
    _wait(lambda: (_outcome(backend, "vid-b") or {}).get("phase") == "completed", what = "B")


def test_another_account_cannot_stop_or_read_the_encoding_clip(ltx):
    backend, enc = ltx
    backend.begin_generate(prompt = "first", steps = 2, video_id = "vid-a")
    assert enc.entered.acquire(timeout = 10)
    assert (
        backend.cancel_generate(expected_video_id = "vid-a", expected_account = "someone-else") is False
    )
    assert backend.generate_outcome("vid-a", "someone-else") is None
    assert video_mod.encode_account_in_flight() == video_mod.current_account_id()
    enc.open(0)
    _wait(lambda: (_outcome(backend, "vid-a") or {}).get("phase") == "completed", what = "A")
    assert video_mod.encode_account_in_flight() is None


def test_a_failed_encode_past_the_slot_fails_its_clip_and_frees_the_tail(ltx):
    backend, enc = ltx
    enc.fail.add(0)
    backend.begin_generate(prompt = "first", steps = 2, video_id = "vid-a")
    assert enc.entered.acquire(timeout = 10)
    enc.open(0)
    _wait(lambda: (_outcome(backend, "vid-a") or {}).get("phase") == "failed", what = "A failed")
    assert "disk full" in _outcome(backend, "vid-a")["error"] or _outcome(backend, "vid-a")["error"]
    assert backend._encode_tail is None
    # The next clip still runs and hands off normally.
    backend.begin_generate(prompt = "second", steps = 2, video_id = "vid-b")
    enc.open(1)
    _wait(lambda: (_outcome(backend, "vid-b") or {}).get("phase") == "completed", what = "B")


def test_an_older_tail_never_takes_a_running_successors_record(ltx):
    backend, enc = ltx
    backend.begin_generate(prompt = "first", steps = 2, video_id = "vid-a")
    assert enc.entered.acquire(timeout = 10)
    backend.begin_generate(prompt = "second", steps = 2, video_id = "vid-b")
    _wait(lambda: backend._gen.get("phase") == "export", what = "B export")
    enc.open(0)
    _wait(lambda: (_outcome(backend, "vid-a") or {}).get("phase") == "completed", what = "A")
    # B owns the record (now encoding past its own slot, or still holding it); A's end never replaced it.
    assert backend.generate_progress().get("video_id") != "vid-a"
    enc.open(1)
    _wait(lambda: (_outcome(backend, "vid-b") or {}).get("phase") == "completed", what = "B")
