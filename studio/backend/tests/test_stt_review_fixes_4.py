# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fourth review pass: status reads that must never block, and the PID registry.

The STT status route reads these sidecars on the event loop, so anything it
touches has to answer while a lifecycle operation holds the sidecar's lock.
"""

import subprocess
import threading
import time

import pytest

from core.inference import stt_ggml_sidecar as ggml_mod
from core.inference import stt_mtmd_sidecar as mtmd_mod
from core.inference.stt_mtmd_sidecar import MtmdSttSidecar


class _SlowlyDyingProcess:
    """A child whose reap blocks, like a llama-server that is slow to exit."""

    def __init__(self, release: threading.Event):
        self.pid = 4242
        self._release = release
        self._returncode = None

    def poll(self):
        return self._returncode

    def terminate(self):
        pass

    def wait(self, timeout = None):
        # Blocks until the test lets go, standing in for a server that takes
        # seconds to release its port and VRAM.
        if not self._release.wait(timeout = timeout):
            raise subprocess.TimeoutExpired("llama-server", timeout)
        self._returncode = -15
        return self._returncode

    def kill(self):
        self._returncode = -9


def _resident(sidecar: MtmdSttSidecar, process) -> None:
    """Publish `process` as the loaded server, as a finished load would."""
    sidecar._process = process
    sidecar._port = 12345
    sidecar._model_id = "qwen3-asr-0.6b"


def test_status_reads_do_not_block_behind_a_reap(monkeypatch):
    """loaded_model/device/is_loading answer while unload() reaps under _lock.

    The status route reads these on the event loop, and training admission
    reads them too, so neither can wait on a dying server.
    """
    monkeypatch.setattr(mtmd_mod, "forget_pid", lambda pid: None)
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    release = threading.Event()
    _resident(sidecar, _SlowlyDyingProcess(release))

    unloading = threading.Thread(target = sidecar.unload, daemon = True)
    unloading.start()
    try:
        # Give unload() time to take _lock and block inside _reap().
        time.sleep(0.2)
        answered = threading.Event()

        def read_status():
            sidecar.loaded_model
            sidecar.device
            sidecar.is_loading()
            answered.set()

        threading.Thread(target = read_status, daemon = True).start()
        assert answered.wait(
            timeout = 5
        ), "a status read blocked behind the reap; the event loop would stall with it"
    finally:
        release.set()
        unloading.join(timeout = 10)


def test_a_reaping_server_stays_visible_to_training_admission(monkeypatch):
    """A dying llama-server still holds VRAM, so it must still read as resident.

    _release_locked() clears the fields, so clearing them before the reap would
    let summarize_resident_stt() report nothing while the process is alive, and
    training would start into memory it does not have.
    """
    monkeypatch.setattr(mtmd_mod, "forget_pid", lambda pid: None)
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    release = threading.Event()
    _resident(sidecar, _SlowlyDyingProcess(release))

    seen = {}

    def unload():
        sidecar.unload()

    unloading = threading.Thread(target = unload, daemon = True)
    unloading.start()
    try:
        time.sleep(0.2)  # inside the reap
        seen["model"] = sidecar.loaded_model
        seen["device"] = sidecar.device
    finally:
        release.set()
        unloading.join(timeout = 10)

    assert (
        seen["model"] == "qwen3-asr-0.6b"
    ), "the reaping server read as gone; training admission would miss its VRAM"
    assert seen["device"] == "llama.cpp"
    # Once the reap is done the fields are cleared, so it reads as gone.
    assert sidecar.loaded_model is None
    assert sidecar._process is None


def test_status_reads_do_not_block_during_a_llama_cpp_update():
    """update_maintenance() holds _lock for the whole install; polls continue."""
    sidecar = MtmdSttSidecar(keep_alive_seconds = 0)
    with sidecar.update_maintenance():
        answered = threading.Event()

        def read_status():
            sidecar.loaded_model
            sidecar.is_loading()
            answered.set()

        threading.Thread(target = read_status, daemon = True).start()
        assert answered.wait(
            timeout = 5
        ), "a status read blocked for the length of the llama.cpp install"


def test_ggml_download_drops_its_adopted_pid(monkeypatch):
    """spawn_download() adopts a PID, so the ggml path must reap it properly.

    A PID left adopted can be reused, and terminate_all would then signal
    whatever inherited it.
    """
    forgotten = []

    class _Finished:
        pid = 7777
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(ggml_mod, "_cached_model_path", lambda model_id: None)
    import core.inference.stt_download_worker as worker_mod

    monkeypatch.setattr(worker_mod, "spawn_download", lambda *a, **k: _Finished())
    monkeypatch.setattr(
        worker_mod, "reap_download", lambda process: forgotten.append(process.pid) or b""
    )
    state = ggml_mod._GgmlDownloadState()
    state._run("tiny", None)

    assert forgotten == [7777], "the GGUF download never dropped its adopted PID"


def test_wait_for_server_sleeps_on_a_non_200_success(monkeypatch):
    """A 2xx that is not 200 must not spin the readiness loop with no delay."""

    class _Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    slept = []
    monkeypatch.setattr(mtmd_mod.urllib.request, "urlopen", lambda *a, **k: _Response())
    monkeypatch.setattr(mtmd_mod.time, "sleep", slept.append)
    monkeypatch.setattr(mtmd_mod, "_SERVER_START_TIMEOUT_SECONDS", 0.5)

    class _Alive:
        def poll(self):
            return None

    assert MtmdSttSidecar._wait_for_server(_Alive(), 1234) is False
    assert slept, "the readiness loop spun without sleeping on a non-200 response"


def test_download_probe_is_memoised_then_dropped_when_a_download_ends(monkeypatch):
    """The status poll asks four times a second; the cache walk runs once."""
    mtmd_mod._forget_downloaded_probe()
    calls = []

    def fake_paths(model_id):
        calls.append(model_id)
        return ("m.gguf", "p.gguf")

    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", fake_paths)
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True
    assert len(calls) == 1, "the memo did not spare the second probe"

    # A finished download changes what is on disk, so the answer is dropped
    # rather than left to expire: the panel settles on its next poll.
    mtmd_mod._forget_downloaded_probe("qwen3-asr-0.6b")
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True
    assert len(calls) == 2

    mtmd_mod._forget_downloaded_probe()


def test_download_probe_expires(monkeypatch):
    """A cache emptied outside Studio is noticed without a restart."""
    mtmd_mod._forget_downloaded_probe()
    monkeypatch.setattr(mtmd_mod, "_DOWNLOADED_PROBE_TTL_SECONDS", 0.0)
    calls = []

    def fake_paths(model_id):
        calls.append(model_id)
        return None

    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", fake_paths)
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is False
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is False
    assert len(calls) == 2, "an expired entry was still served"

    mtmd_mod._forget_downloaded_probe()


def test_an_invalidation_mid_probe_discards_the_stale_answer(monkeypatch):
    """A download finishing under a probe must not be undone by that probe.

    The probe runs outside the lock, so it can read an incomplete snapshot,
    then write False back after the download's invalidation already cleared the
    entry. The model would then read as missing for a whole TTL.
    """
    mtmd_mod._forget_downloaded_probe()

    def probe_then_invalidate(model_id):
        # Stands in for the download completing while this probe is in flight.
        mtmd_mod._forget_downloaded_probe(model_id)
        return None

    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", probe_then_invalidate)
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is False
    with mtmd_mod._downloaded_probe_lock:
        assert (
            "qwen3-asr-0.6b" not in mtmd_mod._downloaded_probe
        ), "a stale answer was written back over the invalidation"

    # The next poll sees the finished download rather than waiting out the TTL.
    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", lambda mid: ("m.gguf", "p.gguf"))
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True
    mtmd_mod._forget_downloaded_probe()


def test_the_entry_is_timestamped_after_the_probe(monkeypatch):
    """A slow cache must not store an entry that is already near expiry."""
    mtmd_mod._forget_downloaded_probe()
    clock = {"t": 1000.0}
    monkeypatch.setattr(mtmd_mod.time, "monotonic", lambda: clock["t"])

    def slow_probe(model_id):
        clock["t"] += 1.5  # probe takes most of the TTL
        return ("m.gguf", "p.gguf")

    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", slow_probe)
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True

    with mtmd_mod._downloaded_probe_lock:
        stored_at = mtmd_mod._downloaded_probe["qwen3-asr-0.6b"][0]
    assert stored_at == 1001.5, "the entry was timestamped before the probe ran"

    # The panel's next poll, 750ms later, is still served from the memo.
    clock["t"] += 0.75
    called = []
    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", lambda mid: called.append(mid))
    assert mtmd_mod.is_model_downloaded("qwen3-asr-0.6b") is True
    assert called == [], "the entry expired early and the probe ran again"
    mtmd_mod._forget_downloaded_probe()


def test_unknown_model_is_never_memoised(monkeypatch):
    called = []
    monkeypatch.setattr(mtmd_mod, "_cached_model_paths", lambda model_id: called.append(model_id))
    assert mtmd_mod.is_model_downloaded("not-a-model") is False
    assert called == []


def test_download_status_reports_progress_without_holding_the_lock():
    """_cache_bytes() walks the cache; a cancel must not queue behind it."""
    state = mtmd_mod._MtmdDownloadState()
    observed = []

    def slow_cache_bytes(model_id = None):
        # The lock is free while this runs, or the assert below deadlocks.
        observed.append(state._lock.acquire(blocking = False))
        if observed[-1]:
            state._lock.release()
        return 1

    state._cache_bytes = slow_cache_bytes
    state._model_id = "qwen3-asr-0.6b"
    state._thread = threading.Thread(target = lambda: time.sleep(0.5), daemon = True)
    state._thread.start()
    try:
        status = state.status()
    finally:
        state._thread.join(timeout = 5)

    assert status["bytes_done"] == 1
    assert observed == [True], "progress was computed while holding the download lock"


@pytest.mark.parametrize("model_id", sorted(mtmd_mod.MTMD_STT_MODELS))
def test_catalogue_probes_stay_answerable(model_id):
    """is_model_downloaded() never raises for a curated id, cache or no cache."""
    mtmd_mod._forget_downloaded_probe()
    assert isinstance(mtmd_mod.is_model_downloaded(model_id), bool)
    mtmd_mod._forget_downloaded_probe()
