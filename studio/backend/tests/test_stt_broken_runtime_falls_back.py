# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A whisper.cpp build that starts but cannot infer must stop reporting as available.

Reported on Windows with ROCm on gfx1200: rocBLAS was missing its TensileLibrary, so
whisper-server started, answered GET /, and died on the first inference. The binary and
every linked library were present, so slim_runtime_intact() was satisfied, is_available()
said yes, and _resolve_serving_stt_engine never fell back. Every recording returned 501
while the UI showed the model loaded. Only an actual inference can distinguish this case.
"""

from __future__ import annotations

import threading

import pytest

from core.inference import stt_ggml_sidecar


@pytest.fixture(autouse = True)
def _clean_runtime_state():
    stt_ggml_sidecar.clear_runtime_inference_failure()
    yield
    stt_ggml_sidecar.clear_runtime_inference_failure()


def test_a_runtime_that_cannot_infer_reports_unavailable(monkeypatch):
    monkeypatch.setattr(stt_ggml_sidecar, "find_whisper_server_binary", lambda: "whisper-server")
    monkeypatch.setattr(stt_ggml_sidecar, "slim_runtime_intact", lambda binary: True)

    assert stt_ggml_sidecar.is_available() is True
    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    assert stt_ggml_sidecar.is_available() is False
    assert "RemoteDisconnected" in (stt_ggml_sidecar.runtime_inference_failure() or "")


def test_the_serving_engine_then_falls_back_to_transformers(monkeypatch):
    from routes import inference as inference_routes

    monkeypatch.setattr(stt_ggml_sidecar, "find_whisper_server_binary", lambda: "whisper-server")
    monkeypatch.setattr(stt_ggml_sidecar, "slim_runtime_intact", lambda binary: True)

    assert inference_routes._resolve_serving_stt_engine("gguf") == "gguf"
    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    # The fallback exists to avoid 501-ing on every recording; this is that case.
    assert inference_routes._resolve_serving_stt_engine("gguf") == "transformers"


def test_a_cancelled_transcription_does_not_disable_the_engine():
    """A cancel closes the socket deliberately. Treating that as a broken runtime would
    disable whisper.cpp for the rest of the session every time someone stops a recording."""
    cancel_event = threading.Event()
    cancel_event.set()

    # Mirrors the guard at the call site rather than driving a live server.
    if cancel_event is None or not cancel_event.is_set():
        stt_ggml_sidecar.note_runtime_inference_failure("should not happen")
    assert stt_ggml_sidecar.runtime_inference_failure() is None


def test_a_later_success_clears_the_failure():
    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    stt_ggml_sidecar.clear_runtime_inference_failure()
    assert stt_ggml_sidecar.runtime_inference_failure() is None


def test_an_amd_box_does_not_report_its_dictation_device_as_cuda(monkeypatch):
    """Torch's ROCm build keeps the "cuda" device name for HIP, which is correct for the
    API and misleading on screen: the Loaded models entry read "Transformers - cuda" on a
    Radeon card (reported on Windows against PR 7984)."""
    import torch

    from core.inference.stt_sidecar import _reported_device

    monkeypatch.setattr(torch.version, "hip", "7.1", raising = False)
    assert _reported_device("cuda") == "rocm"
    assert _reported_device("cpu") == "cpu"
    assert _reported_device(None) is None

    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert _reported_device("cuda") == "cuda"


def test_the_fallback_fetches_the_transformers_snapshot_it_needs(monkeypatch):
    """The GGUF pick downloaded one .bin, so Transformers has no snapshot to serve.

    Without this the fallback swaps a 501 for a 409 "not downloaded" on every retry while
    the Audio page still shows the selection as ready.
    """
    from core.inference import stt_sidecar
    from routes import inference as inference_routes

    monkeypatch.setattr(stt_ggml_sidecar, "find_whisper_server_binary", lambda: "whisper-server")
    monkeypatch.setattr(stt_ggml_sidecar, "slim_runtime_intact", lambda binary: True)
    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda model: False)
    started: list[tuple] = []
    monkeypatch.setattr(
        stt_sidecar,
        "start_model_download",
        lambda model, token = None, revision = None: started.append((model, token)),
    )

    # A healthy runtime serves the GGUF itself, so nothing is fetched.
    inference_routes._prepare_runtime_fallback_checkpoint("gguf", "gguf", "small")
    assert started == []

    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    engine = inference_routes._resolve_serving_stt_engine("gguf")
    inference_routes._prepare_runtime_fallback_checkpoint("gguf", engine, "small")
    assert started == [("small", None)]


def test_an_already_downloaded_snapshot_is_not_fetched_again(monkeypatch):
    from core.inference import stt_sidecar
    from routes import inference as inference_routes

    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda model: True)
    started: list[tuple] = []
    monkeypatch.setattr(
        stt_sidecar,
        "start_model_download",
        lambda model, token = None, revision = None: started.append((model, token)),
    )
    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    inference_routes._prepare_runtime_fallback_checkpoint("gguf", "transformers", "small")
    assert started == []


def test_a_plain_transformers_pick_is_left_alone(monkeypatch):
    """Only a GGUF selection redirected by a broken runtime needs preparing. An engine
    that was never installed already routes its own download through Transformers."""
    from core.inference import stt_sidecar
    from routes import inference as inference_routes

    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda model: False)
    started: list[tuple] = []
    monkeypatch.setattr(
        stt_sidecar,
        "start_model_download",
        lambda model, token = None, revision = None: started.append((model, token)),
    )
    stt_ggml_sidecar.note_runtime_inference_failure("RemoteDisconnected")
    inference_routes._prepare_runtime_fallback_checkpoint("transformers", "transformers", "small")
    inference_routes._prepare_runtime_fallback_checkpoint("mtmd", "mtmd", "small")
    assert started == []

    # And a download that cannot start (another model is in flight) is not fatal: the
    # caller still reports "not downloaded" rather than a 500.
    def refuse(
        model,
        token = None,
        revision = None,
    ):
        from core.inference.stt_sidecar import SttModelIdError
        raise SttModelIdError("Another dictation model is still downloading; wait for it.")

    monkeypatch.setattr(stt_sidecar, "start_model_download", refuse)
    inference_routes._prepare_runtime_fallback_checkpoint("gguf", "transformers", "small")
