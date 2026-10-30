# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp is retired: its engine aliases are served by another engine.

Old frontends, MCP and API callers keep sending ``engine="gguf"``. With PyTorch the
same Whisper key runs on Transformers; without it the request runs llama.cpp's
Qwen3-ASR, then audio.cpp's, and only then reaches Transformers' "needs PyTorch" 501.
whisper-server is installed and healthy in every test here, and every path into it
raises, so nothing can quietly land on it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import routes.inference as ri
from core.inference import stt_audiocpp_sidecar as audiocpp_module
from core.inference import stt_capabilities
from core.inference import stt_ggml_sidecar as ggml_module
from core.inference import stt_mtmd_sidecar as mtmd_module
from core.inference import stt_sidecar as stt_module
from core.inference.audio_cpp_models import DEFAULT_AUDIO_CPP_STT_MODEL
from core.inference.stt_ggml_sidecar import GgmlSttSidecar
from core.inference.stt_mtmd_sidecar import DEFAULT_MTMD_STT_MODEL
from core.inference.stt_sidecar import SttUnavailableError
from models.inference import SttLoadRequest, TranscribeSourceRequest

ALIASES = ("gguf", "ggml", "whisper_cpp", "whisper.cpp")


class _Sidecar:
    loaded_model = None
    device = None

    def __init__(self, engine, transcribed):
        self.engine = engine
        self.transcribed = transcribed

    def transcribe(self, _raw, model, _language, _fast):
        self.transcribed.append((model, self.engine))
        return {"text": "hello"}

    def cancel_transcription(self, _cancel_event):
        return True


class _Host:
    def __init__(self, monkeypatch):
        self.loads: list = []
        self.transcribed: list = []
        self.downloads: list = []
        self.cancels: list = []
        self.unloads: list = []
        self._monkeypatch = monkeypatch
        self.pin(torch = True, audiocpp = True, mtmd = True)

    def pin(self, *, torch, audiocpp, mtmd):
        self._monkeypatch.setattr(stt_module, "is_available", lambda: torch)
        self._monkeypatch.setattr(audiocpp_module, "is_available", lambda: audiocpp)
        self._monkeypatch.setattr(mtmd_module, "is_available", lambda: mtmd)


def _whisper_cpp_must_not_run(*_args, **_kwargs):
    raise AssertionError("whisper.cpp is retired and must never be reached")


@pytest.fixture
def host(monkeypatch):
    state = _Host(monkeypatch)

    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    monkeypatch.setattr(GgmlSttSidecar, "load", _whisper_cpp_must_not_run)
    monkeypatch.setattr(GgmlSttSidecar, "transcribe", _whisper_cpp_must_not_run)
    monkeypatch.setattr(ggml_module, "start_model_download", _whisper_cpp_must_not_run)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "_RETIRED_ALIAS_SERVED", {"load": set(), "download": set()})
    monkeypatch.setattr(ri, "_DIRECT_STT_NOTES", {"load": {}, "download": {}})
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *_a: None)

    def load(
        model,
        engine,
        _cancel_event = None,
        device = None,
        **_kwargs,
    ):
        if engine == "gguf":
            _whisper_cpp_must_not_run()
        if engine == "transformers" and not stt_module.is_available():
            raise SttUnavailableError(
                "Speech-to-text needs PyTorch, Transformers, and PyAV. "
                "Run `unsloth studio update` to install them."
            )
        state.loads.append((model, engine))

    def unload(
        engines,
        expected_model = None,
        wait = True,
    ):
        state.unloads.append((engines, expected_model))
        return []

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load, unload))
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda engine: _Sidecar(engine, state.transcribed))
    monkeypatch.setattr(ri, "_stt_resident_model", lambda _engine: None)

    def start_download(
        module,
        engine,
        model,
        _hf_token,
        _revision = None,
    ):
        state.downloads.append((module.__name__.rsplit(".", 1)[-1], engine, model))
        return "download-1"

    def cancel_download(module, engine, model, _download_id):
        state.cancels.append((module.__name__.rsplit(".", 1)[-1], engine, model))
        return {"downloading": False, "cancelled": True}

    monkeypatch.setattr(ri, "_start_account_stt_download", start_download)
    monkeypatch.setattr(ri, "_cancel_account_stt_download", cancel_download)
    monkeypatch.setattr(
        stt_module, "validate_remote_model", lambda model, hf_token = None: {"model": model}
    )
    for module in (stt_module, audiocpp_module, mtmd_module):
        monkeypatch.setattr(module, "download_status", lambda: {"downloading": True})
    return state


def _load(model, engine):
    return asyncio.run(ri.stt_load(SttLoadRequest(model = model, engine = engine), None, "tester"))


@pytest.mark.parametrize("alias", ALIASES)
def test_an_alias_loads_the_same_key_on_transformers(host, alias):
    _load("small", alias)
    assert host.loads == [("small", "transformers")]


@pytest.mark.parametrize("alias", ALIASES)
def test_an_alias_downloads_through_transformers(host, alias):
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = alias), current_subject = "tester", hf_token = None
        )
    )
    assert host.downloads == [("stt_sidecar", "transformers", "small")]


def test_an_alias_cancels_the_transformers_download(host):
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert host.cancels == [("stt_sidecar", "transformers", "small")]


def test_an_alias_cancel_never_stops_a_download_it_did_not_start(host):
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "transformers"),
            current_subject = "tester",
            hf_token = None,
        )
    )
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert host.cancels == []


def test_an_alias_that_joins_a_direct_download_does_not_claim_it(host, monkeypatch):
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "transformers"),
            current_subject = "tester",
            hf_token = None,
        )
    )
    # The direct download is still running, so the alias request gets the same id back.
    monkeypatch.setattr(
        stt_module, "download_status", lambda: {"downloading": True, "download_id": "download-1"}
    )
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert host.cancels == []


def test_an_alias_overtaken_by_a_direct_download_does_not_claim_it(host, monkeypatch):
    def validate_while_a_direct_download_starts(model, hf_token = None):
        # The alias saw no download; a direct request starts it while the alias validates.
        monkeypatch.setattr(
            stt_module, "validate_remote_model", lambda m, hf_token = None: {"model": m}
        )
        asyncio.run(
            ri.stt_download(
                SttLoadRequest(model = "small", engine = "transformers"),
                current_subject = "tester",
                hf_token = None,
            )
        )
        return {"model": model}

    monkeypatch.setattr(stt_module, "download_status", lambda: {"downloading": False})
    monkeypatch.setattr(
        stt_module, "validate_remote_model", validate_while_a_direct_download_starts
    )
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert host.cancels == []


def _direct_qwen_load_lands_first(host, monkeypatch):
    """The alias's load waits on the registry while a direct Qwen3-ASR load on llama.cpp finishes."""
    load, unload = ri._stt_lifecycle()
    overtaken = []

    def load_after_a_direct_one(model, engine, *args, **kwargs):
        if engine == "mtmd" and not overtaken:
            overtaken.append(True)
            monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load, unload))
            _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
            monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load_after_a_direct_one, unload))
        return load(model, engine, *args, **kwargs)

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load_after_a_direct_one, unload))
    host.pin(torch = False, audiocpp = True, mtmd = True)
    return overtaken


def test_an_alias_load_overtaken_by_a_direct_load_does_not_claim_it(host, monkeypatch):
    overtaken = _direct_qwen_load_lands_first(host, monkeypatch)
    _load("small", "gguf")
    assert overtaken
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_an_alias_transcription_overtaken_by_a_direct_load_does_not_claim_it(host, monkeypatch):
    overtaken = _direct_qwen_load_lands_first(host, monkeypatch)
    asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "small", None, False, "gguf"))
    assert overtaken
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_an_alias_load_nothing_overtook_still_frees_its_model(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert [list(engines) for engines, _ in host.unloads] == [["mtmd"]]


def test_an_alias_cancel_during_a_direct_download_start_leaves_it_running(host, monkeypatch):
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    start = ri._start_account_stt_download

    def start_then_a_legacy_cancel_arrives(*args, **kwargs):
        download_id = start(*args, **kwargs)
        # The direct transfer is cancellable before its request records the handoff.
        asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
        return download_id

    monkeypatch.setattr(ri, "_start_account_stt_download", start_then_a_legacy_cancel_arrives)
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "transformers"),
            current_subject = "tester",
            hf_token = None,
        )
    )
    assert host.cancels == []


def test_an_alias_unload_during_a_direct_load_leaves_it_resident(host, monkeypatch):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    load, unload = ri._stt_lifecycle()

    def load_then_a_legacy_unload_arrives(model, engine, *args, **kwargs):
        load(model, engine, *args, **kwargs)
        asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load_then_a_legacy_unload_arrives, unload))
    host.unloads.clear()
    _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
    assert host.unloads == []


def test_a_failed_direct_download_hands_the_alias_its_download_back(host, monkeypatch):
    from core.inference.stt_sidecar import SttModelCompatibilityError

    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )

    def not_whisper(model, hf_token = None):
        raise SttModelCompatibilityError(f"{model} is not a Whisper checkpoint")

    monkeypatch.setattr(stt_module, "validate_remote_model", not_whisper)
    with pytest.raises(HTTPException):
        asyncio.run(
            ri.stt_download(
                SttLoadRequest(model = "someone/not-whisper", engine = "transformers"),
                current_subject = "tester",
                hf_token = None,
            )
        )
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert host.cancels == [("stt_sidecar", "transformers", "small")]


def test_a_failed_direct_load_hands_the_alias_its_model_back(host, monkeypatch):
    from core.inference.stt_sidecar import SttModelNotDownloadedError

    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    load, unload = ri._stt_lifecycle()

    def not_downloaded(*_args, **_kwargs):
        raise SttModelNotDownloadedError("not downloaded yet")

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (not_downloaded, unload))
    with pytest.raises(HTTPException):
        _load("other/asr-model", "mtmd")
    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load, unload))
    host.unloads.clear()
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert [list(engines) for engines, _ in host.unloads] == [["mtmd"]]


def test_an_alias_unloads_the_transformers_model(host):
    _load("small", "gguf")
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == [(["transformers"], "small")]


def test_an_alias_unload_never_frees_a_model_it_did_not_load(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    # Qwen3-ASR loaded directly on llama.cpp, where an alias request would also run now.
    _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_an_alias_unload_still_frees_audio_cpp_after_llama_cpp_comes_back(host):
    host.pin(torch = False, audiocpp = True, mtmd = False)
    _load("small", "gguf")
    assert host.loads == [(DEFAULT_AUDIO_CPP_STT_MODEL, "audiocpp")]
    # llama.cpp repaired mid-session: a fresh resolve would now pick it.
    host.pin(torch = False, audiocpp = True, mtmd = True)
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert (["audiocpp"], DEFAULT_AUDIO_CPP_STT_MODEL) in host.unloads


def test_an_alias_unload_still_frees_llama_cpp_after_pytorch_is_installed(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    # PyTorch installed while the server runs: a fresh resolve would now pick Transformers.
    host.pin(torch = True, audiocpp = True, mtmd = True)
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == [(["mtmd"], DEFAULT_MTMD_STT_MODEL)]


def test_a_direct_load_takes_the_engine_back_from_the_alias(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "small", None, False, "gguf"))
    host.pin(torch = True, audiocpp = True, mtmd = True)
    # The alias's model is released, then Qwen3-ASR is loaded on llama.cpp directly.
    asyncio.run(ri.stt_unload(current_subject = "tester"))
    _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
    host.unloads.clear()
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_a_direct_load_alone_also_takes_the_engine_back(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    host.pin(torch = True, audiocpp = True, mtmd = True)
    _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_an_alias_load_that_reuses_a_direct_model_does_not_claim_it(host, monkeypatch):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load(DEFAULT_MTMD_STT_MODEL, "mtmd")
    resident = ri._stt_resolved_model_id(DEFAULT_MTMD_STT_MODEL, "mtmd")
    monkeypatch.setattr(_Sidecar, "loaded_model", resident)
    # The alias maps to that same Qwen3-ASR on llama.cpp and only reuses it.
    _load("small", "gguf")
    asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "small", None, False, "gguf"))
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    assert host.unloads == []


def test_a_failed_alias_load_does_not_mark_its_engine(host, monkeypatch):
    from core.inference.stt_sidecar import SttModelNotDownloadedError

    _, unload = ri._stt_lifecycle()

    def not_downloaded(*_args, **_kwargs):
        raise SttModelNotDownloadedError("not downloaded yet")

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (not_downloaded, unload))
    host.pin(torch = False, audiocpp = True, mtmd = True)
    with pytest.raises(HTTPException):
        _load("small", "gguf")
    host.pin(torch = True, audiocpp = True, mtmd = True)
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    # Nothing ran on llama.cpp, so a Qwen3-ASR loaded there directly is not the alias's to free.
    assert host.unloads == []


def test_an_alias_cancel_still_stops_an_audio_cpp_download_after_llama_cpp_comes_back(host):
    host.pin(torch = False, audiocpp = True, mtmd = False)
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    assert host.downloads == [("stt_audiocpp_sidecar", "audiocpp", DEFAULT_AUDIO_CPP_STT_MODEL)]
    # llama.cpp repaired mid-download: a fresh resolve would now pick it.
    host.pin(torch = False, audiocpp = True, mtmd = True)
    asyncio.run(ri.stt_download_cancel(SttLoadRequest(model = "small", engine = "gguf"), "tester"))
    assert ("stt_audiocpp_sidecar", "audiocpp", DEFAULT_AUDIO_CPP_STT_MODEL) in host.cancels


def _capabilities(model, engine, monkeypatch):
    seen = []

    def capabilities_for(model, engine):
        seen.append((model, engine))
        return {}

    monkeypatch.setattr(stt_capabilities, "capabilities_for", capabilities_for)
    asyncio.run(ri.stt_model_capabilities(model = model, engine = engine, current_subject = "tester"))
    return seen


def test_alias_capabilities_describe_the_engine_that_would_serve_it(host, monkeypatch):
    assert _capabilities("small", "gguf", monkeypatch) == [("small", "transformers")]
    host.pin(torch = False, audiocpp = True, mtmd = True)
    assert _capabilities("small", "whisper.cpp", monkeypatch) == [(DEFAULT_MTMD_STT_MODEL, "mtmd")]


def test_an_alias_transcribes_on_transformers(host):
    response = asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "small", None, False, "gguf"))
    assert json.loads(response.body)["text"] == "hello"
    assert host.loads == [("small", "transformers")]
    assert host.transcribed == [("small", "transformers")]


def test_a_source_transcription_asks_transformers_for_capabilities(host, monkeypatch):
    seen = _stop_after_capabilities(monkeypatch)
    _transcribe_source("small", "gguf")
    assert seen == [("small", "transformers")]


def _stop_after_capabilities(monkeypatch):
    from core.inference import audio_inputs

    seen = []

    def capabilities_for(model, engine):
        seen.append((model, engine))
        return {"family": None, "timestamps": "unsupported", "speakers": False}

    def resolve_source(_source):
        raise audio_inputs.AudioInputError(404, "stop here")

    monkeypatch.setattr(stt_capabilities, "capabilities_for", capabilities_for)
    monkeypatch.setattr(audio_inputs, "resolve_source", resolve_source)
    return seen


def _transcribe_source(model, engine):
    body = TranscribeSourceRequest(source = {"input_id": "clip-1"}, model = model, engine = engine)
    with pytest.raises(HTTPException):
        asyncio.run(ri.transcribe_audio_source(body, "tester"))


def test_without_pytorch_an_alias_runs_llama_cpp_qwen3_asr(host, monkeypatch):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    _load("small", "gguf")
    asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "large-v3", None, False, "gguf"))
    asyncio.run(
        ri.stt_download(
            SttLoadRequest(model = "small", engine = "gguf"), current_subject = "tester", hf_token = None
        )
    )
    asyncio.run(ri.stt_unload(engine = "gguf", model = "small", current_subject = "tester"))
    seen = _stop_after_capabilities(monkeypatch)
    _transcribe_source("small", "gguf")

    assert host.loads == [(DEFAULT_MTMD_STT_MODEL, "mtmd"), (DEFAULT_MTMD_STT_MODEL, "mtmd")]
    assert host.transcribed == [(DEFAULT_MTMD_STT_MODEL, "mtmd")]
    assert host.downloads == [("stt_mtmd_sidecar", "mtmd", DEFAULT_MTMD_STT_MODEL)]
    # Only the engine the alias ran on is freed, scoped to the model it was mapped to.
    assert host.unloads == [(["mtmd"], DEFAULT_MTMD_STT_MODEL)]
    assert seen == [(DEFAULT_MTMD_STT_MODEL, "mtmd")]
    # Managed accounts are checked against the repo the default ASR lives in.
    assert ri._stt_repo_reference("small", "gguf") == "unslothai/Qwen3-ASR-0.6B-GGUF"


def test_without_pytorch_or_llama_cpp_an_alias_runs_audio_cpp_qwen3_asr(host):
    host.pin(torch = False, audiocpp = True, mtmd = False)
    _load("small", "gguf")
    asyncio.run(ri._transcribe_audio_bytes(b"RIFFfake", "small", None, False, "gguf"))

    assert host.loads == [
        (DEFAULT_AUDIO_CPP_STT_MODEL, "audiocpp"),
        (DEFAULT_AUDIO_CPP_STT_MODEL, "audiocpp"),
    ]
    assert host.transcribed == [(DEFAULT_AUDIO_CPP_STT_MODEL, "audiocpp")]
    assert ri._stt_repo_reference("small", "gguf") == "audio-cpp/audio.cpp-gguf"


def test_with_no_engine_at_all_an_alias_reports_the_pytorch_501(host):
    host.pin(torch = False, audiocpp = False, mtmd = False)
    with pytest.raises(HTTPException) as raised:
        _load("small", "gguf")
    assert raised.value.status_code == 501
    assert "needs PyTorch" in raised.value.detail
    assert host.loads == []


def test_other_engines_are_unchanged(host):
    host.pin(torch = False, audiocpp = True, mtmd = True)
    # Transformers without PyTorch still answers 501 rather than rerouting.
    with pytest.raises(HTTPException) as raised:
        _load("small", "transformers")
    assert raised.value.status_code == 501
    _load("qwen3-asr-1.7b", "mtmd")
    _load("audio-cpp/audio.cpp-gguf/Qwen3-ASR-1.7B-GGUF", "audiocpp")
    assert host.loads == [
        ("qwen3-asr-1.7b", "mtmd"),
        ("audio-cpp/audio.cpp-gguf/Qwen3-ASR-1.7B-GGUF", "audiocpp"),
    ]


def test_an_unknown_engine_no_longer_suggests_gguf():
    with pytest.raises(HTTPException) as raised:
        ri._resolve_stt_engine("no-such-engine")
    assert raised.value.status_code == 422
    assert "'gguf'" not in raised.value.detail


def test_status_never_offers_whisper_cpp(monkeypatch):
    monkeypatch.setattr(ggml_module, "is_available", lambda: True)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda _model: Path("/cache/ggml.bin"))
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    # Even a whisper-server model somehow still resident is never advertised, so no client
    # picks the retired engine as the one to poll.
    monkeypatch.setattr(GgmlSttSidecar, "loaded_model", property(lambda _self: "small"))
    monkeypatch.setattr(GgmlSttSidecar, "is_loading", lambda _self: True)

    body = json.loads(asyncio.run(ri.stt_status(model = None, current_subject = "owner")).body)
    gguf = body["gguf"]

    assert gguf["available"] is False
    assert gguf["loaded_model"] is None
    assert gguf["loading"] is False
    assert gguf["device"] is None
    assert gguf["models"] == []
    assert gguf["downloaded_models"] == []
    assert gguf["default_model"] is None
    # Kept so old clients still parse the block.
    assert {"loaded_model", "loading", "device", "keep_alive_seconds", "download"} <= set(gguf)


def test_the_whisper_update_route_is_inert(monkeypatch):
    from routes import whisper as whisper_route
    from utils import whisper_cpp_update

    monkeypatch.setattr(whisper_cpp_update, "get_update_status", _whisper_cpp_must_not_run)
    monkeypatch.setattr(
        whisper_route, "get_update_status", _whisper_cpp_must_not_run, raising = False
    )

    status = asyncio.run(
        whisper_route.whisper_update_status(force_refresh = True, current_subject = "owner")
    )

    assert status.supported is False
    assert status.update_available is False
    assert status.job.state == "idle"
