# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenizer-based audio_type detection, covering Gemma 3n
(<audio_soft_token>) and Gemma 4 (<|audio|>) audio-input tokens."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from utils.models.model_config import _AUDIO_TOKEN_PATTERNS, is_audio_input_type


def _classify(tokens: list[str]) -> str | None:
    """Mirror _check_token_patterns: first match in dict order wins."""
    for audio_type, check in _AUDIO_TOKEN_PATTERNS.items():
        if check(tokens):
            return audio_type
    return None


def test_gemma3n_audio_soft_token_is_audio_vlm():
    assert _classify(["<bos>", "<audio_soft_token>", "<image_soft_token>"]) == "audio_vlm"


def test_gemma4_pipe_audio_token_is_audio_vlm():
    # Gemma 4 uses <|audio|> (and <|image|>) instead of *_soft_token.
    assert _classify(["<bos>", "<|image|>", "<|audio|>"]) == "audio_vlm"


def test_csm_uppercase_audio_not_classified_as_audio_vlm():
    # csm uses uppercase <|AUDIO|> + <|audio_eos|>; must stay csm, not audio_vlm.
    tokens = ["<|AUDIO|>", "<|audio_eos|>"]
    assert _classify(tokens) == "csm"


def test_audio_vlm_and_whisper_accept_audio_input():
    assert is_audio_input_type("audio_vlm") is True
    assert is_audio_input_type("whisper") is True
    assert is_audio_input_type("snac") is False
    assert is_audio_input_type(None) is False


def test_non_audio_tokens_classify_none():
    assert _classify(["<bos>", "<eos>", "<pad>"]) is None


def test_orpheus_snac_codebook_beats_a_stray_audio_marker():
    """Orpheus ships 28k <custom_token_N> SNAC codes AND a lone <|audio|>.

    audio_vlm was tested first and won, so a TTS model came back as audio-INPUT:
    is_audio stayed False and the Audio page refused it.
    """
    tokens = ["<|audio|>"] + [f"<custom_token_{i}>" for i in range(28683)]
    assert _classify(tokens) == "snac"
    assert is_audio_input_type(_classify(tokens)) is False


def test_a_codec_family_is_not_shadowed_by_a_stray_audio_marker():
    """The same precedence has to hold for every output codec, not just snac."""
    assert _classify(["<|audio|>", "<|bicodec_semantic_0|>"]) == "bicodec"
    assert (
        _classify(
            [
                "<|audio|>",
                "<|audio_start|>",
                "<|audio_end|>",
                "<|text_start|>",
                "<|text_end|>",
            ]
        )
        == "dac"
    )


def test_gguf_codec_dependency_is_published_before_initialization():
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._healthy = True
    events = []
    backend.init_audio_codec = lambda audio_type: events.append(("init", audio_type))

    assert backend._apply_detected_audio(
        "snac", lambda repo_id: events.append(("reserve", repo_id))
    )
    assert events == [
        ("reserve", "hubertsiuzdak/snac_24khz"),
        ("init", "snac"),
    ]


def test_new_gguf_server_is_unloaded_when_codec_init_fails(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._healthy = True
    events = []

    def failing_init(audio_type):
        events.append(("init", audio_type))
        raise RuntimeError("codec weights missing")

    monkeypatch.setattr(
        backend,
        "_teardown_model",
        lambda: events.append(("unload", None)) or True,
    )
    monkeypatch.setattr(backend, "init_audio_codec", failing_init)

    assert backend._apply_detected_audio_for_new_server("dac") is False
    assert events == [("init", "dac"), ("unload", None)]


def test_new_gguf_server_is_unloaded_when_codec_reservation_conflicts(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._healthy = True
    events = []

    def reject_reservation(repo_id):
        events.append(("reserve", repo_id))
        raise HTTPException(status_code = 409, detail = "cache busy")

    monkeypatch.setattr(
        backend,
        "_teardown_model",
        lambda: events.append(("unload", None)) or True,
    )
    monkeypatch.setattr(
        backend,
        "init_audio_codec",
        lambda audio_type: events.append(("init", audio_type)),
    )

    with pytest.raises(HTTPException) as caught:
        backend._apply_detected_audio_for_new_server("snac", reject_reservation)

    assert caught.value.status_code == 409
    assert events == [
        ("reserve", "hubertsiuzdak/snac_24khz"),
        ("unload", None),
    ]


def _backend_with_failing_codec(monkeypatch):
    """A spawned-server backend whose codec init fails and uses the real teardown."""
    from core.inference.llama_cpp import LlamaCppBackend

    def failing_init(audio_type):
        raise RuntimeError("codec weights missing")

    # Class state a sibling test may have left behind; unloading it here would import torch.
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", None)
    backend = LlamaCppBackend()
    backend._healthy = True
    monkeypatch.setattr(backend, "init_audio_codec", failing_init)
    return backend


def test_the_new_gguf_server_teardown_leaves_no_cancellation_behind(monkeypatch):
    # unload_model arms the load-cancel event so an in-flight download stops. Nothing is
    # downloading here, and leaving it armed makes load_cancelled() report a user cancellation,
    # which skips the tensor-parallel -> layer retry this teardown exists to let run cleanly.
    backend = _backend_with_failing_codec(monkeypatch)

    assert backend._apply_detected_audio_for_new_server("dac") is False
    assert backend.load_cancelled() is False


def test_a_real_cancellation_survives_the_new_gguf_server_teardown(monkeypatch):
    # An unload that landed while the probe ran is a cancellation the load must still see.
    backend = _backend_with_failing_codec(monkeypatch)
    backend._cancel_event.set()

    assert backend._apply_detected_audio_for_new_server("dac") is False
    assert backend.load_cancelled() is True


def test_an_unload_during_audio_abort_teardown_remains_cancelled(monkeypatch):
    import threading

    backend = _backend_with_failing_codec(monkeypatch)
    teardown_started = threading.Event()
    finish_teardown = threading.Event()

    def blocking_kill():
        teardown_started.set()
        assert finish_teardown.wait(5)

    monkeypatch.setattr(backend, "_kill_process", blocking_kill)
    audio_abort = threading.Thread(target = backend._unload_after_audio_abort)
    user_unload = threading.Thread(target = backend.unload_model)

    audio_abort.start()
    assert teardown_started.wait(5)
    user_unload.start()
    assert backend._cancel_event.wait(5)
    finish_teardown.set()
    audio_abort.join(5)
    user_unload.join(5)

    assert not audio_abort.is_alive()
    assert not user_unload.is_alive()
    assert backend.load_cancelled() is True


class _Resp:
    def __init__(
        self,
        status_code: int,
        payload = None,
    ):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def _detect_checked(
    monkeypatch,
    responses,
    model = "acme/tts-model",
):
    """Drive detect_audio_type_checked with a faked Hub, no local cache."""
    from utils.models import model_config as mc

    monkeypatch.setattr(mc, "_audio_detection_cache", {})
    monkeypatch.setattr(mc, "get_cache_path", lambda *a, **k: None)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)

    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kw: responses.pop(0))
    return mc.detect_audio_type_checked(model)


def test_a_gated_repo_is_not_reported_as_definitively_non_audio(monkeypatch):
    # 401 on every tokenizer_config path: nothing was read, so None means unknown.
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(401), _Resp(401)])
    assert audio_type is None
    assert definitive is False


def test_a_readable_repo_without_audio_tokens_is_definitive(monkeypatch):
    # 200 with a plain tokenizer, then a 404 for the LLM/ variant: a real negative.
    plain = {"added_tokens_decoder": {"0": {"content": "<bos>"}}}
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(200, plain), _Resp(404)])
    assert audio_type is None
    assert definitive is True


def test_a_detected_codec_is_definitive(monkeypatch):
    snac = {
        "added_tokens_decoder": {str(i): {"content": f"<custom_token_{i}>"} for i in range(10_001)}
    }
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(200, snac)])
    assert audio_type == "snac"
    assert definitive is True


def test_a_local_path_never_reaches_the_hub(monkeypatch, tmp_path):
    """A filesystem path is not a repo id, so the Hub URL would be nonsense.

    /loras hits this for every adapter directory without its own tokenizer, and a transient
    failure is never cached, so it paid two 15s timeouts per checkpoint on every scan while
    blocking the event loop that called it.
    """
    from utils.models import model_config

    # Recorded rather than raised: the fetch loop catches every exception and treats it as
    # a transient failure, so a raising stub would be swallowed and the test would pass
    # against the unfixed code.
    fetched = []

    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: fetched.append(url))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding = "utf-8")

    result, definitive = model_config._detect_audio_from_tokenizer(str(adapter))
    assert fetched == [], fetched
    assert result is None
    # Nothing was read, so the answer is not definitive and must not be cached.
    assert definitive is False


def test_an_offline_miss_is_not_reprobed_on_every_poll(monkeypatch, tmp_path):
    """/loras probes every checkpoint and its base. Neither answers offline, and a
    non-definitive result is never cached, so the walk repeated on every poll: with 50
    checkpoints that measured 6ms -> 26ms per call, on the event loop."""
    from utils.models import model_config

    monkeypatch.setattr(model_config, "_audio_detection_cache", {})
    monkeypatch.setattr(model_config, "_audio_offline_miss_cache", {})
    probes = []
    monkeypatch.setattr(
        model_config,
        "_detect_audio_from_tokenizer",
        lambda name, token = None, **kw: (probes.append(name), (None, False))[1],
    )

    for _ in range(5):
        assert model_config.detect_audio_type_checked(
            "org/not-downloaded", local_files_only = True
        ) == (None, False)
    assert probes == ["org/not-downloaded"], probes


def test_the_offline_miss_expires_so_a_later_download_is_seen(monkeypatch):
    """Bounded, not permanent: the base may be downloaded, or a training run may finish
    writing the tokenizer it was missing, and neither restarts Studio."""
    from utils.models import model_config

    monkeypatch.setattr(model_config, "_audio_detection_cache", {})
    monkeypatch.setattr(model_config, "_audio_offline_miss_cache", {})
    answers = iter([(None, False), ("snac", True)])
    monkeypatch.setattr(
        model_config,
        "_detect_audio_from_tokenizer",
        lambda name, token = None, **kw: next(answers),
    )
    clock = [1000.0]
    monkeypatch.setattr(model_config.time, "monotonic", lambda: clock[0])

    assert model_config.detect_audio_type_checked("org/m", local_files_only = True)[0] is None
    clock[0] += model_config._AUDIO_OFFLINE_MISS_TTL_S + 1
    assert model_config.detect_audio_type_checked("org/m", local_files_only = True) == ("snac", True)
    # Definitive now, so it is in the real cache and the miss entry is gone.
    assert model_config._audio_offline_miss_cache == {}


def test_an_online_transient_failure_still_retries_immediately(monkeypatch):
    """The bound is deliberately only for probes that touched no network. A gated repo or
    a 5xx must not be remembered, or fixing the token would take a minute to take."""
    from utils.models import model_config

    monkeypatch.setattr(model_config, "_audio_detection_cache", {})
    monkeypatch.setattr(model_config, "_audio_offline_miss_cache", {})
    probes = []
    monkeypatch.setattr(
        model_config,
        "_detect_audio_from_tokenizer",
        lambda name, token = None, **kw: (probes.append(name), (None, False))[1],
    )

    for _ in range(3):
        model_config.detect_audio_type_checked("org/gated", local_files_only = False)
    assert len(probes) == 3, probes


def test_every_pattern_has_a_marker_so_the_parse_can_be_skipped():
    """The marker list is what lets a large text tokenizer_config be settled without
    parsing it. It cannot be derived from the patterns, which are lambdas, so a codec
    added there without a marker here would silently stop being detected."""
    from utils.models.model_config import (
        _AUDIO_TOKEN_MARKERS,
        _AUDIO_TOKEN_PATTERNS,
        _may_hold_audio_tokens,
    )

    # Fails when a codec is added, which is the point: add its marker too.
    assert set(_AUDIO_TOKEN_PATTERNS) == {
        "csm",
        "whisper",
        "bicodec",
        "dac",
        "snac",
        "audio_vlm",
    }

    # Whatever each pattern matches, the marker scan must let it through to the parse.
    samples = {
        "csm": ["<|AUDIO|>", "<|audio_eos|>"],
        "whisper": ["<|startoftranscript|>"],
        "bicodec": ["<|bicodec_semantic_0|>"],
        "dac": ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"],
        "snac": [f"<custom_token_{i}>" for i in range(10001)],
        "audio_vlm": ["<audio_soft_token>"],
    }
    for audio_type, tokens in samples.items():
        assert _classify(tokens) == audio_type, audio_type
        assert _may_hold_audio_tokens(json.dumps(tokens)), audio_type
    assert _may_hold_audio_tokens(json.dumps(["<|image|>", "<|audio|>"]))

    # And an ordinary text tokenizer is settled without a parse.
    assert not _may_hold_audio_tokens(
        json.dumps([f"<|extra_token_{i}|>" for i in range(500)] + ["<bos>", "<eos>"])
    )
    assert all(marker in "".join(_AUDIO_TOKEN_MARKERS) for marker in _AUDIO_TOKEN_MARKERS)


def test_a_large_text_tokenizer_is_not_parsed(monkeypatch, tmp_path):
    """The saving, pinned: an ordinary checkpoint's tokenizer_config is read but never
    handed to json.loads, which was the bulk of a cold /loras scan."""
    import json as json_module

    from utils.models import model_config

    config = {
        "added_tokens_decoder": {
            str(i): {"content": f"<|extra_token_{i}|>", "special": True} for i in range(5000)
        }
    }
    checkpoint = tmp_path / "run"
    checkpoint.mkdir()
    (checkpoint / "tokenizer_config.json").write_text(json_module.dumps(config))

    parsed = []
    real_loads = model_config.json.loads
    monkeypatch.setattr(
        model_config.json,
        "loads",
        lambda raw, *a, **kw: (parsed.append(len(raw)), real_loads(raw, *a, **kw))[1],
    )

    result, definitive = model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    )

    assert result is None
    # Read successfully, so "not audio" is a definitive answer, not an unknown.
    assert definitive is True
    assert parsed == [], parsed


def test_a_half_written_tokenizer_stays_unknown(tmp_path):
    """The skip-the-parse path must not turn a training run's part-written tokenizer into
    a definitive "not audio", which would be cached for the life of the process. It stays
    unknown, exactly as it did when json.loads raised on the truncated text."""
    from utils.models import model_config

    checkpoint = tmp_path / "mid_write"
    checkpoint.mkdir()
    whole = json.dumps({"added_tokens_decoder": {"0": {"content": "<|plain|>"}}})
    (checkpoint / "tokenizer_config.json").write_text(whole[: len(whole) // 2])

    result, definitive = model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    )
    assert result is None
    assert definitive is False

    (checkpoint / "tokenizer_config.json").write_text(whole)
    assert model_config._detect_audio_from_tokenizer(str(checkpoint), local_files_only = True) == (
        None,
        True,
    )


def test_an_adopted_server_retries_the_probe_after_a_codec_reservation_conflict(monkeypatch):
    """The adopted-server retry keeps the running server, so nothing resets the probe flag
    for it. Latching before the codec is applied left a TTS model reporting loaded with
    audio dead: every later duplicate load skipped the retry."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._healthy = True
    monkeypatch.setattr(backend, "_detect_audio_type_strict", lambda: "snac")
    monkeypatch.setattr(backend, "init_audio_codec", lambda audio_type: None)

    def reject_reservation(_repo_id):
        raise HTTPException(status_code = 409, detail = "cache busy")

    with pytest.raises(HTTPException):
        backend._retry_audio_probe(reject_reservation)
    assert backend._audio_probed is False

    assert backend._retry_audio_probe(lambda _repo_id: None) is True
    assert backend._audio_probed is True
    assert backend._is_audio is True


def test_an_adopted_server_probe_that_never_ran_is_not_marked_done(monkeypatch):
    """A transport failure leaves the codec unknown, not absent, so the next duplicate
    load has to ask again."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._healthy = True

    def failing_probe():
        raise RuntimeError("server not answering yet")

    monkeypatch.setattr(backend, "_detect_audio_type_strict", failing_probe)

    assert backend._retry_audio_probe() is True
    assert backend._audio_probed is False
