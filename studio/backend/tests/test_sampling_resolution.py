# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective sampling resolution: per-model recommendation + operator pins.

Precedence per field: operator UNSLOTH_SAMPLING_* pin -> client explicit value ->
per-model recommendation (load_inference_config) -> static schema default.
"""

import pytest

from utils.inference.inference_config import resolve_effective_sampling, SAMPLING_FIELD_NAMES
from utils.inference import inference_config as ic

_SCHEMA_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.01,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
}


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    # The recommended lookup is lru-cached; clear it so a patched config takes effect.
    ic._recommended_sampling.cache_clear()
    for field in SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(ic._SAMPLING_FIELDS[field][0], raising = False)
    yield
    ic._recommended_sampling.cache_clear()


def _all_omitted():
    return {f: None for f in SAMPLING_FIELD_NAMES}


def _set_recommended(monkeypatch, mapping):
    # _recommended_sampling sources from load_inference_config -- the exact block the Chat UI
    # seeds from -- so patch that directly. Fields absent from `mapping` fall to schema defaults.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(mapping))
    ic._recommended_sampling.cache_clear()


def test_recommended_applies_when_client_omits(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 1.0
    assert eff["top_k"] == 64
    assert eff["min_p"] == 0.0
    # A field with no recommendation keeps the static schema default.
    assert eff["top_p"] == 0.95


def test_client_explicit_beats_recommended(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0})
    eff = resolve_effective_sampling("some/model", {**_all_omitted(), "temperature": 0.2})
    assert eff["temperature"] == 0.2


def test_operator_pin_beats_client_and_recommended(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0})
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    eff = resolve_effective_sampling("some/model", {**_all_omitted(), "temperature": 0.2})
    assert eff["temperature"] == 0.9


def test_unknown_model_matches_ui_inference_block(monkeypatch):
    # An unknown model gets the same values the Chat UI would seed (load_inference_config's
    # default.yaml fallback: temp 0.7 / top_k -1), NOT the request schema defaults.
    ui_block = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.01,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(ui_block))
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/unknown-model", _all_omitted())
    assert eff["temperature"] == 0.7
    assert eff["top_k"] == -1
    assert eff["min_p"] == 0.01


def test_empty_recommendation_falls_back_to_schema_defaults(monkeypatch):
    # If load_inference_config yields nothing usable, the resolver falls back to the request
    # schema defaults.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff == _SCHEMA_DEFAULTS


@pytest.mark.parametrize(
    "model",
    ["unsloth/gemma-4-E4B", "unsloth/Qwen3-4B", "unsloth/Qwen3.5-9B", "someorg/unknown-xyz"],
)
def test_recommendation_matches_ui_source(model):
    # Parity guard: what the server recommends for omitted fields equals the Chat UI's source
    # (load_inference_config) for every field the UI adopts (mergeBackendRecommendedInference).
    ic._recommended_sampling.cache_clear()
    ui = ic.load_inference_config(model)
    rec = ic._recommended_sampling(model)
    for f in ic._UI_RECOMMENDED_FIELDS:
        cleaned = ic._clean_sampling_value(f, ui.get(f))
        if cleaned is not None:
            assert rec.get(f) == cleaned, f"{model}:{f} rec={rec.get(f)} ui={ui.get(f)}"


def test_qwen38_reuses_qwen36_sampling_defaults():
    qwen36 = ic.load_inference_config("unsloth/Qwen3.6-27B-GGUF")
    qwen38 = ic.load_inference_config("unsloth/Qwen3.8-27B-GGUF")

    assert qwen38 == qwen36
    assert qwen38 == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    "thinking_mode, expected_temperature, expected_top_p, expected_presence_penalty",
    [
        (True, 1.0, 0.95, 0.0),
        (False, 0.7, 0.8, 1.5),
        (None, 0.7, 0.8, 1.5),
    ],
)
def test_qwen38_sampling_presets_follow_explicit_thinking_mode(
    thinking_mode, expected_temperature, expected_top_p, expected_presence_penalty
):
    eff = resolve_effective_sampling(
        "unsloth/Qwen3.8-27B-GGUF",
        _all_omitted(),
        thinking_mode = thinking_mode,
    )

    assert eff["temperature"] == expected_temperature
    assert eff["top_p"] == expected_top_p
    assert eff["top_k"] == 20
    assert eff["min_p"] == 0.0
    assert eff["presence_penalty"] == expected_presence_penalty


def test_mode_absent_keeps_qwen38_historical_flat_config():
    assert ic.load_inference_config("unsloth/Qwen3.8-27B-GGUF") == ic.load_inference_config(
        "unsloth/Qwen3.8-27B-GGUF", thinking_mode = None
    )


def test_family_without_mode_presets_keeps_historical_config():
    model = "unsloth/Gemma-4-E4B-GGUF"
    historical = ic.load_inference_config(model)
    assert ic.load_inference_config(model, thinking_mode = True) == historical
    assert ic.load_inference_config(model, thinking_mode = False) == historical


def test_qwen38_explicit_client_value_beats_thinking_preset():
    eff = resolve_effective_sampling(
        "unsloth/Qwen3.8-27B-GGUF",
        {**_all_omitted(), "temperature": 0.2},
        thinking_mode = True,
    )
    assert eff["temperature"] == 0.2
    assert eff["top_p"] == 0.95


def test_qwen38_operator_pin_beats_client_and_non_thinking_preset(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    eff = resolve_effective_sampling(
        "unsloth/Qwen3.8-27B-GGUF",
        {**_all_omitted(), "temperature": 0.2},
        thinking_mode = False,
    )
    assert eff["temperature"] == 0.9
    assert eff["top_p"] == 0.8


def test_repetition_penalty_not_auto_recommended(monkeypatch):
    # The Chat UI's mergeBackendRecommendedInference never adopts a backend repetition_penalty
    # (e.g. lfm2's family value 1.05), so the server must not auto-apply one either. It stays at
    # the schema default unless the client sends it or an operator pins it.
    monkeypatch.setattr(
        ic, "load_inference_config", lambda mid: {"temperature": 0.7, "repetition_penalty": 1.05}
    )
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff["temperature"] == 0.7  # a UI-adopted field is recommended
    assert eff["repetition_penalty"] == 1.0  # rep is NOT auto-recommended (matches the UI)
    # An operator can still pin it explicitly.
    monkeypatch.setenv("UNSLOTH_SAMPLING_REPETITION_PENALTY", "1.05")
    eff2 = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff2["repetition_penalty"] == 1.05


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("abc", None),  # unparseable
        ("9.0", None),  # above temperature max (2.0)
        ("-1", None),  # below temperature min (0.0)
        ("   ", None),  # blank
        ("nan", None),  # NaN would pass a naive range check
        ("inf", None),  # non-finite
        ("-inf", None),  # non-finite
    ],
)
def test_operator_override_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", raw)
    assert ic._operator_sampling_override("temperature") == expected


def test_out_of_range_recommendation_is_dropped(monkeypatch):
    # A malformed model recommendation (out of range) is ignored, so the request keeps the
    # schema default rather than forwarding a bad value to llama-server.
    _set_recommended(monkeypatch, {"temperature": 5.0, "top_k": 64})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 0.6  # 5.0 is outside [0, 2] -> schema default
    assert eff["top_k"] == 64  # a valid recommendation is still applied


def test_operator_override_top_k_int_and_range(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "40")
    assert ic._operator_sampling_override("top_k") == 40
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "200")  # above max 100
    assert ic._operator_sampling_override("top_k") is None
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "-1")  # min allowed
    assert ic._operator_sampling_override("top_k") == -1


@pytest.mark.parametrize(
    "field, val",
    [
        ("top_k", 10**400),  # oversized int on an int field: int() ok, but math.isfinite raises
        ("top_k", float("nan")),  # NaN reaching an int field: int(nan) raises ValueError
        ("top_k", float("inf")),  # inf reaching an int field: int(inf) raises OverflowError
        (
            "temperature",
            10**400,
        ),  # oversized int on a float field: float(huge_int) raises OverflowError
    ],
)
def test_clean_sampling_value_rejects_unrepresentable(field, val):
    # None of these may raise; each is unusable and must be dropped to None (regression: an
    # oversized value used to raise OverflowError before the range check could drop it).
    assert ic._clean_sampling_value(field, val) is None


def test_oversized_operator_override_ignored(monkeypatch):
    # A huge integer string parses via int() but overflows float(); math.isfinite would raise
    # OverflowError and 500 the request. It must be ignored like any other bad override and the
    # field must fall back to the schema default -- no exception.
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "9" * 400)
    assert ic._operator_sampling_override("top_k") is None
    _set_recommended(monkeypatch, {})  # no per-model recommendation -> schema default applies
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["top_k"] == 20  # schema default, resolved without raising


def test_oversized_recommendation_ignored(monkeypatch):
    # A malformed per-model recommendation carrying an oversized int must not raise while
    # resolving either; the field simply falls back to the schema default.
    _set_recommended(monkeypatch, {"temperature": 10**400, "top_k": 64})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 0.6  # oversized -> dropped -> schema default
    assert eff["top_k"] == 64  # a valid recommendation is still applied


def test_fill_recommended_sampling_openai_payload(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})

    # Client sent only temperature; top_k / min_p were omitted.
    payload = ChatCompletionRequest(
        model = "m", messages = [{"role": "user", "content": "hi"}], temperature = 0.2
    )
    _fill_recommended_sampling_openai(payload, "some/model")
    assert payload.temperature == 0.2  # explicit client value preserved
    assert payload.top_k == 64  # recommended fills the omitted field
    assert payload.min_p == 0.0
    assert payload.top_p == 0.95  # no recommendation -> schema default unchanged


def test_fill_recommended_sampling_openai_operator_pin_overrides_client(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    monkeypatch.setattr(ic, "load_model_defaults", lambda mid: {})
    monkeypatch.setattr(ic, "get_family_inference_params", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")

    payload = ChatCompletionRequest(
        model = "m", messages = [{"role": "user", "content": "hi"}], temperature = 0.2
    )
    _fill_recommended_sampling_openai(payload, "some/model")
    assert payload.temperature == 0.9  # operator pin wins even over an explicit client value


@pytest.mark.parametrize(
    "request_kwargs, expected_temperature, expected_top_p",
    [
        ({"enable_thinking": True}, 1.0, 0.95),
        ({"enable_thinking": False}, 0.7, 0.8),
        ({}, 0.7, 0.8),
        ({"thinking": {"type": "enabled"}}, 1.0, 0.95),
        ({"reasoning_effort": "none"}, 0.7, 0.8),
        ({"reasoning_effort": "high"}, 1.0, 0.95),
    ],
)
def test_fill_recommended_sampling_openai_uses_normalized_request_mode(
    request_kwargs, expected_temperature, expected_top_p
):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    payload = ChatCompletionRequest(
        model = "m",
        messages = [{"role": "user", "content": "hi"}],
        **request_kwargs,
    )
    _fill_recommended_sampling_openai(payload, "unsloth/Qwen3.8-27B-GGUF")

    assert payload.temperature == expected_temperature
    assert payload.top_p == expected_top_p


@pytest.mark.parametrize(
    "thinking_mode, expected_temperature, expected_top_p",
    [(True, 1.0, 0.95), (False, 0.7, 0.8)],
)
def test_chat_route_lifts_harness_template_kwargs_before_sampling(
    monkeypatch, thinking_mode, expected_temperature, expected_top_p
):
    """Exercise the DeepSeek Harness request shape through the real chat route.

    The client sends ``chat_template_kwargs`` as an OpenAI extra-body field. The
    route must lift it onto the typed request before mode-specific sampling is
    filled; testing those two helpers separately would not pin that ordering.
    """
    import asyncio
    from types import SimpleNamespace

    from models.inference import ChatCompletionRequest
    from routes import inference as inference_route

    class _StopAfterSampling(Exception):
        pass

    async def _no_auto_switch(*_args, **_kwargs):
        return None

    llama_backend = SimpleNamespace(
        is_loaded = True,
        model_identifier = "unsloth/Qwen3.8-27B-GGUF",
        _is_audio = False,
    )
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
        scope = {},
    )
    payload = ChatCompletionRequest(
        model = "deepseek-harness-model",
        messages = [{"role": "user", "content": "hi"}],
        chat_template_kwargs = {"enable_thinking": thinking_mode},
    )
    assert payload.enable_thinking is None

    real_fill = inference_route._fill_recommended_sampling_openai

    def _capture_after_sampling(route_payload, model_id):
        assert route_payload.enable_thinking is thinking_mode
        real_fill(route_payload, model_id)
        raise _StopAfterSampling

    monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_auto_switch)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama_backend)
    monkeypatch.setattr(
        inference_route, "_fill_recommended_sampling_openai", _capture_after_sampling
    )

    with pytest.raises(_StopAfterSampling):
        asyncio.run(inference_route.openai_chat_completions(payload, request, "test-user"))

    assert payload.temperature == expected_temperature
    assert payload.top_p == expected_top_p


@pytest.mark.parametrize(
    "request_kwargs, expected_thinking, expected_effort, expected_preserve, expected_temperature, expected_top_p",
    [
        ({"reasoning_effort": "none"}, False, "none", None, 0.7, 0.8),
        (
            {"enable_thinking": True, "reasoning_effort": "none"},
            True,
            None,
            None,
            1.0,
            0.95,
        ),
        (
            {"enable_thinking": False, "reasoning_effort": "high"},
            False,
            None,
            None,
            0.7,
            0.8,
        ),
        (
            {"thinking": {"type": "enabled"}, "reasoning_effort": "none"},
            False,
            "none",
            None,
            0.7,
            0.8,
        ),
        (
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "medium",
                    "preserve_thinking": True,
                }
            },
            True,
            "medium",
            True,
            1.0,
            0.95,
        ),
        (
            {
                "reasoning_effort": "high",
                "preserve_thinking": False,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "reasoning_effort": "xhigh",
                    "preserve_thinking": True,
                },
            },
            True,
            "high",
            False,
            1.0,
            0.95,
        ),
        (
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "none",
                }
            },
            True,
            None,
            None,
            1.0,
            0.95,
        ),
        (
            {"chat_template_kwargs": {"enable_thinking": "false"}},
            None,
            None,
            None,
            0.7,
            0.8,
        ),
        (
            {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "reasoning_effort": {},
                }
            },
            False,
            None,
            None,
            0.7,
            0.8,
        ),
    ],
)
def test_chat_route_normalizes_reasoning_effort_before_sampling_and_generation(
    monkeypatch,
    request_kwargs,
    expected_thinking,
    expected_effort,
    expected_preserve,
    expected_temperature,
    expected_top_p,
):
    import asyncio
    from types import SimpleNamespace

    from models.inference import ChatCompletionRequest
    from routes import inference as inference_route
    from state.tool_policy import reset_tool_policy

    captured = {}

    def _generate(**kwargs):
        captured.update(kwargs)
        yield "done"

    async def _no_auto_switch(*_args, **_kwargs):
        return None

    reset_tool_policy()
    llama_backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = False,
        supports_reasoning = True,
        reasoning_always_on = False,
        _is_audio = False,
        model_identifier = "unsloth/Qwen3.8-27B-GGUF",
        context_length = 4096,
        generate_chat_completion = _generate,
    )

    class _Request:
        state = SimpleNamespace(skip_api_monitor = True)
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"
        scope = {}

        async def is_disconnected(self):
            return False

    payload = ChatCompletionRequest(
        model = "local-model",
        messages = [{"role": "user", "content": "hi"}],
        **request_kwargs,
    )

    monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_auto_switch)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama_backend)

    response = asyncio.run(
        inference_route.openai_chat_completions(payload, _Request(), "test-user")
    )

    assert response.status_code == 200
    assert payload.temperature == expected_temperature
    assert payload.top_p == expected_top_p
    assert captured["enable_thinking"] is expected_thinking
    assert captured["reasoning_effort"] == expected_effort
    assert captured["preserve_thinking"] is expected_preserve


@pytest.mark.parametrize(
    "reasoning_style, enable_thinking, reasoning_effort, expected",
    [
        ("enable_thinking", True, "none", {"enable_thinking": True}),
        ("enable_thinking", False, "high", {"enable_thinking": False}),
        ("reasoning_effort", True, "none", {"reasoning_effort": "high"}),
        ("reasoning_effort", False, "high", {"reasoning_effort": "low"}),
        ("enable_thinking_effort", True, "none", {"enable_thinking": True}),
        ("enable_thinking_effort", False, "high", {"enable_thinking": False}),
    ],
)
def test_conflicting_controls_resolve_before_model_specific_translation(
    reasoning_style, enable_thinking, reasoning_effort, expected
):
    from core.inference.llama_cpp import LlamaCppBackend
    from routes.inference import _resolve_reasoning_controls

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = reasoning_style
    backend._reasoning_effort_levels = ["high", "max"]
    backend._supports_preserve_thinking = False
    backend._architecture = None

    resolved = _resolve_reasoning_controls(enable_thinking, reasoning_effort)

    assert backend._request_reasoning_kwargs(*resolved) == expected


def test_fill_recommended_sampling_completions_body(monkeypatch):
    # /v1/completions is a raw proxy: recommendations fill omitted fields, but a field with no
    # recommendation and no pin is left absent so llama-server keeps its own default (unlike the
    # chat schema, which carries per-field defaults).
    from routes.inference import _fill_recommended_sampling_completions

    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})

    body = {"prompt": "hi", "temperature": 0.2}
    _fill_recommended_sampling_completions(body, "some/model")
    assert body["temperature"] == 0.2  # explicit client value preserved
    assert body["top_k"] == 64  # recommendation fills the omitted field
    assert body["min_p"] == 0.0
    # No recommendation and no pin -> NOT injected (llama-server keeps its default).
    assert "top_p" not in body
    assert "presence_penalty" not in body
    assert "repeat_penalty" not in body


def test_fill_recommended_sampling_completions_operator_pin(monkeypatch):
    # An operator pin overrides the client's raw-body value, and the repetition pin is written
    # under llama-server's "repeat_penalty" key (the schema field is repetition_penalty).
    from routes.inference import _fill_recommended_sampling_completions

    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    monkeypatch.setenv("UNSLOTH_SAMPLING_REPETITION_PENALTY", "1.2")

    body = {"prompt": "hi", "temperature": 0.2, "repeat_penalty": 1.05}
    _fill_recommended_sampling_completions(body, "some/model")
    assert body["temperature"] == 0.9  # operator pin wins over the client's explicit value
    assert body["repeat_penalty"] == 1.2  # repetition pin lands on llama-server's key
    assert "repetition_penalty" not in body  # never leak the schema field name into the body


@pytest.mark.parametrize("thinking_type", ["DISABLED", "Disabled", "adaptive", "enabled"])
def test_anthropic_sampling_mode_matches_the_generation_resolver(thinking_type):
    """Sampling and generation must read the same sentinel.

    AnthropicThinkingConfig.type is a plain str so unreleased Anthropic tiers stay
    servable, and resolved_enable_thinking() treats only the exact "disabled" as
    off. Lowercasing in the sampling helper made a case variant generate in
    thinking mode while sampling picked the non-thinking preset.
    """
    from models.inference import AnthropicMessagesRequest
    from routes.inference import _normalized_sampling_thinking_mode

    payload = AnthropicMessagesRequest.model_validate(
        {
            "model": "unsloth/Qwen3.8-27B-GGUF",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": thinking_type},
        }
    )
    assert _normalized_sampling_thinking_mode(payload) == payload.resolved_enable_thinking()


def test_anthropic_disabled_sentinel_still_selects_the_non_thinking_preset():
    from models.inference import AnthropicMessagesRequest
    from routes.inference import _normalized_sampling_thinking_mode

    payload = AnthropicMessagesRequest.model_validate(
        {
            "model": "unsloth/Qwen3.8-27B-GGUF",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
        }
    )
    assert _normalized_sampling_thinking_mode(payload) is False
