# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The current-date preference: its endpoints, and the prompts it feeds."""

from datetime import date, datetime, timezone
from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import utils.current_date_prompt_settings as current_date_settings
from core.research.prompts import _system_prompt_with_instructions


@pytest.fixture
def client(monkeypatch):
    calls: dict = {"enabled": True}

    def _set(value):
        calls["set"] = bool(value)
        calls["enabled"] = bool(value)
        return bool(value)

    monkeypatch.setattr(settings, "get_current_date_prompt_enabled", lambda: calls["enabled"])
    monkeypatch.setattr(settings, "set_current_date_prompt_enabled", _set)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), calls


def test_get_current_date_prompt(client):
    c, _ = client
    r = c.get("/current-date-prompt")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["default_enabled"] is True


def test_put_current_date_prompt_disables(client):
    c, calls = client
    r = c.put("/current-date-prompt", json = {"enabled": False})
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert calls["set"] is False


def test_put_current_date_prompt_rejects_non_bool(client):
    c, _ = client
    r = c.put("/current-date-prompt", json = {"enabled": "maybe"})
    assert r.status_code == 422


class TestCurrentDatePromptLine:
    def test_line_states_the_iso_date_when_enabled(self, monkeypatch):
        monkeypatch.setattr(current_date_settings, "get_current_date_prompt_enabled", lambda: True)
        assert (
            current_date_settings.current_date_prompt_line(date(2026, 8, 15))
            == "The current date is 2026-08-15."
        )

    def test_system_prompt_helper_refreshes_a_stale_date(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        already = "The current date is 2026-08-14.\n\nBASE"
        assert (
            inference._apply_current_date_prompt(already)
            == "The current date is 2026-08-15.\n\nBASE"
        )

    def test_system_prompt_helper_does_not_duplicate_the_current_date(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        prompt = "The current date is 2026-08-15.\n\nBASE"
        assert inference._apply_current_date_prompt(prompt) == prompt

    def test_discussing_the_date_phrase_does_not_suppress_injection(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        prompt = "The current date is a phrase this prompt discusses, not a date stamp."
        assert (
            inference._apply_current_date_prompt(prompt)
            == f"The current date is 2026-08-15.\n\n{prompt}"
        )

    def test_line_is_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(current_date_settings, "get_current_date_prompt_enabled", lambda: False)
        assert current_date_settings.current_date_prompt_line(date(2026, 8, 15)) == ""

    def test_request_timezone_decides_the_calendar_date(self):
        request = _types.SimpleNamespace(
            headers = {
                current_date_settings.CURRENT_DATE_TIMEZONE_HEADER: "Pacific/Auckland",
            }
        )
        instant = datetime(2026, 8, 28, 12, 30, tzinfo = timezone.utc)
        assert current_date_settings._request_local_date(request, instant) == date(2026, 8, 29)

    def test_request_offset_is_used_when_timezone_is_unknown(self):
        request = _types.SimpleNamespace(
            headers = {
                current_date_settings.CURRENT_DATE_TIMEZONE_HEADER: "Invalid/Zone",
                current_date_settings.CURRENT_DATE_TIMEZONE_OFFSET_HEADER: "-720",
            }
        )
        instant = datetime(2026, 8, 28, 12, 30, tzinfo = timezone.utc)
        assert current_date_settings._request_local_date(request, instant) == date(2026, 8, 29)

    def test_research_run_stamp_receives_the_http_request(self, monkeypatch):
        import routes.research_runs as research_routes

        request = _types.SimpleNamespace(headers = {"x-unsloth-timezone": "Pacific/Auckland"})
        monkeypatch.setattr(
            research_routes,
            "current_date_prompt_line",
            lambda **kwargs: kwargs["request"].headers["x-unsloth-timezone"],
        )
        payload = research_routes.CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "message-1",
            inferenceRequest = {"model": "local-model"},
        )

        config = research_routes._sanitize_config(payload, {"modelId": "local-model"}, request)

        assert config["currentDate"] == "Pacific/Auckland"

    def test_unreadable_settings_still_default_to_enabled(self, monkeypatch):
        def _explode(*_args, **_kwargs):
            raise RuntimeError("settings db unavailable")

        monkeypatch.setitem(
            sys.modules,
            "storage.studio_db",
            _types.SimpleNamespace(get_app_setting = _explode),
        )
        assert current_date_settings.get_current_date_prompt_enabled() is True


class TestExternalProviderMessages:
    """vLLM, Ollama, OpenAI and custom connections are proxied, so the date goes on the payload."""

    @staticmethod
    def _prepend(messages):
        import routes.inference as inference
        return inference._prepend_current_date_to_messages(messages)

    @pytest.fixture(autouse = True)
    def _enabled(self, monkeypatch):
        import routes.inference as inference
        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )

    def test_date_prefixes_an_existing_system_turn(self):
        out = self._prepend(
            [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
        )
        assert out[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."
        assert out[1] == {"role": "user", "content": "hi"}

    def test_system_turn_is_created_when_absent(self):
        out = self._prepend([{"role": "user", "content": "hi"}])
        assert out[0] == {"role": "system", "content": "The current date is 2026-08-15."}
        assert out[1] == {"role": "user", "content": "hi"}

    def test_ollama_keeps_its_modelfile_system_prompt_when_studio_sends_none(self):
        # Ollama applies the Modelfile SYSTEM only while messages[0] is not a system turn.
        import routes.inference as inference

        messages = [{"role": "user", "content": "hi"}]
        out = inference._prepend_current_date_to_messages(messages, provider_type = "ollama")
        assert out is messages

    def test_ollama_still_dates_a_system_turn_studio_composed(self):
        import routes.inference as inference

        out = inference._prepend_current_date_to_messages(
            [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
            provider_type = "ollama",
        )
        assert out[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."
        assert out[1] == {"role": "user", "content": "hi"}

    def test_other_local_servers_still_get_a_synthesized_system_turn(self):
        import routes.inference as inference
        out = inference._prepend_current_date_to_messages(
            [{"role": "user", "content": "hi"}], provider_type = "llama_cpp"
        )
        assert out[0] == {"role": "system", "content": "The current date is 2026-08-15."}

    def test_date_prefixes_text_inside_structured_content(self):
        messages = [{"role": "system", "content": [{"type": "text", "text": "Be terse."}]}]
        out = self._prepend(messages)
        assert len(out) == 1
        assert out[0]["content"] == [
            {"type": "text", "text": "The current date is 2026-08-15.\n\nBe terse."}
        ]
        assert messages[0]["content"] == [{"type": "text", "text": "Be terse."}]

    def test_date_becomes_a_text_part_when_structured_content_has_none(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png,x"}}],
            }
        ]
        out = self._prepend(messages)
        assert len(out) == 1
        assert out[0]["content"][0] == {
            "type": "text",
            "text": "The current date is 2026-08-15.",
        }
        assert out[0]["content"][1] == messages[0]["content"][0]

    def test_developer_turn_is_used_when_there_is_no_system_turn(self):
        out = self._prepend([{"role": "developer", "content": "Be terse."}])
        assert out[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."

    def test_messages_are_untouched_when_disabled(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "current_date_prompt_line", lambda **_kwargs: "")
        messages = [{"role": "user", "content": "hi"}]
        assert self._prepend(messages) is messages

    def test_a_stale_date_is_refreshed_for_an_interactive_request(self):
        stamped = [
            {"role": "system", "content": "The current date is 2026-08-14.\n\nBe terse."},
            {"role": "user", "content": "hi"},
        ]
        refreshed = self._prepend(stamped)
        assert refreshed[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."
        assert stamped[0]["content"] == "The current date is 2026-08-14.\n\nBe terse."

    def test_a_stale_date_buried_under_instructions_is_refreshed(self):
        buried = [
            {
                "role": "system",
                "content": (
                    "Chat-specific instructions follow.\n"
                    "<chat_instructions>\nBe terse.\n</chat_instructions>\n\n"
                    "Non-overridable rules:\nThe current date is 2026-08-14.\n\nBASE"
                ),
            }
        ]
        refreshed = self._prepend(buried)
        assert "The current date is 2026-08-15." in refreshed[0]["content"]
        assert "The current date is 2026-08-14." not in refreshed[0]["content"]

    def test_a_stale_date_on_a_later_system_turn_is_refreshed(self):
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "The current date is 2026-08-14."},
            {"role": "user", "content": "hi"},
        ]
        refreshed = self._prepend(messages)
        assert refreshed[0]["content"] == "Be terse."
        assert refreshed[1]["content"] == "The current date is 2026-08-15."

    def test_any_api_key_request_is_left_verbatim(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: True)
        messages = [{"role": "user", "content": "hi"}]
        assert inference._prepend_current_date_to_messages(messages, object()) is messages
        assert inference._apply_current_date_prompt("Be terse.", object()) == "Be terse."

    def test_server_tool_loop_can_date_an_api_key_request(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: True)
        messages = [{"role": "user", "content": "hi"}]
        out = inference._prepend_current_date_to_messages(
            messages,
            object(),
            include_api_key = True,
        )
        assert out[0] == {"role": "system", "content": "The current date is 2026-08-15."}

    def test_server_tool_loop_keeps_an_internal_workflow_stamp(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: True)
        monkeypatch.setattr(inference, "_request_is_internal_workflow", lambda _request: True)
        messages = [{"role": "system", "content": "The current date is 2026-08-14."}]
        out = inference._prepend_current_date_to_messages(
            messages,
            object(),
            include_api_key = True,
        )
        assert out is messages

    def test_a_studio_session_request_is_dated(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: False)
        out = inference._prepend_current_date_to_messages(
            [{"role": "user", "content": "hi"}], object()
        )
        assert out[0] == {"role": "system", "content": "The current date is 2026-08-15."}

    def test_a_stale_date_inside_a_text_part_is_refreshed(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "The current date is 2026-08-14."}],
            },
            {"role": "user", "content": "hi"},
        ]
        refreshed = self._prepend(messages)
        assert refreshed[0]["content"][0]["text"] == "The current date is 2026-08-15."
        assert messages[0]["content"][0]["text"] == "The current date is 2026-08-14."

    def test_a_phrase_discussion_inside_a_text_part_does_not_suppress(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The current date is merely a phrase under discussion.",
                    },
                ],
            },
            {"role": "user", "content": "hi"},
        ]
        out = self._prepend(messages)
        assert len(out) == len(messages)
        assert out[0]["content"][0]["text"].startswith("The current date is 2026-08-15.\n\n")
        assert out[1:] == messages[1:]


class TestResearchSystemPrompt:
    """Every Deep Research call (planner, agent, audit, report) goes through this helper."""

    def test_stamped_date_is_prefixed(self):
        prompt = _system_prompt_with_instructions(
            "BASE", {"currentDate": "The current date is 2026-08-15."}
        )
        assert prompt == "The current date is 2026-08-15.\n\nBASE"

    def test_date_precedes_the_non_overridable_rules_with_instructions(self):
        prompt = _system_prompt_with_instructions(
            "BASE",
            {"currentDate": "The current date is 2026-08-15.", "instructions": "Be terse."},
        )
        assert "<chat_instructions>\nBe terse.\n</chat_instructions>" in prompt
        assert prompt.endswith("Non-overridable rules:\nThe current date is 2026-08-15.\n\nBASE")

    def test_stale_manual_date_is_removed_from_instructions(self):
        prompt = _system_prompt_with_instructions(
            "BASE",
            {
                "currentDate": "The current date is 2026-08-15.",
                "instructions": "The current date is 2025-03-01.\nBe terse.",
            },
        )
        assert "The current date is 2025-03-01." not in prompt
        assert prompt.count("The current date is 2026-08-15.") == 1
        assert "<chat_instructions>\nBe terse.\n</chat_instructions>" in prompt

    def test_run_without_a_stamped_date_is_unchanged(self):
        assert _system_prompt_with_instructions("BASE", {}) == "BASE"
        assert _system_prompt_with_instructions("BASE", {"currentDate": ""}) == "BASE"


_QWEN25_LIKE = (
    "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] %}"
    "{% set rest = messages[1:] %}{% else %}"
    "{% set sys = 'You are Qwen, a helpful assistant.' %}{% set rest = messages %}{% endif %}"
    "<|im_start|>system\n{{ sys }}<|im_end|>\n"
    "{% for m in rest %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
)
_CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
)
_GEMMA_LIKE = (
    "{% if messages[0]['role'] == 'system' %}{% set first = messages[0]['content'] + '\n\n' %}"
    "{% set rest = messages[1:] %}{% else %}{% set first = '' %}{% set rest = messages %}{% endif %}"
    "{% for m in rest %}<start_of_turn>{{ m['role'] }}\n"
    "{% if loop.first %}{{ first }}{% endif %}{{ m['content'] }}<end_of_turn>\n{% endfor %}"
)


_REFUSES_SYSTEM = (
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('no system') }}{% endif %}"
    + _CHATML
)
_DROPS_SYSTEM = "{% for m in messages if m['role'] != 'system' %}{{ m['content'] }}{% endfor %}"
_DAY = date(2026, 10, 4)


def _turn(
    template,
    tools = False,
    controls = (),
):
    return current_date_settings.template_system_turn(template, _DAY, tools, controls)


class TestTemplateSystemTurn:
    def test_a_template_with_its_own_default_returns_it(self):
        assert _turn(_QWEN25_LIKE) == (True, "You are Qwen, a helpful assistant.")

    def test_a_template_with_generation_blocks_keeps_its_default(self):
        template = _QWEN25_LIKE.replace(
            "{{ m['content'] }}",
            "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}"
            "{% endgeneration %}{% else %}{{ m['content'] }}{% endif %}",
        )
        assert _turn(template) == (True, "You are Qwen, a helpful assistant.")

    @pytest.mark.parametrize("template", [_CHATML, _GEMMA_LIKE, None, "", "{% if %}"])
    def test_templates_without_a_default_return_nothing(self, template):
        assert _turn(template) == (True, "")

    def test_a_default_that_dates_itself_is_dated_for_the_given_day(self):
        dated = _QWEN25_LIKE.replace(
            "You are Qwen, a helpful assistant.",
            "Today is ' + strftime_now('%B %d, %Y') + ', yesterday was the '"
            " + ((strftime_now('%d') | int) - 1) | string + 'th.",
        )
        assert _turn(dated) == (True, "Today is October 04, 2026, yesterday was the 3th.")

    def test_a_template_that_reads_content_parts_returns_its_default(self):
        parts = (
            "{% for m in messages if m['role'] == 'system' %}{% else %}<|system|>\nBe helpful.\n"
            "{% endfor %}{% for m in messages %}<|{{ m['role'] }}|>\n"
            "{{ m['content'][0]['text'] }}\n{% endfor %}"
        )
        assert _turn(parts) == (True, "Be helpful.")

    @pytest.mark.parametrize("template", [_REFUSES_SYSTEM, _DROPS_SYSTEM])
    def test_a_template_without_a_system_turn_takes_none(self, template):
        assert _turn(template) == (False, None)

    def test_a_tool_request_is_probed_with_a_catalog(self):
        tools_only = (
            "{% if tools and messages[0]['role'] == 'system' %}{{ raise_exception('no') }}"
            "{% endif %}" + _CHATML
        )
        assert _turn(tools_only) == (True, "")
        assert _turn(tools_only, True) == (False, None)

    def test_absent_tools_and_documents_are_passed_as_none_like_transformers(self):
        template = (
            "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] %}"
            "{% set rest = messages[1:] %}{% else %}"
            "{% if tools is none and documents is none %}"
            "{% set sys = 'You are a helpful function-calling assistant.' %}"
            "{% else %}{% set sys = '' %}{% endif %}{% set rest = messages %}{% endif %}"
            "<system>{{ sys }}</system>"
            "{% for m in rest %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>"
            "{% endfor %}"
        )
        assert _turn(template) == (True, "You are a helpful function-calling assistant.")

    def test_a_template_is_probed_with_the_reasoning_controls(self):
        thinking_only = (
            "{% if enable_thinking and messages[0]['role'] == 'system' %}"
            "{{ raise_exception('no') }}{% endif %}" + _CHATML
        )
        assert _turn(thinking_only) == (True, "")
        assert _turn(thinking_only, controls = (("enable_thinking", True),)) == (False, None)

    @pytest.mark.parametrize("token", ["bos_token", "eos_token", "pad_token", "sep_token"])
    def test_a_default_carrying_a_control_token_is_not_replayed(self, token):
        with_token = _QWEN25_LIKE.replace(
            "{% set sys = 'You are Qwen, a helpful assistant.' %}",
            "{% set sys = " + token + " + 'You are Qwen, a helpful assistant.' %}",
        )
        assert _turn(with_token) == (True, None)

    def test_a_default_the_template_rewrites_is_not_replayed(self):
        escaped = _QWEN25_LIKE.replace(
            "'You are Qwen, a helpful assistant.'", "'Say \"hi\".'"
        ).replace("{{ sys }}", "{{ sys | tojson }}")
        assert _turn(escaped) == (True, None)
        plain = _QWEN25_LIKE.replace("{{ sys }}", "{{ sys | tojson }}")
        assert _turn(plain) == (True, "You are Qwen, a helpful assistant.")

    def test_a_structurally_different_system_branch_is_not_treated_as_default_free(self):
        structural = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set sys = 'Think deeply.\\n\\n' + messages[0]['content'] %}"
            "{% set rest = messages[1:] %}{% else %}"
            "{% set sys = 'You are helpful. Think deeply.' %}{% set rest = messages %}"
            "{% endif %}[SYSTEM]{{ sys }}[/SYSTEM]"
            "{% for m in rest %}{{ m['role'] }}:{{ m['content'] }}{% endfor %}"
        )
        assert _turn(structural) == (True, None)

    def test_default_words_inside_a_different_system_branch_are_not_treated_as_absent(self):
        structural = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set sys = 'Always follow policy. ' + messages[0]['content'] %}"
            "{% set rest = messages[1:] %}{% else %}"
            "{% set sys = 'A policy.' %}{% set rest = messages %}{% endif %}"
            "<system>{{ sys }}</system>"
            "{% for m in rest %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>"
            "{% endfor %}"
        )
        assert _turn(structural) == (True, None)

    def test_transformers_tojson_does_not_html_escape_a_default(self):
        html = _QWEN25_LIKE.replace(
            "You are Qwen, a helpful assistant.", "Use <assistant> & answer questions."
        ).replace("{{ sys }}", "{{ sys | tojson }}")
        assert _turn(html) == (True, "Use <assistant> & answer questions.")


class TestDateStaysInTheSystemTurn:
    @pytest.fixture(autouse = True)
    def _clock(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-10-04.",
        )
        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: False)
        monkeypatch.setattr(inference, "_local_template_system_turn", lambda *_a: (True, ""))
        self.inference = inference

    def test_without_a_system_prompt_the_date_is_its_own_system_turn(self):
        assert (
            self.inference._apply_current_date_prompt("", object())
            == "The current date is 2026-10-04."
        )

    def test_a_template_default_system_prompt_is_kept_after_the_date(self, monkeypatch):
        monkeypatch.setattr(
            self.inference, "_local_template_system_turn", lambda *_a: (True, "You are Qwen.")
        )
        assert self.inference._apply_current_date_prompt("", object()) == (
            "The current date is 2026-10-04.\n\nYou are Qwen."
        )
        assert self.inference._apply_current_date_prompt("Be terse.", object()) == (
            "The current date is 2026-10-04.\n\nBe terse."
        )

    def test_a_template_default_that_dates_itself_is_dated_for_the_user(self, monkeypatch):
        monkeypatch.setattr(
            self.inference,
            "_local_template_system_turn",
            lambda today, *_a: (True, f"Today's Date: {today:%B %d, %Y}.\nYou are Granite."),
        )
        expected = (
            "The current date is 2026-10-04.\n\nToday's Date: October 04, 2026.\nYou are Granite."
        )
        assert self.inference._apply_current_date_prompt("", object()) == expected
        assert (
            self.inference._apply_current_date_prompt("", object(), include_api_key = True)
            == expected
        )

    def test_a_tool_turn_states_the_date_without_the_template_default(self, monkeypatch):
        monkeypatch.setattr(
            self.inference, "_local_template_system_turn", lambda *_a: (True, "You are Qwen.")
        )
        assert (
            self.inference._apply_current_date_prompt(
                "", object(), include_api_key = True, template_default = False
            )
            == "The current date is 2026-10-04."
        )

    def test_a_template_without_a_system_turn_gets_no_date(self, monkeypatch):
        monkeypatch.setattr(
            self.inference, "_local_template_system_turn", lambda *_a: (False, None)
        )
        assert self.inference._apply_current_date_prompt("", object()) == ""
        assert self.inference._apply_current_date_prompt("Be terse.", object()) == (
            "The current date is 2026-10-04.\n\nBe terse."
        )

    def test_an_audio_turn_keeps_its_transcription_instruction(self, monkeypatch):
        instruction = self.inference._AUDIO_INPUT_SYSTEM_PROMPT
        assert self.inference._audio_input_system_prompt("", object()) == (
            f"The current date is 2026-10-04.\n\n{instruction}"
        )
        assert self.inference._audio_input_system_prompt("Be terse.", object()) == (
            "The current date is 2026-10-04.\n\nBe terse."
        )
        monkeypatch.setattr(self.inference, "current_date_prompt_line", lambda **_kwargs: "")
        assert self.inference._audio_input_system_prompt("", object()) == instruction

    def test_a_managed_engine_gets_no_unrequested_system_turn(self, monkeypatch):
        from types import SimpleNamespace

        from core.inference import orchestrator

        monkeypatch.undo()
        info = {"engine": "vllm", "chat_template_info": {}}
        backend = SimpleNamespace(active_model_name = "served", models = {"served": info})
        monkeypatch.setattr(
            self.inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
        )
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
        assert self.inference._local_template_system_turn(_DAY) == (False, None)
        assert self.inference._local_template_system_turn(_DAY, tools = True) == (False, None)

    @pytest.mark.parametrize(
        "extra_args",
        [["--chat-template-file", "/srv/chat.jinja"], ["--chat-template=chatml"], ["--no-jinja"]],
    )
    def test_a_gguf_template_chosen_by_extra_args_gets_no_system_turn(
        self, monkeypatch, extra_args
    ):
        from types import SimpleNamespace

        monkeypatch.undo()
        llama = SimpleNamespace(
            is_loaded = True, chat_template = _CHATML, chat_template_override = None, extra_args = []
        )
        monkeypatch.setattr(self.inference, "get_llama_cpp_backend", lambda: llama)
        assert self.inference._local_template_system_turn(_DAY) == (True, "")
        llama.extra_args = extra_args
        assert self.inference._local_template_system_turn(_DAY) == (False, None)

    def test_a_tool_request_probes_the_tool_use_template(self, monkeypatch):
        from types import SimpleNamespace

        from core.inference import orchestrator

        monkeypatch.undo()
        named = [
            {"name": "default", "template": _CHATML},
            {"name": "tool_use", "template": _REFUSES_SYSTEM},
        ]
        info = {"chat_template_info": {"template": named}}
        backend = SimpleNamespace(active_model_name = "hermes", models = {"hermes": info})
        monkeypatch.setattr(
            self.inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
        )
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
        assert self.inference._local_template_system_turn(_DAY) == (True, "")
        assert self.inference._local_template_system_turn(_DAY, tools = True) == (False, None)

    def test_a_text_request_probes_the_mapped_template(self, monkeypatch):
        from types import SimpleNamespace

        from core.inference import orchestrator

        monkeypatch.undo()
        info = {"chat_template_info": {"template": _CHATML, "mapped_template": _QWEN25_LIKE}}
        backend = SimpleNamespace(active_model_name = "mapped", models = {"mapped": info})
        monkeypatch.setattr(
            self.inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
        )
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
        assert self.inference._local_template_system_turn(_DAY) == (
            True,
            "You are Qwen, a helpful assistant.",
        )
        assert self.inference._local_template_system_turn(_DAY, True) == (True, "")

    def test_an_image_request_probes_the_processor_template(self, monkeypatch):
        from types import SimpleNamespace

        from core.inference import orchestrator

        monkeypatch.undo()
        info = {"chat_template_info": {"template": _CHATML, "processor_template": _QWEN25_LIKE}}
        backend = SimpleNamespace(active_model_name = "vlm", models = {"vlm": info})
        monkeypatch.setattr(
            self.inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
        )
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
        assert self.inference._local_template_system_turn(_DAY) == (True, "")
        assert self.inference._local_template_system_turn(_DAY, True) == (
            True,
            "You are Qwen, a helpful assistant.",
        )
        info["chat_template_info"]["processor_template"] = _REFUSES_SYSTEM
        assert self.inference._local_template_system_turn(_DAY, True) == (False, None)

    def test_a_named_template_list_is_probed_through_its_default(self, monkeypatch):
        from types import SimpleNamespace

        from core.inference import orchestrator

        monkeypatch.undo()
        named = [
            {"name": "default", "template": _QWEN25_LIKE},
            {"name": "tool_use", "template": _CHATML},
        ]
        backend = SimpleNamespace(
            active_model_name = "hermes",
            models = {"hermes": {"chat_template_info": {"template": named}}},
        )
        monkeypatch.setattr(
            self.inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
        )
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)
        assert self.inference._local_template_system_turn(_DAY) == (
            True,
            "You are Qwen, a helpful assistant.",
        )

    def test_a_chat_started_days_ago_keeps_every_user_turn_verbatim(self):
        history = [
            {"role": "user", "content": "Mike and Alexis are in bed."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Turn the story back to dinner."},
        ]
        out = self.inference._prepend_current_date_to_messages(history, object())
        assert out == [{"role": "system", "content": "The current date is 2026-10-04."}, *history]

    def test_a_disabled_setting_adds_nothing(self, monkeypatch):
        monkeypatch.setattr(self.inference, "current_date_prompt_line", lambda **_kwargs: "")
        assert self.inference._apply_current_date_prompt("", object()) == ""
