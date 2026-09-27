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
        # Ollama applies the Modelfile SYSTEM only while messages[0] is not a system turn (#10436).
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
        # studio's own workflow keys are excluded too, not just third-party sk-unsloth callers.
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
        # runs created before the field existed, and runs started with the setting off.
        assert _system_prompt_with_instructions("BASE", {}) == "BASE"
        assert _system_prompt_with_instructions("BASE", {"currentDate": ""}) == "BASE"


class TestConversationStartDate:
    """A thread's system line keeps the day it started; later days ride on the newest user turn."""

    @staticmethod
    def _threads(monkeypatch, threads):
        import storage.studio_db as studio_db
        monkeypatch.setattr(studio_db, "get_chat_thread", lambda thread_id: threads.get(thread_id))

    def test_start_date_is_the_creation_day_in_the_browser_zone(self, monkeypatch):
        # 2026-08-15T02:00Z is still the 14th in New York.
        created = int(datetime(2026, 8, 15, 2, tzinfo = timezone.utc).timestamp() * 1000)
        self._threads(monkeypatch, {"t": {"createdAt": created}})
        request = _types.SimpleNamespace(headers = {"x-unsloth-timezone": "America/New_York"})
        assert current_date_settings.conversation_start_date("t", request) == date(2026, 8, 14)

    def test_a_fork_keeps_its_root_threads_start_date(self, monkeypatch):
        day = lambda d: int(datetime(2026, 8, d, 12, tzinfo = timezone.utc).timestamp() * 1000)
        self._threads(
            monkeypatch,
            {
                "root": {"createdAt": day(1)},
                "mid": {"createdAt": day(5), "forkedFromThreadId": "root"},
                "leaf": {"createdAt": day(9), "forkedFromThreadId": "mid"},
                "orphan": {"createdAt": day(9), "forkedFromThreadId": "gone"},
                "loop": {"createdAt": day(9), "forkedFromThreadId": "loop"},
            },
        )
        request = _types.SimpleNamespace(headers = {"x-unsloth-timezone": "UTC"})
        start = current_date_settings.conversation_start_date
        assert start("leaf", request) == date(2026, 8, 1)
        assert start("orphan", request) == date(2026, 8, 9)
        assert start("loop", request) == date(2026, 8, 9)

    def test_a_deep_fork_chain_still_reaches_the_root(self, monkeypatch):
        root = int(datetime(2026, 8, 1, 12, tzinfo = timezone.utc).timestamp() * 1000)
        later = int(datetime(2026, 8, 9, 12, tzinfo = timezone.utc).timestamp() * 1000)
        threads = {"t0": {"createdAt": root}}
        for i in range(1, 40):
            threads[f"t{i}"] = {"createdAt": later, "forkedFromThreadId": f"t{i - 1}"}
        self._threads(monkeypatch, threads)
        request = _types.SimpleNamespace(headers = {"x-unsloth-timezone": "UTC"})
        assert current_date_settings.conversation_start_date("t39", request) == date(2026, 8, 1)

    @pytest.mark.parametrize("thread_id", [None, "", 7, "missing"])
    def test_unknown_threads_have_no_start_date(self, monkeypatch, thread_id):
        self._threads(monkeypatch, {})
        assert current_date_settings.conversation_start_date(thread_id) is None

    def test_a_lookup_failure_has_no_start_date(self, monkeypatch):
        import storage.studio_db as studio_db

        def _boom(_thread_id):
            raise RuntimeError("db locked")

        monkeypatch.setattr(studio_db, "get_chat_thread", _boom)
        assert current_date_settings.conversation_start_date("t") is None


class TestDateChangeNote:
    @pytest.fixture(autouse = True)
    def _clock(self, monkeypatch):
        import routes.inference as inference

        self.today = "2026-08-15"
        self.started = date(2026, 8, 15)
        monkeypatch.setattr(
            inference,
            "current_date_prompt_line",
            lambda **_kwargs: f"The current date is {self.today}.",
        )
        monkeypatch.setattr(
            inference,
            "conversation_start_date",
            lambda thread_id, _request = None: self.started if thread_id == "t" else None,
        )
        monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: False)
        self.inference = inference

    def _proxy(
        self,
        messages,
        thread_id = "t",
        **kwargs,
    ):
        return self.inference._prepend_current_date_to_messages(
            messages, object(), thread_id = thread_id, **kwargs
        )

    def test_same_day_is_byte_identical_to_a_threadless_request(self):
        messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
        assert self._proxy(messages) == self._proxy(messages, thread_id = None)
        assert self.inference._apply_current_date_prompt("Be terse.", object(), thread_id = "t") == (
            "The current date is 2026-08-15.\n\nBe terse."
        )

    def test_midnight_keeps_the_system_turn_and_notes_the_newest_user_turn(self):
        history = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        before = self._proxy(history)
        self.today = "2026-08-16"
        after = self._proxy(history)
        assert after[0] == before[0]
        assert after[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."
        assert after[1:3] == history[1:3]
        assert after[3]["content"] == "second\n\nThe current date is now 2026-08-16."
        assert history[3]["content"] == "second"

    def test_next_request_after_midnight_shares_the_whole_prefix_up_to_the_new_note(self):
        self.today = "2026-08-16"
        turn1 = [{"role": "user", "content": "first"}]
        sent1 = self._proxy(turn1)
        turn2 = turn1 + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        sent2 = self._proxy(turn2)
        assert sent2[0] == sent1[0]
        # the earlier turn loses its note, but only after its own text.
        assert sent1[1]["content"].startswith(sent2[1]["content"])

    def test_threadless_requests_still_refresh_to_today(self):
        self.today = "2026-08-16"
        out = self._proxy([{"role": "user", "content": "hi"}], thread_id = None)
        assert out == [
            {"role": "system", "content": "The current date is 2026-08-16."},
            {"role": "user", "content": "hi"},
        ]

    def test_a_stated_system_date_is_refreshed_to_the_start_date(self):
        self.today = "2026-08-16"
        out = self._proxy(
            [
                {"role": "system", "content": "The current date is 2026-08-16."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert out[0]["content"] == "The current date is 2026-08-15."
        assert out[1]["content"] == "hi\n\nThe current date is now 2026-08-16."

    def test_note_goes_after_media_parts_and_skips_tool_results(self):
        self.today = "2026-08-16"
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        messages = [
            {"role": "user", "content": [image, {"type": "text", "text": "what is this"}]},
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x"}]},
        ]
        out = self.inference._append_current_date_note(messages, object(), thread_id = "t")
        assert out[2] is messages[2]
        assert out[0]["content"] == [
            image,
            {"type": "text", "text": "what is this"},
            {"type": "text", "text": "The current date is now 2026-08-16."},
        ]

    def test_note_skips_a_tool_result_folded_into_a_user_turn(self):
        from core.inference.anthropic_compat import fold_tool_results_into_user

        self.today = "2026-08-16"
        messages = fold_tool_results_into_user(
            [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "ls", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
            ]
        )
        out = self.inference._append_current_date_note(messages, object(), thread_id = "t")
        assert out[2] is messages[2]
        assert out[0]["content"] == "list files\n\nThe current date is now 2026-08-16."

    def test_note_is_not_added_twice(self):
        self.today = "2026-08-16"
        once = self.inference._append_current_date_note(
            [{"role": "user", "content": "hi"}], object(), thread_id = "t"
        )
        assert self.inference._append_current_date_note(once, object(), thread_id = "t") is once

    def test_api_key_callers_never_use_a_thread_start_date(self, monkeypatch):
        monkeypatch.setattr(self.inference, "_request_has_api_key", lambda _request: True)
        monkeypatch.setattr(self.inference, "_request_is_internal_workflow", lambda _r: False)
        self.today = "2026-08-16"
        out = self._proxy([{"role": "user", "content": "hi"}], include_api_key = True)
        assert out == [
            {"role": "system", "content": "The current date is 2026-08-16."},
            {"role": "user", "content": "hi"},
        ]

    def test_disabled_setting_adds_no_note(self, monkeypatch):
        monkeypatch.setattr(self.inference, "current_date_prompt_line", lambda **_kwargs: "")
        messages = [{"role": "user", "content": "hi"}]
        assert (
            self.inference._append_current_date_note(messages, object(), thread_id = "t") is messages
        )
