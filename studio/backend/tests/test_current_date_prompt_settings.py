# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The current-date preference: its endpoints, and the prompts it feeds."""

from datetime import date
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
        monkeypatch.setattr(
            current_date_settings, "get_current_date_prompt_enabled", lambda: True
        )
        assert (
            current_date_settings.current_date_prompt_line(date(2026, 8, 15))
            == "The current date is 2026-08-15."
        )

    def test_system_prompt_helper_is_idempotent(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(
            inference, "current_date_prompt_line", lambda: "The current date is 2026-08-15."
        )
        already = "The current date is 2026-08-14.\n\nBASE"
        assert inference._apply_current_date_prompt(already) == already

    def test_line_is_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            current_date_settings, "get_current_date_prompt_enabled", lambda: False
        )
        assert current_date_settings.current_date_prompt_line(date(2026, 8, 15)) == ""

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
            inference, "current_date_prompt_line", lambda: "The current date is 2026-08-15."
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

    def test_multimodal_system_content_gets_its_own_turn(self):
        out = self._prepend(
            [{"role": "system", "content": [{"type": "text", "text": "Be terse."}]}]
        )
        assert out[0] == {"role": "system", "content": "The current date is 2026-08-15."}
        assert out[1]["content"] == [{"type": "text", "text": "Be terse."}]

    def test_developer_turn_is_used_when_there_is_no_system_turn(self):
        out = self._prepend([{"role": "developer", "content": "Be terse."}])
        assert out[0]["content"] == "The current date is 2026-08-15.\n\nBe terse."

    def test_messages_are_untouched_when_disabled(self, monkeypatch):
        import routes.inference as inference

        monkeypatch.setattr(inference, "current_date_prompt_line", lambda: "")
        messages = [{"role": "user", "content": "hi"}]
        assert self._prepend(messages) is messages

    def test_a_prompt_that_already_states_a_date_is_left_alone(self):
        # Deep Research stamps its own date at run creation, then posts the prompt back through
        # this route; a second line would contradict the first once the run crosses midnight.
        stamped = [
            {"role": "system", "content": "The current date is 2026-08-14.\n\nBe terse."},
            {"role": "user", "content": "hi"},
        ]
        assert self._prepend(stamped) is stamped

    def test_a_stated_date_buried_under_research_instructions_still_suppresses(self):
        # _system_prompt_with_instructions buries the date under a header, so a startswith
        # check would miss it and re-date the prompt.
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
        assert self._prepend(buried) is buried

    def test_only_self_hosted_provider_types_are_dated(self):
        from core.inference.providers import provider_is_self_hosted

        for provider_type in ("vllm", "ollama", "llama_cpp", "custom"):
            assert provider_is_self_hosted(provider_type), provider_type
        # Hosted APIs and Codex state the date in their own context already.
        for provider_type in (
            "openai",
            "openai_codex",
            "anthropic",
            "gemini",
            "deepseek",
            "mistral",
            "kimi",
            "qwen",
            "huggingface",
            "openrouter",
            None,
            ["vllm"],
        ):
            assert not provider_is_self_hosted(provider_type), provider_type


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
        assert prompt.endswith(
            "Non-overridable rules:\nThe current date is 2026-08-15.\n\nBASE"
        )

    def test_run_without_a_stamped_date_is_unchanged(self):
        # runs created before the field existed, and runs started with the setting off.
        assert _system_prompt_with_instructions("BASE", {}) == "BASE"
        assert _system_prompt_with_instructions("BASE", {"currentDate": ""}) == "BASE"
