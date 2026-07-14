# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for Helper LLM startup pre-cache opt-in behavior."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.datasets import AiAssistMappingRequest
from routes import datasets as datasets_route
from routes import settings as settings_route
from utils import helper_precache_settings


def _install_fake_studio_db(monkeypatch, *, stored = None):
    storage_pkg = types.ModuleType("storage")
    studio_db = types.ModuleType("storage.studio_db")
    values: dict[str, object] = {}
    if stored is not None:
        values[helper_precache_settings.HELPER_PRECACHE_SETTING_KEY] = stored

    def get_app_setting(key, fallback = None):
        return values.get(key, fallback)

    def upsert_app_settings(settings):
        values.update(settings)
        return dict(values)

    studio_db.get_app_setting = get_app_setting
    studio_db.upsert_app_settings = upsert_app_settings
    monkeypatch.setitem(sys.modules, "storage", storage_pkg)
    monkeypatch.setitem(sys.modules, "storage.studio_db", studio_db)
    return values


def test_helper_precache_defaults_off_when_setting_missing(monkeypatch):
    monkeypatch.delenv("UNSLOTH_HELPER_MODEL_DISABLE", raising = False)
    _install_fake_studio_db(monkeypatch)

    assert helper_precache_settings.get_helper_precache_enabled() is False
    assert helper_precache_settings.should_preload_helper_on_startup() is False


def test_helper_precache_opt_in_is_blocked_by_existing_disable_env(monkeypatch):
    _install_fake_studio_db(monkeypatch, stored = True)
    monkeypatch.setenv("UNSLOTH_HELPER_MODEL_DISABLE", "true")

    assert helper_precache_settings.get_helper_precache_enabled() is True
    assert helper_precache_settings.should_preload_helper_on_startup() is False


def test_settings_route_persists_helper_precache_toggle(monkeypatch):
    values = _install_fake_studio_db(monkeypatch)
    monkeypatch.delenv("UNSLOTH_HELPER_MODEL_DISABLE", raising = False)

    response = settings_route.update_helper_precache(
        settings_route.HelperPrecachePayload(enabled = True),
        current_subject = "test-user",
    )

    assert response.enabled is True
    assert response.default_enabled is False
    assert response.disabled_by_env is False
    assert values[helper_precache_settings.HELPER_PRECACHE_SETTING_KEY] is True


def test_main_startup_uses_helper_precache_gate_instead_of_unconditional_precache():
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding = "utf-8")
    startup_section = source[
        source.index("cleanup_orphaned_runs") : source.index("# Initialize RSA key pair")
    ]

    assert "_start_helper_precache_if_enabled()" in startup_section
    assert "precache_helper_gguf" not in startup_section
    assert "threading.Thread(target = _precache" not in startup_section


def test_ai_assist_route_still_calls_on_demand_advisor(monkeypatch):
    calls: list[dict] = []
    llm_assist = types.ModuleType("utils.datasets.llm_assist")

    def fake_llm_conversion_advisor(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "suggested_mapping": {"prompt": "user", "answer": "assistant"},
            "system_prompt": "Answer carefully.",
            "dataset_type": "question_answering",
            "is_conversational": False,
            "user_notification": "Columns mapped by AI Assist.",
        }

    llm_assist.llm_conversion_advisor = fake_llm_conversion_advisor
    monkeypatch.setitem(sys.modules, "utils.datasets.llm_assist", llm_assist)

    response = datasets_route.ai_assist_mapping(
        AiAssistMappingRequest(
            columns = ["prompt", "answer"],
            samples = [{"prompt": "x" * 250, "answer": "ok", "extra": "ignored"}],
            dataset_name = "owner/dataset",
            hf_token = "hf_test",
            model_name = "unsloth/test",
            model_type = "text",
        ),
        current_subject = "test-user",
    )

    assert response.success is True
    assert response.suggested_mapping == {"prompt": "user", "answer": "assistant"}
    assert response.system_prompt == "Answer carefully."
    assert calls == [
        {
            "column_names": ["prompt", "answer"],
            "samples": [{"prompt": "x" * 200, "answer": "ok"}],
            "dataset_name": "owner/dataset",
            "hf_token": "hf_test",
            "model_name": "unsloth/test",
            "model_type": "text",
        }
    ]
