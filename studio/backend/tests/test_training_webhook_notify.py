# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for training-completion webhook notifications."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.training import notifications as notif
from routes import settings as settings_route
from utils import notification_settings


def _install_fake_studio_db(monkeypatch, *, stored = None):
    storage_pkg = types.ModuleType("storage")
    studio_db = types.ModuleType("storage.studio_db")
    values: dict[str, object] = {}
    if stored is not None:
        values[notification_settings.TRAINING_WEBHOOK_SETTING_KEY] = stored

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


def _event(**overrides):
    base = dict(
        job_id = "job_1",
        status = "completed",
        model = "unsloth/llama-3-8b",
        total_steps = 60,
        final_loss = 0.4213,
        duration_s = 125.0,
    )
    base.update(overrides)
    return notif.TrainingTerminalEvent(**base)


class _FakeResponse:
    def raise_for_status(self):
        return None


# ── Formatters ────────────────────────────────────────────────────────────


def test_slack_format_uses_text_key():
    body = notif.FORMATTERS["slack"](_event())
    assert set(body) == {"text"}
    assert "llama-3-8b" in body["text"]
    assert "Training complete" in body["text"]


def test_test_event_message_is_clearly_a_test():
    # The "Send test" sample must not look like a finished run.
    text = notif._summary(notif.TrainingTerminalEvent(job_id = "test", status = "test", model = ""))
    assert "connected" in text
    assert "Training complete" not in text and "Training failed" not in text


def test_discord_format_uses_content_key():
    body = notif.FORMATTERS["discord"](_event())
    assert set(body) == {"content"}
    assert "llama-3-8b" in body["content"]


def test_generic_format_carries_fields_and_message():
    body = notif.FORMATTERS["generic"](_event())
    assert body["status"] == "completed"
    assert body["model"] == "unsloth/llama-3-8b"  # HF id preserved
    assert body["total_steps"] == 60
    assert "message" in body


def test_error_event_includes_error_text():
    body = notif.FORMATTERS["slack"](_event(status = "error", error = "CUDA OOM"))
    assert "Training failed" in body["text"]
    assert "CUDA OOM" in body["text"]


def test_local_path_model_is_reduced_to_basename():
    body = notif.FORMATTERS["generic"](_event(model = "/Users/me/models/my-run"))
    assert body["model"] == "my-run"


# ── Format auto-detection from the URL ────────────────────────────────────


def test_detect_slack_url():
    assert notif.detect_format("https://hooks.slack.com/services/T/B/xyz") == "slack"


def test_detect_discord_url():
    assert notif.detect_format("https://discord.com/api/webhooks/1/abc") == "discord"
    assert notif.detect_format("https://discordapp.com/api/webhooks/1/abc") == "discord"


def test_detect_generic_for_other_hosts():
    assert notif.detect_format("https://webhook.site/abc") == "generic"
    assert notif.detect_format("https://ntfy.sh/my-topic") == "generic"


def test_detect_rejects_lookalike_hosts():
    # Substring matches and subdomain spoofs must NOT be treated as Slack/Discord.
    assert notif.detect_format("https://myslack.com/x") == "generic"
    assert notif.detect_format("https://hooks.slack.com.evil.test/x") == "generic"


# ── WebhookSink delivery (format derived from URL) ────────────────────────


def test_webhook_sink_posts_slack_shape_for_slack_url(monkeypatch):
    calls = []

    def fake_post(
        url,
        json = None,
        timeout = None,
    ):
        calls.append((url, json))
        return _FakeResponse()

    monkeypatch.setattr(notif.httpx, "post", fake_post)
    notif.WebhookSink("https://hooks.slack.com/services/x").deliver(_event())

    assert len(calls) == 1
    assert calls[0][0] == "https://hooks.slack.com/services/x"
    assert "text" in calls[0][1]  # slack shape, auto-detected


def test_webhook_sink_retries_then_raises(monkeypatch):
    calls = []

    def fake_post(
        url,
        json = None,
        timeout = None,
    ):
        calls.append(url)
        raise RuntimeError("boom")

    monkeypatch.setattr(notif.httpx, "post", fake_post)
    with pytest.raises(RuntimeError):
        notif.WebhookSink("https://webhook.site/x").deliver(_event())

    assert len(calls) == notif._WEBHOOK_ATTEMPTS


# ── TrainingNotifier ──────────────────────────────────────────────────────


class _RecordingSink:
    def __init__(self):
        self.events = []

    def deliver(self, event):
        self.events.append(event)


class _ExplodingSink:
    url = "https://hook.test/x"

    def deliver(self, event):
        raise RuntimeError("sink failure")


def test_notifier_delivers_to_sink():
    sink = _RecordingSink()
    event = _event()
    for future in notif.TrainingNotifier([sink]).emit(event):
        future.result(timeout = 5)
    assert sink.events == [event]


def test_notifier_swallows_sink_errors():
    # Must not raise even though the sink throws.
    for future in notif.TrainingNotifier([_ExplodingSink()]).emit(_event()):
        future.result(timeout = 5)


def test_notifier_with_no_sinks_is_noop():
    assert notif.TrainingNotifier([]).emit(_event()) == []


def test_get_training_notifier_empty_when_disabled(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    assert notif.get_training_notifier()._sinks == []


def test_get_training_notifier_builds_sink_when_enabled(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    notification_settings.set_training_webhook(True, "https://discord.com/api/webhooks/1/x")
    notifier = notif.get_training_notifier()
    assert len(notifier._sinks) == 1
    assert notifier._sinks[0].url == "https://discord.com/api/webhooks/1/x"
    assert notifier._sinks[0].fmt == "discord"  # derived from the URL


# ── Settings persistence ──────────────────────────────────────────────────


def test_webhook_settings_default_off(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    assert notification_settings.get_training_webhook() == {"enabled": False, "url": ""}


def test_webhook_settings_round_trip(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    notification_settings.set_training_webhook(True, "https://hook.test/abc")
    assert notification_settings.get_training_webhook() == {
        "enabled": True,
        "url": "https://hook.test/abc",
    }


def test_enabling_without_url_is_rejected(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    with pytest.raises(ValueError):
        notification_settings.set_training_webhook(True, "")


def test_invalid_url_is_rejected_when_enabling(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    with pytest.raises(ValueError):
        notification_settings.set_training_webhook(True, "not-a-url")


def test_disable_succeeds_with_partial_url(monkeypatch):
    # Toggling off must not be blocked by a partial/dirty URL draft.
    _install_fake_studio_db(monkeypatch)
    config = notification_settings.set_training_webhook(False, "https://")
    assert config == {"enabled": False, "url": "https://"}


# ── Settings route ────────────────────────────────────────────────────────


def test_settings_route_persists_webhook(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    response = settings_route.update_notifications(
        settings_route.TrainingWebhookPayload(enabled = True, url = "https://hook.test/route"),
        current_subject = "test-user",
    )
    assert response.enabled is True
    assert response.url == "https://hook.test/route"
    assert settings_route.get_notifications(current_subject = "test-user").enabled is True


def test_test_endpoint_rejected_when_disabled(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    # URL is saved but the toggle is off -> the test must not fire.
    notification_settings.set_training_webhook(False, "https://hook.test/x")
    with pytest.raises(Exception):
        settings_route.test_notifications(current_subject = "test-user")


def test_test_endpoint_sends_when_enabled(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    notification_settings.set_training_webhook(True, "https://hook.test/x")
    calls = []

    def fake_post(
        url,
        json = None,
        timeout = None,
    ):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(notif.httpx, "post", fake_post)
    response = settings_route.test_notifications(current_subject = "test-user")
    assert response.ok is True
    assert calls == ["https://hook.test/x"]
