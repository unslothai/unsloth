# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth train --notify-webhook` completion notification."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "studio" / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub matplotlib (a Studio runtime dep) so importing the notifier stays light.
_mpl = types.ModuleType("matplotlib")
_mpl.use = lambda *a, **k: None
_plt = types.ModuleType("matplotlib.pyplot")
_plt.Figure = type("Figure", (), {})
sys.modules.setdefault("matplotlib", _mpl)
sys.modules.setdefault("matplotlib.pyplot", _plt)

import studio.backend.core.training.notifications as notif  # noqa: E402
from unsloth_cli.commands import train as train_mod  # noqa: E402


class _Resp:
    def raise_for_status(self):
        return None


def test_no_webhook_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(notif.httpx, "post", lambda *a, **k: calls.append(1) or _Resp())
    train_mod._send_cli_notification(None, "m", "completed")
    train_mod._send_cli_notification("", "m", "completed")
    assert calls == []


def test_sends_completed_event_with_autodetected_slack_shape(monkeypatch):
    sent = {}

    def fake_post(
        url,
        json = None,
        timeout = None,
    ):
        sent["url"] = url
        sent["json"] = json
        return _Resp()

    monkeypatch.setattr(notif.httpx, "post", fake_post)
    train_mod._send_cli_notification(
        "https://hooks.slack.com/services/x",
        "unsloth/m",
        "completed",
        total_steps = 60,
        final_loss = 0.4,
        duration_s = 120.0,
    )
    assert sent["url"] == "https://hooks.slack.com/services/x"
    assert "text" in sent["json"]  # slack shape, auto-detected from the URL
    assert "Training complete" in sent["json"]["text"]


def test_sends_error_event(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        notif.httpx,
        "post",
        lambda url, json = None, timeout = None: sent.update(json = json) or _Resp(),
    )
    # A setup failure (no progress object) still notifies via an explicit error.
    train_mod._send_cli_notification(
        "https://webhook.site/x", "m", "error", error = "Model preparation failed"
    )
    assert sent["json"]["status"] == "error"
    assert "Model preparation failed" in sent["json"]["message"]


def test_swallows_delivery_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(notif.httpx, "post", boom)
    # A dead webhook must never crash the training command.
    train_mod._send_cli_notification("https://webhook.site/x", "m", "completed")
