# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import cloudflare_tunnel  # noqa: E402
import routes.settings as routes  # noqa: E402
import utils.public_access_settings as public_access  # noqa: E402
from storage import studio_db  # noqa: E402


def _state(intent = "unset", is_colab = False):
    return SimpleNamespace(
        public_access_intent = intent,
        public_access_is_colab = is_colab,
        public_access_port = 8888,
        public_access_ready = True,
    )


def test_auto_start_persistence_is_strict_and_fail_closed(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback: stored.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", lambda values: stored.update(values))
    assert public_access.get_public_access_auto_start() is False
    stored[public_access.PUBLIC_ACCESS_AUTO_START_KEY] = "yes"
    assert public_access.get_public_access_auto_start() is False
    assert public_access.set_public_access_auto_start(True) is True
    assert public_access.get_public_access_auto_start() is True
    with pytest.raises(ValueError):
        routes.PublicAccessAutoStartPayload(enabled = "true")
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda *args: (_ for _ in ()).throw(OSError())
    )
    assert public_access.get_public_access_auto_start() is False


def test_workers_and_stops_are_scoped_to_backend_lifecycle(monkeypatch):
    public_access._start_worker = public_access._stop_worker = None
    public_access._start_worker_admission = public_access._stop_worker_admission = None
    cloudflare_tunnel.stop_studio_tunnel()
    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    entered, reopened, release = (threading.Event() for _ in range(3))
    attempts = []
    real_start = cloudflare_tunnel.start_studio_tunnel

    def _delayed(*args, **kwargs):
        attempts.append(kwargs["admission"])
        if len(attempts) == 1:
            entered.set()
            release.wait(1)
            return real_start(*args, **kwargs)
        reopened.set()

    monkeypatch.setattr(cloudflare_tunnel, "start_studio_tunnel", _delayed)
    monkeypatch.setattr(cloudflare_tunnel, "ensure_cloudflared", lambda: pytest.fail("stale start"))
    monkeypatch.setattr(public_access, "get_public_access_auto_start", lambda: False)
    monkeypatch.setattr(public_access, "_admin_password_ready", lambda: True)
    public_access.start_public_access(_state())
    assert entered.wait(1)
    old_worker = public_access._start_worker
    old_token = attempts[0]
    cloudflare_tunnel.close_studio_tunnel_lifecycle()
    cloudflare_tunnel.open_studio_tunnel_lifecycle()
    monkeypatch.setattr(public_access, "get_public_access_auto_start", lambda: True)
    assert public_access.maybe_auto_start_public_access(_state()) and reopened.wait(1)
    release.set()
    old_worker.join(1)
    current = cloudflare_tunnel.get_studio_tunnel_control_token()
    cloudflare_tunnel.stop_studio_tunnel(admission = old_token)
    assert len(attempts) == 2 and cloudflare_tunnel.get_studio_tunnel_control_token() == current
    assert cloudflare_tunnel.get_studio_tunnel_status()["state"] == "off"


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_request_cannot_adopt_reopened_lifecycle(monkeypatch, operation):
    status = (
        {"state": "off", "managed_by": None, "can_start": True}
        if operation == "start"
        else {"state": "online", "managed_by": "settings"}
    )
    satisfied = {
        "state": "online" if operation == "start" else "stopping",
        "managed_by": "settings",
    }
    cloudflare_tunnel.open_studio_tunnel_lifecycle()

    def _status(_state):
        cloudflare_tunnel.close_studio_tunnel_lifecycle()
        cloudflare_tunnel.open_studio_tunnel_lifecycle()
        return {"block_reason": None, **status}

    monkeypatch.setattr(public_access, "public_access_status", _status)
    with pytest.raises(RuntimeError, match = "server_lifecycle_changed"):
        getattr(public_access, f"{operation}_public_access")(_state())
    monkeypatch.setattr(
        public_access,
        "public_access_status",
        lambda _: (cloudflare_tunnel.stop_studio_tunnel(), satisfied)[1],
    )
    assert getattr(public_access, f"{operation}_public_access")(_state()) == satisfied


def test_management_rejects_api_keys():
    with pytest.raises(HTTPException) as exc:
        routes._require_ui_session(True)
    assert exc.value.status_code == 403
    assert public_access.public_access_status(_state())["streaming_supported"] is True
    source = Path(routes.__file__).read_text()
    assert source.count("_ui_session: None = Depends(_require_ui_session)") == 4


def test_colab_auto_start_setting_is_read_only(monkeypatch):
    monkeypatch.setattr(routes, "set_public_access_auto_start", lambda *_: pytest.fail("persisted"))
    request = SimpleNamespace(app = SimpleNamespace(state = _state(is_colab = True)))
    payload = routes.PublicAccessAutoStartPayload(enabled = True)
    with pytest.raises(HTTPException) as exc:
        routes.update_public_access_auto_start(request, payload, "admin", None)
    assert exc.value.status_code == 409
