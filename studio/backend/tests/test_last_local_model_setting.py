# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level tests for the last-local-model setting endpoints."""

from pathlib import Path
import time
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
import storage.studio_db as studio_db


@pytest.fixture
def client(monkeypatch):
    store: dict = {}

    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda values: store.update(values) or store
    )

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), store


def test_get_with_nothing_stored(client):
    c, _ = client
    r = c.get("/last-local-model")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.pop("server_now"), int)
    assert body == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}


def test_put_then_get_round_trips(client):
    c, store = client
    payload = {"id": "unsloth/gemma-4-E2B-it-GGUF", "kind": "gguf", "gguf_variant": "UD-Q4_K_XL"}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    body = r.json()
    body.pop("server_now")
    assert body == {**payload, "loaded_at": None}
    assert store[settings._last_local_model_key("admin")] == {**payload, "loaded_at": None}

    r = c.get("/last-local-model")
    assert r.status_code == 200
    body = r.json()
    body.pop("server_now")
    assert body == {**payload, "loaded_at": None}


def test_put_accepts_path_qualified_variant(client):
    c, _ = client
    variant = "quants/" + "a" * 80 + "/model-UD-Q4_K_XL-00001-of-00002.gguf"
    payload = {"id": "unsloth/gemma-4-E2B-it-GGUF", "kind": "gguf", "gguf_variant": variant}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    assert r.json()["gguf_variant"] == variant


def test_put_round_trips_loaded_at(client):
    c, _ = client
    payload = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 1765432100000}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    assert r.json()["loaded_at"] == 1765432100000
    assert c.get("/last-local-model").json()["loaded_at"] == 1765432100000


def test_put_without_variant(client):
    c, _ = client
    r = c.put("/last-local-model", json = {"id": "unsloth/Qwen3-4B", "kind": "model"})
    assert r.status_code == 200
    body = r.json()
    body.pop("server_now")
    assert body == {
        "id": "unsloth/Qwen3-4B",
        "kind": "model",
        "gguf_variant": None,
        "loaded_at": None,
    }


def test_put_ignores_stale_timestamped_write(client):
    c, _ = client
    newer = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 2000}
    stale = {
        "id": "unsloth/gemma-4-E2B-it-GGUF",
        "kind": "gguf",
        "gguf_variant": "UD-Q4_K_XL",
        "loaded_at": 1000,
    }
    assert c.put("/last-local-model", json = newer).status_code == 200
    r = c.put("/last-local-model", json = stale)
    assert r.status_code == 200
    assert r.json()["id"] == "unsloth/Qwen3-4B"
    assert c.get("/last-local-model").json()["loaded_at"] == 2000


def test_put_unstamped_write_keeps_last_write_wins(client):
    c, _ = client
    stamped = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 2000}
    assert c.put("/last-local-model", json = stamped).status_code == 200
    assert (
        c.put("/last-local-model", json = {"id": "unsloth/OLMo-4-13B", "kind": "model"}).status_code
        == 200
    )
    assert c.get("/last-local-model").json()["id"] == "unsloth/OLMo-4-13B"


def test_put_clamps_future_dated_stamps(client):
    c, _ = client
    import time as _time

    r = c.put(
        "/last-local-model",
        json = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 9_999_999_999_999},
    )
    assert r.status_code == 200
    cap = int(_time.time() * 1000) + settings._LAST_LOCAL_MODEL_CLOCK_SLACK_MS
    assert r.json()["loaded_at"] <= cap


def test_put_normalizes_slow_client_clocks(client):
    c, store = client
    import time as _time

    now = int(_time.time() * 1000)
    assert (
        c.put(
            "/last-local-model",
            json = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": now, "client_now": now},
        ).status_code
        == 200
    )
    # a clock hours behind: a genuinely later load still lands after the stored one
    slow_now = now - 7_200_000
    r = c.put(
        "/last-local-model",
        json = {
            "id": "unsloth/OLMo-4-13B",
            "kind": "model",
            "loaded_at": slow_now + 60_000,
            "client_now": slow_now,
        },
    )
    assert r.status_code == 200
    assert c.get("/last-local-model").json()["id"] == "unsloth/OLMo-4-13B"
    assert "client_now" not in store[settings._last_local_model_key("admin")]


def test_put_rejects_bad_payloads(client):
    c, _ = client
    assert c.put("/last-local-model", json = {"id": "x", "kind": "lora"}).status_code == 422
    assert c.put("/last-local-model", json = {"id": "", "kind": "gguf"}).status_code == 422
    assert c.put("/last-local-model", json = {"kind": "gguf"}).status_code == 422


def test_get_tolerates_corrupt_stored_value(client):
    c, store = client
    store[settings.LAST_LOCAL_MODEL_SETTING_KEY] = {"id": "x", "kind": "lora"}
    r = c.get("/last-local-model")
    assert r.status_code == 200
    body = r.json()
    body.pop("server_now")
    assert body == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}

    store[settings.LAST_LOCAL_MODEL_SETTING_KEY] = "not-a-dict"
    r = c.get("/last-local-model")
    assert r.status_code == 200
    body = r.json()
    body.pop("server_now")
    assert body == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}


# ── per-subject scoping ─────────────────────────────────────────────


@pytest.fixture
def multi_subject_client(monkeypatch):
    """One backend and one store, with a swappable authenticated subject."""
    store: dict = {}
    subject = {"value": "alice"}

    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda values: store.update(values) or store
    )
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: subject["value"]
    return TestClient(app, raise_server_exceptions = False), store, subject


def test_one_subject_does_not_inherit_anothers_model(multi_subject_client):
    c, _, subject = multi_subject_client
    subject["value"] = "alice"
    c.put("/last-local-model", json = {"id": "alice/model", "kind": "model"})

    subject["value"] = "bob"
    assert c.get("/last-local-model").json()["id"] is None

    c.put("/last-local-model", json = {"id": "bob/model", "kind": "model"})
    assert c.get("/last-local-model").json()["id"] == "bob/model"

    subject["value"] = "alice"
    assert c.get("/last-local-model").json()["id"] == "alice/model"


def test_an_upgraded_install_still_sees_the_shared_row(multi_subject_client):
    """Pre-scoping installs stored one shared record; a subject with no row of
    its own inherits it rather than booting with nothing remembered."""
    c, store, subject = multi_subject_client
    store[settings.LAST_LOCAL_MODEL_SETTING_KEY] = {
        "id": "unsloth/gemma-4-E2B-it-GGUF",
        "kind": "gguf",
        "gguf_variant": "UD-Q4_K_XL",
        "loaded_at": 1000,
    }
    subject["value"] = "alice"
    assert c.get("/last-local-model").json()["id"] == "unsloth/gemma-4-E2B-it-GGUF"

    # Once the subject writes it owns its row and stops following the shared one.
    c.put("/last-local-model", json = {"id": "alice/model", "kind": "model", "loaded_at": 2000})
    assert c.get("/last-local-model").json()["id"] == "alice/model"
    assert store[settings.LAST_LOCAL_MODEL_SETTING_KEY]["id"] == "unsloth/gemma-4-E2B-it-GGUF"


def test_subject_keys_do_not_collide(multi_subject_client):
    _, _, _ = multi_subject_client
    keys = {settings._last_local_model_key(s) for s in ("a", "b", "a:b", "", "  ")}
    # "" and "  " degrade to the shared key; the rest are distinct.
    assert len(keys) == 4
    assert settings._last_local_model_key("") == settings.LAST_LOCAL_MODEL_SETTING_KEY


def test_a_delayed_put_is_dated_from_arrival_not_from_the_load(client):
    """Known limitation, pinned so any change is deliberate. client_now is
    stamped at send, so flight time is indistinguishable from clock skew: a
    retry-delayed PUT lands near arrival and can outrank a newer load."""
    c, store = client
    now = int(time.time() * 1000)

    # The newer load reaches the server first.
    c.put(
        "/last-local-model",
        json = {"id": "newer", "kind": "model", "loaded_at": now, "client_now": now},
    )
    assert c.get("/last-local-model").json()["id"] == "newer"

    # An older load whose PUT sat in a retry for 30s: the shift re-dates it to now.
    old = now - 30_000
    c.put(
        "/last-local-model",
        json = {"id": "older", "kind": "model", "loaded_at": old, "client_now": old},
    )
    assert c.get("/last-local-model").json()["id"] == "older"


def test_a_re_issued_old_shadow_stays_old(client):
    """The other half of the same shift: a shadow re-sent long after its load
    keeps its age, so it cannot displace a newer record."""
    c, _ = client
    now = int(time.time() * 1000)
    c.put(
        "/last-local-model",
        json = {"id": "newer", "kind": "model", "loaded_at": now, "client_now": now},
    )
    # Loaded 30s ago, re-issued now: age is preserved, so it loses.
    c.put(
        "/last-local-model",
        json = {"id": "stale", "kind": "model", "loaded_at": now - 30_000, "client_now": now},
    )
    assert c.get("/last-local-model").json()["id"] == "newer"
