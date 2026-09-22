# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.systemone import catalog, laya_runtime
from routes import systemone
from utils import systemone_settings

_REAL_LOAD = laya_runtime._load_checkpoint
_REAL_TRUNCATED = laya_runtime._state_truncated
_REAL_OWNER_SETTING = systemone_settings._owner_setting
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges, refunds"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def _laya_answers(questions):
    answers = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            answers[name] = {
                "type": "noul",
                "noul": 0.9,
                "confidence": 0.9,
                "action": {"act_probability": 0.1},
            }
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            answers[name] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {k: 1.0 if i == 0 else 0.0 for i, k in enumerate(keys)},
                "confidence": 1.0,
                "action": {"act_probability": 0.0},
            }
        else:
            answers[name] = {
                "type": "score",
                "score": 1.5,
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {
                    str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))
                },
                "confidence": 0.2,
                "action": {"act_probability": 0.3},
            }
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


class FakeAgent:
    def __init__(self, name = "agent"):
        self.name = name
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return _laya_answers(questions)


@pytest.fixture(autouse = True)
def runtime(monkeypatch):
    for name, value in (
        ("_agent", None),
        ("_loaded", None),
        ("_device_name", None),
        ("_loader", None),
        ("_loading", None),
        ("_failure", None),
    ):
        monkeypatch.setattr(laya_runtime, name, value)
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    import storage.studio_db as studio_db

    settings = {systemone_settings.ENABLED_KEY: True}
    monkeypatch.setattr(systemone_settings, "_owner_setting", settings.get)
    monkeypatch.setattr(studio_db, "upsert_app_settings", settings.update)
    loads = []

    def load(checkpoint):
        loads.append(checkpoint.name)
        return FakeAgent(checkpoint.name), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", load)
    monkeypatch.setattr(laya_runtime, "_state_truncated", lambda *args: False)
    yield loads
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(5)


@pytest.fixture
def client():
    app = FastAPI()
    from routes.settings import router as settings_router

    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _post(client, **body):
    body = {
        "state": "Everything is down and we have a demo at noon.",
        "model": "jev-latest",
        "questions": QUESTIONS,
        **body,
    }
    return client.post("/v1/systemone", json = body)


def test_answers_in_jev_wire_format(client, runtime):
    response = _post(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "laya-multilingual"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.9}
    assert body["answers"]["team"] == {
        "type": "choice",
        "choice": "outage",
        "confidence": 1.0,
        "probabilities": {"outage": 1.0, "billing": 0.0},
    }
    tone = body["answers"]["tone"]
    assert set(tone) == {"type", "score", "confidence", "legend", "probabilities"}
    assert tone["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert "X-Unsloth-State-Truncated" not in response.headers
    assert runtime == ["laya-multilingual"]


def test_model_loaded_once_and_questions_forwarded(client, runtime):
    _post(client)
    _post(client, state = {"subject": "Charged twice"}, questions = {"q": {"type": "noul"}})
    assert runtime == ["laya-multilingual"]
    state, questions = laya_runtime._agent.calls[-1]
    assert state == {"subject": "Charged twice"}
    assert questions == {"q": {"type": "noul", "instructions": ""}}


def test_named_checkpoint_swaps_the_resident_model(client, runtime):
    _post(client)
    response = _post(client, model = "laya-english")
    assert response.json()["model"] == "laya-english"
    assert runtime == ["laya-multilingual", "laya-english"]
    assert laya_runtime.status()["loaded_model"] == "laya-english"


@pytest.mark.parametrize(
    "body, status, error_type",
    [
        ({"model": "gpt-4"}, 400, "api_usage_error"),
        ({"questions": {"q": {"type": "maybe"}}}, 400, "api_usage_error"),
        ({"questions": {"q": {"type": "choice"}}}, 400, "invalid_request_error"),
        ({"questions": {"q": {"type": "choice", "criteria": {}}}}, 400, "invalid_request_error"),
        (
            {"questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
            400,
            "invalid_request_error",
        ),
        (
            {"questions": {"q": {"type": "noul", "criteria": {"yes": "x"}}}},
            400,
            "invalid_request_error",
        ),
        ({"images": ["data:image/png;base64,AAAA"]}, 400, "api_usage_error"),
        (
            {"questions": {f"q{i}": {"type": "noul"} for i in range(65)}},
            400,
            "invalid_request_error",
        ),
    ],
)
def test_requests_the_model_cannot_answer_are_refused(client, runtime, body, status, error_type):
    response = _post(client, **body)
    assert response.status_code == status, response.text
    assert response.json()["detail"]["error_type"] == error_type
    assert runtime == []


@pytest.mark.parametrize(
    "body", [{"state": 3}, {"questions": {}}, {"questions": {"q": {"type": "noul", "extra": 1}}}]
)
def test_malformed_requests_are_validation_errors(client, body):
    assert _post(client, **body).status_code == 422


def test_laya_refusal_becomes_a_bad_request(client, monkeypatch):
    def refuse(state, questions):
        raise ValueError("question 'team' options exceed head_max_len=256")

    monkeypatch.setattr(
        laya_runtime, "_load_checkpoint", lambda c: (SimpleNamespace(predict = refuse), "cpu")
    )
    response = _post(client)
    assert response.status_code == 400
    assert "head_max_len" in response.json()["detail"]["message"]


def test_truncated_state_is_flagged(client, monkeypatch):
    monkeypatch.setattr(laya_runtime, "_state_truncated", lambda *args: True)
    assert _post(client).headers["X-Unsloth-State-Truncated"] == "1"


def test_failed_load_backs_off_and_reports_why(client, monkeypatch):
    attempts = []

    def fail(checkpoint):
        attempts.append(checkpoint.name)
        raise FileNotFoundError("Laya checkpoint at /x is missing encoder/")

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", fail)
    first = _post(client)
    assert first.status_code == 503
    assert "missing encoder" in first.json()["detail"]["message"]
    assert int(first.headers["Retry-After"]) >= 1
    assert _post(client).status_code == 503
    assert attempts == ["laya-multilingual"]


def test_missing_package_points_at_studio_update(client, monkeypatch):
    def missing(checkpoint):
        raise ImportError("No module named 'laya'")

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", missing)
    response = _post(client)
    assert response.status_code == 503
    assert "unsloth studio update" in response.json()["detail"]["message"]


def test_slow_load_answers_retry_after_instead_of_hanging(client, monkeypatch):
    release = threading.Event()

    def slow(checkpoint):
        release.wait(5)
        return FakeAgent(), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", slow)
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 0.05)
    response = _post(client)
    assert response.status_code == 503
    assert response.json()["detail"]["error_type"] == "model_loading"
    assert response.headers["Retry-After"] == "5"
    release.set()
    laya_runtime._loader.join(5)
    assert _post(client).status_code == 200


def test_busy_model_answers_overloaded(client, monkeypatch):
    _post(client)
    monkeypatch.setattr(laya_runtime, "RUN_WAIT_S", 0.05)
    with laya_runtime._run_lock:
        response = _post(client)
    assert response.status_code == 529
    assert response.json()["detail"]["error_type"] == "overloaded"


def test_off_by_default_and_says_where_to_turn_it_on(client, monkeypatch, runtime):
    monkeypatch.setattr(systemone_settings, "_owner_setting", {}.get)
    response = _post(client)
    assert response.status_code == 404
    assert "Settings > API" in response.json()["detail"]["message"]
    assert runtime == []


def test_env_kill_switch_beats_the_setting(client, monkeypatch, runtime):
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_DISABLE", "1")
    assert _post(client).status_code == 404
    settings = client.get("/api/settings/systemone").json()
    assert settings["enabled"] is False and settings["enabled_locked"] is True
    assert client.put("/api/settings/systemone", json = {"enabled": True}).status_code == 400
    assert runtime == []


def test_settings_report_choices_and_residency(client):
    settings = client.get("/api/settings/systemone").json()
    assert settings["enabled"] is True
    assert settings["model"] == "laya-multilingual"
    assert settings["device"] == "cpu"
    assert settings["loaded_model"] is None
    assert [m["name"] for m in settings["models"]] == list(catalog.CHECKPOINTS)
    assert all(m["download_bytes"] > 0 for m in settings["models"])
    _post(client)
    settings = client.get("/api/settings/systemone").json()
    assert settings["loaded_model"] == "laya-multilingual"
    assert settings["loaded_device"] == "cpu"
    unloaded = client.post("/api/settings/systemone/unload").json()
    assert unloaded["loaded_model"] is None


def test_changing_settings_drops_the_resident_model(client, runtime):
    _post(client)
    response = client.put(
        "/api/settings/systemone", json = {"model": "laya-english", "device": "gpu"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["model"], body["device"], body["loaded_model"]) == ("laya-english", "gpu", None)
    assert _post(client).json()["model"] == "laya-english"
    assert runtime == ["laya-multilingual", "laya-english"]


def test_turning_it_off_unloads_and_refuses(client, runtime):
    _post(client)
    body = client.put("/api/settings/systemone", json = {"enabled": False}).json()
    assert body["enabled"] is False and body["loaded_model"] is None
    assert _post(client).status_code == 404


@pytest.mark.parametrize("payload", [{"model": "gpt-4"}, {"device": "tpu"}])
def test_invalid_settings_are_refused_without_evicting_the_model(client, payload):
    _post(client)
    assert client.put("/api/settings/systemone", json = payload).status_code == 400
    assert client.get("/api/settings/systemone").json()["loaded_model"] == "laya-multilingual"


def test_env_model_is_locked(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", "laya-english")
    settings = client.get("/api/settings/systemone").json()
    assert settings["model"] == "laya-english" and settings["model_locked"] is True
    assert (
        client.put("/api/settings/systemone", json = {"model": "laya-multilingual"}).status_code
        == 400
    )


def test_settings_change_waits_for_a_running_load(client, monkeypatch):
    release = threading.Event()

    def slow(checkpoint):
        release.wait(5)
        return FakeAgent(), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", slow)
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 0.05)
    assert _post(client).status_code == 503
    assert client.put("/api/settings/systemone", json = {"model": "laya-english"}).status_code == 409
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"
    release.set()
    laya_runtime._loader.join(5)


def test_download_plan_lists_exact_subfolder_files(client, monkeypatch):
    import huggingface_hub

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    tree = [
        Entry("model.safetensors", 9),
        Entry("multilingual/model.safetensors", 600),
        Entry("multilingual/rl_agent_config.json", 1),
        Entry("multilingual/tokenizer/tokenizer.json", 30),
        Entry("multilingual/encoder/config.json", 2),
        Entry("multilingual/README.md", 5),
        Entry("typed-decisions/model.safetensors", 800),
    ]
    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: False)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", lambda self, repo, **kw: tree)
    plan = client.get("/api/settings/systemone/resolve").json()
    assert plan == {
        "repo": catalog.LAYA_REPO,
        "files": [
            "multilingual/encoder/config.json",
            "multilingual/model.safetensors",
            "multilingual/rl_agent_config.json",
            "multilingual/tokenizer/tokenizer.json",
        ],
        "size_bytes": 633,
        "cached": False,
        "error": None,
    }


def test_download_plan_for_a_cached_model_skips_the_hub(client, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    monkeypatch.setattr(
        huggingface_hub.HfApi, "list_repo_tree", lambda *a, **k: pytest.fail("hub listing")
    )
    plan = client.get("/api/settings/systemone/resolve").json()
    assert plan["cached"] is True and plan["files"] == []


def test_is_cached_needs_weights_and_folders(monkeypatch, tmp_path):
    checkpoint = catalog.Checkpoint("laya-local", str(tmp_path), None, "")
    assert laya_runtime.is_cached(checkpoint) is False
    for name in ("encoder", "tokenizer"):
        (tmp_path / name).mkdir()
    assert laya_runtime.is_cached(checkpoint) is False
    for name in ("model.safetensors", "rl_agent_config.json"):
        (tmp_path / name).write_text("x")
    assert laya_runtime.is_cached(checkpoint) is True


def test_local_checkpoint_is_served_under_its_own_name(client, monkeypatch, runtime, tmp_path):
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", str(tmp_path))
    assert _post(client).json()["model"] == catalog.LOCAL_NAME
    assert _post(client, model = catalog.LOCAL_NAME).status_code == 200
    assert runtime == [catalog.LOCAL_NAME]


def test_misspelled_model_setting_is_not_fetched_from_the_hub(client, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("hub download")
    )
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", "laya-multilingul")
    response = _post(client)
    assert response.status_code == 503
    assert "neither a known model nor a directory" in response.json()["detail"]["message"]


def test_oversized_state_is_refused(client, runtime):
    response = _post(client, state = "x" * (systemone.MAX_STATE_CHARS + 1))
    assert response.status_code == 400
    assert runtime == []


def test_waiting_requests_are_bounded(client, monkeypatch):
    monkeypatch.setattr(laya_runtime, "_admission", threading.BoundedSemaphore(1))
    with laya_runtime._admission:
        response = _post(client)
    assert response.status_code == 529
    assert response.headers["Retry-After"] == "1"
    assert _post(client).status_code == 200


def test_local_checkpoint_needs_encoder_and_tokenizer(tmp_path):
    checkpoint = catalog.Checkpoint("laya-local", str(tmp_path), None, "")
    with pytest.raises(FileNotFoundError, match = "encoder/, tokenizer/"):
        laya_runtime._checkpoint_dir(checkpoint)
    for name in ("encoder", "tokenizer"):
        (tmp_path / name).mkdir()
    assert laya_runtime._checkpoint_dir(checkpoint) == tmp_path


def test_hub_checkpoint_downloads_only_its_subfolder_into_studio_cache(monkeypatch, tmp_path):
    import huggingface_hub

    from utils import hf_cache_settings

    for name in ("encoder", "tokenizer"):
        (tmp_path / "multilingual" / name).mkdir(parents = True)
    seen = {}

    def snapshot(repo, **kwargs):
        seen.update(kwargs, repo = repo)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: "/cache/hub")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    assert laya_runtime._checkpoint_dir(catalog.CHECKPOINTS["laya-multilingual"]) == tmp_path
    assert seen["repo"] == catalog.LAYA_REPO
    assert seen["cache_dir"] == "/cache/hub"
    assert seen["local_files_only"] is False
    assert all(p.startswith("multilingual/") for p in seen["allow_patterns"])


def test_device_defaults_to_cpu():
    assert laya_runtime._device() == "cpu"


def test_managed_account_reads_the_owner_switch(monkeypatch):
    import storage.studio_db as studio_db
    from utils import account_context

    readers = []

    def read(key, fallback = None):
        readers.append(account_context.current_account().account_id)
        return True

    monkeypatch.setattr(systemone_settings, "_owner_setting", _REAL_OWNER_SETTING)
    monkeypatch.setattr(studio_db, "get_app_setting", read)
    token = account_context.bind_account(account_context.AccountContext("alice", "alice"))
    try:
        assert systemone_settings.get_enabled() is True
        assert account_context.current_account().account_id == "alice"
    finally:
        account_context.reset_account(token)
    assert readers == [account_context.OWNER_ACCOUNT_ID]


def test_studio_app_serves_systemone_routes():
    from main import app

    # FastAPI keeps included routers as one entry each, so route through the app instead of listing it.
    client = TestClient(app)
    body = {"state": "x", "model": "jev-latest", "questions": {"q": {"type": "noul"}}}
    assert client.post("/v1/systemone", json = body).status_code == 401
    assert client.get("/api/settings/systemone").status_code == 401
    assert client.put("/api/settings/systemone", json = {"enabled": True}).status_code == 401
    assert client.post("/api/settings/systemone/unload").status_code == 401


def test_keyless_inference_scope_covers_systemone():
    from utils.keyless_api_access import scope_covers
    assert scope_covers("inference", "POST", "/v1/systemone")
    assert not scope_covers("inference", "PUT", "/api/settings/systemone")


def test_real_laya_answers_through_the_route(client, monkeypatch):
    path = os.environ.get("SYSTEMONE_TEST_LAYA")
    if not path:
        pytest.skip("set SYSTEMONE_TEST_LAYA to a downloaded convaiinnovations/laya snapshot")
    pytest.importorskip("laya")
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_state_truncated", _REAL_TRUNCATED)
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", path)
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_SUBFOLDER", "multilingual")
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 600.0)
    response = _post(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == catalog.LOCAL_NAME
    assert 0.0 <= body["answers"]["urgent"]["noul"] <= 1.0
    assert body["answers"]["team"]["choice"] in {"outage", "billing"}
    assert 0.0 <= body["answers"]["tone"]["score"] <= 2.0
    assert "X-Unsloth-State-Truncated" not in response.headers
    assert _post(client, state = "word " * 3000).headers.get("X-Unsloth-State-Truncated") == "1"
