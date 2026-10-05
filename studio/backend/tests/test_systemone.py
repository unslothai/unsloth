# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
import sys
import threading
import weakref
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.systemone import catalog, laya_runtime
from routes import systemone
from utils import systemone_settings

_REAL_LOAD = laya_runtime._load_checkpoint
_REAL_PREDICT = laya_runtime._predict
_REAL_OWNER_SETTING = systemone_settings._owner_setting
_REAL_UNAVAILABLE_REASON = systemone_settings.runtime_unavailable_reason
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
    monkeypatch.setattr(
        laya_runtime, "_predict", lambda agent, state, qs: (agent.predict(state, qs), False)
    )
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
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


def test_sdk_request_ids_are_present_and_unique(client):
    from uuid import UUID

    first = _post(client).headers["x-typesafe-request-id"]
    second = _post(client).headers["x-typesafe-request-id"]
    assert UUID(first).version == UUID(second).version == 4
    assert first != second


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
@pytest.mark.parametrize("value", [42, 1.5, False])
def test_invalid_criterion_values_are_rejected_before_loading(client, runtime, kind, value):
    criteria = [value, "high"] if kind == "score" else {"true": value, "false": None}
    response = _post(client, questions = {"q": {"type": kind, "criteria": criteria}})
    assert response.status_code == 422, response.text
    assert "q" in response.json()["detail"]["message"]
    assert "criteria" in response.json()["detail"]["message"]
    assert runtime == []


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_sdk_structured_and_null_criteria_remain_supported(client, kind):
    values = [None, "plain text", {"weight": 3}, ["nested", False]]
    for value in values:
        criteria = [value, "high"] if kind == "score" else {"true": value, "false": None}
        response = _post(client, questions = {"q": {"type": kind, "criteria": criteria}})
        assert response.status_code == 200, response.text
        assert laya_runtime._agent.calls[-1][1]["q"]["criteria"] == criteria
        if kind == "score":
            assert response.json()["answers"]["q"]["legend"]["0"] == value


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
        ({"questions": {"q": {"type": "maybe"}}}, 422, "api_usage_error"),
        ({"questions": {"q": {"type": "choice"}}}, 422, "invalid_request_error"),
        ({"questions": {"q": {"type": "choice", "criteria": {}}}}, 422, "invalid_request_error"),
        (
            {"questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
            422,
            "invalid_request_error",
        ),
        (
            {"questions": {"q": {"type": "noul", "criteria": {"yes": "x"}}}},
            422,
            "invalid_request_error",
        ),
        ({"images": ["data:image/png;base64,AAAA"]}, 400, "api_usage_error"),
        (
            {"questions": {f"q{i}": {"type": "noul"} for i in range(65)}},
            422,
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


def test_laya_refusal_becomes_a_validation_error(client, monkeypatch):
    def refuse(state, questions):
        raise ValueError("question 'team' options exceed head_max_len=256")

    monkeypatch.setattr(
        laya_runtime, "_load_checkpoint", lambda c: (SimpleNamespace(predict = refuse), "cpu")
    )
    response = _post(client)
    assert response.status_code == 422
    assert "head_max_len" in response.json()["detail"]["message"]


def test_truncated_state_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        laya_runtime, "_predict", lambda agent, state, qs: (agent.predict(state, qs), True)
    )
    response = _post(client)
    assert response.status_code == 422
    assert "context window" in response.json()["detail"]["message"]
    assert "shorten" in response.json()["detail"]["message"].lower()
    assert "answers" not in response.json()


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


def test_failed_load_logs_the_underlying_cause(client, monkeypatch, caplog):
    def fail(checkpoint):
        try:
            raise RuntimeError("HIP error: invalid device function")
        except RuntimeError as exc:
            raise ModuleNotFoundError("Could not import module 'AutoTokenizer'") from exc

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", fail)
    with caplog.at_level("WARNING", logger = laya_runtime.__name__):
        response = _post(client)
    assert response.status_code == 503
    assert "HIP error" not in response.json()["detail"]["message"]
    [record] = [r for r in caplog.records if r.getMessage().startswith("System One load failed")]
    assert "HIP error: invalid device function" in caplog.text
    assert record.exc_info is not None


def test_settings_never_report_an_install(client):
    body = client.get("/api/settings/systemone").json()
    assert body["installing"] is False and body["error"] is None


def test_the_vendored_laya_wins_over_an_installed_one(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from pathlib import Path

    for name in [n for n in sys.modules if n == "laya" or n.startswith("laya.")]:
        monkeypatch.delitem(sys.modules, name)
    decoy = tmp_path / "laya"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("raise ImportError('installed laya was imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    laya = laya_runtime._laya()
    assert laya_runtime._laya() is laya and sys.modules["laya"] is laya
    assert Path(laya.__file__).parent == laya_runtime._VENDORED_LAYA
    assert Path(sys.modules["laya.common"].__file__).parent == laya_runtime._VENDORED_LAYA
    assert callable(laya.load) and callable(laya.common.collate_items)


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


def test_settings_update_refuses_a_stale_consent_snapshot(client):
    response = client.put(
        "/api/settings/systemone",
        json = {
            "model": "laya-english",
            "expected_enabled": False,
            "expected_model": "laya-multilingual",
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Decision API settings changed. Try again."
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"


def test_settings_validation_checks_the_snapshot_without_saving(client):
    response = client.post(
        "/api/settings/systemone/validate",
        json = {
            "model": "laya-english",
            "expected_enabled": True,
            "expected_model": "laya-multilingual",
        },
    )
    assert response.status_code == 204
    settings = client.get("/api/settings/systemone").json()
    assert settings["enabled"] is True
    assert settings["model"] == "laya-multilingual"
    updated = client.put(
        "/api/settings/systemone",
        json = {
            "model": "laya-english",
            "expected_enabled": True,
            "expected_model": "laya-multilingual",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["model"] == "laya-english"


def test_env_model_is_locked(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", "laya-english")
    settings = client.get("/api/settings/systemone").json()
    assert settings["model"] == "laya-english" and settings["model_locked"] is True
    assert (
        client.put("/api/settings/systemone", json = {"model": "laya-multilingual"}).status_code
        == 400
    )


def test_stale_check_uses_the_display_name_for_a_local_env_model(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", str(tmp_path))
    settings = client.get("/api/settings/systemone").json()
    assert settings["model"] == "laya-local"
    response = client.post(
        "/api/settings/systemone/validate",
        json = {
            "enabled": True,
            "expected_enabled": True,
            "expected_model": "laya-local",
        },
    )
    assert response.status_code == 204


def test_settings_change_waits_for_a_running_load(client, monkeypatch):
    release = threading.Event()

    def slow(checkpoint):
        release.wait(5)
        return FakeAgent(), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", slow)
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 0.05)
    assert _post(client).status_code == 503
    assert (
        client.post(
            "/api/settings/systemone/validate",
            json = {
                "model": "laya-english",
                "expected_enabled": True,
                "expected_model": "laya-multilingual",
            },
        ).status_code
        == 409
    )
    assert client.put("/api/settings/systemone", json = {"model": "laya-english"}).status_code == 409
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"
    release.set()
    laya_runtime._loader.join(5)


def test_turning_it_off_waits_for_a_claimed_load(client, monkeypatch):
    probing, release = threading.Event(), threading.Event()

    def probe(checkpoint):
        probing.set()
        release.wait(5)
        return False

    monkeypatch.setattr(laya_runtime, "_hub_download_active", probe)
    claim = threading.Thread(
        target = laya_runtime._ensure_loading, args = (catalog.default_checkpoint(),)
    )
    claim.start()
    assert probing.wait(5)
    assert client.put("/api/settings/systemone", json = {"enabled": False}).status_code == 409
    release.set()
    claim.join(5)
    laya_runtime._loader.join(5)
    assert client.get("/api/settings/systemone").json()["enabled"] is True


def test_keyless_callers_only_reach_the_configured_model(client, monkeypatch, runtime):
    import auth.authentication

    monkeypatch.setattr(auth.authentication, "request_admitted_without_credential", lambda r: True)
    refused = _post(client, model = "laya-english")
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "permission_error"
    assert _post(client, model = "laya-multilingual").status_code == 200
    assert _post(client).status_code == 200


def test_unload_returns_cached_device_memory(monkeypatch):
    freed = []
    fake_torch = SimpleNamespace(
        cuda = SimpleNamespace(is_initialized = lambda: True, empty_cache = lambda: freed.append("cuda")),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    laya_runtime.unload()
    assert freed == ["cuda"]


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


def test_download_plan_can_preview_a_model_without_changing_the_setting(client, monkeypatch):
    import huggingface_hub

    seen = []

    def list_tree(self, repo, **kwargs):
        seen.append((repo, kwargs))
        return []

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: False)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", list_tree)
    plan = client.get("/api/settings/systemone/resolve?model=laya-english")
    assert plan.status_code == 200
    assert plan.json()["repo"] == catalog.LAYA_REPO
    assert plan.json()["size_bytes"] == catalog.CHECKPOINTS["laya-english"].download_bytes
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"
    assert seen == [(catalog.LAYA_REPO, {"recursive": True})]


def test_download_plan_refuses_an_unknown_preview_model(client):
    response = client.get("/api/settings/systemone/resolve?model=not-a-model")
    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown Decision API model."


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
    assert response.status_code == 422
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


def test_gpu_on_apple_silicon_runs_mlx_and_falls_back_to_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from utils.hardware import hardware

    monkeypatch.setenv("UNSLOTH_SYSTEMONE_DEVICE", "gpu")
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.MLX)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    real = laya_runtime._mlx_available
    monkeypatch.setattr(laya_runtime, "_mlx_available", lambda: True)
    assert laya_runtime._device() == "mlx"
    monkeypatch.setattr(laya_runtime, "_mlx_available", real)
    # An unsloth-zoo release without the module.
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.decision", None)
    assert laya_runtime._device() == "mps"


def test_mlx_answers_match_laya_on_cpu(monkeypatch):
    path = os.environ.get("SYSTEMONE_TEST_LAYA")
    if not path:
        pytest.skip("set SYSTEMONE_TEST_LAYA to a downloaded convaiinnovations/laya snapshot")
    laya = laya_runtime._laya()
    if not laya_runtime._mlx_available():
        pytest.skip("needs unsloth_zoo.mlx.decision on Apple Silicon")
    from pathlib import Path

    reference = laya.load(path, subfolder = "multilingual", device = "cpu")
    monkeypatch.setattr(laya_runtime, "_checkpoint_dir", lambda checkpoint: Path(path))
    monkeypatch.setattr(laya_runtime, "_device", lambda: "mlx")
    agent, device = _REAL_LOAD(catalog.CHECKPOINTS["laya-multilingual"])
    assert isinstance(agent, laya_runtime._MLXAgent) and device == "mlx"
    for state in (
        "Everything is down and we have a demo at noon.",
        "Hola, ¿me pueden devolver el dinero? " * 80,
    ):
        expected, _ = _REAL_PREDICT(reference, state, QUESTIONS)
        got, _ = _REAL_PREDICT(agent, state, QUESTIONS)
        assert got["usage"] == expected["usage"]
        for name, answer in expected["answers"].items():
            if answer["type"] == "noul":
                assert got["answers"][name]["noul"] == pytest.approx(answer["noul"], abs = 5e-3)
            else:
                assert got["answers"][name]["probabilities"] == pytest.approx(
                    answer["probabilities"], abs = 5e-3
                )


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


def test_switching_models_keeps_serving_until_the_new_checkpoint_is_on_disk(client, monkeypatch):
    assert _post(client).status_code == 200
    fetching = threading.Event()
    release = threading.Event()

    def download(checkpoint, **kwargs):
        fetching.set()
        release.wait(5)
        raise OSError("connection reset")

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_checkpoint_dir", download)
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 0.05)
    assert _post(client, model = "laya-english").status_code == 503
    assert fetching.wait(5)
    assert _post(client).status_code == 200
    release.set()
    laya_runtime._loader.join(5)
    assert _post(client).status_code == 200
    assert laya_runtime.status()["loaded_model"] == "laya-multilingual"


def test_load_waits_for_a_hub_download_of_the_same_repo(client, monkeypatch, runtime):
    from hub.utils import download_registry

    registry = download_registry.get_models_registry()
    monkeypatch.setattr(
        registry, "active_job_refs", lambda repo = None: ["job"] if repo == catalog.LAYA_REPO else []
    )
    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: False)
    response = _post(client)
    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "laya-multilingual is downloading"
    assert runtime == []
    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    assert _post(client).status_code == 200


def test_hub_download_is_refused_while_laya_loads_the_repo(client, monkeypatch):
    from hub.services.models import downloads

    release = threading.Event()

    def slow(checkpoint):
        release.wait(5)
        return FakeAgent(), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", slow)
    monkeypatch.setattr(laya_runtime, "LOAD_WAIT_S", 0.05)
    assert not downloads._load_in_flight(catalog.LAYA_REPO)
    assert _post(client).status_code == 503
    assert downloads._load_in_flight("ConvAIInnovations/Laya")
    release.set()
    laya_runtime._loader.join(5)
    assert not downloads._load_in_flight(catalog.LAYA_REPO)


def test_errors_keep_the_jev_shape_behind_studio_error_handlers(monkeypatch):
    from utils.api_errors import install_api_error_handlers

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(systemone.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)
    response = _post(client, model = "gpt-4")
    assert response.status_code == 400
    assert response.json()["detail"]["error_type"] == "api_usage_error"
    malformed = _post(client, state = 3)
    assert malformed.status_code == 422
    assert malformed.json()["detail"][0]["loc"][:2] == ["body", "state"]


def test_state_limit_counts_characters_not_json_escapes(client):
    text = "\u4f60" * (systemone.MAX_STATE_CHARS // 2)
    assert _post(client, state = text).status_code == 200
    assert _post(client, state = {"message": text}).status_code == 200


def test_load_probes_the_download_registry_without_holding_runtime_state(client, monkeypatch):
    from hub.utils import download_registry

    registry = download_registry.get_models_registry()
    seen, blocked = [], []

    def active_job_refs(repo = None):
        probe = threading.Thread(target = lambda: seen.append(laya_runtime.loading_repo_ids()))
        probe.start()
        probe.join(2)
        blocked.append(probe.is_alive())
        return []

    monkeypatch.setattr(registry, "active_job_refs", active_job_refs)
    assert _post(client).status_code == 200
    assert blocked == [False]
    assert seen == [(catalog.LAYA_REPO,)]


def test_decision_api_cannot_be_enabled_where_laya_is_not_installed(client, monkeypatch):
    settings = {}
    monkeypatch.setattr(systemone_settings, "_owner_setting", settings.get)
    import storage.studio_db as studio_db

    monkeypatch.setattr(studio_db, "upsert_app_settings", settings.update)
    reason = "The Decision API needs PyTorch, which this Studio install does not include."
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: reason)
    preview = client.post(
        "/api/settings/systemone/validate",
        json = {
            "enabled": True,
            "expected_enabled": False,
            "expected_model": "laya-multilingual",
        },
    )
    assert preview.status_code == 400
    assert preview.json()["detail"] == reason
    assert settings == {}
    response = client.put("/api/settings/systemone", json = {"enabled": True})
    assert response.status_code == 400
    assert response.json()["detail"] == reason
    assert client.get("/api/settings/systemone").json()["enabled"] is False
    assert client.put("/api/settings/systemone", json = {"enabled": False}).status_code == 200


def test_runtime_reason_names_what_the_install_lacks(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a)
    )
    expected = "Python 3.10" if sys.version_info < (3, 10) else "PyTorch"
    assert expected in _REAL_UNAVAILABLE_REASON()


def test_settings_drop_a_load_failure_once_its_backoff_expires(client, monkeypatch):
    def fail(checkpoint):
        raise OSError("connection reset")

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", fail)
    assert _post(client).status_code == 503
    assert "connection reset" in client.get("/api/settings/systemone").json()["error"]
    checkpoint, message, _deadline = laya_runtime._failure
    monkeypatch.setattr(laya_runtime, "_failure", (checkpoint, message, 0.0))
    assert client.get("/api/settings/systemone").json()["error"] is None


def test_settings_only_report_a_failure_of_the_selected_model(client, monkeypatch):
    def fail(checkpoint):
        raise OSError("connection reset")

    assert _post(client).status_code == 200
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", fail)
    assert _post(client, model = "laya-english").status_code == 503
    body = client.get("/api/settings/systemone").json()
    assert body["error"] is None and body["loaded_model"] == "laya-multilingual"
    laya_runtime.unload()
    assert _post(client).status_code == 503
    assert "connection reset" in client.get("/api/settings/systemone").json()["error"]


def test_route_module_imports_without_pep604_aliases():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(systemone))
    runtime_unions = [
        node
        for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        for node in ast.walk(stmt.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
    ]
    assert runtime_unions == []


def test_oversized_question_text_is_refused_before_the_model(client, runtime):
    questions = {"q": {"type": "noul", "instructions": "x" * systemone.MAX_QUESTION_CHARS}}
    response = _post(client, questions = questions)
    assert response.status_code == 422
    assert "longer than" in response.json()["detail"]["message"]
    criteria = {f"option {i}": "y" * 200 for i in range(200)}
    assert (
        _post(client, questions = {"q": {"type": "choice", "criteria": criteria}}).status_code == 422
    )
    assert runtime == []
    assert _post(client).status_code == 200


def test_real_laya_answers_through_the_route(client, monkeypatch):
    path = os.environ.get("SYSTEMONE_TEST_LAYA")
    if not path:
        pytest.skip("set SYSTEMONE_TEST_LAYA to a downloaded convaiinnovations/laya snapshot")
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_predict", _REAL_PREDICT)
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
    long_response = _post(client, state = "word " * 3000)
    assert long_response.status_code == 422
    assert "context window" in long_response.json()["detail"]["message"]
    options_response = _post(
        client,
        questions = {
            "route": {
                "type": "choice",
                "instructions": "Choose the best option",
                "criteria": {f"option {i} zero one": None for i in range(255)},
            },
        },
    )
    assert options_response.status_code == 422
    message = options_response.json()["detail"]["message"]
    assert "route" in message and "context window" in message and "criteria" in message
    assert "answers" not in options_response.json()
    assert _post(client).status_code == 200


class WordTokenizer:
    mask_token = "[MASK]"

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        text,
        add_special_tokens = False,
    ):
        self.calls.append(len(text))
        return {"input_ids": [len(word) for word in text.split()]}


def test_long_state_is_tokenized_only_as_far_as_the_model_reads():
    pytest.importorskip("torch")  # laya imports torch
    tok = WordTokenizer()
    ids, cut = laya_runtime._state_ids(tok, "word " * 100_000, 900)
    assert (ids, cut) == ([4] * 900, True)
    assert max(tok.calls) < 100_000


@pytest.mark.parametrize("words, cut", [(900, False), (901, True), (5000, True)])
def test_state_cut_matches_the_full_tokenization(words, cut):
    pytest.importorskip("torch")
    ids, was_cut = laya_runtime._state_ids(WordTokenizer(), "word " * words, 900)
    assert (len(ids), was_cut) == (min(words, 900), cut)


class MovableModel:
    def __init__(self):
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device.type)
        return self

    def float(self):
        return self


@pytest.fixture
def gpu_agent(monkeypatch):
    torch = pytest.importorskip("torch")
    # A CPU without native fp16/bf16: the fallback lands in fp32 whatever this host supports.
    monkeypatch.setattr(laya_runtime, "_precision", lambda device, fp16_checkpoint: (None, None))
    collate = lambda groups, pad: {"attention_mask": torch.ones(1, len(groups[0][0]["ids"]))}
    monkeypatch.setitem(
        sys.modules, "laya", SimpleNamespace(common = SimpleNamespace(collate_items = collate))
    )
    return SimpleNamespace(
        device = torch.device("cuda"),
        dtype = torch.float16,
        model = MovableModel(),
        tok = SimpleNamespace(pad_token_id = 0),
    )


def _items():
    return [{"ids": [1, 5, 2], "markers": [1], "qtype": 0}]


def test_gpu_out_of_memory_falls_back_to_cpu_like_laya(monkeypatch, gpu_agent):
    import torch

    agent = gpu_agent
    ran_on = []

    def run(agent, batch):
        ran_on.append(agent.device.type)
        if agent.device.type != "cpu":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return torch.tensor([[0.5, 1.5]])

    monkeypatch.setattr(laya_runtime, "_run_model", run)
    monkeypatch.setattr(laya_runtime, "_device_name", "cuda:0")
    released = []
    monkeypatch.setattr(laya_runtime, "_release_memory", lambda: released.append(ran_on[:]))
    logits, tokens = laya_runtime._forward(agent, _items())
    assert logits.tolist() == [[0.5, 1.5]] and tokens == 3
    assert ran_on == ["cuda", "cpu"]
    assert released == [["cuda"]]
    assert (agent.device.type, agent.dtype, agent.model.moved_to) == ("cpu", torch.float32, ["cpu"])
    assert laya_runtime._device_name == "cpu"
    laya_runtime._forward(agent, _items())
    assert ran_on == ["cuda", "cpu", "cpu"]


def test_unrelated_gpu_errors_are_not_retried_on_cpu(monkeypatch, gpu_agent):
    agent = gpu_agent

    def run(agent, batch):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(laya_runtime, "_run_model", run)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        laya_runtime._forward(agent, _items())
    assert agent.device.type == "cuda" and agent.model.moved_to == []


@pytest.mark.parametrize("where", ["mx", "mx.metal", None])
def test_release_memory_clears_the_mlx_cache_under_either_name(monkeypatch, where):
    cleared = []
    clear = SimpleNamespace(clear_cache = lambda: cleared.append(where))
    mx = {"mx": clear, "mx.metal": SimpleNamespace(metal = clear), None: SimpleNamespace()}[where]
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    laya_runtime._release_memory()
    assert cleared == ([where] if where else [])


def test_mlx_out_of_memory_falls_back_to_cpu(monkeypatch, gpu_agent):
    import torch

    class Model:
        def __init__(self, message):
            self.message = message

        def logits(self, batch):
            raise RuntimeError(self.message)

    cpu = SimpleNamespace(model = MovableModel(), device = torch.device("cpu"), dtype = torch.float32)
    loads, mlx_models, cleared = [], [], []

    def load(path, device):
        # The MLX model and MLX's buffer cache are both released before the CPU copy loads.
        assert mlx_models[-1]() is None and cleared
        loads.append((path, device))
        if len(loads) == 1:
            raise MemoryError
        return cpu

    def run_out_of_memory(message):
        agent.model = Model(message)
        mlx_models.append(weakref.ref(agent.model))
        cleared.clear()
        monkeypatch.setattr(laya_runtime, "_agent", agent)
        monkeypatch.setattr(laya_runtime, "_loaded", "checkpoint")
        return laya_runtime._forward(agent, _items())

    monkeypatch.setattr(sys.modules["laya"], "load", load, raising = False)
    monkeypatch.setitem(
        sys.modules, "mlx.core", SimpleNamespace(clear_cache = lambda: cleared.append(True))
    )
    ran_on = []
    monkeypatch.setattr(
        laya_runtime,
        "_run_model",
        lambda agent, batch: ran_on.append(agent.device.type) or torch.tensor([[0.5, 1.5]]),
    )
    monkeypatch.setattr(laya_runtime, "_device_name", "mlx")
    agent = laya_runtime._MLXAgent.__new__(laya_runtime._MLXAgent)
    agent.folder, agent.tok, agent.model = "ckpt", gpu_agent.tok, Model("shape mismatch")
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        laya_runtime._forward(agent, _items())
    assert loads == [] and agent.device == "mlx"

    with pytest.raises(MemoryError):
        run_out_of_memory("[malloc] Unable to allocate 2147483648 bytes.")
    assert (laya_runtime._agent, laya_runtime._loaded, laya_runtime._device_name) == (
        None,
        None,
        None,
    )

    logits, tokens = run_out_of_memory(
        "[METAL] Command buffer execution failed: Insufficient Memory."
    )
    assert logits.tolist() == [[0.5, 1.5]] and tokens == 3
    assert loads == [("ckpt", "cpu")] * 2
    assert (agent.model, agent.device.type, agent.dtype, laya_runtime._device_name) == (
        cpu.model,
        "cpu",
        torch.float32,
        "cpu",
    )
    laya_runtime._forward(agent, _items())
    assert ran_on == ["cpu", "cpu"] and len(loads) == 2


def test_fast_path_matches_laya_predict():
    path = os.environ.get("SYSTEMONE_TEST_LAYA")
    if not path:
        pytest.skip("set SYSTEMONE_TEST_LAYA to a downloaded convaiinnovations/laya snapshot")
    laya = laya_runtime._laya()
    agent = laya.load(path, subfolder = "multilingual", device = "cpu")
    questions = {name: laya_runtime._to_laya(q) for name, q in QUESTIONS.items()}
    questions["zeta"] = {
        "type": "choice",
        "instructions": "Pick one",
        "criteria": {"zeta": None, "alpha": "first", "mid": ""},
    }
    for state in (
        "I was charged twice.",
        {"turns": ["hi", "refund please"]},
        "Über 请 word " * 20_000,
    ):
        expected = agent.predict(state, questions)
        result, truncated = _REAL_PREDICT(agent, state, questions)
        assert result["usage"] == expected["usage"]
        for name, answer in result["answers"].items():
            want = {k: v for k, v in expected["answers"][name].items() if k != "action"}
            assert answer == want
            if "probabilities" in answer:
                assert list(answer["probabilities"]) == list(want["probabilities"])
        assert truncated == (len(state) > 200 if isinstance(state, str) else False)


def _save_checkpoint(folder, dtype):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    folder.mkdir(parents = True, exist_ok = True)
    save_file(
        {"big.weight": torch.zeros(64, 64, dtype = dtype), "temperature": torch.ones(3)},
        str(folder / "model.safetensors"),
    )
    return folder


def test_stored_dtype_is_read_from_the_safetensors_header(tmp_path):
    torch = pytest.importorskip("torch")
    assert laya_runtime._stored_fp16(_save_checkpoint(tmp_path / "half", torch.float16))
    assert not laya_runtime._stored_fp16(_save_checkpoint(tmp_path / "full", torch.float32))
    assert not laya_runtime._stored_fp16(tmp_path / "missing")


def test_precision_follows_the_checkpoint_then_the_device(monkeypatch):
    torch = pytest.importorskip("torch")
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", None)
    assert laya_runtime._precision(cuda, True) == (torch.float16, torch.float16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (7, 5))
    assert laya_runtime._precision(cuda, False) == (torch.float16, torch.float16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (9, 0))
    assert laya_runtime._precision(cuda, False) == (torch.bfloat16, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_FP32", "1")
    assert laya_runtime._precision(cuda, True) == (None, None)
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_FP32")
    import platform

    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert laya_runtime._precision(torch.device("cpu"), True) == (None, None)
    assert laya_runtime._precision(torch.device("mps"), True) == (None, None)


def _tiny_decision_model(torch):
    nn = torch.nn
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 32)
            self.norm = nn.LayerNorm(32)
            self.layer = nn.TransformerEncoderLayer(
                32, 4, 64, 0.0, batch_first = True, norm_first = True
            )
            self.out = nn.Linear(32, 1)

        def forward(self, ids):
            return self.out(self.layer(self.norm(self.emb(ids)))).float()

    model = Tiny().eval()
    # Weights exactly representable in fp16, as a checkpoint saved in fp16 loads into fp32.
    for param in model.parameters():
        param.data = param.data.half().float()
    return model


def test_fp16_weights_are_bit_identical_under_bf16_autocast():
    torch = pytest.importorskip("torch")
    import copy

    reference = _tiny_decision_model(torch)
    half = copy.deepcopy(reference)
    laya_runtime._cast_matmul_weights(half, torch.float16)
    assert (
        half.emb.weight.dtype == half.out.weight.dtype == half.layer.self_attn.in_proj_weight.dtype
    )
    assert half.emb.weight.dtype == torch.float16
    assert half.norm.weight.dtype == half.layer.norm1.weight.dtype == torch.float32
    ids = torch.randint(0, 50, (3, 17))
    assert half.emb(ids).dtype == torch.float32
    with torch.inference_mode(), torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        assert torch.equal(reference(ids), half(ids))


def test_fp16_overflow_reruns_in_fp32(monkeypatch, gpu_agent):
    torch = pytest.importorskip("torch")
    agent = gpu_agent
    calls = []

    def run(agent, batch):
        calls.append(agent.dtype)
        value = float("inf") if agent.dtype == torch.float16 else 1.0
        return torch.tensor([[value, 0.5]])

    monkeypatch.setattr(laya_runtime, "_run_model", run)
    logits, _ = laya_runtime._forward(agent, _items())
    assert calls == [torch.float16, torch.float32] and agent.dtype == torch.float32
    assert logits.tolist() == [[1.0, 0.5]]


def _tiny_laya_encoder(tmp_path):
    pytest.importorskip("torch")
    from transformers import ModernBertConfig

    config = ModernBertConfig(
        vocab_size = 300,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        pad_token_id = 0,
        bos_token_id = 2,
        eos_token_id = 1,
        cls_token_id = 1,
        sep_token_id = 1,
        mask_token_id = 4,
        global_attn_every_n_layers = 3,
        local_attention = 16,
    )
    encoder_dir = tmp_path / "encoder"
    config.save_pretrained(encoder_dir)
    return str(encoder_dir), {"encoder": "tiny", "head_layers": 1, "act_costs": {"escalate": 0.5}}


def test_skip_init_build_matches_laya_after_loading(tmp_path):
    torch = pytest.importorskip("torch")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    reference = laya.common.build_model(cfg, encoder_dir = encoder_dir).eval()
    fast = laya_runtime._build_model(cfg, encoder_dir, laya.common.build_model).eval()
    embedding = fast.encoder.get_input_embeddings()
    assert embedding.num_embeddings == 300 and fast.encoder.config.vocab_size == 300
    assert embedding.padding_idx == reference.encoder.get_input_embeddings().padding_idx
    # laya loads strictly, so every parameter and persistent buffer comes from the checkpoint.
    fast.load_state_dict(reference.state_dict(), strict = True)
    ours = dict(fast.named_parameters()) | dict(fast.named_buffers())
    theirs = dict(reference.named_parameters()) | dict(reference.named_buffers())
    assert ours.keys() == theirs.keys()
    # Includes the rotary inv_freq buffers, which the checkpoint does not carry.
    assert any("inv_freq" in name for name in theirs)
    assert all(torch.equal(ours[name], theirs[name]) for name in theirs)
    assert repr(fast) == repr(reference)
    assert fast.encoder.config.to_dict() == reference.encoder.config.to_dict()
    ids = torch.randint(5, 300, (2, 20))
    args = (
        ids,
        torch.ones_like(ids),
        torch.tensor([[3, 7]] * 2),
        torch.ones(2, 2, dtype = torch.bool),
        torch.zeros(2, dtype = torch.long),
    )
    with torch.inference_mode():
        assert torch.equal(fast(*args)[0], reference(*args)[0])


def test_skip_init_embedding_is_built_in_the_serving_dtype(tmp_path):
    torch = pytest.importorskip("torch")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    fast = laya_runtime._build_model(cfg, encoder_dir, laya.common.build_model, torch.float16)
    assert fast.encoder.get_input_embeddings().weight.dtype == torch.float16
    assert fast.encoder.embeddings.norm.weight.dtype == torch.float32


def test_build_without_an_encoder_dir_is_laya_own(tmp_path):
    pytest.importorskip("torch")
    sentinel = object()
    calls = []

    def original(cfg, encoder_dir = None):
        calls.append(encoder_dir)
        return sentinel

    assert laya_runtime._build_model({}, None, original) is sentinel
    assert laya_runtime._build_model({}, str(tmp_path / "missing"), original) is sentinel
    assert calls == [None, str(tmp_path / "missing")]


def test_fallback_build_releases_the_speculative_encoder(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import gc

    from transformers import ModernBertModel

    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    # An input embedding that is not a plain nn.Embedding sends the build back to laya's own builder.
    monkeypatch.setattr(ModernBertModel, "get_input_embeddings", lambda self: torch.nn.Identity())
    gc.collect()
    before = sum(isinstance(o, ModernBertModel) for o in gc.get_objects())
    during = []

    def original(cfg, encoder_dir = None):
        during.append(sum(isinstance(o, ModernBertModel) for o in gc.get_objects()))
        return "laya"

    assert laya_runtime._build_model(cfg, encoder_dir, original) == "laya"
    assert during == [before]


def test_build_hook_is_restored_even_when_loading_fails(monkeypatch):
    pytest.importorskip("torch")  # laya imports torch
    laya = laya_runtime._laya()
    original = laya.agent.build_model
    seen = []

    def boom(path, **kwargs):
        seen.append(laya.agent.build_model is not original)
        raise FileNotFoundError(path)

    monkeypatch.setattr(laya, "load", boom)
    with pytest.raises(FileNotFoundError):
        laya_runtime._load_laya("missing", device = "cpu")
    assert seen == [True] and laya.agent.build_model is original


def test_high_special_token_ids_still_skip_the_embedding_init(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import ModernBertConfig

    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    # ModernBERT-large puts its special tokens at the end of the vocabulary (pad 50283 of 50368).
    config = ModernBertConfig.from_pretrained(encoder_dir)
    config.pad_token_id, config.bos_token_id, config.eos_token_id = 297, 295, 296
    config.cls_token_id, config.sep_token_id = 295, 296
    config.save_pretrained(encoder_dir)
    built = []
    real_skip_init = torch.nn.utils.skip_init

    def skip_init(*args, **kwargs):
        built.append(args[1])
        return real_skip_init(*args, **kwargs)

    from transformers import AutoModel

    real_from_config = AutoModel.from_config
    placeholder_vocab = []

    def from_config(config, **kwargs):
        placeholder_vocab.append(config.vocab_size)
        return real_from_config(config, **kwargs)

    torch.nn.utils.skip_init, AutoModel.from_config = skip_init, from_config
    try:
        fast = laya_runtime._build_model(cfg, encoder_dir, laya.common.build_model).eval()
    finally:
        torch.nn.utils.skip_init, AutoModel.from_config = real_skip_init, real_from_config
    # One row, not one past the highest special id: otherwise the embedding is still randomly initialised.
    assert placeholder_vocab == [1]
    reference = laya.common.build_model(cfg, encoder_dir = encoder_dir).eval()
    assert built == [300]
    assert (
        fast.encoder.get_input_embeddings().padding_idx
        == 297
        == reference.encoder.get_input_embeddings().padding_idx
    )
    fast.load_state_dict(reference.state_dict(), strict = True)
    assert repr(fast) == repr(reference)
    assert fast.encoder.config.to_dict() == reference.encoder.config.to_dict()
    ids = torch.randint(5, 300, (2, 20))
    args = (
        ids,
        torch.ones_like(ids),
        torch.tensor([[3, 7]] * 2),
        torch.ones(2, 2, dtype = torch.bool),
        torch.zeros(2, dtype = torch.long),
    )
    with torch.inference_mode():
        assert torch.equal(fast(*args)[0], reference(*args)[0])


def _tiny_decision_batch(
    torch,
    rows = 4,
    tokens = 40,
):
    generator = torch.Generator().manual_seed(0)
    lengths = [tokens, 23, 9, 31][:rows]
    ids = torch.randint(5, 300, (rows, tokens), generator = generator)
    mask = torch.zeros(rows, tokens, dtype = torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
        ids[row, length:] = 297
    counts = [2, 5, 1, 3][:rows]
    positions = torch.zeros(rows, 5, dtype = torch.long)
    markers = torch.zeros(rows, 5, dtype = torch.bool)
    for row, count in enumerate(counts):
        positions[row, :count] = torch.randperm(lengths[row], generator = generator)[:count]
        markers[row, :count] = True
    return ids, mask, positions, markers, torch.tensor([0, 1, 2, 1][:rows])


@pytest.mark.parametrize("head_layers", [1, 2, 3])
def test_marker_head_matches_laya_forward(tmp_path, head_layers):
    torch = pytest.importorskip("torch")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    torch.manual_seed(0)
    model = laya.common.build_model(
        {**cfg, "head_layers": head_layers}, encoder_dir = encoder_dir
    ).eval()
    assert laya_runtime._marker_head(model)
    args = _tiny_decision_batch(torch)
    with torch.inference_mode():
        want = model(*args)[0]
        got = laya_runtime._decision_logits(model, *args)
        assert got.dtype == want.dtype == torch.float32
        torch.testing.assert_close(got, want, atol = 1e-5, rtol = 1e-4)
        # Without padding the mask is dropped altogether.
        unpadded = [arg[:1] for arg in args]
        torch.testing.assert_close(
            laya_runtime._decision_logits(model, *unpadded, padded = False),
            model(*unpadded)[0],
            atol = 1e-5,
            rtol = 1e-4,
        )


def test_models_without_laya_head_keep_their_own_forward(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    assert not laya_runtime._marker_head(
        laya.common.build_model({**cfg, "head_layers": 0}, encoder_dir = encoder_dir)
    )
    model = laya.common.build_model(cfg, encoder_dir = encoder_dir)
    model.scorer[2].approximate = "tanh"
    assert not laya_runtime._marker_head(model)
    model = laya.common.build_model(cfg, encoder_dir = encoder_dir)
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_FAST", "0")
    assert not laya_runtime._marker_head(model)


def test_collate_matches_laya():
    torch = pytest.importorskip("torch")
    collate_items = laya_runtime._laya().common.collate_items
    items = [
        {"ids": [2, 7, 9, 1], "markers": [1, 2], "qtype": 1},
        {"ids": [2, 8, 1], "markers": [1], "qtype": 0},
        {"ids": [2, 5, 6, 7, 8, 9, 1], "markers": [1, 3, 5, 6], "qtype": 2},
    ]
    want = collate_items([items], 297)
    got = laya_runtime._collate(items, 297)
    for name in laya_runtime._INPUTS:
        assert got[name].dtype == want[name].dtype and torch.equal(got[name], want[name]), name


def test_cuda_graphs_replay_the_eager_logits(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("needs a CUDA GPU (ROCm never builds graphs)")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    torch.manual_seed(0)
    model = laya.common.build_model({**cfg, "head_layers": 2}, encoder_dir = encoder_dir).eval()
    agent = SimpleNamespace(model = model, device = torch.device("cpu"), dtype = torch.float32)
    monkeypatch.setattr(laya_runtime, "_precision", lambda device, fp16_checkpoint: (None, None))
    laya_runtime._place(agent, torch.device("cuda"), False)
    ids, mask, positions, markers, qtype = _tiny_decision_batch(torch, rows = 3, tokens = 40)
    batch = dict(zip(laya_runtime._INPUTS, (ids, mask, positions, markers, qtype)))
    graphed = laya_runtime._run_model(agent, batch).clone()
    graphs = agent.__dict__["_unsloth_graphs"]
    if graphs.broken:
        # transformers 4.x: ModernBERT cannot be captured, so the model runs eagerly with the same logits.
        import transformers

        assert int(transformers.__version__.split(".")[0]) < 5
        with torch.inference_mode():
            reference = model(*(value.cuda() for value in batch.values()))[0]
        torch.testing.assert_close(graphed, reference, atol = 1e-4, rtol = 1e-4)
        return
    assert list(graphs.graphs) == [(4, 64, 8)]
    # The same padded bucket run eagerly: rows 3 -> 4 repeat row 0, tokens 40 -> 64 and options 5 -> 8 are masked.
    padded = {
        "input_ids": torch.zeros(4, 64, dtype = torch.long),
        "attention_mask": torch.zeros(4, 64, dtype = torch.long),
        "marker_pos": torch.zeros(4, 8, dtype = torch.long),
        "marker_mask": torch.zeros(4, 8, dtype = torch.bool),
        "qtype": torch.zeros(4, dtype = torch.long),
    }
    for name, value in batch.items():
        if value.dim() == 1:
            padded[name][:3] = value
        else:
            padded[name][:3, : value.shape[1]] = value
        padded[name][3:] = padded[name][:1]
    with torch.inference_mode():
        eager = laya_runtime._decision_logits(
            model, *(padded[name].cuda() for name in laya_runtime._INPUTS)
        )
    assert torch.equal(graphed, eager[:3, :5])
    with torch.inference_mode():
        reference = model(*(value.cuda() for value in batch.values()))[0]
    torch.testing.assert_close(graphed, reference, atol = 1e-4, rtol = 1e-4)
    # A second request with other contents replays the same graph.
    batch["input_ids"] = torch.randint(5, 300, ids.shape)
    with torch.inference_mode():
        reference = model(*(value.cuda() for value in batch.values()))[0]
    torch.testing.assert_close(
        laya_runtime._run_model(agent, batch), reference, atol = 1e-4, rtol = 1e-4
    )
    assert len(graphs.graphs) == 1
    # Moving or recasting the model drops the graphs that point at its old weights.
    laya_runtime._place(agent, torch.device("cuda"), False)
    assert "_unsloth_graphs" not in agent.__dict__
    # Batches too large to be worth padding into a bucket run eagerly.
    assert (
        laya_runtime._CUDAGraphs(agent).run(
            {"input_ids": torch.zeros(16, 1024), "marker_pos": torch.zeros(16, 2)}
        )
        is None
    )
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_CUDA_GRAPHS", "0")
    laya_runtime._run_model(agent, batch)
    assert "_unsloth_graphs" not in agent.__dict__


def test_encoders_that_keep_the_padding_id_match_laya(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import RobertaConfig

    laya = laya_runtime._laya()
    # RoBERTa keeps pad_token_id for its position ids, so a build with a stand-in padding id answers differently.
    config = RobertaConfig(
        vocab_size = 300,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        max_position_embeddings = 40,
        pad_token_id = 1,
        bos_token_id = 0,
        eos_token_id = 2,
    )
    encoder_dir = tmp_path / "encoder"
    config.save_pretrained(encoder_dir)
    cfg = {"encoder": "tiny", "head_layers": 1, "act_costs": {"escalate": 0.5}}
    reference = laya.common.build_model(cfg, encoder_dir = str(encoder_dir)).eval()
    fast = laya_runtime._build_model(cfg, str(encoder_dir), laya.common.build_model).eval()
    fast.load_state_dict(reference.state_dict(), strict = True)
    ids = torch.randint(5, 300, (2, 20))
    ids[:, 15:] = 1
    args = (
        ids,
        (ids != 1).long(),
        torch.tensor([[3, 7]] * 2),
        torch.ones(2, 2, dtype = torch.bool),
        torch.zeros(2, dtype = torch.long),
    )
    with torch.inference_mode():
        assert torch.equal(fast(*args)[0], reference(*args)[0])


def _cuda_tiny_agent(
    tmp_path,
    monkeypatch,
    torch,
    capture = True,
):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("needs a CUDA GPU")
    import transformers

    if capture and int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("transformers 4.x ModernBERT cannot be captured")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    torch.manual_seed(0)
    model = laya.common.build_model({**cfg, "head_layers": 2}, encoder_dir = encoder_dir).eval()
    agent = SimpleNamespace(model = model, device = torch.device("cpu"), dtype = torch.float32)
    monkeypatch.setattr(laya_runtime, "_precision", lambda device, fp16_checkpoint: (None, None))
    laya_runtime._place(agent, torch.device("cuda"), False)
    return agent


def _padded_eager(model, batch, key, torch):
    rows = batch["input_ids"].shape[0]
    shapes = {
        "input_ids": key[:2],
        "attention_mask": key[:2],
        "marker_pos": (key[0], key[2]),
        "marker_mask": (key[0], key[2]),
        "qtype": key[:1],
    }
    padded = {name: torch.zeros(shape, dtype = batch[name].dtype) for name, shape in shapes.items()}
    for name, value in batch.items():
        if value.dim() == 1:
            padded[name][:rows] = value
        else:
            padded[name][:rows, : value.shape[1]] = value
        padded[name][rows:] = padded[name][:1]
    with torch.inference_mode():
        return laya_runtime._decision_logits(
            model, *(padded[name].cuda() for name in laya_runtime._INPUTS)
        )


def test_cuda_graphs_sharing_a_pool_replay_in_any_order(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    agent = _cuda_tiny_agent(tmp_path, monkeypatch, torch)
    generator = torch.Generator().manual_seed(1)
    shapes = [(1, 20, 2), (3, 40, 5), (2, 100, 3), (6, 200, 9)]

    def request(rows, tokens, options):
        ids = torch.randint(5, 300, (rows, tokens), generator = generator)
        mask = torch.ones(rows, tokens, dtype = torch.long)
        mask[1:, tokens // 2 :] = 0
        positions = torch.randint(0, tokens // 2, (rows, options), generator = generator)
        markers = torch.ones(rows, options, dtype = torch.bool)
        markers[1:, options - 1] = False
        return dict(
            zip(
                laya_runtime._INPUTS,
                (ids, mask, positions, markers, torch.randint(0, 3, (rows,), generator = generator)),
            )
        )

    graphs = None
    # Capture order A B C D, then replay out of order, interleaving buckets.
    for index in [0, 1, 2, 3, 0, 2, 1, 3, 3, 0, 1, 0, 2]:
        batch = request(*shapes[index])
        got = laya_runtime._run_model(agent, batch).clone()
        graphs = agent.__dict__["_unsloth_graphs"]
        key = graphs.graphs and next(
            k
            for k in graphs.graphs
            if k[0] >= shapes[index][0] and k[1] >= shapes[index][1] and k[2] >= shapes[index][2]
        )
        want = _padded_eager(agent.model, batch, key, torch)
        assert torch.equal(got, want[: shapes[index][0], : shapes[index][2]]), (index, key)
    assert len(graphs.graphs) == 4 and not graphs.broken


def test_cuda_graphs_over_the_memory_budget_run_eagerly(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    agent = _cuda_tiny_agent(tmp_path, monkeypatch, torch)
    batch = dict(zip(laya_runtime._INPUTS, _tiny_decision_batch(torch, rows = 3, tokens = 40)))
    graphs = agent.__dict__["_unsloth_graphs"] = laya_runtime._CUDAGraphs(agent)
    # The first capture alone takes more than the budget.
    graphs.max_pool_bytes = 1 << 20
    reserved = iter([0, 2 << 20])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: next(reserved))
    logits = laya_runtime._run_model(agent, batch)
    assert graphs.broken and not graphs.graphs
    with torch.inference_mode():
        reference = agent.model(*(value.cuda() for value in batch.values()))[0]
    torch.testing.assert_close(logits, reference, atol = 1e-4, rtol = 1e-4)


def test_rocm_never_builds_cuda_graphs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    agent = _cuda_tiny_agent(tmp_path, monkeypatch, torch, capture = False)
    # transformers 4.x would torch.compile ModernBERT, which a faked ROCm version sends looking for HIP.
    agent.model.encoder.config.reference_compile = False
    monkeypatch.setattr(torch.version, "hip", "6.2")
    monkeypatch.setattr(laya_runtime, "_CUDAGraphs", None)
    batch = dict(zip(laya_runtime._INPUTS, _tiny_decision_batch(torch, rows = 3, tokens = 40)))
    laya_runtime._run_model(agent, batch)
    assert "_unsloth_graphs" not in agent.__dict__


def test_chunks_split_rows_and_trim_padding():
    torch = pytest.importorskip("torch")
    lengths = [10, 3, 7, 2, 9]
    batch = laya_runtime._collate(
        [{"ids": list(range(2, 2 + n)), "markers": [1], "qtype": 0} for n in lengths], 0
    )
    parts = list(laya_runtime._chunks(batch, 20))
    # 20 // 10 tokens = 2 rows per forward, each trimmed to its own longest row.
    assert [tuple(p["input_ids"].shape) for p in parts] == [(2, 10), (2, 7), (1, 9)]
    for name in laya_runtime._INPUTS:
        if name in ("input_ids", "attention_mask"):
            continue
        assert torch.equal(torch.cat([p[name] for p in parts]), batch[name])
    assert list(laya_runtime._chunks(batch, 10**6)) == [batch]
    # A row longer than the budget still runs, one row at a time.
    assert [p["input_ids"].shape[0] for p in laya_runtime._chunks(batch, 4)] == [1] * 5


def test_chunked_forward_matches_one_forward(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    laya = laya_runtime._laya()
    encoder_dir, cfg = _tiny_laya_encoder(tmp_path)
    torch.manual_seed(0)
    model = laya.common.build_model({**cfg, "head_layers": 2}, encoder_dir = encoder_dir).eval()
    agent = SimpleNamespace(model = model, device = torch.device("cpu"), dtype = torch.float32)
    batch = dict(zip(laya_runtime._INPUTS, _tiny_decision_batch(torch)))
    with torch.inference_mode():
        want = model(*batch.values())[0]
    monkeypatch.setitem(laya_runtime._CHUNK_TOKENS, "cpu", 40)
    got = laya_runtime._run_model(agent, batch)
    torch.testing.assert_close(got, want, atol = 1e-5, rtol = 1e-4)
    calls = []
    monkeypatch.setattr(
        laya_runtime,
        "_run_chunk",
        lambda agent, part: calls.append(part)
        or torch.zeros(part["input_ids"].shape[0], part["marker_pos"].shape[1]),
    )
    assert laya_runtime._run_model(agent, batch).shape == want.shape and len(calls) == 4
    # The kill switch keeps laya's single forward.
    calls.clear()
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_FAST", "0")
    laya_runtime._run_model(agent, batch)
    assert len(calls) == 1


def test_mlx_requests_run_in_chunks(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    seen = []

    class Model:
        def logits(self, batch):
            seen.append(tuple(batch["input_ids"].shape))
            return np.full(
                (batch["input_ids"].shape[0], batch["marker_pos"].shape[1]), float(len(seen))
            )

    agent = SimpleNamespace(device = "mlx", model = Model(), tok = SimpleNamespace(pad_token_id = 0))
    monkeypatch.setitem(laya_runtime._CHUNK_TOKENS, "mlx", 8)
    items = [{"ids": [2, 5, 6, 1], "markers": [1, 2], "qtype": 0} for _ in range(5)]
    logits, usage = laya_runtime._forward(agent, items)
    assert seen == [(2, 4), (2, 4), (1, 4)] and usage == 20
    assert logits[:, 0].tolist() == [1, 1, 2, 2, 3]


def test_cuda_graph_results_survive_the_next_replay(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    agent = _cuda_tiny_agent(tmp_path, monkeypatch, torch)
    first = dict(zip(laya_runtime._INPUTS, _tiny_decision_batch(torch, rows = 3, tokens = 40)))
    second = {**first, "input_ids": torch.randint(5, 300, first["input_ids"].shape)}
    a = laya_runtime._run_model(agent, first)
    kept = a.clone()
    laya_runtime._run_model(agent, second)
    assert torch.equal(a, kept)


def test_chunk_budget_splits_only_where_memory_is_short(monkeypatch):
    torch = pytest.importorskip("torch")
    # A CPU runs one forward: every extra forward costs a fixed overhead, and host RAM is rarely the limit.
    assert laya_runtime._chunk_budget(torch.device("cpu")) is None
    assert laya_runtime._chunk_budget("mlx") == laya_runtime._CHUNK_TOKENS["mlx"]
    gib = 1 << 30
    for total, hip, want in (
        (8 * gib, None, 16384),  # a small card keeps the floor
        (24 * gib, None, 26214),
        (180 * gib, None, 196608),  # the 64 x 1024 request stays one forward
        (96 * gib, "7.1", 31457),  # ROCm activations are larger per token
    ):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device, t = total: SimpleNamespace(total_memory = t),
        )
        monkeypatch.setattr(torch.version, "hip", hip)
        assert laya_runtime._chunk_budget(torch.device("cuda")) == want
    monkeypatch.setitem(laya_runtime._CHUNK_TOKENS, "cuda", None)
    assert laya_runtime._chunk_budget(torch.device("cuda")) is None


def _fake_mlx(monkeypatch, **zoo):
    mx = SimpleNamespace(float16 = "float16", float32 = "float32")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core = mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.decision", SimpleNamespace(**zoo))


def test_mlx_runs_fp16_checkpoints_in_fp16(monkeypatch, tmp_path):
    import transformers

    _fake_mlx(
        monkeypatch, load_decision_model = lambda folder, compute_dtype = "float32": compute_dtype
    )
    fake_laya = SimpleNamespace(
        common = SimpleNamespace(clamp_temperature = float),
        agent = SimpleNamespace(Agent = SimpleNamespace(_to_internal = None)),
    )
    monkeypatch.setitem(sys.modules, "laya", fake_laya)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: None)
    (tmp_path / "rl_agent_config.json").write_text("{}")
    assert laya_runtime._MLXAgent(tmp_path, True).model == "float16"
    assert laya_runtime._MLXAgent(tmp_path, False).model == "float32"
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_FP32", "1")
    assert laya_runtime._MLXAgent(tmp_path, True).model == "float32"
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_FP32")
    # An unsloth-zoo whose loader predates compute_dtype.
    _fake_mlx(monkeypatch, load_decision_model = lambda folder, dtype = "float32": dtype)
    old = laya_runtime._MLXAgent(tmp_path, True)
    assert old.dtype == old.model == "float32"

    monkeypatch.setattr(laya_runtime, "_checkpoint_dir", lambda checkpoint: tmp_path)
    monkeypatch.setattr(laya_runtime, "_device", lambda: "mlx")
    monkeypatch.setattr(laya_runtime, "_MLXAgent", lambda folder, fp16: fp16)
    assert _REAL_LOAD(catalog.CHECKPOINTS["laya-english"]) == (True, "mlx")


def test_mlx_fp16_overflow_reruns_in_fp32(monkeypatch, gpu_agent):
    import numpy as np

    _fake_mlx(monkeypatch)

    class Model:
        dtype = "float16"

        def set_dtype(self, dtype):
            self.dtype = dtype

        def logits(self, batch):
            return np.array([[np.inf if self.dtype == "float16" else 1.0, 0.5]])

    agent = SimpleNamespace(device = "mlx", dtype = "float16", model = Model(), tok = gpu_agent.tok)
    logits, _ = laya_runtime._forward(agent, _items())
    assert logits.tolist() == [[1.0, 0.5]] and agent.dtype == agent.model.dtype == "float32"
    # An fp32 agent has nothing wider to retry in: its non-finite logits come back without a set_dtype call.
    agent.model.dtype = "float16"  # only makes the fake emit inf
    agent.model.set_dtype = None
    logits, _ = laya_runtime._forward(agent, _items())
    assert np.isinf(logits[0, 0])


def _mcp_decide(arguments):
    from fastmcp import Client
    async def call():
        async with Client(systemone.decisions_mcp) as mcp:
            return await mcp.call_tool("decide", arguments, raise_on_error = False)

    return asyncio.run(call())


def test_decisions_mcp_answers_like_the_route(client):
    route = _post(client).json()
    result = _mcp_decide(
        {"state": "Everything is down and we have a demo at noon.", "questions": QUESTIONS}
    )
    assert not result.is_error
    assert result.structured_content == route


def test_decisions_mcp_reports_the_route_errors(monkeypatch, runtime):
    result = _mcp_decide({"state": "x", "questions": {}})
    assert result.is_error and "At least one question" in result.content[0].text
    long_state = "x" * (systemone.MAX_STATE_CHARS + 1)
    result = _mcp_decide({"state": long_state, "questions": QUESTIONS})
    assert result.is_error and "State is longer than" in result.content[0].text
    monkeypatch.setattr(systemone_settings, "_owner_setting", {}.get)
    result = _mcp_decide({"state": "x", "questions": QUESTIONS})
    assert result.is_error and "Settings > API" in result.content[0].text
    assert runtime == []


def test_chat_calls_studio_decisions_without_a_server_or_key(client):
    import json

    from core.inference.mcp_client import call_tool_sync, close_mcp_sessions, list_tools_async

    url = client.get("/api/settings/systemone").json()["mcp_url"]
    assert url == f"http://127.0.0.1:80{systemone.MCP_PATH}/"
    tools = asyncio.run(list_tools_async(url, timeout = 10))
    assert [tool["name"] for tool in tools] == ["decide"]
    text = call_tool_sync(url, None, "decide", {"state": "x", "questions": QUESTIONS}, scope = "chat")
    close_mcp_sessions(url, None)
    assert json.loads(text)["answers"]["urgent"] == {"type": "noul", "noul": 0.9}


def test_decisions_mcp_endpoint_needs_studio_auth(monkeypatch):
    from main import app
    from starlette.applications import Starlette
    from starlette.routing import Mount

    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "decide", "arguments": {"state": "x", "questions": QUESTIONS}},
    }
    listing = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    headers = {"Accept": "application/json, text/event-stream"}
    assert (
        TestClient(app).post(f"{systemone.MCP_PATH}/", json = call, headers = headers).status_code
        == 401
    )

    async def signed_in(credentials):
        return "tester"

    monkeypatch.setattr(systemone, "get_current_subject", signed_in)
    mcp_app = systemone.decisions_mcp.http_app(path = "/", stateless_http = True, json_response = True)
    served = Starlette(
        routes = [Mount(systemone.MCP_PATH, systemone.RequireStudioAuth(mcp_app))],
        lifespan = mcp_app.lifespan,
    )
    with TestClient(served) as http:
        listed = http.post(
            f"{systemone.MCP_PATH}/",
            json = listing,
            headers = {**headers, "Authorization": "Bearer t"},
        )
        answered = http.post(
            f"{systemone.MCP_PATH}/", json = call, headers = {**headers, "Authorization": "Bearer t"}
        )
        monkeypatch.setattr(systemone_settings, "_owner_setting", {}.get)
        hidden = http.post(
            f"{systemone.MCP_PATH}/",
            json = listing,
            headers = {**headers, "Authorization": "Bearer t"},
        )
    assert answered.status_code == 200, answered.text
    assert answered.json()["result"]["structuredContent"]["answers"]["urgent"]["noul"] == 0.9
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["decide"]
    assert hidden.json()["result"]["tools"] == []


def test_managed_accounts_can_add_studio_decisions_but_not_other_loopback():
    from core.inference.mcp_client import validate_mcp_address
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("alice-id", "alice")
    run_as(alice, validate_mcp_address, f"http://127.0.0.1:8888{systemone.MCP_PATH}/")
    with pytest.raises(HTTPException):
        run_as(alice, validate_mcp_address, "http://127.0.0.1:8888/mcp")


def test_mlx_overflow_in_a_later_chunk_reruns_the_whole_request_in_fp32(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    _fake_mlx(monkeypatch)
    seen = []

    class Model:
        dtype = "float16"

        def set_dtype(self, dtype):
            self.dtype = dtype

        def logits(self, batch):
            seen.append((self.dtype, batch["input_ids"].shape[0]))
            rows = batch["input_ids"].shape[0]
            bad = self.dtype == "float16" and len(seen) == 3
            return np.full((rows, batch["marker_pos"].shape[1]), np.inf if bad else 1.0)

    agent = SimpleNamespace(
        device = "mlx", dtype = "float16", model = Model(), tok = SimpleNamespace(pad_token_id = 0)
    )
    monkeypatch.setitem(laya_runtime._CHUNK_TOKENS, "mlx", 8)
    items = [{"ids": [2, 5, 6, 1], "markers": [1, 2], "qtype": 0} for _ in range(5)]
    logits, _ = laya_runtime._forward(agent, items)
    assert np.isfinite(logits).all() and logits.shape == (5, 2)
    # Three fp16 chunks, the last overflowing, then all three again in fp32.
    assert seen == [
        ("float16", 2),
        ("float16", 2),
        ("float16", 1),
        ("float32", 2),
        ("float32", 2),
        ("float32", 1),
    ]
