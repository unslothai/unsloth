# SPDX-License-Identifier: AGPL-3.0-only
"""Voice mode's decision checkpoint switched live: the stored pick wins over the launch env var."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.inference import voice_decisions as vd


@pytest.fixture
def checkpoints(monkeypatch):
    from core.systemone import catalog

    known = {"laya-ft:v3", "laya-ft:v4"}
    monkeypatch.setattr(
        catalog,
        "resolve",
        lambda name: SimpleNamespace(name = name, layout = "laya") if name in known else None,
    )
    monkeypatch.setenv(vd.ENV_CHECKPOINT, "laya-ft:v4")
    return known


def test_setting_wins_over_env_and_clearing_falls_back(checkpoints):
    assert vd.checkpoint_name() == "laya-ft:v4"
    assert vd.checkpoint_source() == "env"
    vd.set_checkpoint("laya-ft:v3")
    assert vd.checkpoint_name() == "laya-ft:v3"
    assert vd.checkpoint_source() == "setting"
    vd.set_checkpoint(None)
    assert vd.checkpoint_name() == "laya-ft:v4"
    assert vd.checkpoint_source() == "env"


def test_unknown_checkpoint_is_refused(checkpoints):
    with pytest.raises(ValueError):
        vd.set_checkpoint("laya-ft:missing")
    assert vd.checkpoint_name() == "laya-ft:v4"


def test_switch_rebuilds_on_the_new_checkpoint(checkpoints, monkeypatch, tmp_path):
    from core.inference import voice_models as vm

    asked = []

    def ask(self, state, question):
        asked.append(self.checkpoint.name)
        return {"type": "noul", "noul": 0.5, "confidence": 0.5}

    monkeypatch.setattr(vd.DecisionVoice, "ask", ask)
    models = vm.VoiceModels(root = tmp_path)
    models.judge("one")
    vd.set_checkpoint("laya-ft:v3")
    models.judge("two")  # still the resident v4 until the switch is applied
    models.switch_decision()
    models.judge("three")
    assert asked == ["laya-ft:v4", "laya-ft:v4", "laya-ft:v3"]


def test_route_sets_and_reports_the_pick(checkpoints, monkeypatch):
    from auth.authentication import get_current_subject
    from core.inference import voice_models as vm
    from core.systemone import catalog, laya_runtime
    from routes import inference

    switched = []
    fake = SimpleNamespace(
        switch_decision = lambda: switched.append(True),
        decision_state = lambda: {"state": "loading", "error": None},
        judge = lambda text: 0.5,
    )
    monkeypatch.setattr(vm, "get_voice_models", lambda: fake)
    monkeypatch.setattr(catalog, "fine_tunes", lambda: [])
    monkeypatch.setattr(laya_runtime, "select", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)

    got = client.get("/api/inference/voice/decision").json()
    assert (got["current"], got["source"], got["launch_default"]) == (
        "laya-ft:v4",
        "env",
        "laya-ft:v4",
    )
    stock = {m["name"]: m for m in got["models"]}["laya-multilingual"]
    # The picker row's badges: family, size; a stock entry has no folder, so no age or GGUF export.
    assert (stock["family"], stock["gguf"], stock["mtime"]) == ("Laya", None, None)
    assert stock["size_bytes"] > 0

    put = client.put("/api/inference/voice/decision", json = {"name": "laya-ft:v3"})
    assert put.status_code == 200
    assert (put.json()["current"], put.json()["source"], put.json()["state"]) == (
        "laya-ft:v3",
        "setting",
        "loading",
    )
    assert switched == [True]

    assert client.put("/api/inference/voice/decision", json = {"name": "nope"}).status_code == 400
    cleared = client.put("/api/inference/voice/decision", json = {"name": None}).json()
    assert (cleared["current"], cleared["source"]) == ("laya-ft:v4", "env")


def _clef_folder(folder):
    folder.mkdir(parents = True)
    for name in (
        "config.json",
        "unsloth_decision_config.json",
        "joint_head.safetensors",
        "joint_head_config.json",
    ):
        (folder / name).write_text("{}")
    return folder


def test_custom_folder_decision_models_are_found_and_kept_in_bounds(monkeypatch, tmp_path):
    from core.systemone import catalog, laya_runtime

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    root = (tmp_path / "shared").resolve()
    model = _clef_folder(root / "voice-decisions-v4-qwen")
    (root / "notes").mkdir()
    outside = _clef_folder(tmp_path / "elsewhere")
    # The picker's custom folders; the real list refuses tmp paths (/private/var is denied).
    monkeypatch.setattr(catalog, "_custom_roots", lambda: [(root, False)])

    found = catalog.folder_decisions()
    assert [(c.name, c.layout, c.label) for c in found] == [
        (catalog.FOLDER_PREFIX + str(model.resolve()), "clef", "voice-decisions-v4-qwen")
    ]
    assert catalog.resolve(found[0].name) == found[0]
    assert catalog.resolve(catalog.FOLDER_PREFIX + str(outside)) is None
    assert catalog.resolve(catalog.FOLDER_PREFIX + str(root / "notes")) is None

    vd.set_checkpoint(found[0].name)
    assert vd.checkpoint_name() == found[0].name


def test_a_refused_warm_up_reports_an_error(checkpoints, monkeypatch, tmp_path):
    from core.inference import voice_models as vm
    from core.systemone import laya_runtime
    from routes import inference

    def ask(self, state, question):
        raise RuntimeError("Decision model unavailable: Unload the resident chat model first.")

    monkeypatch.setattr(vd.DecisionVoice, "ask", ask)
    monkeypatch.setattr(
        laya_runtime,
        "status",
        lambda: {"loaded_model": None, "loading_model": None, "error_model": None, "error": None},
    )
    models = vm.VoiceModels(root = tmp_path)
    monkeypatch.setattr(vm, "get_voice_models", lambda: models)
    inference._warm_voice_decision()
    assert models.decision_state() == {
        "state": "error",
        "error": "Decision model unavailable: Unload the resident chat model first.",
    }
    models.switch_decision()
    assert models.decision_state()["state"] == "not_loaded"


def _as_account(monkeypatch, account_id, role):
    from routes import inference
    from utils import account_context

    ctx = account_context.AccountContext(account_id, account_id, role)
    monkeypatch.setattr(account_context, "current_account", lambda: ctx)
    monkeypatch.setattr(account_context, "current_account_id", lambda: account_id)
    monkeypatch.setattr(inference, "current_account_id", lambda: account_id)


def _voice_client(routers):
    from auth.authentication import get_current_subject

    app = FastAPI()
    for router in routers:
        app.include_router(router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def test_voice_log_routes_are_owner_only_and_outcomes_stay_with_their_account(
    monkeypatch, tmp_path
):
    from core.inference import voice_log
    from routes import inference

    monkeypatch.setenv("UNSLOTH_VOICE_LOG_DIR", str(tmp_path))
    client = _voice_client([inference.router, inference.studio_router])
    alice = voice_log.append("judge", {"text": "my pin"}, "hold", 0.2, 1.0, account = "alice")
    owners = voice_log.append("judge", {"text": "hello"}, "hold", 0.2, 1.0, account = "owner")

    _as_account(monkeypatch, "bob", "user")
    assert client.get("/api/inference/voice/laya/stats").status_code == 403
    assert client.get("/api/inference/voice/laya/export?only=all").status_code == 403
    assert client.delete("/api/inference/voice/laya/log").status_code == 403
    for row_id in (alice, owners, "missing"):
        posted = client.post(
            "/api/inference/voice/outcome", json = {"id": row_id, "outcome": "flushed"}
        )
        assert posted.status_code == 404

    _as_account(monkeypatch, "alice", "user")
    posted = client.post("/api/inference/voice/outcome", json = {"id": alice, "outcome": "flushed"})
    assert posted.status_code == 200

    _as_account(monkeypatch, "owner", "owner")
    assert client.get("/api/inference/voice/laya/stats").json()["rows"] == 2
    assert {r["id"]: r["outcome"] for r in voice_log.read_rows()} == {
        alice: "flushed",
        owners: None,
    }
    assert client.delete("/api/inference/voice/laya/log").json() == {"removed": 2}


def test_a_loading_decision_checkpoint_answers_503(monkeypatch):
    from core.inference import voice_models as vm
    from routes import inference

    def loading(*args):
        raise vd.DecisionUnavailable("Decision model unavailable: laya-ft:v4 is loading", 5)

    def broken(*args):
        raise RuntimeError("something else")

    fake = SimpleNamespace(
        available = lambda name: True,
        reason = lambda name: "",
        judge = loading,
        filler = loading,
        interject = loading,
    )
    monkeypatch.setattr(vm, "get_voice_models", lambda: fake)
    client = _voice_client([inference.router])
    for path, body in (
        ("judge", {"text": "hi"}),
        ("filler", {"text": "hi"}),
        ("interject", {"prompt": "a", "utterance": "b"}),
    ):
        r = client.post(f"/api/inference/voice/{path}", json = body)
        assert (r.status_code, r.headers.get("retry-after")) == (503, "5"), path
        assert "is loading" in r.json()["detail"]

    fake.judge = broken
    r = TestClient(client.app, raise_server_exceptions = False).post(
        "/api/inference/voice/judge", json = {"text": "hi"}
    )
    assert r.status_code == 500


def test_decision_voice_raises_unavailable_while_loading():
    from core.systemone import laya_runtime

    def decide(checkpoint, state, question):
        raise laya_runtime.Unavailable(503, "model_loading", "laya-ft:v4 is loading", 5)

    voice = vd.DecisionVoice.__new__(vd.DecisionVoice)
    voice.checkpoint, voice._decide = object(), decide
    with pytest.raises(vd.DecisionUnavailable) as err:
        voice.judge("hi")
    assert err.value.retry_after == 5 and isinstance(err.value, RuntimeError)


def test_retiring_a_voice_skips_one_a_request_holds(monkeypatch):
    from core.inference import model_slots
    from routes import inference

    busy = SimpleNamespace(voice = True, request = SimpleNamespace(), refs = 1, generations = set())
    idle = SimpleNamespace(voice = True, request = SimpleNamespace(), refs = 0, generations = set())
    monkeypatch.setattr(model_slots, "slots", [busy, idle])
    monkeypatch.setattr(model_slots, "visible", lambda: [busy, idle])
    dropped = []
    monkeypatch.setattr(model_slots, "drop", dropped.append)
    monkeypatch.setattr(inference, "_voice_task", lambda request: "tts")
    monkeypatch.setattr(inference, "_kept_slot_reports", lambda slot: [])

    inference._retire_other_voices(SimpleNamespace())
    assert dropped == [idle]
    assert model_slots.slots == [busy]


def test_a_symlinked_child_outside_the_folder_is_not_listed(monkeypatch, tmp_path):
    import os

    from core.systemone import catalog, laya_runtime

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    root = (tmp_path / "shared").resolve()
    root.mkdir()
    outside = _clef_folder(tmp_path / "elsewhere")
    os.symlink(outside, root / "linked")
    inside = _clef_folder(root / "real")
    os.symlink(inside, root / "alias")
    monkeypatch.setattr(catalog, "_custom_roots", lambda: [(root, False)])

    found = catalog.folder_decisions()
    assert [c.name for c in found] == [catalog.FOLDER_PREFIX + str(inside)]
    assert all(catalog.folder_decision(c.name) == c for c in found)


def test_vad_routes_check_a_managed_accounts_model_access(monkeypatch):
    """VAD load/detect gate the detector repo like STT load does, before any download or load."""
    from fastapi import HTTPException

    from core.inference import vad_audiocpp_sidecar
    from hub.dependencies import get_request_hf_token
    from hub.services.models import account_access
    from routes import inference

    checked = []

    def deny(reference, *args, **kwargs):
        checked.append(reference)
        raise HTTPException(status_code = 403, detail = "Model not allowed")

    monkeypatch.setattr(account_access, "require_model_access", deny)
    calls = []
    fake = SimpleNamespace(
        load = lambda model, *a, **k: calls.append(("load", model)) or model,
        download = lambda *a, **k: calls.append(("download", a)),
        detect = lambda *a, **k: calls.append(("detect", a)) or {},
    )
    monkeypatch.setattr(vad_audiocpp_sidecar, "get_audio_cpp_vad_sidecar", lambda: fake)
    client = _voice_client([inference.router, inference.studio_router])
    client.app.dependency_overrides[get_request_hf_token] = lambda: None
    _as_account(monkeypatch, "bob", "user")
    monkeypatch.setattr(account_access, "managed_account", lambda: True)

    load = client.post("/api/inference/audio/vad/load", json = {"model": "someorg/not-allowed-vad"})
    detect = client.post(
        "/api/inference/audio/vad/detect?model=someorg/not-allowed-vad", content = b"RIFF"
    )
    assert load.status_code == 403 and detect.status_code == 403
    assert checked == ["someorg/not-allowed-vad", "someorg/not-allowed-vad"] and calls == []


def test_a_managed_voice_load_checks_access_before_any_hub_lookup(monkeypatch):
    """voice=true runs the model-access check first; a refused repo is never resolved on the Hub."""
    from fastapi import HTTPException

    from core.inference import audio_cpp_models
    from hub.services.models import account_access
    from routes import inference

    events = []

    def deny(reference, *args, **kwargs):
        events.append(("access", reference))
        raise HTTPException(status_code = 403, detail = "Model not allowed")

    monkeypatch.setattr(account_access, "require_model_access", deny)
    monkeypatch.setattr(
        audio_cpp_models, "resolve", lambda *a, **k: events.append(("resolve", a[0])) or None
    )
    client = _voice_client([inference.router, inference.studio_router])
    _as_account(monkeypatch, "bob", "user")
    monkeypatch.setattr(account_access, "managed_account", lambda: True)

    response = client.post(
        "/api/inference/load", json = {"model_path": "secretorg/private-llm-GGUF", "voice": True}
    )
    assert response.status_code == 403
    assert events == [("access", "secretorg/private-llm-GGUF")]
