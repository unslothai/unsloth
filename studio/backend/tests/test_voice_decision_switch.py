# SPDX-License-Identifier: AGPL-3.0-only
"""Voice mode's decision checkpoint switched live: the stored pick wins over the launch env var."""

from __future__ import annotations

import threading
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

    switched, warmed = [], threading.Event()
    fake = SimpleNamespace(
        switch_decision = lambda: switched.append(True),
        decision_state = lambda: {"state": "loading", "error": None},
        warm_decision = warmed.set,
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
    assert got["models"] == []

    put = client.put("/api/inference/voice/decision", json = {"name": "laya-ft:v3"})
    assert put.status_code == 200
    assert (put.json()["current"], put.json()["source"], put.json()["state"]) == (
        "laya-ft:v3",
        "setting",
        "loading",
    )
    assert switched == [True]
    assert warmed.wait(5)

    assert client.put("/api/inference/voice/decision", json = {"name": "nope"}).status_code == 400
    cleared = client.put("/api/inference/voice/decision", json = {"name": None}).json()
    assert (cleared["current"], cleared["source"]) == ("laya-ft:v4", "env")


def test_a_stock_checkpoint_is_not_offered_and_reads_as_not_set_up(monkeypatch, tmp_path):
    from auth.authentication import get_current_subject
    from core.inference import voice_models as vm
    from core.systemone import catalog
    from routes import inference

    with pytest.raises(ValueError, match = "not trained for voice mode"):
        vd.set_checkpoint("laya-english")
    with pytest.raises(ValueError, match = "not trained for voice mode"):
        vd.set_checkpoint("default")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({vd.SETTING_KEY: "laya-english"}, read_back = False)
    monkeypatch.setenv(vd.ENV_CHECKPOINT, "laya-multilingual")
    monkeypatch.delenv(vm.ENV_DIR, raising = False)
    monkeypatch.delenv(vm.ENV_HUB, raising = False)
    assert (vd.checkpoint_name(), vd.checkpoint_source()) == (None, None)

    models = vm.VoiceModels()
    monkeypatch.setattr(vm, "get_voice_models", lambda: models)
    monkeypatch.setattr(catalog, "fine_tunes", lambda: [])
    monkeypatch.setattr(catalog, "folder_decisions", lambda: [])
    client = _voice_client([inference.router, inference.studio_router])
    # 503 makes the browser fall back to its normal pause.
    assert (
        client.post("/api/inference/voice/judge", json = {"text": "Tell me a joke."}).status_code
        == 503
    )
    assert "decision" not in models._models

    got = client.get("/api/inference/voice/decision").json()
    assert (got["current"], got["source"], got["launch_default"]) == (None, None, None)
    assert got["builtin_configured"] is False
    assert got["models"] == []
    put = client.put("/api/inference/voice/decision", json = {"name": "laya-english"})
    assert put.status_code == 400 and "not trained for voice mode" in put.json()["detail"]


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
    models.warm_decision()
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


def test_decision_voice_raises_unavailable_while_loading(checkpoints):
    from core.systemone import laya_runtime

    def loading(checkpoint, state, question):
        raise laya_runtime.Unavailable(503, "model_loading", "laya-ft:v4 is loading", 5)

    with pytest.raises(vd.DecisionUnavailable) as err:
        vd.DecisionVoice("laya-ft:v4", decide = loading).judge("hi")
    assert err.value.retry_after == 5 and isinstance(err.value, RuntimeError)
    assert err.value.loading is True

    def busy(checkpoint, state, question):
        raise laya_runtime.Unavailable(409, "gpu_busy", "Unload the resident chat model first.", 1)

    with pytest.raises(vd.DecisionUnavailable) as err:
        vd.DecisionVoice("laya-ft:v4", decide = busy).judge("hi")
    assert err.value.loading is False


def test_a_question_for_a_model_that_did_not_load_says_why(monkeypatch, tmp_path):
    from core.inference import voice_models as vm
    from routes import inference

    monkeypatch.delenv(vd.ENV_CHECKPOINT, raising = False)
    models = vm.VoiceModels(root = tmp_path / "nope")
    with pytest.raises(vm.VoiceModelUnavailable, match = "judge model is unavailable: FileNotFound"):
        models.judge("hi")

    def gone(text):
        raise vm.VoiceModelUnavailable("Voice judge model is unavailable: switched")

    fake = SimpleNamespace(available = lambda name: True, reason = lambda name: "", judge = gone)
    monkeypatch.setattr(vm, "get_voice_models", lambda: fake)
    r = _voice_client([inference.router]).post("/api/inference/voice/judge", json = {"text": "hi"})
    assert (r.status_code, r.json()["detail"]) == (
        503,
        "Voice judge model is unavailable: switched",
    )


def test_retiring_a_voice_skips_one_a_request_holds(monkeypatch):
    from core.inference import audio_cpp_models, model_slots
    from models.inference import LoadRequest
    from routes import inference
    from utils import multi_model_settings

    def kept_voice(refs):
        return model_slots.ExtraSlot(
            SimpleNamespace(is_loaded = False, is_active = False),
            SimpleNamespace(active_model_name = None),
            "owner",
            request = LoadRequest(model_path = "audio-cpp/Kokoro-82M-GGUF", voice = True),
            refs = refs,
            voice = True,
        )

    held, idle = kept_voice(refs = 1), kept_voice(refs = 0)
    dropped, reached = [], []

    def stop_before_the_new_slot(*args, **kwargs):
        reached.append(True)
        raise RuntimeError("stop before the new slot starts")

    monkeypatch.setattr(model_slots, "slots", [held, idle])
    monkeypatch.setattr(model_slots, "drop", dropped.append)
    monkeypatch.setattr(
        audio_cpp_models,
        "resolve",
        lambda path, *a, **k: SimpleNamespace(task = "tts", unsupported = None)
        if path.startswith("audio-cpp/")
        else None,
    )
    monkeypatch.setattr(multi_model_settings, "get_multi_model_enabled", lambda: True)
    monkeypatch.setattr(
        inference, "get_inference_backend", lambda: SimpleNamespace(active_model_name = "org/Chat")
    )
    monkeypatch.setattr(inference, "LlamaCppBackend", stop_before_the_new_slot)

    from auth.authentication import get_current_subject
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    for router in (inference.router, inference.studio_router):
        app.include_router(router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    TestClient(app, raise_server_exceptions = False).post(
        "/api/inference/load", json = {"model_path": "audio-cpp/Other-TTS", "voice": True}
    )
    assert reached == [True]
    assert dropped == [idle]
    assert model_slots.slots == [held]


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


def test_the_trace_probe_loads_no_decision_model(monkeypatch, tmp_path):
    from core.inference import voice_models
    from routes import inference

    def fail():
        raise AssertionError("the trace probe must not touch the voice models")

    monkeypatch.setattr(voice_models, "get_voice_models", fail)
    client = _voice_client([inference.studio_router])
    monkeypatch.delenv("UNSLOTH_VAD_DEBUG_DIR", raising = False)
    assert client.get("/api/inference/voice/trace").json() == {"trace": False}
    monkeypatch.setenv("UNSLOTH_VAD_DEBUG_DIR", str(tmp_path))
    assert client.get("/api/inference/voice/trace").json() == {"trace": True}


def test_a_clef_folder_without_the_optional_decision_config_is_still_found(monkeypatch, tmp_path):
    from core.systemone import catalog, laya_runtime
    from utils.models.model_config import CLEF_ADAPTER_MARKERS, CLEF_MARKERS

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    root = (tmp_path / "shared").resolve()
    for name, markers in (("merged-clef", CLEF_MARKERS), ("adapter-clef", CLEF_ADAPTER_MARKERS)):
        folder = root / name
        folder.mkdir(parents = True)
        for marker in markers:
            (folder / marker).write_text("{}")
        assert not (folder / "unsloth_decision_config.json").exists()
    monkeypatch.setattr(catalog, "_custom_roots", lambda: [(root, False)])
    found = sorted((c.label, c.layout) for c in catalog.folder_decisions())
    assert found == [("adapter-clef", "clef"), ("merged-clef", "clef")]


def test_a_d1_in_a_custom_folder_keeps_the_mlx_backend(monkeypatch, tmp_path):
    import json

    from core.systemone import catalog, laya_runtime

    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    root = (tmp_path / "shared").resolve()
    from utils.models.model_config import CLEF_MARKERS

    (root / "my-d1").mkdir(parents = True)
    (root / "my-d1" / "config.json").write_text(json.dumps({"model_type": "d1_omni"}))
    (root / "my-clef").mkdir()
    for marker in CLEF_MARKERS:
        (root / "my-clef" / marker).write_text("{}")
    monkeypatch.setattr(catalog, "_custom_roots", lambda: [(root, False)])
    found = sorted((c.label, c.layout, c.backend) for c in catalog.folder_decisions())
    assert found == [("my-clef", "clef", "pytorch"), ("my-d1", "clef", "mlx")]


def test_the_decision_switch_takes_the_models_lock_off_the_event_loop(checkpoints, monkeypatch):
    import asyncio

    from core.inference import voice_models as vm
    from core.systemone import catalog, laya_runtime
    from routes import inference

    where = []

    def switch_decision():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            where.append("worker thread")
        else:
            where.append("event loop")

    fake = SimpleNamespace(
        switch_decision = switch_decision,
        decision_state = lambda: {"state": "loading", "error": None},
        warm_decision = lambda: None,
    )
    monkeypatch.setattr(vm, "get_voice_models", lambda: fake)
    monkeypatch.setattr(catalog, "fine_tunes", lambda: [])
    monkeypatch.setattr(laya_runtime, "select", lambda *a, **k: None)
    put = _voice_client([inference.studio_router]).put(
        "/api/inference/voice/decision", json = {"name": "laya-ft:v3"}
    )
    assert put.status_code == 200
    assert where == ["worker thread"]
