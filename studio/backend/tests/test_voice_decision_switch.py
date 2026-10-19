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
    assert (got["current"], got["source"], got["launch_default"]) == ("laya-ft:v4", "env", "laya-ft:v4")
    assert "laya-multilingual" in {m["name"] for m in got["models"]}

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
    for name in ("config.json", "unsloth_decision_config.json", "joint_head.safetensors", "joint_head_config.json"):
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
