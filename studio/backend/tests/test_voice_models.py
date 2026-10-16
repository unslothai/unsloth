# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
from pathlib import Path

import pytest

from core.inference import voice_models as vm


class _FakeJudge:
    def __call__(self, text):
        return 0.9 if text.endswith("?") else 0.2


class _FakeFiller:
    def predict(self, text, prev_q = False):
        return {"say": True, "class": "code", "confidence": 0.8, "phrase": "On it.", "slot": None, "echo": False, "ms": 0.1}


class _FakeInterject:
    def predict(self, prompt, utterance, elapsed_s = 2.0):
        return {"action": "ignore", "confidence": 0.7, "probs": {"restart": 0.1, "ignore": 0.7, "queue": 0.2}, "merged_prompt": None}


def _stubbed(tmp_path, monkeypatch, **loaders):
    missing = {name: _boom for name in vm.MODELS}
    monkeypatch.setattr(vm.VoiceModels, "_loaders", {**missing, **loaders})
    return vm.VoiceModels(tmp_path)


def _boom(root):
    raise RuntimeError("no torch here")


def test_missing_dir_marks_every_model_unavailable(tmp_path):
    models = vm.VoiceModels(tmp_path / "nope")
    status = models.status()
    assert status["judge"] is False and status["filler"] is False and status["interject"] is False
    assert status["dir"] == str(tmp_path / "nope")
    assert "not found" in models.reason("judge")


def test_a_loader_failure_is_remembered_and_never_raises(tmp_path, monkeypatch):
    calls = []

    def flaky(root):
        calls.append(root)
        raise ImportError("No module named torch")

    models = _stubbed(tmp_path, monkeypatch, judge = flaky, filler = lambda root: _FakeFiller())
    assert models.available("judge") is False
    assert models.available("judge") is False
    assert len(calls) == 1
    assert "ImportError" in models.reason("judge")
    assert models.status()["filler"] is True
    assert models.status()["errors"]["judge"].startswith("ImportError")


def test_calls_go_through_to_the_loaded_models(tmp_path, monkeypatch):
    models = _stubbed(
        tmp_path, monkeypatch,
        judge = lambda root: _FakeJudge(), filler = lambda root: _FakeFiller(), interject = lambda root: _FakeInterject(),
    )
    assert models.judge("are you there?") == 0.9
    assert models.filler("write me a snake game")["class"] == "code"
    assert models.interject("write a poem", "ok", 1.0)["action"] == "ignore"
    assert models.status() == {"judge": True, "filler": True, "interject": True, "dir": str(tmp_path), "errors": {}}


def test_env_var_picks_the_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(vm.ENV_DIR, str(tmp_path))
    assert vm.models_dir() == tmp_path
    monkeypatch.delenv(vm.ENV_DIR)
    assert vm.models_dir() == Path(vm.DEFAULT_DIR).expanduser()


def test_featurize_adds_the_context_token():
    assert vm.featurize("  Hello There ") == "hello there <end>"
    assert vm.featurize("the second one", prev_q = True) == "<prevq> the second one <end>"


@pytest.mark.parametrize(
    "text, slot",
    [
        ("write me a snake game in python", "a snake game"),
        ("why is my install failing with this error", "the install error"),
        ("can you plan my week", "your week"),
        ("hello", None),
        ("the second one", None),
    ],
)
def test_echo_slot_rules(text, slot):
    got, conf = vm.extract_slot(text)
    assert got == slot
    assert (conf > 0) == (slot is not None)


def test_filler_phrase_and_echo_come_from_the_bank(tmp_path, monkeypatch):
    bank = {
        "classes": {"code": {"phrases": [{"text": "On it.", "weight": 1}]}, "minimal": {"phrases": [{"text": "Sure.", "weight": 1}]}},
        "echo_templates": {"templates": [{"text": "okay, {slot}.", "weight": 1}], "mix": {"echo_when_slot": 1.0}},
    }
    (tmp_path / "filler").mkdir()
    (tmp_path / "filler" / "phrases.json").write_text(json.dumps(bank))

    import numpy as np

    class _Gate:
        def predict_proba(self, xs):
            return np.array([[0.1, 0.9]] if "snake" in xs[0] or "hi" in xs[0] else [[0.8, 0.2]])

    class _Cls:
        classes_ = ["code", "minimal"]

        def predict_proba(self, xs):
            return np.array([[0.7, 0.3]] if "snake" in xs[0] else [[0.2, 0.8]])

    def fake_pickle_load(f):
        return {"gate": _Gate(), "cls": _Cls()}

    monkeypatch.setattr(vm.pickle, "load", fake_pickle_load)
    (tmp_path / "filler" / "models" / "tfidf_lr").mkdir(parents = True)
    (tmp_path / "filler" / "models" / "tfidf_lr" / "model.pkl").write_bytes(b"")
    filler = vm._Filler(tmp_path, seed = 0)

    out = filler.predict("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["echo"] is True
    assert out["slot"] == "a snake game" and out["phrase"] == "Okay, a snake game."
    out = filler.predict("hi there friend")
    assert out["class"] == "minimal" and out["phrase"] == "Sure." and out["slot"] is None
    out = filler.predict("yeah")
    assert out == {"say": False, "class": "none", "confidence": 0.8, "phrase": "", "slot": None, "echo": False, "ms": out["ms"]}


def test_merge_prompt_keeps_corrections_and_additions_apart():
    assert vm.merge_prompt("write a poem about the sea.", "no wait, about mountains") == "write a poem about the sea. Correction: no wait, about mountains"
    assert vm.merge_prompt("write a poem", "and also make it rhyme") == "write a poem. Also: make it rhyme"


def test_runtime_bias_rules():
    even = {"restart": 1 / 3, "ignore": 1 / 3, "queue": 1 / 3}
    corr = vm.apply_runtime_bias(even, "no make it shorter", 3.0)
    assert max(corr, key = corr.get) == "restart"
    fast = vm.apply_runtime_bias(even, "and the title", 0.5)
    assert fast["ignore"] > fast["restart"] == fast["queue"]
    bare = vm.apply_runtime_bias(even, "no no um", 3.0)
    assert bare["restart"] == pytest.approx(bare["ignore"])
    other = vm.apply_runtime_bias(even, "no not you honey", 3.0)
    assert other["ignore"] > 0.6
    assert sum(other.values()) == pytest.approx(1.0)


def test_hand_feature_layout_is_fixed():
    f = vm.hand_features("write a poem", "no wait", 1.0)
    assert len(f) == 21 + len(vm._CUE_WORDS)
    assert f[9] == 1.0 and f[13] == 1.0 and f[17] == 1.0


def test_interject_unpickler_maps_the_training_class():
    assert vm._InterjectUnpickler.find_class(None, "train_tfidf", "Model") is vm._InterjectTfidf


@pytest.mark.skipif(not vm.models_dir().is_dir(), reason = "voice models dir not present")
def test_real_models_end_to_end():
    models = vm.VoiceModels()
    status = models.status()
    assert status == {"judge": True, "filler": True, "interject": True, "dir": str(vm.models_dir()), "errors": {}}, status
    assert models.judge("what's the capital of france") > models.judge("what's the capital of")
    out = models.filler("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["phrase"]
    out = models.interject("write a poem about the sea", "no wait make it about the mountains", 1.0)
    assert out["action"] == "restart" and out["merged_prompt"].startswith("write a poem about the sea. Correction:")
    assert models.interject("write a poem about the sea", "ok", 0.5)["action"] == "ignore"
    assert os.environ.get(vm.ENV_DIR) or models.root == Path(vm.DEFAULT_DIR).expanduser()
