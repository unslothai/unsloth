# SPDX-License-Identifier: AGPL-3.0-only
"""Voice mode on a Decision fine-tune: state shapes match the v2 training set, answers map to the app's wire."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.inference import voice_decisions as vd


def _voice(monkeypatch, answers):
    from core.systemone import catalog

    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(name = name, layout = "laya"))
    calls = []

    def decide(checkpoint, state, question):
        calls.append((state, json.loads(json.dumps(question))))
        return {
            "answers": {next(iter(question)): answers[next(iter(question))]},
            "truncated": False,
        }

    return vd.DecisionVoice("laya-ft:run", decide = decide), calls


def test_states_match_training_builder():
    assert vd.judge_state("Go to the") == "Go to the"
    assert vd.filler_state("yes", True) == "(the assistant just asked them a question)\nyes"
    assert vd.filler_state("what is love", False) == "what is love"
    assert vd.interject_state("how do i x", "and in python", 4.6) == (
        "Request: how do i x\nSpoken over the thinking (5 seconds into thinking): and in python"
    )
    assert (
        vd.interject_state("how do i x", "ok", 0)
        == "Request: how do i x\nSpoken over the thinking: ok"
    )


def test_judge_is_the_noul_probability(monkeypatch):
    voice, calls = _voice(
        monkeypatch, {"turn_done": {"type": "noul", "noul": 0.83, "confidence": 0.83}}
    )
    assert voice.judge("are you there") == pytest.approx(0.83)
    assert calls[0][1] == vd.TURN_DONE


def test_filler_class_and_its_probability(monkeypatch):
    bank = {"classes": {"minimal": {"desc": "short"}, "none": {"desc": "nothing"}}}
    voice, calls = _voice(
        monkeypatch,
        {
            "filler": {
                "type": "choice",
                "choice": "minimal",
                "confidence": 0.4,
                "probabilities": {"minimal": 0.7, "none": 0.3},
            }
        },
    )
    clf = vd.DecisionFillerClassifier(voice, bank)
    assert clf.classify("what's the point of art", False) == ("minimal", 0.7)
    assert calls[0][1]["filler"]["criteria"] == {"minimal": "short", "none": "nothing"}


@pytest.mark.parametrize(
    ("choice", "action", "merged"),
    [
        ("amend", "restart", "write a poem\nabout cats"),
        ("replace", "restart", "about cats"),
        ("queue", "queue", None),
        ("ignore", "ignore", None),
    ],
)
def test_interject_maps_four_classes_onto_three_actions(monkeypatch, choice, action, merged):
    probs = {"amend": 0.1, "replace": 0.2, "queue": 0.3, "ignore": 0.4}
    probs[choice] = 0.9
    voice, _ = _voice(
        monkeypatch,
        {
            "interject": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.5,
                "probabilities": probs,
            }
        },
    )
    out = vd.DecisionInterject(voice).predict("write a poem", "about cats", 3.0)
    assert out["action"] == action and out["kind"] == choice
    assert out["merged_prompt"] == merged
    assert out["confidence"] == 0.9
    assert out["probs"]["restart"] == pytest.approx(probs["amend"] + probs["replace"])
    assert "ms" in out


def test_voice_models_routes_through_the_checkpoint(monkeypatch, tmp_path):
    from core.inference import voice_models as vm
    from core.systemone import catalog

    monkeypatch.setenv(vd.ENV_CHECKPOINT, "laya-ft:run")
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(name = name, layout = "laya"))
    monkeypatch.setattr(
        vd.DecisionVoice,
        "ask",
        lambda self, state, q: {"type": "noul", "noul": 0.9, "confidence": 0.9},
    )
    models = vm.VoiceModels(root = tmp_path)
    assert models.judge("done") == pytest.approx(0.9)
    assert models.backend("judge") == "decision:laya"
    assert models.status()["backends"]["judge"] == "decision:laya"


def test_unset_env_keeps_the_v1_loaders(monkeypatch, tmp_path):
    from core.inference import voice_models as vm

    monkeypatch.delenv(vd.ENV_CHECKPOINT, raising = False)
    models = vm.VoiceModels(root = tmp_path)
    assert (
        models.available("judge") is False
    )  # nothing under tmp_path, and no checkpoint configured
    assert "decision" not in models._models
