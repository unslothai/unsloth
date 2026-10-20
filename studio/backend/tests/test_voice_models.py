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
    def predict(
        self,
        text,
        prev_q = False,
    ):
        return {
            "say": True,
            "class": "code",
            "confidence": 0.8,
            "phrase": "On it.",
            "slot": None,
            "echo": False,
            "ms": 0.1,
        }


class _FakeInterject:
    def predict(
        self,
        prompt,
        utterance,
        elapsed_s = 2.0,
    ):
        return {
            "action": "ignore",
            "confidence": 0.7,
            "probs": {"restart": 0.1, "ignore": 0.7, "queue": 0.2},
            "merged_prompt": None,
        }


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
        tmp_path,
        monkeypatch,
        judge = lambda root: _FakeJudge(),
        filler = lambda root: _FakeFiller(),
        interject = lambda root: _FakeInterject(),
    )
    assert models.judge("are you there?") == 0.9
    assert models.filler("write me a snake game")["class"] == "code"
    assert models.interject("write a poem", "ok", 1.0)["action"] == "ignore"
    status = models.status()
    assert {k: status[k] for k in vm.MODELS} == {"judge": True, "filler": True, "interject": True}
    assert status["dir"] == str(tmp_path) and status["errors"] == {}
    assert status["backends"] == {
        "judge": None,
        "filler": None,
        "interject": None,
        "filler_echo": None,
    }


def test_local_dir_wins_and_the_hub_is_the_fallback(tmp_path, monkeypatch):
    local = tmp_path / "interject" / "models" / "tfidf_lr"
    local.mkdir(parents = True)
    (local / "model.pkl").write_bytes(b"x")
    assert vm.resolve_model_dir("interject", tmp_path, "someone") == tmp_path / "interject"
    with pytest.raises(FileNotFoundError):
        vm.resolve_model_dir("judge", tmp_path / "nope", None)
    assert vm.resolve_model_dir("judge", tmp_path, None) == tmp_path / "judge"
    seen = {}
    import huggingface_hub

    def fake_download(repo, allow_patterns):
        seen["repo"], seen["patterns"] = repo, allow_patterns
        return str(tmp_path / "snap")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert vm.resolve_model_dir("judge", tmp_path, "someone") == tmp_path / "snap"
    assert seen["repo"] == "someone/voice-judge" and seen["patterns"] == vm.HUB_FILES["judge"]


def test_env_var_picks_the_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(vm.ENV_DIR, str(tmp_path))
    assert vm.models_dir() == tmp_path
    monkeypatch.delenv(vm.ENV_DIR)
    assert vm.models_dir() is None


def test_nothing_is_fetched_without_a_dir_or_hub(monkeypatch):
    # No personal default: unset, the routes answer 503 and the app keeps its word rules.
    import huggingface_hub

    monkeypatch.delenv(vm.ENV_DIR, raising = False)
    monkeypatch.delenv(vm.ENV_HUB, raising = False)
    monkeypatch.delenv("UNSLOTH_VOICE_DECISION_CHECKPOINT", raising = False)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("fetched")
    )
    assert vm.models_hub() is None
    models = vm.VoiceModels()
    assert [models.available(name) for name in vm.MODELS] == [False, False, False]
    assert "no voice models configured" in models.reason("judge")
    assert models.status()["dir"] is None


def test_decision_filler_uses_the_bundled_bank_without_a_dir(monkeypatch):
    from types import SimpleNamespace

    from core.inference import voice_decisions as vd
    from core.systemone import catalog

    monkeypatch.delenv(vm.ENV_DIR, raising = False)
    monkeypatch.delenv(vm.ENV_HUB, raising = False)
    monkeypatch.setenv(vd.ENV_CHECKPOINT, "laya-ft:run")
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(name = name, layout = "laya"))
    monkeypatch.setattr(
        vd.DecisionVoice,
        "ask",
        lambda self, state, q: {
            "type": "choice",
            "choice": "code",
            "confidence": 0.9,
            "probabilities": {"code": 0.9},
        },
    )
    out = vm.VoiceModels().filler("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["phrase"]


def test_featurize_adds_the_context_token():
    assert vm.featurize("  Hello There ") == "hello there <end>"
    assert vm.featurize("the second one", prev_q = True) == "<prevq> the second one <end>"


@pytest.mark.parametrize(
    "text, slot",
    [
        ("write me a snake game in python", "a snake game"),
        ("why is my install failing with this error", "the install error"),
        ("can you plan my week", "your week"),
        ("what's your skill level", "my skill level"),
        ("tell me about your training data", "my training data"),
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
        "classes": {
            "code": {"phrases": [{"text": "On it.", "weight": 1}]},
            "minimal": {"phrases": [{"text": "Sure.", "weight": 1}]},
        },
        "echo_templates": {
            "templates": [{"text": "okay, {slot}.", "weight": 1}],
            "mix": {"echo_when_slot": 1.0},
        },
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

    def fake_pickle_load(unpickler):
        return {"gate": _Gate(), "cls": _Cls()}

    monkeypatch.setattr(vm._SklearnUnpickler, "load", fake_pickle_load)
    (tmp_path / "filler" / "models" / "tfidf_lr").mkdir(parents = True)
    (tmp_path / "filler" / "models" / "tfidf_lr" / "model.pkl").write_bytes(b"")
    filler = vm._Filler(tmp_path / "filler", seed = 0)
    assert filler.backend == "tfidf" and filler.echo_backend == "rules"

    out = filler.predict("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["echo"] is True
    assert out["slot"] == "a snake game" and out["phrase"] == "Okay, a snake game."
    out = filler.predict("hi there friend")
    assert out["class"] == "minimal" and out["phrase"] == "Sure." and out["slot"] is None
    out = filler.predict("yeah")
    assert out == {
        "say": False,
        "class": "none",
        "confidence": 0.8,
        "phrase": "",
        "slot": None,
        "echo": False,
        "ms": out["ms"],
    }


def test_merge_prompt_keeps_corrections_and_additions_apart():
    assert (
        vm.merge_prompt("write a poem about the sea.", "no wait, about mountains")
        == "write a poem about the sea. Correction: no wait, about mountains"
    )
    assert (
        vm.merge_prompt("write a poem", "and also make it rhyme")
        == "write a poem. Also: make it rhyme"
    )


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


def _write_trained_interject(root: Path) -> None:
    """A model.pkl shaped like train_tfidf.Model, pickled under that module name."""
    import pickle
    import sys
    import types

    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    module = types.ModuleType("train_tfidf")

    class Model:
        pass

    Model.__module__ = "train_tfidf"
    Model.__qualname__ = "Model"
    module.Model = Model
    sys.modules["train_tfidf"] = module
    try:
        rows = [
            ("write a poem about the sea", "no wait make it about mountains", 2.0, "restart"),
            ("summarize this article", "mm hmm", 1.0, "ignore"),
            ("explain recursion", "and also show an example in python", 3.0, "queue"),
        ] * 2
        m = Model()
        m.vu_w = TfidfVectorizer().fit([r[1] for r in rows])
        m.vu_c = TfidfVectorizer(analyzer = "char_wb", ngram_range = (2, 3)).fit([r[1] for r in rows])
        m.vp_w = TfidfVectorizer().fit([r[0] for r in rows])
        h = np.array([vm.hand_features(r[0], r[1], r[2]) for r in rows])
        m.sc = StandardScaler().fit(h)
        x = sp.hstack(
            [
                m.vu_w.transform([r[1] for r in rows]),
                m.vu_c.transform([r[1] for r in rows]),
                0.4 * m.vp_w.transform([r[0] for r in rows]),
                sp.csr_matrix(m.sc.transform(h)),
            ]
        ).tocsr()
        m.clf = LogisticRegression(max_iter = 200).fit(x, [r[3] for r in rows])
        folder = root / "models" / "tfidf_lr"
        folder.mkdir(parents = True)
        (folder / "model.pkl").write_bytes(pickle.dumps(m))
    finally:
        del sys.modules["train_tfidf"]


def test_interject_loads_a_trained_sklearn_pickle(tmp_path):
    pytest.importorskip("sklearn")
    _write_trained_interject(tmp_path)
    verdict = vm._Interject(tmp_path).predict(
        "write a poem about the sea", "no wait make it about mountains"
    )
    assert verdict["action"] in vm.INTERJECT_ACTIONS


def test_filler_tfidf_loads_a_trained_sklearn_pickle(tmp_path):
    pytest.importorskip("sklearn")
    import pickle

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = [vm.featurize(t) for t in ("write some code", "hello there", "fix this bug", "thanks")]
    gate = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, [1, 0, 1, 0])
    cls = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(
        texts, ["code", "chat", "code", "chat"]
    )
    (tmp_path / "model.pkl").write_bytes(pickle.dumps({"gate": gate, "cls": cls}))
    label, confidence = vm._TfidfClassifier(tmp_path).classify("write some code", False)
    assert label in ("code", "chat", "none") and 0.0 <= confidence <= 1.0


class _RunsCode:
    def __reduce__(self):
        return (os.system, ("true",))


@pytest.mark.parametrize("loader", ["interject", "filler"])
def test_voice_pickles_refuse_a_function_global(tmp_path, loader):
    import pickle

    folder = tmp_path / "models" / "tfidf_lr" if loader == "interject" else tmp_path
    folder.mkdir(parents = True, exist_ok = True)
    (folder / "model.pkl").write_bytes(pickle.dumps({"gate": _RunsCode(), "cls": None}))
    with pytest.raises(pickle.UnpicklingError, match = "not allowed"):
        if loader == "interject":
            vm._Interject(tmp_path)
        else:
            vm._TfidfClassifier(tmp_path)


class _FakeSession:
    """Stands in for onnxruntime: 12 class logits, or per-token B/I/O logits."""

    def __init__(self, logits_for):
        self.logits_for = logits_for

    def get_inputs(self):
        return [type("I", (), {"name": "input_ids"})(), type("I", (), {"name": "attention_mask"})()]

    def run(self, _outputs, feed):
        assert set(feed) == {"input_ids", "attention_mask"}
        assert feed["input_ids"].dtype.name == "int64"
        return [self.logits_for(feed["input_ids"].shape[1])]


class _FakeTokenizer:
    def __init__(self, path):
        self.max = None

    def enable_truncation(self, n):
        self.max = n

    def encode(self, text):
        words, ids, offsets, pos = text.split(), [], [], 0
        for w in words:
            a = text.index(w, pos)
            ids.append(len(ids) + 5)
            offsets.append((a, a + len(w)))
            pos = a + len(w)
        ids, offsets = ids[: self.max], offsets[: self.max]
        return type("E", (), {"ids": ids, "attention_mask": [1] * len(ids), "offsets": offsets})()


def _onnx_layout(tmp_path, classes):
    for name in (vm.FILLER_ONNX, vm.ECHO_ONNX):
        d = tmp_path / "filler" / "models" / name
        d.mkdir(parents = True)
        (d / "model_int8.onnx").write_bytes(b"")
        (d / "tokenizer.json").write_text("{}")
    (tmp_path / "filler" / "models" / vm.FILLER_ONNX / "classes.json").write_text(
        json.dumps(classes)
    )
    (tmp_path / "filler" / "phrases.json").write_text(
        json.dumps(
            {
                "classes": {"code": {"phrases": [{"text": "On it.", "weight": 1}]}},
                "echo_templates": {
                    "templates": [{"text": "okay, {slot}.", "weight": 1}],
                    "mix": {"echo_when_slot": 1.0},
                },
            }
        )
    )


def test_onnx_filler_is_preferred_when_its_folder_exists(tmp_path, monkeypatch):
    import numpy as np
    import types

    classes = ["none", "minimal", "code"]
    _onnx_layout(tmp_path, classes)

    def cls_logits(n):
        return np.array([[0.0, 0.0, 4.0]], dtype = np.float32)

    def tag_logits(
        n,
    ):  # "write me a snake game in python": B on "a", I on "snake", "game", O elsewhere
        bio = np.zeros((1, n, 3), dtype = np.float32)
        bio[0, :, 0] = 3.0
        if n >= 5:
            bio[0, 2] = [0, 3.0, 0]
            bio[0, 3] = [0, 0, 3.0]
            bio[0, 4] = [0, 0, 3.0]
        return bio

    sessions = iter([_FakeSession(cls_logits), _FakeSession(tag_logits)])
    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        types.SimpleNamespace(
            SessionOptions = lambda: types.SimpleNamespace(intra_op_num_threads = 0),
            InferenceSession = lambda path, so, providers: next(sessions),
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer = types.SimpleNamespace(from_file = _FakeTokenizer)),
    )

    filler = vm._Filler(tmp_path / "filler", seed = 0)
    assert filler.backend == "onnx" and filler.echo_backend == "onnx"
    out = filler.predict("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["confidence"] > 0.9
    assert (
        out["slot"] == "a snake game"
        and out["echo"] is True
        and out["phrase"] == "Okay, a snake game."
    )
    assert set(out) == {"say", "class", "confidence", "phrase", "slot", "echo", "ms"}


def test_onnx_echo_tagger_falls_back_to_rules_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vm,
        "_OnnxClassifier",
        lambda d: type("C", (), {"classify": lambda self, t, q: ("code", 0.9)})(),
    )
    (tmp_path / "filler" / "models" / vm.FILLER_ONNX).mkdir(parents = True)
    (tmp_path / "filler" / "phrases.json").write_text(
        json.dumps(
            {
                "classes": {"code": {"phrases": [{"text": "On it.", "weight": 1}]}},
                "echo_templates": {
                    "templates": [{"text": "okay, {slot}.", "weight": 1}],
                    "mix": {"echo_when_slot": 0.0},
                },
            }
        )
    )
    filler = vm._Filler(tmp_path / "filler", seed = 0)
    assert filler.backend == "onnx" and filler.echo_backend == "rules"
    out = filler.predict("why is my install failing with this error")
    assert out["slot"] == "the install error" and out["echo"] is False and out["phrase"] == "On it."


@pytest.mark.skipif(
    vm.models_dir() is None or not vm.models_dir().is_dir(), reason = "voice models dir not set"
)
def test_real_models_end_to_end():
    models = vm.VoiceModels()
    status = models.status()
    assert {k: status[k] for k in vm.MODELS} == {
        "judge": True,
        "filler": True,
        "interject": True,
    }, status
    assert status["errors"] == {} and status["dir"] == str(vm.models_dir())
    assert status["backends"] == {
        "judge": "onnx",
        "filler": "onnx",
        "interject": "tfidf",
        "filler_echo": "onnx",
    }
    assert models.judge("what's the capital of france") > models.judge("what's the capital of")
    out = models.filler("write me a snake game in python")
    assert out["say"] and out["class"] == "code" and out["phrase"]
    assert (
        out["slot"] == models._models["filler"].echo.extract("write me a snake game in python")[0]
    )
    assert out["ms"] < 100
    out = models.interject("write a poem about the sea", "no wait make it about the mountains", 1.0)
    assert out["action"] == "restart" and out["merged_prompt"].startswith(
        "write a poem about the sea. Correction:"
    )
    assert models.interject("write a poem about the sea", "ok", 0.5)["action"] == "ignore"
    assert models.root == vm.models_dir()


_AGREEMENT_OR_REFUSAL = (
    r"\b(ok|okay|sure|got it|alright|all right|yes|yeah|yep|no|nope|no problem|sounds good"
    r"|will do|on it|of course|absolutely|good question|(?<!if )i can do that|can't|cannot"
    r"|right|almost there|here we go)\b"
)


def test_filler_bank_never_sounds_like_an_answer():
    # The model decides yes or no; a filler said before it must not. This is the bank that ships.
    import re

    bank = json.loads(vm.BUNDLED_PHRASES.read_text(encoding = "utf-8"))
    texts = [p["text"] for c in bank["classes"].values() for p in c["phrases"]]
    texts += [t["text"] for t in bank["echo_templates"]["templates"]]
    assert len(texts) > 50
    bad = [t for t in texts if re.search(_AGREEMENT_OR_REFUSAL, t, re.I)]
    assert bad == []
