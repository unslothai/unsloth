# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


@pytest.fixture
def decision_mlx(monkeypatch):
    # By path, as unsloth/__init__.py loads it on Apple Silicon; needs neither mlx nor torch to import.
    for name in ("unsloth._decision_common", "_decision_mlx_under_test"):
        monkeypatch.delitem(sys.modules, name, raising = False)
    spec = importlib.util.spec_from_file_location(
        "_decision_mlx_under_test", _MODELS / "decision_mlx.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_install_without_torch_is_told_to_install_it(decision_mlx, monkeypatch):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "torch" else find_spec(name, *args),
    )
    decision_mlx._decision_zoo.cache_clear()
    with pytest.raises(ImportError, match = "decision models on MLX need PyTorch"):
        decision_mlx._decision_zoo()


def test_adapters_refuse_a_full_finetune_before_loading(decision_mlx, monkeypatch, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding = "utf-8")

    def never():
        raise AssertionError("loaded before refusing")

    monkeypatch.setattr(decision_mlx, "_decision_zoo", never)
    with pytest.raises(ValueError, match = "save_pretrained_merged"):
        decision_mlx._load_clef(tmp_path, None, False, True, None, False, {})


def _d1_zoo(tmp_path, family = None):
    pipeline = types.SimpleNamespace(
        family = "d1", processor = "processor", model = types.SimpleNamespace(freeze = lambda: None)
    )
    network = lambda: types.SimpleNamespace(train = lambda: None)
    calls = {}
    zoo = types.SimpleNamespace(
        FAMILIES = {"d1": types.SimpleNamespace(matches = lambda folder: family == "d1")},
        detect_family = lambda folder: pytest.fail(
            "detect_family refuses inputs the Clef loader takes"
        ),
        load_language_model_as_d1 = lambda folder, token = None: pipeline,
        load_language_model_as_clef = lambda *args, **kwargs: pytest.fail(
            "a d1 was given a Clef head"
        ),
        D1Network = lambda pipeline, checkpointing: network(),
        d1_training_network = lambda pipeline, **kwargs: (calls.update(lora = kwargs), network())[1],
        clef_training_network = lambda *args, **kwargs: pytest.fail("a d1 took the Clef network"),
        d1_training_item = lambda *args: calls.update(item = args)
        or {"slots": [], "input_ids": [0] * 8},
        clef_training_item = lambda *args, **kwargs: pytest.fail("a d1 took the Clef item"),
        save_d1_model = lambda *args: calls.update(saved = args),
        save_clef_model = lambda *args: pytest.fail("a d1 took the Clef saver"),
    )
    (tmp_path / "config.json").write_text("{}", encoding = "utf-8")
    return zoo, pipeline, calls


def _load_d1(
    decision_mlx,
    tmp_path,
    head = "d1",
    load_in_4bit = False,
):
    return decision_mlx._load_lm_as_clef(
        tmp_path, None, load_in_4bit, False, None, None, False, False, 0, decision_head = head
    )


@pytest.mark.parametrize("head, family", [("d1", None), ("clef", "d1")])
def test_a_d1_trains_saves_and_tokenizes_as_one(decision_mlx, monkeypatch, tmp_path, head, family):
    zoo, pipeline, calls = _d1_zoo(tmp_path, family)
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    monkeypatch.setattr(
        decision_mlx,
        "_decision_annotate",
        lambda model, **attributes: [setattr(model, *pair) for pair in attributes.items()]
        and model,
    )
    model, tokenizer = _load_d1(decision_mlx, tmp_path, head)
    assert tokenizer == "processor" and model.decision_config["layout"] == "d1"
    common = types.SimpleNamespace(clamp_temperature = float)
    monkeypatch.setattr(decision_mlx, "_laya", lambda: types.SimpleNamespace(common = common))
    decision_mlx._clef_item(zoo, pipeline, "s", {"q": {}}, 64, ["image"], ["clip"])
    assert calls["item"] == (pipeline, "s", {"q": {}}, ["image"], ["clip"])
    decision_mlx._clef_item(zoo, pipeline, "s", {"q": {}}, 8, ["image"], ["clip"])
    with pytest.raises(ValueError, match = "past max_seq_length 7"):
        decision_mlx._clef_item(zoo, pipeline, "s", {"q": {}}, 7, ["image"], ["clip"])
    model = decision_mlx.FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    assert calls["lora"]["r"] == 8 and model.is_clef and model._unsloth_lora
    model.encoder = types.SimpleNamespace(
        trainable_parameters = dict, named_modules = list, freeze = lambda: None
    )
    decision_mlx._decision_save_merged(model, "out")
    assert calls["saved"] == (pipeline, "out", tmp_path, None)
    model.decision_config.update(head_temperature = 2.0, temperature = [1.0, 1.5, 3.0])
    decision_mlx._decision_save_merged(model, "out")
    assert calls["saved"][3] == {"choice": 2.0, "score": 3.0, "noul": 6.0}


def test_a_reloaded_d1_answers_at_its_calibration_until_it_trains(
    decision_mlx, monkeypatch, tmp_path
):
    zoo, *_ = _d1_zoo(tmp_path)
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    monkeypatch.setattr(
        decision_mlx,
        "_decision_annotate",
        lambda model, **attributes: [setattr(model, *pair) for pair in attributes.items()]
        and model,
    )
    buckets = {"noul": 8.0, "choice:small": 2.0}
    (tmp_path / "calibration.json").write_text(
        json.dumps({"temperatures": buckets, "default": 4.0}), encoding = "utf-8"
    )
    model, _ = _load_d1(decision_mlx, tmp_path)
    config = model.decision_config
    served = [config["head_temperature"] * t for t in config["temperature"]]
    assert dict(zip(decision_mlx.QUESTION_TYPES, served)) == {
        "choice": 2.0,
        "score": 4.0,
        "noul": 8.0,
    }
    for fitted in (
        {"noul": 2.0, "choice:small": 10.0, "default": 1.0},
        {"noul": 10.0, "choice:small": 1.0, "default": 2.0},
    ):
        (tmp_path / "calibration.json").write_text(
            json.dumps({"temperatures": fitted, "default": fitted["default"]}), encoding = "utf-8"
        )
        config = decision_mlx._d1_temperatures(tmp_path)
        assert all(0.5 <= t <= 5.0 for t in config["temperature"])
    tuned = decision_mlx.FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    assert (
        tuned.decision_config["head_temperature"] == 1.0
        and tuned.decision_config["temperature"] == [1.0] * 3
    )
    full, _ = decision_mlx._load_lm_as_clef(
        tmp_path, None, False, True, None, None, False, False, 0, decision_head = "d1"
    )
    assert full.decision_config["head_temperature"] == 1.0


def test_a_d1_score_answers_its_median_level(decision_mlx):
    zoo = types.SimpleNamespace(
        d1_answer = lambda pipeline, question, probabilities: {"type": "score", "score": 1.0}
    )
    predicted = decision_mlx._d1_predicted(
        zoo, None, {"type": "score"}, {"0": 0.40, "1": 0.35, "2": 0.25}
    )
    assert predicted["score"] == 1.0 and predicted["answer"] == 1


def test_a_language_model_keeps_the_clef_loader(decision_mlx, monkeypatch, tmp_path):
    zoo, *_ = _d1_zoo(tmp_path)
    zoo.load_language_model_as_clef = lambda *args, **kwargs: types.SimpleNamespace(
        tokenizer = "t",
        model = types.SimpleNamespace(freeze = lambda: None),
        head = types.SimpleNamespace(unfreeze = lambda: None),
    )
    zoo.ClefNetwork = lambda pipeline, checkpointing: types.SimpleNamespace(train = lambda: None)
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    monkeypatch.setattr(decision_mlx, "_decision_annotate", lambda model, **attributes: model)
    assert _load_d1(decision_mlx, tmp_path, "clef")[1] == "t"


def test_a_d1_ignores_4bit_out_loud_and_other_heads_are_refused(
    decision_mlx, monkeypatch, tmp_path
):
    zoo, *_ = _d1_zoo(tmp_path)
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    monkeypatch.setattr(decision_mlx, "_decision_annotate", lambda model, **attributes: model)
    with pytest.warns(UserWarning, match = "load_in_4bit"):
        _load_d1(decision_mlx, tmp_path, load_in_4bit = True)
    with pytest.raises(ValueError, match = "must be one of"):
        _load_d1(decision_mlx, tmp_path, "other")


def test_an_older_zoo_is_told_to_upgrade_for_a_d1(decision_mlx, monkeypatch, tmp_path):
    zoo, *_ = _d1_zoo(tmp_path)
    del zoo.load_language_model_as_d1
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    with pytest.raises(ImportError, match = "newer unsloth-zoo"):
        _load_d1(decision_mlx, tmp_path)


@pytest.mark.parametrize("head", ["d1", "clef"])
def test_a_language_model_naming_its_own_model_file_is_refused(
    decision_mlx, monkeypatch, tmp_path, head
):
    # mlx-vlm / mlx-lm exec a config.json model_file, so it must never reach the zoo loaders.
    zoo, *_ = _d1_zoo(tmp_path)
    zoo.load_language_model_as_d1 = lambda *args, **kwargs: pytest.fail(
        "model_file reached the d1 loader"
    )
    zoo.load_language_model_as_clef = lambda *args, **kwargs: pytest.fail(
        "model_file reached the Clef loader"
    )
    monkeypatch.setattr(decision_mlx, "_decision_zoo", lambda: zoo)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "lfm3_vl", "model_file": "evil.py"}), encoding = "utf-8"
    )
    with pytest.raises(ValueError, match = "repository code"):
        _load_d1(decision_mlx, tmp_path, head)


def test_a_decision_checkpoint_naming_its_own_model_file_is_refused(
    decision_mlx, monkeypatch, tmp_path
):
    monkeypatch.setattr(decision_mlx, "_checkpoint_folder", lambda *args: tmp_path)
    monkeypatch.setattr(decision_mlx, "is_clef_checkpoint", lambda folder: True)
    monkeypatch.setattr(decision_mlx, "_is_plain_lm", lambda *args: False)
    monkeypatch.setattr(
        decision_mlx,
        "_load_clef",
        lambda *args, **kwargs: pytest.fail("model_file reached the loader"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["D1Model"], "model_file": "evil.py"}), encoding = "utf-8"
    )
    with pytest.raises(ValueError, match = "repository code"):
        decision_mlx.FastDecisionModel.from_pretrained(str(tmp_path))
