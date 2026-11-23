# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    t = tmod.UnslothTrainer()
    t.model_name = "Qwen2ForCausalLM"
    t.tokenizer = _Tokenizer()
    return t


_ROWS = [{"instruction": f"Say {w}", "input": "", "output": w} for w in ("hi", "yes", "no")]


@pytest.mark.parametrize("as_dict", [False, True])
def test_a_saved_to_disk_folder_trains_as_saved(trainer, tmp_path, as_dict):
    from datasets import Dataset, DatasetDict

    saved = tmp_path / "saved"
    dataset = Dataset.from_list(_ROWS)
    (DatasetDict(train = dataset) if as_dict else dataset).save_to_disk(str(saved))

    result = trainer.load_and_format_dataset(None, local_datasets = [str(saved)])

    assert result is not None
    dataset_info, _ = result
    assert dataset_info["final_format"] == "alpaca"
    assert len(dataset_info["dataset"]) == 3


@pytest.mark.parametrize("name", ["state.json", "dataset_dict.json"])
def test_a_folder_with_a_data_file_named_like_a_manifest_still_loads_its_rows(
    trainer, tmp_path, name
):
    folder = tmp_path / "rows"
    folder.mkdir()
    (folder / name).write_text("\n".join(json.dumps(row) for row in _ROWS))

    result = trainer.load_and_format_dataset(None, local_datasets = [str(folder)])

    assert result is not None
    assert len(result[0]["dataset"]) == 3


@pytest.mark.parametrize("layout", ["dataset", "dict", "state_json_rows"])
def test_the_saved_dataset_loader_shared_with_the_mlx_worker(tmp_path, layout):
    from datasets import Dataset, DatasetDict
    from utils.datasets.saved import load_saved_dataset

    folder = tmp_path / layout
    if layout == "state_json_rows":
        folder.mkdir()
        (folder / "state.json").write_text("\n".join(json.dumps(row) for row in _ROWS))
        assert load_saved_dataset(folder) is None
        return
    dataset = Dataset.from_list(_ROWS)
    (DatasetDict(train = dataset) if layout == "dict" else dataset).save_to_disk(str(folder))

    assert load_saved_dataset(folder)["output"] == ["hi", "yes", "no"]


def test_the_mlx_worker_resolves_a_studio_dataset_name_before_the_saved_probe(
    tmp_path, monkeypatch
):
    from datasets import Dataset
    import utils.paths
    from core.training.worker import _resolve_mlx_local_dataset_path
    from utils.datasets.saved import load_saved_dataset

    saved = tmp_path / "datasets" / "my_saved"
    Dataset.from_list(_ROWS).save_to_disk(str(saved))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils.paths, "resolve_dataset_path", lambda name: saved)

    assert load_saved_dataset("my_saved") is None
    assert load_saved_dataset(_resolve_mlx_local_dataset_path("my_saved"))["output"] == [
        "hi",
        "yes",
        "no",
    ]


def test_a_saved_to_disk_eval_folder_loads_as_saved(trainer, tmp_path):
    from datasets import Dataset

    train, evals = tmp_path / "train", tmp_path / "eval"
    Dataset.from_list(_ROWS).save_to_disk(str(train))
    Dataset.from_list(_ROWS[:2]).save_to_disk(str(evals))

    result = trainer.load_and_format_dataset(
        None, local_datasets = [str(train)], local_eval_datasets = [str(evals)], eval_steps = 0.5
    )

    assert result is not None
    assert len(result[1]) == 2


@pytest.mark.parametrize("escape", ["state_file", "dict_split"])
def test_a_saved_manifest_pointing_outside_its_folder_is_refused(tmp_path, escape):
    from datasets import Dataset, DatasetDict
    from utils.datasets.saved import load_saved_dataset

    Dataset.from_list(_ROWS).save_to_disk(str(tmp_path / "outside"))
    folder = tmp_path / "saved"
    if escape == "state_file":
        Dataset.from_list(_ROWS).save_to_disk(str(folder))
        state = json.loads((folder / "state.json").read_text())
        arrow = next((tmp_path / "outside").glob("*.arrow")).name
        state["_data_files"] = [{"filename": f"../outside/{arrow}"}]
        (folder / "state.json").write_text(json.dumps(state))
    else:
        DatasetDict(train = Dataset.from_list(_ROWS)).save_to_disk(str(folder))
        (folder / "dataset_dict.json").write_text(json.dumps({"splits": ["../outside"]}))

    with pytest.raises(ValueError, match = "outside its folder"):
        load_saved_dataset(folder)
