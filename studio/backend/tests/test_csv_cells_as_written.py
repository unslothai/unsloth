# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

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
from hub.services.datasets.local import _load_local_preview_slice  # noqa: E402
from hub.utils.dataset_format import check_dataset_format  # noqa: E402
from utils.datasets.format_detection import detect_vlm_dataset_structure  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


ALPACA_CSV = (
    "instruction,output\n"
    "List the side effects.,None\n"
    "What is the airline code?,N/A\n"
    "What does the API return on a miss?,null\n"
    "Price for item 7?,NA\n"
    "What is 8 x 9?,72\n"
    "What is 23 / 4?,5.75\n"
    "Zip code of Holtsville NY?,00501\n"
    "Say nothing.,\n"
)
WRITTEN = ["None", "N/A", "null", "NA", "72", "5.75", "00501", ""]


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


def _responses(dataset):
    return [text.split("### Response:\n", 1)[1].removesuffix("</s>") for text in dataset["text"]]


def test_uploaded_csv_trains_na_and_none_answers_as_written(trainer, tmp_path):
    train = tmp_path / "train.csv"
    train.write_text(ALPACA_CSV)
    evals = tmp_path / "eval.csv"
    evals.write_text(ALPACA_CSV)

    result = trainer.load_and_format_dataset(
        None,
        local_datasets = [str(train)],
        local_eval_datasets = [str(evals)],
        eval_steps = 0.1,
    )

    assert result is not None
    dataset_info, eval_dataset = result
    assert dataset_info["final_format"] == "alpaca"
    assert _responses(dataset_info["dataset"]) == WRITTEN
    assert _responses(eval_dataset) == WRITTEN


def test_upload_preview_shows_csv_cells_as_written(tmp_path):
    path = tmp_path / "upload.csv"
    path.write_text(ALPACA_CSV)

    preview, _ = _load_local_preview_slice(dataset_path = path, train_split = "train", preview_size = 10)

    assert [row["output"] for row in preview] == WRITTEN[:-1] + [None]


def test_cpt_csv_still_trains_the_text_column_not_the_id(trainer, tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("id,passage\n1,The first document.\n2,The second document.\n")

    dataset_info, _ = trainer.load_and_format_dataset(None, local_datasets = [str(path)], is_cpt = True)

    assert list(dataset_info["dataset"]["text"]) == [
        "The first document.</s>",
        "The second document.</s>",
    ]


def test_vision_csv_still_trains_the_caption_not_a_numeric_label(tmp_path):
    path = tmp_path / "vision.csv"
    path.write_text(
        "image,caption,label\nhttps://example.com/a.png,cat,1042\nhttps://example.com/b.png,dog,7\n"
    )

    preview, _ = _load_local_preview_slice(dataset_path = path, train_split = "train", preview_size = 10)

    assert check_dataset_format(preview, is_vlm = True)["detected_text_column"] == "caption"
    assert detect_vlm_dataset_structure(preview)["text_column"] == "caption"
