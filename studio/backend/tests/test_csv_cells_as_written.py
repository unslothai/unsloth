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

    numbers = tmp_path / "numbers.csv"
    numbers.write_text("question,answer\nq1,72\nq2,\nq3,00501\n")
    preview, _ = _load_local_preview_slice(
        dataset_path = numbers, train_split = "train", preview_size = 10
    )
    assert [row["answer"] for row in preview] == ["72", None, "00501"]


def test_cpt_csv_still_trains_the_text_column_not_the_id(trainer, tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text(
        "id,score,is_clean,notes,passage\n1,1e5,True,,NA\n2,.5,False,N/A,The second document.\n"
    )

    dataset_info, _ = trainer.load_and_format_dataset(None, local_datasets = [str(path)], is_cpt = True)

    assert list(dataset_info["dataset"]["text"]) == ["NA</s>", "The second document.</s>"]

    mixed = tmp_path / "mixed.csv"
    mixed.write_text("content,source\n101,corpus\nTrue,corpus\n")
    dataset_info, _ = trainer.load_and_format_dataset(
        None, local_datasets = [str(mixed)], is_cpt = True
    )
    assert list(dataset_info["dataset"]["text"]) == ["101</s>", "True</s>"]

    numbers = tmp_path / "numbers.jsonl"
    numbers.write_text(
        '{"content": "101", "source": "corpus"}\n{"content": "102", "source": "corpus"}\n'
    )
    dataset_info, _ = trainer.load_and_format_dataset(
        None, local_datasets = [str(numbers)], is_cpt = True
    )
    assert list(dataset_info["dataset"]["text"]) == ["101</s>", "102</s>"]


def test_cpt_csv_trains_the_body_column_and_keeps_the_warning(trainer, tmp_path):
    path = tmp_path / "posts.csv"
    path.write_text(
        "title,body\nFirst post,The first post has a long body of text.\n"
        "Second post,The second post has an even longer body of text.\n"
    )

    dataset_info, _ = trainer.load_and_format_dataset(None, local_datasets = [str(path)], is_cpt = True)

    assert list(dataset_info["dataset"]["text"]) == [
        "The first post has a long body of text.</s>",
        "The second post has an even longer body of text.</s>",
    ]
    assert any("auto-selecting 'body'" in w for w in trainer.training_progress.warnings)


def test_cpt_csv_with_an_id_column_trains_the_text_without_a_column_warning(trainer, tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("id,passage\n1,The first document.\n2,The second document.\n")

    dataset_info, _ = trainer.load_and_format_dataset(None, local_datasets = [str(path)], is_cpt = True)

    assert list(dataset_info["dataset"]["text"]) == [
        "The first document.</s>",
        "The second document.</s>",
    ]
    assert not any("auto-selecting" in w for w in trainer.training_progress.warnings)


def test_csv_files_missing_a_column_still_load_together(trainer, tmp_path):
    wide = tmp_path / "a.csv"
    wide.write_text("instruction,input,output\nZip code?,Holtsville,00501\n")
    narrow = tmp_path / "b.csv"
    narrow.write_text("instruction,output\nPrice?,NA\n")

    dataset_info, _ = trainer.load_and_format_dataset(None, local_datasets = [str(wide), str(narrow)])

    assert len(dataset_info["dataset"]) == 2


def test_vision_detection_skips_typed_cells_only_in_csv_read_as_text(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"")
    path = tmp_path / "vision.csv"
    path.write_text(f"file_name,image,caption,label\n101,{image},cat,False\n102,{image},dog,True\n")

    preview, _ = _load_local_preview_slice(dataset_path = path, train_split = "train", preview_size = 10)

    detected = check_dataset_format(preview, is_vlm = True)
    assert (detected["detected_image_column"], detected["detected_text_column"]) == (
        "image",
        "caption",
    )
    structure = detect_vlm_dataset_structure(preview)
    assert (structure["image_column"], structure["text_column"]) == ("image", "caption")

    unresolved = tmp_path / "unresolved.csv"
    unresolved.write_text("photo,file_name,caption\n101,missing/a.png,cat\n")
    preview, _ = _load_local_preview_slice(
        dataset_path = unresolved, train_split = "train", preview_size = 10
    )
    assert detect_vlm_dataset_structure(preview)["image_column"] == "file_name"

    vqa = tmp_path / "vqa.jsonl"
    vqa.write_text(
        '{"image": "https://example.com/a.png", "question": "How many?", "answer": "2"}\n'
    )
    preview, _ = _load_local_preview_slice(dataset_path = vqa, train_split = "train", preview_size = 10)
    assert check_dataset_format(preview, is_vlm = True)["detected_text_column"] == "answer"
    assert detect_vlm_dataset_structure(preview)["text_column"] == "answer"

    ocr = tmp_path / "ocr.csv"
    ocr.write_text(
        "image,caption\n"
        + "https://example.com/a.png,12345678\n" * 10
        + "https://example.com/b.png,hello\n"
    )
    preview, _ = _load_local_preview_slice(dataset_path = ocr, train_split = "train", preview_size = 10)
    assert check_dataset_format(preview, is_vlm = True)["detected_text_column"] == "caption"
    assert detect_vlm_dataset_structure(preview)["text_column"] == "caption"

    missing = tmp_path / "missing.csv"
    missing.write_text(
        "image,caption,text\nhttps://example.com/a.png,None,hi\nhttps://example.com/b.png,a dog,yo\n"
    )
    preview, _ = _load_local_preview_slice(
        dataset_path = missing, train_split = "train", preview_size = 10
    )
    assert check_dataset_format(preview, is_vlm = True)["detected_text_column"] == "text"
    assert detect_vlm_dataset_structure(preview)["text_column"] == "text"
