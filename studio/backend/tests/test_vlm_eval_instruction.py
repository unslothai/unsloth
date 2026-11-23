# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A vision dataset without a question column trains on one fixed prompt; the eval split and the row order must not change it."""

import ast
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

from utils.datasets import format_and_template_dataset, llm_assist

HELPER_SENTENCE = "Describe what this picture shows."
MEDICAL = "Analyze this medical image and describe the key findings."
LATEX = "Convert this image to LaTeX notation."


@pytest.fixture(autouse = True)
def _helper_writes_one_sentence(monkeypatch):
    monkeypatch.setattr(
        llm_assist,
        "llm_generate_vlm_instruction",
        lambda **kwargs: {"instruction": HELPER_SENTENCE, "confidence": 0.85},
    )


def _format(
    captions,
    dataset_name,
    vlm_fallback_instruction = None,
):
    ds = Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8))] * len(captions), "caption": captions}
    )
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = dataset_name,
        **(
            {}
            if vlm_fallback_instruction is None
            else {"vlm_fallback_instruction": vlm_fallback_instruction}
        ),
    )
    assert info["success"], info["errors"]
    return info


def _prompts(info):
    return {
        next(part["text"] for part in row["messages"][0]["content"] if part["type"] == "text")
        for row in info["dataset"]
    }


def test_eval_split_reuses_the_train_instruction():
    train = _format(["Radiograph shows a lytic lesion.", "Axial view of the knee."], "org/scans")
    eval_ = _format(
        ["Sagittal view of the spine.", "Coronal slice of the wrist."],
        "org/scans",
        vlm_fallback_instruction = train.get("vlm_instruction"),
    )

    assert _prompts(train) == {MEDICAL}
    assert _prompts(eval_) == {MEDICAL}


@pytest.mark.parametrize(
    "captions, dataset_name, expected",
    [
        (["Axial view of the knee.", "Radiograph shows a lytic lesion."], "org/scans", MEDICAL),
        # Row 0 also holds a long word, so the OCR hint ties LaTeX on it; only row 0 matches OCR.
        ([r"\displaystyle \frac{a}{b}", "x^{2}", r"\sqrt{y}"], "org/latex_ocr", LATEX),
    ],
)
def test_instruction_does_not_hinge_on_the_first_row(captions, dataset_name, expected):
    # A max_steps run shuffles train by random_seed before formatting, which picks the first row.
    assert _prompts(_format(captions, dataset_name)) == {expected}


def test_question_column_rows_report_no_fixed_instruction():
    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))] * 2,
            "question": ["What bone is this?", "Is it broken?"],
            "answer": ["Femur.", "No."],
        }
    )
    info = format_and_template_dataset(
        ds, model_name = "unsloth/Qwen2.5-VL-3B-Instruct", tokenizer = None, is_vlm = True
    )

    assert info["vlm_instruction"] is None
    assert _prompts(info) == {"What bone is this?", "Is it broken?"}


def _eval_split_calls(module):
    source = Path(__file__).resolve().parents[1] / "core" / "training" / module
    calls = [
        node
        for node in ast.walk(ast.parse(source.read_text(encoding = "utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "format_and_template_dataset"
        and getattr(node.args[0], "id", None) == "eval_dataset"
        and not any(
            k.arg == "is_vlm" and getattr(k.value, "value", None) is False for k in node.keywords
        )
    ]
    assert calls
    return calls


TRAINING_MODULES = [("trainer.py", "dataset_info"), ("worker.py", "vlm_info")]


@pytest.mark.parametrize("module, train_result", TRAINING_MODULES)
def test_training_formats_the_eval_split_with_the_train_instruction(module, train_result):
    for call in _eval_split_calls(module):
        keywords = {k.arg: k.value for k in call.keywords}
        assert "vlm_fallback_instruction" in keywords
        names = {
            n.id for n in ast.walk(keywords["vlm_fallback_instruction"]) if isinstance(n, ast.Name)
        }
        assert train_result in names


@pytest.mark.parametrize("module, train_result", TRAINING_MODULES)
def test_eval_split_questions_survive_a_caption_only_train_split(module, train_result):
    # Caption-only train infers a fixed prompt; an eval split with its own questions must keep them.
    train = _format(["Radiograph shows a lytic lesion.", "Axial view of the knee."], "org/scans")
    assert train["vlm_instruction"] == MEDICAL
    eval_ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))] * 2,
            "question": ["What bone is this?", "Is it broken?"],
            "answer": ["Femur.", "No."],
        }
    )
    for call in _eval_split_calls(module):
        # Replay the call site's keywords that read the training result, against the real formatter.
        replayed = {
            k.arg: eval(compile(ast.Expression(k.value), module, "eval"), {train_result: train})
            for k in call.keywords
            if k.arg
            and train_result in {n.id for n in ast.walk(k.value) if isinstance(n, ast.Name)}
        }
        assert replayed
        info = format_and_template_dataset(
            eval_ds,
            model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
            tokenizer = None,
            is_vlm = True,
            dataset_name = "org/scans",
            **replayed,
        )
        assert info["success"], info["errors"]
        assert _prompts(info) == {"What bone is this?", "Is it broken?"}


def test_content_hint_scan_does_not_decode_the_images(monkeypatch):
    from datasets.features.image import Image as ImageFeature
    from utils.datasets import generate_smart_vlm_instruction

    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))] * 50,
            "caption": ["Radiograph shows a lytic lesion."] * 50,
        }
    )
    decoded = []
    real_decode = ImageFeature.decode_example

    def counting_decode(
        self,
        value,
        token_per_repo_id = None,
    ):
        decoded.append(1)
        return real_decode(self, value, token_per_repo_id = token_per_repo_id)

    monkeypatch.setattr(ImageFeature, "decode_example", counting_decode)
    info = generate_smart_vlm_instruction(
        ds, text_column = "caption", image_column = "image", dataset_name = "org/scans"
    )

    assert info["instruction"] == MEDICAL
    # Row 0 is read whole for its column names; the 100-row hint scan reads captions only.
    assert len(decoded) <= 1, len(decoded)


def test_content_hint_scan_keeps_rows_a_custom_transform_reads():
    # A with_transform callback that reads the image must still get it: the text-column projection would drop it.
    from utils.datasets import generate_smart_vlm_instruction

    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))] * 20,
            "caption": ["Radiograph shows a lytic lesion."] * 20,
        }
    )

    def transform(batch):
        assert "image" in batch, "the transform lost its image input"
        return {"image": batch["image"], "caption": batch["caption"]}

    ds = ds.with_transform(transform)
    info = generate_smart_vlm_instruction(
        ds, text_column = "caption", image_column = "image", dataset_name = "org/scans"
    )
    assert info["instruction"] == MEDICAL
