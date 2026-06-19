# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.training import TrainingStartRequest
from utils.datasets.chat_templates import apply_chat_template_to_dataset
from utils.datasets.format_conversion import (
    convert_alpaca_to_chatml,
    convert_chatml_to_alpaca,
    standardize_chat_format,
)
from utils.datasets.iterable import is_streaming_dataset

datasets = pytest.importorskip("datasets")

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token = "</s>"
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize = False,
        add_generation_prompt = False,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in conversation
        )


def _iterable_dataset(rows):
    return datasets.IterableDataset.from_generator(lambda: iter(rows))


def test_chat_template_mapping_omits_eager_kwargs_for_streaming(monkeypatch):
    seen_kwargs = []
    original_map = datasets.IterableDataset.map

    def spy_map(self, *args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return original_map(self, *args, **kwargs)

    monkeypatch.setattr(datasets.IterableDataset, "map", spy_map)

    dataset = _iterable_dataset(
        [
            {
                "conversations": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
    )
    result = apply_chat_template_to_dataset(
        {
            "dataset": dataset,
            "final_format": "chatml_conversations",
            "chat_column": "conversations",
            "is_standardized": True,
        },
        tokenizer = _Tokenizer(),
        batch_size = 1,
        num_proc = 2,
    )

    assert result["success"] is True
    row = next(iter(result["dataset"]))
    assert "user: Hi" in row["text"]
    assert seen_kwargs
    assert all("num_proc" not in kwargs for kwargs in seen_kwargs)
    assert all("desc" not in kwargs for kwargs in seen_kwargs)


def test_format_conversion_omits_eager_kwargs_for_streaming(monkeypatch):
    seen_kwargs = []
    original_map = datasets.IterableDataset.map

    def spy_map(self, *args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return original_map(self, *args, **kwargs)

    monkeypatch.setattr(datasets.IterableDataset, "map", spy_map)

    converted = convert_chatml_to_alpaca(
        _iterable_dataset(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Question"},
                        {"from": "gpt", "value": "Answer"},
                    ]
                }
            ]
        ),
        batch_size = 1,
        num_proc = 2,
    )

    row = next(iter(converted))
    assert row["instruction"] == "Question"
    assert row["output"] == "Answer"
    assert seen_kwargs
    assert all("num_proc" not in kwargs for kwargs in seen_kwargs)
    assert all("desc" not in kwargs for kwargs in seen_kwargs)


def test_streaming_start_rejects_train_on_completions_before_backend_start():
    training_route = _load_route_module(
        "training_route_module_for_streaming_completion_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        train_on_completions = True,
        max_steps = 10,
    )

    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: pytest.fail("backend should not start"),
    )

    with patch.object(training_route, "get_training_backend", return_value = backend):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                training_route.start_training(request, current_subject = "test-user")
            )

    assert exc_info.value.status_code == 422
    assert "train_on_completions" in exc_info.value.detail


@pytest.mark.parametrize("eval_split", [None, "train"])
def test_streaming_start_requires_separate_eval_split(eval_split):
    training_route = _load_route_module(
        "training_route_module_for_streaming_eval_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        train_split = "train",
        eval_split = eval_split,
        eval_steps = 0.1,
        max_steps = 10,
    )

    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: pytest.fail("backend should not start"),
    )

    with patch.object(training_route, "get_training_backend", return_value = backend):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                training_route.start_training(request, current_subject = "test-user")
            )

    assert exc_info.value.status_code == 422
    assert "separate eval_split" in exc_info.value.detail


def test_dataset_slice_bounds_are_non_negative():
    with pytest.raises(ValidationError):
        TrainingStartRequest(
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            dataset_slice_start = -1,
        )

    with pytest.raises(ValidationError):
        TrainingStartRequest(
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            dataset_slice_start = 5,
            dataset_slice_end = 4,
        )


def test_dataset_slice_accepts_equal_and_ordered_bounds():
    # start == end is intentionally allowed (single-row slice); start < end too.
    equal = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        dataset_slice_start = 5,
        dataset_slice_end = 5,
    )
    assert equal.dataset_slice_start == 5
    assert equal.dataset_slice_end == 5

    ordered = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        dataset_slice_start = 2,
        dataset_slice_end = 9,
    )
    assert ordered.dataset_slice_end == 9


def test_is_streaming_dataset_detects_hf_iterable():
    assert is_streaming_dataset(_iterable_dataset([{"a": 1}])) is True


def test_is_streaming_dataset_false_for_plain_list():
    assert is_streaming_dataset([{"a": 1}]) is False


def test_is_streaming_dataset_torch_branch_when_datasets_unavailable():
    torch = pytest.importorskip("torch")

    class _TorchIterable(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([1, 2, 3])

    # Force the `from datasets import IterableDataset` import to fail so the
    # torch detection branch is exercised.
    with patch.dict(sys.modules, {"datasets": None}):
        assert is_streaming_dataset(_TorchIterable()) is True


def test_is_streaming_dataset_false_when_both_backends_unavailable():
    # Both `datasets` and `torch.utils.data` imports fail -> graceful False.
    with patch.dict(sys.modules, {"datasets": None, "torch.utils.data": None}):
        assert is_streaming_dataset(object()) is False


def test_standardize_chat_format_omits_eager_kwargs_for_streaming(monkeypatch):
    seen_kwargs = []
    original_map = datasets.IterableDataset.map

    def spy_map(self, *args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return original_map(self, *args, **kwargs)

    monkeypatch.setattr(datasets.IterableDataset, "map", spy_map)

    dataset = _iterable_dataset(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello"},
                ]
            }
        ]
    )
    result = standardize_chat_format(
        dataset,
        tokenizer = _Tokenizer(),
        batch_size = 1,
        num_proc = 2,
    )

    next(iter(result))
    assert seen_kwargs
    assert all("num_proc" not in kwargs for kwargs in seen_kwargs)
    assert all("desc" not in kwargs for kwargs in seen_kwargs)


def test_convert_alpaca_to_chatml_omits_eager_kwargs_for_streaming(monkeypatch):
    seen_kwargs = []
    original_map = datasets.IterableDataset.map

    def spy_map(self, *args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return original_map(self, *args, **kwargs)

    monkeypatch.setattr(datasets.IterableDataset, "map", spy_map)

    converted = convert_alpaca_to_chatml(
        _iterable_dataset(
            [{"instruction": "Question", "input": "", "output": "Answer"}]
        ),
        batch_size = 1,
        num_proc = 2,
    )

    row = next(iter(converted))
    assert "conversations" in row
    assert seen_kwargs
    assert all("num_proc" not in kwargs for kwargs in seen_kwargs)
    assert all("desc" not in kwargs for kwargs in seen_kwargs)


def test_streaming_start_happy_path_reaches_backend():
    training_route = _load_route_module(
        "training_route_module_for_streaming_happy_path_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        train_split = "train",
        eval_split = "validation",
        eval_steps = 0.1,
        max_steps = 10,
    )

    captured = {}

    def _start_training(**kwargs):
        captured.update(kwargs)
        return True

    backend = SimpleNamespace(
        current_job_id = "job_test",
        is_training_active = lambda: False,
        start_training = _start_training,
    )

    with patch.object(training_route, "get_training_backend", return_value = backend):
        with patch.object(training_route, "load_model_defaults", return_value = {}):
            response = asyncio.run(
                training_route.start_training(request, current_subject = "test-user")
            )

    assert response.status == "queued"
    assert captured["dataset_streaming"] is True
    assert captured["max_steps"] == 10
    assert captured["eval_split"] == "validation"
