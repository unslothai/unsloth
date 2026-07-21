# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.training import TrainingStartRequest
from utils.datasets.chat_templates import apply_chat_template_to_dataset
from utils.datasets.format_conversion import convert_chatml_to_alpaca
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


# --- Streaming keeps dataset.map() lazy: eager-only kwargs (num_proc/desc) are
#     omitted for IterableDatasets, which reject them. One per module. ---


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


# --- Streaming detection ---


def test_is_streaming_dataset_detects_hf_iterable():
    assert is_streaming_dataset(_iterable_dataset([{"a": 1}])) is True


def test_is_streaming_dataset_false_for_plain_list():
    assert is_streaming_dataset([{"a": 1}]) is False


# --- Raw-text / CPT streaming: keep the lazy filter, skip the len()-based
#     counting that would TypeError on an IterableDataset (the BLOCKER fix). ---


def test_drop_invalid_text_rows_streaming_keeps_filter_skips_len():
    from utils.datasets.raw_text import _drop_invalid_text_rows

    stream = datasets.Dataset.from_list(
        [{"text": "keep1"}, {"text": None}, {"text": "keep2"}]
    ).to_iterable_dataset()
    assert not hasattr(stream, "__len__")

    filtered, notices = _drop_invalid_text_rows(
        stream, mode_title = "Raw text", split_scope = "this dataset"
    )

    # Result still streams; only string-'text' rows survive.
    assert [row["text"] for row in filtered] == ["keep1", "keep2"]
    assert any(n.level == "info" for n in notices)


# --- Request validation ---


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


@pytest.mark.parametrize(
    "bad_hf_dataset",
    ["../../etc/passwd", "org/../../secret", "a" * 257],
)
def test_hf_dataset_rejects_unsafe_values(bad_hf_dataset):
    with pytest.raises(ValidationError):
        TrainingStartRequest(
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            hf_dataset = bad_hf_dataset,
        )


def test_project_name_rejects_values_over_ui_limit():
    with pytest.raises(ValidationError):
        TrainingStartRequest(
            model_name = "unsloth/test",
            project_name = "x" * 81,
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
        )


# --- Start-route streaming compatibility guards ---


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


def test_streaming_start_rejects_missing_max_steps():
    training_route = _load_route_module(
        "training_route_module_for_streaming_max_steps_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        max_steps = 0,
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
    assert "max_steps" in exc_info.value.detail


def test_streaming_start_rejects_embedding_models():
    # The embedding training path loads the full dataset (no streaming) and uses
    # len/select, so the route must reject streaming for embedding runs even on a
    # direct API call (the UI blocker doesn't cover that).
    training_route = _load_route_module(
        "training_route_module_for_streaming_embedding_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        is_embedding = True,
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

    assert exc_info.value.status_code == 400
    assert "embedding" in exc_info.value.detail


@pytest.mark.parametrize(
    "training_type, format_type",
    [
        ("LoRA/QLoRA", "raw"),  # raw-text format
        ("Continued Pretraining", "chatml"),  # CPT
    ],
)
def test_streaming_start_accepts_raw_text_and_cpt(training_type, format_type):
    # Streaming + raw-text / CPT is supported: _drop_invalid_text_rows skips its
    # len()-based checks for IterableDatasets, so the start route must NOT reject.
    training_route = _load_route_module(
        "training_route_module_for_streaming_raw_cpt_accept_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = training_type,
        hf_dataset = "org/dataset",
        format_type = format_type,
        dataset_streaming = True,
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
    assert captured["format_type"] == format_type


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


# streaming rejects HF slice syntax in train_split / eval_split


@pytest.mark.parametrize(
    "field, value",
    [
        ("train_split", "train[:50%]"),
        ("train_split", "train[:20]"),
        ("eval_split", "validation[:1000]"),
    ],
)
def test_streaming_rejects_bracketed_split_syntax(field, value):
    # The model_validator _validate_streaming_splits raises ValidationError when
    # dataset_streaming=True and a split contains "[" (HF slice syntax).
    kwargs = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "hf_dataset": "org/dataset",
        "format_type": "chatml",
        "dataset_streaming": True,
        "max_steps": 10,
        field: value,
    }
    with pytest.raises(ValidationError) as exc_info:
        TrainingStartRequest(**kwargs)
    detail = str(exc_info.value)
    assert "slice" in detail.lower() or "bracket" in detail.lower() or "[" in detail


# streaming rejects mixed sources (local_datasets)


def test_streaming_start_rejects_local_datasets():
    # dataset_streaming + local_datasets -> 400, 'local' in detail
    training_route = _load_route_module(
        "training_route_module_for_streaming_local_datasets_test",
        "routes/training.py",
    )
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        max_steps = 10,
    )
    # Bypass Pydantic's local-path validation by injecting directly after construction.
    object.__setattr__(request, "local_datasets", ["/some/local/file.jsonl"])

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

    assert exc_info.value.status_code == 400
    assert (
        "local" in exc_info.value.detail.lower()
        or "hf-only" in exc_info.value.detail.lower()
    )


# _drop_invalid_text_rows handles from_generator with column_names=None


def test_drop_invalid_text_rows_from_generator_none_column_names():
    # from_generator IterableDatasets have column_names=None; resolve_column_names
    # must fall back to first-row probe. _drop_invalid_text_rows must not raise
    # TypeError and must filter correctly.
    from utils.datasets.raw_text import _drop_invalid_text_rows

    def _gen():
        yield {"text": "valid row"}
        yield {"text": None}  # invalid, should be dropped
        yield {"text": "another row"}

    stream = datasets.IterableDataset.from_generator(_gen)
    # Precondition: column_names is None on a raw from_generator dataset.
    assert (
        stream.column_names is None
    ), "precondition failed: expected column_names=None for from_generator dataset"

    filtered, notices = _drop_invalid_text_rows(
        stream, mode_title = "Raw text", split_scope = "test split"
    )

    rows = list(filtered)
    assert [r["text"] for r in rows] == ["valid row", "another row"]
    # At least one info/warning notice about dropped rows.
    assert len(notices) >= 1


# _preflight_first_batch returns error string on empty dataloader


def test_preflight_first_batch_returns_error_on_empty_stream():
    # StopIteration from an empty dataloader must return a clear
    # error string (not None). Test via a minimal stub, no real model needed.
    import types
    import sys

    # Minimal stub trainer whose get_train_dataloader() yields nothing.
    class _EmptyLoader:
        def __iter__(self):
            return iter([])

    class _StubTrainer:
        def get_train_dataloader(self):
            return _EmptyLoader()

    # Load UnslothTrainer class from trainer.py via importlib to avoid heavy imports.
    trainer_path = _BACKEND_ROOT / "core" / "training" / "trainer.py"
    spec = importlib.util.spec_from_file_location("trainer_module", trainer_path)
    trainer_mod = importlib.util.module_from_spec(spec)
    # Provide a minimal sys.modules shim so top-level imports in trainer.py don't
    # crash when optional heavy deps (torch, unsloth) are absent.
    _orig_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    try:
        spec.loader.exec_module(trainer_mod)
    except Exception:
        # trainer.py has optional heavy imports; access _preflight_first_batch directly.
        pass

    # If we successfully loaded the module, find the trainer class.
    trainer_cls = None
    for name, obj in vars(trainer_mod).items() if "trainer_mod" in dir() else []:
        # Only real classes — when heavy deps are stubbed with MagicMock,
        # hasattr() is always True on a mock, so guard on isinstance(obj, type)
        # to avoid picking a mock instance (object.__new__ would then reject it).
        if isinstance(obj, type) and hasattr(obj, "_preflight_first_batch"):
            trainer_cls = obj
            break

    if trainer_cls is None:
        pytest.skip(
            "Could not load trainer module (missing optional deps: torch/unsloth)."
        )

    # Build a bare instance without calling __init__ (avoids needing real deps).
    instance = object.__new__(trainer_cls)
    instance.trainer = _StubTrainer()
    instance.model_name = "stub-model"

    result = instance._preflight_first_batch()

    assert result is not None, (
        "_preflight_first_batch must return an error string (not None) when the "
        "training dataloader is empty."
    )
    assert isinstance(result, str)
    # The message should indicate there are no training rows / empty dataset.
    assert any(
        kw in result.lower() for kw in ("empty", "no training", "no rows", "stream")
    )
