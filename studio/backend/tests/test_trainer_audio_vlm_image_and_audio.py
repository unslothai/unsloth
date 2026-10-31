# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An audio VLM (Gemma 3n, Gemma 4, LFM3-Omni) trained on a dataset with both audio and images.

pre_detect routes such a run to the audio VLM path (is_vlm is False), so that path has to carry the
images itself or they are silently dropped.
"""

from __future__ import annotations

import logging
import sys
import textwrap
import types
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

datasets = pytest.importorskip("datasets")
PIL_Image = pytest.importorskip("PIL.Image")

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


class _NoAudioCast:
    """A real Dataset whose Audio cast is a no-op: decoding needs torchcodec, absent on CPU runners."""

    def __init__(self, ds):
        self._ds = ds

    def cast_column(self, column, feature):
        if isinstance(feature, datasets.Audio):
            return self
        return _NoAudioCast(self._ds.cast_column(column, feature))

    def map(self, fn, **kwargs):
        kwargs.pop("num_proc", None)
        return _NoAudioCast(self._ds.map(fn, **kwargs))

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __iter__(self):
        return iter(self._ds)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


def _dataset(columns, per_row = None):
    image = PIL_Image.new("RGB", (8, 8), "red")
    rows = {
        "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
        "text": ["a red square", "still red"],
    }
    for name in columns:
        rows[name] = [[image] * n for n in per_row] if per_row else [image, image]
    return _NoAudioCast(datasets.Dataset.from_dict(rows))


@pytest.fixture
def audio_vlm_trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/gemma-3n-E2B-it"
    t.is_audio_vlm = True
    return t


def _user_part_types(dataset):
    user = next(m for m in dataset[0]["messages"] if m["role"] == "user")
    return [part["type"] for part in user["content"]]


def test_mixed_dataset_keeps_the_image_in_the_user_turn(audio_vlm_trainer):
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(_dataset(["image"]))

    assert _user_part_types(formatted) == ["image", "audio", "text"]
    assert audio_vlm_trainer._audio_vlm_image_col == "image"


def test_audio_only_dataset_is_unchanged(audio_vlm_trainer):
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(_dataset([]))

    assert getattr(audio_vlm_trainer, "_audio_vlm_image_col", None) is None
    assert _user_part_types(formatted) == ["audio", "text"]


def test_a_mapped_image_column_is_used(audio_vlm_trainer):
    mapping = {"audio": "audio", "text": "text", "photo": "image"}
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(_dataset(["photo"]), mapping)

    assert _user_part_types(formatted)[0] == "image"
    assert audio_vlm_trainer._audio_vlm_image_col == "photo"


def test_a_list_of_images_per_row_gets_one_slot_per_image(audio_vlm_trainer):
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(_dataset(["images"], per_row = [1, 2]))

    assert audio_vlm_trainer._audio_vlm_image_col == "images"
    users = [next(m for m in row["messages"] if m["role"] == "user") for row in formatted]
    assert [[p["type"] for p in u["content"]].count("image") for u in users] == [1, 2]


def test_a_file_name_column_of_audio_paths_is_not_taken_for_images(audio_vlm_trainer):
    rows = {
        "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
        "text": ["one", "two"],
        "file_name": ["clips/a.wav", "clips/b.wav"],
    }
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(
        _NoAudioCast(datasets.Dataset.from_dict(rows))
    )

    assert getattr(audio_vlm_trainer, "_audio_vlm_image_col", None) is None
    assert _user_part_types(formatted) == ["audio", "text"]


@pytest.mark.parametrize("feature", ["image", "images"])
def test_an_image_typed_column_counts_even_when_row_zero_is_empty(audio_vlm_trainer, feature):
    image = PIL_Image.new("RGB", (8, 8), "red")
    many = feature == "images"
    rows = {
        "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
        "text": ["one", "two"],
        feature: [[] if many else None, [image] if many else image],
    }
    schema = datasets.Features(
        {
            "audio": {
                "array": datasets.Sequence(datasets.Value("float32")),
                "sampling_rate": datasets.Value("int64"),
            },
            "text": datasets.Value("string"),
            feature: datasets.Sequence(datasets.Image()) if many else datasets.Image(),
        }
    )
    ds = _NoAudioCast(datasets.Dataset.from_dict(rows, features = schema))

    assert audio_vlm_trainer._resolve_audio_vlm_image_column(ds, None, "audio") == feature


def _collator_block():
    lines = (_BACKEND / "core" / "training" / "trainer.py").read_text(encoding = "utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == 'audio_col_name = getattr(self, "_audio_vlm_audio_col", "audio")'
    )
    end = next(
        i
        for i in range(start, len(lines))
        if lines[i].strip() == "data_collator = audio_vlm_collate_fn"
    )
    return textwrap.dedent("\n".join(lines[start:end]))


class _RecordingProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(pad_token_id = 0)
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "x"

    def __call__(self, text, audio, return_tensors, padding, **kwargs):
        import torch
        self.calls.append(kwargs)
        return {"input_ids": torch.ones((len(text), 2), dtype = torch.long)}


def _collate(
    image_col,
    row_image = None,
    rows = None,
):
    pytest.importorskip("torch")
    processor = _RecordingProcessor()
    namespace = {
        "processor": processor,
        "training_args": {},
        "self": types.SimpleNamespace(_audio_vlm_audio_col = "audio", _audio_vlm_image_col = image_col),
        "logger": logging.getLogger(__name__),
        "types": types,
        "apply_completion_masking": None,
        "audio_response_masker": None,
    }
    exec(_collator_block(), namespace)
    image = PIL_Image.new("L", (4, 4))
    row_image = image if row_image is None else row_image
    examples = [
        {"messages": [], "audio": {"array": [0.0]}, "image": r}
        for r in (rows if rows is not None else [row_image, row_image])
    ]
    namespace["audio_vlm_collate_fn"](examples)
    return processor.calls[0]


def test_collator_passes_one_rgb_image_per_example():
    kwargs = _collate("image")

    images = kwargs.get("images")
    assert images is not None, "images never reached the processor"
    assert [len(per_example) for per_example in images] == [1, 1]
    assert all(per_example[0].mode == "RGB" for per_example in images)


def test_collator_passes_every_image_of_a_list_row():
    image = PIL_Image.new("L", (4, 4))
    images = _collate("image", row_image = [image, image])["images"]

    assert [len(per_example) for per_example in images] == [2, 2]
    assert all(i.mode == "RGB" for per_example in images for i in per_example)


def test_collator_without_images_does_not_pass_the_kwarg():
    assert "images" not in _collate(None)


def test_a_none_image_row_gets_no_slot_and_no_image(audio_vlm_trainer):
    image = PIL_Image.new("RGB", (8, 8), "red")
    rows = {
        "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
        "text": ["one", "two"],
        "image": [None, image],
    }
    schema = datasets.Features(
        {
            "audio": {
                "array": datasets.Sequence(datasets.Value("float32")),
                "sampling_rate": datasets.Value("int64"),
            },
            "text": datasets.Value("string"),
            "image": datasets.Image(),
        }
    )
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(
        _NoAudioCast(datasets.Dataset.from_dict(rows, features = schema))
    )
    users = [next(m for m in row["messages"] if m["role"] == "user") for row in formatted]
    assert [[p["type"] for p in u["content"]].count("image") for u in users] == [0, 1]

    image = PIL_Image.new("L", (4, 4))
    assert [len(x) for x in _collate("image", rows = [None, image])["images"]] == [0, 1]
    assert "images" not in _collate("image", rows = [None, None])


@pytest.mark.parametrize("many", [False, True])
def test_a_non_decoding_image_feature_is_decoded(audio_vlm_trainer, many):
    image = PIL_Image.new("RGB", (8, 8), "red")
    inner = datasets.Image(decode = False)
    schema = datasets.Features(
        {
            "audio": {
                "array": datasets.Sequence(datasets.Value("float32")),
                "sampling_rate": datasets.Value("int64"),
            },
            "text": datasets.Value("string"),
            "image": datasets.Sequence(inner) if many else inner,
        }
    )
    rows = {
        "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
        "text": ["one", "two"],
        "image": [[image], [image, image]] if many else [image, image],
    }
    formatted = audio_vlm_trainer._format_audio_vlm_dataset(
        _NoAudioCast(datasets.Dataset.from_dict(rows, features = schema))
    )

    cell = formatted[1]["image"]
    cells = cell if many else [cell]
    assert len(cells) == (2 if many else 1)
    assert all(isinstance(x, PIL_Image.Image) for x in cells)


def test_a_messages_column_with_image_parts_is_not_taken_for_images(audio_vlm_trainer):
    image = PIL_Image.new("RGB", (8, 8), "red")
    turn = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    ds = _NoAudioCast(
        datasets.Dataset.from_dict(
            {
                "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
                "text": ["one", "two"],
                "messages": [turn, turn],
            }
        )
    )

    assert audio_vlm_trainer._resolve_audio_vlm_image_column(ds, None, "audio") is None


def test_an_untyped_path_column_counts_when_row_zero_is_empty(audio_vlm_trainer):
    ds = _NoAudioCast(
        datasets.Dataset.from_dict(
            {
                "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 3,
                "text": ["one", "two", "three"],
                "image": [None, None, "pics/a.png"],
            }
        )
    )

    assert audio_vlm_trainer._resolve_audio_vlm_image_column(ds, None, "audio") == "image"
