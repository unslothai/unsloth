# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402

_VISION_LORA_YAML = _REPO_ROOT / "studio" / "backend" / "assets" / "configs" / "vision_lora.yaml"
_VISION_CONFIG = {
    "model_type": "qwen2_vl",
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "vision_config": {},
}
_TEXT_CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
_TEXT_TOKENIZER = {"added_tokens_decoder": {"151643": {"content": "<|endoftext|>"}}}
_AUDIO_VLM_TOKENIZER = {"added_tokens_decoder": {"262273": {"content": "<audio_soft_token>"}}}
_IMAGE_ROWS = [{"image": "photos/cat.png", "text": "a cat"}]
_TEXT_ROWS = [{"instruction": "Say hi", "output": "hi"}]
_AUDIO_ROWS = [{"audio": "clips/hello.wav", "text": "hello"}]
_FILE_NAME_ROWS = [{"instruction": "Summarise", "output": "A readme", "file_name": "README.md"}]
_PROSE_IMAGE_ROWS = [{"instruction": "Describe", "output": "A cat", "image": "a cat"}]
_PROSE_SPEECH_ROWS = [
    {"instruction": "Rewrite", "output": "Hello!", "speech": "Welcome to our meeting."}
]


class _RecordingTrainer:
    training_thread = None

    def __init__(self):
        self.is_vlm = False
        self.load_kwargs = None
        self.prepare_kwargs = None

    def load_model(self, **kwargs):
        self.load_kwargs = kwargs
        self.is_vlm = kwargs.get("is_dataset_image", False)
        return True

    def prepare_model_for_training(self, **kwargs):
        self.prepare_kwargs = kwargs
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        return True

    def get_training_progress(self):
        return SimpleNamespace(error = None, is_completed = True)


def _train(tmp_path, monkeypatch, *args):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_UPDATE_DOWNLOAD_COUNTS", False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    trainer = _RecordingTrainer()
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *a: trainer)
    app = typer.Typer()
    app.command()(train_module.train)

    result = CliRunner().invoke(app, [*args, "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    return trainer


def _local_model(
    tmp_path,
    config,
    tokenizer_config = None,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (model / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding = "utf-8")
    return str(model)


def _local_parquet(tmp_path, rows):
    from datasets import Dataset

    path = tmp_path / "data.parquet"
    Dataset.from_list(rows).to_parquet(str(path))
    return str(path)


def _hub(monkeypatch, vision_repos, rows):
    train_module.ensure_studio_backend_path()
    import utils.datasets.cache_safe as cache_safe
    import utils.models as models
    from datasets import Dataset

    loaded = []

    def load_dataset(path, **kwargs):
        loaded.append((path, kwargs.get("split")))
        return Dataset.from_list(rows)

    monkeypatch.setattr(cache_safe, "load_dataset_cache_safe", load_dataset)
    monkeypatch.setattr(
        models, "is_vision_model_checked", lambda name, **kwargs: (name in vision_repos, True)
    )
    monkeypatch.setattr(models, "detect_audio_type_checked", lambda *args, **kwargs: (None, True))
    return loaded


@pytest.mark.parametrize(
    "config, tokenizer_config, rows, image, audio",
    [
        (_VISION_CONFIG, _TEXT_TOKENIZER, _IMAGE_ROWS, True, False),
        (_VISION_CONFIG, _TEXT_TOKENIZER, _TEXT_ROWS, False, False),
        (_TEXT_CONFIG, _TEXT_TOKENIZER, _IMAGE_ROWS, False, False),
        (_TEXT_CONFIG, _AUDIO_VLM_TOKENIZER, _AUDIO_ROWS, False, True),
        (_VISION_CONFIG, _TEXT_TOKENIZER, _FILE_NAME_ROWS, False, False),
        (_VISION_CONFIG, _TEXT_TOKENIZER, _PROSE_IMAGE_ROWS, False, False),
        (_TEXT_CONFIG, _AUDIO_VLM_TOKENIZER, _PROSE_SPEECH_ROWS, False, False),
    ],
)
def test_train_tells_the_trainer_whether_a_local_dataset_has_images_or_audio(
    tmp_path, monkeypatch, config, tokenizer_config, rows, image, audio
):
    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, config, tokenizer_config),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs.get("is_dataset_image", False) is image
    assert trainer.load_kwargs.get("is_dataset_audio", False) is audio


@pytest.mark.parametrize("header, image", [(b"\x89PNG\r\n\x1a\n", True), (b"# Read me\n", False)])
def test_an_extensionless_file_counts_as_an_image_only_when_it_is_one(
    tmp_path, monkeypatch, header, image
):
    referenced = tmp_path / "README"
    referenced.write_bytes(header)
    rows = [{"instruction": "Summarise", "output": "A readme", "file_name": str(referenced)}]

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _VISION_CONFIG, _TEXT_TOKENIZER),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs.get("is_dataset_image", False) is image


@pytest.mark.parametrize(
    "header, audio",
    [
        (b"\x00\x00\x00\x20ftypM4A ", True),
        (b"\x1a\x45\xdf\xa3\x01\x00", True),
        (b"\xff\xfb\x90\x00", True),
        (b"\xff\xf1\x50\x80", True),
        (b"0&\xb2u\x8ef\xcf\x11", True),
        (b"#!AMR\n\x3c", True),
        (b"FORM\x00\x00\x00\x00AIFF", True),
        (b"caff\x00\x01\x00\x00", True),
        (b"Hello there\n", False),
    ],
)
def test_an_extensionless_audio_file_is_recognised_by_its_container(
    tmp_path, monkeypatch, header, audio
):
    clip = tmp_path / "clip"
    clip.write_bytes(header)
    rows = [{"audio": str(clip), "text": "hello"}]

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _TEXT_CONFIG, _AUDIO_VLM_TOKENIZER),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs.get("is_dataset_audio", False) is audio


def test_an_image_set_with_a_leading_empty_row_still_trains_the_vision_model(tmp_path, monkeypatch):
    rows = [{"image": None, "text": "no picture"}, {"image": "photos/cat.png", "text": "a cat"}]

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _VISION_CONFIG, _TEXT_TOKENIZER),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs["is_dataset_image"] is True


@pytest.mark.parametrize(
    "clip", [b"RIFF\x24\x00\x00\x00WAVEfmt ", {"bytes": b"fLaC\x00\x00\x00\x22", "path": None}]
)
def test_byte_backed_audio_trains_the_audio_model(tmp_path, monkeypatch, clip):
    rows = [{"audio": clip, "text": "hello"}]

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _TEXT_CONFIG, _AUDIO_VLM_TOKENIZER),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs["is_dataset_audio"] is True


def test_an_extensionless_image_url_is_checked_by_what_it_serves(tmp_path, monkeypatch):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(b"\x89PNG\r\n\x1a\n" if self.path == "/cat" else b"plain text")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        results = []
        for name in ("cat", "readme"):
            directory = tmp_path / name
            directory.mkdir()
            trainer = _train(
                directory,
                monkeypatch,
                "--model",
                _local_model(directory, _VISION_CONFIG, _TEXT_TOKENIZER),
                "--local-dataset",
                _local_parquet(directory, [{"image": f"{base}/{name}", "text": "x"}]),
            )
            results.append(trainer.load_kwargs.get("is_dataset_image", False))
    finally:
        server.shutdown()

    assert results == [True, False]


def test_a_native_audio_model_is_told_its_dataset_has_audio(tmp_path, monkeypatch):
    train_module.ensure_studio_backend_path()
    import utils.models as models

    monkeypatch.setattr(models, "detect_audio_type_checked", lambda *a, **k: ("whisper", True))
    monkeypatch.setattr(models, "is_vision_model_checked", lambda *a, **k: (False, True))

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        "openai/whisper-tiny",
        "--local-dataset",
        _local_parquet(tmp_path, _AUDIO_ROWS),
    )

    assert trainer.load_kwargs["is_dataset_audio"] is True


@pytest.mark.parametrize("name", ["clip.amr", "clip.aiff", "clip.caf", "clip.bin"])
def test_audio_paths_with_other_suffixes_and_path_dicts_count_as_audio(tmp_path, monkeypatch, name):
    clip = tmp_path / name
    clip.write_bytes(b"#!AMR\n\x3c\x00")
    rows = [{"audio": {"bytes": None, "path": str(clip)}, "text": "hello"}]

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _TEXT_CONFIG, _AUDIO_VLM_TOKENIZER),
        "--local-dataset",
        _local_parquet(tmp_path, rows),
    )

    assert trainer.load_kwargs["is_dataset_audio"] is True


def test_every_local_dataset_path_is_checked_for_images(tmp_path, monkeypatch):
    from datasets import Dataset

    text = tmp_path / "text.parquet"
    Dataset.from_list(_TEXT_ROWS).to_parquet(str(text))
    images = tmp_path / "images.parquet"
    Dataset.from_list(_IMAGE_ROWS).to_parquet(str(images))

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _VISION_CONFIG, _TEXT_TOKENIZER),
        "--local-dataset",
        str(text),
        "--local-dataset",
        str(images),
    )

    assert trainer.load_kwargs["is_dataset_image"] is True


def test_a_transient_vision_probe_still_checks_the_dataset(tmp_path, monkeypatch):
    train_module.ensure_studio_backend_path()
    import utils.models as models

    monkeypatch.setattr(models, "detect_audio_type_checked", lambda *a, **k: (None, True))
    monkeypatch.setattr(models, "is_vision_model_checked", lambda *a, **k: (False, False))

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        "Qwen/Qwen2-VL-2B-Instruct",
        "--local-dataset",
        _local_parquet(tmp_path, _IMAGE_ROWS),
    )

    assert trainer.load_kwargs["is_dataset_image"] is True


@pytest.mark.parametrize("as_dict", [False, True])
def test_a_saved_to_disk_image_dataset_trains_the_vision_model(tmp_path, monkeypatch, as_dict):
    from datasets import Dataset, DatasetDict
    from PIL import Image

    saved = tmp_path / "saved"
    dataset = Dataset.from_list([{"image": Image.new("RGB", (8, 8)), "text": "a square"}])
    (DatasetDict(train = dataset) if as_dict else dataset).save_to_disk(str(saved))

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _VISION_CONFIG, _TEXT_TOKENIZER),
        "--local-dataset",
        str(saved),
    )

    assert trainer.load_kwargs["is_dataset_image"] is True


def test_shipped_vision_lora_config_trains_the_vision_lora(tmp_path, monkeypatch):
    loaded = _hub(monkeypatch, {"unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"}, _IMAGE_ROWS)

    trainer = _train(tmp_path, monkeypatch, "--config", str(_VISION_LORA_YAML))

    assert loaded == [("philschmid/amazon-product-descriptions-vlm", "train")]
    assert trainer.load_kwargs["is_dataset_image"] is True
    assert trainer.prepare_kwargs["target_modules"] == "all-linear"


def test_a_gated_upstream_vlm_is_checked_through_its_public_mirror(tmp_path, monkeypatch):
    _hub(monkeypatch, {"unsloth/gemma-3-4b-it-unsloth-bnb-4bit"}, _IMAGE_ROWS)

    trainer = _train(
        tmp_path, monkeypatch, "--model", "google/gemma-3-4b-it", "--dataset", "unsloth/LaTeX_OCR"
    )

    assert trainer.load_kwargs["is_dataset_image"] is True


def test_a_home_relative_local_dataset_is_checked_for_images(tmp_path, monkeypatch):
    from datasets import Dataset

    Dataset.from_list(_IMAGE_ROWS).to_parquet(str(tmp_path / "images.parquet"))
    monkeypatch.setenv("HOME", str(tmp_path))

    trainer = _train(
        tmp_path,
        monkeypatch,
        "--model",
        _local_model(tmp_path, _VISION_CONFIG, _TEXT_TOKENIZER),
        "--local-dataset",
        "~/images.parquet",
    )

    assert trainer.load_kwargs["is_dataset_image"] is True
