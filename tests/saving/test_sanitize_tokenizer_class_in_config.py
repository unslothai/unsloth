import json
from pathlib import Path

import pytest

from unsloth_zoo.saving_utils import sanitize_tokenizer_class_in_config


def test_sanitize_tokenizer_class_rewrites_tokenizers_backend(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding = "utf-8")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "model_max_length": 8192}),
        encoding = "utf-8",
    )

    sanitize_tokenizer_class_in_config(None, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert saved_config["model_max_length"] == 8192


def test_sanitize_tokenizer_class_leaves_loadable_classes_alone(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding = "utf-8")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast"}),
        encoding = "utf-8",
    )

    sanitize_tokenizer_class_in_config(None, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "LlamaTokenizerFast"


def test_sanitize_tokenizer_class_uses_source_tokenizer_class_when_available(tmp_path):
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )

    class Qwen2TokenizerFast:
        pass

    sanitize_tokenizer_class_in_config(Qwen2TokenizerFast(), tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "Qwen2TokenizerFast"


def test_sanitize_tokenizer_class_supports_filename_prefix(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding = "utf-8")
    prefixed_config = tmp_path / "adapter-tokenizer_config.json"
    prefixed_config.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )

    sanitize_tokenizer_class_in_config(None, tmp_path, filename_prefix = "adapter")

    saved_config = json.loads(prefixed_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert not (tmp_path / "tokenizer_config.json").exists()


def test_sanitize_tokenizer_class_skips_without_tokenizer_json_or_source_class(tmp_path):
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )

    sanitize_tokenizer_class_in_config(None, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "TokenizersBackend"


@pytest.mark.parametrize(
    "tokenizer_class",
    ("PreTrainedTokenizerFast", "GemmaTokenizer", "Qwen2Tokenizer"),
)
def test_sanitize_tokenizer_class_is_noop_for_exportable_values(tmp_path, tokenizer_class):
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"tokenizer_class": tokenizer_class}),
        encoding = "utf-8",
    )

    sanitize_tokenizer_class_in_config(None, tmp_path)

    saved_config = json.loads(tokenizer_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == tokenizer_class
