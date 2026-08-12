import json
import types

from unsloth.save import (
    _preserve_tokenizer_class,
    _resolve_export_tokenizer_class,
)


def test_preserve_tokenizer_class_restores_from_local_base_model(tmp_path):
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        encoding = "utf-8",
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )
    (export_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")

    tokenizer = types.SimpleNamespace(name_or_path = str(base_model))

    _preserve_tokenizer_class(tokenizer, export_dir)

    saved_config = json.loads((export_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "Qwen2Tokenizer"


def test_preserve_tokenizer_class_falls_back_to_pretrained_tokenizer_fast(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )
    (export_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")

    tokenizer = types.SimpleNamespace(name_or_path = "missing/base-model")

    _preserve_tokenizer_class(tokenizer, export_dir)

    saved_config = json.loads((export_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_preserve_tokenizer_class_supports_processor_tokenizer(tmp_path):
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizer"}),
        encoding = "utf-8",
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )
    (export_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")

    processor = types.SimpleNamespace(
        tokenizer = types.SimpleNamespace(name_or_path = str(base_model)),
    )

    _preserve_tokenizer_class(processor, export_dir)

    saved_config = json.loads((export_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "LlamaTokenizer"


def test_preserve_tokenizer_class_supports_filename_prefix(tmp_path):
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GemmaTokenizer"}),
        encoding = "utf-8",
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    prefixed_config = export_dir / "adapter-tokenizer_config.json"
    prefixed_config.write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding = "utf-8",
    )
    (export_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")

    tokenizer = types.SimpleNamespace(name_or_path = str(base_model))

    _preserve_tokenizer_class(tokenizer, export_dir, filename_prefix = "adapter")

    saved_config = json.loads(prefixed_config.read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "GemmaTokenizer"
    assert not (export_dir / "tokenizer_config.json").exists()


def test_preserve_tokenizer_class_leaves_loadable_class_unchanged(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast", "other": True}),
        encoding = "utf-8",
    )

    tokenizer = types.SimpleNamespace(name_or_path = "unused")

    _preserve_tokenizer_class(tokenizer, export_dir)

    saved_config = json.loads((export_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
    assert saved_config["tokenizer_class"] == "LlamaTokenizerFast"
    assert saved_config["other"] is True


def test_resolve_export_tokenizer_class_prefers_live_subclass(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "tokenizer.json").write_text("{}", encoding = "utf-8")

    class Qwen3_5Tokenizer:
        pass

    tokenizer = Qwen3_5Tokenizer()
    tokenizer.name_or_path = "unused/base-model"

    assert _resolve_export_tokenizer_class(tokenizer, export_dir) == "Qwen3_5Tokenizer"
