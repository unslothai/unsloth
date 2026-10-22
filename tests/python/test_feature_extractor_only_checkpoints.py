"""Encoder-only audio checkpoints (AST, WavLM, wav2vec2 bases) ship only a feature extractor: FastModel
returns it instead of failing on the missing tokenizer, and keeps it right padded."""

import ast
from pathlib import Path

import pytest

VISION_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "vision.py"


def _is_bare_feature_extractor():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_bare_feature_extractor"
    )
    ns = {}
    exec(compile(ast.Module([node], []), str(VISION_PATH), "exec"), ns)
    return ns["_is_bare_feature_extractor"]


def test_bare_feature_extractor_detection():
    transformers = pytest.importorskip("transformers")
    check = _is_bare_feature_extractor()
    fe = transformers.Wav2Vec2FeatureExtractor()
    assert check(fe)
    tok_cls = getattr(transformers, "PreTrainedTokenizerBase")
    assert not check(object.__new__(tok_cls))
    assert not check(None)


def test_last_resort_tries_feature_extractor_and_keeps_padding():
    source = VISION_PATH.read_text(encoding = "utf-8")
    assert "AutoFeatureExtractor.from_pretrained(" in source
    assert (
        "if not _is_bare_feature_extractor(tokenizer):\n            tokenizer.padding_side = _padding_side"
        in source
    )


def _config_maps_audio_auto_class():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    wanted = {"_AUDIO_AUTO_CLASS_NAMES", "_config_maps_audio_auto_class"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    ns = {}
    exec(compile(ast.Module(nodes, []), str(VISION_PATH), "exec"), ns)
    return ns["_config_maps_audio_auto_class"]


def test_audio_configs_get_their_processor():
    transformers = pytest.importorskip("transformers")
    maps = _config_maps_audio_auto_class()
    for cfg in ("Wav2Vec2Config", "HubertConfig", "WhisperConfig", "MoonshineConfig", "ASTConfig"):
        if hasattr(transformers, cfg):
            assert maps(getattr(transformers, cfg)()), cfg
    for cfg in ("LlamaConfig", "Qwen2Config", "T5Config"):
        assert not maps(getattr(transformers, cfg)()), cfg
