# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

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
    # AutoProcessor's fallback keeps the padding_side = "left" it was given; waveforms stay right padded.
    assert (
        'tokenizer.padding_side = "right" if _is_bare_feature_extractor(tokenizer) else _padding_side'
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


def test_tokenizer_fallback_keeps_audio_processors():
    source = VISION_PATH.read_text(encoding = "utf-8")
    # Granite Speech names its audio frontend audio_processor.
    assert '("image_processor", "feature_extractor", "audio_processor")' in source


LOADER_PATH = VISION_PATH.with_name("loader.py")


def _resolve_audio_auto_model():
    transformers = pytest.importorskip("transformers")
    tree = ast.parse(LOADER_PATH.read_text(encoding = "utf-8"))
    wanted = {"_AUDIO_HEAD_AUTO_CLASS_NAMES", "_resolve_audio_auto_model"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]

    def resolve_model_class(auto_model, config, **kwargs):
        try:
            return auto_model._model_mapping[type(config)]
        except KeyError:
            return None

    ns = {"resolve_model_class": resolve_model_class}
    exec(compile(ast.Module(nodes, []), str(LOADER_PATH), "exec"), ns)
    return ns["_resolve_audio_auto_model"], transformers


def test_default_auto_model_loads_the_checkpoints_own_audio_head():
    resolve, transformers = _resolve_audio_auto_model()
    cases = [
        (transformers.Wav2Vec2Config(architectures = ["Wav2Vec2ForCTC"]), "AutoModelForCTC"),
        (
            transformers.ASTConfig(architectures = ["ASTForAudioClassification"]),
            "AutoModelForAudioClassification",
        ),
        (transformers.WavLMConfig(architectures = ["WavLMModel"]), "AutoModel"),
        (transformers.WavLMConfig(architectures = ["WavLMForXVector"]), "AutoModelForAudioXVector"),
        (
            transformers.WhisperConfig(architectures = ["WhisperForAudioClassification"]),
            "AutoModelForAudioClassification",
        ),
    ]
    for config, expected in cases:
        assert resolve(config).__name__ == expected, config.architectures
    assert resolve(transformers.LlamaConfig(architectures = ["LlamaForCausalLM"])) is None
    # with a causal-LM mapping, a bare checkpoint keeps today's route (Qwen2Model embedding repos)
    assert (
        resolve(transformers.LlamaConfig(architectures = ["LlamaModel"]), include_bare = False) is None
    )
    assert resolve(transformers.Wav2Vec2Config()) is None  # no architectures: keep today's route
    source = LOADER_PATH.read_text(encoding = "utf-8")
    flat = "".join(source.split())
    assert "_resolve_audio_auto_model(model_config,**_probe_hub_kwargs)" in flat
    assert "auto_model=_resolve_non_vlm_auto_model(model_config,**_probe_hub_kwargs)" in flat
    assert "include_bare=notcausal_maps" in flat


def test_audio_processor_frontends_are_right_padded_too():
    source = VISION_PATH.read_text(encoding = "utf-8")
    assert 'for _audio_attr in ("feature_extractor", "audio_processor"):' in source
