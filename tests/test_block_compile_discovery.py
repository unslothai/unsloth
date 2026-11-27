# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block compile decoder-layer discovery on meta-device default-config models (CPU)."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_PATH = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "_block_compile.py")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in (
        "UNSLOTH_BLOCK_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_BLOCK_COMPILE_SSM",
    ):
        monkeypatch.delenv(key, raising = False)


def _load():
    spec = importlib.util.spec_from_file_location("_ublock_discovery", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _meta_model(model_type, auto = "AutoModelForCausalLM"):
    try:
        config = transformers.AutoConfig.for_model(model_type)
    except Exception as e:
        pytest.skip(f"{model_type} not in this transformers: {str(e)[:80]}")
    try:
        with torch.device("meta"):
            return getattr(transformers, auto).from_config(config)
    except Exception as e:  # default config not buildable in this transformers version
        pytest.skip(f"{model_type}: {type(e).__name__}: {str(e)[:80]}")


def _found(M, model):
    names = {id(m): n for n, m in model.named_modules()}
    return {
        names[id(layer_list)]: sorted({type(m).__name__ for m in layer_list})
        for layer_list in M._layer_lists(model)
    }


def _eager_ssm(M, model):
    return sorted(
        {
            type(layer).__name__
            for layer, legacy in M._layers(model)
            if not legacy and M._is_ssm_layer(layer)
        }
    )


# model_type -> (layer lists found, classes kept eager as Mamba-style SSM layers)
_EXPECTED = {
    "llama": ({"model.layers": ["LlamaDecoderLayer"]}, []),
    "qwen3": ({"model.layers": ["Qwen3DecoderLayer"]}, []),
    "falcon_h1": ({"model.layers": ["FalconH1DecoderLayer"]}, []),
    "gpt_neox": ({"gpt_neox.layers": ["GPTNeoXLayer"]}, []),
    "gpt2": ({"transformer.h": ["GPT2Block"]}, []),
    "olmo_hybrid": (
        {
            "model.layers": [
                "OlmoHybridAttentionDecoderLayer",
                "OlmoHybridLinearAttentionDecoderLayer",
            ]
        },
        [],
    ),
    "mllama": (
        {"model.layers": ["MllamaCrossAttentionDecoderLayer", "MllamaSelfAttentionDecoderLayer"]},
        [],
    ),
    "jamba": (
        {"model.layers": ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]},
        ["JambaMambaDecoderLayer"],
    ),
    "zamba2": (
        {"model.layers": ["Zamba2HybridLayer", "Zamba2MambaDecoderLayer"]},
        ["Zamba2HybridLayer", "Zamba2MambaDecoderLayer"],
    ),
    "nemotron_h": ({"model.layers": ["NemotronHBlock"]}, ["NemotronHBlock"]),
    "mamba2": ({"backbone.layers": ["Mamba2Block"]}, ["Mamba2Block"]),
    "mamba": ({"backbone.layers": ["MambaBlock"]}, ["MambaBlock"]),
    "falcon_mamba": ({"backbone.layers": ["FalconMambaBlock"]}, ["FalconMambaBlock"]),
}


@pytest.mark.parametrize("model_type", sorted(_EXPECTED))
def test_text_decoder_lists_are_found(model_type):
    M = _load()
    model = _meta_model(model_type)
    lists, eager = _EXPECTED[model_type]
    found = _found(M, model)
    compiled = sorted(
        {c for classes in found.values() for c in classes} - set(_eager_ssm(M, model))
    )
    assert compiled == sorted({c for classes in lists.values() for c in classes} - set(eager))
    assert set(_eager_ssm(M, model)) <= set(eager)
    # transformers < 5 builds zamba2's hybrid layer as a plain Module: the list is not found there.
    if model_type != "zamba2" or found:
        assert found == lists


@pytest.mark.parametrize(
    "model_type, lists",
    [
        (
            "mllama",
            {
                "model.language_model.layers": [
                    "MllamaCrossAttentionDecoderLayer",
                    "MllamaSelfAttentionDecoderLayer",
                ]
            },
        ),
        ("qwen2_5_vl", {"model.language_model.layers": ["Qwen2_5_VLDecoderLayer"]}),
        ("gemma3", {"model.language_model.layers": ["Gemma3DecoderLayer"]}),
        ("llava", {"model.language_model.layers": ["LlamaDecoderLayer"]}),
        ("pixtral", {}),
    ],
)
def test_vision_towers_are_excluded(model_type, lists):
    M = _load()
    auto = "AutoModel" if model_type == "pixtral" else "AutoModelForImageTextToText"
    assert _found(M, _meta_model(model_type, auto)) == lists


def test_nemotron_h_keeps_only_mamba_blocks_eager():
    # One class for every block kind: the SSM decision is per layer, not per class.
    M = _load()
    model = _meta_model("nemotron_h")
    kinds = {}
    for layer, _ in M._layers(model):
        kinds.setdefault(type(layer.mixer).__name__, set()).add(M._is_ssm_layer(layer))
    assert kinds.pop("NemotronHMamba2Mixer") == {True}
    assert kinds and all(v == {False} for v in kinds.values())


class _Embed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)


def _layer_class(name, base = None):
    base = base or transformers.modeling_layers.GradientCheckpointingLayer

    def __init__(self):
        base.__init__(self)
        self.proj = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
    ):
        return self.proj(hidden_states) + hidden_states

    return type(name, (base,), {"__init__": __init__, "forward": forward})


def _fake_compile(calls):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(type(args[0]).__name__)
            return fn(*args, **kw)

        return compiled

    return fake_compile


def test_mixed_shared_and_tower_layers(monkeypatch):
    M = _load()
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    Attn, Ssm, Tower = (
        _layer_class("ToyAttnLayer"),
        _layer_class("ToySsmLayer"),
        _layer_class("ToyVisionLayer"),
    )
    Plain = _layer_class("ToyPlainBlock", torch.nn.Module)

    class ToyMambaMixer(torch.nn.Module):
        pass

    shared = Attn()
    text = _Embed()
    ssm = Ssm()
    ssm.mixer = ToyMambaMixer()
    text.layers = torch.nn.ModuleList([shared, ssm, shared, Attn()])
    model = torch.nn.Module()
    model.model = text
    # GradientCheckpointingLayer subclasses, but no token embedding beside them / a tower path.
    model.vision_tower = _Embed()
    model.vision_tower.layers = torch.nn.ModuleList([Tower()])
    model.blocks_without_embedding = torch.nn.ModuleList([Tower()])
    model.plain = torch.nn.Module()
    model.plain.wte = torch.nn.Embedding(4, 8)
    model.plain.h = torch.nn.ModuleList(
        [Plain(), Plain()]
    )  # no GradientCheckpointingLayer: suffix rule
    assert [c.__name__ for c in M._decoder_layer_classes(model)] == [
        "ToyAttnLayer",
        "ToySsmLayer",
        "ToyPlainBlock",
    ]
    assert len(M._layers(model)) == 5
    limit = (
        torch._dynamo.config.recompile_limit
        if hasattr(torch._dynamo.config, "recompile_limit")
        else None
    )
    wrapped = M.patch_block_compile(model)
    assert wrapped == ["ToyAttnLayer", "ToyPlainBlock"]
    assert ssm._unsloth_block_mode == M._OFF
    x = torch.randn(2, 4, 8, requires_grad = True)
    for layer in list(text.layers) + list(model.plain.h):
        layer(x).sum().backward()
    assert "ToySsmLayer" not in calls and {"ToyAttnLayer", "ToyPlainBlock"} <= set(calls)
    assert not getattr(Tower.forward, M._MARK, False)
    if limit is not None:
        torch._dynamo.config.recompile_limit = limit


def test_ssm_opt_in(monkeypatch):
    M = _load()
    monkeypatch.setattr(torch, "compile", _fake_compile([]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE_SSM", "1")
    Ssm = _layer_class("ToySsmOptInLayer")
    layer = Ssm()
    layer.ssm_state_size = 4  # Mamba-style mixer attribute
    text = _Embed()
    text.layers = torch.nn.ModuleList([layer])
    assert M.patch_block_compile(text) == ["ToySsmOptInLayer"]
    assert "_unsloth_block_mode" not in layer.__dict__


def test_skip_list_still_applies(monkeypatch):
    M = _load()
    monkeypatch.setattr(torch, "compile", _fake_compile([]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    Skip = _layer_class("GptOssDecoderLayer")
    text = _Embed()
    text.layers = torch.nn.ModuleList([Skip()])
    assert M.patch_block_compile(text) == []
    assert not getattr(Skip.forward, M._MARK, False)


def test_cache_limit_covers_every_layer_kind(monkeypatch):
    M = _load()
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    monkeypatch.setattr(config, name, 8)
    M._raise_cache_limit(5)
    assert getattr(config, name) == 20
    M._raise_cache_limit(1)
    assert getattr(config, name) == 20


def test_same_class_outside_the_decoder_stays_stock(monkeypatch):
    # T5's encoder and decoder share T5Block: only the decoder list is found, the wrap is per class.
    M = _load()
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    Shared = _layer_class("ToySharedBlockLayer")
    model = torch.nn.Module()
    model.encoder = _Embed()
    model.encoder.layers = torch.nn.ModuleList([Shared()])
    model.decoder = _Embed()
    model.decoder.layers = torch.nn.ModuleList([Shared()])
    assert M.patch_block_compile(model) == ["ToySharedBlockLayer"]
    x = torch.randn(2, 4, 8, requires_grad = True)
    model.encoder.layers[0](x).sum().backward()
    assert calls == []
    model.decoder.layers[0](x).sum().backward()
    assert calls
