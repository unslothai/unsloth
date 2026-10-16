# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5's hosted int8 ConvRot Gemma 4 encoder: the default, its planning, its load and its fallback."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.inference import video_ltx25 as v25
from core.inference import video_ltx25_te_worker as worker

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias = False)
        self.k_proj = nn.Linear(d, d // 2, bias = False)
        self.o_proj = nn.Linear(d, d, bias = False)
        self.q_norm = nn.LayerNorm(d)


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, 2 * d, bias = False)
        self.up_proj = nn.Linear(d, 2 * d, bias = False)
        self.down_proj = nn.Linear(2 * d, d, bias = False)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _Mlp(d)


class _TinyGemma(nn.Module):
    """The shape of Gemma4UnifiedForConditionalGeneration the worker walks: model.language_model.layers."""

    def __init__(self, d = 256, layers = 2):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, d)
        self.model.language_model.layers = nn.ModuleList(_Layer(d) for _ in range(layers))

BASE = "Lightricks/LTX-2.5-Diffusers"


def test_unset_request_resolves_to_the_hosted_file(monkeypatch):
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    for mode in (None, "auto", "int8"):
        source = v25.ltx25_te_prequant_sources(mode, BASE)["text_encoder"]
        assert (source.kind, source.location, source.filename) == (
            "repo",
            v25.LTX25_TE_HOSTED_REPO,
            v25.LTX25_TE_HOSTED_FILE,
        )
    # none / off pin the bf16 encoder: no hosted file, the dense shards stay in the plan.
    assert v25.ltx25_te_prequant_sources("none", BASE) == {}
    assert v25.ltx25_te_prequant_sources("off", BASE) == {}
    # Only the base it was cut from, and never the single-file release (its encoder file carries the projections).
    assert v25.ltx25_te_prequant_sources(None, "someone/ltx-2.5-finetune") == {}
    assert v25.ltx25_te_prequant_sources(None, v25.LTX25_SINGLE_FILE_REPO) == {}
    assert v25.ltx25_te_prequant_sources("fp8", BASE) == {}
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "0")
    assert v25.ltx25_te_prequant_sources(None, BASE) == {}


def test_the_video_planner_stages_the_hosted_file():
    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    fam = detect_video_family(BASE)
    sources = VideoBackend._te_prequant_sources(fam, None, base = BASE)
    assert sources["text_encoder"].filename == v25.LTX25_TE_HOSTED_FILE
    assert VideoBackend._te_prequant_sources(fam, "none", base = BASE) == {}


def test_client_hands_the_worker_the_hosted_file(monkeypatch):
    monkeypatch.setattr(v25, "ensure_text_encoder_sidecar", lambda: "/nonexistent")
    monkeypatch.setattr(v25.Ltx25TextEncoderClient, "start", lambda self: None)
    common = dict(
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
    )
    client = v25.start_text_encoder(repo = "/snap/ltx25", base_repo = BASE, te_quant = "int8", **common)
    spec = client.config["te_prequant"]
    assert (spec["repo"], spec["filename"], spec["base"]) == (
        v25.LTX25_TE_HOSTED_REPO,
        v25.LTX25_TE_HOSTED_FILE,
        v25.LTX25_TE_HOSTED_BASE,
    )
    # The scoped snapshot may have skipped the dense shards, so the fallback knows the Hub id.
    assert client.config["hub_repo"] == BASE
    assert v25.start_text_encoder(repo = BASE, te_quant = None, **common).config["te_prequant"] is None


def _write_tiny_encoder(folder: Path) -> _TinyGemma:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = _TinyGemma().to(torch.bfloat16)
    te = folder / "text_encoder"
    te.mkdir(parents = True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(te / "model.safetensors"))
    (te / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in model.state_dict()}})
    )
    (te / "config.json").write_text(json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"]}))
    return model


def _build(pack: Path, out: Path) -> dict:
    """The builder runs the worker's own at-load path (here against the stand-in transformers) and saves it."""
    import build_ltx25_te_int8_convrot as builder

    return builder.build(str(pack), out, base = BASE, device = "cpu")


def _fake_transformers(monkeypatch, folder: Path):
    from safetensors.torch import load_file

    class _Cls(_TinyGemma):
        @classmethod
        def from_pretrained(cls, repo, subfolder = None, **_kw):
            path = Path(repo) / (subfolder or "") / "model.safetensors"
            if not path.is_file():
                raise OSError(f"no weights at {path}")
            model = _TinyGemma().to(torch.bfloat16)
            model.load_state_dict(load_file(str(path)))
            return model

        def __init__(self, _config = None):
            super().__init__()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__ = "5.10.2",
            Gemma4UnifiedForConditionalGeneration = _Cls,
            AutoTokenizer = SimpleNamespace(from_pretrained = lambda *a, **k: object()),
            AutoConfig = SimpleNamespace(
                from_pretrained = lambda *a, **k: SimpleNamespace(tie_word_embeddings = False)
            ),
        ),
    )


def test_hosted_load_equals_the_at_load_build(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "pack")
    _fake_transformers(monkeypatch, tmp_path / "pack")
    out = tmp_path / "hosted.safetensors"
    first = _build(tmp_path / "pack", out)
    assert first["quantized_linears"] == 12
    # Deterministic: a rebuild is byte-identical.
    again = _build(tmp_path / "pack", tmp_path / "again.safetensors")
    assert again["sha256"] == first["sha256"]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == first["sha256"]

    base = {"repo": str(tmp_path / "pack"), "device": "cpu", "dtype": "bfloat16", "te_quant": "int8"}
    spec = {"repo": "unsloth/LTX-2.5-FP8", "filename": "x", "mirror_path": str(out), "base": BASE}
    config_hosted = dict(base, te_prequant = dict(spec))
    _tok, hosted = worker._load(config_hosted)
    config_at_load = dict(base)
    _tok, at_load = worker._load(config_at_load)
    assert config_hosted["_te_source"] == "hosted"
    assert config_at_load["_te_source"] == "quantized_at_load"
    a, b = hosted.state_dict(), at_load.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key
    x = torch.randn(3, 256, dtype = torch.bfloat16)
    for layer_a, layer_b in zip(hosted.model.language_model.layers, at_load.model.language_model.layers):
        assert type(layer_a.mlp.down_proj).__name__ == "Int8ConvRotLinear"
        assert torch.equal(layer_a.self_attn.q_proj(x), layer_b.self_attn.q_proj(x))


def test_missing_or_foreign_hosted_file_falls_back_to_the_at_load_build(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "pack")
    _fake_transformers(monkeypatch, tmp_path / "pack")
    base = {"repo": str(tmp_path / "pack"), "device": "cpu", "dtype": "bfloat16", "te_quant": "int8"}
    missing = {
        "repo": "unsloth/LTX-2.5-FP8",
        "filename": "x",
        "mirror_path": str(tmp_path / "absent.safetensors"),
        "base": BASE,
    }
    monkeypatch.setattr(worker, "_hosted_path", lambda spec, common: spec["mirror_path"])
    config = dict(base, te_prequant = missing)
    _tok, model = worker._load(config)
    assert config["_te_source"] == "quantized_at_load"
    assert type(model.model.language_model.layers[0].mlp.gate_proj).__name__ == "Int8ConvRotLinear"
    # Built from another base: refused, the dense encoder is quantized instead.
    out = tmp_path / "hosted.safetensors"
    _build(tmp_path / "pack", out)
    config = dict(base, te_prequant = dict(missing, mirror_path = str(out), base = "someone/else"))
    worker._load(config)
    assert config["_te_source"] == "quantized_at_load"


def test_none_keeps_bf16_and_dense_fallback_reads_the_hub_id(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "hub")
    _fake_transformers(monkeypatch, tmp_path / "hub")
    # A scoped snapshot without the encoder shards: the dense read retries the pack's Hub id.
    (tmp_path / "snap").mkdir()
    config = {
        "repo": str(tmp_path / "snap"),
        "hub_repo": str(tmp_path / "hub"),
        "device": "cpu",
        "dtype": "bfloat16",
        "te_quant": None,
    }
    _tok, model = worker._load(config)
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, nn.Linear)
    assert config.get("_te_source") is None
