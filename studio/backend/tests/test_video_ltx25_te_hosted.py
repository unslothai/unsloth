# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5's hosted int8 ConvRot Gemma 4 encoder: when a load takes it, its planning, its load and its fallback."""

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

    def __init__(
        self,
        d = 256,
        layers = 2,
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, d)
        self.model.language_model.layers = nn.ModuleList(_Layer(d) for _ in range(layers))


BASE = "Lightricks/LTX-2.5-Diffusers"


def _confirmed(monkeypatch, value):
    from core.inference import diffusion_te_prequant as tpq

    calls = []

    def _fake(sources, **_kw):
        calls.append(tuple(sorted(sources)))
        return value

    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", _fake)
    return calls


def test_unset_request_takes_the_hosted_file_only_when_confirmed(monkeypatch):
    # Auto takes the hosted int8 file when it is on disk or listed by the Hub, else keeps the dense shards (bf16).
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    calls = _confirmed(monkeypatch, True)
    for mode in (None, "auto", ""):
        assert v25.ltx25_te_prequant_sources(mode, BASE)["text_encoder"].filename == (
            v25.LTX25_TE_HOSTED_FILE
        )
    assert calls == [("text_encoder",)] * 3
    _confirmed(monkeypatch, False)
    for mode in (None, "auto", ""):
        assert v25.ltx25_te_prequant_sources(mode, BASE) == {}
    # An explicit int8 never needs the confirmation: a miss falls back to the at-load quantization.
    assert v25.ltx25_te_prequant_sources("int8", BASE)["text_encoder"].filename == (
        v25.LTX25_TE_HOSTED_FILE
    )
    # Kill switch: the bf16 auto from before the hosted file, confirmed or not.
    calls = _confirmed(monkeypatch, True)
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "0")
    assert v25.ltx25_te_prequant_sources(None, BASE) == {}
    assert v25.ltx25_te_prequant_sources("auto", BASE) == {}
    assert calls == []
    # Auto on a custom pack / the single-file release never takes it either.
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    assert v25.ltx25_te_prequant_sources(None, "someone/LTX-2.5-Diffusers") == {}
    assert v25.ltx25_te_prequant_sources(None, v25.LTX25_SINGLE_FILE_REPO) == {}


def test_resolve_settles_the_automatic_pick_per_load(monkeypatch):
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    _confirmed(monkeypatch, True)
    for mode in (None, "auto"):
        resolved = v25.ltx25_resolve_te_request(mode, BASE)
        assert resolved == "int8"
        assert v25.ltx25_te_quant_scheme(resolved) == "int8"
    # A pre-fetch that could not get the file keeps bf16, never the at-load quantization.
    failed = v25.ltx25_resolve_te_request(None, BASE, {"text_encoder": "EntryNotFoundError: 404"})
    assert failed == "none" and v25.ltx25_te_quant_scheme(failed) is None
    assert v25.ltx25_resolve_te_request(None, BASE, {"text_encoder_2": "x"}) == "int8"
    _confirmed(monkeypatch, False)
    assert v25.ltx25_resolve_te_request(None, BASE) == "none"
    assert v25.ltx25_resolve_te_request("auto", "someone/ltx25-pack") == "none"
    # Explicit requests are untouched: int8 keeps its hosted-else-at-load path, off / none keep bf16.
    for mode in ("int8", "none", "off", "fp8"):
        assert v25.ltx25_resolve_te_request(mode, BASE, {"text_encoder": "x"}) == mode
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "0")
    assert v25.ltx25_resolve_te_request(None, BASE) is None
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "1")
    assert v25.ltx25_resolve_te_request(None, BASE) is None
    assert v25.ltx25_te_quant_scheme(v25.ltx25_resolve_te_request(None, BASE)) == "int8"


def test_int8_request_resolves_to_the_hosted_file(monkeypatch):
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    source = v25.ltx25_te_prequant_sources("int8", BASE)["text_encoder"]
    assert (source.kind, source.location, source.filename) == (
        "repo",
        v25.LTX25_TE_HOSTED_REPO,
        v25.LTX25_TE_HOSTED_FILE,
    )
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "1")
    for mode in (None, "auto"):
        assert (
            v25.ltx25_te_prequant_sources(mode, BASE)["text_encoder"].filename
            == v25.LTX25_TE_HOSTED_FILE
        )
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    # none / off pin the bf16 encoder: no hosted file, the dense shards stay in the plan.
    assert v25.ltx25_te_prequant_sources("none", BASE) == {}
    assert v25.ltx25_te_prequant_sources("off", BASE) == {}
    # Only the base it was cut from, and never the single-file release (its encoder file carries the projections).
    assert v25.ltx25_te_prequant_sources("int8", "someone/ltx-2.5-finetune") == {}
    # Exact identity: a custom pack that only shares the folder name keeps its own encoder weights.
    assert v25.ltx25_te_prequant_sources("int8", "/packs/custom/LTX-2.5-Diffusers") == {}
    assert v25.ltx25_te_prequant_sources("int8", "someone/LTX-2.5-Diffusers") == {}
    cached = "/hf/hub/models--Lightricks--LTX-2.5-Diffusers/snapshots/abc123"
    assert (
        v25.ltx25_te_prequant_sources("int8", cached)["text_encoder"].location
        == v25.LTX25_TE_HOSTED_REPO
    )
    assert v25.ltx25_te_prequant_sources("int8", v25.LTX25_SINGLE_FILE_REPO) == {}
    assert v25.ltx25_te_prequant_sources("fp8", BASE) == {}


def test_the_video_planner_stages_the_hosted_file(monkeypatch):
    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    _confirmed(monkeypatch, False)
    fam = detect_video_family(BASE)
    sources = VideoBackend._te_prequant_sources(fam, "int8", base = BASE)
    assert sources["text_encoder"].filename == v25.LTX25_TE_HOSTED_FILE
    assert VideoBackend._te_prequant_sources(fam, None, base = BASE) == {}
    assert VideoBackend._te_prequant_sources(fam, "none", base = BASE) == {}
    # Confirmed, the unset request stages the hosted file too (and so drops the dense text_encoder/ shards).
    _confirmed(monkeypatch, True)
    assert VideoBackend._te_prequant_sources(fam, None, base = BASE)["text_encoder"].filename == (
        v25.LTX25_TE_HOSTED_FILE
    )
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
    # The dense fallback's peak travels with the hosted file, so the worker can refuse it on a short host.
    assert client.config["dense_te_gb"] == v25.LTX25_TE_BF16_GB


def _write_tiny_encoder(folder: Path) -> _TinyGemma:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = _TinyGemma().to(torch.bfloat16)
    te = folder / "text_encoder"
    te.mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()}, str(te / "model.safetensors")
    )
    (te / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in model.state_dict()}})
    )
    (te / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"]})
    )
    return model


def _build(pack: Path, out: Path) -> dict:
    """The builder runs the worker's own at-load path (here against the stand-in transformers) and saves it."""
    import build_ltx25_te_int8_convrot as builder
    return builder.build(str(pack), out, base = BASE, device = "cpu")


def _fake_transformers(monkeypatch, folder: Path):
    from safetensors.torch import load_file
    class _Cls(_TinyGemma):
        @classmethod
        def from_pretrained(
            cls,
            repo,
            subfolder = None,
            **_kw,
        ):
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

    base = {
        "repo": str(tmp_path / "pack"),
        "device": "cpu",
        "dtype": "bfloat16",
        "te_quant": "int8",
    }
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
    for layer_a, layer_b in zip(
        hosted.model.language_model.layers, at_load.model.language_model.layers
    ):
        assert type(layer_a.mlp.down_proj).__name__ == "Int8ConvRotLinear"
        assert torch.equal(layer_a.self_attn.q_proj(x), layer_b.self_attn.q_proj(x))


def test_missing_or_foreign_hosted_file_falls_back_to_the_at_load_build(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "pack")
    _fake_transformers(monkeypatch, tmp_path / "pack")
    base = {
        "repo": str(tmp_path / "pack"),
        "device": "cpu",
        "dtype": "bfloat16",
        "te_quant": "int8",
    }
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


def test_a_rejected_hosted_file_refuses_the_dense_fallback_without_room(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "pack")
    _fake_transformers(monkeypatch, tmp_path / "pack")
    monkeypatch.setattr(worker, "_hosted_path", lambda spec, common: spec["mirror_path"])
    monkeypatch.setattr(worker, "_available_host_bytes", lambda: 8 * 1000**3)
    config = {
        "repo": str(tmp_path / "pack"),
        "device": "cpu",
        "dtype": "bfloat16",
        "te_quant": "int8",
        "dense_te_gb": 23.9,
        "te_prequant": {
            "repo": "r",
            "filename": "x",
            "mirror_path": str(tmp_path / "absent"),
            "base": BASE,
        },
    }
    with pytest.raises(RuntimeError, match = "needs about 24 GB"):
        worker._load(config)
    monkeypatch.setattr(worker, "_available_host_bytes", lambda: 64 * 1000**3)
    worker._load(config)
    assert config["_te_source"] == "quantized_at_load"


@pytest.mark.parametrize(
    "dtype, free_gb, refused",
    [
        ("float32", 30, True),
        ("float32", 50, False),
        ("bfloat16", 30, False),
        ("float16", 30, False),
    ],
)
def test_the_dense_fallback_room_check_prices_the_load_dtype(dtype, free_gb, refused, monkeypatch):
    monkeypatch.setattr(worker, "_available_host_bytes", lambda: free_gb * 1000**3)
    config = {"dtype": dtype, "dense_te_gb": v25.LTX25_TE_BF16_GB}
    if refused:
        with pytest.raises(RuntimeError, match = "needs about 48 GB"):
            worker._require_room_for_dense(config, OSError("hosted file absent"))
    else:
        worker._require_room_for_dense(config, OSError("hosted file absent"))


def test_offline_dense_fallback_still_tries_the_cached_hub_id(tmp_path, monkeypatch):
    _write_tiny_encoder(tmp_path / "hub")
    _fake_transformers(monkeypatch, tmp_path / "hub")
    (tmp_path / "snap").mkdir()
    config = {
        "repo": str(tmp_path / "snap"),
        "hub_repo": str(tmp_path / "hub"),
        "device": "cpu",
        "dtype": "bfloat16",
        "te_quant": None,
        "local_files_only": True,
    }
    _tok, model = worker._load(config)
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, nn.Linear)


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


def _hf_cache_entry(root: Path, repo: str, filename: str, payload: bytes) -> Path:
    commit = "0" * 40
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True)
    (folder / "refs" / "main").write_text(commit)
    target = folder / "snapshots" / commit / filename
    target.parent.mkdir(parents = True)
    target.write_bytes(payload)
    return target


def test_hosted_path_resolves_a_copy_under_the_other_cache_root(tmp_path, monkeypatch):
    # The server stages with reuse_other_cache_root, so a copy cached only under huggingface_hub's own root counts as
    # staged (and the dense shards are dropped). The worker must find it too, offline.
    import huggingface_hub.constants as hub_constants

    repo, filename = "unsloth/LTX-2.5-FP8", "LTX-2.5-text_encoder-INT8-ConvRot.safetensors"
    default_root, configured = tmp_path / "default", tmp_path / "configured"
    configured.mkdir()
    staged = _hf_cache_entry(default_root, repo, filename, b"hosted")
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(default_root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spec = {"repo": repo, "filename": filename, "mirror_path": None}
    common = {"token": None, "cache_dir": str(configured), "local_files_only": True}
    assert Path(worker._hosted_path(spec, common)).read_bytes() == staged.read_bytes()
    # A copy under the configured root still wins.
    mine = _hf_cache_entry(configured, repo, filename, b"configured")
    assert Path(worker._hosted_path(spec, common)) == mine


def test_hosted_path_ignores_the_base_revision(tmp_path, monkeypatch):
    # ``common`` carries the BASE repo's planned commit; the hosted file lives in unsloth/LTX-2.5-FP8, where that
    # commit does not exist, so it must be read at the hosted repo's own ref.
    import huggingface_hub.constants as hub_constants

    repo, filename = "unsloth/LTX-2.5-FP8", "LTX-2.5-text_encoder-INT8-ConvRot.safetensors"
    root = tmp_path / "cache"
    staged = _hf_cache_entry(root, repo, filename, b"hosted")
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spec = {"repo": repo, "filename": filename, "mirror_path": None}
    common = {
        "token": None,
        "cache_dir": str(root),
        "local_files_only": True,
        "revision": "f" * 40,
    }
    assert Path(worker._hosted_path(spec, common)).read_bytes() == staged.read_bytes()
    assert common["revision"] == "f" * 40


def test_builder_drops_a_tied_lm_head(tmp_path, monkeypatch):
    import build_ltx25_te_int8_convrot as builder

    class _Tied(_TinyGemma):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(256, 16, bias = False)
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def _load(config):
        config["_te_source"] = "quantized_at_load"
        return object(), _Tied().to(torch.bfloat16)

    monkeypatch.setattr(worker, "_load", _load)
    out = tmp_path / "hosted.safetensors"
    builder.build("unused", out, base = BASE, device = "cpu")
    from safetensors import safe_open

    with safe_open(str(out), "pt") as fh:
        keys = set(fh.keys())
    assert "model.language_model.embed_tokens.weight" in keys
    assert "lm_head.weight" not in keys


def test_builder_builds_on_the_cpu_when_the_card_is_absent(tmp_path, monkeypatch):
    import build_ltx25_te_int8_convrot as builder

    seen: dict = {}

    def _load(config):
        seen.update(config)
        config["_te_source"] = "quantized_at_load"
        return object(), _TinyGemma().to(torch.bfloat16)

    monkeypatch.setattr(worker, "_load", _load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    builder.build("unused", tmp_path / "hosted.safetensors", base = BASE, device = "cuda:0")
    assert seen["device"] == "cpu" and seen["resident"] is False
