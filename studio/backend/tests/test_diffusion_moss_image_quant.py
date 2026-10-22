# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image quantisation wiring: the frozen keep-list and ConvRot spec the hosted builds were made with, the hosted
file names for SFT and Base, the shared text encoder, the automatic scheme order, and a clean fallback when the hosted
repos are absent. CPU only, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

SFT = "OpenMOSS-Team/MOSS-Image-SFT"
BASE = "OpenMOSS-Team/MOSS-Image-Base"
SFT_REPO = "unsloth/MOSS-Image-SFT-FP8"
BASE_REPO = "unsloth/MOSS-Image-Base-FP8"
KEEP = ("text_embedder", "time_embedding", "modulation")


def _fam(repo = SFT):
    from core.inference.diffusion_families import detect_family

    fam = detect_family(repo)
    assert fam is not None and fam.name == "moss-image"
    return fam


@pytest.fixture
def offline_empty(tmp_path, monkeypatch):
    """No mirror, no Hub, an empty cache: the hosted repos do not exist yet."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_MIRROR", raising = False)
    cache = tmp_path / "hub"
    cache.mkdir()
    return str(cache)


def test_keep_list_and_convrot_spec_match_the_hosted_builds():
    fam = _fam()
    assert fam.profile.quant_exclude_tokens == KEEP
    spec = fam.profile.int8_convrot
    assert spec.group == 256 and spec.default_on
    assert spec.suffixes == (
        "proj.q",
        "proj.k",
        "proj.v",
        "proj.o",
        "proj.ffn_up_gate",
        "proj.ffn_down",
    )
    assert spec.repo == SFT_REPO and spec.variant_repos == (BASE_REPO,)
    assert spec.filename == "MOSS-Image-SFT-INT8-ConvRot.safetensors"


@pytest.mark.parametrize("family", ["moss-image", "moss-image-base"])
@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_quant_tables_answer_to_both_recorded_family_names(family, scheme):
    """The Base files record family "moss-image-base": the validator's exclude check reads it from the file."""
    from core.inference import diffusion_transformer_quant as tq

    tokens = tq.exclude_tokens_for_scheme(scheme, family)
    assert set(KEEP) <= set(tokens)
    # The hosted files' recorded exclude lists (int8 adds the generic M = 1 tokens; fp8 is the keep-list alone).
    recorded = {
        "int8": [
            "norm",
            "_mod",
            "modulation",
            "timestep_embed",
            "guidance_embed",
            "time_text_embed",
            "pooled",
            "time_embed",
            "text_embedder",
            "time_embedding",
            "modulation",
        ],
        "fp8": list(KEEP),
    }[scheme]
    assert set(tokens) <= set(recorded)


@pytest.mark.parametrize(
    "base, repo, names",
    [
        (
            SFT,
            SFT_REPO,
            {
                "int8": (
                    "MOSS-Image-SFT-INT8-ConvRot.safetensors",
                    "MOSS-Image-SFT-INT8.safetensors",
                ),
                "fp8": ("MOSS-Image-SFT-FP8.safetensors",),
            },
        ),
        (
            BASE,
            BASE_REPO,
            {
                "int8": (
                    "MOSS-Image-Base-INT8-ConvRot.safetensors",
                    "MOSS-Image-Base-INT8.safetensors",
                ),
                "fp8": ("MOSS-Image-Base-FP8.safetensors",),
            },
        ),
    ],
)
def test_each_base_resolves_its_own_hosted_files(base, repo, names):
    from core.inference.diffusion_prequant import candidate_filenames_of, resolve_prequant_source

    fam = _fam(base)
    for scheme, expected in names.items():
        source = resolve_prequant_source(fam, scheme, base_repo = base)
        assert source is not None and source.location == repo, (base, scheme)
        candidates = candidate_filenames_of(source)
        positions = [candidates.index(n) for n in expected]
        assert positions == sorted(positions), candidates
        other = "SFT" if "Base" in repo else "Base"
        assert not any(f"MOSS-Image-{other}-" in n for n in candidates), candidates
    # Base never takes SFT's artifact for a scheme it has no row for
    assert resolve_prequant_source(fam, "nvfp4", base_repo = base) is None


@pytest.mark.parametrize("family", ["moss-image", "moss-image-base"])
def test_auto_never_picks_fp8_but_an_explicit_fp8_is_allowed(family):
    from core.inference import diffusion_transformer_quant as tq

    for cap in ((10, 0), (9, 0), (8, 9), (8, 0)):
        order = tq._auto_scheme_order(family, "cuda", cap)
        assert order and order[0] == "int8", (cap, order)
        assert "fp8" not in order and "mxfp8" not in order, (cap, order)
    assert not tq._family_denied(family, "fp8")
    assert "fp8" in tq._auto_scheme_order("qwen-image-2.1", "cuda", (10, 0))


def test_auto_keeps_the_5b_dit_bf16_while_it_fits():
    """Small-DiT rule (as FLUX.2 / FLUX.3-klein, Z-Image): bf16 while a resident plan fits, INT8-ConvRot once bf16
    would offload, an explicit precision honoured (the rule reads only AUTO)."""
    from types import SimpleNamespace

    from core.inference.diffusion import _auto_keeps_bf16_reason, _auto_quant_eager_reason
    from core.inference.diffusion_families import (
        SMALL_DIT_BF16_WHEN_RESIDENT,
        small_dit_rule_excludes,
    )
    from core.inference.diffusion_transformer_quant import auto_bf16_when_resident_reason

    fam = _fam()
    assert auto_bf16_when_resident_reason("moss-image") == SMALL_DIT_BF16_WHEN_RESIDENT
    # Both 5B bases sit inside the rule; only FLUX.2 klein 9B variants are carved out.
    for base in ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base"):
        assert not small_dit_rule_excludes(base)
        assert _auto_keeps_bf16_reason(fam, "pipeline", base) is not None
    assert _auto_keeps_bf16_reason(fam, "pipeline") is not None
    assert _auto_keeps_bf16_reason(fam, "gguf") is None

    def plan(policy, required, budget):
        return SimpleNamespace(
            offload_policy = policy,
            estimates = {"resident_required_mib": required, "safe_device_budget_mib": budget},
        )

    assert _auto_quant_eager_reason(fam, plan("none", 36060, 156376)) is not None
    assert _auto_quant_eager_reason(fam, plan("none", 36060, 20000)) is None
    assert _auto_quant_eager_reason(fam, plan("group", 36060, 20000)) is None


def test_text_encoder_file_and_equivalence():
    from core.inference import diffusion_te_prequant as tp
    from core.inference.diffusion_precision import resolve_te_quant_request
    from core.inference.diffusion_text_encoder_trim import family_trims_lm_head

    fam = _fam()
    assert fam.te_quant_auto == "int8"
    mode, auto = resolve_te_quant_request(None, fam.te_quant_auto)
    assert mode == "int8" and auto
    source = tp.resolve_te_prequant_source(fam, "text_encoder", "int8")
    assert source.location == SFT_REPO
    assert source.filename == "MOSS-Image-SFT-text_encoder-INT8-ConvRot.safetensors"
    assert (
        source.fallback_filenames == ()
        and tp.resolve_te_prequant_source(fam, "text_encoder", "fp8") is None
    )
    assert family_trims_lm_head("moss-image")
    # Base reuses the SFT file: the validator accepts the SFT base_model_id on a Base load, and nothing else
    assert tp.te_base_equivalent(SFT, BASE) and tp.te_base_equivalent(BASE, SFT)
    assert not tp.te_base_equivalent(SFT, "Qwen/Qwen3.5-9B")
    assert 0.684 < tp.te_int8_convrot_budget_scale(fam) < 0.7


def test_absent_hosted_repos_fall_back_to_runtime_quant_and_the_dense_encoder(offline_empty):
    from core.inference import diffusion_te_prequant as tp
    from core.inference.diffusion_denoiser_prequant import (
        denoiser_prequant_cached,
        denoiser_prequant_pipe_kwargs,
    )
    for base in (SFT, BASE):
        fam = _fam(base)
        for scheme in ("int8", "fp8"):
            assert not denoiser_prequant_cached(
                fam, scheme, base_repo = base, cache_dir = offline_empty
            )
            # {} = the released denoiser is loaded and quantised at runtime (int8 rotates with the same spec)
            assert (
                denoiser_prequant_pipe_kwargs(
                    fam,
                    base,
                    scheme = scheme,
                    dtype = None,
                    device = "cpu",
                    local_files_only = True,
                    cache_dir = offline_empty,
                )
                == {}
            )
        sources = {"text_encoder": tp.resolve_te_prequant_source(fam, "text_encoder", "int8")}
        assert tp.te_prequant_sources_confirmed(sources, local_files_only = True) is False


def test_automatic_text_encoder_without_the_file_stays_dense_bf16():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_auto_policy import RESOLVED_FELL_BACK
    from core.inference.diffusion_precision import AutoTeQuant, quantize_text_encoders

    encoder = torch.nn.Sequential(torch.nn.Linear(256, 256))
    weight = encoder[0].weight
    pipe = SimpleNamespace(text_encoder = encoder)
    target = SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    outcome = quantize_text_encoders(pipe, target, mode = AutoTeQuant("int8"), family = "moss-image")
    assert outcome.mode is None and outcome.status == RESOLVED_FELL_BACK
    assert pipe.text_encoder[0].weight is weight and weight.dtype == torch.float32


def test_denoiser_seed_resolves_the_in_tree_class(monkeypatch):
    """The hosted builds are seeded into the pipeline assembly; the class is Studio's port, not a diffusers attr."""
    pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("diffusers without AutoencoderKLFlux2")
    from core.inference import diffusion_denoiser_prequant as ddp
    from core.inference import diffusion_prequant as pq
    from core.inference.diffusion_moss_image import MOSSImageTransformer

    seen = {}
    sentinel = object()

    def fake_load(transformer_cls, base_repo, source, **kwargs):
        seen["cls"], seen["base"], seen["family"] = transformer_cls, base_repo, kwargs.get("family")
        return sentinel

    monkeypatch.setattr(pq, "load_prequantized_transformer", fake_load)
    monkeypatch.setattr(
        ddp,
        "denoiser_prequant_source",
        lambda *a, **k: SimpleNamespace(kind = "repo", location = BASE_REPO),
    )
    out = ddp.denoiser_prequant_pipe_kwargs(
        _fam(BASE), BASE, scheme = "int8", dtype = None, device = "cpu"
    )
    assert out == {"transformer": sentinel}
    assert seen == {"cls": MOSSImageTransformer, "base": BASE, "family": "moss-image"}


def test_a_handed_in_encoder_gets_the_vision_drop_and_offload_guard(monkeypatch):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("diffusers without AutoencoderKLFlux2")
    from core.inference import diffusion_moss_image as moss

    class FakeGatedDeltaNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(4, 4, 3, groups = 4)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2)
            self.layer = FakeGatedDeltaNet()

    class Stub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(moss, "assert_transformers_supported", lambda: None)
    monkeypatch.setattr(moss, "_tokenizer_class", lambda repo, common: Stub)
    monkeypatch.setattr(
        moss.MOSSImageScheduler, "from_pretrained", classmethod(lambda cls, *a, **k: "sched")
    )
    monkeypatch.setattr(
        diffusers.AutoencoderKLFlux2, "from_pretrained", classmethod(lambda cls, *a, **k: "vae")
    )
    monkeypatch.setattr(moss, "MOSSImagePipeline", lambda **kw: kw)
    encoder = Encoder()
    parts = moss.load_moss_image_pipeline(
        "OpenMOSS-Team/MOSS-Image-SFT",
        torch.bfloat16,
        transformer = torch.nn.Linear(2, 2),
        text_encoder = encoder,
        local_files_only = True,
    )
    assert parts["text_encoder"] is encoder and encoder.visual is None
    assert getattr(encoder.layer, "_unsloth_conv_guard", False)


def test_tokenizer_class_comes_from_its_config(tmp_path, offline_empty):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    from core.inference import diffusion_moss_image as moss

    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"})
    )
    assert moss._tokenizer_class(str(tmp_path), {}) is transformers.Qwen2Tokenizer
    assert moss._tokenizer_class(str(tmp_path / "missing"), {}) is transformers.AutoTokenizer
    assert (
        moss._tokenizer_class(SFT, {"local_files_only": True, "cache_dir": offline_empty})
        is transformers.AutoTokenizer
    )


def test_activation_casts_read_a_declared_compute_dtype():
    """diffusers' GGUFLinear stores uint8 weights and declares ``compute_dtype``; a plain Linear casts to its weight."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("diffusers without AutoencoderKLFlux2")
    from core.inference.diffusion_moss_image import _compute_dtype

    plain = torch.nn.Linear(2, 2).to(torch.bfloat16)
    assert _compute_dtype(plain) is torch.bfloat16
    packed = torch.nn.Linear(2, 2)
    packed.compute_dtype = torch.bfloat16
    assert _compute_dtype(packed) is torch.bfloat16


def test_measured_peak_seeds_the_quantised_placement():
    """The compiled-tier seed keeps a 5 GB int8 denoiser resident on a 16 GB card instead of the flat 8 GiB estimate."""
    from core.inference.diffusion_memory import measured_image_runtime_mib

    assert measured_image_runtime_mib("moss-image", "default") == 3584
    assert measured_image_runtime_mib("moss-image", "max") == 3584
    assert measured_image_runtime_mib("moss-image", "off", dense_transformer_mib = 9800) is None


def test_moss_denies_nvfp4():
    # NVFP4 W4A4 scored LPIPS 0.383 against bf16 (SFT, 12 prompts); INT8-ConvRot 0.058.
    for repo in (SFT, BASE):
        assert "nvfp4" in _fam(repo).profile.scheme_deny


def test_fp8_te_cast_keeps_the_gated_delta_conv1d_in_compute_dtype():
    """Qwen3.5's gated-delta layers hand ``linear_attn.conv1d.weight`` straight to the causal-conv kernel without
    calling the module, so the layerwise upcast hook never runs: the cast must leave conv1d out."""
    import torch

    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    except ImportError:
        pytest.skip("transformers without Qwen3.5 (MOSS-Image needs >= 5.3)")
    from core.inference.diffusion_precision import _cast_fp8

    cfg = Qwen3_5TextConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        linear_num_key_heads = 1,
        linear_num_value_heads = 2,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4,
    )
    encoder = Qwen3_5TextModel(cfg).to(torch.bfloat16)
    _cast_fp8(encoder, SimpleNamespace(dtype = torch.bfloat16))

    convs = [m for n, m in encoder.named_modules() if n.endswith("linear_attn.conv1d")]
    assert convs, "the tiny config built no gated-delta layer"
    assert all(m.weight.dtype == torch.bfloat16 for m in convs)
    fp8 = [
        m
        for m in encoder.modules()
        if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.float8_e4m3fn
    ]
    assert fp8, "the cast stored no Linear in fp8"
