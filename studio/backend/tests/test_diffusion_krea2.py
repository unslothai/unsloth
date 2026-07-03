"""Unit tests for the Krea 2 per-component pipeline loader (CPU-only, no network)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from core.inference.diffusion_krea2 import (
    KREA2_FAMILY_NAME,
    _load_model_index,
    load_krea2_pipeline,
    remap_rope_parameters,
)


# ── rope_parameters (transformers 5.x) -> rope_scaling (4.x) remap ──────────


def test_remap_rope_parameters_copies_5x_values():
    cfg = SimpleNamespace(
        rope_scaling = None,
        rope_theta = 1000000.0,
        rope_parameters = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5000000,
            "rope_type": "default",
        },
    )
    remap_rope_parameters(cfg)
    # rope_theta is hoisted to the top-level slot, the rest lands in rope_scaling.
    assert cfg.rope_theta == 5000000
    assert cfg.rope_scaling == {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    }


def test_remap_rope_parameters_noop_on_5x_runtime_or_plain_4x_config():
    # rope_scaling already parsed (5.x runtime exposing the alias): untouched.
    parsed = {"rope_type": "default", "mrope_section": [1, 2, 3]}
    cfg = SimpleNamespace(rope_scaling = parsed, rope_theta = 7.0, rope_parameters = {"x": 1})
    remap_rope_parameters(cfg)
    assert cfg.rope_scaling is parsed
    assert cfg.rope_theta == 7.0
    # No rope_parameters at all (a plain 4.x-exported config): untouched.
    cfg = SimpleNamespace(rope_scaling = None, rope_theta = 7.0)
    remap_rope_parameters(cfg)
    assert cfg.rope_scaling is None


# ── model_index.json resolution ──────────────────────────────────────────────


def test_load_model_index_from_local_path(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"is_distilled": True, "patch_size": 2})
    )
    assert _load_model_index(str(tmp_path)) == {"is_distilled": True, "patch_size": 2}


# ── pipeline assembly threads the model_index init config ────────────────────


def test_load_krea2_pipeline_threads_init_config(monkeypatch, tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "is_distilled": True,
                "patch_size": 2,
                "text_encoder_select_layers": [2, 5, 8],
            }
        )
    )

    captured: dict = {}

    class _FromPretrained:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            captured.setdefault("components", {})[self.tag] = (repo_id, kwargs)
            return SimpleNamespace(tag = self.tag)

    def _pipeline_ctor(**kwargs):
        captured["pipeline"] = kwargs
        return SimpleNamespace(**kwargs)

    fake_diffusers = SimpleNamespace(
        FlowMatchEulerDiscreteScheduler = _FromPretrained("scheduler"),
        AutoencoderKLQwenImage = _FromPretrained("vae"),
        Krea2Transformer2DModel = _FromPretrained("transformer"),
        Krea2Pipeline = _pipeline_ctor,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(
        "core.inference.diffusion_krea2.load_krea2_tokenizer",
        lambda repo_id, hf_token = None: SimpleNamespace(tag = "tokenizer"),
    )
    monkeypatch.setattr(
        "core.inference.diffusion_krea2.load_krea2_text_encoder",
        lambda repo_id, dtype, hf_token = None: SimpleNamespace(tag = "text_encoder"),
    )

    pipe = load_krea2_pipeline(str(tmp_path), "bf16")

    # Turbo's fixed-mu schedule rides on is_distilled; dropping any of these would
    # silently degrade generations, so the ctor kwargs are asserted exactly.
    assert captured["pipeline"]["is_distilled"] is True
    assert captured["pipeline"]["patch_size"] == 2
    assert captured["pipeline"]["text_encoder_select_layers"] == [2, 5, 8]
    assert pipe.transformer.tag == "transformer"
    # A prebuilt transformer (single-file/quant path) must be used as-is.
    prebuilt = SimpleNamespace(tag = "prebuilt")
    pipe = load_krea2_pipeline(str(tmp_path), "bf16", transformer = prebuilt)
    assert pipe.transformer is prebuilt


# ── registry / trust / int8 exclusion wiring ─────────────────────────────────


def test_krea2_family_wiring():
    from core.inference.diffusion import _is_trusted_diffusion_repo
    from core.inference.diffusion_families import detect_family, family_sd_cpp_supported
    from core.inference.diffusion_transformer_quant import TQ_INT8, exclude_tokens_for_scheme

    fam = detect_family("krea/Krea-2-Turbo")
    assert fam is not None and fam.name == KREA2_FAMILY_NAME
    # The vendor repo is non-GGUF allowlisted; no sd.cpp mapping -> diffusers fallback.
    assert _is_trusted_diffusion_repo("krea/Krea-2-Turbo")
    assert not family_sd_cpp_supported(fam)
    # Krea2TimestepEmbedding runs at M = batch; int8 (torch._int_mm, M > 16) must skip it.
    assert "time_embed" in exclude_tokens_for_scheme(TQ_INT8)


# ── training wiring ──────────────────────────────────────────────────────────


def test_krea2_training_registry():
    from core.inference.diffusion_families import trainable_family_names
    from core.training.diffusion_train_common import (
        family_train_infos,
        get_trainer,
        train_defaults,
    )
    from core.training.diffusion_dit_trainer import run_dit_lora_training

    assert "krea-2" in trainable_family_names()
    assert get_trainer("krea-2") is run_dit_lora_training
    # The Krea 2 authors' recommended starting point (their reference script defaults).
    assert train_defaults("krea-2") == {
        "lora_rank": 32,
        "learning_rate": 3e-4,
        "resolution": 512,
    }
    info = {i["name"]: i for i in family_train_infos()}["krea-2"]
    assert info["default_base"] == "krea/Krea-2-Turbo"
    assert info["supports_compile"] is True


def test_krea2_spec_registered_with_authors_targets():
    from core.training.diffusion_dit_trainer import _KREA2_TARGETS, _SPECS

    spec = _SPECS["krea-2"]
    assert spec.force_bf16 is True
    assert spec.lora_targets == _KREA2_TARGETS
    # The authors' full recommended set: attention + SwiGLU + text fusion + embedders.
    for t in ("to_q", "to_gate", "ff.up", "text_fusion.projector", "time_mod_proj"):
        assert t in _KREA2_TARGETS


def test_krea2_collate_and_forward_roundtrip():
    import torch
    from core.training.diffusion_dit_trainer import _SPECS

    spec = _SPECS["krea-2"]
    # Two fixed-length embed entries collate to a plain concat with the mask batched.
    entries = [
        (torch.randn(1, 8, 12, 16), torch.ones(1, 8, dtype = torch.int64)),
        (torch.randn(1, 8, 12, 16), torch.ones(1, 8, dtype = torch.int64)),
    ]
    pe_b, mask_b = spec.collate(entries, "cpu", torch.float32)
    assert pe_b.shape == (2, 8, 12, 16)
    assert mask_b.shape == (2, 8)

    captured = {}

    class _FakeTransformer:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            # Echo the packed sequence: unpack(pack(x)) == x proves the inlined
            # packing mirrors Krea2Pipeline exactly (they are mutual inverses).
            return (kwargs["hidden_states"],)

    noisy = torch.randn(2, 16, 1, 8, 8)
    timesteps = torch.tensor([250.0, 750.0])
    pred = spec.forward(
        _FakeTransformer(), noisy, timesteps, None, (pe_b, mask_b), None, "cpu", torch.float32
    )
    assert torch.equal(pred, noisy)
    # [B, (H/2)*(W/2), C*4] patches, one shared [(txt+img), 3] position grid, and the
    # [0, 1] timestep convention.
    assert captured["hidden_states"].shape == (2, 16, 64)
    assert captured["position_ids"].shape == (8 + 16, 3)
    assert torch.allclose(captured["timestep"], torch.tensor([0.25, 0.75]))
    assert captured["encoder_attention_mask"] is mask_b
