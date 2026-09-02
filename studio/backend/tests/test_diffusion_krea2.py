# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the Krea 2 per-component pipeline loader (CPU-only, no network)."""

from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    # rope_scaling already parsed (a 5.x runtime exposing the alias): untouched.
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
    (tmp_path / "model_index.json").write_text(json.dumps({"is_distilled": True, "patch_size": 2}))
    assert _load_model_index(str(tmp_path)) == {"is_distilled": True, "patch_size": 2}


def test_load_model_index_wraps_truncated_local_json(tmp_path):
    (tmp_path / "model_index.json").write_text('{"patch_size":', encoding = "utf-8")

    with pytest.raises(ValueError, match = r"model_index\.json.*local model directory") as exc_info:
        _load_model_index(str(tmp_path))

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_load_model_index_wraps_invalid_utf8(tmp_path):
    (tmp_path / "model_index.json").write_bytes(b'\xff{"patch_size": 2}')

    with pytest.raises(ValueError, match = r"model_index\.json.*local model directory") as exc_info:
        _load_model_index(str(tmp_path))

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_load_model_index_accepts_utf8_bom(tmp_path):
    (tmp_path / "model_index.json").write_bytes(b'\xef\xbb\xbf{"patch_size": 2}')

    assert _load_model_index(str(tmp_path)) == {"patch_size": 2}


@pytest.mark.parametrize(
    "payload", ["[]", "null", "3", "2.5", '"str"', "true", "false", '[{"a": 1}]']
)
def test_load_model_index_rejects_non_object_json(tmp_path, payload):
    # All of these parsed and reached the caller, which then died on ``.get`` one frame away.
    (tmp_path / "model_index.json").write_text(payload, encoding = "utf-8")

    with pytest.raises(
        ValueError, match = r"model_index\.json.*must contain a JSON object"
    ) as exc_info:
        _load_model_index(str(tmp_path))

    assert exc_info.value.__cause__ is None


def test_load_model_index_wraps_unreadable_local_file(monkeypatch, tmp_path):
    # Present but unreadable (0600, EIO, a Windows AV lock): the OSError used to be swallowed and
    # re-reported as "not found". Faulted at the read because chmod is a no-op as root.
    (tmp_path / "model_index.json").write_text('{"patch_size": 2}', encoding = "utf-8")
    original = Path.read_text

    def _deny(self, *args, **kwargs):
        if self.name == "model_index.json":
            raise PermissionError(errno.EACCES, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _deny)

    with pytest.raises(ValueError, match = r"model_index\.json.*local model directory") as exc_info:
        _load_model_index(str(tmp_path))

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_load_model_index_wraps_a_nesting_bomb(monkeypatch, tmp_path):
    # Valid JSON and valid UTF-8, so neither guard above sees it; the parser blows the stack.
    # Faulted directly because the depth is not portable: 3.14 parses what 3.10-3.13 reject.
    (tmp_path / "model_index.json").write_text('{"a": 1}', encoding = "utf-8")
    monkeypatch.setattr(
        json, "loads", lambda *args, **kwargs: (_ for _ in ()).throw(RecursionError("too deep"))
    )

    with pytest.raises(ValueError, match = r"model_index\.json.*local model directory") as exc_info:
        _load_model_index(str(tmp_path))

    assert isinstance(exc_info.value.__cause__, RecursionError)


def test_load_model_index_missing_local_file(tmp_path):
    with pytest.raises(FileNotFoundError, match = r"model_index\.json not found in local model dir"):
        _load_model_index(str(tmp_path))


def test_load_model_index_wraps_malformed_hub_cache_content(monkeypatch, tmp_path):
    import huggingface_hub

    downloaded = tmp_path / "downloaded-model_index.json"
    downloaded.write_text('{"patch_size":', encoding = "utf-8")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *_args, **_kwargs: str(downloaded)
    )

    with pytest.raises(ValueError, match = r"model_index\.json.*Hub/cache") as exc_info:
        _load_model_index("krea/Krea-2-Turbo", local_files_only = True)

    assert str(downloaded) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


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
        # Hand-written fakes with EXACT signatures, so they have to follow the production one:
        # load_krea2_pipeline now passes local_files_only down to every component load.
        lambda repo_id, hf_token = None, local_files_only = False: SimpleNamespace(tag = "tokenizer"),
    )
    monkeypatch.setattr(
        "core.inference.diffusion_krea2.load_krea2_text_encoder",
        lambda repo_id, dtype, hf_token = None, local_files_only = False: SimpleNamespace(
            tag = "text_encoder"
        ),
    )

    pipe = load_krea2_pipeline(str(tmp_path), "bf16")

    # Turbo's fixed-mu schedule rides on is_distilled and dropping any of these silently degrades generations, so the ctor kwargs are asserted exactly.
    assert captured["pipeline"]["is_distilled"] is True
    assert captured["pipeline"]["patch_size"] == 2
    assert captured["pipeline"]["text_encoder_select_layers"] == [2, 5, 8]
    assert pipe.transformer.tag == "transformer"
    # A prebuilt transformer (single-file/quant path) must be used as-is.
    prebuilt = SimpleNamespace(tag = "prebuilt")
    pipe = load_krea2_pipeline(str(tmp_path), "bf16", transformer = prebuilt)
    assert pipe.transformer is prebuilt


def test_a_corrupt_index_is_rejected_before_any_component_is_built(monkeypatch, tmp_path):
    """A few KB against the ~35 GB it configures, so it is read first. Read last, a clear message
    still costs a full load to reach, which is most of what the opaque traceback cost."""
    (tmp_path / "model_index.json").write_text('{"_class_name": "Krea2Pipe', encoding = "utf-8")

    built: list = []

    class _Records:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            built.append(self.tag)
            return SimpleNamespace(tag = self.tag)

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            FlowMatchEulerDiscreteScheduler = _Records("scheduler"),
            AutoencoderKLQwenImage = _Records("vae"),
            Krea2Transformer2DModel = _Records("transformer"),
            Krea2Pipeline = lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setattr(
        "core.inference.diffusion_krea2.load_krea2_tokenizer",
        lambda repo_id, hf_token = None, local_files_only = False: built.append("tokenizer"),
    )
    monkeypatch.setattr(
        "core.inference.diffusion_krea2.load_krea2_text_encoder",
        lambda repo_id, dtype, hf_token = None, local_files_only = False: built.append("text_encoder"),
    )

    with pytest.raises(ValueError, match = r"model_index\.json"):
        load_krea2_pipeline(str(tmp_path), "bf16")

    assert built == []


# ── registry / trust / int8 exclusion wiring ─────────────────────────────────


def test_load_krea2_pipeline_requires_krea_capable_diffusers(monkeypatch):
    # On diffusers < 0.39 (no Krea2Pipeline) the loader must fail fast with the upgrade hint, not a bare AttributeError mid-load.
    import pytest

    fake = SimpleNamespace(__version__ = "0.38.0")
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    with pytest.raises(RuntimeError, match = "0.39"):
        load_krea2_pipeline("krea/Krea-2-Turbo", "bf16")


def test_krea2_family_wiring():
    from core.inference.diffusion import _is_trusted_diffusion_repo
    from core.inference.diffusion_families import (
        default_generation_params,
        detect_family,
        family_sd_cpp_supported,
    )
    from core.inference.diffusion_transformer_quant import TQ_INT8, exclude_tokens_for_scheme

    fam = detect_family("krea/Krea-2-Turbo")
    assert fam is not None and fam.name == KREA2_FAMILY_NAME
    # Both vendor repos are non-GGUF allowlisted (Turbo for inference, Raw for training); no sd.cpp mapping, so diffusers fallback.
    assert _is_trusted_diffusion_repo("krea/Krea-2-Turbo")
    assert _is_trusted_diffusion_repo("krea/Krea-2-Raw")
    assert not family_sd_cpp_supported(fam)
    # Krea2TimestepEmbedding runs at M = batch; int8 (torch._int_mm, M above 16) must skip it.
    assert "time_embed" in exclude_tokens_for_scheme(TQ_INT8)
    # Adapters train on Raw but run on Turbo, so the family carries a deploy override.
    assert fam.deploy_base_repo == "krea/Krea-2-Turbo"
    # The OpenAI /v1/images/generations route reads (steps, guidance) from this table. Krea Turbo is distilled (8 steps, no
    # CFG); Raw is the undistilled base at 52 steps / CFG 3.5, so its more specific key must beat "krea".
    assert default_generation_params("krea/Krea-2-Turbo") == (8, 0.0)
    assert default_generation_params("krea/Krea-2-Raw") == (52, 3.5)


# ── training wiring ──────────────────────────────────────────────────────────


def test_krea2_training_registry(dit_train_host):
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
    # Krea's guidance: train LoRAs on the undistilled Raw model and run them on Turbo, so Raw leads the training bases.
    assert info["default_base"] == "krea/Krea-2-Raw"
    assert info["base_repos"] == ["krea/Krea-2-Raw", "krea/Krea-2-Turbo"]
    assert info["supports_compile"] is True
    # Deploy previews the adapter on Turbo, not the Raw checkpoint it trained on, so the UI loads the distilled recipe; other families leave this None.
    assert info["deploy_base"] == "krea/Krea-2-Turbo"
    assert {i["name"]: i for i in family_train_infos()}["flux.1"]["deploy_base"] is None


def test_krea2_spec_registered_with_authors_targets():
    from core.training.diffusion_dit_trainer import _KREA2_TARGETS, _SPECS

    spec = _SPECS["krea-2"]
    assert spec.force_bf16 is True
    assert spec.lora_targets == _KREA2_TARGETS
    # The authors' full recommended set: attention + SwiGLU + text fusion + embedders.
    for t in ("to_q", "to_gate", "ff.up", "text_fusion.projector", "time_mod_proj"):
        assert t in _KREA2_TARGETS


def test_krea2_collate_and_forward_roundtrip():
    # spec.forward imports Krea2Pipeline (prepare_position_ids), so this needs a real diffusers install; CI runs without one.
    pytest.importorskip("diffusers")
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
            # Echo the packed sequence: unpack(pack(x)) == x proves the inlined packing mirrors Krea2Pipeline exactly.
            return (kwargs["hidden_states"],)

    noisy = torch.randn(2, 16, 1, 8, 8)
    timesteps = torch.tensor([250.0, 750.0])
    pred = spec.forward(
        _FakeTransformer(), noisy, timesteps, None, (pe_b, mask_b), None, "cpu", torch.float32
    )
    assert torch.equal(pred, noisy)
    # [B, (H/2)*(W/2), C*4] patches, one shared [(txt+img), 3] position grid, and the [0, 1] timestep convention.
    assert captured["hidden_states"].shape == (2, 16, 64)
    assert captured["position_ids"].shape == (8 + 16, 3)
    assert torch.allclose(captured["timestep"], torch.tensor([0.25, 0.75]))
    assert captured["encoder_attention_mask"] is mask_b
