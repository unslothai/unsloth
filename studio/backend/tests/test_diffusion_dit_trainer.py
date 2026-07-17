# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the flow-matching DiT LoRA trainer (FLUX.1 / FLUX.2 / Qwen-Image / Z-Image).

CPU-only: cover family resolution, the per-family spec table, the QLoRA prequant
heuristic, the bf16-only guard, and the gated-repo name check. The full training loop is
exercised by the live GPU smokes, not here."""

from __future__ import annotations

import sys
import types

import pytest

from core.training.diffusion_dit_trainer import (
    _FLUX2_TARGETS,
    _FLUX_TARGETS,
    _GATED_TRAIN_REPOS,
    _QWEN_TARGETS,
    _SPECS,
    _ZIMAGE_TARGETS,
    _apply_mxfp8_training,
    _assert_gated_access,
    _mx_module_filter,
    _repo_is_prequantized,
    _resolve_base_precision,
    _select_lora_targets,
    _should_compile,
    run_dit_lora_training,
)
from core.training.diffusion_train_common import (
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    family_train_infos,
    train_precision_modes,
)


def test_specs_cover_the_dit_families():
    assert set(_SPECS) == {
        "flux.1", "qwen-image", "z-image", "krea-2", "flux.2-klein", "flux.2-dev"
    }
    # FLUX / Qwen share the added-kv attention target set; Z-Image and Krea 2 are single-stream.
    assert "add_q_proj" in _SPECS["flux.1"].lora_targets
    assert "add_q_proj" in _SPECS["qwen-image"].lora_targets
    assert "add_q_proj" not in _SPECS["z-image"].lora_targets
    assert "add_q_proj" not in _SPECS["krea-2"].lora_targets
    # Z-Image, Qwen, Krea 2 and both FLUX.2 variants are bf16-only.
    assert _SPECS["z-image"].force_bf16 is True
    assert _SPECS["qwen-image"].force_bf16 is True
    assert _SPECS["krea-2"].force_bf16 is True
    assert _SPECS["flux.2-klein"].force_bf16 is True
    assert _SPECS["flux.2-dev"].force_bf16 is True


def test_flux2_specs_share_targets_and_split_conditioners():
    # dev and Klein share the transformer (and so the LoRA target set) but load different
    # conditioning pipelines and save through their own pipeline class.
    klein, dev = _SPECS["flux.2-klein"], _SPECS["flux.2-dev"]
    assert klein.lora_targets == dev.lora_targets == _FLUX2_TARGETS
    # The fused single-stream projection is targeted; the plain to_out suffix is not (it
    # would also match the double-stream ModuleList container, which peft cannot wrap).
    assert "to_qkv_mlp_proj" in _FLUX2_TARGETS
    assert "to_out.0" in _FLUX2_TARGETS
    assert "to_out" not in _FLUX2_TARGETS
    assert klein.load_conditioners is not dev.load_conditioners
    assert klein.save is not dev.save
    assert klein.load_transformer is dev.load_transformer
    # The Mistral stack makes dev far heavier than the 4B Klein.
    assert dev.dense_bf16_gb > klein.dense_bf16_gb


def test_select_lora_targets_uses_family_default_for_generic_config():
    # normalized() fills lora_target_modules with the generic DEFAULT_LORA_TARGETS when a
    # caller doesn't set it, so that value must resolve to the family's targets (which add
    # the DiT-specific projections), not stay stuck on the generic SDXL list.
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _FLUX_TARGETS) == _FLUX_TARGETS
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _QWEN_TARGETS) == _QWEN_TARGETS
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _ZIMAGE_TARGETS) == _ZIMAGE_TARGETS


def test_select_lora_targets_explicit_override_wins():
    # Any OTHER explicit tuple is a deliberate override and must win over the family spec.
    override = ("to_q", "to_k")
    assert _select_lora_targets(override, _FLUX_TARGETS) == override
    # The default request path (config carrying the generic default) reaches the spec.
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg.lora_target_modules == DEFAULT_LORA_TARGETS
    assert (
        _select_lora_targets(cfg.lora_target_modules, _SPECS["flux.1"].lora_targets)
        == _FLUX_TARGETS
    )


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", True),
        ("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", True),
        ("some/model-int4", True),
        ("black-forest-labs/FLUX.1-dev", False),
        ("Tongyi-MAI/Z-Image-Turbo", False),
    ],
)
def test_prequant_heuristic(repo, expected):
    assert _repo_is_prequantized(repo) is expected


def test_zimage_rejects_fp16_before_loading():
    # bf16-only families must refuse an explicit fp16 request up front (no model load).
    cfg = DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image-Turbo",
        data_dir = "does-not-exist",
        output_dir = "o",
        mixed_precision = "fp16",
    )
    with pytest.raises(ValueError, match = "bf16"):
        run_dit_lora_training(cfg)


def test_flux2_rejects_fp16_before_loading():
    # Both FLUX.2 variants resolve from their repo names, and both are bf16-only: an
    # explicit fp16 request fails in normalized() itself, before anything loads. Klein's
    # base is ungated, so this exercises the precision guard directly (no token in play).
    ok = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.2-klein-4B", data_dir = "d", output_dir = "o"
    ).normalized()
    assert ok.resolved_family == "flux.2-klein"
    assert (
        DiffusionLoraConfig(
            base_model = "black-forest-labs/FLUX.2-dev", data_dir = "d", output_dir = "o"
        )
        .normalized()
        .resolved_family
        == "flux.2-dev"
    )
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.2-klein-4B",
        data_dir = "does-not-exist",
        output_dir = "o",
        mixed_precision = "fp16",
    )
    with pytest.raises(ValueError, match = "bf16"):
        run_dit_lora_training(cfg)


def test_flux2_bases_pass_the_trusted_base_gate():
    # The FLUX.2 bases are training-side additions to the loader's trust allowlist, so the
    # pre-download trust gate must accept them (and still refuse an arbitrary repo).
    from core.training.diffusion_train_common import _assert_trusted_base_model

    _assert_trusted_base_model("black-forest-labs/FLUX.2-klein-4B")
    _assert_trusted_base_model("black-forest-labs/FLUX.2-dev")
    with pytest.raises(ValueError, match = "untrusted"):
        _assert_trusted_base_model("someone/random-flux2-finetune")


def test_gated_access_requires_token():
    assert "black-forest-labs/flux.1-dev" in _GATED_TRAIN_REPOS
    assert "black-forest-labs/flux.2-dev" in _GATED_TRAIN_REPOS
    # No token -> clear, actionable error before any download.
    with pytest.raises(ValueError, match = "gated"):
        _assert_gated_access("black-forest-labs/FLUX.1-dev", None)
    with pytest.raises(ValueError, match = "gated"):
        _assert_gated_access("black-forest-labs/FLUX.1-dev", "   ")
    with pytest.raises(ValueError, match = "gated"):
        _assert_gated_access("black-forest-labs/FLUX.2-dev", None)
    # With a token, or for a non-gated repo, it is a no-op.
    _assert_gated_access("black-forest-labs/FLUX.1-dev", "hf_realtoken")
    _assert_gated_access("black-forest-labs/FLUX.2-dev", "hf_realtoken")
    _assert_gated_access("Tongyi-MAI/Z-Image-Turbo", None)
    _assert_gated_access("black-forest-labs/FLUX.2-klein-4B", None)  # Klein is open


def test_family_train_infos_lists_dit_families():
    infos = {i["name"]: i for i in family_train_infos()}
    for fam in ("sdxl", "flux.1", "qwen-image", "z-image", "flux.2-klein", "flux.2-dev"):
        assert fam in infos, f"{fam} missing from family_train_infos"
        assert infos[fam]["default_base"]
        assert infos[fam]["base_repos"]
        assert "resolution" in infos[fam]["defaults"]
    # FLUX default bases are the gated dev repos; their notes flag the license requirement.
    assert infos["flux.1"]["default_base"] == "black-forest-labs/FLUX.1-dev"
    assert "gated" in infos["flux.1"]["vram_note"].lower()
    assert infos["flux.2-dev"]["default_base"] == "black-forest-labs/FLUX.2-dev"
    assert "gated" in infos["flux.2-dev"]["vram_note"].lower()
    # Klein-4B is open.
    assert infos["flux.2-klein"]["default_base"] == "black-forest-labs/FLUX.2-klein-4B"
    assert "gated" not in infos["flux.2-klein"]["vram_note"].lower()
    # Z-Image defaults to the prequant nf4 repo for QLoRA.
    assert "4bit" in infos["z-image"]["default_base"].lower()


def test_family_train_infos_sdxl_supports_compile_without_precision_modes(monkeypatch):
    # Regional compile now applies to every family (the SDXL trainer compiles its U-Net
    # blocks too), but base_precision stays DiT-only, so SDXL advertises no precision modes
    # while a DiT family (z-image) keeps its own. Pin the precision list so the assertion
    # holds regardless of the test host's GPU capability.
    import core.training.diffusion_train_common as dtc

    monkeypatch.setattr(dtc, "train_precision_modes", lambda: (["nf4", "bf16", "auto"], "auto"))
    infos = {i["name"]: i for i in family_train_infos()}
    assert infos["sdxl"]["supports_compile"] is True
    assert infos["sdxl"]["precision_modes"] == []
    assert infos["z-image"]["supports_compile"] is True
    assert infos["z-image"]["precision_modes"] == ["nf4", "bf16", "auto"]


# ── mxfp8 base precision (DiT dense speed mode) ───────────────────────────────
def _linear(
    in_features,
    out_features,
    bias = False,
):
    import torch.nn as nn
    return nn.Linear(in_features, out_features, bias = bias)


def test_mx_module_filter_accepts_dense_block_linear():
    # A bias-free 3072x3072 attention/FFN linear at a normal block fqn is a valid mxfp8 target.
    assert _mx_module_filter(_linear(3072, 3072), "blocks.0.ff.up") is True


def test_mx_module_filter_skips_biased_linear():
    # The torchao 0.17 MX training path drops the bias term (its linear override computes
    # input @ weight_t only), so an mxfp8'd biased FROZEN linear would silently lose its bias and
    # corrupt the base output the LoRA regresses against. Biased linears must stay bf16.
    assert _mx_module_filter(_linear(3072, 3072, bias = True), "blocks.0.ff.up") is False


def test_resolve_base_precision_explicit_mxfp8_requires_blackwell(monkeypatch):
    # An explicit mxfp8 request on a non-Blackwell CUDA GPU must fail fast: its MX GEMM has no
    # kernel below sm100 and would otherwise crash at the first training step, after a full dense
    # transformer load. /info only advertises mxfp8 on sm100+, so this mirrors that gate.
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    cfg = types.SimpleNamespace(base_precision = "mxfp8", mixed_precision = "bf16", base_model = "x")
    with pytest.raises(ValueError, match = "Blackwell"):
        _resolve_base_precision(cfg, None, "cuda")


def test_resolve_base_precision_explicit_mxfp8_ok_on_blackwell(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    cfg = types.SimpleNamespace(base_precision = "mxfp8", mixed_precision = "bf16", base_model = "x")
    assert _resolve_base_precision(cfg, None, "cuda") == "mxfp8"


def test_mx_module_filter_skips_lora_and_proj_out():
    # LoRA-owned modules (adapters stay high precision) and the output projection are
    # excluded, mirroring the fp8 filter's guards.
    lin = _linear(3072, 3072)
    assert _mx_module_filter(lin, "blocks.0.attn.to_q.lora_A.default") is False
    assert _mx_module_filter(lin, "proj_out") is False
    assert _mx_module_filter(lin, "x.proj_out.y") is False


def test_mx_module_filter_rejects_non_block_aligned_dims():
    # MX block scaling tiles 32-wide, so a dim not divisible by 32 (3000) is rejected.
    assert _mx_module_filter(_linear(3000, 3072), "blocks.0.ff.up") is False


def test_mx_module_filter_rejects_non_linear():
    import torch.nn as nn

    # A non-Linear module is never a target even if it exposes matching feature counts.
    assert _mx_module_filter(nn.LayerNorm(3072), "blocks.0.norm") is False


def test_should_compile_auto_mxfp8_on_cuda():
    # auto compiles the dense speed modes on cuda; int8 stays eager (torchao subclass);
    # an explicit "off" wins over the mode.
    cfg = DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o")
    assert _should_compile(cfg, False, "cuda", base_precision = "mxfp8") is True
    assert _should_compile(cfg, False, "cuda", base_precision = "int8") is False
    off = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", compile_transformer = "off"
    )
    assert _should_compile(off, False, "cuda", base_precision = "mxfp8") is False


def test_apply_mxfp8_training_failure_falls_back_with_warning(monkeypatch):
    # An unavailable torchao MX path must never be fatal: force both API revisions' imports to
    # raise, then assert the helper returns False and emits exactly one warning naming mxfp8.
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", None)
    monkeypatch.setitem(sys.modules, "torchao.prototype.moe_training.config", None)
    events = []
    ok = _apply_mxfp8_training(object(), lambda e: events.append(e))
    assert ok is False
    warnings = [e for e in events if e["type"] == "warning"]
    assert len(warnings) == 1
    assert "mxfp8" in warnings[0]["message"]


def test_mxfp8_training_config_falls_back_to_the_torchao_0_17_api(monkeypatch):
    # torchao 0.17 removed prototype.mx_formats.MXLinearConfig in favour of the
    # MXFP8TrainingOpConfig recipe API; the config helper must fall back to it so the advertised
    # mxfp8 mode keeps engaging on those installs instead of silently training dense bf16.
    from types import SimpleNamespace

    from core.training.diffusion_dit_trainer import _mxfp8_training_config

    calls = {}

    class _Recipe:
        MXFP8_RCEIL = "mxfp8_rceil"

    class _OpConfig:
        @staticmethod
        def from_recipe(recipe):
            calls["recipe"] = recipe
            return "cfg-0.17"

    fake_config = SimpleNamespace(MXFP8TrainingOpConfig = _OpConfig, MXFP8TrainingRecipe = _Recipe)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", None)
    monkeypatch.setitem(
        sys.modules, "torchao.prototype.moe_training", SimpleNamespace(config = fake_config)
    )
    monkeypatch.setitem(sys.modules, "torchao.prototype.moe_training.config", fake_config)
    assert _mxfp8_training_config() == "cfg-0.17"
    assert calls["recipe"] == _Recipe.MXFP8_RCEIL


def _patch_capability(monkeypatch, capability):
    # Drive train_precision_modes' GPU probe: pretend CUDA is present at the given tensor
    # core capability (fp8 needs sm89+, mxfp8 needs sm100+). The torchao probe is stubbed
    # functional so these tests exercise the CAPABILITY gate on hosts without torchao
    # (the CPU-only CI runner does not install it). is_bf16_supported must be stubbed True
    # too: the dense modes gate on it, and an Ada/Blackwell GPU is by definition bf16-capable,
    # so without this the modes collapse to nf4 on a CPU runner where the real probe is False
    # (the test otherwise only passes on a bf16 GPU host).
    import torch

    import core.training.diffusion_train_common as dtc

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(dtc, "has_functional_torchao", lambda: True)


def test_train_precision_modes_blackwell_lists_mxfp8(monkeypatch):
    # sm100 (Blackwell) exposes both fp8 and mxfp8, ordered before the "auto" pick.
    _patch_capability(monkeypatch, (10, 0))
    modes, recommended = train_precision_modes()
    assert "mxfp8" in modes and "fp8" in modes
    assert modes.index("mxfp8") < modes.index("auto")
    assert modes.index("fp8") < modes.index("auto")
    assert recommended == "auto"


def test_train_precision_modes_ada_has_fp8_without_mxfp8(monkeypatch):
    # sm89 (Ada) is fp8-capable but not block-scaled mxfp8-capable.
    _patch_capability(monkeypatch, (8, 9))
    modes, _ = train_precision_modes()
    assert "fp8" in modes
    assert "mxfp8" not in modes


def test_train_precision_modes_newer_blackwell_has_mxfp8(monkeypatch):
    # Any capability >= sm100 keeps mxfp8 (sm120 here).
    _patch_capability(monkeypatch, (12, 0))
    modes, _ = train_precision_modes()
    assert "mxfp8" in modes


def test_train_precision_modes_pre_ampere_is_nf4_only(monkeypatch):
    # A pre-Ampere GPU EMULATES bf16 (is_bf16_supported() True) but has no native bf16 tensor
    # cores; the DiT trainer requires native bf16, so /info must offer nf4 only. Otherwise it
    # advertises a start that evicts resident models and then fails the trainer's bf16 guard.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda *a, **k: True
    )  # emulation reports True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))  # Turing
    modes, recommended = train_precision_modes()
    assert modes == ["nf4"]
    assert recommended == "nf4"
