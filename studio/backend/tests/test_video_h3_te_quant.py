# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted QUANTIZED conditioner route for MiniMax-H3's Diffusers path.

Mirrors tests/test_video_prequant.py, which covers the denoiser half of the same idea: the
resolver, the download-plan hooks that decide whether the base repo's dense ``text_encoder/``
shards are staged, the ConvRot INT8 arithmetic, and the memory floor recomputed from what the
load actually holds.

Network-free and (except for the two arithmetic tests) torch-free. The 27 GB artifact itself is
exercised end to end on a GPU, not here."""

from __future__ import annotations

import types

import pytest

from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family
from core.inference.video_minimax_h3 import (
    H3_DIFFUSERS_VRAM_BASE_GB,
    H3_TEXT_ENCODER_BF16_GB,
    H3_TRANSFORMER_BF16_GB,
    estimate_h3_diffusers_vram_gb,
    h3_diffusers_vram_base_gb,
    h3_transformer_resident_gb,
)
from core.inference.video_minimax_h3_te import (
    H3_TE_CONVROT_GROUP,
    H3_TE_QUANT_FILES,
    H3_TE_QUANT_REPO,
    H3_TE_READ_LAYER,
    h3_te_quant_filename,
    h3_te_quant_scheme,
    h3_te_remap_key,
    h3_te_resident_gb,
)

torch = pytest.importorskip("torch", reason = "the ConvRot arithmetic tests need torch")


def _fam(modular_workflow = "fl2va", name = "minimax-h3"):
    return types.SimpleNamespace(name = name, modular_workflow = modular_workflow)


# ── the resolver ─────────────────────────────────────────────────────────────────
def test_only_int8_has_a_hosted_conditioner():
    assert h3_te_quant_scheme("int8") == "int8"
    assert h3_te_quant_scheme("INT8") == "int8"
    # Valid text_encoder_quant modes with no hosted H3 artifact resolve to nothing rather than
    # raising: the request is well formed, this family just cannot serve it.
    for mode in ("fp8", "fp8_dynamic", "nvfp4", "fp8-dynamic", "", None):
        assert h3_te_quant_scheme(mode) is None


def test_the_hosted_filename_is_the_comfy_component_repo():
    # H3_TE_QUANT_REPO must stay the repo the Diffusers path already pulls its VAEs from, or this
    # adds a second component dependency nobody staged.
    from core.inference.video_minimax_h3 import H3_COMPONENT_REPO

    assert H3_TE_QUANT_REPO == H3_COMPONENT_REPO
    assert (
        h3_te_quant_filename("int8")
        == "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    assert h3_te_quant_filename("fp8") is None
    assert h3_te_quant_filename(None) is None


def test_every_hosted_scheme_is_a_valid_text_encoder_quant_request():
    # A scheme this module offers but normalize_te_quant refuses could never be requested.
    from core.inference.diffusion_precision import normalize_te_quant
    for scheme in H3_TE_QUANT_FILES:
        assert normalize_te_quant(scheme) == scheme


# ── the name mapping ─────────────────────────────────────────────────────────────
def test_remap_puts_comfy_names_into_the_transformers_tree():
    # ComfyUI flattens the Qwen3-VL tree; transformers nests both halves under model.
    assert (
        h3_te_remap_key("model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        h3_te_remap_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    )
    assert h3_te_remap_key("visual.blocks.3.attn.qkv.bias") == "model.visual.blocks.3.attn.qkv.bias"
    # NOT idempotent, on purpose: re-mapping an already-transformers name produces a key no module
    # owns, so an artifact re-uploaded in transformers naming fails the strict load loudly instead
    # of half-matching.
    assert h3_te_remap_key("model.language_model.layers.0.mlp.up_proj.weight") == (
        "model.language_model.language_model.layers.0.mlp.up_proj.weight"
    )


# ── the ConvRot arithmetic ───────────────────────────────────────────────────────
def test_the_convrot_hadamard_is_symmetric_and_its_own_inverse():
    from core.inference.video_minimax_h3_te import build_convrot_hadamard

    h = build_convrot_hadamard(H3_TE_CONVROT_GROUP)
    assert h.shape == (H3_TE_CONVROT_GROUP, H3_TE_CONVROT_GROUP)
    # Exactly, not approximately: the entries are +-1/16 and the products are sums of 256 of them.
    assert torch.equal(h, h.T)
    assert (h @ h - torch.eye(H3_TE_CONVROT_GROUP)).abs().max().item() == 0.0
    # Regular Hadamard: a power of 4 only. 128 is a power of 2 and must still be refused.
    for bad in (128, 100, 2):
        with pytest.raises(ValueError):
            build_convrot_hadamard(bad)


def test_rotating_the_activation_undoes_the_rotation_baked_into_the_weight():
    """The load-bearing identity: x_rot @ W_rot.T == x @ W.T.

    If this were false the INT8 conditioner would decode to noise, and a load that only checked
    file sizes would never notice. Dequantizing WITHOUT the rotation is the control."""
    from core.inference.video_minimax_h3_te import (
        _int8_convrot_linear_class,
        build_convrot_hadamard,
        rotate_convrot_activation,
    )

    torch.manual_seed(0)
    group = H3_TE_CONVROT_GROUP
    out_features, in_features = 64, group * 3
    weight = torch.randn(out_features, in_features) * 0.02
    x = torch.randn(5, in_features)

    h = build_convrot_hadamard(group)
    # The offline half, exactly as the artifact was baked: W_rot = W @ H.T blockwise.
    grouped = weight.reshape(out_features, in_features // group, group)
    weight_rot = grouped.matmul(h.T).reshape(out_features, in_features)
    # Round trip through INT8 per output channel, as the hosted checkpoint stores it.
    scale = (weight_rot.abs().amax(dim = -1, keepdim = True) / 127.0).clamp(min = 1e-30)
    qdata = torch.round(weight_rot / scale).clamp(-127, 127).to(torch.int8)

    module = _int8_convrot_linear_class()(in_features, out_features, bias = False, group_size = group)
    module.load_state_dict({"weight": qdata, "weight_scale": scale}, strict = True, assign = True)
    with torch.no_grad():
        got = module(x)
    reference = torch.nn.functional.linear(x, weight)
    # Everything but the INT8 rounding is exact, so this is a quantization-error bound, not a
    # numerical-slop one.
    assert (got - reference).norm() / reference.norm() < 0.02

    # Control: skipping the activation rotation is not "slightly worse", it is unrelated.
    unrotated = torch.nn.functional.linear(x, qdata.float() * scale)
    assert (unrotated - reference).norm() / reference.norm() > 0.5
    # And the rotation itself is an involution.
    assert torch.allclose(
        rotate_convrot_activation(rotate_convrot_activation(x, h, group), h, group), x, atol = 1e-5
    )


def test_the_int8_module_keeps_its_weight_quantized():
    """The resident footprint IS the point, so a cached dense view would defeat the whole path."""
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    group = H3_TE_CONVROT_GROUP
    module = _int8_convrot_linear_class()(group, 8, bias = False, group_size = group)
    module.load_state_dict(
        {
            "weight": torch.zeros(8, group, dtype = torch.int8),
            "weight_scale": torch.ones(8, 1),
        },
        strict = True,
        assign = True,
    )
    with torch.no_grad():
        module(torch.randn(2, group))
    assert module.weight.dtype is torch.int8
    assert sum(t.numel() * t.element_size() for t in module.buffers()) == 8 * group + 8 * 4


# ── the read layer ───────────────────────────────────────────────────────────────
def test_the_read_layer_matches_the_diffusers_pipeline():
    """MiniMax-H3 conditions on hidden_states[50]. If diffusers ever moves it, the 50-layer
    artifact stops being lossless and this must fail rather than ship a silent approximation."""
    pytest.importorskip("diffusers", reason = "reads the installed pipeline's own constant")
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    assert MiniMaxH3ModularPipeline.text_encoder_layer.fget(None) == H3_TE_READ_LAYER


# ── the memory floor ─────────────────────────────────────────────────────────────
def test_the_default_floor_is_unchanged():
    """Backwards compatibility: nothing that does not pass a size sees a different number."""
    assert h3_diffusers_vram_base_gb() == pytest.approx(H3_DIFFUSERS_VRAM_BASE_GB, abs = 0.001)
    assert estimate_h3_diffusers_vram_gb(960, 544, 124) == pytest.approx(73.68, abs = 0.02)
    assert estimate_h3_diffusers_vram_gb(1344, 768, 345) == pytest.approx(96.98, abs = 0.02)


def test_an_unpinned_floor_is_the_largest_component_not_the_sum():
    # Every component runs under CPU offload, so quantizing ONE of the two 66 GB components buys
    # nothing while both stay in the rotation.
    assert h3_diffusers_vram_base_gb(transformer_gb = 20.3) == pytest.approx(68.5, abs = 0.001)
    assert h3_diffusers_vram_base_gb(text_encoder_gb = 27.2) == pytest.approx(68.1, abs = 0.001)


def test_a_pinned_denoiser_makes_the_floor_additive():
    """The pre-quantized denoiser is pinned out of the offload rotation, so it is resident
    alongside whatever else is running. Treating that as a max under-states the floor by the whole
    denoiser, which is the direction that lets a doomed generation start."""
    pinned = h3_diffusers_vram_base_gb(transformer_gb = 20.3, transformer_pinned = True)
    assert pinned == pytest.approx(20.3 + 66.7 + 2.6, abs = 0.001)
    assert pinned > h3_diffusers_vram_base_gb(transformer_gb = 20.3)
    # A tiny conditioner cannot drag the floor under the VAEs, which still rotate through.
    assert h3_diffusers_vram_base_gb(
        text_encoder_gb = 1.0, transformer_gb = 20.3, transformer_pinned = True
    ) == pytest.approx(20.3 + 11.1 + 2.6, abs = 0.001)


def test_the_floor_matches_the_two_measured_runs():
    """Calibration, against torch.cuda.max_memory_allocated over a real 960x544x124 generation with
    the int8 denoiser pinned. Both must be covered, and neither by an absurd margin."""
    for text_encoder_gb, measured in ((H3_TEXT_ENCODER_BF16_GB, 94.62), (27.2, 55.20)):
        predicted = estimate_h3_diffusers_vram_gb(
            960,
            544,
            124,
            text_encoder_gb = text_encoder_gb,
            transformer_gb = 20.3,
            transformer_pinned = True,
        )
        assert measured <= predicted <= measured + 1.0, (text_encoder_gb, predicted, measured)


def test_quantizing_the_conditioner_is_what_finally_moves_the_floor():
    dense = estimate_h3_diffusers_vram_gb(
        960, 544, 124, transformer_gb = 20.3, transformer_pinned = True
    )
    quantized = estimate_h3_diffusers_vram_gb(
        960, 544, 124, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_pinned = True
    )
    # The saving is the conditioner delta and nothing else.
    assert dense - quantized == pytest.approx(H3_TEXT_ENCODER_BF16_GB - 27.2, abs = 0.001)
    assert quantized < H3_DIFFUSERS_VRAM_BASE_GB


def test_resident_sizes_track_the_engaged_scheme_only():
    assert h3_te_resident_gb("int8", bf16_gb = H3_TEXT_ENCODER_BF16_GB) == 27.2
    # A scheme with no hosted artifact, and a declined request recorded as None, both keep the
    # dense budget. Under-stating the floor is the expensive direction.
    for mode in (None, "fp8", "nvfp4"):
        assert h3_te_resident_gb(mode, bf16_gb = H3_TEXT_ENCODER_BF16_GB) == H3_TEXT_ENCODER_BF16_GB
    assert h3_transformer_resident_gb("int8") == 20.3
    assert h3_transformer_resident_gb(None) == H3_TRANSFORMER_BF16_GB


# ── the download plan ────────────────────────────────────────────────────────────
def test_only_a_modular_family_drops_its_dense_encoder():
    assert VideoBackend._h3_te_quant_scheme(_fam(), "int8") == "int8"
    # A conventional family casts its own dense encoder in place and still needs those shards.
    assert VideoBackend._h3_te_quant_scheme(_fam(modular_workflow = None), "int8") is None
    assert VideoBackend._h3_te_quant_scheme(_fam(), "fp8") is None
    assert VideoBackend._h3_te_quant_scheme(_fam(), None) is None


def test_an_unsupported_request_never_breaks_the_plan():
    # The plan runs before validate_load_request has had the last word on some paths, and a raise
    # here would cost the whole download plan rather than one optimisation.
    assert VideoBackend._h3_te_quant_scheme(_fam(), "not-a-scheme") is None
    assert VideoBackend._h3_te_quant_scheme(object(), "int8") is None


def test_the_real_family_resolves_the_hosted_conditioner():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.modular_workflow
    assert VideoBackend._h3_te_quant_scheme(fam, "int8") == "int8"


def test_an_unresolvable_artifact_keeps_the_dense_shards():
    """Only an artifact that really exists on the Hub earns the right to drop 62 GB of encoder."""

    class _Boom:
        def model_info(self, *_args, **_kwargs):
            raise RuntimeError("hub down")

    assert VideoBackend._h3_te_quant_hub_files("int8", _Boom()) == (None, [])
    assert VideoBackend._h3_te_quant_hub_files(None, _Boom()) == (None, [])

    class _Empty:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(siblings = [])

    assert VideoBackend._h3_te_quant_hub_files("int8", _Empty()) == (None, [])


def test_a_resolvable_artifact_is_staged_in_place_of_the_dense_shards():
    wanted = h3_te_quant_filename("int8")

    class _Api:
        def model_info(self, repo_id, **_kwargs):
            assert repo_id == H3_TE_QUANT_REPO
            return types.SimpleNamespace(
                siblings = [
                    types.SimpleNamespace(rfilename = wanted, size = 27_141_342_152),
                    types.SimpleNamespace(rfilename = "vae/other.safetensors", size = 5),
                ]
            )

    repo, files = VideoBackend._h3_te_quant_hub_files("int8", _Api())
    assert repo == H3_TE_QUANT_REPO
    # Exactly the one artifact, at its real size: the disk preflight is sized off this.
    assert files == [(wanted, 27_141_342_152)]


def test_the_dense_encoder_is_dropped_from_the_base_pull_but_its_config_is_kept():
    """The loader meta-inits from <base>/text_encoder/config.json, so dropping that would break
    the very load that made the skip safe."""
    from core.inference.diffusion_te_prequant import is_prequant_covered_weight

    covered = ("text_encoder",)
    assert is_prequant_covered_weight("text_encoder/model-00001-of-00014.safetensors", covered)
    assert not is_prequant_covered_weight("text_encoder/config.json", covered)
    assert not is_prequant_covered_weight("text_encoder/tokenizer.json", covered)
    # Other components are untouched.
    assert not is_prequant_covered_weight("vae/diffusion_pytorch_model.safetensors", covered)
    assert not is_prequant_covered_weight(
        "transformer/diffusion_pytorch_model.safetensors", covered
    )


# ── the resolved record ──────────────────────────────────────────────────────────
def test_a_declined_request_reads_as_a_fallback_not_as_never_asked():
    """The record has to keep the REQUEST on the left. Erasing it to None makes a refused fp8
    request indistinguishable from a load nobody asked to quantize."""
    from core.inference.diffusion_auto_policy import build_resolved_record

    record = build_resolved_record(
        {
            "text_encoder_quant": (
                "fp8",
                "off",
                "no hosted quantized fp8 conditioner for minimax-h3; "
                "loaded the released bfloat16 encoder instead",
            )
        }
    )
    entry = record["text_encoder_quant"]
    assert entry["requested"] == "fp8"
    assert entry["value"] == "off"
    assert entry["status"] != "as_requested"
