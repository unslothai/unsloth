# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 family registration, the HunyuanImage structured exclusion, and the
curated krea/Krea-2-LoRA-* catalog entries. Pure-module tests: no torch, no network."""

import pytest

from core.inference.diffusion import _is_trusted_diffusion_repo
from core.inference.diffusion_auto_policy import family_bf16_components_gb
from core.inference.diffusion_families import (
    IDEOGRAM4_FAMILY_NAME,
    default_generation_params,
    detect_family,
    excluded_model_reason,
)
from core.inference.diffusion_lora import _CURATED, list_loras


# ── ideogram-4 family detection ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "repo_id",
    [
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "ideogram-ai/ideogram-4-nf4-diffusers",
    ],
)
def test_detect_family_ideogram4_repos(repo_id):
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == IDEOGRAM4_FAMILY_NAME
    assert fam.pipeline_class == "Ideogram4Pipeline"
    assert fam.transformer_class == "Ideogram4Transformer2DModel"
    # The vendor ships no bf16 repo: the raw-float8 export is the family base.
    assert fam.base_repo == "ideogram-ai/ideogram-4-fp8"


def test_detect_family_ideogram4_override():
    fam = detect_family("some/local-path", override = "ideogram-4")
    assert fam is not None and fam.name == IDEOGRAM4_FAMILY_NAME
    assert detect_family("x", override = "ideogram4").name == IDEOGRAM4_FAMILY_NAME


def test_ideogram4_repos_are_trusted_non_gguf():
    # The three official vendor pipelines load via from_pretrained, which is gated
    # to the unsloth org + the explicit allowlist.
    for rid in (
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "ideogram-ai/ideogram-4-nf4-diffusers",
    ):
        assert _is_trusted_diffusion_repo(rid)
    assert not _is_trusted_diffusion_repo("ideogram-ai/some-future-repo")


# ── FLUX.1 Krea dev (flux.1 family variant) ──────────────────────────────────
@pytest.mark.parametrize(
    "repo_id",
    [
        "black-forest-labs/FLUX.1-Krea-dev",
        "QuantStack/FLUX.1-Krea-dev-GGUF",
        # A local GGUF pick where the family keyword lives in the filename.
        "QuantStack/FLUX.1-Krea-dev-GGUF/flux1-krea-dev-Q4_K_M.gguf",
    ],
)
def test_detect_family_flux1_krea_dev(repo_id):
    # Krea's FLUX.1-dev finetune keeps the exact dev layout, so it must resolve to the
    # existing flux.1 family (FluxPipeline), never to krea-2 (a different arch).
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "flux.1"
    assert fam.pipeline_class == "FluxPipeline"


def test_flux1_krea_dev_is_trusted_non_gguf():
    # The gated official pipeline loads via from_pretrained -> needs the allowlist.
    assert _is_trusted_diffusion_repo("black-forest-labs/FLUX.1-Krea-dev")


def test_flux1_krea_dev_generation_defaults():
    # Model-card recipe: 28 steps at guidance 4.5. The generic "krea" key (Krea-2-Turbo's
    # 8-step no-CFG shape) must NOT swallow it, and the krea-2 defaults must stay intact.
    assert default_generation_params("black-forest-labs/FLUX.1-Krea-dev") == (28, 4.5)
    assert default_generation_params("QuantStack/FLUX.1-Krea-dev-GGUF") == (28, 4.5)
    assert default_generation_params("krea/Krea-2-Turbo") == (8, 0.0)
    assert default_generation_params("krea/Krea-2-Raw") == (52, 3.5)


# ── lumina-2 family ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "repo_id",
    [
        "Alpha-VLLM/Lumina-Image-2.0",
        # A same-arch finetune must group here via the lumina-image-2.0 token.
        "neta-art/NetaYume-Lumina-Image-2.0",
    ],
)
def test_detect_family_lumina2_repos(repo_id):
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "lumina-2"
    assert fam.pipeline_class == "Lumina2Pipeline"
    assert fam.transformer_class == "Lumina2Transformer2DModel"
    assert fam.base_repo == "Alpha-VLLM/Lumina-Image-2.0"
    # Published bf16-only upstream; the fp16 fallback stays off.
    assert fam.fp16_incompatible is True


def test_detect_family_lumina2_override_and_next_rejected():
    assert detect_family("x", override = "lumina-2").name == "lumina-2"
    assert detect_family("x", override = "lumina2").name == "lumina-2"
    # Lumina-Next is a DIFFERENT arch (LuminaText2ImgPipeline): it must stay unknown
    # instead of resolving here and crashing mid-load.
    assert detect_family("Alpha-VLLM/Lumina-Next-SFT-diffusers") is None


def test_lumina2_is_trusted_non_gguf():
    # The official pipeline loads via from_pretrained -> needs the allowlist.
    assert _is_trusted_diffusion_repo("Alpha-VLLM/Lumina-Image-2.0")
    assert not _is_trusted_diffusion_repo("Alpha-VLLM/some-future-repo")


def test_lumina2_generation_defaults():
    # Model-card recipe: 50 steps at guidance 4.0 (cfg_trunc_ratio is added by the
    # backend generate call itself, not the defaults table).
    assert default_generation_params("Alpha-VLLM/Lumina-Image-2.0") == (50, 4.0)


def test_lumina2_prequant_wiring():
    # Hosted int8/fp8 checkpoints (gate-validated) serve the family default base.
    from core.inference.diffusion_families import family_prequant_repo

    fam = detect_family("Alpha-VLLM/Lumina-Image-2.0")
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/Lumina-Image-2.0-FP8"


def test_lumina2_bf16_component_table_present():
    fam = detect_family("Alpha-VLLM/Lumina-Image-2.0")
    sizes = family_bf16_components_gb(fam)
    assert sizes is not None
    transformer_gb, encoders_gb, vae_gb = sizes
    # 2.6B DiT + Gemma2-2B, both fp32 on disk -> ~5.2 GB each bf16-resident.
    assert 4.0 <= transformer_gb <= 7.0
    assert 4.0 <= encoders_gb <= 7.0
    assert vae_gb <= 0.5


def test_ideogram4_generation_defaults():
    # Model-card settings: 48 steps, guidance 7 (the backend keeps the pipeline's
    # recommended tapered schedule when the request matches exactly).
    assert default_generation_params("ideogram-ai/ideogram-4-fp8") == (48, 7.0)


def test_ideogram4_bf16_reservation_table_present():
    # The memory planner reserves this bf16 footprint for a narrow (fp8) ideogram-4 base even
    # when the blob-cache estimate is absent (empty cache / a best-effort download probe that
    # swallowed a transient HF error), so the ~54 GB pipeline never plans a resident placement
    # it cannot fit. If this constant table ever went None, that fp8 OOM safeguard would
    # silently disable, so pin that it is present and sums to the expected ~54 GB.
    fam = detect_family("ideogram-ai/ideogram-4-fp8")
    table = family_bf16_components_gb(fam, fam.base_repo)
    assert table is not None
    assert sum(table) > 50.0  # transformer (37.2) + bf16 text encoder (16.3) + VAE (0.2)


def test_ideogram4_memory_table_counts_both_dits():
    fam = detect_family("ideogram-ai/ideogram-4-fp8")
    components = family_bf16_components_gb(fam)
    assert components is not None
    transformer_gb, text_encoders_gb, _vae_gb = components
    # Two ~9.3B DiTs (conditional + unconditional) at bf16: well above one DiT's
    # ~18.6 GB. A single-DiT entry here would let auto planning under-reserve and OOM.
    assert transformer_gb > 30.0
    assert text_encoders_gb > 5.0


# ── structured exclusions ────────────────────────────────────────────────────
def test_hunyuanimage_is_excluded_with_reason():
    reason = excluded_model_reason("tencent/HunyuanImage-3.0")
    assert reason is not None and "diffusers" in reason
    # Not detectable as any family: the exclusion reason is the load error surface.
    assert detect_family("tencent/HunyuanImage-3.0") is None


def test_excluded_model_reason_none_for_supported_and_unknown():
    assert excluded_model_reason("unsloth/Z-Image-Turbo-GGUF") is None
    assert excluded_model_reason("someorg/some-model") is None


def test_validate_load_request_surfaces_exclusion_reason():
    from core.inference.diffusion import DiffusionBackend
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "trust_remote_code"):
        backend.validate_load_request("tencent/HunyuanImage-3.0")


# ── curated krea LoRA catalog ────────────────────────────────────────────────
def test_curated_krea2_loras_present_and_well_formed():
    krea = [e for e in _CURATED if e.repo_id and e.repo_id.startswith("krea/Krea-2-LoRA-")]
    assert len(krea) == 9
    for entry in krea:
        assert entry.source == "hub" and entry.fmt == "safetensors"
        assert entry.families == ("krea-2",)
        # Every official style repo carries a single "{style}.safetensors" at the root.
        style = entry.repo_id.split("Krea-2-LoRA-")[-1]
        assert entry.weight_name == f"{style}.safetensors"


def test_list_loras_family_filter_gates_krea_entries():
    krea_ids = {e.id for e in _CURATED if e.families == ("krea-2",)}
    assert krea_ids  # curated entries exist
    listed_for_krea = {e.id for e in list_loras(family = "krea-2")}
    assert krea_ids <= listed_for_krea
    listed_for_flux = {e.id for e in list_loras(family = "flux.1")}
    assert not (krea_ids & listed_for_flux)


# ── ideogram-4 fp8 transformer remap ─────────────────────────────────────────
def test_convert_fp8_state_dict_dequantizes_and_splits_qkv():
    # The vendor fp8 transformer stores fused attention.qkv (Q/K/V rows stacked) +
    # attention.o, each with a per-output-channel weight_scale; diffusers expects split
    # to_q/to_k/to_v/to_out.0 with the scale already applied. The converter must undo
    # both, or every attention weight loads wrong (garbage) and on meta (a load crash).
    torch = pytest.importorskip("torch")

    from core.inference.diffusion_ideogram4 import _convert_fp8_state_dict

    hidden = 4  # tiny stand-in for attention_head_dim * num_attention_heads
    # Reference (real) weights, then a fake per-channel fp8 encoding: value / scale.
    q = torch.randn(hidden, hidden)
    k = torch.randn(hidden, hidden)
    v = torch.randn(hidden, hidden)
    o = torch.randn(hidden, hidden)
    ff = torch.randn(hidden, hidden)
    fused = torch.cat([q, k, v], dim = 0)  # [3 * hidden, hidden]
    qkv_scale = torch.rand(3 * hidden) + 0.5
    o_scale = torch.rand(hidden) + 0.5
    ff_scale = torch.rand(hidden) + 0.5
    norm = torch.randn(hidden)  # dense (unscaled) weight passes through
    raw = {
        "layers.0.attention.qkv.weight": fused / qkv_scale[:, None],
        "layers.0.attention.qkv.weight_scale": qkv_scale,
        "layers.0.attention.o.weight": o / o_scale[:, None],
        "layers.0.attention.o.weight_scale": o_scale,
        "layers.0.feed_forward.w1.weight": ff / ff_scale[:, None],
        "layers.0.feed_forward.w1.weight_scale": ff_scale,
        "layers.0.attention_norm1.weight": norm,
    }
    out = _convert_fp8_state_dict(raw, hidden, torch.bfloat16)

    # Every converted tensor is cast to the requested compute dtype (the load_state_dict
    # copy would silently up/down-cast otherwise).
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    # Re-run in float32 for the exact value checks below (bf16 loses precision).
    out = _convert_fp8_state_dict(raw, hidden, torch.float32)

    # No scale keys leak through; fused/renamed keys are gone.
    assert not any(key.endswith("_scale") for key in out)
    assert "layers.0.attention.qkv.weight" not in out
    assert "layers.0.attention.o.weight" not in out
    # QKV split back to the reference weights in Q/K/V order.
    torch.testing.assert_close(out["layers.0.attention.to_q.weight"], q)
    torch.testing.assert_close(out["layers.0.attention.to_k.weight"], k)
    torch.testing.assert_close(out["layers.0.attention.to_v.weight"], v)
    # o renamed to to_out.0 with the scale applied.
    torch.testing.assert_close(out["layers.0.attention.to_out.0.weight"], o)
    # A non-attention fp8 weight keeps its name, scale applied.
    torch.testing.assert_close(out["layers.0.feed_forward.w1.weight"], ff)
    # A dense weight passes through unchanged.
    torch.testing.assert_close(out["layers.0.attention_norm1.weight"], norm)


def test_ideogram4_repo_is_fp8_detects_local_layout(tmp_path):
    # A local mirror of the fp8 base never string-matches base_repo, so memory planning
    # relies on this shard-header probe to reserve the bf16 footprint. The fp8 layout is
    # marked by a companion ``*.weight_scale``; the bnb-4bit (nf4) mirror carries none and
    # must read as not-fp8 so it stays (correctly) planned against its compressed bytes.
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")

    from core.inference.diffusion_ideogram4 import ideogram4_repo_is_fp8

    fp8 = tmp_path / "fp8"
    (fp8 / "transformer").mkdir(parents = True)
    st.save_file(
        {
            "layers.0.attention.o.weight": torch.zeros(2, 2),
            "layers.0.attention.o.weight_scale": torch.ones(2),
        },
        str(fp8 / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    assert ideogram4_repo_is_fp8(str(fp8)) is True

    nf4 = tmp_path / "nf4"
    (nf4 / "transformer").mkdir(parents = True)
    st.save_file(
        {"layers.0.attention.to_q.weight": torch.zeros(2, 2)},
        str(nf4 / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    assert ideogram4_repo_is_fp8(str(nf4)) is False

    # A directory with no transformer shards at all resolves to False, not an error.
    assert ideogram4_repo_is_fp8(str(tmp_path / "missing")) is False


def test_create_causal_mask_patch_is_self_disabling_and_idempotent():
    # The patch adapts the pipeline's inputs_embeds kwarg to the installed transformers
    # create_causal_mask signature; on a matching signature it must forward unchanged,
    # and a second apply must not double-wrap.
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    import core.inference.diffusion_ideogram4 as ig4
    from diffusers.pipelines.ideogram4 import pipeline_ideogram4 as pipe_mod

    original = pipe_mod.create_causal_mask
    try:
        ig4._CAUSAL_MASK_PATCHED = False
        ig4._patch_create_causal_mask()
        wrapped = pipe_mod.create_causal_mask
        assert wrapped is not original  # the patch installed a wrapper
        ig4._patch_create_causal_mask()  # idempotent: no re-wrap
        assert pipe_mod.create_causal_mask is wrapped
    finally:
        pipe_mod.create_causal_mask = original
        ig4._CAUSAL_MASK_PATCHED = False
