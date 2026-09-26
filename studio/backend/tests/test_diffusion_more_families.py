# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 family registration, the HunyuanImage structured exclusion, and the
curated krea/Krea-2-LoRA-* catalog entries. Pure-module tests: no torch, no network."""

import pytest

from core.inference.diffusion import _is_trusted_diffusion_repo
from core.inference.diffusion_auto_policy import family_bf16_components_gb
from core.inference.diffusion_families import (
    IDEOGRAM4_FAMILY_NAME,
    assert_flux2_gguf_matches_base,
    default_generation_params,
    detect_family,
    excluded_model_reason,
    family_sd_cpp_supported,
    sd_cpp_companion_only_repo_ids,
    sd_cpp_text_encoders_for,
)
from core.inference.sd_cpp_args import text_encoder_flags_for_family
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
    # The three official vendor pipelines load via from_pretrained, gated to the unsloth org + the explicit allowlist.
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
    # Krea's FLUX.1-dev finetune keeps the exact dev layout, so it resolves to flux.1, never krea-2 (a different arch).
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "flux.1"
    assert fam.pipeline_class == "FluxPipeline"


def test_flux1_krea_dev_is_trusted_non_gguf():
    # The gated official pipeline loads via from_pretrained, so it needs the allowlist.
    assert _is_trusted_diffusion_repo("black-forest-labs/FLUX.1-Krea-dev")


def test_flux1_krea_dev_generation_defaults():
    # Model-card recipe: 28 steps at guidance 4.5. The generic "krea" key (Turbo's 8-step no-CFG shape) must NOT swallow it, and the krea-2 defaults must stay intact.
    assert default_generation_params("black-forest-labs/FLUX.1-Krea-dev") == (28, 4.5)
    assert default_generation_params("QuantStack/FLUX.1-Krea-dev-GGUF") == (28, 4.5)
    assert default_generation_params("krea/Krea-2-Turbo") == (8, 0.0)
    assert default_generation_params("krea/Krea-2-Raw") == (52, 3.5)


def test_flux_dev_and_krea_do_not_inherit_the_schnell_nvfp4_checkpoint():
    # The NVFP4 artifact is schnell-only: dev and Krea-dev inheriting it failed validation with no dense fallback.
    from core.inference.diffusion_families import family_prequant_repo

    fam = detect_family("black-forest-labs/FLUX.1-schnell")
    assert fam is not None and fam.name == "flux.1"
    assert (
        family_prequant_repo(fam, "nvfp4", base_repo = "black-forest-labs/FLUX.1-schnell")
        == "unsloth/FLUX.1-schnell-NVFP4"
    )
    for base, fp8_repo in (
        ("black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev-FP8"),
        ("black-forest-labs/FLUX.1-Krea-dev", "unsloth/FLUX.1-Krea-dev-FP8"),
    ):
        assert family_prequant_repo(fam, "nvfp4", base_repo = base) is None
        assert family_prequant_repo(fam, "fp8", base_repo = base) == fp8_repo
        assert family_prequant_repo(fam, "int8", base_repo = base) == fp8_repo


def test_flux2_klein_generation_defaults_distinguish_base_from_distilled():
    for size in ("4B", "9B"):
        assert default_generation_params(f"unsloth/FLUX.2-klein-base-{size}") == (50, 4.0)
        assert default_generation_params(f"unsloth/FLUX.2-klein-{size}") == (4, 1.0)


# ── z-image: the undistilled base ────────────────────────────────────────────
def test_zimage_base_is_trusted_so_the_gguf_keeps_its_companion_base():
    # unsloth/Z-Image-GGUF carries base_model: Tongyi-MAI/Z-Image, and _resolve_base_repo drops a
    # tag that fails this gate. While it did, that pick fell back to the Turbo companions and
    # denoised on their shift 3.0 scheduler instead of the base's 6.0.
    assert _is_trusted_diffusion_repo("Tongyi-MAI/Z-Image")
    assert _is_trusted_diffusion_repo("Tongyi-MAI/Z-Image-Turbo")
    assert not _is_trusted_diffusion_repo("someone/Z-Image-finetune")


def test_zimage_base_has_no_hosted_prequant_to_inherit():
    # Both hosted checkpoints are baked from the Turbo transformer. Falling back to them for the
    # undistilled base made planning treat an unrelated artifact as usable: auto declined the dense
    # path when it was uncached, and an explicit int8/fp8 request downloaded it, hit the
    # base_model_id refusal, then had no dense shards staged to fall back to.
    from core.inference.diffusion_families import family_prequant_repo

    fam = detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert fam is not None and fam.name == "z-image"
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/Z-Image-Turbo-FP8"
        assert (
            family_prequant_repo(fam, scheme, base_repo = "Tongyi-MAI/Z-Image-Turbo")
            == "unsloth/Z-Image-Turbo-FP8"
        )
        assert family_prequant_repo(fam, scheme, base_repo = "Tongyi-MAI/Z-Image") is None
        # However the id was typed, and through the mirror the loader actually fetches.
        assert family_prequant_repo(fam, scheme, base_repo = "  tongyi-mai/Z-IMAGE ") is None


def test_prequant_exclusion_does_not_break_a_family_type_that_lacks_the_field():
    # family_prequant_repo is shared with the VIDEO loader, whose VideoFamily has no
    # prequant_excluded_bases. A plain attribute read raises AttributeError here, and the only
    # caller wraps this in a bare except that turns any raise into "no hosted checkpoint", so
    # every video family would quietly drop to the dense path whenever a base_repo is passed.
    from core.inference.diffusion_families import family_prequant_repo
    from core.inference.diffusion_prequant import resolve_prequant_source
    from core.inference.video_families import detect_video_family

    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert h3 is not None
    assert not hasattr(h3, "prequant_excluded_bases")
    for scheme in ("int8", "fp8"):
        # The base_repo argument is the trigger: an empty base short-circuits before the read.
        assert (
            family_prequant_repo(h3, scheme, base_repo = "MiniMaxAI/MiniMax-H3")
            == "unsloth/MiniMax-H3-FP8"
        )
        source = resolve_prequant_source(h3, scheme, base_repo = "MiniMaxAI/MiniMax-H3")
        assert source is not None and source.location == "unsloth/MiniMax-H3-FP8"


def test_zimage_base_generation_defaults_are_not_the_distilled_recipe():
    # The base is undistilled: 20 steps at guidance 4. The more specific "z-image-turbo" key sits
    # ahead of "z-image", so the 9-step CFG-free Turbo recipe must not swallow it.
    assert default_generation_params("Tongyi-MAI/Z-Image") == (20, 4.0)
    assert default_generation_params("unsloth/Z-Image-GGUF") == (20, 4.0)
    assert default_generation_params("Tongyi-MAI/Z-Image-Turbo") == (9, 0.0)
    assert default_generation_params("unsloth/Z-Image-Turbo-GGUF") == (9, 0.0)


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
    # Lumina-Next is a DIFFERENT arch (LuminaText2ImgPipeline): it must stay unknown, not resolve here and crash mid-load.
    assert detect_family("Alpha-VLLM/Lumina-Next-SFT-diffusers") is None


def test_lumina2_is_trusted_non_gguf():
    # The official pipeline loads via from_pretrained -> needs the allowlist.
    assert _is_trusted_diffusion_repo("Alpha-VLLM/Lumina-Image-2.0")
    assert not _is_trusted_diffusion_repo("Alpha-VLLM/some-future-repo")


def test_lumina2_generation_defaults():
    # Model-card recipe: 50 steps at guidance 4.0 (cfg_trunc_ratio is added by the backend generate call itself).
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


# ── hunyuanimage-2.1 family ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "repo_id",
    [
        "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        "QuantStack/HunyuanImage-2.1-GGUF",
        # A local GGUF pick whose family keyword lives in the filename (QuantStack drops the dash, covered by the hunyuanimage2.1 alias).
        "QuantStack/HunyuanImage-2.1-GGUF/HunyuanImage2.1-Q4_K_M.gguf",
    ],
)
def test_detect_family_hunyuanimage21_repos(repo_id):
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "hunyuanimage-2.1"
    assert fam.pipeline_class == "HunyuanImagePipeline"
    assert fam.transformer_class == "HunyuanImageTransformer2DModel"
    assert fam.base_repo == "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    # The call's guidance knob is distilled_guidance_scale; there is no guidance_scale kwarg.
    assert fam.cfg_kwarg == "distilled_guidance_scale"
    # Published bf16-only upstream; the fp16 fallback stays off.
    assert fam.fp16_incompatible is True


def test_detect_family_hunyuanimage21_override_and_30_still_excluded():
    assert detect_family("x", override = "hunyuanimage-2.1").name == "hunyuanimage-2.1"
    assert detect_family("x", override = "hunyuanimage2.1").name == "hunyuanimage-2.1"
    # HunyuanImage-3.0 has no diffusers pipeline, so its structured exclusion must survive the 2.1 family.
    assert detect_family("tencent/HunyuanImage-3.0") is None
    assert excluded_model_reason("tencent/HunyuanImage-3.0") is not None
    assert excluded_model_reason("hunyuanvideo-community/HunyuanImage-2.1-Diffusers") is None


def test_hunyuanimage21_is_trusted_non_gguf():
    # The mirror pipeline loads via from_pretrained -> needs the allowlist.
    assert _is_trusted_diffusion_repo("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
    assert not _is_trusted_diffusion_repo("hunyuanvideo-community/some-future-repo")


def test_hunyuanimage21_generation_defaults():
    # Card recipe: 50 steps; guidance feeds distilled_guidance_scale, while CFG runs inside the repo's guider components.
    assert default_generation_params("hunyuanvideo-community/HunyuanImage-2.1-Diffusers") == (
        50,
        3.25,
    )


def test_hunyuanimage21_prequant_wiring():
    # Hosted int8/fp8 checkpoints, verified bit-identical to on-the-fly quantize.
    from core.inference.diffusion_families import family_prequant_repo
    fam = detect_family("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/HunyuanImage-2.1-FP8"


def test_hunyuanimage21_bf16_component_table_present():
    fam = detect_family("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
    sizes = family_bf16_components_gb(fam)
    assert sizes is not None
    transformer_gb, encoders_gb, vae_gb = sizes
    # 17B DiT (32.5 GB bf16 on disk) + Qwen2.5-VL 15.5 GB + ByT5 0.8 GB.
    assert 30.0 <= transformer_gb <= 35.0
    assert 15.0 <= encoders_gb <= 18.0
    assert vae_gb <= 1.0


# ── hidream-i1 family ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "repo_id",
    [
        "HiDream-ai/HiDream-I1-Full",
        "HiDream-ai/HiDream-I1-Dev",
        "HiDream-ai/HiDream-I1-Fast",
    ],
)
def test_detect_family_hidream_repos(repo_id):
    # One family covers all three variants (same 17B MoE arch + 4-TE stack).
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == "hidream-i1"
    assert fam.pipeline_class == "HiDreamImagePipeline"
    assert fam.transformer_class == "HiDreamImageTransformer2DModel"
    assert fam.base_repo == "HiDream-ai/HiDream-I1-Full"
    # Published bf16-only upstream; the fp16 fallback stays off.
    assert fam.fp16_incompatible is True


def test_hidream_override_and_trust():
    assert detect_family("x", override = "hidream-i1").name == "hidream-i1"
    assert detect_family("x", override = "hidream").name == "hidream-i1"
    # The three official repos load via from_pretrained so they are allowlisted; the Llama TE4 rides the trusted unsloth mirror.
    for rid in (
        "HiDream-ai/HiDream-I1-Full",
        "HiDream-ai/HiDream-I1-Dev",
        "HiDream-ai/HiDream-I1-Fast",
    ):
        assert _is_trusted_diffusion_repo(rid)
    assert not _is_trusted_diffusion_repo("HiDream-ai/some-future-repo")
    assert _is_trusted_diffusion_repo("unsloth/Meta-Llama-3.1-8B-Instruct")


def test_hidream_generation_defaults():
    # Upstream inference.py: Full 50 steps / guidance 5; Dev and Fast are distilled and guidance-free at 28 / 16 steps. The specific keys must beat the generic "hidream".
    assert default_generation_params("HiDream-ai/HiDream-I1-Full") == (50, 5.0)
    assert default_generation_params("HiDream-ai/HiDream-I1-Dev") == (28, 0.0)
    assert default_generation_params("HiDream-ai/HiDream-I1-Fast") == (16, 0.0)


def test_hidream_bf16_component_table_present():
    fam = detect_family("HiDream-ai/HiDream-I1-Full")
    sizes = family_bf16_components_gb(fam)
    assert sizes is not None
    transformer_gb, encoders_gb, vae_gb = sizes
    # 17B MoE DiT 34.2 GB; TEs are CLIP-L 0.5 + CLIP-G 2.8 + T5-XXL 9.5 plus the ~16 GB Llama TE4 mirror, so ~28.8 GB.
    assert 32.0 <= transformer_gb <= 37.0
    assert 26.0 <= encoders_gb <= 32.0
    assert vae_gb <= 0.5


def test_ideogram4_generation_defaults():
    # Model-card settings: 48 steps, guidance 7 (an exact match keeps the pipeline's recommended tapered schedule).
    assert default_generation_params("ideogram-ai/ideogram-4-fp8") == (48, 7.0)


def test_ideogram4_bf16_reservation_table_present():
    # The memory planner reserves this bf16 footprint for a narrow (fp8) ideogram-4 base even with no blob-cache
    # estimate, so the ~54 GB pipeline never plans a resident placement it cannot fit. Pin its presence and sum.
    fam = detect_family("ideogram-ai/ideogram-4-fp8")
    table = family_bf16_components_gb(fam, fam.base_repo)
    assert table is not None
    assert sum(table) > 50.0  # transformer (37.2) + bf16 text encoder (16.3) + VAE (0.2)


def test_ideogram4_memory_table_counts_both_dits():
    fam = detect_family("ideogram-ai/ideogram-4-fp8")
    components = family_bf16_components_gb(fam)
    assert components is not None
    transformer_gb, text_encoders_gb, _vae_gb = components
    # Two ~9.3B bf16 DiTs, well above one DiT's ~18.6 GB: a single-DiT entry would let auto planning under-reserve and OOM.
    assert transformer_gb > 30.0
    assert text_encoders_gb > 5.0


def test_hidream_prequant_wiring():
    # Hosted int8/fp8 checkpoints (28/28 per-case gate pairs each; int8 bit-identical to on-the-fly) serve the family default base.
    from core.inference.diffusion_families import family_prequant_repo
    fam = detect_family("HiDream-ai/HiDream-I1-Full")
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/HiDream-I1-Full-FP8"


def test_hidream_distilled_variants_have_no_hosted_prequant_to_inherit():
    # Dev and Fast are distillations of Full, so the hosted Full checkpoint is baked from other
    # weights. Inheriting it made a Dev / Fast pick plan the Full artifact, drop its own shards,
    # download several GB and only then hit the base_model_id refusal.
    from core.inference.diffusion_families import family_prequant_repo
    for repo_id in ("HiDream-ai/HiDream-I1-Dev", "HiDream-ai/HiDream-I1-Fast"):
        fam = detect_family(repo_id)
        for scheme in ("int8", "fp8"):
            assert family_prequant_repo(fam, scheme, base_repo = repo_id) is None
            # However the id was typed, and through the mirror the loader actually fetches.
            assert family_prequant_repo(fam, scheme, base_repo = f"  {repo_id.upper()} ") is None
            assert (
                family_prequant_repo(
                    fam, scheme, base_repo = repo_id.replace("HiDream-ai", "unsloth")
                )
                is None
            )


def test_qwen_image_2512_routes_to_its_own_hosted_prequant():
    # 2512 is a different checkpoint with its own baked artifacts. Falling back to the Qwen-Image ones
    # made a 2512 pick plan an artifact base_model_id refuses, after its shards had been dropped.
    from core.inference.diffusion_families import family_prequant_repo

    fam = detect_family("Qwen/Qwen-Image-2512")
    assert fam is not None and fam.name == "qwen-image"
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/Qwen-Image-FP8"
        assert (
            family_prequant_repo(fam, scheme, base_repo = "Qwen/Qwen-Image")
            == "unsloth/Qwen-Image-FP8"
        )
        for base_repo in (
            "Qwen/Qwen-Image-2512",
            "unsloth/Qwen-Image-2512",
            " QWEN/QWEN-IMAGE-2512 ",
        ):
            assert (
                family_prequant_repo(fam, scheme, base_repo = base_repo)
                == "unsloth/Qwen-Image-2512-FP8"
            )


def test_qwen_image_2512_prequant_filenames_match_its_repo():
    # The names derive from the repo name, so the variant repo must be asked for <Model>-<SCHEME>
    # in both containers. The .pt is what that repo actually serves today and is asserted to stay
    # in the chain: preferring safetensors is only allowed to ADD a name in front of it, never to
    # replace it, or every checkpoint already published would stop resolving.
    from core.inference.diffusion_prequant import candidate_filenames_of, resolve_prequant_source
    fam = detect_family("Qwen/Qwen-Image-2512")
    for scheme, safetensors_name, pickle_name in (
        ("int8", "Qwen-Image-2512-INT8.safetensors", "Qwen-Image-2512-INT8.pt"),
        ("fp8", "Qwen-Image-2512-FP8.safetensors", "Qwen-Image-2512-FP8.pt"),
    ):
        source = resolve_prequant_source(fam, scheme, base_repo = "Qwen/Qwen-Image-2512")
        assert source is not None
        assert source.location == "unsloth/Qwen-Image-2512-FP8"
        names = list(candidate_filenames_of(source))
        assert names[0] == safetensors_name, names
        assert pickle_name in names[1:], names
        # And the legacy repo-agnostic spelling stays last, for a repo predating the model-named one.
        assert names[-1] == f"transformer_{scheme}.pt", names


def test_hidream_quant_schemes_not_denied_and_no_extra_excludes():
    # Measured on a B200: int8 and fp8 both engage and render cleanly, including 2-3 token prompts on int8. The routed
    # MoE expert Linears only see the concatenated image+text stream (M >> 16), so torch._int_mm's minimum never binds.
    from core.inference.diffusion_transformer_quant import (
        _FAMILY_SCHEME_DENY,
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )

    assert "hidream-i1" not in _FAMILY_SCHEME_DENY
    assert exclude_tokens_for_scheme("int8", "hidream-i1") == _INT8_EXCLUDE_NAME_TOKENS
    assert exclude_tokens_for_scheme("fp8", "hidream-i1") == ()


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
    # The vendor fp8 transformer stores fused attention.qkv + attention.o with per-output-channel weight_scale, while
    # diffusers wants split to_q/to_k/to_v/to_out.0 with the scale applied. Undo both or every attention weight loads wrong.
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

    # Every converted tensor is cast to the requested compute dtype (the load_state_dict copy would silently re-cast).
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
    # A local mirror of the fp8 base never string-matches base_repo, so memory planning relies on this shard-header
    # probe. The fp8 layout is marked by a companion ``*.weight_scale``; the bnb-4bit mirror carries none.
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
    # The patch adapts the pipeline's inputs_embeds kwarg to the installed transformers create_causal_mask signature; a match forwards unchanged and a second apply must not double-wrap.
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


# ── FLUX.2 klein size resolution ─────────────────────────────────────────────
def test_flux2_klein_9b_resolves_its_own_base_and_text_encoder():
    """A klein-9B GGUF must not inherit the family's 4B default.

    One family covers both klein sizes and defaults to 4B, relying on the base_model card tag for
    the real base. That tag is only honoured for repos on the trust allowlist, so omitting the 9B
    entries silently loaded a 9B checkpoint against a 4B config (inner_dim 4096 vs 3072), which
    surfaces as a bare shape mismatch inside the GGUF quantizer. klein-BASE-9B is 9B too, and the
    text-encoder rule matched the literal "klein-9b", so it was handed the 4B encoder.
    """
    for repo in (
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
    ):
        assert _is_trusted_diffusion_repo(repo), repo

    for repo_id, want_te in (
        ("unsloth/FLUX.2-klein-9B-GGUF", "qwen_3_8b"),
        ("unsloth/FLUX.2-klein-base-9B-GGUF", "qwen_3_8b"),
        ("unsloth/FLUX.2-klein-4B-GGUF", "qwen_3_4b"),
        ("unsloth/FLUX.2-klein-base-4B-GGUF", "qwen_3_4b"),
    ):
        encoders = sd_cpp_text_encoders_for(detect_family(repo_id), repo_id, None)
        assert want_te in encoders[0][1], (repo_id, encoders)


def test_flux2_gguf_base_mismatch_check_fails_open(tmp_path):
    """The size check never turns a working load into a failing one."""
    fam = detect_family("unsloth/FLUX.2-klein-9B-GGUF")
    empty = tmp_path / "not-a-gguf.gguf"
    empty.write_bytes(b"")
    # No path, an unreadable file, an unmapped base, and a non-FLUX.2 family are all pass-through.
    assert_flux2_gguf_matches_base(fam, "black-forest-labs/FLUX.2-klein-4B", None)
    assert_flux2_gguf_matches_base(fam, "black-forest-labs/FLUX.2-klein-4B", empty)
    assert_flux2_gguf_matches_base(fam, "unsloth/Something-FP8", empty)
    assert_flux2_gguf_matches_base(
        detect_family("unsloth/FLUX.1-dev-GGUF"), "black-forest-labs/FLUX.2-klein-4B", empty
    )


def test_qwen_image_21_is_reachable_end_to_end_not_just_detectable():
    """A family whose base repo is not trusted is not a family at all.

    Detection resolving is the easy half and was never the problem: the load is refused several
    layers later, by a check that reads a different list, so the entry shipped looking complete and
    every non-GGUF pick of it died with "restricted to unsloth/* repos". This asserts the whole
    chain the picker actually walks, which is why it is one test rather than four.
    """
    from core.inference.diffusion_families import (
        _PIPELINE_MIN_DIFFUSERS,
        detect_family,
        detect_family_by_pipeline_class,
    )

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    # Not swallowed by the generic family, and not swallowing it either.
    assert detect_family("Qwen/Qwen-Image").name == "qwen-image"
    assert detect_family("Qwen/Qwen-Image-2512").name == "qwen-image"
    for alias in ("qwen_image_21", "qwenimage21", "qwen-image-21"):
        assert detect_family("", override = alias) is fam, alias
    # The class really is what the published model_index.json names.
    assert detect_family_by_pipeline_class("QwenImage21Pipeline") is fam
    assert fam.pipeline_class in _PIPELINE_MIN_DIFFUSERS

    # The gate that made the entry inert. _is_trusted_diffusion_repo is asked of the base repo by
    # validate_load_request BEFORE anything is built, so a family base missing from that list is
    # unloadable however correct the rest of the entry is.
    assert _is_trusted_diffusion_repo(fam.base_repo), (
        f"{fam.base_repo} is the family's own base and is not in _TRUSTED_NON_GGUF_REPOS, so "
        "every non-GGUF pick of this family is refused before the pipeline is built"
    )


def test_every_image_family_base_repo_is_loadable():
    """The general form of the above, so the next family cannot ship inert the same way."""
    from core.inference.diffusion_families import _FAMILIES

    unreachable = [
        f.name for f in _FAMILIES if f.base_repo and not _is_trusted_diffusion_repo(f.base_repo)
    ]
    assert (
        not unreachable
    ), f"these families declare a base repo that validate_load_request refuses: {unreachable}"


def test_qwen_image_21_gguf_reaches_sd_cpp_with_its_own_vae_and_a_qwen3vl_encoder():
    """The no-GPU route for unsloth/Qwen-Image-2.1-GGUF.

    Every assertion here is a way the route was observed to fail quietly rather than loudly:
    a family without both sd.cpp assets silently falls back to diffusers, the qwen-image VAE
    decodes 2.1 latents to noise instead of erroring, and a fixed flow shift overrides the
    resolution-dependent schedule upstream picks for this architecture.
    """
    fam = detect_family("unsloth/Qwen-Image-2.1-GGUF")
    assert fam is not None and fam.name == "qwen-image-2.1"
    assert family_sd_cpp_supported(fam)

    assert fam.sd_cpp_vae == (
        "unsloth/Qwen-Image-2.1-FP8",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    )
    # Not the qwen-image VAE: a different class for a different latent space, which decodes to
    # noise rather than raising if it is ever substituted here.
    assert "2.1" in fam.sd_cpp_vae[1]

    encoders = sd_cpp_text_encoders_for(fam, "unsloth/Qwen-Image-2.1-GGUF", None)
    # The encoder, then the vision projector native editing reads through --llm_vision.
    assert len(encoders) == 2
    assert encoders[1] == ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision")
    repo, filename, kind = encoders[0]
    assert repo == "unsloth/Qwen3-VL-8B-Instruct-GGUF"
    # Which rung is the family's call, pinned by exact name in
    # test_diffusion_compat_preflight.py::test_qwen_image_2_1_takes_the_dynamic_4bit_text_encoder.
    # Restating it here broke when #11542 moved it to UD-Q4_K_XL. What this route needs is that
    # the declared file is what reaches sd.cpp, and that it stays a 4-bit GGUF: the CPU RAM win
    # is the reason the no-GPU route exists (bf16 is 16.4 GB).
    assert filename == fam.sd_cpp_text_encoders[0][1]
    assert filename.endswith(".gguf") and "Q4_K" in filename, filename
    assert kind == "llm"
    assert text_encoder_flags_for_family(fam.name) == ("--llm",)

    assert fam.sd_cpp_sampling_method == "euler"
    assert fam.sd_cpp_flow_shift is None

    # The encoder repo is fetch-only and must not be offered as a loadable model. The VAE repo is
    # the base itself, so it must NOT be classified that way.
    companions = sd_cpp_companion_only_repo_ids()
    assert "unsloth/qwen3-vl-8b-instruct-gguf" in companions
    # The VAE ships inside a repo that is itself loadable, so it must NOT be classified fetch-only.
    # Putting it in the GGUF repo instead WOULD be: that repo appears in no family field, so
    # companions-minus-loadable would mark the home of every denoiser as a companion and hide it.
    assert "unsloth/qwen-image-2.1-fp8" not in companions
    assert "qwen/qwen-image-2.1" not in companions


def test_the_pinned_prebuilt_is_one_that_can_load_qwen_image_21():
    """The route is only real if the binary the installer pins understands the architecture.

    The tag STRING cannot answer this: every mirror build resolves to the same master-813 base, so
    the August build and the current one are indistinguishable by name. What is asserted here is the
    pin itself, against the release verified to render this family (26.9s at 1024 on one B200,
    Q4_K_M denoiser, bf16 VAE, Q4_K_M Qwen3-VL encoder). Bump both together or not at all.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "install_sd_cpp_prebuilt.py"
    spec = importlib.util.spec_from_file_location("install_sd_cpp_prebuilt_pin", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_TAG == "master-813-bfbef5b-u1d02858", (
        "the pinned prebuilt must be one built from a tree carrying Qwen-Image-2.1; "
        f"{module.DEFAULT_TAG} is not"
    )


def test_a_minimum_that_has_not_shipped_does_not_prescribe_an_impossible_upgrade():
    """``pip install -U 'diffusers>=0.41.0'`` has no candidate while 0.41.0 is unreleased, so the
    refusal has to name the pinned main build Studio actually installs for this class.

    And it has to name a remedy that WORKS for the cause that produces this refusal. Measured on a
    host whose git exits non-zero: the install keeps diffusers 0.40.0, and the old text sent the
    reader to `pip install -r diffusers-main.txt`, which resolves the same git+https requirement
    and fails identically. The quoted zip URL needs no git, so it is the one line here that has to
    stay true, hence the check that it names the commit the pin file actually carries."""
    import pathlib
    import re as _re

    from core.inference.diffusion_families import (
        _PIPELINE_MIN_DIFFUSERS,
        _UNRELEASED_MIN_DIFFUSERS,
        _too_old_message,
    )

    message = _too_old_message("QwenImage21Pipeline", "qwen-image-2.1", "0.40.0")
    assert "pip install -U 'diffusers>=0.41.0'" not in message
    assert "has not been released yet" in message
    assert "git --version" in message, "the likely cause has to be checkable by the reader"

    pin = pathlib.Path(__file__).resolve().parents[1] / "requirements" / "diffusers-main.txt"
    commit = _re.search(r"@([0-9a-fA-F]{40})\b", pin.read_text(encoding = "utf-8"))
    assert commit is not None, "the main pin must carry a full commit for the zip route to exist"
    assert (
        f"https://github.com/huggingface/diffusers/archive/{commit.group(1).lower()}.zip" in message
    ), message

    # A released minimum keeps the ordinary remedy.
    released = _too_old_message("Krea2Pipeline", "krea-2", "0.38.0")
    assert "pip install -U 'diffusers>=0.39.0'" in released

    # Every unreleased entry must still be a minimum some class actually declares, so a stale one
    # cannot sit here unnoticed after its release ships.
    declared = set(_PIPELINE_MIN_DIFFUSERS.values())
    assert _UNRELEASED_MIN_DIFFUSERS <= declared, sorted(_UNRELEASED_MIN_DIFFUSERS - declared)


def test_qwen_image_21_takes_reference_images_but_is_not_an_edit_only_family():
    """2.1 is unified, so it is the FLUX.2 shape and not the Qwen-Image-Edit one.

    ``QwenImage21Pipeline.__call__`` takes ``image`` as optional condition images beside the prompt,
    with no ``strength`` and the size from width/height, which is what the reference workflow passes.
    ``edit`` would mean the pipeline IS the edit pipeline with no plain text-to-image, which is
    Qwen-Image-Edit, a different model with a different pipeline class. Getting this wrong in either
    direction is silent: False refuses reference images outright, True would demand an input image
    for every generation.
    """
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    assert fam.reference is True
    assert fam.edit is False
    assert fam.pipeline_class == "QwenImage21Pipeline"
    # It also has no separate img2img or inpaint pipeline upstream: the one class covers both jobs.
    assert fam.img2img_pipeline_class is None
    assert fam.inpaint_pipeline_class is None

    # The edit family is a different model entirely, and must not have been merged into this one.
    edit = detect_family("Qwen/Qwen-Image-Edit-2511")
    assert edit is not None and edit.name == "qwen-image-edit" and edit.edit is True
    assert edit.pipeline_class != fam.pipeline_class
