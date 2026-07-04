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


def test_ideogram4_generation_defaults():
    # Model-card settings: 48 steps, guidance 7 (the backend keeps the pipeline's
    # recommended tapered schedule when the request matches exactly).
    assert default_generation_params("ideogram-ai/ideogram-4-fp8") == (48, 7.0)


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
