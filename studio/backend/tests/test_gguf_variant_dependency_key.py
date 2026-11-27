# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dependency key must move whenever the companion set moves, and stay put when it does not."""

from hub.services.models.gguf_variants import _variant_dependency_key


KLEIN_REPO = "unsloth/FLUX.2-klein-GGUF"


def test_klein_4b_and_9b_in_one_repo_get_different_keys():
    """sd_cpp_text_encoders_for keys klein-4B and klein-9B by repo id and filename: same repo and
    family but different companions, so a repo-wide footprint is wrong"""
    key_4b = _variant_dependency_key(KLEIN_REPO, "flux.2-klein-4b-Q4_K_M.gguf")
    key_9b = _variant_dependency_key(KLEIN_REPO, "flux.2-klein-9b-Q4_K_M.gguf")
    assert key_4b is not None and key_9b is not None
    assert key_4b != key_9b


def test_renamed_klein_uses_its_cached_header_size(monkeypatch):
    monkeypatch.setattr(
        "core.inference.diffusion_compat.flux2_inner_dim_for_pick",
        lambda *args, **kwargs: 4096,
    )
    renamed = _variant_dependency_key(KLEIN_REPO, "renamed-Q4_K_M.gguf")
    known_9b = _variant_dependency_key(KLEIN_REPO, "flux.2-klein-9b-Q4_K_M.gguf")
    assert renamed == known_9b


def test_unidentified_klein_filenames_do_not_share_a_footprint(monkeypatch):
    monkeypatch.setattr(
        "core.inference.diffusion_compat.flux2_inner_dim_for_pick",
        lambda *args, **kwargs: None,
    )
    first = _variant_dependency_key(KLEIN_REPO, "renamed-a-Q4_K_M.gguf")
    second = _variant_dependency_key(KLEIN_REPO, "renamed-b-Q4_K_M.gguf")
    assert first is not None and second is not None
    assert first != second


def test_two_quants_of_the_same_model_share_one_key():
    """Quants of one model share one key, since the client resolves one footprint per key."""
    assert _variant_dependency_key(
        KLEIN_REPO, "flux.2-klein-4b-Q4_K_M.gguf"
    ) == _variant_dependency_key(KLEIN_REPO, "flux.2-klein-4b-Q8_0.gguf")


def test_different_families_in_one_neutral_repo_get_different_keys():
    """detect_family_for_pick falls back to `repo_id/filename`, so a repo whose id
    matches no family can still hold GGUFs of two families with entirely different
    base repos."""
    key_flux = _variant_dependency_key("someone/mixed-gguf", "flux.1-dev-Q4_K_M.gguf")
    key_qwen = _variant_dependency_key("someone/mixed-gguf", "qwen-image-Q4_K_M.gguf")
    assert key_flux is not None and key_qwen is not None
    assert key_flux != key_qwen


def test_a_text_model_has_no_key():
    """No family resolves, so there is nothing to group by. Null, not a made-up key:
    the client treats an unkeyed listing as one group, which is the old behavior."""
    assert (
        _variant_dependency_key(
            "unsloth/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )
        is None
    )


def test_the_key_never_raises_and_makes_no_network_call(monkeypatch):
    """Runs once per row, so a family-detection failure must cost nothing and open no socket."""
    import hub.services.models.gguf_variants as gv

    def _boom(*args, **kwargs):
        raise RuntimeError("family registry unavailable")

    monkeypatch.setattr("core.inference.diffusion_families.detect_family_for_pick", _boom)
    assert gv._variant_dependency_key(KLEIN_REPO, "flux.2-klein-4b-Q4_K_M.gguf") is None


def test_the_listing_carries_the_key_per_variant():
    """dependency_key sits on the variant row; a per-repo response is the wrong granularity."""
    from hub.schemas.inventory import GgufVariantDetail

    field = GgufVariantDetail.model_fields["dependency_key"]
    assert field.default is None
    detail = GgufVariantDetail(filename = "a.gguf", quant = "Q4_K_M")
    assert detail.dependency_key is None
