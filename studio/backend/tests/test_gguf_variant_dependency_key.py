# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The per-variant dependency key a GGUF listing carries.

The companion download footprint (text encoders, VAE, tokenizer, configs) is not a
property of the repo, so a client that resolves it once per repo advertises a
GB-wrong "Full required size" on every row that is not the one it sampled. The key
exists so the client can group rows correctly, which means it has to move whenever
the companion set moves and stay put when it does not.
"""

from hub.services.models.gguf_variants import _variant_dependency_key


KLEIN_REPO = "unsloth/FLUX.2-klein-GGUF"


def test_klein_4b_and_9b_in_one_repo_get_different_keys():
    """sd_cpp_text_encoders_for hands klein-9B Qwen3-8B and klein-4B the family
    default, keyed on repo id + filename. Same repo, same family, different
    companions, so a repo-wide footprint is wrong for one of the two rows."""
    key_4b = _variant_dependency_key(KLEIN_REPO, "flux.2-klein-4b-Q4_K_M.gguf")
    key_9b = _variant_dependency_key(KLEIN_REPO, "flux.2-klein-9b-Q4_K_M.gguf")
    assert key_4b is not None and key_9b is not None
    assert key_4b != key_9b


def test_two_quants_of_the_same_model_share_one_key():
    """The common case must stay a single group: the client resolves one footprint
    per key, and splitting quants of one model would multiply the requests."""
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
    """It runs inside the variant listing, once per row. A family-detection failure
    must cost the listing nothing, and a socket here would stall the picker."""
    import hub.services.models.gguf_variants as gv

    def _boom(*args, **kwargs):
        raise RuntimeError("family registry unavailable")

    monkeypatch.setattr("core.inference.diffusion_families.detect_family_for_pick", _boom)
    assert gv._variant_dependency_key(KLEIN_REPO, "flux.2-klein-4b-Q4_K_M.gguf") is None


def test_the_listing_carries_the_key_per_variant():
    """The field is on the variant row, not the response, because the response is
    the repo and the repo is exactly the wrong granularity."""
    from hub.schemas.inventory import GgufVariantDetail

    field = GgufVariantDetail.model_fields["dependency_key"]
    assert field.default is None
    detail = GgufVariantDetail(filename = "a.gguf", quant = "Q4_K_M")
    assert detail.dependency_key is None
