# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The stored Hub listing is a fallback for update/companion state, never for disk facts."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from hub.schemas.inventory import GgufVariantDetail, GgufVariantsResponse
from hub.services.models import gguf_variants
from hub.utils import variant_hub_state


@pytest.fixture()
def hub_state_root(tmp_path, monkeypatch):
    root = tmp_path / "hub-state"
    monkeypatch.setattr(variant_hub_state, "state_root", lambda **_kwargs: root)
    monkeypatch.setattr(variant_hub_state, "_endpoint", lambda: "https://huggingface.co")
    return root


def _response(**overrides) -> GgufVariantsResponse:
    fields = dict(
        repo_id = "Org/Cached",
        variants = [
            GgufVariantDetail(
                quant = "Q4_K_M",
                filename = "model-Q4_K_M.gguf",
                size_bytes = 1024,
                download_size_bytes = 2048,
                downloaded = True,
                update_available = True,
                pending_drafter_filename = "mtp-drafter.gguf",
                pending_drafter_size_bytes = 64,
                dependency_key = "family",
            )
        ],
        has_vision = True,
        default_variant = "Q4_K_M",
        dependencies_resolved = True,
    )
    fields.update(overrides)
    return GgufVariantsResponse(**fields)


def test_a_live_answer_is_stored_without_the_disk_facts(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())

    stored = variant_hub_state.stored_listing("Org/Cached")
    assert stored is not None
    entry = stored["variants"][0]
    assert entry["update_available"] is True
    assert entry["pending_drafter_filename"] == "mtp-drafter.gguf"
    # What the cache walk answers must never be restored from a record.
    for field in ("downloaded", "partial", "size_bytes", "cache_path", "dependency_key"):
        assert field not in entry


def test_a_record_from_another_credential_or_endpoint_is_not_this_callers_answer(
    hub_state_root, monkeypatch
):
    variant_hub_state.record_listing("Org/Cached", _response(), hf_token = "hf_gated")
    assert variant_hub_state.stored_listing("Org/Cached", hf_token = "hf_gated") is not None
    assert variant_hub_state.stored_listing("Org/Cached") is None, "gated state served anonymously"
    assert variant_hub_state.stored_listing("Org/Cached", hf_token = "hf_other") is None

    monkeypatch.setattr(variant_hub_state, "_endpoint", lambda: "https://modelscope.example")
    assert variant_hub_state.stored_listing("Org/Cached", hf_token = "hf_gated") is None


def test_an_expired_record_is_not_served(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    now = time.time()
    assert variant_hub_state.stored_listing("Org/Cached", max_age_seconds = 60, now = now + 61) is None
    assert (
        variant_hub_state.stored_listing("Org/Cached", max_age_seconds = 60, now = now + 5) is not None
    )


def test_a_corrupt_or_half_written_record_is_ignored(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    path = next((hub_state_root / "variant-hub").glob("*.json"))
    path.write_text("{not json", encoding = "utf-8")
    assert variant_hub_state.stored_listing("Org/Cached") is None


def test_merging_keeps_every_disk_fact_and_takes_only_the_hub_ones(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    stored = variant_hub_state.stored_listing("Org/Cached")

    # The disk answer no longer knows about the update, and reports what is really on disk.
    local = _response(
        has_vision = False,
        default_variant = None,
        dependencies_resolved = False,
        variants = [
            GgufVariantDetail(
                quant = "Q4_K_M",
                filename = "model-Q4_K_M.gguf",
                size_bytes = 1024,
                downloaded = True,
                cache_path = "/cache/old",
                update_available = False,
            )
        ],
    )
    merged = variant_hub_state.merged_response(local, stored)

    assert merged.variants[0].update_available is True
    assert merged.variants[0].pending_drafter_filename == "mtp-drafter.gguf"
    assert merged.variants[0].pending_drafter_size_bytes == 64
    assert merged.variants[0].cache_path == "/cache/old", "disk facts must survive the merge"
    assert merged.variants[0].size_bytes == 1024
    assert merged.default_variant == "Q4_K_M"
    assert merged.hub_state == "cached"
    assert merged.observed_at is not None


def test_a_quant_absent_from_disk_is_not_invented(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    stored = variant_hub_state.stored_listing("Org/Cached")
    local = _response(
        variants = [
            GgufVariantDetail(
                quant = "Q8_0", filename = "model-Q8_0.gguf", size_bytes = 512, downloaded = True
            )
        ]
    )
    merged = variant_hub_state.merged_response(local, stored)
    assert [v.quant for v in merged.variants] == ["Q8_0"]
    assert merged.variants[0].update_available is False


def _answer(**overrides) -> gguf_variants.VariantsAnswer:
    return gguf_variants.VariantsAnswer(_response(**overrides), None)


def _live(**overrides) -> gguf_variants.VariantsAnswer:
    """An answer the Hub produced: what _compute_with_cleanables stamps in its worker thread."""
    return gguf_variants.VariantsAnswer(_response(hub_state = "live", **overrides), None)


def test_a_live_answer_is_marked_live_and_recorded(hub_state_root, monkeypatch):
    answer = gguf_variants._hub_state_answer(_live(), "Org/Cached", None, offline = False)
    assert answer.response.hub_state == "live"
    assert variant_hub_state.stored_listing("Org/Cached") is not None


def test_an_answer_that_could_not_reach_the_hub_is_marked_cached(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    # dependencies_resolved is True here on purpose: the disk alone can prove the companion set,
    # so that field does not mean the Hub answered.
    disk = _answer(dependencies_resolved = True, variants = _response().variants[:1])
    answer = gguf_variants._hub_state_answer(disk, "Org/Cached", None, offline = False)
    assert answer.response.hub_state == "cached"
    assert answer.response.variants[0].update_available is True


def test_a_local_or_mlx_answer_never_reads_or_writes_the_record(hub_state_root):
    for kwargs in ({"resolved_locally": True}, {"variant_format": "mlx"}):
        answer = _live(**kwargs)
        returned = gguf_variants._hub_state_answer(answer, "Org/Cached", None, offline = False)
        assert returned is answer, "a local or MLX answer was rebuilt for no reason"
        assert (
            variant_hub_state.stored_listing("Org/Cached") is None
        ), "nothing about a local path or an MLX listing belongs in the record"


def test_an_offline_answer_keeps_its_rows_and_offers_nothing_stored(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    disk = _answer(
        variants = [
            GgufVariantDetail(
                quant = "Q4_K_M",
                filename = "model-Q4_K_M.gguf",
                size_bytes = 1024,
                downloaded = True,
                update_available = False,
            )
        ]
    )
    answer = gguf_variants._hub_state_answer(disk, "Org/Cached", None, offline = True)
    assert answer.response.hub_state == "none", "an offline listing must not claim Hub state"
    assert (
        answer.response.variants[0].update_available is False
    ), "an offline request must not be answered from the record"


def test_the_wrapper_falls_back_to_the_stored_listing_when_the_hub_fails(
    hub_state_root, monkeypatch
):
    variant_hub_state.record_listing("Org/Cached", _response())
    calls = []

    async def failing(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("offline"):
            return gguf_variants.VariantsAnswer(
                _response(
                    dependencies_resolved = False,
                    variants = [
                        GgufVariantDetail(
                            quant = "Q4_K_M",
                            filename = "model-Q4_K_M.gguf",
                            size_bytes = 1024,
                            downloaded = True,
                            cache_path = "/cache/old",
                            update_available = False,
                        )
                    ],
                ),
                None,
            )
        raise ConnectionError("Hub unreachable")

    monkeypatch.setattr(gguf_variants, "_compute_gguf_variants_answer", failing)
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer("Org/Cached"))

    assert calls and calls[-1].get("offline") is True, "the rows must be re-read from disk"
    assert answer.response.hub_state == "cached"
    assert answer.response.variants[0].cache_path == "/cache/old"
    assert answer.response.variants[0].update_available is True


def test_the_wrapper_reraises_when_the_hub_fails_and_nothing_is_stored(hub_state_root, monkeypatch):
    async def failing(*args, **kwargs):
        raise ConnectionError("Hub unreachable")

    monkeypatch.setattr(gguf_variants, "_compute_gguf_variants_answer", failing)
    with pytest.raises(ConnectionError):
        asyncio.run(gguf_variants.get_gguf_variants_answer("Org/Cached"))


def test_a_missing_repo_stays_a_404_instead_of_a_stored_answer(hub_state_root, monkeypatch):
    from fastapi import HTTPException

    variant_hub_state.record_listing("Org/Cached", _response())

    async def gated(*args, **kwargs):
        raise HTTPException(status_code = 404, detail = "not found")

    monkeypatch.setattr(gguf_variants, "_compute_gguf_variants_answer", gated)
    with pytest.raises(HTTPException):
        asyncio.run(gguf_variants.get_gguf_variants_answer("Org/Cached"))


def test_disk_proof_of_dependencies_is_not_mistaken_for_a_hub_answer(hub_state_root):
    """With the Hub unreadable the disk alone can prove the companion set, and then
    dependencies_resolved is true. That must not overwrite the record with a disk answer."""
    first = _response(variants = _response().variants[:1])
    variant_hub_state.record_listing("Org/Cached", first)
    before = variant_hub_state.stored_listing("Org/Cached")

    disk_only = _answer(
        dependencies_resolved = True,
        variants = [
            GgufVariantDetail(
                quant = "Q4_K_M",
                filename = "model-Q4_K_M.gguf",
                size_bytes = 1024,
                downloaded = True,
                update_available = False,
            )
        ],
    )
    answer = gguf_variants._hub_state_answer(disk_only, "Org/Cached", None, offline = False)

    after = variant_hub_state.stored_listing("Org/Cached")
    assert after == before, "a disk answer replaced the record"
    assert answer.response.hub_state == "cached"


def test_a_live_answer_overwrites_the_record(hub_state_root):
    variant_hub_state.record_listing("Org/Cached", _response())
    assert variant_hub_state.stored_listing("Org/Cached")["variants"][0]["update_available"] is True

    current = _answer(
        variants = [
            GgufVariantDetail(
                quant = "Q4_K_M",
                filename = "model-Q4_K_M.gguf",
                size_bytes = 1024,
                downloaded = True,
                update_available = False,
            )
        ],
        default_variant = "Q4_K_M",
    )
    gguf_variants._hub_state_answer(
        _live(variants = current.response.variants, default_variant = "Q4_K_M"),
        "Org/Cached",
        None,
        offline = False,
    )
    stored = variant_hub_state.stored_listing("Org/Cached")
    assert stored is not None
    assert stored["variants"][0]["update_available"] is False, "the newest answer did not win"
