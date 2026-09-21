# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_resolve_variant_gguf_files: a whole repo listing is a verdict, not a licence to guess.

The load path used to fall through to a synthesised ``{repo}-{variant}.gguf`` name even
when the listing it had just read showed downloadable GGUFs but none for the requested
variant -- a guaranteed-404 filename issued straight to the download tier, surfacing as
an opaque EntryNotFoundError/500 (#11343). With a listing in hand the resolver must
name the variants the picker advertised instead.
"""

import pytest

import core.inference.llama_cpp as llama_cpp

# The shape #11343 reported: one main file, the quant token packed inside a longer
# token ("PQ2_0"), so the file answers only to tokens its own name carries.
BONSAI_FILES = ["Bonsai-2-27B-PQ2_0-CRACK.gguf"]
BONSAI_REPO = "dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF"


@pytest.fixture
def hub_unreachable(monkeypatch):
    """A listing the Hub never returns."""

    def boom(repo_id, token = None):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr("huggingface_hub.list_repo_files", boom)


@pytest.fixture
def no_cached_variant(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_cached_variant_resolution", lambda repo, variant: (None, []))


def test_a_whole_listing_that_names_no_file_for_the_variant_raises_helpfully(
    monkeypatch, no_cached_variant
):
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files", lambda repo_id, token = None: list(BONSAI_FILES)
    )

    with pytest.raises(ValueError) as exc_info:
        llama_cpp._resolve_variant_gguf_files(BONSAI_REPO, "Q7_0")

    msg = str(exc_info.value)
    assert "Q7_0" in msg
    # The offered table uses the SAME identity the variants picker advertised,
    # whatever the label extraction made of the filename (here "Q2_0").
    assert "Available variants: Q2_0" in msg


def test_a_whole_listing_of_only_companion_ggufs_is_not_a_variant_table(
    monkeypatch, no_cached_variant
):
    """No main files means nothing to offer the user: keep the convention guess."""
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, token = None: ["mtp-draft.gguf", "mmproj-f16.gguf"],
    )

    filename, _ = llama_cpp._resolve_variant_gguf_files("unsloth/Llama-3.2-1B-GGUF", "Q4_K_M")
    assert filename == "Llama-3.2-1B-Q4_K_M.gguf"


def test_a_failed_listing_still_falls_through_to_synthesis(hub_unreachable, no_cached_variant):
    """A transiently unreachable Hub stays fail-soft; the user is not billed a verdict."""
    filename, _ = llama_cpp._resolve_variant_gguf_files("unsloth/Llama-3.2-1B-GGUF", "Q4_K_M")
    assert filename == "Llama-3.2-1B-Q4_K_M.gguf"


def test_a_whole_listing_beats_the_cache_for_an_incorrect_variant(monkeypatch, no_cached_variant):
    """The raise fires after the cache tier ran, so cache is still a legal escape."""
    calls = []

    def fake_cached(repo, variant):
        calls.append(variant)
        return None, []

    monkeypatch.setattr(llama_cpp, "_cached_variant_resolution", fake_cached)
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files", lambda repo_id, token = None: list(BONSAI_FILES)
    )

    with pytest.raises(ValueError):
        llama_cpp._resolve_variant_gguf_files(BONSAI_REPO, "Q7_0")
    assert calls == ["Q7_0"], "the listing verdict came before the cache tier"


def test_a_whole_listing_still_names_a_variant_it_advertises(monkeypatch, no_cached_variant):
    """The happy path is untouched: the file the menu advertised is what resolves."""
    files = [
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Bonsai-2-27B-PQ2_0-CRACK.gguf",
    ]
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo_id, token = None: files)

    filename, shards = llama_cpp._resolve_variant_gguf_files("unsloth/Llama-3.2-1B-GGUF", "Q4_K_M")
    assert filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    assert shards == []
