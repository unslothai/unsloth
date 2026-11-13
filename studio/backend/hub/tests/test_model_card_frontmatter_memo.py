# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cached-models listing parses each README's front matter once per unchanged file."""

from collections import OrderedDict

import yaml

from hub.services.models import cache_inventory

CARD = "---\npipeline_tag: text-generation\nlibrary_name: transformers\ntags:\n- a\n- ' b '\n- 3\nmodel-index: [{name: x}]\n---\n# Model\n"
FIELDS = {"pipeline_tag": "text-generation", "library_name": "transformers", "tags": ["a", "b"]}


def _counting_yaml(monkeypatch, settle_ns = -1):
    calls = []
    original = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda text: (calls.append(1), original(text))[1])
    monkeypatch.setattr(cache_inventory, "_CARD_FIELDS", OrderedDict(), raising = False)
    # Files the test writes are inside the settle window; most tests treat them as settled.
    monkeypatch.setattr(cache_inventory, "_CARD_FIELDS_SETTLE_NS", settle_ns, raising = False)
    return calls


def test_an_unchanged_card_is_parsed_once_and_returned_as_a_copy(tmp_path, monkeypatch):
    calls = _counting_yaml(monkeypatch)
    card = tmp_path / "README.md"
    card.write_text(CARD, encoding = "utf-8")
    first = cache_inventory._model_card_fields(card)
    assert first == FIELDS
    first["tags"].append("mutated")
    for _ in range(3):
        assert cache_inventory._model_card_fields(card) == FIELDS
    assert len(calls) == 1


def test_listing_metadata_carries_the_card_fields(tmp_path, monkeypatch):
    _counting_yaml(monkeypatch)
    (tmp_path / "README.md").write_text(CARD, encoding = "utf-8")
    for _ in range(2):
        meta = cache_inventory._cached_model_local_metadata(tmp_path, snapshot = tmp_path)
        assert {k: meta[k] for k in FIELDS} == FIELDS
        assert list(meta)[-3:] == ["pipeline_tag", "library_name", "tags"]


def test_a_changed_card_is_parsed_again(tmp_path, monkeypatch):
    calls = _counting_yaml(monkeypatch)
    card = tmp_path / "README.md"
    card.write_text(CARD, encoding = "utf-8")
    assert cache_inventory._model_card_fields(card)["pipeline_tag"] == "text-generation"
    card.write_text(
        CARD.replace("text-generation", "image-text-to-text") + "more\n", encoding = "utf-8"
    )
    for _ in range(2):
        assert cache_inventory._model_card_fields(card)["pipeline_tag"] == "image-text-to-text"
    assert len(calls) == 2


def test_a_card_inside_the_settle_window_is_parsed_every_time(tmp_path, monkeypatch):
    calls = _counting_yaml(monkeypatch, settle_ns = 3600 * 10**9)
    card = tmp_path / "README.md"
    card.write_text(CARD, encoding = "utf-8")
    for _ in range(2):
        assert cache_inventory._model_card_fields(card) == FIELDS
    assert len(calls) == 2


def test_missing_and_plain_cards_give_no_fields(tmp_path, monkeypatch):
    _counting_yaml(monkeypatch)
    assert cache_inventory._model_card_fields(tmp_path / "missing.md") == {}
    plain = tmp_path / "README.md"
    plain.write_text("# no front matter\n", encoding = "utf-8")
    for _ in range(2):
        assert cache_inventory._model_card_fields(plain) == {}
