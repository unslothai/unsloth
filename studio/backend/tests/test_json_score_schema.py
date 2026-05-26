# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from eval.json_score.schema import (
    ArrayNode,
    LeafNode,
    ObjectNode,
    json_schema_to_node,
    normalize_schema,
)


def test_shorthand_string_is_leaf():
    node = normalize_schema("money")
    assert node == LeafNode("money", {})


def test_leaf_dict_with_type_and_params():
    node = normalize_schema({"type": "money", "rel_tol": 0.01})
    assert node == LeafNode("money", {"rel_tol": 0.01})


def test_object_schema():
    node = normalize_schema({"name": "string", "vat_id": "categorical"})
    assert node == ObjectNode(
        {"name": LeafNode("string", {}), "vat_id": LeafNode("categorical", {})}
    )


def test_array_schema():
    node = normalize_schema(["string"])
    assert node == ArrayNode(LeafNode("string", {}))


def test_nested_schema():
    node = normalize_schema(
        {"vendor": {"name": "string"}, "items": [{"price": "money"}]}
    )
    assert node == ObjectNode(
        {
            "vendor": ObjectNode({"name": LeafNode("string", {})}),
            "items": ArrayNode(ObjectNode({"price": LeafNode("money", {})})),
        }
    )


def test_empty_array_schema_is_error():
    with pytest.raises(ValueError, match="single-element list"):
        normalize_schema([])


def test_multi_element_array_schema_is_error():
    with pytest.raises(ValueError, match="single-element list"):
        normalize_schema(["string", "money"])


def test_unknown_comparator_is_error():
    with pytest.raises(ValueError, match="Unknown comparator"):
        normalize_schema("not_a_comparator")


def test_bad_leaf_param_is_error():
    with pytest.raises(ValueError, match="Invalid params"):
        normalize_schema({"type": "money", "bogus": 1})


# ── Standard JSON Schema translation ────────────────────────────────────────


def test_json_schema_types_map_to_comparators():
    node = json_schema_to_node(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "salary": {"type": "number"},
                "headcount": {"type": "integer"},
                "remote": {"type": "boolean"},
                "status": {"type": "string", "enum": ["open", "closed"]},
                "posted_date": {"type": "string", "format": "date"},
            },
        }
    )
    assert node == ObjectNode(
        {
            "title": LeafNode("string", {}),
            "salary": LeafNode("numeric", {}),
            "headcount": LeafNode("numeric", {}),
            "remote": LeafNode("categorical", {}),
            "status": LeafNode("categorical", {}),
            "posted_date": LeafNode("date", {}),
        }
    )


def test_json_schema_nested_object_and_array():
    node = json_schema_to_node(
        {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "salaries": {"type": "array", "items": {"type": "number"}},
            },
        }
    )
    assert node == ObjectNode(
        {
            "company": ObjectNode({"name": LeafNode("string", {})}),
            "tags": ArrayNode(LeafNode("string", {})),
            "salaries": ArrayNode(LeafNode("numeric", {})),
        }
    )


def test_json_schema_with_type_named_field_is_preserved():
    # A property literally named "type" must become a field, not be mistaken
    # for a leaf comparator spec.
    node = json_schema_to_node(
        {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "amount": {"type": "number"},
            },
        }
    )
    assert node == ObjectNode(
        {"type": LeafNode("string", {}), "amount": LeafNode("numeric", {})}
    )


def test_normalize_detects_json_schema_via_dollar_schema():
    # Regression: a pasted draft-2020-12 schema used to raise
    # "Unknown comparator 'https://json-schema.org/...'".
    node = normalize_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/schemas/job-posting.json",
            "title": "Job Posting",
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }
    )
    assert node == ObjectNode({"title": LeafNode("string", {})})


def test_legacy_field_schema_not_treated_as_json_schema():
    # No $schema/$id and no JSON-Schema-only type → still the comparator mapping.
    node = normalize_schema({"total": {"type": "money"}, "currency": "categorical"})
    assert node == ObjectNode(
        {"total": LeafNode("money", {}), "currency": LeafNode("categorical", {})}
    )
