# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from eval.json_score.schema import (
    ArrayNode,
    LeafNode,
    ObjectNode,
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
    node = normalize_schema({"vendor": {"name": "string"}, "items": [{"price": "money"}]})
    assert node == ObjectNode(
        {
            "vendor": ObjectNode({"name": LeafNode("string", {})}),
            "items": ArrayNode(ObjectNode({"price": LeafNode("money", {})})),
        }
    )


def test_empty_array_schema_is_error():
    with pytest.raises(ValueError, match = "single-element list"):
        normalize_schema([])


def test_multi_element_array_schema_is_error():
    with pytest.raises(ValueError, match = "single-element list"):
        normalize_schema(["string", "money"])


def test_unknown_comparator_is_error():
    with pytest.raises(ValueError, match = "Unknown comparator"):
        normalize_schema("not_a_comparator")


def test_bad_leaf_param_is_error():
    with pytest.raises(ValueError, match = "Invalid params"):
        normalize_schema({"type": "money", "bogus": 1})
