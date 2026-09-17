# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nested tool-schema objects reach llama-server as a permissive union (#10839)."""

import copy
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_tool_schema import (
    _PERMISSIVE_OBJECT,
    llama_grammar_tools,
    relax_nested_object_key_order,
)

# Notion's notion-query-data-sources ``data`` branches, property order as the server declares it.
_SQL_BRANCH = {
    "type": "object",
    "properties": {
        "data_source_urls": {"maxItems": 100, "type": "array", "items": {"type": "string"}},
        "query": {"type": "string"},
        "mode": {"type": "string", "enum": ["sql"]},
        "params": {
            "maxItems": 100,
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
        },
    },
    "required": ["data_source_urls", "query"],
    "additionalProperties": False,
}
_VIEW_BRANCH = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["view"]},
        "view_url": {"type": "string"},
        "start_cursor": {"type": "string", "minLength": 1, "maxLength": 200},
        "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
        "is_archived": {"type": "boolean"},
    },
    "required": ["mode", "view_url"],
    "additionalProperties": False,
}


def _notion_parameters():
    return {
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "properties": {
            "data": {
                "description": "The data required for querying data sources",
                "anyOf": [copy.deepcopy(_SQL_BRANCH), copy.deepcopy(_VIEW_BRANCH)],
            }
        },
        "required": ["data"],
        "additionalProperties": {},
    }


def _tool(parameters, name = "mcp__notion__notion-query-data-sources"):
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def _relaxed(original):
    if "$ref" in original or "anyOf" in original or "oneOf" in original:
        notes = {
            key: original[key] for key in ("description", "nullable", "title") if key in original
        }
        return {**notes, "anyOf": [_PERMISSIVE_OBJECT, original]}
    kind = original.get("type")
    scalar = {
        key: value
        for key, value in original.items()
        if key not in ("properties", "required", "additionalProperties", "description", "title")
    }
    others = (
        [{**scalar, "type": t} for t in kind if t != "object"] if isinstance(kind, list) else []
    )
    return {**original, "anyOf": [_PERMISSIVE_OBJECT, *others]}


_PAGING = {
    "type": "object",
    "description": "Paging",
    "properties": {"start_cursor": {"type": "string"}, "page_size": {"type": "integer"}},
}
_DESCRIBED_REF = {"$ref": "#/$defs/Paging", "description": "Pagination for the next page"}


def test_wrapper_keeps_the_fields_chat_templates_read():
    parameters = {
        "type": "object",
        "properties": {"filter": copy.deepcopy(_PAGING), "next": copy.deepcopy(_DESCRIBED_REF)},
        "$defs": {"Paging": copy.deepcopy(_PAGING)},
    }
    relaxed = relax_nested_object_key_order(parameters)["properties"]

    assert relaxed["filter"] == {
        "type": "object",
        "description": "Paging",
        "properties": {"start_cursor": {"type": "string"}, "page_size": {"type": "integer"}},
        "anyOf": [{"type": "object", "additionalProperties": True}],
    }
    assert relaxed["next"] == {
        "description": "Pagination for the next page",
        "anyOf": [{"type": "object", "additionalProperties": True}, _DESCRIBED_REF],
    }


def test_type_list_alternatives_keep_their_own_constraints():
    target = {
        "type": ["object", "string", "null"],
        "description": "A cursor or a query object",
        "pattern": "^s:[a-z0-9_]+$",
        "minLength": 3,
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    parameters = {"type": "object", "properties": {"target": copy.deepcopy(target)}}

    assert relax_nested_object_key_order(parameters)["properties"]["target"]["anyOf"] == [
        {"type": "object", "additionalProperties": True},
        {"type": "string", "pattern": "^s:[a-z0-9_]+$", "minLength": 3},
        {"type": "null", "pattern": "^s:[a-z0-9_]+$", "minLength": 3},
    ]


@pytest.mark.parametrize("template", ["gemma-4.jinja", "gemma-4-edge.jinja"])
def test_gemma_prompt_still_declares_nested_fields(template):
    sandbox = pytest.importorskip("jinja2.sandbox")
    source = (Path(_BACKEND_DIR) / "assets" / "chat_templates" / template).read_text(
        encoding = "utf-8"
    )
    parameters = {
        "type": "object",
        "properties": {"filter": copy.deepcopy(_PAGING), "next": copy.deepcopy(_DESCRIBED_REF)},
        "$defs": {"Paging": copy.deepcopy(_PAGING)},
    }
    environment = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    rendered = environment.from_string(source).render(
        messages = [{"role": "user", "content": "hi"}],
        tools = llama_grammar_tools([_tool(parameters, name = "query")]),
        add_generation_prompt = True,
        bos_token = "<bos>",
    )
    assert "start_cursor" in rendered
    assert "page_size" in rendered
    assert "Pagination for the next page" in rendered


@pytest.mark.parametrize("template", ["gemma-4.jinja", "gemma-4-edge.jinja"])
@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "$ref"])
def test_gemma_prompt_preserves_direct_fields_beside_composition(template, keyword):
    sandbox = pytest.importorskip("jinja2.sandbox")
    source = (Path(_BACKEND_DIR) / "assets" / "chat_templates" / template).read_text(
        encoding = "utf-8"
    )
    view = copy.deepcopy(_VIEW_BRANCH)
    view[keyword] = (
        "#/$defs/View"
        if keyword == "$ref"
        else [
            {**copy.deepcopy(_VIEW_BRANCH), "required": ["mode", "view_url", field]}
            for field in ("start_cursor", "page_size")
        ]
    )
    parameters = {
        "type": "object",
        "properties": {"data": view},
        "required": ["data"],
        "$defs": {"View": copy.deepcopy(_VIEW_BRANCH)},
    }
    tools = [_tool(parameters, name = "query")]
    original = copy.deepcopy(tools)
    environment = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    compiled = environment.from_string(source)
    context = {
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": True,
        "bos_token": "<bos>",
    }
    expected = compiled.render(tools = tools, **context)
    assert "start_cursor" in expected
    assert 'required:[<|"|>mode<|"|>,<|"|>view_url<|"|>]' in expected
    assert compiled.render(tools = llama_grammar_tools(tools), **context) == expected
    assert tools == original


def test_flat_parameters_are_returned_unchanged():
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["query"],
    }
    assert relax_nested_object_key_order(parameters) is parameters


def test_root_object_is_never_wrapped():
    relaxed = relax_nested_object_key_order(_notion_parameters())
    assert relaxed["type"] == "object"
    assert relaxed["required"] == ["data"]
    assert list(relaxed["properties"]) == ["data"]


def test_nested_object_becomes_a_union_that_keeps_the_original():
    parameters = _notion_parameters()
    relaxed = relax_nested_object_key_order(parameters)

    data = relaxed["properties"]["data"]
    assert data["description"] == "The data required for querying data sources"
    assert data["anyOf"] == [_relaxed(_SQL_BRANCH), _relaxed(_VIEW_BRANCH)]
    assert parameters == _notion_parameters()


def test_relaxed_schema_admits_any_key_order_and_the_original_contract():
    jsonschema = pytest.importorskip("jsonschema")
    relaxed = relax_nested_object_key_order(_notion_parameters())

    call = {
        "data": {
            "mode": "view",
            "view_url": "https://app.notion.com/p/x?v=y",
            "page_size": 1,
            "is_archived": False,
            "start_cursor": "s:mcp_non_archived_1",
        }
    }
    jsonschema.validate(call, relaxed)
    jsonschema.validate(call, _notion_parameters())


def test_relaxing_twice_changes_nothing():
    once = relax_nested_object_key_order(_notion_parameters())
    assert relax_nested_object_key_order(once) is once


def test_objects_under_items_and_refs_are_wrapped_where_they_are_used():
    row = {"type": "object", "properties": {"id": {"type": "string"}, "note": {"type": "string"}}}
    parameters = {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": copy.deepcopy(row)},
            "first": {"type": "array", "prefixItems": [copy.deepcopy(row)]},
            "page": {"$ref": "#/$defs/Row"},
        },
        "$defs": {"Row": copy.deepcopy(row)},
    }
    relaxed = relax_nested_object_key_order(parameters)

    assert relaxed["properties"]["rows"]["items"] == _relaxed(row)
    assert relaxed["properties"]["first"]["prefixItems"] == [_relaxed(row)]
    assert relaxed["properties"]["page"] == _relaxed({"$ref": "#/$defs/Row"})
    assert relaxed["$defs"]["Row"] == row


def test_allof_ref_is_wrapped_around_the_use_and_not_the_definition():
    # A wrapped $defs target compiles to "{}" under llama.cpp's allOf merge, dropping every key.
    paging = {
        "type": "object",
        "properties": {"start_cursor": {"type": "string"}, "page_size": {"type": "integer"}},
    }
    filter_use = {"allOf": [{"$ref": "#/$defs/Filter"}], "description": "paging"}
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "filter": copy.deepcopy(filter_use)},
        "required": ["query"],
        "$defs": {"Filter": copy.deepcopy(paging)},
    }
    relaxed = relax_nested_object_key_order(parameters)

    assert relaxed["properties"]["filter"] == _relaxed(filter_use)
    assert relaxed["$defs"] is parameters["$defs"]

    recursive_root = {"$defs": {"Node": copy.deepcopy(paging)}, "allOf": [{"$ref": "#/$defs/Node"}]}
    assert relax_nested_object_key_order(recursive_root) is recursive_root


def test_references_are_followed_to_the_object_llama_cpp_builds():
    parts = [
        {"type": "object", "properties": {"start_cursor": {"type": "string"}}},
        {"type": "object", "properties": {"page_size": {"type": "integer"}}},
    ]
    parameters = {
        "type": "object",
        "properties": {
            "composed": {"$ref": "#/$defs/Paging"},
            "aliased": {"$ref": "#/$defs/Alias"},
            "looped": {"$ref": "#/$defs/Loop"},
        },
        "$defs": {
            "Paging": {"allOf": copy.deepcopy(parts)},
            "Alias": {"$ref": "#/$defs/Paging"},
            "Loop": {"$ref": "#/$defs/Loop"},
        },
    }
    relaxed = relax_nested_object_key_order(parameters)

    assert relaxed["properties"]["composed"] == _relaxed({"$ref": "#/$defs/Paging"})
    assert relaxed["properties"]["aliased"] == _relaxed({"$ref": "#/$defs/Alias"})
    assert relaxed["properties"]["looped"] == {"$ref": "#/$defs/Loop"}
    assert relaxed["$defs"] is parameters["$defs"]


def test_references_to_union_definitions_are_wrapped_at_the_use_site():
    # pydantic emits these for a TypeAliasType union field (oneOf) and a RootModel union field (anyOf).
    view = {
        "type": "object",
        "properties": {
            "kind": {"const": "view"},
            "start_cursor": {"type": "string"},
            "page_size": {"type": "integer"},
        },
        "required": ["kind"],
    }
    sql = {
        "type": "object",
        "properties": {"kind": {"const": "sql"}, "query": {"type": "string"}},
        "required": ["kind", "query"],
    }
    parameters = {
        "type": "object",
        "properties": {
            "aliased": {"$ref": "#/$defs/ChoiceAlias"},
            "rooted": {"$ref": "#/$defs/ChoiceRoot"},
            "looped": {"$ref": "#/$defs/Loop"},
        },
        "$defs": {
            "ChoiceAlias": {
                "oneOf": [{"$ref": "#/$defs/View"}, {"$ref": "#/$defs/Sql"}],
                "discriminator": {"propertyName": "kind"},
            },
            "ChoiceRoot": {"anyOf": [{"$ref": "#/$defs/View"}, {"$ref": "#/$defs/Sql"}]},
            "Loop": {"anyOf": [{"$ref": "#/$defs/Loop"}, {"type": "string"}]},
            "View": copy.deepcopy(view),
            "Sql": copy.deepcopy(sql),
        },
    }
    relaxed = relax_nested_object_key_order(parameters)

    assert relaxed["properties"]["aliased"] == _relaxed({"$ref": "#/$defs/ChoiceAlias"})
    assert relaxed["properties"]["rooted"] == _relaxed({"$ref": "#/$defs/ChoiceRoot"})
    assert relaxed["properties"]["looped"] == {"$ref": "#/$defs/Loop"}
    assert relaxed["$defs"] is parameters["$defs"]


def test_a_caller_union_shaped_like_the_wrapper_is_read_as_written():
    from core.inference.tool_loop_controller import coerce_arguments_by_schema

    properties = {
        "payload": {
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "object", "properties": {"code": {"type": "integer"}}},
            ]
        }
    }
    coerced = coerce_arguments_by_schema({"payload": {"code": "001"}}, properties, repair = True)
    assert coerced == {"payload": {"code": "001"}}


def test_root_union_branches_are_the_root_and_stay_bare():
    by_id = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["id"],
    }
    by_url = {"type": "object", "properties": {"url": {"type": "string"}, "c": {"type": "string"}}}
    parameters = {"type": "object", "anyOf": [by_id, by_url]}
    assert relax_nested_object_key_order(parameters) is parameters

    root_ref = {"$ref": "#/$defs/Args", "$defs": {"Args": copy.deepcopy(by_id)}}
    assert relax_nested_object_key_order(root_ref) is root_ref


@pytest.mark.parametrize("forced", [False, True])
def test_passthrough_healer_types_arguments_against_the_original_schema(forced):
    import json

    from core.inference.passthrough_healing import heal_openai_message
    from routes.inference import _build_passthrough_payload

    data = {
        "type": "object",
        "properties": {
            "view_url": {"type": "string"},
            "start_cursor": {"type": "string"},
            "page_size": {"type": "integer"},
            "is_archived": {"type": "boolean"},
        },
        "required": ["view_url"],
    }
    tool = _tool({"type": "object", "properties": {"data": data}, "required": ["data"]}, name = "q")
    body = _build_passthrough_payload(
        [{"role": "user", "content": "hi"}],
        [tool, _tool({"type": "object", "properties": {}}, name = "other")],
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        stream = False,
        tool_choice = {"type": "function", "function": {"name": "q"}} if forced else "auto",
        max_tokens = 16,
        stop = None,
        backend_ctx = 4096,
    )
    assert body["tool_choice"] == ("required" if forced else "auto")
    assert [tool["function"]["name"] for tool in body["tools"]] == (
        ["q"] if forced else ["q", "other"]
    )
    assert body["tools"][0]["function"]["parameters"]["properties"]["data"] == _relaxed(data)

    call = {
        "name": "q",
        "arguments": {
            "data": {"view_url": "u", "page_size": "1", "is_archived": "false", "start_cursor": "c"}
        },
    }
    message = {"role": "assistant", "content": f"<tool_call>{json.dumps(call)}</tool_call>"}
    assert heal_openai_message(message, {"q"}, body["tools"])

    arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {
        "data": {"view_url": "u", "page_size": 1, "is_archived": False, "start_cursor": "c"}
    }


@pytest.mark.parametrize("union", ["anyOf", "oneOf"])
@pytest.mark.parametrize("null_first", [False, True])
def test_passthrough_healer_types_nullable_object_arguments(union, null_first):
    import json

    from core.inference.passthrough_healing import heal_openai_message
    from routes.inference import _build_passthrough_payload

    data = {
        "type": "object",
        "properties": {
            "start_cursor": {"type": "string"},
            "page_size": {"type": "integer"},
            "is_archived": {"type": "boolean"},
        },
    }
    branches = [{"type": "null"}, data] if null_first else [data, {"type": "null"}]
    tool = _tool({"type": "object", "properties": {"data": {union: branches}}}, name = "q")
    body = _build_passthrough_payload(
        [{"role": "user", "content": "hi"}],
        [tool],
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        stream = False,
        tool_choice = "auto",
        max_tokens = 16,
        stop = None,
        backend_ctx = 4096,
    )
    for value in (
        {"start_cursor": "001", "page_size": "1", "is_archived": "false"},
        '{"start_cursor":"001","page_size":"1","is_archived":"false"}',
        None,
    ):
        call = {"name": "q", "arguments": {"data": value}}
        message = {"role": "assistant", "content": f"<tool_call>{json.dumps(call)}</tool_call>"}
        assert heal_openai_message(message, {"q"}, body["tools"])
        expected = (
            None if value is None else {"start_cursor": "001", "page_size": 1, "is_archived": False}
        )
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"data": expected}


def test_nullable_nested_object_is_wrapped():
    options = {
        "type": ["object", "null"],
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    parameters = {"type": "object", "properties": {"options": copy.deepcopy(options)}}
    assert relax_nested_object_key_order(parameters)["properties"]["options"] == _relaxed(options)


def test_nested_objects_without_two_optional_keys_are_left_alone():
    parameters = {
        "type": "object",
        "properties": {
            "labels": {"type": "object", "additionalProperties": {"type": "string"}},
            "blob": {"type": "object"},
            "pinned": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
            "single": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            },
        },
    }
    assert relax_nested_object_key_order(parameters) is parameters


def test_a_property_named_properties_is_not_read_as_the_keyword():
    parameters = {
        "type": "object",
        "properties": {"properties": {"type": "string"}, "type": {"type": "string"}},
    }
    assert relax_nested_object_key_order(parameters) is parameters

    inner = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    parameters = {"type": "object", "properties": {"properties": copy.deepcopy(inner)}}
    relaxed = relax_nested_object_key_order(parameters)
    assert relaxed["properties"] == {"properties": _relaxed(inner)}


def test_catalog_rewrites_only_parameters_and_keeps_untouched_entries():
    flat = _tool({"type": "object", "properties": {"q": {"type": "string"}}}, name = "search")
    bare = {"type": "function", "function": {"name": "noop"}}
    nested = _tool(_notion_parameters())
    tools = [flat, bare, "not-a-tool", nested]

    out = llama_grammar_tools(tools)

    assert out[0] is flat
    assert out[1] is bare
    assert out[2] == "not-a-tool"
    assert out[3]["function"]["name"] == nested["function"]["name"]
    assert out[3]["function"]["parameters"]["properties"]["data"]["anyOf"][1] == _relaxed(
        _VIEW_BRANCH
    )
    assert nested == _tool(_notion_parameters())


def test_catalog_without_nested_objects_is_the_same_list():
    tools = [_tool({"type": "object", "properties": {"q": {"type": "string"}}}, name = "search")]
    assert llama_grammar_tools(tools) is tools
    assert llama_grammar_tools(None) is None


def test_passthrough_body_carries_the_relaxed_catalog():
    from routes.inference import _build_passthrough_payload

    body = _build_passthrough_payload(
        [{"role": "user", "content": "hi"}],
        [_tool(_notion_parameters())],
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        stream = False,
        tool_choice = "auto",
        max_tokens = 16,
        stop = None,
        backend_ctx = 4096,
    )
    data = body["tools"][0]["function"]["parameters"]["properties"]["data"]
    assert data["anyOf"][1]["anyOf"] == [_PERMISSIVE_OBJECT]
    assert data["anyOf"][1]["properties"] == _VIEW_BRANCH["properties"]
