# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from datasets import Dataset

from utils.datasets import format_and_template_dataset

_TEMPLATE = (
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}"
    "{% for message in messages %}<{{ message.role }}>{{ message.content }}{% endfor %}"
)


class _ToolsTokenizer:
    chat_template = _TEMPLATE
    eos_token = "</s>"

    def apply_chat_template(
        self,
        conversation,
        tools = None,
        **_kwargs,
    ):
        sandbox = pytest.importorskip("jinja2.sandbox")
        env = sandbox.ImmutableSandboxedEnvironment()
        env.filters["tojson"] = lambda value: json.dumps(value)
        return env.from_string(self.chat_template).render(messages = conversation, tools = tools)


class _NoToolsTokenizer(_ToolsTokenizer):
    def apply_chat_template(
        self,
        conversation,
        tools = None,
        **kwargs,
    ):
        if tools is not None:
            raise ValueError("tools are not supported")
        return super().apply_chat_template(conversation, **kwargs)


class _SystemWithToolsRejectedTokenizer(_ToolsTokenizer):
    def apply_chat_template(
        self,
        conversation,
        tools = None,
        **kwargs,
    ):
        if tools is not None and conversation[0]["role"] == "system":
            raise ValueError("system turns with tools are not supported")
        return super().apply_chat_template(conversation, tools = tools, **kwargs)


_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}

_MESSAGES = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "It is 21C."},
]


def _format(
    rows,
    tokenizer = None,
    **kwargs,
):
    result = format_and_template_dataset(
        Dataset.from_list(rows),
        model_name = "stub-model",
        tokenizer = tokenizer or _ToolsTokenizer(),
        num_proc = 1,
        **kwargs,
    )
    assert result["success"], result["errors"]
    return list(result["dataset"]["text"])


def _catalog(*tools):
    return "<tools>" + "".join(json.dumps(tool) for tool in tools) + "</tools>"


@pytest.mark.parametrize(
    "tools",
    [[_WEATHER], json.dumps([_WEATHER])],
    ids = ["list", "json_string"],
)
def test_tools_column_reaches_the_trained_text(tools):
    texts = _format([{"messages": _MESSAGES, "tools": tools}])

    assert texts == [_catalog(_WEATHER) + "<user>Weather in Paris?<assistant>It is 21C."]


def test_flat_json_string_tools_are_trained_in_the_chat_tool_shape():
    flat = json.dumps(_WEATHER["function"])

    texts = _format([{"messages": _MESSAGES, "tools": [flat]}])

    assert texts[0].startswith(_catalog(_WEATHER))


def test_each_row_trains_only_its_own_tools():
    texts = _format(
        [
            {"messages": _MESSAGES, "tools": [_WEATHER]},
            {"messages": _MESSAGES, "tools": [_SEARCH]},
            {"messages": _MESSAGES, "tools": []},
        ]
    )

    assert texts[0].startswith(_catalog(_WEATHER))
    assert texts[1].startswith(_catalog(_SEARCH))
    assert texts[2] == "<user>Weather in Paris?<assistant>It is 21C."


def test_tools_and_system_column_are_both_trained():
    texts = _format([{"system": "Be brief.", "messages": _MESSAGES, "tools": [_WEATHER]}])

    assert texts == [
        _catalog(_WEATHER) + "<system>Be brief.<user>Weather in Paris?<assistant>It is 21C."
    ]


def test_system_retry_keeps_the_tools_column_for_training():
    texts = _format(
        [{"system": "Be brief.", "messages": _MESSAGES, "tools": [_WEATHER]}],
        _SystemWithToolsRejectedTokenizer(),
    )

    assert texts == [_catalog(_WEATHER) + "<user>Weather in Paris?<assistant>It is 21C."]


def test_user_mapping_keeps_the_tools_column_for_training():
    texts = _format(
        [{"question": "Weather in Paris?", "answer": "It is 21C.", "tools": [_WEATHER]}],
        custom_format_mapping = {"question": "user", "answer": "assistant"},
    )

    assert texts == [_catalog(_WEATHER) + "<user>Weather in Paris?<assistant>It is 21C."]


@pytest.mark.parametrize(
    ("metadata", "system"),
    [
        ({"__system_prompt": "Be brief."}, "<system>Be brief."),
        ({"__label_mapping": {}}, ""),
    ],
    ids = ["system_prompt", "label_mapping"],
)
def test_user_mapping_metadata_keeps_the_tools_column_for_training(metadata, system):
    mapping = {"question": "user", "answer": "assistant", **metadata}

    texts = _format(
        [{"question": "Weather in Paris?", "answer": "It is 21C.", "tools": [_WEATHER]}],
        custom_format_mapping = mapping,
    )

    assert texts == [_catalog(_WEATHER) + system + "<user>Weather in Paris?<assistant>It is 21C."]


def test_rows_still_train_when_the_template_rejects_tools():
    texts = _format([{"messages": _MESSAGES, "tools": [_WEATHER]}], _NoToolsTokenizer())

    assert texts == ["<user>Weather in Paris?<assistant>It is 21C."]


@pytest.mark.parametrize(
    "encode",
    [json.dumps, lambda tools: [json.dumps(tool) for tool in tools]],
    ids = ["json_string", "json_string_list"],
)
def test_json_string_tools_keep_their_null_defaults(encode):
    tool = {
        "type": "function",
        "function": {
            "name": "search_docs",
            "parameters": {
                "type": "object",
                "properties": {"lang": {"type": "string", "default": None}},
            },
        },
    }

    texts = _format([{"messages": _MESSAGES, "tools": encode([tool])}])

    assert texts[0].startswith(_catalog(tool))
