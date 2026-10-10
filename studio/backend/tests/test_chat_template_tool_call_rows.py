# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset, format_and_template_dataset
from utils.datasets.chat_templates import keep_renderable_chat_template

_GEMMA4_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "assets" / "chat_templates" / "gemma-4.jinja"
)

_QWEN35_TEMPLATE = """
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + (message.content or '') }}
{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}
{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
{%- for args_name, args_value in tool_call.arguments | items %}
{{- '<parameter=' + args_name + '>\\n' + args_value | string + '\\n</parameter>\\n' }}
{%- endfor %}
{{- '</function>\\n</tool_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
"""

_LLAMA3_TEMPLATE = """
{%- for message in messages %}
{%- if not (message.role == 'tool' or 'tool_calls' in message) %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' }}
{%- elif 'tool_calls' in message %}
{%- if not message.tool_calls|length == 1 %}{{- raise_exception('one tool call per message') }}{%- endif %}
{%- set tool_call = message.tool_calls[0].function %}
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n{"name": "' + tool_call.name + '", "parameters": ' + tool_call.arguments | tojson + '}<|eot_id|>' }}
{%- else %}
{{- '<|start_header_id|>ipython<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' }}
{%- endif %}
{%- endfor %}
"""

_DEEPSEEK_TEMPLATE = """
{%- for message in messages %}
{%- if message['role'] == 'user' %}{{- '<User>' + message['content'] }}{%- endif %}
{%- if message['role'] == 'assistant' and message['content'] is none %}
{%- for tool in message['tool_calls'] %}
{{- '<call>' + tool['function']['name'] + '\\n' + tool['function']['arguments'] + '</call>' }}
{%- endfor %}
{%- endif %}
{%- if message['role'] == 'assistant' and message['content'] is not none %}{{- '<Assistant>' + message['content'] }}{%- endif %}
{%- if message['role'] == 'tool' %}{{- '<output>' + message['content'] }}{%- endif %}
{%- endfor %}
"""


class _JinjaTokenizer:
    eos_token = ""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation,
        tokenize = False,
        add_generation_prompt = False,
        **_kwargs,
    ):
        jinja2 = pytest.importorskip("jinja2")
        sandbox = pytest.importorskip("jinja2.sandbox")

        def _raise(message):
            raise jinja2.exceptions.TemplateError(message)

        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        env.filters["tojson"] = lambda value, **opts: json.dumps(value, **opts)
        env.globals["raise_exception"] = _raise
        return env.from_string(self.chat_template).render(
            messages = conversation,
            add_generation_prompt = add_generation_prompt,
            bos_token = "",
        )


def _tool_call_row(arguments):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "21C", "tool_call_id": "call_0"},
        {"role": "assistant", "content": "It is 21C in Paris."},
    ]


def _plain_row():
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]


def _format(rows, template):
    dataset_info = {
        "dataset": Dataset.from_list([{"messages": row} for row in rows]),
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }
    return apply_chat_template_to_dataset(dataset_info, _JinjaTokenizer(template), num_proc = 1)


def test_json_string_tool_arguments_are_rendered_as_parameters():
    result = _format([_tool_call_row('{"city": "Paris", "unit": null}')], _QWEN35_TEMPLATE)

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert "<parameter=city>\nParis\n</parameter>" in text
    assert "<parameter=unit>\nNone\n</parameter>" in text


def test_gemma4_learns_its_own_tool_call_format():
    result = _format(
        [_tool_call_row('{"city": "Paris"}')], _GEMMA4_TEMPLATE.read_text(encoding = "utf-8")
    )

    assert result["success"] is True
    assert 'call:get_weather{city:<|"|>Paris<|"|>}' in result["dataset"][0]["text"]


def test_tool_call_dataset_with_null_filled_keys_renders_on_llama3():
    rows = [_plain_row(), _tool_call_row('{"city": "Paris"}')]

    result = _format(rows, _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    texts = result["dataset"]["text"]
    assert len(texts) == 2
    assert "<|start_header_id|>assistant<|end_header_id|>\n\nHello!" in texts[0]
    assert '{"name": "get_weather", "parameters": {"city": "Paris"}}' in texts[1]


def test_dict_arguments_do_not_gain_other_tools_null_parameters():
    weather = _tool_call_row({"city": "Paris"})
    search = _tool_call_row({"query": "cats"})
    search[1]["tool_calls"][0]["function"]["name"] = "web_search"

    result = _format([weather, search], _QWEN35_TEMPLATE)

    assert result["success"] is True
    weather_text, search_text = result["dataset"]["text"]
    assert "<parameter=query>" not in weather_text
    assert "<parameter=city>" not in search_text


def test_non_json_tool_arguments_are_kept_verbatim():
    result = _format([_tool_call_row("not json")], _LLAMA3_TEMPLATE)

    assert result["success"] is True
    assert '"parameters": "not json"' in result["dataset"][0]["text"]


def test_tool_call_turn_with_null_content_and_string_arguments_still_renders():
    row = _tool_call_row('{"city": "Paris"}')
    row[1]["content"] = None

    result = _format([row, _plain_row()], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert len(result["dataset"]) == 2
    assert '<call>get_weather\n{"city": "Paris"}</call>' in result["dataset"][0]["text"]


def test_null_content_is_dropped_before_the_row_as_loaded_is_tried():
    template = (
        "{%- for message in messages %}"
        "{%- if message.content is defined %}[{{ message.content }}]{%- else %}-{%- endif %}"
        "{%- endfor %}"
    )
    null_content = _tool_call_row('{"city": "Paris"}')
    null_content[1]["content"] = None

    result = _format([null_content, _tool_call_row('{"city": "Paris"}')], template)

    assert result["dataset"]["text"] == [
        "[Weather in Paris?]-[21C][It is 21C in Paris.]",
        "[Weather in Paris?][][21C][It is 21C in Paris.]",
    ]


def test_dropped_row_reports_the_template_error_of_the_cleaned_row():
    parallel = _tool_call_row('{"city": "Paris"}')
    parallel[1]["tool_calls"].append({**parallel[1]["tool_calls"][0], "id": "call_1"})

    result = _format([_plain_row(), parallel], _LLAMA3_TEMPLATE)

    assert result["success"] is True
    assert "one tool call per message" in result["dropped_rows_warning"]


def test_template_probe_counts_rows_after_cleaning():
    tokenizer = _JinjaTokenizer(_LLAMA3_TEMPLATE)
    dataset = Dataset.from_list(
        [{"messages": _plain_row()}, {"messages": _tool_call_row('{"city": "Paris"}')}]
    )

    note = keep_renderable_chat_template(tokenizer, dataset, "messages", _QWEN35_TEMPLATE)

    assert note is None
    assert tokenizer.chat_template == _LLAMA3_TEMPLATE


def _sharegpt_tool_row(call):
    return {
        "conversations": [
            {"from": "human", "value": "Weather in Paris?"},
            {"from": "function_call", "value": call},
            {"from": "observation", "value": '{"temp": 18}'},
            {"from": "gpt", "value": "It is 18C in Paris."},
        ]
    }


def _format_sharegpt(rows, template):
    return format_and_template_dataset(
        Dataset.from_list(rows),
        model_name = "stub-model",
        tokenizer = _JinjaTokenizer(template),
        num_proc = 1,
    )


def test_sharegpt_function_call_and_observation_train_as_tool_turns():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert (
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        '{"name": "get_weather", "parameters": {"city": "Paris"}}' in text
    )
    assert '<|start_header_id|>ipython<|end_header_id|>\n\n{"temp": 18}' in text
    assert "function_call" not in text
    assert "observation" not in text


def test_sharegpt_function_call_list_trains_every_call():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": '{"city": "Rome"}'},
        ]
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert "<|im_start|>assistant\n<tool_call>\n<function=get_weather>" in text
    assert "<parameter=city>\nParis\n</parameter>" in text
    assert "<parameter=city>\nRome\n</parameter>" in text
    assert '<|im_start|>tool\n{"temp": 18}' in text


def test_sharegpt_function_call_that_is_not_json_is_kept_as_written():
    result = _format_sharegpt([_sharegpt_tool_row("get_weather(Paris)")], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert "<|im_start|>function_call\nget_weather(Paris)" in result["dataset"][0]["text"]


def test_sharegpt_function_call_is_kept_when_the_template_ignores_tool_calls():
    plain_chatml = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n'"
        " + message['content'] + '<|im_end|>\\n'}}{% endfor %}"
    )
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], plain_chatml)

    assert result["success"] is True, result["errors"]
    assert f"<|im_start|>function_call\n{call}" in result["dataset"][0]["text"]


def test_sharegpt_function_call_renders_on_a_template_gating_calls_on_null_content():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '<call>get_weather\n{"city": "Paris"}</call>' in text
    assert '<output>{"temp": 18}' in text


def test_sharegpt_call_name_in_the_prompt_does_not_hide_an_ignored_call():
    plain_chatml = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n'"
        " + message['content'] + '<|im_end|>\\n'}}{% endfor %}"
    )
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    row = _sharegpt_tool_row(call)
    row["conversations"][0]["value"] = "Use get_weather for Paris."

    result = _format_sharegpt([row], plain_chatml)

    assert f"<|im_start|>function_call\n{call}" in result["dataset"][0]["text"]


def test_sharegpt_function_call_keeps_explicit_null_arguments():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris", "unit": None}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert '"parameters": {"city": "Paris", "unit": null}' in result["dataset"][0]["text"]


def test_sharegpt_parallel_calls_are_split_for_one_call_templates():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_time", "arguments": {"city": "Rome"}},
        ]
    )

    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"time": "noon"}'})

    result = _format_sharegpt([row], _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    paris = text.index('{"name": "get_weather", "parameters": {"city": "Paris"}}')
    rome = text.index('{"name": "get_time", "parameters": {"city": "Rome"}}')
    assert paris < text.index('{"temp": 18}') < rome < text.index('{"time": "noon"}')
    assert "function_call" not in text


def test_sharegpt_parallel_calls_sharing_one_result_are_not_reordered():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_time", "arguments": {"city": "Rome"}},
        ]
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _LLAMA3_TEMPLATE)

    text = result["dataset"][0]["text"]
    assert text.index("Rome") < text.index('{"temp": 18}')


def test_sharegpt_repeated_call_name_is_not_lost_on_a_first_call_only_template():
    first_call_only = (
        "{%- for message in messages %}"
        "{%- if message.tool_calls %}{%- set call = message.tool_calls[0].function %}"
        "{{- '<call ' + call.name + '>' + call.arguments + '</call>' }}"
        "{%- elif message.role == 'tool' %}{{- '<result from=get_weather>' + message.content }}"
        "{%- else %}{{- '<' + message.role + '>' + message.content }}{%- endif %}"
        "{%- endfor %}"
    )
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"temp": 21}'})

    result = _format_sharegpt([row], first_call_only)

    assert result["success"] is True, result["errors"]
    assert "Rome" in result["dataset"][0]["text"]
