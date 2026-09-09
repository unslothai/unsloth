# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment

from core.inference.template_capabilities import template_supports_tools


@pytest.mark.parametrize(
    "template",
    [
        "{# message.tool_calls are unsupported #}{{ message.content }}",
        "{{ 'message.tool_calls are unsupported' }}",
        "{{ \"message['tool_calls'] is unsupported\" }}",
        "{{ '{% if tools %}example{% endif %}' }}",
        "{# {% if tools %}example{% endif %} #}",
        "{% raw %}{% if tools %}example{% endif %}{% endraw %}",
        "{% if not tools %}plain{% endif %}",
        "{% if tools is none %}plain{% endif %}",
        "{% if tools is undefined %}plain{% endif %}",
        "{{ not tools }}",
        "{{ tools is not none }}",
        "{{ tools | length }}",
        "{% if tools %}{{ tools is defined }}{% endif %}",
        "{% if tools is not none %}{{ raise_exception('tools unsupported') }}{% endif %}",
        "{% if tools %}{{ raise_exception('unsupported: ' ~ tools) }}{% endif %}",
        "{% if message.role == 'tool' %}{{ raise_exception('unsupported role') }}{% endif %}",
        "{% if message.role != 'tool' %}{{ message.content }}{% endif %}",
        "{% if not message.tool_calls %}{{ message.content }}{% endif %}",
        "{% if builtin_tools or tools_in_user_message %}plain{% endif %}",
        "{% set unused = tools %}{{ message.content }}",
        "{% if tools %}",
    ],
)
def test_non_tool_templates_stay_disabled(template):
    assert template_supports_tools(template) is False


@pytest.mark.parametrize(
    "template",
    [
        "{{ tools | tojson }}",
        "{% set available_tools = tools %}{{ available_tools | tojson }}",
        "{% if tools and not available_tools %}{% set available_tools = tools %}"
        "{% endif %}{% if available_tools %}{{ available_tools | tojson }}{% endif %}",
        "{% for tool in tools %}{{ tool | tojson }}{% endfor %}",
        "{% if tools is not none %}{{ tools | tojson }}{% endif %}",
        "{% if documents %}docs{% elif tools %}{{ tools | tojson }}{% endif %}",
        "{%+ if tools +%}{{ tools | tojson }}{% endif %}",
        "{% if message.role == 'tool' %}{{ message.content }}{% endif %}",
        "{% if 'tool' == message['role'] %}{{ message.content }}{% endif %}",
        "{% if message.tool_calls %}{{ message.tool_calls | tojson }}{% endif %}",
        "{% if tool_calls is defined %}{{ tool_calls | tojson }}{% endif %}",
        "{% generation %}{{ message.tool_calls | tojson }}{% endgeneration %}",
        "{% for message in messages %}{% if loop.index > 1 %}{% break %}{% endif %}"
        "{{ message.tool_calls | tojson }}{% endfor %}",
        "{{ '{% raw %}' }}{{ tools | tojson }}{{ '{% endraw %}' }}",
    ],
)
def test_executable_tool_templates_are_detected(template):
    assert template_supports_tools(template) is True


def _published_template(name):
    return (Path(__file__).parent / "data" / "chat_templates" / f"{name}.jinja").read_text(
        encoding = "utf-8"
    )


@pytest.mark.parametrize("name, detected", [("granite-3.3", True), ("phi-4-mini", False)])
def test_detection_does_not_bypass_the_safetensors_parser_gate(name, detected):
    from routes.inference import _detect_safetensors_features

    template = _published_template(name)
    assert template_supports_tools(template) is detected
    flags = _detect_safetensors_features(SimpleNamespace(active_model_name = name), template)
    assert flags["supports_tools"] is False


@pytest.mark.parametrize("name, emits_schema", [("granite-3.3", True), ("phi-4-mini", False)])
def test_published_templates_render_studios_tool_argument(name, emits_schema):
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    class Tokenizer:
        chat_template = _published_template(name)

        def apply_chat_template(self, messages, **kwargs):
            return (
                Environment()
                .from_string(self.chat_template)
                .render(messages = messages, eos_token = "", **kwargs)
            )

    tokenizer = Tokenizer()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    with_tools = apply_chat_template_for_generation(tokenizer, messages, tools = tools)
    without_tools = apply_chat_template_for_generation(tokenizer, messages)
    assert ("get_weather" in with_tools) is emits_schema
    assert (with_tools != without_tools) is emits_schema


@pytest.mark.parametrize(
    "template, expected",
    [
        (
            "{% set ns = namespace(available_tools=[]) %}"
            "{% if tools %}{% set ns = namespace(available_tools=tools) %}{% endif %}"
            "{{ ns.available_tools | tojson }}",
            True,
        ),
        (
            "{% set ns = namespace(available_tools=[]) %}"
            "{% if tools %}{% set ns.available_tools = tools %}{% endif %}"
            "{{ ns.available_tools | tojson }}",
            True,
        ),
        (
            "{% set ns = namespace(available_tools=[]) %}"
            "{% set ns.available_tools = tools %}"
            "{% set catalog = ns['available_tools'] %}{{ catalog | tojson }}",
            True,
        ),
        (
            "{% set ns = namespace(available_tools=[]) %}"
            "{% set other = namespace(available_tools=[]) %}"
            "{% set ns.available_tools = tools %}{{ other.available_tools | tojson }}",
            False,
        ),
        (
            "{% macro format_metadata(tools) %}{{ tools | tojson }}{% endmacro %}"
            "{{ message.content }}",
            False,
        ),
        (
            "{% macro format_metadata(tools) %}{{ tools | tojson }}{% endmacro %}"
            "{{ format_metadata([]) }}",
            False,
        ),
        (
            "{% macro format_metadata(tools) %}{{ tools | tojson }}{% endmacro %}"
            "{{ format_metadata(tools) }}",
            True,
        ),
        (
            "{% macro format_metadata(catalog) %}{{ catalog | tojson }}{% endmacro %}"
            "{{ format_metadata(catalog=tools) }}",
            True,
        ),
        (
            "{% macro format_metadata(tools=[]) %}{{ tools | tojson }}{% endmacro %}"
            "{{ format_metadata() }}",
            False,
        ),
        (
            "{% macro format_metadata(catalog=tools) %}{{ catalog | tojson }}{% endmacro %}"
            "{{ format_metadata() }}",
            True,
        ),
        (
            "{% macro format_metadata(catalog, selected=catalog) %}"
            "{{ selected | tojson }}{% endmacro %}{{ format_metadata(tools) }}",
            True,
        ),
        (
            "{% set ns = namespace(available_tools=tools) %}"
            "{% macro format_metadata(data) %}{{ data.available_tools | tojson }}{% endmacro %}"
            "{{ format_metadata(ns) }}",
            True,
        ),
        (
            "{% macro format_metadata() %}{{ tools | tojson }}{% endmacro %}"
            "{{ format_metadata() }}",
            True,
        ),
        (
            "{% macro format_metadata(tools) %}plain{% endmacro %}{{ format_metadata(tools) }}",
            False,
        ),
        (
            "{% if tools %}{% macro unused(tools) %}{{ tools | tojson }}"
            "{% endmacro %}{% endif %}{{ message.content }}",
            False,
        ),
        (
            "{% macro unused() %}{% set catalog = tools %}{% endmacro %}"
            "{{ catalog | default('plain') }}",
            False,
        ),
    ],
)
def test_alias_and_macro_detection_matches_rendered_catalog(template, expected):
    _assert_catalog_rendering(template, expected)


def _assert_catalog_rendering(template, expected):
    from core.inference.llama_cpp import detect_reasoning_flags

    render = Environment().from_string(template)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    common = {"message": {"role": "user", "content": "Hi"}}
    with_tools = render.render(tools = tools, **common)
    without_tools = render.render(tools = [], **common)
    assert ("get_weather" in with_tools) is expected
    if expected:
        assert with_tools != without_tools
    assert template_supports_tools(template) is expected
    assert detect_reasoning_flags(template)["supports_tools"] is expected


@pytest.mark.parametrize(
    "template, expected",
    [
        ("{% set catalog=[] %}{{ catalog|tojson }}{% set catalog=tools %}", False),
        ("{% set catalog=tools %}{% set catalog=[] %}{{ catalog|tojson }}", False),
        ("{% set catalog=tools %}{% set catalog=catalog|list %}{{ catalog|tojson }}", True),
        ("{% set catalog=tools|length %}{{ catalog }}", False),
        ("{% set tools=[] %}{{ tools|tojson }}", False),
        (
            "{% if tools %}{% set catalog=tools|list %}{% endif %}"
            "{{ catalog|default([])|tojson }}",
            True,
        ),
        ("{% set catalog=tools|default([])|list %}{{ catalog|tojson }}", True),
        ("{% set catalog=tools or [] %}{{ catalog|tojson }}", True),
        (
            "{% set catalog=tools %}{% if tools %}{% set catalog=[] %}"
            "{% else %}{% set catalog=[] %}{% endif %}{{ catalog|tojson }}",
            False,
        ),
        (
            "{% set catalog=[] %}{% with catalog=tools, output=catalog %}"
            "{{ output|tojson }}{% endwith %}",
            False,
        ),
        ("{% set catalog %}{{ tools|tojson }}{% endset %}{{ catalog }}", True),
        ("{% set catalog %}{{ tools|tojson }}{% endset %}", False),
        ("{% set catalog=[] if tools else [] %}{{ catalog|tojson }}", False),
        ("{% set catalog=tools and [] %}{{ catalog|tojson }}", False),
        (
            "{% set ns=namespace(catalog=tools) %}{% set ns.catalog=[] %}"
            "{{ ns.catalog|tojson }}",
            False,
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% set ns=namespace(catalog=[]) %}"
            "{{ ns.catalog|tojson }}",
            False,
        ),
        (
            "{% set catalog=tools %}{% if true %}{% set catalog=[] %}{% endif %}"
            "{{ catalog|tojson }}",
            False,
        ),
        (
            "{% set catalog=tools %}{% if false %}{% set catalog=[] %}{% endif %}"
            "{{ catalog|tojson }}",
            True,
        ),
        (
            "{% set catalog=[] %}{% macro show() %}{{ catalog|tojson }}{% endmacro %}"
            "{{ show() }}{% set catalog=tools %}",
            False,
        ),
        (
            "{% set catalog=[] %}{% macro show() %}{{ catalog|tojson }}{% endmacro %}"
            "{% set catalog=tools %}{{ show() }}",
            True,
        ),
        (
            "{% set catalog=[] %}{% for item in tools %}{% set catalog=item %}{% endfor %}"
            "{{ catalog|tojson }}",
            False,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}"
            "{% for item in tools %}{% set ns.catalog=item %}{% endfor %}"
            "{{ ns.catalog|tojson }}",
            True,
        ),
    ],
)
def test_alias_assignment_flow_matches_rendered_catalog(template, expected):
    _assert_catalog_rendering(template, expected)


@pytest.mark.parametrize(
    "template, schema_flags",
    [
        ("{% if tools and false %}tool instructions{% endif %}", ()),
        ("{% if false and tools %}tool instructions{% endif %}", ()),
        ("{% if tools and (false or false) %}tool instructions{% endif %}", ()),
        ("{% if tools and 1 == 2 %}tool instructions{% endif %}", ()),
        ("{% if tools or true %}plain{% endif %}", ()),
        (
            "{% set flag=true %}{% if tools or flag %}plain{% endif %}",
            (),
        ),
        (
            "{% if tools and false %}plain{% elif tools %}{{ tools|tojson }}{% endif %}",
            (False, True),
        ),
        (
            "{% if flag %}{% set catalog=tools %}{% endif %}"
            "{% if not flag %}{{ catalog|default([])|tojson }}{% endif %}",
            (),
        ),
        (
            "{% if flag %}{% set catalog=tools %}{% endif %}"
            "{% if flag %}{{ catalog|default([])|tojson }}{% endif %}",
            (True,),
        ),
        (
            "{% if not flag %}{% set catalog=tools %}{% endif %}"
            "{% if flag %}{{ catalog|default([])|tojson }}{% endif %}",
            (),
        ),
        (
            "{% if flag %}{% set catalog=tools %}{% endif %}{% set flag=false %}"
            "{% if not flag %}{{ catalog|default([])|tojson }}{% endif %}",
            (True,),
        ),
        (
            "{% if flag and other %}{% set catalog=tools %}{% endif %}"
            "{% if not flag or not other %}{{ catalog|default([])|tojson }}{% endif %}",
            (),
        ),
        (
            "{% if flag or other %}{% set catalog=tools %}{% endif %}"
            "{% if not flag and not other %}{{ catalog|default([])|tojson }}{% endif %}",
            (),
        ),
        (
            "{% set catalog=[] %}{% if tools %}{% do catalog.extend(tools) %}"
            "{% endif %}{{ catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set catalog=[] %}{% do catalog.append(tools[0]) %}{{ catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set catalog=[] %}{% do catalog.extend(tools) %}"
            "{% do catalog.clear() %}{{ catalog|tojson }}",
            (),
        ),
        (
            "{% set catalog=[] %}{% if flag %}{% do catalog.extend(tools) %}{% endif %}"
            "{% if not flag %}{{ catalog|tojson }}{% endif %}",
            (),
        ),
        (
            "{% set catalog=[] %}{% for tool in tools %}{% do catalog.append(tool) %}"
            "{% endfor %}{{ catalog|tojson }}",
            (False, True),
        ),
        (
            "{% if tools %}{% set ns=namespace({'catalog': tools}) %}{% endif %}"
            "{{ ns.catalog|default([])|tojson }}",
            (False, True),
        ),
        (
            "{% set mapping={'catalog': tools, 'label': 'plain'} %}"
            "{% set ns=namespace(mapping) %}{{ ns.catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set ns=namespace({'catalog': tools}, catalog=[]) %}{{ ns.catalog|tojson }}",
            (),
        ),
        (
            "{% set ns=namespace({'catalog': tools, 'label': 'plain'}) %}{{ ns.label }}",
            (),
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% for x in [1] %}"
            "{% set ns.catalog=[] %}{% endfor %}{{ ns.catalog|tojson }}",
            (),
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% for x in [] %}"
            "{% set ns.catalog=[] %}{% endfor %}{{ ns.catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% for x in [1] if false %}"
            "{% set ns.catalog=[] %}{% endfor %}{{ ns.catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% for x in [1] %}"
            "{% set ns=namespace(catalog=[]) %}{% set ns.catalog=[] %}"
            "{% endfor %}{{ ns.catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set ns=namespace(catalog=tools) %}{% for value in [tools, []] %}"
            "{% set ns.catalog=value %}{% endfor %}{{ ns.catalog|tojson }}",
            (),
        ),
        (
            "{% set catalog=[] %}{% if tools %}{% set catalog, ignored=tools, none %}"
            "{% endif %}{{ catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set catalog=tools %}{% set catalog, ignored=[], tools %}{{ catalog|tojson }}",
            (),
        ),
        (
            "{% set catalog=tools %}{% set ignored=[] %}"
            "{% set ignored, catalog=catalog, ignored %}{{ catalog|tojson }}",
            (),
        ),
        (
            "{% set pair=[tools, 'plain'] %}{% set catalog, label=pair %}{{ catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set wrapper={'catalog': tools, 'label': 'plain'} %}{{ wrapper['label'] }}",
            (),
        ),
        (
            "{% set wrapper={'catalog': tools, 'label': 'plain'} %}{{ wrapper.catalog|tojson }}",
            (False, True),
        ),
        (
            "{% set wrapper={'inner': {'catalog': tools, 'label': 'plain'}} %}"
            "{{ wrapper.inner.label }}",
            (),
        ),
        ("{% set wrapper=[tools, 'plain'] %}{{ wrapper[1] }}", ()),
        ("{% set wrapper=[tools, 'plain'] %}{{ wrapper[0]|tojson }}", (False, True)),
        (
            "{% set ns=namespace(catalog=[]) %}{% with %}{% if flag %}"
            "{% set ns.catalog=tools %}{% endif %}{% endwith %}"
            "{% if not flag %}{{ ns.catalog|tojson }}{% endif %}",
            (),
        ),
    ],
)
def test_reviewed_paths_match_rendered_catalog(template, schema_flags):
    from core.inference.llama_cpp import detect_reasoning_flags

    render = Environment(extensions = ["jinja2.ext.do"]).from_string(template)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    for flag in (False, True):
        for other in (False, True):
            output = render.render(tools = tools, flag = flag, other = other)
            assert ("get_weather" in output) is (flag in schema_flags)
    expected = bool(schema_flags)
    assert template_supports_tools(template) is expected
    assert detect_reasoning_flags(template)["supports_tools"] is expected


def _pathological(label):
    """Built here rather than parametrized: pytest puts the node id in
    PYTEST_CURRENT_TEST, and Windows caps an environment variable at 32767
    characters, so a 70 KB template in the id errors the test at setup."""
    if label == "nested_if":
        # Jinja parses by recursive descent, so nesting alone exhausts the stack
        # before this module ever sees a tree.
        return "{% if tools %}" * 5000 + "{{ tools|tojson }}" + "{% endif %}" * 5000
    if label == "parenthesised_guard":
        return "{% if " + "(" * 200 + "tools" + ")" * 200 + " %}{{ tools|tojson }}{% endif %}"
    if label == "wide_or":
        # A wide boolean guard recurses in the node repr used as a condition-fact key.
        return (
            "{% if " + " or ".join(f"v{i}" for i in range(2000)) + " %}"
            "{{ tools|tojson }}{% endif %}"
        )
    # A long attribute chain recurses in _value_aliases.
    return "{{ tools" + ".a" * 3000 + " }}"


@pytest.mark.parametrize("label", ["nested_if", "parenthesised_guard", "wide_or", "deep_attribute"])
def test_pathological_templates_disable_tools_instead_of_raising(label):
    """detect_reasoning_flags runs on the GGUF metadata read and the llama-server launch,
    so a template too deep to analyse must leave tools off rather than break the load.
    The substring scan this replaced could not raise at all."""
    from core.inference.llama_cpp import detect_reasoning_flags

    template = _pathological(label)
    assert template_supports_tools(template) is False
    assert detect_reasoning_flags(template, f"vendor/{label}")["supports_tools"] is False


_TOOL_BODY = "{% if tools %}{{ tools|tojson }}{% endif %}"


@pytest.mark.parametrize(
    "template",
    [
        # A Hugging Face named-template map and the Hermes-3 list form: both are
        # unhashable, so lru_cache would raise on them before any fail-closed branch.
        {"default": _TOOL_BODY, "tool_use": "x"},
        [{"name": "default", "template": _TOOL_BODY}],
        # Hashable but not a template: `"tool" not in template` would raise instead.
        _TOOL_BODY.encode(),
        None,
        object(),
    ],
)
def test_non_string_templates_are_turned_away_before_the_cache(template):
    assert template_supports_tools(template) is False


@pytest.mark.parametrize(
    "template",
    [{"default": _TOOL_BODY, "tool_use": "x"}, [{"name": "default", "template": _TOOL_BODY}]],
)
def test_named_template_containers_reach_the_classifier_without_raising(template):
    """routes.inference passes the raw chat_template through when template selection
    yields nothing, so a named-template map or list reaches detect_reasoning_flags."""
    from core.inference.llama_cpp import detect_reasoning_flags
    assert detect_reasoning_flags(template, "vendor/named")["supports_tools"] is False


@pytest.mark.parametrize(
    "template, expected",
    [
        # A filter can reject every item of a literal iterable, so the body never runs
        # and Jinja takes the else.
        (
            "{% if tools %}{% for x in [1] if false %}x{% else %}{{ tools|tojson }}"
            "{% endfor %}{% endif %}",
            True,
        ),
        ("{% for x in [1] if true %}{{ tools|tojson }}{% else %}plain{% endfor %}", True),
        # break and continue end the path: nothing after them in the body runs.
        ("{% for x in [1] %}{% break %}{{ tools|tojson }}{% endfor %}", False),
        ("{% for x in [1] %}{% continue %}{{ tools|tojson }}{% endfor %}", False),
        ("{% for x in [1] %}{{ tools|tojson }}{% break %}{% endfor %}", True),
        (
            "{% set catalog=[] %}{% set alias=catalog %}{% do alias.extend(tools) %}"
            "{% do catalog.clear() %}{{ catalog|tojson }}",
            False,
        ),
        # A rebind breaks the sharing.
        (
            "{% set catalog=[] %}{% set alias=catalog %}{% set alias=[] %}"
            "{% do alias.extend(tools) %}{{ catalog|tojson }}",
            False,
        ),
        # A field named tool_calls on an object the template built itself proves
        # nothing; on an untracked message value it still does.
        ("{% set ns=namespace(tool_calls='plain') %}{{ ns.tool_calls }}", False),
        ("{% set holder={'tool_calls': 'plain'} %}{{ holder.tool_calls }}", False),
        ("{{ message.tool_calls|tojson }}", True),
        ("{% set ns=namespace(role='tool') %}{% if ns.role == 'tool' %}plain{% endif %}", False),
        ("{% if message.role == 'tool' %}{{ message.content }}{% endif %}", True),
        # A subscript whose key is a name bound to a constant selects one field.
        (
            "{% set key='label' %}{% set wrapper={'catalog':tools,'label':'plain'} %}"
            "{{ wrapper[key] }}",
            False,
        ),
        (
            "{% set key='catalog' %}{% set wrapper={'catalog':tools,'label':'plain'} %}"
            "{{ wrapper[key]|tojson }}",
            True,
        ),
        # {% call %} renders through the macro, not through the caller block.
        (
            "{% macro render(catalog, caller=None) %}{{ catalog|tojson }}{% endmacro %}"
            "{% if tools %}{% call render(tools) %}{% endcall %}{% endif %}",
            True,
        ),
        (
            "{% macro render(catalog, caller=None) %}plain{% endmacro %}"
            "{% if tools %}{% call render(tools) %}{% endcall %}{% endif %}",
            False,
        ),
    ],
)
def test_reviewed_round_seven_paths_match_rendered_catalog(template, expected):
    from core.inference.llama_cpp import detect_reasoning_flags

    render = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"]).from_string(
        template
    )
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    output = render.render(
        tools = tools,
        messages = [{"role": "user", "content": "Hi"}],
        message = {"role": "tool", "content": "get_weather said sunny", "tool_calls": tools},
    )
    assert ("get_weather" in output) is expected
    assert template_supports_tools(template) is expected
    assert detect_reasoning_flags(template)["supports_tools"] is expected


@pytest.mark.parametrize(
    "template, expected",
    [
        # `loop.first` reprs identically in every loop, so an outer loop's condition
        # facts must not prune a branch of a nested one.
        (
            "{% for m in messages %}{% if loop.first %}{% for t in tools %}"
            "{% if not loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}"
            "{% endif %}{% endfor %}",
            True,
        ),
        (
            "{% for m in messages %}{% if loop.first %}{% for t in tools %}"
            "{% if loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}"
            "{% endif %}{% endfor %}",
            True,
        ),
        # A template without the do extension mutates through `{% set _ = ... %}`.
        ("{% set catalog=[] %}{% set _x = catalog.append(tools) %}{{ catalog|tojson }}", True),
        # Methods outside append/extend/clear still move tool data into the receiver.
        ("{% set catalog=[] %}{% do catalog.insert(0, tools) %}{{ catalog|tojson }}", True),
        ("{% set d={} %}{% do d.update({'c': tools}) %}{{ d.c|tojson }}", True),
        # An unrecognised method handed nothing tool-shaped still changes nothing.
        ("{% set catalog=[] %}{% do catalog.insert(0, 'plain') %}{{ catalog|tojson }}", False),
    ],
)
def test_loop_facts_and_mutation_shapes_match_rendered_catalog(template, expected):
    render = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"]).from_string(
        template
    )
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {}}},
    ]
    output = render.render(
        tools = tools,
        messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    )
    assert ("get_weather" in output) is expected
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    "template, expected",
    [
        # Rebinding the root detaches it in both directions: the earlier alias still
        # refers to the old container, so the new one stays empty.
        ("{% set a=[] %}{% set b=a %}{% set a=[] %}{% do b.extend(tools) %}{{ a|tojson }}", False),
        # A caller block runs only if the macro invokes caller().
        (
            "{% macro render(caller=None) %}plain{% endmacro %}"
            "{% call render() %}{{ tools|tojson }}{% endcall %}",
            False,
        ),
        (
            "{% macro render(caller=None) %}{{ caller() }}{% endmacro %}"
            "{% call render() %}{{ tools|tojson }}{% endcall %}",
            True,
        ),
        # A keyword argument names the field its value lands under.
        (
            "{% set d={} %}{% if tools %}{% do d.update(catalog=tools) %}{% endif %}"
            "{{ d.catalog|tojson }}",
            True,
        ),
        (
            "{% set d={} %}{% do d.update(catalog=tools) %}{{ d.label|default('x') }}",
            False,
        ),
        # A destructive call takes the provenance back out with the value.
        (
            "{% set catalog={'schema':tools,'label':'plain'} %}{% do catalog.pop('schema') %}"
            "{{ catalog|tojson }}",
            False,
        ),
        (
            "{% set catalog={'schema':tools,'label':'plain'} %}{% do catalog.pop('label') %}"
            "{{ catalog|tojson }}",
            True,
        ),
        # break and continue stop the scan but keep what the path already mutated.
        (
            "{% set ns=namespace(catalog=[]) %}{% for x in [1] %}{% if tools %}"
            "{% set ns.catalog=tools %}{% endif %}{% continue %}{% endfor %}"
            "{{ ns.catalog|tojson }}",
            True,
        ),
        # A filter block that reduces its input leaves no schema behind.
        ("{% filter first %}{{ tools|tojson }}{% endfilter %}", False),
        ("{% filter upper %}{{ tools|tojson }}{% endfilter %}", True),
        # A macro declaration binds its name, shadowing the catalog.
        ("{% macro tools() %}plain{% endmacro %}{{ tools }}", False),
    ],
)
def test_round_nine_paths_match_rendered_catalog(template, expected):
    render = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"]).from_string(
        template
    )
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    output = render.render(tools = tools, messages = [{"role": "user", "content": "hi"}])
    assert ("get_weather" in output.lower() or "GET_WEATHER" in output) is expected
    assert template_supports_tools(template) is expected


def test_an_unresolved_subscript_key_still_selects_every_field():
    """The constant-key resolution narrows a subscript only when the key is known.
    An unknown key has to keep selecting every field, or a catalog reached through a
    computed key would be missed."""
    template = "{% set wrapper={'catalog':tools,'label':'plain'} %}{{ wrapper[key]|tojson }}"
    assert template_supports_tools(template) is True


@pytest.mark.parametrize(
    "template",
    [
        "{% set catalog=[] %}{% set alias=catalog %}{% do alias.extend(tools) %}"
        "{{ catalog|tojson }}",
        "{% set catalog=[] %}{% set alias=catalog %}{% do catalog.extend(tools) %}"
        "{{ alias|tojson }}",
        "{% set ns=namespace(catalog=[]) %}{% set alias=ns.catalog %}"
        "{% do alias.extend(tools) %}{{ ns.catalog|tojson }}",
        "{% set a=[] %}{% set b=a %}{% do b.extend(tools) %}{{ a|tojson }}",
    ],
)
def test_mutation_through_an_alias_is_a_known_under_approximation(template):
    """These templates DO render the catalog. State is keyed by name, so a mutation
    made through one name is not seen through another bound to the same container,
    and detection stays off.

    This is deliberate. An identity map was tried and withdrawn: it needs the whole
    state to be keyed by object rather than by name, and the partial version produced
    three further divergences of its own. Erring off costs a template that already
    renders its catalog, which is the safe direction, and no published template does
    this: it appears in none of the 120 checked from 106 repositories."""
    render = Environment(extensions = ["jinja2.ext.do"]).from_string(template)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    assert "get_weather" in render.render(tools = tools)
    assert template_supports_tools(template) is False


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # The else arm of a negated guard is reached exactly when the catalog is
        # present, so it has to read the same as the positive spelling.
        ("{% if not tools %}plain{% else %}You may call tools.{% endif %}", True),
        ("{% if tools is not defined %}plain{% else %}You may call tools.{% endif %}", True),
        ("{% if tools is none %}plain{% else %}You may call tools.{% endif %}", True),
        # An elif makes the else reachable for more than one reason, so the guard
        # does not carry across it.
        ("{% if not tools %}plain{% elif other %}x{% else %}You may call tools.{% endif %}", False),
        # A negated guard on something else is not a tool guard.
        ("{% if not messages %}plain{% else %}nothing here{% endif %}", False),
    ],
)
def test_negated_tool_guard_matches_its_positive_spelling(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3968600730: a positional mapping handed to update keeps its own keys.
        ("{% set d={} %}{% do d.update({'catalog': tools}) %}{{ d.label|default('x') }}", False),
        ("{% set d={} %}{% do d.update({'catalog': tools}) %}{{ d.catalog|tojson }}", True),
        # 3968600737: a literal that always passes the filter never takes the else.
        ("{% for x in [1] if true %}plain{% else %}{{ tools|tojson }}{% endfor %}", False),
        ("{% for x in [1] if flag %}plain{% else %}{{ tools|tojson }}{% endfor %}", True),
        ("{% for x in [] %}plain{% else %}{{ tools|tojson }}{% endfor %}", True),
        # 3968600770: continue ends the iteration, not the loop.
        (
            "{% for value in [false, tools] %}{% if not value %}{% continue %}{% endif %}"
            "{{ value|tojson }}{% endfor %}",
            True,
        ),
        ("{% for m in messages %}{% break %}{{ tools|tojson }}{% endfor %}", False),
        (
            "{% set ns=namespace(catalog=[]) %}{% for x in [tools, []] %}{% set ns.catalog=x %}"
            "{% break %}{% endfor %}{{ ns.catalog|tojson }}",
            True,
        ),
        # 3968600779: an alias inherits every constructed descendant, not just the root.
        (
            "{% set wrapper={'message':{'role':'tool','content':'plain'}} %}"
            "{% set current=wrapper %}{% if current.message.role == 'tool' %}"
            "{{ current.message.content }}{% endif %}",
            False,
        ),
        (
            "{% set current=messages[0] %}{% if current.role == 'tool' %}"
            "{{ current.content }}{% endif %}",
            True,
        ),
        # 3968600789: a namespace write inside a macro escapes it.
        (
            "{% set ns=namespace(catalog=[]) %}{% macro load() %}{% set ns.catalog=tools %}"
            "{% endmacro %}{{ load() }}{{ ns.catalog|tojson }}",
            True,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}{% macro load(c) %}{% set ns.catalog=c %}"
            "{% endmacro %}{% if tools %}{% set _=load(tools) %}{% endif %}{{ ns.catalog|tojson }}",
            True,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}{% macro noop() %}plain{% endmacro %}"
            "{{ noop() }}{{ ns.catalog|tojson }}",
            False,
        ),
        # 3968600805: dict() keeps each keyword's field path, as namespace() does.
        ("{% set wrapper=dict(catalog=tools, label='plain') %}{{ wrapper.label }}", False),
        ("{% set wrapper=dict(catalog=tools) %}{{ wrapper.catalog|tojson }}", True),
        # 3968600814: a shallow copy is the receiver again.
        (
            "{% set wrapper={'catalog':tools} %}{% set clone=wrapper.copy() %}"
            "{{ clone.catalog|tojson }}",
            True,
        ),
        (
            "{% set wrapper={'catalog':tools} %}{% set clone=wrapper.copy() %}"
            "{{ clone.label|default('x') }}",
            False,
        ),
        # 3968600825: fold comparisons against names bound to a literal.
        ("{% set flag=true %}{% if flag == false %}{{ tools|tojson }}{% endif %}", False),
        ("{% set flag=true %}{% if flag == true %}{{ tools|tojson }}{% endif %}", True),
        ("{% set n=3 %}{% if n > 5 %}{{ tools|tojson }}{% endif %}", False),
        ("{% set n=3 %}{% if n < 5 %}{{ tools|tojson }}{% endif %}", True),
        ("{% if flag == false %}{{ tools|tojson }}{% endif %}", True),
        # 3968600841: the other spellings of a non-empty catalog guard.
        ("{% if tools != [] %}You may use tools.{% endif %}", True),
        ("{% if tools != none %}You may use tools.{% endif %}", True),
        ("{% if tools|length > 0 %}You may use tools.{% endif %}", True),
        ("{% if tools|length >= 1 %}You may use tools.{% endif %}", True),
        ("{% if 0 < tools|length %}You may use tools.{% endif %}", True),
        ("{% if tools == [] %}plain{% else %}You may use tools.{% endif %}", True),
        ("{% if messages != [] %}nothing here{% endif %}", False),
        ("{% if tools|length > 3 %}nothing here{% endif %}", False),
        ("{% if tools == [] %}nothing here{% endif %}", False),
    ],
)
def test_round_eleven_paths(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize("blocks", [12, 40, 200])
def test_unrelated_conditions_do_not_exhaust_the_budget(blocks):
    """Paths that differ only in a fact nothing reads again are merged.

    Without that a template pays the full Cartesian product of its conditions, and
    the budget catch turns a tool-capable template off. Qwen3-Coder spends a large
    share of the budget on its own, so the headroom is not theoretical.
    """
    template = (
        "".join("{%% if flag%d %%}plain{%% endif %%}" % index for index in range(blocks))
        + "{% if tools %}{{ tools|tojson }}{% endif %}"
    )
    assert template_supports_tools(template) is True


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3969190325: every feasible macro outcome survives, not just the last scanned.
        (
            "{% macro load() %}{% if tools %}{% set ns.catalog=tools %}"
            "{% else %}{% set ns.catalog=[] %}{% endif %}{% endmacro %}"
            "{% set ns=namespace(catalog=[]) %}{{ load() }}{{ ns.catalog|tojson }}",
            True,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}{% macro noop() %}plain{% endmacro %}"
            "{{ noop() }}{{ ns.catalog|tojson }}",
            False,
        ),
        # 3969190330: rebinding a root makes its members external again.
        (
            "{% set wrapper={'message':{'role':'plain'}} %}{% set wrapper=payload %}"
            "{% if wrapper.message.role == 'tool' %}{{ wrapper.message.content }}{% endif %}",
            True,
        ),
        (
            "{% set wrapper={'message':{'role':'plain'}} %}"
            "{% if wrapper.message.role == 'tool' %}{{ wrapper.message.content }}{% endif %}",
            False,
        ),
        # 3969190339: dict() builds a record just as a literal does.
        (
            "{% set wrapper=dict(role='tool', content='plain') %}"
            "{% if wrapper.role == 'tool' %}{{ wrapper.content }}{% endif %}",
            False,
        ),
        # 3969190349: an update replaces the fields it names.
        (
            "{% set d={'catalog':tools} %}{% do d.update({'catalog':[]}) %}{{ d.catalog|tojson }}",
            False,
        ),
        ("{% set d={'catalog':tools} %}{% do d.update(catalog=[]) %}{{ d.catalog|tojson }}", False),
        (
            "{% set d={'catalog':tools,'other':tools} %}{% do d.update({'catalog':[]}) %}"
            "{{ d.other|tojson }}",
            True,
        ),
        # 3969190360: a literal iterable knows which iteration is first and last.
        (
            "{% for x in [1] %}{% if not loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}",
            False,
        ),
        ("{% for x in [1] %}{% if loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}", True),
        (
            "{% for x in [1,2] %}{% if not loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}",
            True,
        ),
        (
            "{% for m in messages %}{% if not loop.first %}{{ tools|tojson }}{% endif %}{% endfor %}",
            True,
        ),
        # 3969190379: in-place mutators render None, so the argument never reaches output.
        ("{% set catalog=[] %}{{ catalog.append(tools) }}", False),
        ("{{ tools.clear() }}", False),
        ("{% set catalog=[] %}{% do catalog.append(tools) %}{{ catalog|tojson }}", True),
        # 3969190387: get() selects the field, and its default counts too.
        ("{% set wrapper={'catalog':tools} %}{{ wrapper.get('catalog')|tojson }}", True),
        ("{% set wrapper={'label':'x'} %}{{ wrapper.get('label') }}", False),
        ("{% set wrapper={} %}{{ wrapper.get('x', tools)|tojson }}", True),
        # 3969190396: iterating a mapping walks its keys, not its values.
        ("{% for key in {'x': tools} %}{{ key }}{% endfor %}", False),
        ("{% for v in tools %}{{ v|tojson }}{% endfor %}", True),
        # 3969190408: an inline loop filter guards the body, as GLM-4-32B spells it.
        (
            "{% for message in messages if message.role == 'tool' %}{{ message.content }}"
            "{% endfor %}",
            True,
        ),
        ("{% for m in messages if m.role != 'system' %}{{ m.content }}{% endfor %}", False),
    ],
)
def test_round_twelve_paths(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3969739986: caller() reached only on a dead path is not an invocation.
        (
            "{% macro wrap(caller=None) %}{% if false %}{{ caller() }}{% endif %}{% endmacro %}"
            "{% call wrap() %}{{ tools|tojson }}{% endcall %}",
            False,
        ),
        (
            "{% macro wrap(caller=None) %}{{ caller() }}{% endmacro %}"
            "{% call wrap() %}{{ tools|tojson }}{% endcall %}",
            True,
        ),
        (
            "{% macro w(caller=None) %}{% if flag %}{{ caller() }}{% endif %}{% endmacro %}"
            "{% call w() %}{{ tools|tojson }}{% endcall %}",
            True,
        ),
        ("{% call unknown() %}{{ tools|tojson }}{% endcall %}", True),
        (
            "{% macro w(caller=None) %}{{ caller() }}{% endmacro %}{% call w() %}plain{% endcall %}",
            False,
        ),
        # 3969739997: with a filter the loop position describes the accepted sequence.
        (
            "{% for x in [false, true] if x %}{% if loop.first %}{{ tools|tojson }}{% endif %}"
            "{% endfor %}",
            True,
        ),
        # 3969740007: a mutator called in output position still mutates.
        ("{% set catalog=tools|list %}{{ catalog.clear() }}{{ catalog|tojson }}", False),
        ("{% set catalog=tools|list %}{{ catalog|tojson }}", True),
        # 3969740018: a truth-tested count is a guard, in both spellings.
        ("{% if tools|length %}You may call tools.{% endif %}", True),
        ("{% if not tools|length %}plain{% else %}You may call tools.{% endif %}", True),
        ("{% if messages|length %}nothing here{% endif %}", False),
        # 3969740028: the role literal may be staged in a variable.
        (
            "{% set tool_role='tool' %}{% if message.role == tool_role %}{{ message.content }}"
            "{% endif %}",
            True,
        ),
        ("{% set r='user' %}{% if message.role == r %}{{ message.content }}{% endif %}", False),
        # 3969740040: pop hands back its default when the field is missing.
        ("{% set d={} %}{{ d.pop('missing', tools)|tojson }}", True),
        ("{% set d={'a':tools} %}{% do d.pop('a') %}{{ d|tojson }}", False),
        # 3969740046: rebuilding a container drops the old subtree.
        (
            "{% set wrapper={'message': {'role':'user'}} %}{% set wrapper={'message': message} %}"
            "{% if wrapper.message.role == 'tool' %}{{ wrapper.message.content }}{% endif %}",
            True,
        ),
        (
            "{% set wrapper={'message': {'role':'user'}} %}"
            "{% if wrapper.message.role == 'tool' %}{{ wrapper.message.content }}{% endif %}",
            False,
        ),
        # 3969740055: raise_exception aborts the render, and a guard that proves the
        # catalog empty means the output it reaches carries no schema.
        ("{% if tools %}{{ raise_exception('unsupported') }}{% endif %}{{ tools|tojson }}", False),
        ("{{ raise_exception('always') }}{{ tools|tojson }}", False),
        ("{{ tools|tojson }}{{ raise_exception('after') }}", True),
        ("{% if not tools %}{{ raise_exception('none') }}{% endif %}{{ tools|tojson }}", True),
        ("{% if not tools %}{{ tools|tojson }}{% endif %}", False),
        ("{% if tools %}{{ tools|tojson }}{% endif %}", True),
    ],
)
def test_round_thirteen_paths(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3970832364: get() is the mapping spelling of a field read.
        (
            "{% if message.get('tool_calls') %}{{ message.get('tool_calls')|tojson }}{% endif %}",
            True,
        ),
        ("{% if message.get('content') %}{{ message.get('content') }}{% endif %}", False),
        ("{% set w={'role':'tool'} %}{% if w.get('role') == 'tool' %}plain{% endif %}", False),
        # 3970832372: {% do load() %} discards the value but still runs the macro.
        (
            "{% set ns=namespace(catalog=[]) %}{% macro load() %}{% set ns.catalog=tools %}"
            "{% endmacro %}{% do load() %}{{ ns.catalog|tojson }}",
            True,
        ),
        # 3970832382: unmatched arguments arrive as the implicit varargs and kwargs.
        ("{% macro show() %}{{ kwargs|tojson }}{% endmacro %}{{ show(catalog=tools) }}", True),
        ("{% macro show() %}{{ varargs|tojson }}{% endmacro %}{{ show(tools) }}", True),
        ("{% macro show() %}{{ kwargs|tojson }}{% endmacro %}{{ show(label='x') }}", False),
        # 3970832395: a default is only reachable when the key can be absent.
        ("{% set d={'present': []} %}{{ d.pop('present', tools)|tojson }}", False),
        ("{% set d={'present': 'x'} %}{{ d.pop('present', tools)|tojson }}", False),
        ("{% set d={} %}{{ d.pop('missing', tools)|tojson }}", True),
        ("{% set d={'other': 1} %}{{ d.pop('missing', tools)|tojson }}", True),
        ("{{ payload.pop('x', tools)|tojson }}", True),
        ("{% set w={'label':'x'} %}{{ w.get('label', tools)|tojson }}", False),
        # 3970832398: constant Jinja tests fold, so their branches are not scanned.
        ("{% if false is true %}{{ tools|tojson }}{% endif %}", False),
        ("{% if 1 is none %}{{ tools|tojson }}{% endif %}", False),
        ("{% if true is true %}{{ tools|tojson }}{% endif %}", True),
        ("{% if x is defined %}{{ tools|tojson }}{% endif %}", True),
        # 3970832411: break leaves the loop incomplete, so Jinja renders the else arm.
        ("{% for x in [1] %}{% break %}{% else %}{{ tools|tojson }}{% endfor %}", True),
        ("{% for x in [1] %}plain{% else %}{{ tools|tojson }}{% endfor %}", False),
        # 3970832421: a namespace write inside {% set %}...{% endset %} outlives it.
        (
            "{% set ns=namespace(catalog=[]) %}{% set captured %}{% set ns.catalog=tools %}x"
            "{% endset %}{{ ns.catalog|tojson }}",
            True,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}{% set captured %}plain{% endset %}"
            "{{ ns.catalog|tojson }}",
            False,
        ),
        # 3970832427: reverse and sort move the members, so the indices go unknown.
        ("{% set c=[tools, []] %}{% do c.reverse() %}{{ c[1]|tojson }}", True),
        ("{% set c=[[], []] %}{% do c.reverse() %}{{ c[1]|tojson }}", False),
    ],
)
def test_round_fourteen_paths(template, expected):
    assert template_supports_tools(template) is expected


def test_reordering_a_list_over_approximates_rather_than_losing_the_catalog():
    """`reverse` makes every index under the receiver unknown rather than tracking
    where each member went, so selecting the index the catalog moved AWAY from also
    matches. That is deliberate: this detector's damaging failure is hiding tool
    controls on a template that supports them, so it errs towards showing them."""
    assert (
        template_supports_tools("{% set c=[tools, []] %}{% do c.reverse() %}{{ c[0]|tojson }}")
        is True
    )


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3968192941: a name bound to a macro calls the same body.
        (
            "{% macro show() %}{{ tools|tojson }}{% endmacro %}{% set render=show %}"
            "{{ render() }}",
            True,
        ),
        ("{% macro show() %}plain{% endmacro %}{% set render=show %}{{ render() }}", False),
        # 3968192960: the role may be staged in a name before the comparison.
        (
            "{% set role=message.role %}{% if role == 'tool' %}{{ message.content }}{% endif %}",
            True,
        ),
        (
            "{% set role=message.role %}{% if role == 'user' %}{{ message.content }}{% endif %}",
            False,
        ),
        # A role read off a template-built record still means nothing.
        (
            "{% set w={'role':'tool'} %}{% set role=w.role %}{% if role == 'tool' %}plain"
            "{% endif %}",
            False,
        ),
    ],
)
def test_round_ten_leftovers_now_closed(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3971620568: an arm selected only where the value is proved empty.
        ("{{ tools if not tools else [] }}", False),
        ("{{ tools if tools else [] }}", True),
        ("{{ [] if not tools else tools }}", True),
        # 3971620577: recursion is simulated to a bounded depth.
        (
            "{% macro show(n,c) %}{% if n %}{{ show(n-1,c) }}{% else %}{{ c|tojson }}"
            "{% endif %}{% endmacro %}{{ show(1,tools) }}",
            True,
        ),
        (
            "{% macro show(n,c) %}{% if n %}{{ show(n-1,c) }}{% else %}plain{% endif %}"
            "{% endmacro %}{{ show(1,tools) }}",
            False,
        ),
        ("{% macro f(c) %}{{ f(c) }}{% endmacro %}{{ f(tools) }}", False),
        # 3971620584: a macro chosen through an expression is still that macro.
        (
            "{% macro show() %}{{ tools|tojson }}{% endmacro %}"
            "{% set render = show if flag else show %}{{ render() }}",
            True,
        ),
        (
            "{% macro show() %}plain{% endmacro %}{% set render = show if flag else show %}"
            "{{ render() }}",
            False,
        ),
        # 3971620595: an else arm reached after break keeps that path's mutations.
        (
            "{% set ns=namespace(catalog=[]) %}{% for x in [1] %}{% set ns.catalog=tools %}"
            "{% break %}{% else %}{{ ns.catalog|tojson }}{% endfor %}",
            True,
        ),
        (
            "{% set ns=namespace(catalog=[]) %}{% for x in [1] %}{% break %}{% else %}"
            "{{ ns.catalog|tojson }}{% endfor %}",
            False,
        ),
        # 3971620599: template-built is asked of the field, not just its container.
        (
            "{% set ns=namespace() %}{% set ns.role=message.role %}"
            "{% if ns.role == 'tool' %}{{ message.content }}{% endif %}",
            True,
        ),
        ("{% set ns=namespace(role='tool') %}{% if ns.role == 'tool' %}plain{% endif %}", False),
        (
            "{% set ns=namespace() %}{% set ns.role='tool' %}{% if ns.role == 'tool' %}plain"
            "{% endif %}",
            False,
        ),
        # 3971620605: a scalar-returning method reduces its input.
        ("{{ tools.count(tools[0]) }}", False),
        # 3971620613: an indexed removal shifts the later indices down.
        ("{% set c=[[],tools] %}{% do c.pop(0) %}{{ c[0]|tojson }}", True),
        ("{% set c=[[],tools] %}{% do c.pop(0) %}{{ c[1]|tojson }}", False),
        ("{% set c=[tools,[]] %}{% do c.pop(1) %}{{ c[0]|tojson }}", True),
        # 3971620621: a raise in an always-evaluated position aborts the expression.
        ("{{ raise_exception('unsupported') or tools|tojson }}", False),
        ("{{ tools|tojson or raise_exception('never') }}", True),
        ("{{ raise_exception('always') }}{{ tools|tojson }}", False),
        ("{{ tools|tojson }}{{ raise_exception('after') }}", True),
    ],
)
def test_round_fifteen_paths(template, expected):
    assert template_supports_tools(template) is expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # 3971892918: a call's arguments are evaluated before the call.
        ("{{ dict(error=raise_exception('unsupported'), catalog=tools)|tojson }}", False),
        ("{{ dict(catalog=tools)|tojson }}", True),
        # 3971892927: a macro selected out of a static collection.
        (
            "{% macro show() %}{{ tools|tojson }}{% endmacro %}{% set render=[show][0] %}"
            "{{ render() }}",
            True,
        ),
        ("{% macro show() %}plain{% endmacro %}{% set render=[show][0] %}{{ render() }}", False),
        # 3971892936: iterating a mapping walks its keys, even when it was named first.
        (
            "{% set by_name={'weather': tools} %}{% for name in by_name %}{{ name }}{% endfor %}",
            False,
        ),
        (
            "{% set by_name={'weather': tools} %}{% for name in by_name %}"
            "{{ by_name[name]|tojson }}{% endfor %}",
            True,
        ),
        (
            "{% set by_name={'w': tools} %}{% set by_name=tools %}{% for v in by_name %}"
            "{{ v|tojson }}{% endfor %}",
            True,
        ),
        ("{% set c=[tools] %}{% for v in c %}{{ v|tojson }}{% endfor %}", True),
        # 3971892941: a statically empty slice renders nothing; other slices do not.
        ("{{ tools[0:0]|tojson }}", False),
        ("{{ tools[0:1]|tojson }}", True),
        ("{{ tools[1:]|tojson }}", True),
        ("{{ tools[:2]|tojson }}", True),
        ("{% for m in messages[1:] %}{{ m.tool_calls|tojson }}{% endfor %}", True),
    ],
)
def test_round_sixteen_paths(template, expected):
    assert template_supports_tools(template) is expected
