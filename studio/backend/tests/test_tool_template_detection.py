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
