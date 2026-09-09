# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-capability hints from executable Jinja syntax."""

from functools import lru_cache

from jinja2 import Environment, TemplateSyntaxError, nodes
from jinja2.ext import Extension


class _Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle = True)


_ENVIRONMENT = Environment(extensions = [_Generation, "jinja2.ext.loopcontrols", "jinja2.ext.do"])


def _field(node):
    if isinstance(node, nodes.Getattr):
        return node.attr
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        return node.arg.value
    return None


def _reference_key(node):
    if isinstance(node, nodes.Name):
        return (node.name,)
    if isinstance(node, nodes.NSRef):
        return (node.name, node.attr)
    if isinstance(node, (nodes.Getattr, nodes.Getitem)):
        parent = _reference_key(node.node)
        field = _field(node)
        if parent is not None and isinstance(field, str):
            return (*parent, field)
    return None


def _tool_reference(node, aliases):
    return _reference_key(node) in aliases or _field(node) == "tool_calls"


def _walk(node):
    yield node
    if not isinstance(node, nodes.Macro):
        for child in node.iter_child_nodes():
            yield from _walk(child)


def _positive_test(node, aliases):
    if _tool_reference(node, aliases):
        return True
    if isinstance(node, (nodes.And, nodes.Or)):
        return _positive_test(node.left, aliases) or _positive_test(node.right, aliases)
    if isinstance(node, nodes.Test):
        return node.name == "defined" and _tool_reference(node.node, aliases)
    if isinstance(node, nodes.Not):
        return (
            isinstance(node.node, nodes.Test)
            and node.node.name in ("none", "undefined")
            and _tool_reference(node.node.node, aliases)
        )
    if isinstance(node, nodes.Compare) and len(node.ops) == 1:
        operand = node.ops[0]
        if operand.op == "eq":
            return any(
                _field(role) == "role" and isinstance(value, nodes.Const) and value.value == "tool"
                for role, value in ((node.expr, operand.expr), (operand.expr, node.expr))
            )
    return False


def _is_payload(node):
    if isinstance(node, (nodes.Not, nodes.Test, nodes.Compare)):
        return False
    if isinstance(node, nodes.Filter) and node.name in ("length", "count"):
        return False
    return not (
        isinstance(node, nodes.Call)
        and isinstance(node.node, nodes.Name)
        and node.node.name == "raise_exception"
    )


def _emits_tools(node, aliases, macros, active):
    if not _is_payload(node):
        return False
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name):
        macro = macros.get(node.node.name)
        if macro is not None:
            if macro.name in active:
                return False
            parameters = [argument.name for argument in macro.args]
            local_aliases = {key for key in aliases if key[0] not in parameters}
            arguments = dict(zip(parameters, node.args))
            arguments.update((keyword.key, keyword.value) for keyword in node.kwargs)
            defaults = dict(
                zip(parameters[len(parameters) - len(macro.defaults) :], macro.defaults)
            )
            for name in parameters:
                value = arguments.get(name, defaults.get(name))
                source_aliases = aliases if name in arguments else local_aliases.copy()
                if value is not None and _emits_tools(value, source_aliases, macros, active):
                    local_aliases.add((name,))
                source = _reference_key(value)
                if source is not None:
                    for key in source_aliases:
                        if key[: len(source)] == source:
                            local_aliases.add((name, *key[len(source) :]))
            local_macros = {name: value for name, value in macros.items() if name not in parameters}
            return _scan(macro.body, local_aliases, local_macros, active | {macro.name})
    return _tool_reference(node, aliases) or any(
        _emits_tools(child, aliases, macros, active) for child in node.iter_child_nodes()
    )


def _emits(body):
    for statement in body:
        outputs = (node for node in _walk(statement) if isinstance(node, nodes.Output))
        for output in outputs:
            for value in output.nodes:
                if not _is_payload(value):
                    continue
                if isinstance(value, nodes.TemplateData) and not value.data.strip():
                    continue
                return True
    return False


def _scan(body, aliases, macros, active):
    syntax = [node for statement in body for node in _walk(statement)]
    macros = {**macros, **{node.name: node for node in syntax if isinstance(node, nodes.Macro)}}
    assignments = [node for node in syntax if isinstance(node, nodes.Assign)]
    for _ in assignments:
        previous = len(aliases)
        for assignment in assignments:
            key = _reference_key(assignment.target)
            if key is not None and _tool_reference(assignment.node, aliases):
                aliases.add(key)
            value = assignment.node
            if (
                key is not None
                and isinstance(value, nodes.Call)
                and isinstance(value.node, nodes.Name)
                and value.node.name == "namespace"
            ):
                for keyword in value.kwargs:
                    if _tool_reference(keyword.value, aliases):
                        aliases.add((*key, keyword.key))
        if len(aliases) == previous:
            break

    for node in syntax:
        if isinstance(node, nodes.Output):
            if any(_emits_tools(value, aliases, macros, active) for value in node.nodes):
                return True
        elif isinstance(node, nodes.For):
            if _tool_reference(node.iter, aliases) and _emits(node.body):
                return True
        elif isinstance(node, nodes.If):
            if _positive_test(node.test, aliases) and _emits(node.body):
                return True
    return False


@lru_cache(maxsize = 128)
def template_supports_tools(template: str) -> bool:
    """Inspect syntax only; rendering and parser support remain backend checks."""
    if "tool" not in template:
        return False
    try:
        tree = _ENVIRONMENT.parse(template)
    except TemplateSyntaxError:
        return False
    return _scan(tree.body, {("tools",), ("tool_calls",)}, {}, set())
