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


def _tool_reference(node, aliases):
    return (isinstance(node, nodes.Name) and node.name in aliases) or _field(node) in (
        "tools",
        "tool_calls",
    )


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


def _emits_tools(node, aliases):
    if not _is_payload(node):
        return False
    return _tool_reference(node, aliases) or any(
        _emits_tools(child, aliases) for child in node.iter_child_nodes()
    )


def _emits(body):
    for statement in body:
        outputs = (
            [statement] if isinstance(statement, nodes.Output) else statement.find_all(nodes.Output)
        )
        for output in outputs:
            for value in output.nodes:
                if not _is_payload(value):
                    continue
                if isinstance(value, nodes.TemplateData) and not value.data.strip():
                    continue
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

    aliases = {"tools", "tool_calls"}
    assignments = list(tree.find_all(nodes.Assign))
    for _ in assignments:
        previous = len(aliases)
        for assignment in assignments:
            if isinstance(assignment.target, nodes.Name) and _tool_reference(
                assignment.node, aliases
            ):
                aliases.add(assignment.target.name)
        if len(aliases) == previous:
            break

    for output in tree.find_all(nodes.Output):
        if any(_emits_tools(value, aliases) for value in output.nodes):
            return True
    for loop in tree.find_all(nodes.For):
        if _tool_reference(loop.iter, aliases) and _emits(loop.body):
            return True
    return any(
        _positive_test(branch.test, aliases) and _emits(branch.body)
        for branch in tree.find_all(nodes.If)
    )
