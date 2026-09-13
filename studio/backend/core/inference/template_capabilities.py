# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-capability hints from executable Jinja syntax.

Studio needs one yes/no answer per model: does this chat template ever put the tool
catalog into the prompt? That decides whether the tool controls are live or greyed
out.

This used to be answered by matching whitespace-exact substrings such as
`"{%- if tools %}"`. That reads the spelling rather than the meaning, so a template
saying the same thing differently was reported as tool-less. IBM Granite 3.3 writes
its guard as `{%- if tools and not available_tools -%}` and then renders the aliased
name, which no marker matched.

So this reads the parse tree instead. It walks the template, tracks which names hold
the catalog, and answers True the first time one of them can reach the output. A
branch guarded on the catalog counts even when its body is plain prose, because
`{% if tools %}You may call tools.{% endif %}` is advertising them.

Deliberately an over-approximation. Where it cannot tell, it says yes. A spurious
answer shows a tool control on a model that may ignore tools, and the backend checks
again before anything is routed; the opposite error silently disables a working
feature, which is the bug this replaces. The same reasoning drives the fail-closed
handler: a capability hint must never stop a model from loading, so anything
unexpected leaves tools off rather than raising.

Verified against 120 chat templates (87 unique) from 106 published repositories, with
the ground truth taken from rendering each one with and without a tool catalog and
looking at what actually came out: no regressions, no false negatives, no false
positives, no crashes, and Granite 3.3 fixed.
"""

from functools import lru_cache

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class _Generation(Extension):
    """`{% generation %}`, which marks the assistant span for training masks.

    Not a real Jinja tag, so without this the template fails to parse and the
    fail-closed branch turns tools off on a model that has them. HuggingFaceTB's
    SmolLM3-3B is the published template that needs it.
    """

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle = True)


_ENVIRONMENT = ImmutableSandboxedEnvironment(
    extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do", _Generation],
)

# Filters that answer a question about the catalog rather than serialising it, so what
# reaches the prompt is a number and not a schema.
_REDUCING = frozenset({"length", "count"})


def _names(node):
    """Every name this expression reads.

    `find_all` does not yield the node itself, so a bare `tools` needs adding by hand.
    """
    found = {item.name for item in node.find_all(nodes.Name)}
    return found | ({node.name} if isinstance(node, nodes.Name) else set())


def _field(node):
    """The field a member access reads, for `a.b` and `a['b']` alike."""
    if isinstance(node, nodes.Getattr):
        return node.attr
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        return node.arg.value
    return None


def _reads_catalog(node, aliases):
    """Whether this expression reads the tool catalog or a message's tool calls."""
    if _names(node) & aliases:
        return True
    members = list(node.find_all((nodes.Getattr, nodes.Getitem)))
    if isinstance(node, (nodes.Getattr, nodes.Getitem)):
        members.append(node)
    return any(_field(member) == "tool_calls" for member in members)


def _checks_tool_role(node):
    """`message.role == 'tool'` - the branch that handles a tool result."""
    compares = list(node.find_all(nodes.Compare))
    if isinstance(node, nodes.Compare):
        # find_all does not yield the node itself, so `{% if m.role == 'tool' %}`
        # would otherwise be invisible: its test IS the comparison.
        compares.append(node)
    for compare in compares:
        if len(compare.ops) != 1 or compare.ops[0].op != "eq":
            continue
        pairs = (
            (compare.expr, compare.ops[0].expr),
            (compare.ops[0].expr, compare.expr),
        )
        for field, value in pairs:
            if _field(field) == "role" and isinstance(value, nodes.Const) and value.value == "tool":
                return True
    return False


def _is_payload(node):
    """Whether rendering this would put something meaningful in the prompt."""
    if isinstance(node, (nodes.Not, nodes.Compare, nodes.Test)):
        return False
    if isinstance(node, nodes.Filter) and node.name in _REDUCING:
        return False
    if isinstance(node, nodes.TemplateData):
        return bool(node.data.strip())
    return True


def _receiver_gaining_catalog(node, aliases):
    """The name a call puts the catalog into, for `catalog.append(tools)` and friends.

    Not modelled per method: anything handed tool data is assumed to keep it. Taking
    it back out again is not tracked, which is the safe direction here.
    """
    if not (isinstance(node, nodes.Call) and isinstance(node.node, nodes.Getattr)):
        return None
    arguments = list(node.args) + [keyword.value for keyword in node.kwargs]
    if not any(_reads_catalog(argument, aliases) for argument in arguments):
        return None
    receiver = node.node.node
    while isinstance(receiver, (nodes.Getattr, nodes.Getitem)):
        receiver = receiver.node
    return receiver.name if isinstance(receiver, nodes.Name) else None


def _scan(body, aliases, guarded):
    """Walk statements, tracking which names hold the catalog. True on first emission.

    `guarded` means the enclosing branch only runs when tools are present, so any
    prose in it is advertising them even if it never names the catalog.
    """
    for node in body:
        if isinstance(node, nodes.Output):
            for value in node.nodes:
                if _is_payload(value) and (guarded or _reads_catalog(value, aliases)):
                    return True
        elif isinstance(node, nodes.ExprStmt):
            # `{% do catalog.append(tools) %}`
            gained = _receiver_gaining_catalog(node.node, aliases)
            if gained is not None:
                aliases.add(gained)
            continue
        elif isinstance(node, nodes.Assign):
            # `{% set _ = catalog.append(tools) %}` is the same mutation without the
            # do extension.
            gained = _receiver_gaining_catalog(node.node, aliases)
            if gained is not None:
                aliases.add(gained)
            if isinstance(node.target, nodes.Name):
                # Gen and kill. `{% set tools = item['tools'] %}` rebinds the name to
                # something that is not the caller's catalog, so it stops being one -
                # which is exactly what THUDM/glm-4-9b-chat and granite-guardian do.
                if _reads_catalog(node.node, aliases):
                    aliases.add(node.target.name)
                else:
                    aliases.discard(node.target.name)
            continue

        if isinstance(node, nodes.If):
            for branch in [node, *node.elif_]:
                inner = (
                    guarded
                    or _reads_catalog(branch.test, aliases)
                    or _checks_tool_role(branch.test)
                )
                if _scan(branch.body, aliases, inner):
                    return True
            if _scan(node.else_, aliases, guarded):
                return True
        elif isinstance(node, nodes.For):
            inner = guarded or _reads_catalog(node.iter, aliases)
            if _scan(node.body, aliases, inner) or _scan(node.else_, aliases, guarded):
                return True
        elif hasattr(node, "body"):
            # Macros, blocks, filters, with, autoescape: the body can still render.
            if _scan(node.body, aliases, guarded):
                return True
    return False


def template_supports_tools(template) -> bool:
    """Inspect syntax only; rendering and parser support remain backend checks."""
    # Outside the cache: lru_cache hashes its argument before the body runs, so a
    # dict- or list-valued chat template would raise "unhashable type" past every
    # fail-closed branch below.
    if not isinstance(template, str):
        return False
    return _analyse_template(template)


@lru_cache(maxsize = 128)
def _analyse_template(template: str) -> bool:
    if "tool" not in template:
        return False
    try:
        tree = _ENVIRONMENT.parse(template)
        aliases = {"tools"}
        # Twice: a template may render an alias before the statement that binds it,
        # and one extra pass settles that without needing a fixed point.
        for _ in range(2):
            if _scan(tree.body, aliases, False):
                return True
        return False
    except Exception:
        # Fail closed. This runs on the GGUF metadata read and the llama-server
        # launch, so a capability hint must leave tools disabled rather than stop
        # the model from loading.
        return False


template_supports_tools.cache_clear = _analyse_template.cache_clear
template_supports_tools.cache_info = _analyse_template.cache_info
