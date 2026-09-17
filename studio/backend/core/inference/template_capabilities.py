# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does this chat template ever put the tool catalog into the prompt?

Answers Studio's one yes/no per model, deciding whether the tool controls are live.

Do not go back to matching substrings: that reads spelling, not meaning, and called
Granite 3.3 tool-less for writing `{%- if tools and not available_tools -%}`.

Over-approximates on purpose. A spurious yes shows a control the backend re-checks;
a spurious no silently disables a working feature, which is the bug this replaces.
"""

from functools import lru_cache

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class _Generation(Extension):
    """Not real Jinja, so without it HuggingFaceTB/SmolLM3-3B fails to parse."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle = True)


_ENVIRONMENT = ImmutableSandboxedEnvironment(
    extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do", _Generation],
)

# Ask about the catalog rather than serialise it: a number reaches the prompt.
_REDUCING = frozenset({"length", "count"})


def _names(node):
    """Every name this expression reads. find_all skips the node, so add it by hand."""
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
        # `{% if m.role == 'tool' %}` IS the comparison, which find_all skips.
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


def _bound_names(node):
    """The names a `{% set %}` or `{% for %}` target binds, tuple targets included."""
    if isinstance(node, nodes.Tuple):
        return {name for item in node.items for name in _bound_names(item)}
    # `{% set ns.catalog = ... %}` is an NSRef, already naming the container.
    while isinstance(node, (nodes.Getattr, nodes.Getitem)):
        node = node.node
    return {node.name} if isinstance(node, (nodes.Name, nodes.NSRef)) else set()


def _rebound_names(node):
    """Names a target REBINDS: `{% set ns.catalog = x %}` cannot un-hold `ns`."""
    if isinstance(node, nodes.Name):
        return {node.name}
    if isinstance(node, nodes.Tuple):
        return {name for item in node.items for name in _rebound_names(item)}
    return set()


def _receiver_gaining_catalog(node, aliases, guarded = False):
    """Names `catalog.append(tools)` fills. Handed tool data is assumed kept.

    Under a tools guard the call runs only when tools exist, so the receiver holds
    tool-conditional content whatever the argument was.
    """
    if not (isinstance(node, nodes.Call) and isinstance(node.node, nodes.Getattr)):
        return set()
    arguments = list(node.args) + [keyword.value for keyword in node.kwargs]
    if not guarded and not any(_reads_catalog(argument, aliases) for argument in arguments):
        return set()
    return _bound_names(node.node.node)


def _scan_maybe(body, aliases, guarded, bound = frozenset(), killed = frozenset()):
    """Walk a body that may not run. Keep what it learns, drop what it unbinds; the
    body's own `bound`/`killed` names escape in neither direction."""
    local = (set(aliases) | set(bound)) - set(killed)
    found = _scan(body, local, guarded)
    aliases |= local - (set(bound) - aliases)
    return found


def _join(aliases, arms, exhaustive):
    """Union the arms: a name holds the catalog if it does on any. Falling past every
    arm is a path too, so the incoming set counts unless an `{% else %}` is present -
    which is what keeps a rebinding on every arm a rebinding."""
    merged = set() if exhaustive else set(aliases)
    for arm in arms:
        merged |= arm
    aliases.clear()
    aliases.update(merged)


def _scan(body, aliases, guarded):
    """Walk statements, tracking which names hold the catalog; True on first emission.
    `guarded` means the branch runs only when tools exist, so its prose advertises them."""
    for node in body:
        if isinstance(node, nodes.Output):
            for value in node.nodes:
                # `{{ m.content if m.role == 'tool' else '' }}` carries its tool-role
                # check inside the expression, where an `{% if %}` would otherwise
                # hold it. The marker scan matched that spelling.
                if _is_payload(value) and (
                    guarded
                    or _reads_catalog(value, aliases)
                    or _checks_tool_role(value)
                ):
                    return True
        elif isinstance(node, nodes.ExprStmt):
            # `{% do catalog.append(tools) %}`
            aliases |= _receiver_gaining_catalog(node.node, aliases, guarded)
            continue
        elif isinstance(node, nodes.Assign):
            # `{% set _ = catalog.append(tools) %}`: the same mutation without `do`.
            aliases |= _receiver_gaining_catalog(node.node, aliases, guarded)
            if (_reads_catalog(node.node, aliases) or guarded
                    or _checks_tool_role(node.node)):
                # LiquidAI's LFM2 fills `ns.system_prompt` inside the guard and
                # renders it outside, so a namespace field has to carry the catalog.
                # Anything stored under a tools guard counts too, even a constant:
                # `{% if tools %}{% set intro = 'You may call functions.' %}{% endif %}`
                # only runs when tools exist, so rendering `intro` later advertises
                # them - which the marker scan caught and a catalog-only rule misses.
                # A stored tool-role predicate counts the same way: branching on
                # `{% set handles_tool = m.role == 'tool' %}` renders exactly what the
                # inline spelling does, which is already a yes.
                aliases |= _bound_names(node.target)
            else:
                # Plain names only: writing one field says nothing about the rest of
                # the container. glm-4-9b-chat rebinds `tools` off a message.
                aliases -= _rebound_names(node.target)
            continue

        if isinstance(node, nodes.If):
            arms = []
            for branch in [node, *node.elif_]:
                inner = (
                    guarded
                    or _reads_catalog(branch.test, aliases)
                    or _checks_tool_role(branch.test)
                )
                arm = set(aliases)
                if _scan(branch.body, arm, inner):
                    return True
                arms.append(arm)
            if node.else_:
                arm = set(aliases)
                if _scan(node.else_, arm, guarded):
                    return True
                arms.append(arm)
            _join(aliases, arms, exhaustive = bool(node.else_))
        elif isinstance(node, nodes.For):
            over_catalog = _reads_catalog(node.iter, aliases)
            # `{% for m in messages if m.role == 'tool' %}` keeps its guard in the
            # loop's own filter, where an `{% if %}` would otherwise hold it.
            filtered = node.test is not None and (
                _reads_catalog(node.test, aliases) or _checks_tool_role(node.test)
            )
            # Each item is catalog data, so the loop variable carries it.
            bound = _bound_names(node.target) if over_catalog else frozenset()
            if _scan_maybe(node.body, aliases, guarded or over_catalog or filtered, bound):
                return True
            if _scan_maybe(node.else_, aliases, guarded):
                return True
        elif isinstance(node, nodes.With):
            # `{% with catalog = tools %}` binds like a set, for the block only.
            bound = set()
            killed = set()
            for target, value in zip(node.targets, node.values):
                if _reads_catalog(value, aliases):
                    bound |= _bound_names(target)
                else:
                    killed |= _rebound_names(target)
            if _scan_maybe(node.body, aliases, guarded, bound, killed):
                return True
        elif hasattr(node, "body"):
            # Macros, blocks, filters, autoescape: the body can still render.
            if _scan_maybe(node.body, aliases, guarded):
                return True
    return False


def template_supports_tools(template) -> bool:
    """Inspect syntax only; rendering and parser support remain backend checks."""
    # Outside the cache: lru_cache hashes first, so a dict-valued template would
    # raise past every fail-closed branch below.
    if not isinstance(template, str):
        return False
    # `str.__str__`, not `str(...)`: an override would raise out here, unhandled.
    return _analyse_template(str.__str__(template))


@lru_cache(maxsize = 128)
def _analyse_template(template: str) -> bool:
    if "tool" not in template:
        return False
    try:
        tree = _ENVIRONMENT.parse(template)
        aliases = {"tools"}
        # Twice: a template may render an alias before the statement that binds it.
        for _ in range(2):
            if _scan(tree.body, aliases, False):
                return True
        return False
    except Exception:
        # Fail closed: this runs on the model-load path and must not stop it.
        return False


template_supports_tools.cache_clear = _analyse_template.cache_clear
template_supports_tools.cache_info = _analyse_template.cache_info
