# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-capability hints from executable Jinja syntax."""

from dataclasses import dataclass, field
from functools import lru_cache

from jinja2 import Environment, nodes
from jinja2.ext import Extension


class _Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle = True)


_ENVIRONMENT = Environment(extensions = [_Generation, "jinja2.ext.loopcontrols", "jinja2.ext.do"])
_UNKNOWN = object()

# In-place mutators: `{{ catalog.append(tools) }}` renders "None", so the argument
# reaches the receiver but never the output.
# Methods whose result is a number, so nothing of the receiver or the arguments
# survives into the rendered value. The `length` and `count` FILTERS are already
# excluded by `_is_payload`; these are the method spellings.
_SCALAR_RETURNING_METHODS = frozenset({"count", "index"})

# How many nested activations of one macro to simulate. A schema formatter that
# recurses over nested parameter objects reaches its catalog a level or two down, and
# refusing at the first repeat reported those templates as tool-less. The step budget
# is what stops this from being expensive.
_RECURSION_LIMIT = 3

_MUTATORS_RETURNING_NONE = frozenset(
    {"append", "extend", "insert", "update", "add", "clear", "sort", "reverse", "discard"}
)


class _AnalysisLimit(Exception):
    pass


@dataclass
class _State:
    aliases: set
    macros: dict = field(default_factory = dict)
    facts: dict = field(default_factory = dict)
    assigned: set = field(default_factory = set)
    mutated: set = field(default_factory = set)
    # Objects the template built itself, so a field named `tool_calls` or `role` on
    # them carries no meaning. Survives a scope change: a namespace built outside a
    # loop is still template-built inside it.
    constructed: set = field(default_factory = set)
    # Names bound to a constant, so a subscript written through one resolves to a
    # single field rather than to every field.
    consts: dict = field(default_factory = dict)
    # Names bound straight to a field of something else, so `{% set role =
    # message.role %}` still reads as a role check when the comparison uses `role`.
    origins: dict = field(default_factory = dict)
    # Keys holding a mapping the template built, mapped to the field names that
    # mapping was written with. Answers both `{% for name in by_name %}` walking keys
    # and whether a get/pop default can be reached.
    mappings: dict = field(default_factory = dict)
    # Set when the path hit break, so a literal loop stops simulating further items
    # for it. `continue` only ends the current iteration, so it is tracked apart:
    # the path skips the rest of the body but still sees the next item.
    terminated: bool = False
    continued: bool = False
    budget: list = field(default_factory = lambda: [8192])

    def copy(self, scoped = False):
        return _State(
            self.aliases.copy(),
            self.macros.copy(),
            self.facts.copy(),
            set() if scoped else self.assigned.copy(),
            set() if scoped else self.mutated.copy(),
            self.constructed.copy(),
            self.consts.copy(),
            self.origins.copy(),
            self.mappings.copy(),
            self.terminated,
            self.continued,
            self.budget,
        )


def _reading_call(node):
    """`d.get('k')` reads field k, so it is the mapping spelling of `d.k` and `d['k']`.

    Only the single-argument form: with a default the value may come from elsewhere,
    and `_value_aliases` handles that case on its own.
    """
    return (
        isinstance(node, nodes.Call)
        and isinstance(node.node, nodes.Getattr)
        and node.node.attr == "get"
        and len(node.args) == 1
        and not node.kwargs
    )


def _origin_field(node, state):
    """The field a plain name was bound from, so `{% set role = message.role %}`
    still answers `role` when the comparison names the alias."""
    if isinstance(node, nodes.Name):
        origin = state.origins.get(node.name)
        if origin is not None:
            return origin[-1]
    return _UNKNOWN


def _field(node):
    if isinstance(node, nodes.Getattr):
        return node.attr
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        return node.arg.value
    if _reading_call(node):
        return node.args[0].value if isinstance(node.args[0], nodes.Const) else _UNKNOWN
    return _UNKNOWN


def _member(node, state):
    """`_field`, plus subscripts whose key is a name bound to a constant."""
    member = _field(node)
    if (
        member is _UNKNOWN
        and isinstance(node, nodes.Getitem)
        and isinstance(node.arg, nodes.Name)
        and node.arg.name in state.consts
    ):
        return state.consts[node.arg.name]
    return member


def _reference_key(node):
    if isinstance(node, nodes.Name):
        return (node.name,)
    if isinstance(node, nodes.NSRef):
        return (node.name, node.attr)
    if isinstance(node, (nodes.Getattr, nodes.Getitem)) or _reading_call(node):
        base = node.node.node if _reading_call(node) else node.node
        parent = _reference_key(base)
        member = _field(node)
        if parent is not None and isinstance(member, (str, int)):
            return (*parent, member)
    return None


def _names(node):
    return {item.name for item in node.find_all(nodes.Name)} | (
        {node.name} if isinstance(node, nodes.Name) else set()
    )


def _as_const(node, state):
    """The literal a node is known to be, either written out or bound to a name."""
    if isinstance(node, nodes.Const):
        return node
    if state is not None and isinstance(node, nodes.Name) and node.name in state.consts:
        return nodes.Const(state.consts[node.name])
    return None


def _constant_truth(node, state = None):
    fact = state.facts.get(repr(node)) if state is not None else None
    if fact is not None:
        return fact[0]
    if isinstance(node, nodes.Const):
        return bool(node.value)
    if isinstance(node, (nodes.List, nodes.Tuple, nodes.Dict)):
        return bool(node.items)
    if isinstance(node, nodes.Compare):
        # `{% set flag = true %}{% if flag == false %}` is dead at render time, so
        # resolve names the template bound to a literal before folding.
        folded = _as_const(node.expr, state)
        operands = [_as_const(operand.expr, state) for operand in node.ops]
        if folded is not None and all(operand is not None for operand in operands):
            replacement = nodes.Compare(
                folded,
                [nodes.Operand(operand.op, value) for operand, value in zip(node.ops, operands)],
            )
            try:
                # The rebuilt node is synthetic, so it carries no environment of its
                # own and has to be handed an evaluation context explicitly.
                return bool(replacement.as_const(nodes.EvalContext(_ENVIRONMENT)))
            except Exception:
                pass
    if isinstance(node, nodes.Test):
        # `{% if false is true %}` and `{% if 1 in [] %}` never run their bodies.
        folded = _as_const(node.node, state)
        arguments = [_as_const(argument, state) for argument in node.args]
        if folded is not None and all(argument is not None for argument in arguments):
            try:
                replacement = nodes.Test(folded, node.name, arguments, [], None, None)
                return bool(replacement.as_const(nodes.EvalContext(_ENVIRONMENT)))
            except Exception:
                pass
    if isinstance(node, nodes.Not):
        value = _constant_truth(node.node, state)
        return None if value is None else not value
    if isinstance(node, (nodes.And, nodes.Or)):
        left = _constant_truth(node.left, state)
        right = _constant_truth(node.right, state)
        decisive = isinstance(node, nodes.Or)
        if left is decisive or right is decisive:
            return decisive
        if left is not None and right is not None:
            return not decisive
    return None


def _assume(node, truth, state):
    state.budget[0] -= 1
    if state.budget[0] < 0:
        raise _AnalysisLimit
    known = _constant_truth(node, state)
    if known is not None:
        return [state.copy()] if known is truth else []
    if isinstance(node, nodes.Not):
        return _assume(node.node, not truth, state)
    if isinstance(node, (nodes.And, nodes.Or)):
        both = truth if isinstance(node, nodes.And) else not truth
        if both:
            return [
                right
                for left in _assume(node.left, truth, state)
                for right in _assume(node.right, truth, left)
            ]
        return _assume(node.left, truth, state) + [
            right
            for left in _assume(node.left, not truth, state)
            for right in _assume(node.right, truth, left)
        ]
    result = state.copy()
    result.facts[repr(node)] = (truth, _names(node))
    return [result]


def _signature(state):
    """Everything about a path that can still change the verdict."""
    return (
        frozenset(state.aliases),
        tuple(sorted((name, id(node)) for name, node in state.macros.items())),
        frozenset((expression, fact[0]) for expression, fact in state.facts.items()),
        frozenset(state.assigned),
        frozenset(state.mutated),
        frozenset(state.constructed),
        frozenset((name, repr(value)) for name, value in state.consts.items()),
        frozenset(state.origins.items()),
        frozenset((name, members) for name, members in state.mappings.items()),
        state.terminated,
        state.continued,
    )


def _live_names(body):
    """Names each suffix of `body` still refers to, so a fact about a name nothing
    reads again can be dropped and the two paths that carried it collapse into one.

    Without this a template pays for the full Cartesian product of its conditions:
    twelve unrelated `{% if flagN %}` blocks are 4096 paths, which exhausts the
    budget and fails closed on a template that does support tools. Qwen3-Coder
    already spends most of the budget, so the headroom is not theoretical.
    """
    suffixes = [frozenset()] * (len(body) + 1)
    for index in range(len(body) - 1, -1, -1):
        suffixes[index] = _names(body[index]) | suffixes[index + 1]
    return suffixes


def _collapse(states, live):
    """Drop facts nothing reads again, then merge paths that became identical."""
    seen = {}
    for state in states:
        if state.facts:
            state.facts = {
                expression: fact for expression, fact in state.facts.items() if fact[1] & live
            }
        seen.setdefault(_signature(state), state)
    return list(seen.values()) if len(seen) < len(states) else states


def _forget(key, state):
    state.facts = {
        expression: fact for expression, fact in state.facts.items() if key[0] not in fact[1]
    }


def _select(paths, member):
    return {
        suffix[1:] if suffix else ()
        for suffix in paths
        if not suffix or member is _UNKNOWN or suffix[0] in (member, _UNKNOWN)
    }


def _empty_slice(node):
    """`tools[0:0]` renders an empty list whatever the catalog holds."""
    if not (isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Slice)):
        return False
    start, stop = node.arg.start, node.arg.stop
    start_value = 0 if start is None else (start.value if isinstance(start, nodes.Const) else None)
    stop_value = stop.value if isinstance(stop, nodes.Const) else None
    return (
        isinstance(start_value, int)
        and isinstance(stop_value, int)
        and start_value >= 0
        and stop_value >= 0
        and stop_value <= start_value
    )


def _template_built(node, state):
    """The member-name shortcuts below read a field off an untracked value, so they
    only mean anything when the field is one the template wrote itself.

    Asked of the field rather than its container: `{% set ns = namespace() %}
    {% set ns.role = message.role %}` leaves ns template-built but ns.role external,
    and the role check on it has to count.
    """
    key = state.origins.get(node.name) if isinstance(node, nodes.Name) else _reference_key(node)
    if key is None or len(key) < 2:
        return False
    return key[:-1] in state.constructed and key in state.constructed


def _tool_reference(node, state):
    key = _reference_key(node)
    if key in state.aliases:
        return True
    return _field(node) == "tool_calls" and not _template_built(node, state)


def _negated_guard(node, state):
    """True when the branch that is NOT taken is the one with a tool catalog.

    `{% if not tools %}plain{% else %}You may call tools.{% endif %}` advertises
    tools in its else arm exactly as `{% if tools %}...{% endif %}` does in its
    body, so the two spellings have to agree.
    """
    if isinstance(node, nodes.Not):
        return _positive_test(node.node, state)
    if _counts_tools(node, state):
        return True
    if isinstance(node, nodes.Test) and node.name in ("none", "undefined"):
        return _tool_reference(node.node, state)
    if isinstance(node, nodes.Compare) and len(node.ops) == 1 and node.ops[0].op == "eq":
        return any(
            (_tool_reference(a, state) or _counts_tools(a, state)) and _empty_literal(b)
            for a, b in ((node.expr, node.ops[0].expr), (node.ops[0].expr, node.expr))
        )
    return False


def _empty_literal(node):
    """`[]`, `{}`, `()`, `''`, `0`, `none` - the values a catalog is compared against
    to ask whether it is empty."""
    if isinstance(node, nodes.Const):
        return not node.value
    if isinstance(node, (nodes.List, nodes.Tuple, nodes.Dict)):
        return not node.items
    return False


def _counts_tools(node, state):
    """`tools|length` and friends, so `tools|length > 0` reads as a tool guard."""
    return (
        isinstance(node, nodes.Filter)
        and node.name in ("length", "count")
        and node.node is not None
        and _tool_reference(node.node, state)
    )


def _non_empty_tools(node, state):
    """A comparison that holds exactly when the catalog is present and non-empty.

    Templates spell the guard as `{% if tools != [] %}` or `{% if tools|length > 0 %}`
    as often as `{% if tools %}`, and all three advertise tools the same way.
    """
    if not (isinstance(node, nodes.Compare) and len(node.ops) == 1):
        return False
    operand = node.ops[0]
    left, right, op = node.expr, operand.expr, operand.op
    if op == "ne":
        return any(
            (_tool_reference(a, state) or _counts_tools(a, state)) and _empty_literal(b)
            for a, b in ((left, right), (right, left))
        )
    # `tools|length > 0` and the mirrored `0 < tools|length`.
    mirror = {"gt": "lt", "lt": "gt", "gteq": "lteq", "lteq": "gteq"}
    for counted, bound, sense in ((left, right, op), (right, left, mirror.get(op, op))):
        if not _counts_tools(counted, state) or not isinstance(bound, nodes.Const):
            continue
        if (sense == "gt" and bound.value == 0) or (sense == "gteq" and bound.value == 1):
            return True
    return False


def _positive_test(node, state):
    if _constant_truth(node) is not None:
        return False
    if _tool_reference(node, state):
        return True
    if isinstance(node, nodes.And):
        return _positive_test(node.left, state) or _positive_test(node.right, state)
    if isinstance(node, nodes.Or):
        return (
            _constant_truth(node.right, state) is not True and _positive_test(node.left, state)
        ) or (_constant_truth(node.left, state) is not True and _positive_test(node.right, state))
    if isinstance(node, nodes.Test):
        return node.name == "defined" and _tool_reference(node.node, state)
    if isinstance(node, nodes.Not):
        return (
            isinstance(node.node, nodes.Test)
            and node.node.name in ("none", "undefined")
            and _tool_reference(node.node.node, state)
        )
    if _counts_tools(node, state):
        return True
    if _non_empty_tools(node, state):
        return True
    if isinstance(node, nodes.Compare) and len(node.ops) == 1:
        operand = node.ops[0]
        if operand.op == "eq":
            return any(
                (_field(role) == "role" or _origin_field(role, state) == "role")
                and not _template_built(role, state)
                and (literal := _as_const(value, state)) is not None
                and literal.value == "tool"
                for role, value in ((node.expr, operand.expr), (operand.expr, node.expr))
            )
    return False


def _known_empty(node, state):
    """True when this path already established that the value renders as empty.

    `{% if not tools %}{{ tools|tojson }}{% endif %}` emits `[]`, which is no schema:
    the guard proves the catalog is falsy on the only path that reaches the output.
    """
    inner = node
    while isinstance(inner, nodes.Filter) and inner.node is not None:
        inner = inner.node
    if not isinstance(inner, (nodes.Name, nodes.Getattr, nodes.Getitem)):
        return False
    return _constant_truth(inner, state) is False


def _raises(node, state = None):
    """Whether evaluating this expression always reaches `raise_exception`.

    Only the positions certain to be evaluated: the left operand that `and` and `or`
    short-circuit on, the input of a filter, and every part of a concatenation.
    `{{ tools|tojson or raise_exception(...) }}` does NOT raise when the catalog is
    present, so a right operand does not count.
    """
    if (
        isinstance(node, nodes.Call)
        and isinstance(node.node, nodes.Name)
        and node.node.name == "raise_exception"
    ):
        return True
    if isinstance(node, (nodes.And, nodes.Or)):
        return _raises(node.left, state)
    if isinstance(node, (nodes.Filter, nodes.Not)) and node.node is not None:
        return _raises(node.node, state)
    if isinstance(node, nodes.Concat):
        return any(_raises(item, state) for item in node.nodes)
    if isinstance(node, nodes.Call):
        # Arguments are evaluated before the call, so one that raises aborts it.
        if any(_raises(argument, state) for argument in node.args) or any(
            _raises(keyword.value, state) for keyword in node.kwargs
        ):
            return True
        # A macro whose body raises unconditionally aborts wherever it is invoked.
        if state is not None and isinstance(node.node, nodes.Name):
            return any(
                _always_raises(macro.body, state)
                for macro in _macro_group(state.macros.get(node.node.name))
            )
    return False


def _always_raises(body, state):
    """Whether every render of this body reaches a raise. Only statements that always
    run count, so a raise inside an `{% if %}` does not."""
    return any(
        isinstance(statement, nodes.Output)
        and any(_raises(value, state) for value in statement.nodes)
        for statement in body
    )


def _is_payload(node):
    if isinstance(node, (nodes.Not, nodes.Test, nodes.Compare)):
        return False
    if isinstance(node, nodes.Filter) and node.name in ("length", "count"):
        return False
    if isinstance(node, nodes.TemplateData):
        return bool(node.data.strip())
    return not (
        isinstance(node, nodes.Call)
        and isinstance(node.node, nodes.Name)
        and node.node.name == "raise_exception"
    )


def _replace(paths, key, derived):
    paths.difference_update({old for old in paths if old[: len(key)] == key})
    paths.update((*key, *suffix) for suffix in derived)


def _value_aliases(value, state, active):
    if value is None or not _is_payload(value):
        return set()
    if isinstance(value, nodes.Name):
        return {key[1:] for key in state.aliases if key[0] == value.name}
    if isinstance(value, (nodes.Getattr, nodes.Getitem)):
        if _empty_slice(value):
            return set()
        if _field(value) == "tool_calls" and not _template_built(value, state):
            return {()}
        return _select(_value_aliases(value.node, state, active), _member(value, state))
    if isinstance(value, nodes.Dict):
        result = set()
        for pair in value.items:
            key = pair.key.value if isinstance(pair.key, nodes.Const) else _UNKNOWN
            _replace(result, (key,), _value_aliases(pair.value, state, active))
        return result
    if isinstance(value, (nodes.List, nodes.Tuple)):
        return {
            (index, *suffix)
            for index, item in enumerate(value.items)
            for suffix in _value_aliases(item, state, active)
        }
    if isinstance(value, nodes.CondExpr):
        return set().union(
            *(
                set()
                if _known_empty(expression, branch)
                else _value_aliases(expression, branch, active)
                for expression, truth in ((value.expr1, True), (value.expr2, False))
                if expression is not None
                for branch in _assume(value.test, truth, state)
            ),
            set(),
        )
    if isinstance(value, nodes.And):
        return set().union(
            *(
                _value_aliases(value.right, branch, active)
                for branch in _assume(value.left, True, state)
            )
        )
    if isinstance(value, nodes.Or):
        return set().union(
            *(
                _value_aliases(expression, branch, active)
                for expression, truth in ((value.left, True), (value.right, False))
                for branch in _assume(value.left, truth, state)
            )
        )
    if isinstance(value, nodes.Call) and isinstance(value.node, nodes.Getattr):
        # `catalog.pop('schema')` evaluates to whatever sat at that field.
        removed = _removed_key(value.node.attr, value)
        if removed is not None:
            result = _select(_value_aliases(value.node.node, state, active), removed)
            # `d.pop('missing', tools)` renders the default, but only when the field
            # can actually be absent: a literal that visibly holds the key never
            # reaches it.
            if not _definitely_has(value.node.node, removed, state):
                for fallback in value.args[1:]:
                    result |= _value_aliases(fallback, state, active)
            return result
        # A shallow copy is the receiver again as far as provenance goes; without this
        # the fallback reads `copy` as a data field and loses everything under it.
        if value.node.attr == "copy" and not value.args and not value.kwargs:
            return _value_aliases(value.node.node, state, active)
        if value.node.attr == "get" and value.args:
            member = value.args[0].value if isinstance(value.args[0], nodes.Const) else _UNKNOWN
            result = _select(_value_aliases(value.node.node, state, active), member)
            # The default is what a missing field falls back to, so it counts - unless
            # the receiver is a literal that visibly holds the key.
            if not _definitely_has(value.node.node, member, state):
                for fallback in value.args[1:]:
                    result |= _value_aliases(fallback, state, active)
            return result
        # append/extend/update and the other in-place mutators return None: the data
        # goes into the receiver, not into the rendered result.
        if value.node.attr in _MUTATORS_RETURNING_NONE | _SCALAR_RETURNING_METHODS:
            return set()
    if isinstance(value, nodes.Call) and isinstance(value.node, nodes.Name):
        if value.node.name in ("namespace", "dict"):
            result = set().union(*(_value_aliases(arg, state, active) for arg in value.args))
            for keyword in value.kwargs:
                _replace(result, (keyword.key,), _value_aliases(keyword.value, state, active))
            return result
        group = _macro_group(state.macros.get(value.node.name))
        if len(group) > 1:
            # `{% set render = plain if flag else show %}` could be either, so the
            # catalog counts if ANY of them renders it.
            return set().union(
                *(
                    _value_aliases(
                        nodes.Call(
                            nodes.Name(candidate.name, "load"), value.args, value.kwargs, None, None
                        ),
                        _with_macro(state, value.node.name, candidate),
                        active,
                    )
                    for candidate in group
                ),
                set(),
            )
        macro = group[0] if group else None
        if macro is not None:
            if active.count(macro.name) >= _RECURSION_LIMIT:
                # Deep enough. A macro that recurses without bound renders nothing
                # either, so giving up here costs only unbounded recursion.
                return set()
            local = state.copy(scoped = True)
            parameters = [argument.name for argument in macro.args]
            arguments = dict(zip(parameters, value.args))
            arguments.update((keyword.key, keyword.value) for keyword in value.kwargs)
            defaults = dict(
                zip(parameters[len(parameters) - len(macro.defaults) :], macro.defaults)
            )
            # `{% macro wrap(caller=None) %}` declares the parameter only to document
            # it; inside a {% call %} Jinja binds `caller` to the block regardless, so
            # the declaration must not clear a binding the caller supplied.
            supplied_caller = state.macros.get("caller")
            for name in parameters:
                if name == "caller" and supplied_caller is not None:
                    continue
                _replace(local.aliases, (name,), set())
                local.macros.pop(name, None)
                _forget((name,), local)
            for parameter in macro.args:
                if parameter.name == "caller" and supplied_caller is not None:
                    continue
                expression = arguments.get(parameter.name, defaults.get(parameter.name))
                _bind(
                    parameter,
                    expression,
                    local,
                    active,
                    source = state if parameter.name in arguments else local.copy(),
                )
            # Jinja collects arguments the signature does not name into the implicit
            # `varargs` and `kwargs`, so `{% macro show() %}{{ kwargs|tojson }}` sees
            # what was passed by keyword.
            extra_positional = value.args[len(parameters) :]
            _replace(
                local.aliases,
                ("varargs",),
                {
                    (index, *suffix)
                    for index, argument in enumerate(extra_positional)
                    for suffix in _value_aliases(argument, state, active)
                },
            )
            extra_keywords = set()
            for keyword in value.kwargs:
                if keyword.key not in parameters:
                    _replace(
                        extra_keywords,
                        (keyword.key,),
                        _value_aliases(keyword.value, state, active),
                    )
            _replace(local.aliases, ("kwargs",), extra_keywords)
            # The macro's own names stay live: nothing outside it constrains its body.
            emits, children = _scan(macro.body, local, (*active, macro.name), tail = _names(macro))
            # A namespace write inside a macro escapes it, so the caller sees it:
            # {% macro load() %}{% set ns.catalog = tools %}{% endmacro %}{{ load() }}
            # leaves the catalog in ns. _export_scope already knows which of a
            # scope's mutations outlive it, and the macro's own parameters were
            # dropped from `assigned` by the scoped copy above.
            # Different paths through the macro can leave different things in the
            # namespace. The question this analyser answers is whether the catalog
            # can reach the output on any feasible path, so the outcomes are unioned
            # rather than overwritten - otherwise the last child scanned wins.
            merged = set()
            for child in children:
                exported = _export_scope(state, child)
                merged |= exported.aliases
                state.mutated.update(exported.mutated)
            if children:
                state.aliases.clear()
                state.aliases.update(merged)
                # The macro's own facts are deliberately NOT carried out. Keeping one
                # child's facts alongside every child's aliases pairs a mutation made
                # under one condition with a different condition, which no render
                # takes. The caller's facts already describe what holds out here.
            return {()} if emits else set()
    # Other expressions serialize or transform their inputs.
    return (
        {()}
        if any(_value_aliases(child, state, active) for child in value.iter_child_nodes())
        else set()
    )


def _bind(
    target,
    value,
    state,
    active,
    source = None,
):
    private = source is None
    source = state.copy() if private else source
    if isinstance(target, (nodes.Tuple, nodes.List)):
        if isinstance(value, (nodes.Tuple, nodes.List)):
            for item, expression in zip(target.items, value.items):
                _bind(item, expression, state, active, source = source)
        else:
            paths = _value_aliases(value, source, active)
            for index, item in enumerate(target.items):
                _bind_paths(item, _select(paths, index), state)
        return
    paths = _value_aliases(value, source, active)
    if private and source.mutated - state.mutated:
        # Evaluating the value ran a macro whose namespace writes escape it. Those
        # landed on the private copy, so carry them over before the target is bound.
        carried = _export_scope(state, source)
        state.aliases.clear()
        state.aliases.update(carried.aliases)
        state.mutated.update(carried.mutated)
        state.facts = carried.facts
    _bind_paths(target, paths, state)
    key = _reference_key(target)
    if key is not None:
        source_key = (
            _reference_key(value)
            if isinstance(value, (nodes.Name, nodes.Getattr, nodes.Getitem))
            else None
        )
        members = _mapping_keys(value)
        if members is None:
            state.mappings.pop(key, None)
        else:
            state.mappings[key] = members
        if isinstance(value, nodes.Const):
            # `{% set ns.role = 'tool' %}` writes a literal, so the field is the
            # template's own and a role check on it means nothing.
            state.constructed.add(key)
        elif _constructs_object(value):
            # The replacement populates its own members, so the old subtree goes
            # first: otherwise `{% set w={'message': message} %}` keeps the previous
            # literal's `w.message` marked template-built while it now holds input.
            state.constructed.difference_update(
                [built for built in state.constructed if built[: len(key)] == key]
            )
            _mark_constructed(key, value, state)
        elif source_key is not None and source_key in state.constructed:
            # The nested members were built by the template too, so the alias has to
            # carry them: otherwise `current.message.role` reads as external data.
            inherited = [
                (*key, *built[len(source_key) :])
                for built in state.constructed
                if built[: len(source_key)] == source_key
            ]
            state.constructed.update(inherited)
        else:
            # Not just the root: `{% set wrapper = payload %}` makes every member of
            # the old wrapper external again, so a role check on one has to count.
            state.constructed.difference_update(
                [built for built in state.constructed if built[: len(key)] == key]
            )
    if isinstance(target, nodes.Name):
        if isinstance(value, nodes.Const) and isinstance(value.value, (str, int)):
            state.consts[target.name] = value.value
        else:
            state.consts.pop(target.name, None)
        origin = _reference_key(value) if value is not None else None
        if origin is not None and len(origin) > 1 and origin not in state.constructed:
            state.origins[target.name] = origin
        else:
            state.origins.pop(target.name, None)
        # `{% set render = show %}` hands the name the macro, so calling it runs the
        # same body. _bind_paths has already dropped any macro under this name.
        # A conditional picks one of its arms, and either may be a macro.
        selected = tuple(
            macro
            for candidate in _macro_sources(value)
            for macro in _macro_group(state.macros.get(candidate))
        )
        if selected:
            state.macros[target.name] = selected if len(selected) > 1 else selected[0]
    truth = _constant_truth(value, source) if value is not None else None
    if isinstance(target, nodes.Name) and truth is not None:
        state.facts[repr(nodes.Name(target.name, "load"))] = (truth, {target.name})


def _with_macro(state, name, macro):
    """The same state with one macro table entry narrowed to a single candidate."""
    narrowed = state.copy()
    narrowed.macros[name] = macro
    narrowed.macros[macro.name] = macro
    return narrowed


def _macro_group(entry):
    """A macro table entry as a tuple: an expression may have selected several."""
    if entry is None:
        return ()
    return entry if isinstance(entry, tuple) else (entry,)


def _macro_sources(value):
    """The names an assigned expression could evaluate to, for carrying a macro across
    `{% set render = show %}` and `{% set render = show if flag else other %}`."""
    if isinstance(value, nodes.Name):
        return [value.name]
    if isinstance(value, nodes.CondExpr):
        return [
            name
            for arm in (value.expr1, value.expr2)
            if arm is not None
            for name in _macro_sources(arm)
        ]
    if isinstance(value, (nodes.List, nodes.Tuple)):
        return [name for item in value.items for name in _macro_sources(item)]
    if isinstance(value, nodes.Getitem):
        # `{% set render = [show][0] %}`: a lookup table of formatters.
        return _macro_sources(value.node)
    if isinstance(value, nodes.Dict):
        return [name for pair in value.items for name in _macro_sources(pair.value)]
    return []


def _mark_constructed(key, value, state):
    """A literal and everything nested in it were all built by the template."""
    state.constructed.add(key)
    pairs = []
    if isinstance(value, nodes.Dict):
        pairs = [
            (pair.key.value, pair.value)
            for pair in value.items
            if isinstance(pair.key, nodes.Const)
        ]
    elif isinstance(value, (nodes.List, nodes.Tuple)):
        pairs = list(enumerate(value.items))
    elif isinstance(value, nodes.Call):
        pairs = [(keyword.key, keyword.value) for keyword in value.kwargs]
    for member, item in pairs:
        if _constructs_object(item):
            _mark_constructed((*key, member), item, state)
        elif isinstance(item, nodes.Const):
            # A literal scalar member is template-built too, and recording it is what
            # says this record HAS the field, which decides whether a get/pop default
            # can ever be reached. A member holding an expression is NOT recorded: it
            # may carry external data, and `{% set w={'message': message} %}` must
            # leave w.message external.
            state.constructed.add((*key, member))


def _constructs_object(value):
    if isinstance(value, (nodes.Dict, nodes.List, nodes.Tuple)):
        return True
    return (
        isinstance(value, nodes.Call)
        and isinstance(value.node, nodes.Name)
        and value.node.name in ("namespace", "dict")
    )


def _bind_paths(target, paths, state):
    if isinstance(target, (nodes.Tuple, nodes.List)):
        for index, item in enumerate(target.items):
            _bind_paths(item, _select(paths, index), state)
        return
    key = _reference_key(target)
    if key is None:
        return
    _replace(state.aliases, key, paths)
    _forget(key, state)
    if isinstance(target, nodes.Name):
        state.assigned.add(target.name)
        state.macros.pop(target.name, None)
    elif key[0] not in state.assigned:
        # Recorded when the write happens, not when the scope closes: a mutation that
        # ran BEFORE a local rebind hit the outer object and still has to escape.
        state.mutated.add(key)


def _mutate(call, state, active):
    if not isinstance(call, nodes.Call) or not isinstance(call.node, nodes.Getattr):
        return
    key = _reference_key(call.node.node)
    if key is None:
        return
    method = call.node.attr
    if method in _SCALAR_RETURNING_METHODS:
        # `catalog.count(x)` reads the receiver and answers a number: nothing moves.
        return
    positional = {suffix for arg in call.args for suffix in _value_aliases(arg, state, active)}
    if method == "update":
        # An update REPLACES the fields it names, so whatever they held before is
        # gone whether or not the new value carries provenance of its own.
        overwritten = [keyword.key for keyword in call.kwargs]
        for argument in call.args:
            if isinstance(argument, nodes.Dict):
                overwritten.extend(
                    pair.key.value for pair in argument.items if isinstance(pair.key, nodes.Const)
                )
        for member in overwritten:
            _replace(state.aliases, (*key, member), set())
    if method == "update" and all(isinstance(arg, nodes.Dict) for arg in call.args):
        # A positional mapping keeps its own keys, exactly as the keyword form below
        # does: d.update({'catalog': tools}) puts the catalog at d.catalog and leaves
        # every other field of d alone.
        pass
    elif method != "extend":
        # append, insert, add, setdefault: the argument lands somewhere under the
        # receiver rather than being spliced into it. Any unrecognised method handed
        # tool data is treated the same way rather than ignored.
        positional = {(_UNKNOWN, *suffix) for suffix in positional}
    # A keyword names the field its value lands under: d.update(catalog=tools) puts
    # the catalog at d.catalog.
    paths = positional | {
        (keyword.key, *suffix)
        for keyword in call.kwargs
        for suffix in _value_aliases(keyword.value, state, active)
    }
    if method in ("reverse", "sort") and not call.args:
        # The members are the same but their positions are not, so every index under
        # the receiver becomes unknown. This over-approximates - selecting the index
        # the catalog moved away from also matches - which is the safe direction for
        # a detector whose failure mode is silently hiding tool controls.
        moved = {
            alias for alias in state.aliases if alias[: len(key)] == key and len(alias) > len(key)
        }
        if moved:
            _replace(
                state.aliases,
                key,
                {(_UNKNOWN, *alias[len(key) + 1 :]) for alias in moved},
            )
            state.mutated.add(key)
            _forget(key, state)
        return
    removed = _removed_key(method, call)
    if method != "clear" and removed is None and not paths:
        return
    if method == "clear" or removed is _UNKNOWN:
        _replace(state.aliases, key, set())
    elif removed is not None:
        # pop/remove/discard take the value back out, so its provenance goes too.
        _replace(state.aliases, (*key, removed), set())
        if isinstance(removed, int) and not isinstance(removed, bool):
            # A list closes the gap, so everything after the hole moves down one.
            shifted = {
                alias
                for alias in state.aliases
                if alias[: len(key)] == key
                and len(alias) > len(key)
                and isinstance(alias[len(key)], int)
                and alias[len(key)] > removed
            }
            state.aliases.difference_update(shifted)
            state.aliases.update(
                (*key, alias[len(key)] - 1, *alias[len(key) + 1 :]) for alias in shifted
            )
    else:
        state.aliases.update((*key, *suffix) for suffix in paths)
    if key[0] not in state.assigned:
        state.mutated.add(key)
    _forget(key, state)


# Filters that reduce their input to a measurement or a single element, so whatever
# went in is no longer readable on the other side.
_REDUCING = frozenset(
    {"length", "count", "first", "last", "min", "max", "sum", "random", "wordcount"}
)


def _keeps_content(node):
    while isinstance(node, nodes.Filter):
        if node.name in _REDUCING:
            return False
        node = node.node
    return True


def _unknown_callee(call, state):
    """Whether a `{% call %}` targets something this analyser cannot read.

    A macro it knows is scanned, and the caller block bound into it as `caller`, so
    reachability falls out of the ordinary scan. Anything else could invoke the block
    for reasons not visible here, so the block is scanned unconditionally.
    """
    return not (isinstance(call.node, nodes.Name) and call.node.name in state.macros)


def _mapping_keys(value):
    """The field names a mapping literal was written with, or None if this is not one.

    Every written key counts, including one whose value came from outside: what makes
    a `get`/`pop` default unreachable is the key being THERE, not what it holds.
    """
    if isinstance(value, nodes.Dict):
        return frozenset(
            pair.key.value for pair in value.items if isinstance(pair.key, nodes.Const)
        )
    if (
        isinstance(value, nodes.Call)
        and isinstance(value.node, nodes.Name)
        and value.node.name == "dict"
    ):
        return frozenset(keyword.key for keyword in value.kwargs)
    return None


def _definitely_has(node, member, state):
    """Whether the receiver is a literal this template built that visibly holds
    `member`, so a `get`/`pop` default can never be reached."""
    if member is _UNKNOWN:
        return False
    if isinstance(node, nodes.Dict):
        return any(
            isinstance(pair.key, nodes.Const) and pair.key.value == member for pair in node.items
        )
    key = _reference_key(node)
    if key is None:
        return False
    if member in state.mappings.get(key, ()):
        return True
    return key in state.constructed and (*key, member) in state.constructed


def _removed_key(method, call):
    """The field a destructive call takes back out, or None. `_UNKNOWN` when the call
    removes something we cannot name, which drops the whole receiver's provenance."""
    if method not in ("pop", "remove", "discard", "popitem"):
        return None
    if not call.args:
        return _UNKNOWN
    return call.args[0].value if isinstance(call.args[0], nodes.Const) else _UNKNOWN


def _export_scope(parent, child):
    result = parent.copy()
    result.facts.update(
        {
            expression: fact
            for expression, fact in child.facts.items()
            if not fact[1] & child.assigned
        }
    )
    for key in child.mutated:
        _replace(
            result.aliases,
            key,
            {alias[len(key) :] for alias in child.aliases if alias[: len(key)] == key},
        )
        result.mutated.add(key)
        _forget(key, result)
    return result


def _scan_if(node, state, active, guarded, tail):
    remaining = [state]
    results = []
    for branch in [node, *node.elif_]:
        next_remaining = []
        for current in remaining:
            for positive in _assume(branch.test, True, current):
                emits, states = _scan(
                    branch.body,
                    positive,
                    active,
                    guarded or _positive_test(branch.test, current),
                    tail,
                )
                if emits:
                    return True, []
                results.extend(states)
            next_remaining.extend(_assume(branch.test, False, current))
        remaining = next_remaining
    for current in remaining:
        # With an elif in the chain the else arm is reached for more than one
        # reason, so only a plain if/else carries the negated guard across.
        else_guarded = guarded or (not node.elif_ and _negated_guard(node.test, current))
        emits, states = _scan(node.else_, current, active, else_guarded, tail)
        if emits:
            return True, []
        results.extend(states)
    return False, results


def _scan_loop(node, state, active, guarded, tail):
    # A fact established on one iteration is read on the next, so nothing the loop
    # itself mentions may be pruned inside it.
    inner_tail = tail | _names(node)
    literal = isinstance(node.iter, (nodes.List, nodes.Tuple))
    values = node.iter.items if literal else [None]
    states = [state]
    finished = []
    if not values:
        emits, children = _scan(node.else_, state.copy(scoped = True), active, guarded, inner_tail)
        return emits, [_export_scope(state, child) for child in children]
    else_reachable = True
    for value in values:
        results = []
        for parent in states:
            local = parent.copy(scoped = True)
            if value is None:
                # `{% for key in {'x': tools} %}` walks the keys, not the values, so
                # nothing under the mapping reaches the target. The mapping may have
                # been given a name first, which `state.mappings` records.
                over_keys = isinstance(node.iter, nodes.Dict) or (
                    _reference_key(node.iter) in parent.mappings
                )
                _bind_paths(
                    node.target,
                    set()
                    if over_keys
                    else _select(_value_aliases(node.iter, parent, active), _UNKNOWN),
                    local,
                )
            else:
                _bind(node.target, value, local, active)
            # `loop.first` reprs the same in every loop, so an outer loop's facts would
            # otherwise prune branches of a nested one.
            _forget(("loop",), local)
            if literal and node.test is None:
                # A literal iterable is simulated item by item, so which iteration
                # this is happens to be known: `{% for x in [1] %}{% if not
                # loop.first %}` never runs its body. With a filter the position
                # describes the accepted sequence, not the source, so it stays
                # unknown rather than being read off the source index.
                position = values.index(value)
                for member, truth in (
                    ("first", position == 0),
                    ("last", position == len(values) - 1),
                ):
                    local.facts[repr(nodes.Getattr(nodes.Name("loop", "load"), member, "load"))] = (
                        truth,
                        {"loop"},
                    )
            candidates = [local] if node.test is None else _assume(node.test, True, local)
            for candidate in candidates:
                emits, children = _scan(
                    node.body,
                    candidate,
                    active,
                    guarded
                    or _tool_reference(node.iter, parent)
                    or (node.test is not None and _positive_test(node.test, candidate)),
                    inner_tail,
                )
                if emits:
                    return True, []
                for child in children:
                    exported = _export_scope(parent, child)
                    if child.terminated:
                        # break left the loop, so later items of a literal iterable
                        # never run for this path: park it instead of simulating on.
                        exported.terminated = False
                        exported.continued = False
                        finished.append(exported)
                    else:
                        # continue only skipped the rest of this iteration.
                        exported.continued = False
                        results.append(exported)
            if node.test is not None:
                rejected = _assume(node.test, False, local)
                if not rejected:
                    # This item always passes the filter, so the body runs at least
                    # once and the else arm is unreachable.
                    else_reachable = False
                results.extend(_export_scope(parent, child) for child in rejected)
        states = results
    # A filter can reject every item of a literal iterable, in which case the body
    # never runs and Jinja takes the else. Only an unfiltered literal is guaranteed
    # to iterate.
    # An else arm reached after a break must see the mutations that path already made,
    # so it is scanned from each parked state as well as from the pre-loop one.
    entries = []
    if not literal or (node.test is not None and else_reachable):
        entries.append(state)
    entries.extend(finished)
    for entry in entries:
        emits, children = _scan(node.else_, entry.copy(scoped = True), active, guarded, inner_tail)
        if emits:
            return True, []
        states.extend(_export_scope(entry, child) for child in children)
    return False, states + finished


def _scan(
    body,
    state,
    active,
    guarded = False,
    tail = frozenset(),
):
    states = [state]
    stopped = []
    live = _live_names(body)
    for index, node in enumerate(body):
        # Everything still readable once this statement is done, here or further out.
        rest = live[index + 1] | tail
        results = []
        for current in states:
            current.budget[0] -= 1
            if current.budget[0] < 0:
                raise _AnalysisLimit
            if isinstance(node, nodes.Output):
                # Consecutive `{{ }}` are parsed into one Output, so its children are
                # walked in order: what precedes a raise renders, what follows it
                # does not, and a mutating call lands before the next statement.
                aborted = False
                for value in node.nodes:
                    if _raises(value, current):
                        # Checked first: the raise happens before this expression's
                        # own payload would be rendered, not after it.
                        _mutate(value, current, active)
                        aborted = True
                        break
                    if (
                        _is_payload(value)
                        and (guarded or _value_aliases(value, current, active))
                        and not _known_empty(value, current)
                    ):
                        return True, []
                    _mutate(value, current, active)
                if aborted:
                    # The render stops here, so this path reaches no later output.
                    continue
            elif isinstance(node, nodes.Assign):
                # `{% set _ = xs.append(...) %}` is how templates mutate without the do
                # extension, so the call mutates even though this is an assignment.
                # Bind first: `{% set x = catalog.pop('schema') %}` hands x the value the
                # mutation is about to take out of catalog.
                _bind(node.target, node.node, current, active)
                _mutate(node.node, current, active)
            elif isinstance(node, nodes.ExprStmt):
                # `{% do load() %}` discards the return value but still runs the
                # macro, so it is evaluated for the state it leaves behind.
                _value_aliases(node.node, current, active)
                _mutate(node.node, current, active)
            elif isinstance(node, nodes.Macro):
                current.macros[node.name] = node
                # The declaration binds the name, so `{% macro tools() %}` shadows the
                # catalog and `{{ tools }}` renders the macro object instead.
                _replace(current.aliases, (node.name,), set())
            elif isinstance(node, nodes.If):
                emits, children = _scan_if(node, current, active, guarded, rest)
                if emits:
                    return True, []
                results.extend(children)
                continue
            elif isinstance(node, nodes.For):
                emits, children = _scan_loop(node, current, active, guarded, rest)
                if emits:
                    return True, []
                results.extend(children)
                continue
            elif isinstance(node, (nodes.Break, nodes.Continue)):
                # Nothing after this in the body runs, so the path stops being scanned.
                # It still carries whatever it already mutated, which outlives the loop.
                if isinstance(node, nodes.Break):
                    current.terminated = True
                else:
                    current.continued = True
                stopped.append(current)
                continue
            elif isinstance(node, nodes.CallBlock):
                # {% call macro(...) %}: the body is the caller block, so the generic
                # handler below would only ever see an empty one. The invocation is
                # where the catalog actually reaches the output.
                # Binding the block as a macro named `caller` lets the ordinary scan
                # decide whether it runs, so a caller() sitting in a branch that
                # cannot execute does not drag the block in with it.
                invoked = current.copy()
                invoked.macros["caller"] = nodes.Macro("caller", [], [], node.body)
                emitted = _value_aliases(node.call, invoked, active)
                # The macro ran, so whatever it wrote to an outer namespace escapes,
                # exactly as it does for a plain call.
                if invoked.mutated - current.mutated:
                    carried = _export_scope(current, invoked)
                    current.aliases.clear()
                    current.aliases.update(carried.aliases)
                    current.mutated.update(carried.mutated)
                if emitted:
                    return True, []
                if _unknown_callee(node.call, current):
                    emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded, rest)
                    if emits:
                        return True, []
            elif isinstance(node, nodes.FilterBlock):
                # The block's own filter decides what survives: `{% filter first %}`
                # emits one character of the catalog, which is no schema at all.
                if _keeps_content(node.filter):
                    emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded, rest)
                    if emits:
                        return True, []
            elif isinstance(node, nodes.AssignBlock):
                # `{% set catalog|length %}` binds the filtered result, so a filter
                # that keeps nothing of the catalog leaves nothing to find.
                emits, captured = _scan(node.body, current.copy(scoped = True), active, guarded, rest)
                # Jinja keeps a namespace write made inside the capture, so the
                # escaping mutations are exported before the target is bound.
                escaped = set()
                for child in captured:
                    exported = _export_scope(current, child)
                    escaped |= exported.aliases
                    current.mutated.update(exported.mutated)
                if captured:
                    current.aliases.clear()
                    current.aliases.update(escaped)
                kept = emits and _keeps_content(node.filter)
                _bind_paths(node.target, {()} if kept else set(), current)
            elif isinstance(node, nodes.With):
                local = current.copy(scoped = True)
                for target, value in zip(node.targets, node.values):
                    _bind(target, value, local, active, source = current)
                emits, children = _scan(node.body, local, active, guarded, rest)
                if emits:
                    return True, []
                results.extend(_export_scope(current, child) for child in children)
                continue
            elif hasattr(node, "body"):
                emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded, rest)
                if emits:
                    return True, []
            results.append(current)
        states = _collapse(results, rest)
    return False, states + stopped


def template_supports_tools(template) -> bool:
    """Inspect syntax only; rendering and parser support remain backend checks."""
    # Outside the cache: lru_cache hashes its argument before the body runs, so a
    # dict- or list-valued chat template (a Hugging Face named-template map, or the
    # Hermes-3 [{"name", "template"}] list) would raise "unhashable type" past every
    # fail-closed branch below. routes.inference passes the raw value through when
    # template selection yields nothing, so this is reachable, not theoretical.
    if not isinstance(template, str):
        return False
    return _analyse_template(template)


@lru_cache(maxsize = 128)
def _analyse_template(template: str) -> bool:
    if "tool" not in template:
        return False
    try:
        tree = _ENVIRONMENT.parse(template)
        emits, _ = _scan(tree.body, _State({("tools",), ("tool_calls",)}), ())
        return emits
    except Exception:
        # Fail closed. Besides the expected TemplateSyntaxError and _AnalysisLimit, a
        # deeply nested template exhausts the interpreter stack -- in Jinja's own
        # recursive-descent parser (`{% if %}` nesting, parenthesised guards), in
        # _value_aliases (a long attribute chain), or in the node repr used as a fact
        # key (a wide boolean guard). detect_reasoning_flags runs on the GGUF metadata
        # read and the llama-server launch, so a capability hint must leave tools
        # disabled rather than stop the model from loading, which the substring scan
        # this replaced could never do.
        return False


# Callers that measure or reset the analysis cache reach it through the public name.
template_supports_tools.cache_clear = _analyse_template.cache_clear
template_supports_tools.cache_info = _analyse_template.cache_info
