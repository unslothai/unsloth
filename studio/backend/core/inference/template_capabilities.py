# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-capability hints from executable Jinja syntax."""

from dataclasses import dataclass, field
from functools import lru_cache

from jinja2 import Environment, TemplateSyntaxError, nodes
from jinja2.ext import Extension


class _Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle = True)


_ENVIRONMENT = Environment(extensions = [_Generation, "jinja2.ext.loopcontrols", "jinja2.ext.do"])
_UNKNOWN = object()


class _AnalysisLimit(Exception):
    pass


@dataclass
class _State:
    aliases: set
    macros: dict = field(default_factory = dict)
    facts: dict = field(default_factory = dict)
    assigned: set = field(default_factory = set)
    mutated: set = field(default_factory = set)
    budget: list = field(default_factory = lambda: [8192])

    def copy(self, scoped = False):
        return _State(
            self.aliases.copy(),
            self.macros.copy(),
            self.facts.copy(),
            set() if scoped else self.assigned.copy(),
            set() if scoped else self.mutated.copy(),
            self.budget,
        )


def _field(node):
    if isinstance(node, nodes.Getattr):
        return node.attr
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        return node.arg.value
    return _UNKNOWN


def _reference_key(node):
    if isinstance(node, nodes.Name):
        return (node.name,)
    if isinstance(node, nodes.NSRef):
        return (node.name, node.attr)
    if isinstance(node, (nodes.Getattr, nodes.Getitem)):
        parent = _reference_key(node.node)
        member = _field(node)
        if parent is not None and isinstance(member, (str, int)):
            return (*parent, member)
    return None


def _names(node):
    return {item.name for item in node.find_all(nodes.Name)} | (
        {node.name} if isinstance(node, nodes.Name) else set()
    )


def _constant_truth(node, state = None):
    fact = state.facts.get(repr(node)) if state is not None else None
    if fact is not None:
        return fact[0]
    if isinstance(node, nodes.Const):
        return bool(node.value)
    if isinstance(node, (nodes.List, nodes.Tuple, nodes.Dict)):
        return bool(node.items)
    if (
        isinstance(node, nodes.Compare)
        and isinstance(node.expr, nodes.Const)
        and all(isinstance(operand.expr, nodes.Const) for operand in node.ops)
    ):
        try:
            return bool(node.as_const())
        except nodes.Impossible:
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


def _tool_reference(node, state):
    key = _reference_key(node)
    return key in state.aliases or _field(node) == "tool_calls"


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
        if _field(value) == "tool_calls":
            return {()}
        return _select(_value_aliases(value.node, state, active), _field(value))
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
                _value_aliases(expression, branch, active)
                for expression, truth in ((value.expr1, True), (value.expr2, False))
                for branch in _assume(value.test, truth, state)
            )
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
    if isinstance(value, nodes.Call) and isinstance(value.node, nodes.Name):
        if value.node.name == "namespace":
            result = set().union(*(_value_aliases(arg, state, active) for arg in value.args))
            for keyword in value.kwargs:
                _replace(result, (keyword.key,), _value_aliases(keyword.value, state, active))
            return result
        macro = state.macros.get(value.node.name)
        if macro is not None:
            if macro.name in active:
                return set()
            local = state.copy(scoped = True)
            parameters = [argument.name for argument in macro.args]
            arguments = dict(zip(parameters, value.args))
            arguments.update((keyword.key, keyword.value) for keyword in value.kwargs)
            defaults = dict(
                zip(parameters[len(parameters) - len(macro.defaults) :], macro.defaults)
            )
            for name in parameters:
                _replace(local.aliases, (name,), set())
                local.macros.pop(name, None)
                _forget((name,), local)
            for parameter in macro.args:
                expression = arguments.get(parameter.name, defaults.get(parameter.name))
                _bind(
                    parameter,
                    expression,
                    local,
                    active,
                    source = state if parameter.name in arguments else local.copy(),
                )
            emits, _ = _scan(macro.body, local, active | {macro.name})
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
    source = state.copy() if source is None else source
    if isinstance(target, (nodes.Tuple, nodes.List)):
        if isinstance(value, (nodes.Tuple, nodes.List)):
            for item, expression in zip(target.items, value.items):
                _bind(item, expression, state, active, source = source)
        else:
            paths = _value_aliases(value, source, active)
            for index, item in enumerate(target.items):
                _bind_paths(item, _select(paths, index), state)
        return
    _bind_paths(target, _value_aliases(value, source, active), state)
    truth = _constant_truth(value, source) if value is not None else None
    if isinstance(target, nodes.Name) and truth is not None:
        state.facts[repr(nodes.Name(target.name, "load"))] = (truth, {target.name})


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
    else:
        state.mutated.add(key)


def _mutate(call, state, active):
    if not isinstance(call, nodes.Call) or not isinstance(call.node, nodes.Getattr):
        return
    key = _reference_key(call.node.node)
    if key is None:
        return
    method = call.node.attr
    if method not in ("append", "extend", "clear"):
        return
    paths = set().union(*(_value_aliases(arg, state, active) for arg in call.args))
    if method == "clear":
        _replace(state.aliases, key, set())
    elif paths:
        if method == "append":
            paths = {(_UNKNOWN, *suffix) for suffix in paths}
        state.aliases.update((*key, *suffix) for suffix in paths)
    state.mutated.add(key)
    _forget(key, state)


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
        if key[0] in child.assigned:
            continue
        _replace(
            result.aliases,
            key,
            {alias[len(key) :] for alias in child.aliases if alias[: len(key)] == key},
        )
        result.mutated.add(key)
        _forget(key, result)
    return result


def _scan_if(node, state, active, guarded):
    remaining = [state]
    results = []
    for branch in [node, *node.elif_]:
        next_remaining = []
        for current in remaining:
            for positive in _assume(branch.test, True, current):
                emits, states = _scan(
                    branch.body, positive, active, guarded or _positive_test(branch.test, current)
                )
                if emits:
                    return True, []
                results.extend(states)
            next_remaining.extend(_assume(branch.test, False, current))
        remaining = next_remaining
    for current in remaining:
        emits, states = _scan(node.else_, current, active, guarded)
        if emits:
            return True, []
        results.extend(states)
    return False, results


def _scan_loop(node, state, active, guarded):
    literal = isinstance(node.iter, (nodes.List, nodes.Tuple))
    values = node.iter.items if literal else [None]
    states = [state]
    if not values:
        emits, children = _scan(node.else_, state.copy(scoped = True), active, guarded)
        return emits, [_export_scope(state, child) for child in children]
    for value in values:
        results = []
        for parent in states:
            local = parent.copy(scoped = True)
            if value is None:
                _bind_paths(
                    node.target, _select(_value_aliases(node.iter, parent, active), _UNKNOWN), local
                )
            else:
                _bind(node.target, value, local, active)
            candidates = [local] if node.test is None else _assume(node.test, True, local)
            for candidate in candidates:
                emits, children = _scan(
                    node.body, candidate, active, guarded or _tool_reference(node.iter, parent)
                )
                if emits:
                    return True, []
                results.extend(_export_scope(parent, child) for child in children)
            if node.test is not None:
                results.extend(
                    _export_scope(parent, child) for child in _assume(node.test, False, local)
                )
        states = results
    if not literal:
        emits, children = _scan(node.else_, state.copy(scoped = True), active, guarded)
        if emits:
            return True, []
        states.extend(_export_scope(state, child) for child in children)
    return False, states


def _scan(
    body,
    state,
    active,
    guarded = False,
):
    states = [state]
    for node in body:
        results = []
        for current in states:
            current.budget[0] -= 1
            if current.budget[0] < 0:
                raise _AnalysisLimit
            if isinstance(node, nodes.Output):
                if any(
                    _is_payload(value) and (guarded or _value_aliases(value, current, active))
                    for value in node.nodes
                ):
                    return True, []
            elif isinstance(node, nodes.Assign):
                _bind(node.target, node.node, current, active)
            elif isinstance(node, nodes.ExprStmt):
                _mutate(node.node, current, active)
            elif isinstance(node, nodes.Macro):
                current.macros[node.name] = node
            elif isinstance(node, nodes.If):
                emits, children = _scan_if(node, current, active, guarded)
                if emits:
                    return True, []
                results.extend(children)
                continue
            elif isinstance(node, nodes.For):
                emits, children = _scan_loop(node, current, active, guarded)
                if emits:
                    return True, []
                results.extend(children)
                continue
            elif isinstance(node, nodes.AssignBlock):
                emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded)
                _bind_paths(node.target, {()} if emits else set(), current)
            elif isinstance(node, nodes.With):
                local = current.copy(scoped = True)
                for target, value in zip(node.targets, node.values):
                    _bind(target, value, local, active, source = current)
                emits, children = _scan(node.body, local, active, guarded)
                if emits:
                    return True, []
                results.extend(_export_scope(current, child) for child in children)
                continue
            elif hasattr(node, "body"):
                emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded)
                if emits:
                    return True, []
            results.append(current)
        states = results
    return False, states


@lru_cache(maxsize = 128)
def template_supports_tools(template: str) -> bool:
    """Inspect syntax only; rendering and parser support remain backend checks."""
    if "tool" not in template:
        return False
    try:
        tree = _ENVIRONMENT.parse(template)
        emits, _ = _scan(tree.body, _State({("tools",), ("tool_calls",)}), set())
        return emits
    except (TemplateSyntaxError, _AnalysisLimit):
        return False
